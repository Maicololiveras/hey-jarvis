"""Jarvis daemon — main orchestrator loop.

Wires together all Jarvis modules:
  AudioPipeline  →  WakeEvent / SegmentEvent
  StateMachine   →  DORMIDO / ACTIVO transitions
  STT            →  speech-to-text via faster-whisper
  QueryRouter    →  AI backend dispatch (claude-p, maix-engine)
  EngramBridge   →  memory enrichment + session persistence
  TTS (functions)→  edge-tts / pyttsx3 speech output
  JarvisUI       →  circular visual feedback window
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any

import numpy as np

from . import config
from .models import TickEvent, WakeEvent, SegmentEvent, UICommand
from .state_machine import StateMachine, State
from .audio_pipeline import AudioPipeline
from .stt import STT
from .query_router import QueryRouter
from .engram_bridge import EngramBridge
from . import tts as tts_module  # module-level functions: speak(), is_speaking()
from .jarvis_ui import JarvisUI
from .logging_setup import setup_logging
from .local_model import get_server_from_config

log = logging.getLogger(__name__)


class JarvisDaemon:
    """Main orchestrator that wires all Jarvis modules together.

    Lifecycle:
        1. ``__init__``  — loads config, creates all module instances.
        2. ``run``       — enters the blocking event loop (AudioPipeline.stream_events).
        3. ``_shutdown`` — graceful cleanup on KeyboardInterrupt or fatal error.
    """

    def __init__(self, ui: JarvisUI | None = None) -> None:
        # NOTE: setup_logging() is called in __main__.py — do NOT call it again here

        cfg = config.load()
        jarvis_cfg: dict[str, Any] = cfg.get("jarvis", {})
        stt_cfg = config.get_stt_config()

        # ── Module initialization ──────────────────────────────────────
        self.state = StateMachine(
            silence_timeout=jarvis_cfg.get("silence_timeout_seconds", 5.0),
        )
        self.audio = AudioPipeline(
            config.get_audio_config(),
            config.get_wake_word_config(),
        )

        self.stt = STT(
            model_path=stt_cfg.get("model_path", ""),
            device=stt_cfg.get("device", "cpu"),
            compute_type=stt_cfg.get("compute_type", "int8"),
            fast_model_path=stt_cfg.get("fast_model_path", ""),
            engine=stt_cfg.get("engine", "faster-whisper"),
            api_key_env=stt_cfg.get("api_key_env", "GROQ_API_KEY"),
            language=jarvis_cfg.get("language", "auto"),
            groq_model=stt_cfg.get("model", "whisper-large-v3"),
        )
        self._use_precise_stt = bool(stt_cfg.get("use_precise_pass", False))
        # Pre-load only the models we will actually use to avoid paying
        # startup and runtime cost for an unused precise pass.
        self.stt.preload(fast=True, precise=self._use_precise_stt)

        self.router = QueryRouter(config.get_query_config())
        self.engram = EngramBridge(
            config.get_engram_config(),
            config.get_query_config(),
        )

        # JarvisUI is either the PyQt6 widget or the NullUI fallback.
        # When PyQt6 is available it requires a QApplication — the caller
        # (e.g. __main__.py) is responsible for creating one before
        # instantiating the daemon, or we create a headless NullUI.
        # The caller can pass a pre-created UI instance so that the daemon
        # sends commands to the same widget displayed on screen.
        self.ui: JarvisUI = ui if ui is not None else JarvisUI()
        self.ui.configure_backend_selector(
            current_backend=self.router.get_default_backend(),
            available_backends=self._ui_available_backends(),
            on_backend_selected=self._on_backend_changed,
        )

        # ── Maix AI Engine preload check ────────────────────────────────
        self._local_model_server = get_server_from_config(cfg)
        if self._local_model_server:
            log.info(
                "[JarvisDaemon] Preloading maix-engine client: %s",
                self._local_model_server.url,
            )
            self._local_model_server.start()
        else:
            log.warning("[JarvisDaemon] maix-engine not configured, skipping preload")

        # ── Session tracking ───────────────────────────────────────────
        self._exchanges: list[dict[str, Any]] = []
        self._last_activity: float = time.time()

        # ── Two-stage pipeline queues ─────────────────────────────────
        # Stage 1: raw audio segments waiting for STT + query processing
        self._query_queue: queue.Queue[SegmentEvent] = queue.Queue(maxsize=3)
        self._max_queue_size = 3
        # Stage 2: (user_text, response_text, language, backend) ready for TTS
        self._response_queue: queue.Queue[tuple[str, str, str | None, str]] = (
            queue.Queue(maxsize=5)
        )
        # Sentinel to stop the worker thread on shutdown
        self._worker_stop = threading.Event()

        # Start the background query worker
        self._worker_thread = threading.Thread(
            target=self._query_worker,
            name="jarvis-query-worker",
            daemon=True,
        )
        self._worker_thread.start()

        log.info("[JarvisDaemon] All modules initialized (pipeline worker started)")

    def _ui_available_backends(self) -> list[str]:
        """Return UI-selectable backends in a stable user-facing order."""
        preferred_order = ("claude-p", "opencode", "groq", "maix-engine")
        available = set(self.router.get_available_backends())
        return [backend for backend in preferred_order if backend in available]

    def _on_backend_changed(self, backend: str) -> None:
        """Handle UI backend selection changes."""
        try:
            previous = self.router.get_default_backend()
            self.router.set_default_backend(backend)
            self.ui.configure_backend_selector(
                current_backend=self.router.get_default_backend(),
                available_backends=self._ui_available_backends(),
                on_backend_selected=self._on_backend_changed,
            )
            log.info(
                "[JarvisDaemon] Backend changed via UI: %s -> %s",
                previous,
                backend,
            )
        except Exception as exc:
            log.error("[JarvisDaemon] Failed to change backend to %s: %s", backend, exc)

    # ------------------------------------------------------------------
    # Background query worker (Stage 1 → Stage 2)
    # ------------------------------------------------------------------

    def _query_worker(self) -> None:
        """Background thread: pulls SegmentEvents from _query_queue,
        runs STT + QueryRouter, and puts results into _response_queue.

        This runs continuously so queries are processed even while TTS
        is playing on the main thread.
        """
        log.info(
            "[Worker] Query worker thread started (session=%s)", self.state.session_id
        )
        while not self._worker_stop.is_set():
            try:
                event = self._query_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                session_id = self.state.session_id
                log.info(
                    "[Worker] Processing query (session=%s, %.1fs audio)",
                    session_id,
                    event.duration_seconds,
                )

                # 1. STT — fast transcribe
                fast_text, language = self.stt.transcribe(event.audio, fast=True)
                if not fast_text or not fast_text.strip():
                    log.debug(
                        "[Worker] Empty transcription (fast), skipping (session=%s)",
                        session_id,
                    )
                    continue

                log.info(
                    "[Worker] Fast transcribe (session=%s, %s): %s",
                    session_id,
                    language,
                    fast_text[:100],
                )

                # Voice command: "Jarvis dormido" → finish queue then sleep
                if self._is_sleep_command(fast_text):
                    log.info(
                        "[Worker] Sleep command detected (session=%s): %r",
                        session_id,
                        fast_text,
                    )
                    self._response_queue.put(
                        (
                            fast_text,
                            "__SLEEP__",
                            language or "es",
                            "command",
                        )
                    )
                    continue

                text = fast_text
                if self._use_precise_stt:
                    text, language = self.stt.transcribe(event.audio, fast=False)
                    if not text or not text.strip():
                        text = fast_text
                    log.info(
                        "[Worker] Precise transcribe (session=%s, %s): %s",
                        session_id,
                        language,
                        text[:100],
                    )

                # 2. Enrich with Engram context
                enriched_prompt = self.engram.enrich_prompt(
                    text, language=language or "es"
                )

                # 3. Query backend (blocking — this is why we run in a thread)
                result = self.router.query(enriched_prompt)

                if not result.ok:
                    log.error(
                        "[Worker] Query failed (session=%s): %s",
                        session_id,
                        result.error,
                    )
                    self._response_queue.put(
                        (
                            text,
                            "Lo siento, hubo un error procesando tu consulta.",
                            language or "es",
                            "error",
                        )
                    )
                else:
                    log.info(
                        "[Worker] Response ready (session=%s, backend=%s, %.0fms): %s...",
                        session_id,
                        result.backend,
                        result.latency_ms,
                        result.text[:100],
                    )
                    self._response_queue.put(
                        (
                            text,
                            result.text,
                            language,
                            result.backend,
                        )
                    )

            except Exception:
                log.exception(
                    "[Worker] Unhandled exception processing query (session=%s)",
                    self.state.session_id,
                )

        log.info(
            "[Worker] Query worker thread stopped (session=%s)", self.state.session_id
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop — runs forever until interrupted.

        Iterates over ``AudioPipeline.stream_events()`` which yields
        ``WakeEvent`` and ``SegmentEvent`` objects.  Each event is
        dispatched to the appropriate handler based on type and current
        state.

        Between events, the loop drains _response_queue and plays TTS
        for any responses that the worker has prepared.
        """
        log.info("[JarvisDaemon] Starting main loop")

        try:
            for event in self.audio.stream_events():
                # ── Drain response queue (play ready responses) ───────
                if self.state.is_activo:
                    self._drain_response_queue()

                # ── Silence timeout check (only while ACTIVO) ─────────
                if self.state.is_activo:
                    both_queues_empty = (
                        self._query_queue.empty() and self._response_queue.empty()
                    )
                    if both_queues_empty and self.state.check_silence_timeout(
                        tts_playing=tts_module.is_speaking(),
                    ):
                        log.info(
                            "[JarvisDaemon] No speech for 5s, returning to DORMIDO"
                        )
                        self._deactivate()
                        continue

                    # ── Feed live audio levels to UI waveform ─────────
                    self._feed_ui_waveform()

                # ── Dispatch by event type ────────────────────────────
                if isinstance(event, WakeEvent):
                    self._handle_wake(event)
                elif isinstance(event, SegmentEvent):
                    self._handle_segment(event)
                elif isinstance(event, TickEvent):
                    continue

        except KeyboardInterrupt:
            log.info("[JarvisDaemon] Interrupted — shutting down")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_wake(self, event: WakeEvent) -> None:
        """Handle wake word detection → transition to ACTIVO."""
        if not self.state.is_dormido:
            log.info(
                "[JarvisDaemon] Duplicate wake ignored (session=%s)",
                self.state.session_id,
            )
            return

        self.state.activate()
        log.info(
            "[JarvisDaemon] Wake word detected (score=%.3f, session=%s)",
            event.score,
            self.state.session_id,
        )
        self.ui.send_command(UICommand("show"))
        self._speak_with_ui_feedback("Si, te escucho", "es")
        self._enter_active_listening("wake-greeting")

    def _handle_segment(self, event: SegmentEvent) -> None:
        """Handle a speech segment by enqueueing it for the background worker.

        ALL segments (foreground and background) go through _query_queue
        so the worker thread handles STT + query processing while the
        main thread stays free for TTS playback and UI updates.
        """
        if not self.state.is_activo:
            return

        self.state.record_audio_activity()

        # Send segment audio as waveform burst for visual feedback
        self.ui.send_command(UICommand("update_waveform", event.audio))
        self.ui.send_command(UICommand("set_state", "processing"))

        label = "background" if event.background else "foreground"
        log.info(
            "[JarvisDaemon] Enqueueing %s segment (session=%s, %.1fs audio, queue=%d)",
            label,
            self.state.session_id,
            event.duration_seconds,
            self._query_queue.qsize(),
        )

        try:
            self._query_queue.put_nowait(event)
        except queue.Full:
            try:
                self._query_queue.get_nowait()
                self._query_queue.put_nowait(event)
                log.warning(
                    "[JarvisDaemon] Queue full — dropped oldest, enqueued new segment (session=%s)",
                    self.state.session_id,
                )
            except queue.Empty:
                pass

    def _drain_response_queue(self) -> None:
        """Play all ready responses from the worker thread (Stage 2).

        Called from the main loop on every tick while ACTIVO.  Each
        response is spoken via TTS, the exchange is recorded, and then
        the next response (if any) plays immediately.
        """
        played = False
        while not self._response_queue.empty():
            try:
                user_text, response_text, language, backend = (
                    self._response_queue.get_nowait()
                )
            except queue.Empty:
                break

            # Handle sleep command: finish remaining responses then deactivate
            if response_text == "__SLEEP__":
                log.info(
                    "[JarvisDaemon] Sleep command: draining remaining responses then sleeping (session=%s)",
                    self.state.session_id,
                )
                # Drain any remaining responses first
                while not self._response_queue.empty():
                    try:
                        ut, rt, lg, bk = self._response_queue.get_nowait()
                        if rt != "__SLEEP__":
                            self._speak_with_ui_feedback(rt, lg or None)
                    except queue.Empty:
                        break
                self._speak_with_ui_feedback("Entendido, me voy a dormir.", "es")
                self._deactivate()
                return

            played = True
            log.info(
                "[TTS] Playing queued response (session=%s, %s): %s...",
                self.state.session_id,
                backend,
                response_text[:80],
            )

            # Speak on the main/daemon thread (coordinates mute window + UI)
            self._speak_with_ui_feedback(response_text, language or None)

            # Track exchange for session history
            self._exchanges.append(
                {
                    "user": user_text,
                    "response": response_text[:500],
                    "language": language,
                    "backend": backend,
                    "timestamp": time.time(),
                }
            )

        if played:
            self._enter_active_listening("response-complete")

    @staticmethod
    def _is_sleep_command(text: str) -> bool:
        """Check if the transcribed text is a sleep/deactivate command."""
        normalized = text.strip().lower()
        sleep_patterns = (
            "jarvis dormido",
            "jarvis dormir",
            "jarvis duerme",
            "jarvis apágate",
            "jarvis apagate",
            "jarvis para",
            "jarvis stop",
            "jarvis sleep",
        )
        return any(pattern in normalized for pattern in sleep_patterns)

    def _enter_active_listening(self, reason: str) -> None:
        """Return to active listening after TTS without requiring wake word."""
        self.audio.clear_mute_window()
        self.state.record_audio_activity()
        log.info(
            "[JarvisDaemon] TTS complete, entering active listening (session=%s, %s)",
            self.state.session_id,
            reason,
        )
        self.ui.send_command(UICommand("set_state", "listening"))

    # ------------------------------------------------------------------
    # UI waveform helpers
    # ------------------------------------------------------------------

    def _feed_ui_waveform(self, speaking_override: bool | None = None) -> None:
        """Send audio level data to the UI based on the current state.

        - **listening**: forward the last raw mic chunk from AudioPipeline
          so the waveform visualizes real microphone input.
        - **speaking**: generate a synthetic pulsing signal because V1
          cannot tap the TTS audio output (ffplay / edge-tts).
        - **processing / idle**: the UI handles its own animation.
        """
        if not self.state.is_activo:
            return

        speaking = (
            tts_module.is_speaking() if speaking_override is None else speaking_override
        )

        if speaking:
            # Speaking: synthesize a short audio-like chunk so the UI FFT sees
            # a stronger, continuously changing signal during TTS playback.
            t = time.time()
            sample_count = 1024
            x = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
            envelope = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 3.2 * t))
            shimmer = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 0.9 * t))
            pulse = envelope * (
                0.75 * np.sin(2.0 * math.pi * (3.0 + shimmer) * x + t * 10.0)
                + 0.45 * np.sin(2.0 * math.pi * (7.0 + shimmer * 1.5) * x + t * 16.0)
                + 0.25 * np.sin(2.0 * math.pi * 13.0 * x + t * 7.0)
            )
            self.ui.send_command(UICommand("update_waveform", pulse.astype(np.float32)))
        else:
            # Listening: forward real mic audio for live waveform
            chunk = self.audio.last_chunk
            if chunk is not None and len(chunk) > 0:
                self.ui.send_command(UICommand("update_waveform", chunk))

    def _speak_with_ui_feedback(self, text: str, language: str | None) -> float:
        """Run blocking TTS on a worker thread while keeping UI animation alive."""
        session_id = self.state.session_id
        self.ui.send_command(UICommand("set_state", "speaking"))
        log.info(
            "[TTS] Starting playback (session=%s, language=%s): %s...",
            session_id,
            language,
            text[:80],
        )

        result: dict[str, float] = {"duration": 0.0}

        def _run_tts() -> None:
            log.debug("[TTS] Worker thread speaking (session=%s)", session_id)
            result["duration"] = tts_module.speak(text, language)

        # Keep the microphone muted for the real TTS playback window, not an estimate.
        self.audio.set_mute_window(3600.0)
        worker = threading.Thread(target=_run_tts, name="jarvis-tts", daemon=True)
        worker.start()

        try:
            while worker.is_alive():
                self._feed_ui_waveform(speaking_override=True)
                time.sleep(0.05)

            worker.join()
        finally:
            self.audio.clear_mute_window()

        log.info(
            "[TTS] Playback finished (session=%s, duration=%.2fs)",
            session_id,
            result["duration"],
        )

        return result["duration"]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _deactivate(self) -> None:
        """Transition to DORMIDO: hide UI, save session to Engram.

        Drains any remaining responses before deactivating so no work
        is silently lost.
        """
        session_id = self.state.session_id
        # Play any remaining queued responses before going dormant
        self._drain_response_queue()

        log.info(
            "[JarvisDaemon] Deactivating session=%s — saving session with %d exchanges",
            session_id,
            len(self._exchanges),
        )
        self.router.clear_history()
        self.state.deactivate()
        log.info("[JarvisDaemon] Session %s ended", session_id)
        self.ui.send_command(
            UICommand("update_waveform", np.zeros(48, dtype=np.float32))
        )
        self.ui.send_command(UICommand("set_state", "idle"))
        self.ui.send_command(UICommand("hide"))

        # Persist session to Engram memory
        if self._exchanges:
            try:
                self.engram.save_session_summary(self._exchanges)
            except Exception as exc:
                log.error(
                    "[JarvisDaemon] Failed to save session %s: %s",
                    session_id,
                    exc,
                )
            self._exchanges.clear()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Clean shutdown of all modules."""
        log.info("[JarvisDaemon] Shutting down (session=%s)", self.state.session_id)

        # Stop the query worker thread
        self._worker_stop.set()
        self._worker_thread.join(timeout=3.0)
        if self._worker_thread.is_alive():
            log.warning(
                "[JarvisDaemon] Worker thread did not stop in time (session=%s)",
                self.state.session_id,
            )

        # Save any remaining exchanges before exit
        if self._exchanges:
            try:
                self.engram.save_session_summary(self._exchanges)
            except Exception:
                pass

        # Stop local model server if running
        if self._local_model_server:
            self._local_model_server.stop()

        self.audio.close()
        log.info("[JarvisDaemon] Shutdown complete (session=%s)", self.state.session_id)
