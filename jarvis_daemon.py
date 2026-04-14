"""Jarvis daemon — main orchestrator loop.

Wires together all Jarvis modules:
  AudioPipeline  →  WakeEvent / SegmentEvent
  StateMachine   →  DORMIDO / ACTIVO transitions
  STT            →  speech-to-text via faster-whisper
  QueryRouter    →  AI backend dispatch (claude-p, local-qwen)
  EngramBridge   →  memory enrichment + session persistence
  TTS (functions)→  edge-tts / pyttsx3 speech output
  JarvisUI       →  circular visual feedback window
"""

from __future__ import annotations

import logging
import math
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

        # ── Session tracking ───────────────────────────────────────────
        self._exchanges: list[dict[str, Any]] = []
        self._last_activity: float = time.time()

        log.info("[JarvisDaemon] All modules initialized")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop — runs forever until interrupted.

        Iterates over ``AudioPipeline.stream_events()`` which yields
        ``WakeEvent`` and ``SegmentEvent`` objects.  Each event is
        dispatched to the appropriate handler based on type and current
        state.
        """
        log.info("[JarvisDaemon] Starting main loop")

        try:
            for event in self.audio.stream_events():
                # ── Silence timeout check (only while ACTIVO) ─────────
                if self.state.is_activo:
                    if self.state.check_silence_timeout(
                        tts_playing=tts_module.is_speaking(),
                    ):
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
            return

        log.info(
            "[JarvisDaemon] Wake word detected (score=%.3f)",
            event.score,
        )
        self.state.activate()
        self.ui.send_command(UICommand("show"))
        duration = self._speak_with_ui_feedback("Si, te escucho", "es")
        # Short mute to avoid echo, but don't over-mute — user needs to talk right after
        self.audio.set_mute_window(duration + 0.3)
        self.ui.send_command(UICommand("set_state", "listening"))
        # Reset audio activity timer so silence timeout starts AFTER greeting
        self.state.record_audio_activity()

    def _handle_segment(self, event: SegmentEvent) -> None:
        """Handle a speech segment: STT → enrich → route → TTS.

        Only processes segments while in ACTIVO state.  Short or empty
        transcriptions are silently ignored.
        """
        if not self.state.is_activo:
            return

        self.state.record_audio_activity()

        # Send segment audio as final waveform burst before processing
        self.ui.send_command(UICommand("update_waveform", event.audio))

        # 1. Fast transcribe (base model) — quick check ────────────────
        self.ui.send_command(UICommand("set_state", "processing"))
        fast_text, language = self.stt.transcribe(event.audio, fast=True)

        if not fast_text or not fast_text.strip():
            log.debug("[JarvisDaemon] Empty transcription (fast), ignoring")
            self.ui.send_command(UICommand("set_state", "listening"))
            return

        log.info(
            "[JarvisDaemon] Fast transcribe (%s): %s",
            language,
            fast_text[:100],
        )

        text = fast_text
        if self._use_precise_stt:
            # The precise pass improves accuracy, but it adds noticeable
            # latency. Keep it opt-in for production responsiveness.
            text, language = self.stt.transcribe(event.audio, fast=False)

            if not text or not text.strip():
                text = fast_text  # fallback to fast if medium returns empty

            log.info(
                "[JarvisDaemon] Precise transcribe (%s): %s",
                language,
                text[:100],
            )

        # 3. Enrich with Engram context ─────────────────────────────────
        enriched_prompt = self.engram.enrich_prompt(text, language=language or "es")

        # 3. Query backend ──────────────────────────────────────────────
        result = self.router.query(enriched_prompt)

        if not result.ok:
            log.error("[JarvisDaemon] Query failed: %s", result.error)
            self._speak_with_ui_feedback(
                "Lo siento, hubo un error procesando tu consulta.",
                language or "es",
            )
            self.ui.send_command(UICommand("set_state", "listening"))
            return

        log.info(
            "[JarvisDaemon] Response from %s (%.0fms): %s...",
            result.backend,
            result.latency_ms,
            result.text[:100],
        )

        # 4. Speak response ─────────────────────────────────────────────
        duration = self._speak_with_ui_feedback(result.text, language or None)

        # Mute the mic for the TTS duration + 1s buffer to avoid echo.
        self.audio.set_mute_window(duration + 1.0)

        # 5. Track exchange ─────────────────────────────────────────────
        self._exchanges.append(
            {
                "user": text,
                "response": result.text[:500],
                "language": language,
                "backend": result.backend,
                "timestamp": time.time(),
            }
        )

        # 6. Back to listening ──────────────────────────────────────────
        self.state.record_audio_activity()
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
        self.ui.send_command(UICommand("set_state", "speaking"))

        result: dict[str, float] = {"duration": 0.0}

        def _run_tts() -> None:
            result["duration"] = tts_module.speak(text, language)

        worker = threading.Thread(target=_run_tts, name="jarvis-tts", daemon=True)
        worker.start()

        while worker.is_alive():
            self._feed_ui_waveform(speaking_override=True)
            time.sleep(0.05)

        worker.join()
        return result["duration"]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _deactivate(self) -> None:
        """Transition to DORMIDO: hide UI, save session to Engram."""
        log.info(
            "[JarvisDaemon] Deactivating — saving session with %d exchanges",
            len(self._exchanges),
        )
        self.state.deactivate()
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
                log.error("[JarvisDaemon] Failed to save session: %s", exc)
            self._exchanges.clear()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Clean shutdown of all modules."""
        log.info("[JarvisDaemon] Shutting down...")

        # Save any remaining exchanges before exit
        if self._exchanges:
            try:
                self.engram.save_session_summary(self._exchanges)
            except Exception:
                pass

        self.audio.close()
        log.info("[JarvisDaemon] Shutdown complete")
