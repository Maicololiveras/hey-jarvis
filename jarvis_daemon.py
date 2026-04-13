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
import time
from typing import Any

import numpy as np

from . import config
from .models import WakeEvent, SegmentEvent, UICommand
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
        setup_logging()

        cfg = config.load()
        jarvis_cfg: dict[str, Any] = cfg.get("jarvis", {})

        # ── Module initialization ──────────────────────────────────────
        self.state = StateMachine(
            silence_timeout=jarvis_cfg.get("silence_timeout_seconds", 5.0),
        )
        self.audio = AudioPipeline(
            config.get_audio_config(),
            config.get_wake_word_config(),
        )

        stt_cfg = config.get_stt_config()
        self.stt = STT(
            model_path=stt_cfg.get("model_path", ""),
            device=stt_cfg.get("device", "cpu"),
            compute_type=stt_cfg.get("compute_type", "int8"),
        )

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
        self.ui.send_command(UICommand("set_state", "listening"))

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

        # 1. Transcribe ─────────────────────────────────────────────────
        self.ui.send_command(UICommand("set_state", "processing"))
        text, language = self.stt.transcribe(event.audio)

        if not text or not text.strip():
            log.debug("[JarvisDaemon] Empty transcription, ignoring")
            self.ui.send_command(UICommand("set_state", "listening"))
            return

        log.info(
            "[JarvisDaemon] Transcribed (%s): %s",
            language,
            text[:100],
        )

        # 2. Enrich with Engram context ─────────────────────────────────
        enriched_prompt = self.engram.enrich_prompt(text)

        # 3. Query backend ──────────────────────────────────────────────
        result = self.router.query(enriched_prompt)

        if not result.ok:
            log.error("[JarvisDaemon] Query failed: %s", result.error)
            tts_module.speak(
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
        self.ui.send_command(UICommand("set_state", "speaking"))
        duration = tts_module.speak(result.text, language or None)

        # Mute the mic for the TTS duration + 1s buffer to avoid echo.
        self.audio.set_mute_window(duration + 1.0)

        # 5. Track exchange ─────────────────────────────────────────────
        self._exchanges.append({
            "user": text,
            "response": result.text[:500],
            "language": language,
            "backend": result.backend,
            "timestamp": time.time(),
        })

        # 6. Back to listening ──────────────────────────────────────────
        self.state.record_audio_activity()
        self.ui.send_command(UICommand("set_state", "listening"))

    # ------------------------------------------------------------------
    # UI waveform helpers
    # ------------------------------------------------------------------

    def _feed_ui_waveform(self) -> None:
        """Send audio level data to the UI based on the current state.

        - **listening**: forward the last raw mic chunk from AudioPipeline
          so the waveform visualizes real microphone input.
        - **speaking**: generate a synthetic pulsing signal because V1
          cannot tap the TTS audio output (ffplay / edge-tts).
        - **processing / idle**: the UI handles its own animation.
        """
        if not self.state.is_activo:
            return

        if tts_module.is_speaking():
            # Speaking: synthetic pulse (can't tap ffplay output in V1)
            t = time.time()
            num_bands = 48
            pulse = np.array([
                0.3 + 0.7 * abs(math.sin(2.0 * math.pi * 2.0 * t + i * 0.15))
                for i in range(num_bands)
            ], dtype=np.float32)
            self.ui.send_command(UICommand("update_waveform", pulse))
        else:
            # Listening: forward real mic audio for live waveform
            chunk = self.audio.last_chunk
            if chunk is not None and len(chunk) > 0:
                self.ui.send_command(UICommand("update_waveform", chunk))

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
        self.ui.send_command(UICommand("update_waveform", np.zeros(48, dtype=np.float32)))
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
