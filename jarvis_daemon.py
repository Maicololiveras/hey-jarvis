"""Jarvis daemon — main orchestrator loop.

Wires together all Jarvis modules:
  AudioPipeline  →  WakeEvent / SegmentEvent / TickEvent
  StateMachine   →  DORMIDO / ACTIVO transitions
  STT            →  speech-to-text via faster-whisper
  QueryRouter    →  AI backend dispatch
  EngramBridge   →  memory enrichment + session persistence
  TTS            →  edge-tts internal playback
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
from . import tts as tts_module
from .audio_pipeline import AudioPipeline
from .engram_bridge import EngramBridge
from .jarvis_ui import JarvisUI
from .models import SegmentEvent, TickEvent, UICommand, WakeEvent
from .query_router import QueryRouter
from .state_machine import StateMachine
from .stt import STT

log = logging.getLogger(__name__)


class QueryTask:
    """A single user segment moving through STT -> router -> TTS."""

    def __init__(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
        is_followup: bool,
    ) -> None:
        self.audio = audio
        self.text = text
        self.language = language
        self.is_followup = is_followup
        self.created_at = time.time()
        self.future: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.response: dict[str, Any] | None = None

    def complete(self, response_text: str, language: str | None, backend: str) -> None:
        self.future.put_nowait(
            {
                "response": response_text,
                "language": language,
                "backend": backend,
                "ok": True,
                "error": None,
            }
        )

    def fail(self, error: str) -> None:
        self.future.put_nowait(
            {
                "response": "",
                "language": self.language,
                "backend": "",
                "ok": False,
                "error": error,
            }
        )

    def poll_response(self) -> dict[str, Any] | None:
        if self.response is not None:
            return self.response
        try:
            self.response = self.future.get_nowait()
        except queue.Empty:
            return None
        return self.response


class QueryQueue:
    """Thread-safe queue that processes queries sequentially via a worker thread."""

    def __init__(self, router: QueryRouter, engram: EngramBridge) -> None:
        self._queue: queue.Queue[QueryTask] = queue.Queue(maxsize=10)
        self._router = router
        self._engram = engram
        self._worker: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="jarvis-query-queue",
            daemon=True,
        )
        self._worker.start()
        log.info("[QueryQueue] Worker started")

    def stop(self) -> None:
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        log.info("[QueryQueue] Worker stopped")

    def enqueue(self, task: QueryTask) -> None:
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            task.fail("cola llena, intenta de nuevo")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                enriched_prompt = self._engram.enrich_prompt(
                    task.text,
                    language=task.language or "es",
                )
                result = self._router.query(enriched_prompt)
                if result.ok:
                    task.complete(result.text, task.language, result.backend)
                else:
                    task.complete(
                        f"Lo siento, hubo un error: {result.error}",
                        task.language,
                        result.backend,
                    )
            except Exception as exc:
                log.exception("[QueryQueue] Task error: %s", exc)
                task.fail(str(exc))


class JarvisDaemon:
    """Main orchestrator that keeps ACTIVO responsive while work happens."""

    def __init__(self, ui: JarvisUI | None = None) -> None:
        cfg = config.load()
        jarvis_cfg: dict[str, Any] = cfg.get("jarvis", {})
        stt_cfg = config.get_stt_config()

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
        self.stt.preload(fast=True, precise=self._use_precise_stt)

        self.router = QueryRouter(config.get_query_config())
        self.engram = EngramBridge(
            config.get_engram_config(),
            config.get_query_config(),
        )
        self.query_queue = QueryQueue(self.router, self.engram)
        self.query_queue.start()

        self.ui: JarvisUI = ui if ui is not None else JarvisUI()
        self.ui.configure_backend_selector(
            current_backend=self.router.get_default_backend(),
            available_backends=self.router.get_available_backends(),
            on_backend_selected=self.set_active_backend,
        )

        self._pending_tasks: list[QueryTask] = []
        self._speech_worker: threading.Thread | None = None
        self._speech_result: dict[str, Any] = {"duration": 0.0}
        self._speech_task: QueryTask | None = None
        self._speech_text: str = ""
        self._speech_language: str | None = None

        self._exchanges: list[dict[str, Any]] = []
        self._last_activity: float = time.time()

        log.info("[JarvisDaemon] All modules initialized")

    def set_active_backend(self, backend: str) -> None:
        """Change the active query backend for future requests."""
        self.router.set_default_backend(backend)
        self.ui.configure_backend_selector(
            current_backend=backend,
            available_backends=self.router.get_available_backends(),
            on_backend_selected=self.set_active_backend,
        )

        try:
            cfg = config.load()
            cfg.setdefault("query", {})["default_backend"] = backend
            config.save(cfg)
            log.info("[JarvisDaemon] Active backend set to %s", backend)
        except Exception:
            log.exception("[JarvisDaemon] Failed to persist active backend %s", backend)

    def run(self) -> None:
        """Main loop — keeps processing audio while query/TTS work continues."""
        log.info("[JarvisDaemon] Starting main loop")

        try:
            for event in self.audio.stream_events():
                self._advance_async_state()

                if isinstance(event, SegmentEvent):
                    self._handle_segment(event)
                    self._advance_async_state()
                    continue

                if self.state.is_activo:
                    self._feed_ui_waveform()
                    if (
                        not self._pending_tasks
                        and self._speech_worker is None
                        and self.state.check_silence_timeout(
                            tts_playing=tts_module.is_speaking(),
                        )
                    ):
                        self._deactivate()
                        continue

                if isinstance(event, WakeEvent):
                    self._handle_wake(event)
                elif isinstance(event, TickEvent):
                    continue

        except KeyboardInterrupt:
            log.info("[JarvisDaemon] Interrupted — shutting down")
        finally:
            self._shutdown()

    def _handle_wake(self, event: WakeEvent) -> None:
        """Handle wake word detection -> transition to ACTIVO."""
        if not self.state.is_dormido:
            return

        log.info("[JarvisDaemon] Wake word detected (score=%.3f)", event.score)
        self.state.clear_context()
        self.router.clear_history()
        self.state.activate()
        self.ui.send_command(UICommand("show"))
        self.state.record_audio_activity()
        self._start_speaking("Si, te escucho", "es")

    def _handle_segment(self, event: SegmentEvent) -> None:
        """Handle a speech segment: STT -> async query queue -> later TTS."""
        if not self.state.is_activo:
            return

        self.state.record_audio_activity()
        self.ui.send_command(UICommand("update_waveform", event.audio))
        if self._speech_worker is None:
            self.ui.send_command(UICommand("set_state", "processing"))

        fast_text, language = self.stt.transcribe(event.audio, fast=True)
        if not fast_text or not fast_text.strip():
            log.debug("[JarvisDaemon] Empty transcription (fast), ignoring")
            self._set_idle_active_ui_state()
            return

        log.info("[JarvisDaemon] Fast transcribe (%s): %s", language, fast_text[:100])

        text = fast_text
        if self._use_precise_stt:
            precise_text, precise_language = self.stt.transcribe(
                event.audio, fast=False
            )
            if precise_text and precise_text.strip():
                text = precise_text
                language = precise_language
            log.info("[JarvisDaemon] Precise transcribe (%s): %s", language, text[:100])

        task = QueryTask(
            audio=event.audio,
            text=text,
            language=language,
            is_followup=self._is_followup(text),
        )
        self._pending_tasks.append(task)
        self.query_queue.enqueue(task)

        queue_depth = len(self._pending_tasks)
        if queue_depth > 1 or self._speech_worker is not None:
            log.info(
                "[JarvisDaemon] Queued follow-up request (depth=%d): %s",
                queue_depth,
                text[:80],
            )

        self._set_idle_active_ui_state()

    def _advance_async_state(self) -> None:
        """Advance background query/TTS work without blocking the audio loop."""
        if self._speech_worker is not None:
            if self._speech_worker.is_alive():
                return
            self._speech_worker.join()
            self._speech_worker = None
            self._finalize_speaking()

        if not self.state.is_activo:
            return

        if not self._pending_tasks:
            self._set_idle_active_ui_state()
            return

        current = self._pending_tasks[0]
        if current.response is None:
            if (time.time() - current.created_at) > 60.0:
                current.fail("timeout waiting for backend response")
            else:
                self.ui.send_command(UICommand("set_state", "processing"))
                return

        response = current.poll_response()
        if response is None:
            self.ui.send_command(UICommand("set_state", "processing"))
            return

        self._pending_tasks.pop(0)
        self._speak_task_response(current, response)

    def _speak_task_response(
        self,
        task: QueryTask,
        response: dict[str, Any],
    ) -> None:
        result_text = (
            response.get("response")
            or "Lo siento, hubo un error procesando tu consulta."
        )
        result_language = response.get("language") or task.language or "es"
        result_backend = response.get("backend") or self.router.get_default_backend()

        if response.get("ok"):
            log.info(
                "[JarvisDaemon] Response from %s: %s...",
                result_backend,
                result_text[:100],
            )
        else:
            log.error("[JarvisDaemon] Query failed: %s", response.get("error"))

        response["backend"] = result_backend
        response["language"] = result_language
        response["response"] = result_text
        task.response = response
        self._start_speaking(result_text, result_language, task=task)

    def _start_speaking(
        self,
        text: str,
        language: str | None,
        task: QueryTask | None = None,
    ) -> None:
        """Start TTS on a worker thread so the mic loop keeps running."""
        if self._speech_worker is not None:
            raise RuntimeError("TTS worker already active")

        self._speech_task = task
        self._speech_text = text
        self._speech_language = language
        self._speech_result = {"duration": 0.0}
        self.ui.send_command(UICommand("set_state", "speaking"))

        def _run_tts() -> None:
            self._speech_result["duration"] = tts_module.speak(text, language)

        self._speech_worker = threading.Thread(
            target=_run_tts,
            name="jarvis-tts",
            daemon=True,
        )
        self._speech_worker.start()

    def _finalize_speaking(self) -> None:
        """Finalize TTS completion and resume ACTIVO listening/queue processing."""
        task = self._speech_task
        self._speech_task = None

        if task is not None and task.response is not None:
            response = task.response
            result_text = response.get("response", "")
            result_language = response.get("language") or task.language
            result_backend = response.get("backend", "")

            self._exchanges.append(
                {
                    "user": task.text,
                    "response": result_text[:500],
                    "language": result_language,
                    "backend": result_backend,
                    "timestamp": time.time(),
                    "is_followup": task.is_followup,
                }
            )
            self.state.increment_turn()
            self.state.set_context(task.text, result_text)

        self._speech_text = ""
        self._speech_language = None
        self._set_idle_active_ui_state()

    def _set_idle_active_ui_state(self) -> None:
        """Update UI state while Jarvis remains ACTIVO."""
        if not self.state.is_activo:
            return
        if self._speech_worker is not None or tts_module.is_speaking():
            self.ui.send_command(UICommand("set_state", "speaking"))
        elif self._pending_tasks:
            self.ui.send_command(UICommand("set_state", "processing"))
        else:
            self.ui.send_command(UICommand("set_state", "listening"))

    def _feed_ui_waveform(self, speaking_override: bool | None = None) -> None:
        """Send audio level data to the UI based on the current state."""
        if not self.state.is_activo:
            return

        speaking = (
            tts_module.is_speaking() if speaking_override is None else speaking_override
        )

        if speaking:
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
            return

        chunk = self.audio.last_chunk
        if chunk is not None and len(chunk) > 0:
            self.ui.send_command(UICommand("update_waveform", chunk))

    def _is_followup(self, text: str) -> bool:
        """Detect if this is a follow-up in a multi-turn conversation."""
        if not self._exchanges:
            return False

        text_lower = text.lower().strip()
        followup_indicators = (
            "y",
            "ademas",
            "tambien",
            "sobre eso",
            "mas",
            "pero",
            "entonces",
            "ok",
            "bueno",
            "si",
            "si claro",
            "perfecto",
            "ah",
            "ahhh",
            "entiendo",
            "ya",
            "ya veo",
            "continua",
            "sigue",
        )

        for indicator in followup_indicators:
            if text_lower.startswith(indicator) or text_lower == indicator:
                return True

        return text_lower in ("eso", "eso mismo", "por que", "como")

    def _deactivate(self) -> None:
        """Transition to DORMIDO: hide UI and save the session."""
        log.info(
            "[JarvisDaemon] Deactivating — saving session with %d exchanges",
            len(self._exchanges),
        )
        self.state.clear_context()
        self.router.clear_history()
        self.state.deactivate()
        self.ui.send_command(
            UICommand("update_waveform", np.zeros(48, dtype=np.float32))
        )
        self.ui.send_command(UICommand("set_state", "idle"))
        self.ui.send_command(UICommand("hide"))

        if self._exchanges:
            try:
                self.engram.save_session_summary(self._exchanges)
            except Exception as exc:
                log.error("[JarvisDaemon] Failed to save session: %s", exc)
            self._exchanges.clear()

    def _shutdown(self) -> None:
        """Clean shutdown of all modules."""
        log.info("[JarvisDaemon] Shutting down...")

        if self._speech_worker and self._speech_worker.is_alive():
            self._speech_worker.join(timeout=5.0)

        if self._exchanges:
            try:
                self.engram.save_session_summary(self._exchanges)
            except Exception:
                pass

        self.query_queue.stop()
        self.audio.close()
        log.info("[JarvisDaemon] Shutdown complete")
