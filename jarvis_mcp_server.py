"""FastMCP entrypoint exposing thin wrappers around Jarvis modules."""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import re
from threading import RLock
from typing import Iterator

_REPO_ROOT = Path(__file__).resolve().parent
_PARENT_DIR = _REPO_ROOT.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import numpy as np
from fastmcp import FastMCP

from jarvis import config, tts
from jarvis.audio_pipeline import select_audio_device
from jarvis.engram_bridge import EngramBridge
from jarvis.logging_setup import LOG_FILE
from jarvis.query_router import QueryRouter
from jarvis.state_machine import State, StateMachine
from jarvis.stt import STT


class JarvisMCPService:
    """Thin wrapper service reusing the existing Jarvis modules."""

    def __init__(self) -> None:
        cfg = config.load()
        jarvis_cfg = cfg.get("jarvis", {})
        stt_cfg = config.get_stt_config()

        self._state_machine = StateMachine(
            silence_timeout=jarvis_cfg.get("silence_timeout_seconds", 5.0),
        )
        self._router = QueryRouter(config.get_query_config())
        self._engram = EngramBridge(
            config.get_engram_config(),
            config.get_query_config(),
        )
        self._stt = STT(
            model_path=stt_cfg.get("model_path", ""),
            device=stt_cfg.get("device", "cpu"),
            compute_type=stt_cfg.get("compute_type", "int8"),
            fast_model_path=stt_cfg.get("fast_model_path", ""),
            engine=stt_cfg.get("engine", "faster-whisper"),
            api_key_env=stt_cfg.get("api_key_env", "GROQ_API_KEY"),
            language=jarvis_cfg.get("language", "auto"),
            groq_model=stt_cfg.get("model", "whisper-large-v3"),
        )
        self._lock = RLock()
        self._processing = 0

    @contextmanager
    def _processing_scope(self) -> Iterator[None]:
        with self._lock:
            self._state_machine.check_error_timeout()
            if self._state_machine.is_error:
                raise RuntimeError("Jarvis is recovering from an error")
            if self._processing == 0 and self._state_machine.is_dormido:
                self._state_machine.activate()
            self._processing += 1

        try:
            yield
        finally:
            with self._lock:
                self._processing = max(0, self._processing - 1)
                if (
                    self._processing == 0
                    and self._state_machine.has_active_session
                    and not self._state_machine.is_error
                ):
                    self._state_machine.deactivate()

    def _detect_prompt_language(self, prompt: str) -> str:
        configured = config.load().get("jarvis", {}).get("language", "auto")
        if configured and configured != "auto":
            return str(configured)
        return tts.detect_language(prompt)

    def _jarvis_process_running(self) -> bool:
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-Command",
                    "Get-CimInstance Win32_Process | Where-Object "
                    "{ $_.CommandLine -like '*-m jarvis*' -and $_.Name -eq 'python.exe' } | "
                    "Select-Object -ExpandProperty ProcessId",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            return False

        return bool(result.stdout.strip())

    def _status_from_log(self) -> str | None:
        if not self._jarvis_process_running() or not LOG_FILE.exists():
            return None

        try:
            lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None

        state_pattern = re.compile(r"State: [A-Z]+ -> ([A-Z]+)")
        for line in reversed(lines[-400:]):
            match = state_pattern.search(line)
            if match:
                return match.group(1)

        return None

    def status(self) -> dict[str, str]:
        self._state_machine.check_error_timeout()
        with self._lock:
            if self._processing > 0:
                return {
                    "state": self._state_machine.state.value,
                    "source": "mcp-wrapper",
                }

        logged = self._status_from_log()
        if logged is not None:
            return {"state": logged, "source": str(LOG_FILE)}

        return {
            "state": self._state_machine.state.value,
            "source": "local-state-machine",
        }

    def speak(self, text: str, language: str = "es") -> dict[str, object]:
        with self._lock:
            self._state_machine.check_error_timeout()
            if self._state_machine.is_error:
                raise RuntimeError("Jarvis is recovering from an error")
            should_deactivate = False
            if self._state_machine.is_dormido:
                self._state_machine.activate()
                should_deactivate = True
            self._state_machine.start_speaking()

        try:
            duration = tts.speak(text, language)
            return {
                "ok": True,
                "text": text,
                "language": language,
                "estimated_duration_seconds": duration,
            }
        except Exception:
            self._state_machine.enter_error()
            raise
        finally:
            with self._lock:
                if self._state_machine.state is State.HABLANDO:
                    self._state_machine.finish_speaking()
                if (
                    should_deactivate
                    and self._state_machine.has_active_session
                    and not self._state_machine.is_error
                ):
                    self._state_machine.deactivate()

    def query(self, prompt: str, backend: str = "auto") -> dict[str, object]:
        selected_backend = None if backend == "auto" else backend
        language = self._detect_prompt_language(prompt)

        with self._processing_scope():
            self._state_machine.start_processing()
            try:
                if selected_backend is None:
                    enriched_prompt = self._engram.enrich_prompt(
                        prompt, language=language
                    )
                else:
                    enriched_prompt = self._engram.enrich_prompt(
                        prompt,
                        language=language,
                        backend=selected_backend,
                    )
                result = self._router.query(enriched_prompt, backend=selected_backend)
            except Exception:
                self._state_machine.enter_error()
                raise

        return {
            "ok": result.ok,
            "text": result.text,
            "backend": result.backend
            or (selected_backend or self._router.get_default_backend()),
            "latency_ms": result.latency_ms,
            "error": result.error,
        }

    def listen(self, duration_seconds: int = 5) -> dict[str, object]:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than 0")

        import sounddevice as sd

        audio_cfg = config.get_audio_config()
        sample_rate = int(audio_cfg.get("sample_rate", 16000))
        channels = int(audio_cfg.get("channels", 1))
        device_index, device_name = select_audio_device(audio_cfg)
        frame_count = int(sample_rate * duration_seconds)

        with self._processing_scope():
            self._state_machine.start_listening()
            try:
                audio = sd.rec(
                    frame_count,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                    device=device_index,
                )
                sd.wait()

                if channels > 1:
                    mono_audio = np.mean(audio, axis=1, dtype=np.float32)
                else:
                    mono_audio = audio.reshape(-1).astype(np.float32)

                text, language = self._stt.transcribe(
                    mono_audio, strict=False, fast=False
                )
                if text and text.strip():
                    self._state_machine.receive_text()
            except Exception:
                self._state_machine.enter_error()
                raise

        return {
            "ok": True,
            "text": text,
            "language": language,
            "duration_seconds": duration_seconds,
            "device": device_name,
        }


service = JarvisMCPService()
mcp = FastMCP(
    "Hey Jarvis MCP",
    instructions="Thin FastMCP wrappers around the existing Jarvis TTS, STT, query, and state modules.",
)


@mcp.tool
def jarvis_speak(text: str, language: str = "es") -> dict[str, object]:
    """Speak text through the existing Jarvis TTS module."""

    return service.speak(text, language)


@mcp.tool
def jarvis_query(prompt: str, backend: str = "auto") -> dict[str, object]:
    """Query the existing Jarvis router and return the backend response."""

    return service.query(prompt, backend)


@mcp.tool
def jarvis_status() -> dict[str, str]:
    """Return the best available Jarvis state snapshot."""

    return service.status()


@mcp.tool
def jarvis_listen(duration_seconds: int = 5) -> dict[str, object]:
    """Record audio briefly and transcribe it through the existing Jarvis STT."""

    return service.listen(duration_seconds)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
