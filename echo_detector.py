"""Text-based TTS echo detection for STT transcripts."""

from __future__ import annotations

import logging
import re
import threading
import time
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - optional dependency
    fuzz = None

log = logging.getLogger(__name__)


class EchoDetector:
    """Track recent TTS utterances and filter echoed transcripts."""

    _MATCH_THRESHOLD = 0.7
    _COOLDOWN_SECONDS = 0.3
    _DELAYED_ECHO_SECONDS = 1.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent_tts: list[dict[str, float | str]] = []

    def track_tts_start(self, text: str, duration: float) -> None:
        """Register an utterance before playback starts."""
        normalized = self._normalize(text)
        if not normalized:
            return

        now = time.monotonic()
        duration = max(0.0, float(duration))
        with self._lock:
            self._prune(now)
            self._recent_tts.append(
                {
                    "text": text,
                    "normalized": normalized,
                    "start": now,
                    "end": now + duration,
                    "cooldown_end": now + duration + self._COOLDOWN_SECONDS,
                    "expires": now + duration + self._DELAYED_ECHO_SECONDS,
                }
            )

    def check(self, transcription: str) -> str | None:
        """Return cleaned text or ``None`` when the transcript is echo only."""
        normalized = self._normalize(transcription)
        if not normalized:
            return transcription

        candidates = self._active_candidates()
        if not candidates:
            return transcription

        cleaned = self.cleanup_leading_echo(transcription)
        if cleaned != transcription:
            log.info("Echo prefix stripped, salvaged: %s", cleaned)
            return cleaned

        for candidate in candidates:
            score = self._ratio(normalized, str(candidate["normalized"]))
            if score > self._MATCH_THRESHOLD:
                log.info("Echo detected and discarded: %s", transcription)
                return None

        return transcription

    def cleanup_leading_echo(self, transcription: str) -> str:
        """Strip an echoed TTS prefix when useful speech follows it."""
        words = transcription.strip().split()
        if len(words) < 2:
            return transcription

        best_remainder = transcription
        best_length = 0
        for candidate in self._active_candidates():
            tts_words = str(candidate["text"]).strip().split()
            max_prefix = min(len(words) - 1, len(tts_words))
            for size in range(max_prefix, 0, -1):
                spoken_prefix = " ".join(tts_words[:size])
                transcript_prefix = " ".join(words[:size])
                score = self._ratio(
                    self._normalize(transcript_prefix),
                    self._normalize(spoken_prefix),
                )
                if score <= self._MATCH_THRESHOLD:
                    continue

                remainder = " ".join(words[size:]).strip()
                if remainder and size > best_length:
                    best_remainder = remainder
                    best_length = size
                break

        return best_remainder

    def _active_candidates(self) -> list[dict[str, float | str]]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return [item for item in self._recent_tts if self._window_for(item, now)]

    def _prune(self, now: float) -> None:
        self._recent_tts = [item for item in self._recent_tts if now <= item["expires"]]

    @classmethod
    def _window_for(cls, item: dict[str, float | str], now: float) -> str | None:
        if now <= item["end"]:
            return "during_tts"
        if now <= item["cooldown_end"]:
            return "cooldown"
        if now <= item["expires"]:
            return "delayed_echo"
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"[^\w\sáéíóúñü¿¡]", " ", lowered)
        return " ".join(lowered.split())

    @staticmethod
    def _ratio(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if fuzz is not None:
            return fuzz.ratio(left, right) / 100.0
        return SequenceMatcher(None, left, right).ratio()
