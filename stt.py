"""Speech-to-Text module — wraps faster-whisper transcription.

Extracted from claudio_daemon.py (TASK-006).  Key improvements over the
original inline code:

- Lazy model loading (first ``transcribe`` call, not import time).
- Config-driven: model_path / device / compute_type taken from config.
- Language auto-detected by Whisper (no hardcoded ``language="es"``).
- Returns ``(text, language_code)`` so callers know the detected language.
- Proper ``logging`` instead of ``print()``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

# Sample rate expected by Whisper models.
_WHISPER_SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# Hallucination catalogue
# ---------------------------------------------------------------------------

# Canned phrases that Whisper emits on silence / noise.
_CANNED_HALLUCINATIONS: tuple[str, ...] = (
    "subtítulos",
    "subtitulos",
    "gracias por ver",
    "subtítulos realizados",
    "amara.org",
    "amara .org",
    "www.",
    "transcripción en español",
    "transcripcion en español",
    "palabras esperadas",
    "en español rioplatense",
)

# Lightweight subset used in non-strict (menu) mode.
_CANNED_HALLUCINATIONS_LIGHT: tuple[str, ...] = (
    "subtítulos",
    "gracias por ver",
    "amara.org",
)


class STT:
    """Speech-to-text engine backed by *faster-whisper*.

    The model is loaded lazily on the first call to :meth:`transcribe`,
    keeping import time and memory usage low until actually needed.

    Parameters
    ----------
    model_path:
        Path to the faster-whisper model directory (or a HuggingFace
        model id like ``"small"``).
    device:
        ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation type — ``"int8"``, ``"float16"``, ``"float32"``, etc.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        strict: bool = True,
    ) -> tuple[str, str]:
        """Transcribe *audio* to text.

        Parameters
        ----------
        audio:
            1-D float32 array of PCM samples at 16 kHz.
        strict:
            When ``True`` (default), aggressive hallucination filtering is
            applied — suitable for idle / wake-word detection states.
            When ``False``, only the most obvious canned hallucinations
            are discarded — suitable for menu / conversation states where
            short or uncertain utterances still matter.

        Returns
        -------
        tuple[str, str]
            ``(text, detected_language_code)``.  ``text`` may be empty if
            the segment was classified as a hallucination or silence.
            ``detected_language_code`` is the ISO-639-1 code Whisper
            inferred (e.g. ``"es"``, ``"en"``).
        """
        model = self._ensure_model()

        # --- write a temporary WAV (faster-whisper expects a file path) ---
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            pcm = (audio.flatten() * 32767).astype(np.int16)
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_WHISPER_SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())

            t0 = time.time()
            segments, info = model.transcribe(
                tmp,
                beam_size=5,
                vad_filter=False,  # Silero handles VAD externally
                temperature=0.0,
                condition_on_previous_text=False,
                # Loosened thresholds so short user responses are not
                # classified as silence by the model's defaults.
                no_speech_threshold=0.95,
                compression_ratio_threshold=3.0,
                log_prob_threshold=-2.5,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            detected_lang = getattr(info, "language", "")
            elapsed = time.time() - t0

            # --- hallucination filtering ---
            text = self._filter(text, strict=strict, elapsed=elapsed)
            if text:
                log.info("Transcription (%.2fs, lang=%s): '%s'", elapsed, detected_lang, text)

            return text.strip(), detected_lang
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Hallucination filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Return ``True`` if *text* looks like a Whisper hallucination.

        Two heuristics:
        1. Contains a known canned phrase.
        2. A single word accounts for >= 60 % of all words (repetition).
        """
        if not text:
            return True

        text_lower = text.lower()
        for phrase in _CANNED_HALLUCINATIONS:
            if phrase in text_lower:
                return True

        words = [w.strip(".,!?¿¡").lower() for w in text.split() if w.strip()]
        if len(words) >= 4:
            counts = Counter(words)
            _top_word, top_count = counts.most_common(1)[0]
            if top_count / len(words) >= 0.6:
                return True

        return False

    def _filter(self, text: str, *, strict: bool, elapsed: float) -> str:
        """Apply hallucination filtering according to *strict* mode."""
        if strict and self._is_hallucination(text):
            log.debug("Discarded hallucination (%.2fs): '%s'", elapsed, text[:80])
            return ""

        if not strict and text:
            if any(p in text.lower() for p in _CANNED_HALLUCINATIONS_LIGHT):
                log.debug("Canned phrase filtered (%.2fs): '%s'", elapsed, text[:80])
                return ""

        return text

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self):
        """Load the Whisper model on first use."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel  # noqa: WPS433 (nested import)

        log.info(
            "Loading Whisper model from '%s' (device=%s, compute=%s) ...",
            self._model_path,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_path,
            device=self._device,
            compute_type=self._compute_type,
        )
        log.info("Whisper model loaded.")
        return self._model
