"""TTS module for Hey Jarvis — edge-tts with internal playback.

Synthesizes speech with edge-tts, decodes the returned MP3 in memory, and
plays it internally via sounddevice so Jarvis knows exactly when playback
starts and ends.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import av
except ImportError:  # pragma: no cover - environment-dependent optional import
    av = None

from jarvis.config import get_tts_config

logger = logging.getLogger(__name__)

# Rough chars-per-second estimate for speech duration.
_CHARS_PER_SECOND = 15.0

# Spanish-indicator patterns — diacritics and high-frequency words.
_SPANISH_DIACRITICS = re.compile(r"[áéíóúñ¿¡ü]", re.IGNORECASE)
_SPANISH_WORDS = re.compile(
    r"\b(?:que|del|por|una|con|los|las|para|como|más|pero|tiene|esto|esta|"
    r"puede|todo|hace|hola|sí|también|porque|cuando|donde|desde|entre|"
    r"sobre|después|antes|muy|bien|aquí|ahora|algo|otro|cada|"
    r"el|en|es|no|ya|le|se|lo|la|de|al|un)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Detect language from *text* using simple heuristics."""
    if _SPANISH_DIACRITICS.search(text):
        return "es"

    words = text.split()
    if not words:
        return "en"

    matches = len(_SPANISH_WORDS.findall(text))
    ratio = matches / len(words)
    if ratio >= 0.20:
        return "es"

    return "en"


_last_speech_end: float = 0.0
_speaking: bool = False
_playback_completed_event: threading.Event = threading.Event()


def is_speaking() -> bool:
    """Return ``True`` if a TTS utterance is currently playing."""
    return _speaking


def get_last_speech_end_time() -> float:
    """Return epoch timestamp when the last utterance finished (0.0 if none)."""
    return _last_speech_end


def wait_for_playback_complete(timeout: float | None = None) -> bool:
    """Block until TTS playback finishes."""
    return _playback_completed_event.wait(timeout=timeout)


def _pick_voice(language: str, cfg: dict) -> str:
    """Return the configured voice name for *language*."""
    if language == "es":
        return cfg.get("voice_es", "es-CO-GonzaloNeural")
    return cfg.get("voice_en", "en-US-GuyNeural")


async def _synthesize_online(text: str, voice: str, rate: str) -> bytes:
    """Synthesise *text* with edge-tts and return encoded audio bytes."""
    import edge_tts  # lazy — optional dep

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    audio_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_chunks.append(chunk["data"])

    audio_bytes = b"".join(audio_chunks)
    if not audio_bytes:
        raise RuntimeError("edge-tts returned no audio chunks")
    return audio_bytes


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded edge-tts audio bytes into float32 PCM samples."""
    if av is None:
        raise RuntimeError(
            "PyAV is required for internal edge-tts playback but is not installed"
        )

    frames: list[np.ndarray] = []
    sample_rate = 24000
    with av.open(io.BytesIO(audio_bytes), format="mp3") as container:
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate or sample_rate)
        for frame in container.decode(audio=0):
            pcm = frame.to_ndarray()
            if pcm.ndim == 1:
                pcm = pcm[np.newaxis, :]
            frames.append(np.asarray(pcm, dtype=np.float32))

    if not frames:
        raise RuntimeError("Decoded edge-tts audio was empty")

    audio = np.concatenate(frames, axis=1).T
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    return audio, sample_rate


def _play_audio(audio: np.ndarray, sample_rate: int) -> float:
    """Play decoded PCM samples internally and return actual duration."""
    pcm = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    frame_count = len(pcm) if pcm.ndim == 1 else pcm.shape[0]
    duration = float(frame_count / sample_rate)
    sd.play(pcm, samplerate=sample_rate, blocking=True)
    sd.stop()
    return duration


def speak(text: str, language: Optional[str] = None) -> float:
    """Speak *text* and return playback duration in seconds."""
    global _speaking, _last_speech_end

    if not text or not text.strip():
        logger.warning("speak() called with empty text — skipping")
        _playback_completed_event.set()
        return 0.0

    cfg = get_tts_config()
    lang = language or detect_language(text)
    voice = _pick_voice(lang, cfg)
    rate = cfg.get("rate", "+0%")
    duration = len(text) / _CHARS_PER_SECOND

    logger.info(
        "TTS speak [lang=%s voice=%s est=%.1fs]: %.60s%s",
        lang,
        voice,
        duration,
        text,
        "..." if len(text) > 60 else "",
    )

    _playback_completed_event.clear()
    _speaking = True
    try:
        audio_bytes = asyncio.run(_synthesize_online(text, voice, rate))
        audio, sample_rate = _decode_audio_bytes(audio_bytes)
        duration = _play_audio(audio, sample_rate)
    except Exception:
        logger.error(
            "edge-tts internal playback failed [lang=%s voice=%s rate=%s]",
            lang,
            voice,
            rate,
            exc_info=True,
        )
    finally:
        _speaking = False
        _last_speech_end = time.time()
        _playback_completed_event.set()

    return duration
