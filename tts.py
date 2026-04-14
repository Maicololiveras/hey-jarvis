"""TTS module for Hey Jarvis — edge-tts + sounddevice with language detection.

Provides language-aware voice selection, offline fallback via pyttsx3,
and speaking-state tracking for mute-window coordination (TASK-013).

Primary playback: PyAV decodes MP3 → sounddevice plays raw PCM.
Fallback playback: ffplay subprocess (if sounddevice/PyAV fail).
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional

import numpy as np

from jarvis.config import get_tts_config

logger = logging.getLogger(__name__)

# Windows flag to hide console window when launching subprocesses.
_CREATE_NO_WINDOW = 0x08000000

# Track active ffplay process for cleanup on shutdown (fallback path only).
_active_ffplay: subprocess.Popen | None = None

# Threading event to signal playback completion to other modules.
_playback_completed = threading.Event()
_playback_completed.set()  # Not playing initially.


def kill_active_playback() -> None:
    """Stop any running playback. Call on shutdown or restart.

    Stops sounddevice playback first, then kills any ffplay fallback process.
    """
    global _active_ffplay

    # Stop sounddevice playback (primary path).
    try:
        import sounddevice as sd

        sd.stop()
        logger.info("Stopped sounddevice playback")
    except Exception:
        pass

    _playback_completed.set()

    # Kill ffplay if it's running (fallback path).
    if _active_ffplay and _active_ffplay.poll() is None:
        try:
            _active_ffplay.kill()
            logger.info("Killed active ffplay (PID %d)", _active_ffplay.pid)
        except Exception:
            pass
    _active_ffplay = None

    # Also kill any orphan ffplay processes.
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "ffplay.exe"],
            capture_output=True,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception:
        pass


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


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str:
    """Detect language from *text* using simple heuristics.

    Returns ``"es"`` if Spanish diacritics or common Spanish words are found,
    otherwise ``"en"``.
    """
    if _SPANISH_DIACRITICS.search(text):
        return "es"

    words = text.split()
    if not words:
        return "en"

    matches = len(_SPANISH_WORDS.findall(text))
    ratio = matches / len(words)

    # If >=20 % of words are common Spanish tokens, assume Spanish.
    if ratio >= 0.20:
        return "es"

    return "en"


# ---------------------------------------------------------------------------
# ffplay resolver — kept as FALLBACK if sounddevice/PyAV fail
# ---------------------------------------------------------------------------


def _resolve_ffplay() -> str:
    """Return path to *ffplay* binary, searching PATH and WinGet installs."""
    path = shutil.which("ffplay")
    if path:
        return path

    candidates = glob.glob(
        os.path.join(
            os.path.expanduser("~"),
            "AppData/Local/Microsoft/WinGet/Packages/*ffmpeg*/*/bin/ffplay.exe",
        )
    )
    if candidates:
        return candidates[0]

    return "ffplay"  # fallback — let the OS resolve it or fail loudly


def _resolve_playback_binary(playback: Optional[str]) -> str:
    """Return the configured playback binary, preserving ffplay lookup."""
    player = (playback or "ffplay").strip()
    if not player or player.lower() == "ffplay":
        return _resolve_ffplay()
    return player


def _normalize_edge_rate(rate: object) -> Optional[str]:
    """Return an edge-tts compatible rate string like '+5%'."""
    if not isinstance(rate, str):
        return None

    value = rate.strip()
    if not value:
        return None

    if re.fullmatch(r"[+-]?\d+%", value):
        return value if value[0] in "+-" else f"+{value}"

    logger.warning("Invalid TTS rate %r; expected formats like '+5%%' or '-10%%'", rate)
    return None


def _rate_percent_to_multiplier(rate: object) -> Optional[float]:
    """Convert config rate like '+5%' into a numeric multiplier."""
    normalized = _normalize_edge_rate(rate)
    if normalized is None:
        return None

    percent = int(normalized[:-1])
    return max(0.1, 1.0 + (percent / 100.0))


# ---------------------------------------------------------------------------
# PyAV decoder: MP3 → numpy PCM array
# ---------------------------------------------------------------------------


def _decode_mp3_to_pcm(filepath: str) -> tuple[np.ndarray, int]:
    """Decode an MP3 file to a float32 numpy array using PyAV.

    Returns
    -------
    tuple[np.ndarray, int]
        (audio_data, sample_rate) where audio_data is float32 in [-1, 1].
    """
    import av

    container = av.open(filepath)

    # Use AudioResampler to convert to s16 mono for sounddevice
    resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)

    frames: list[np.ndarray] = []
    sample_rate = 24000

    for frame in container.decode(audio=0):
        resampled_frames = resampler.resample(frame)
        for resampled in resampled_frames:
            arr = resampled.to_ndarray()
            frames.append(arr)

    container.close()

    if not frames:
        raise RuntimeError(f"No audio frames decoded from {filepath}")

    # Concatenate: to_ndarray() gives (channels, samples) for s16
    raw = np.concatenate(frames, axis=1 if frames[0].ndim == 2 else 0)

    # Transpose to (samples, channels) if needed
    if raw.ndim == 2:
        raw = raw.T

    # Convert int16 to float32 in [-1.0, 1.0] for sounddevice
    audio_data = raw.astype(np.float32) / 32768.0

    return audio_data, sample_rate


# ---------------------------------------------------------------------------
# Speaking-state tracking
# ---------------------------------------------------------------------------

_last_speech_end: float = 0.0
_speaking: bool = False


def estimate_duration(text: str) -> float:
    """Return the rough duration estimate used for TTS coordination."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    return len(cleaned) / _CHARS_PER_SECOND


def is_speaking() -> bool:
    """Return ``True`` if a TTS utterance is currently playing.

    Checks both the module flag and sounddevice stream state for accuracy.
    """
    if not _speaking:
        return False

    # Double-check with sounddevice — the stream may have finished.
    try:
        import sounddevice as sd

        stream = sd.get_stream()
        if stream is not None and stream.active:
            return True
    except Exception:
        pass

    # If _speaking is True but no sd stream is active, trust the flag
    # (could be ffplay fallback or pyttsx3).
    return _speaking


def get_last_speech_end_time() -> float:
    """Return epoch timestamp when the last utterance finished (0.0 if none)."""
    return _last_speech_end


def wait_for_playback(timeout: Optional[float] = None) -> bool:
    """Block until the current playback finishes.

    Parameters
    ----------
    timeout:
        Maximum seconds to wait.  ``None`` means wait forever.

    Returns
    -------
    bool
        ``True`` if playback completed, ``False`` if timed out.
    """
    return _playback_completed.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# Voice selection
# ---------------------------------------------------------------------------


def _pick_voice(language: str, cfg: dict) -> str:
    """Return the configured voice name for *language*."""
    if language == "es":
        return cfg.get("voice_es", "es-CO-GonzaloNeural")
    return cfg.get("voice_en", "en-US-GuyNeural")


# ---------------------------------------------------------------------------
# Core TTS — online (edge-tts) with sounddevice primary, ffplay fallback
# ---------------------------------------------------------------------------


async def _speak_online(text: str, voice: str, cfg: dict) -> None:
    """Synthesise *text* with edge-tts and play via sounddevice (PyAV decode).

    Falls back to ffplay subprocess if sounddevice/PyAV playback fails.
    """
    import edge_tts  # lazy — optional dep

    output = tempfile.mktemp(suffix=".mp3")
    try:
        rate = _normalize_edge_rate(cfg.get("rate"))
        communicate_kwargs = {"rate": rate} if rate else {}
        communicate = edge_tts.Communicate(text, voice, **communicate_kwargs)
        await communicate.save(output)

        # --- Primary path: PyAV decode + sounddevice playback ---
        try:
            _play_with_sounddevice(output)
            return
        except Exception:
            logger.warning(
                "sounddevice playback failed, falling back to ffplay",
                exc_info=True,
            )

        # --- Fallback path: ffplay subprocess ---
        _play_with_ffplay(output, cfg)
    finally:
        # Clean up temp file.
        try:
            os.unlink(output)
        except OSError:
            pass


def _play_with_sounddevice(filepath: str) -> None:
    """Decode MP3 with PyAV and play via sounddevice."""
    import sounddevice as sd

    _playback_completed.clear()
    try:
        audio_data, sample_rate = _decode_mp3_to_pcm(filepath)
        logger.debug(
            "Decoded MP3: %d samples, %d Hz, %s shape",
            audio_data.shape[0],
            sample_rate,
            audio_data.shape,
        )

        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()  # Block until playback finishes (interruptible via sd.stop()).
    finally:
        _playback_completed.set()


def _play_with_ffplay(filepath: str, cfg: dict) -> None:
    """Play audio file via ffplay subprocess (fallback)."""
    global _active_ffplay

    playback = _resolve_playback_binary(cfg.get("playback"))
    logger.debug("Playing TTS via ffplay fallback: %s -> %s", playback, filepath)

    _playback_completed.clear()
    try:
        proc = subprocess.Popen(
            [playback, "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        _active_ffplay = proc
        proc.wait()  # Block until playback finishes.
        _active_ffplay = None
    finally:
        _playback_completed.set()


def _speak_offline(text: str, language: str, cfg: dict) -> None:
    """Fallback TTS using pyttsx3 (Windows SAPI5)."""
    import pyttsx3  # lazy — optional dep

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    target = "spanish" if language == "es" else "english"
    for v in voices:
        if target in v.name.lower():
            engine.setProperty("voice", v.id)
            break

    multiplier = _rate_percent_to_multiplier(cfg.get("rate"))
    if multiplier is not None:
        current_rate = engine.getProperty("rate")
        if isinstance(current_rate, (int, float)):
            engine.setProperty("rate", max(50, int(current_rate * multiplier)))

    engine.say(text)
    engine.runAndWait()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def speak(text: str, language: Optional[str] = None) -> float:
    """Speak *text* and return estimated duration in seconds.

    Parameters
    ----------
    text:
        The utterance to synthesise.
    language:
        ``"es"`` or ``"en"``.  When ``None``, auto-detected from *text*.

    Returns
    -------
    float
        Estimated speech duration (``len(text) / 15.0``).
    """
    global _speaking, _last_speech_end

    if not text or not text.strip():
        logger.warning("speak() called with empty text — skipping")
        return 0.0

    cfg = get_tts_config()
    lang = language or detect_language(text)
    voice = _pick_voice(lang, cfg)
    offline_fallback = cfg.get("offline_fallback", True)

    duration = estimate_duration(text)

    logger.info(
        "TTS speak [lang=%s voice=%s est=%.1fs]: %.60s%s",
        lang,
        voice,
        duration,
        text,
        "..." if len(text) > 60 else "",
    )

    _speaking = True
    try:
        asyncio.run(_speak_online(text, voice, cfg))
    except Exception:
        logger.warning("edge-tts failed, attempting offline fallback", exc_info=True)
        if offline_fallback:
            try:
                _speak_offline(text, lang, cfg)
            except Exception:
                logger.error("Offline TTS also failed", exc_info=True)
        else:
            logger.error("Offline fallback disabled — utterance dropped")
    finally:
        _speaking = False
        _last_speech_end = time.time()

    return duration
