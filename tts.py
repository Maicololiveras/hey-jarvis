"""TTS module for Hey Jarvis — edge-tts + ffplay with language detection.

Provides language-aware voice selection, offline fallback via pyttsx3,
and speaking-state tracking for mute-window coordination (TASK-013).
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
import time
from typing import Optional

from jarvis.config import get_tts_config

logger = logging.getLogger(__name__)

# Windows flag to hide console window when launching subprocesses.
_CREATE_NO_WINDOW = 0x08000000

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
# ffplay resolver (same logic as speak.py)
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


# ---------------------------------------------------------------------------
# Speaking-state tracking
# ---------------------------------------------------------------------------

_last_speech_end: float = 0.0
_speaking: bool = False


def is_speaking() -> bool:
    """Return ``True`` if a TTS utterance is currently playing."""
    return _speaking


def get_last_speech_end_time() -> float:
    """Return epoch timestamp when the last utterance finished (0.0 if none)."""
    return _last_speech_end


# ---------------------------------------------------------------------------
# Voice selection
# ---------------------------------------------------------------------------

def _pick_voice(language: str, cfg: dict) -> str:
    """Return the configured voice name for *language*."""
    if language == "es":
        return cfg.get("voice_es", "es-CO-GonzaloNeural")
    return cfg.get("voice_en", "en-US-GuyNeural")


# ---------------------------------------------------------------------------
# Core TTS — online (edge-tts) and offline (pyttsx3)
# ---------------------------------------------------------------------------

async def _speak_online(text: str, voice: str) -> None:
    """Synthesise *text* with edge-tts and play via ffplay."""
    import edge_tts  # lazy — optional dep

    output = tempfile.mktemp(suffix=".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output)

    ffplay = _resolve_ffplay()
    logger.debug("Playing TTS via %s -> %s", ffplay, output)

    proc = subprocess.Popen(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", output],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
    )
    proc.wait()  # block until playback finishes so _speaking stays accurate


def _speak_offline(text: str, language: str) -> None:
    """Fallback TTS using pyttsx3 (Windows SAPI5)."""
    import pyttsx3  # lazy — optional dep

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    target = "spanish" if language == "es" else "english"
    for v in voices:
        if target in v.name.lower():
            engine.setProperty("voice", v.id)
            break

    engine.setProperty("rate", 180)
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

    duration = len(text) / _CHARS_PER_SECOND

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
        asyncio.run(_speak_online(text, voice))
    except Exception:
        logger.warning("edge-tts failed, attempting offline fallback", exc_info=True)
        if offline_fallback:
            try:
                _speak_offline(text, lang)
            except Exception:
                logger.error("Offline TTS also failed", exc_info=True)
        else:
            logger.error("Offline fallback disabled — utterance dropped")
    finally:
        _speaking = False
        _last_speech_end = time.time()

    return duration
