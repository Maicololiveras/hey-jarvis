"""Audio device selection and preprocessing functions for the Jarvis pipeline.

Extracted from claudio_daemon.py. These are standalone functions that operate
on numpy arrays and use the audio config dict from jarvis.config for all
tunable parameters.

Provides:
- select_audio_device  — pick the best input device (API priority + override)
- preprocess_chunk     — highpass + pre-gain for VAD input
- preprocess_segment   — adaptive AGC normalization for Whisper input
- _highpass_filter     — scipy-free single-pole IIR highpass
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default audio constants (used when config keys are missing)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "sample_rate": 16000,
    "pre_gain": 2.0,
    "highpass_cutoff_hz": 80,
    "agc_target_peak": 0.9,
    "agc_max_gain": 10.0,
    "agc_min_peak": 0.01,
    "device_override": None,
}

# Host-API preference order for Windows audio device selection.
AUDIO_DEVICE_API_PRIORITY: tuple[str, ...] = (
    "MME",
    "DirectSound",
    "WDM-KS",
    "WASAPI",
)

# Generic Windows input device aliases that should be skipped in favour of
# the actual hardware mic when available.
_GENERIC_INPUT_ALIASES: set[str] = {
    "asignador de sonido microsoft - input",
    "controlador primario de captura de sonido",
    "microsoft sound mapper - input",
    "primary sound capture driver",
}

# Environment variable that overrides automatic device selection.
_ENV_DEVICE_OVERRIDE = "JARVIS_AUDIO_DEVICE"


# ---------------------------------------------------------------------------
# Internal highpass filter state (module-level, reset via reset_highpass)
# ---------------------------------------------------------------------------
_hp_prev_x: float = 0.0
_hp_prev_y: float = 0.0
_hp_alpha: float = 0.0  # computed lazily on first call


def reset_highpass() -> None:
    """Reset the highpass filter state.

    Call this when starting a new audio stream so leftover state from a
    previous session does not bleed into the new one.
    """
    global _hp_prev_x, _hp_prev_y, _hp_alpha
    _hp_prev_x = 0.0
    _hp_prev_y = 0.0
    _hp_alpha = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _highpass_filter(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate: int,
) -> np.ndarray:
    """Apply a single-pole high-pass IIR filter in-place.

    Uses the recurrence:
        y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        alpha = RC / (RC + dt),  where RC = 1 / (2 * pi * cutoff_hz)

    This is a minimal, scipy-free implementation suitable for real-time
    chunk-by-chunk processing.  Filter state is kept at module level so
    consecutive chunks produce a continuous output.

    Parameters
    ----------
    data:
        1-D float32 audio samples.
    cutoff_hz:
        High-pass cutoff frequency in Hz.
    sample_rate:
        Audio sample rate in Hz.

    Returns
    -------
    np.ndarray:
        Filtered float32 samples (same length as *data*).
    """
    global _hp_prev_x, _hp_prev_y, _hp_alpha

    # Compute alpha lazily (or recompute if cutoff changed).
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)

    if _hp_alpha != alpha:
        _hp_alpha = alpha
        logger.debug(
            "Highpass filter alpha=%.6f (cutoff=%dHz, sr=%dHz)",
            alpha,
            cutoff_hz,
            sample_rate,
        )

    out = np.empty_like(data, dtype=np.float32)
    prev_x = _hp_prev_x
    prev_y = _hp_prev_y
    for i in range(data.shape[0]):
        x = float(data[i])
        y = alpha * (prev_y + x - prev_x)
        out[i] = y
        prev_x = x
        prev_y = y
    _hp_prev_x = prev_x
    _hp_prev_y = prev_y
    return out


def preprocess_chunk(chunk: np.ndarray, audio_cfg: dict[str, Any]) -> np.ndarray:
    """Pipeline applied to every raw mic chunk before it reaches VAD.

    Removes DC offset / low-frequency rumble via a highpass filter, then
    applies a fixed pre-gain boost so Silero VAD sees a stronger signal
    when the user speaks softly.

    Adaptive AGC is intentionally NOT applied here — per-chunk
    normalization would amplify ambient silence into false voice events.

    Parameters
    ----------
    chunk:
        1-D float32 audio samples from the mic callback.
    audio_cfg:
        The ``audio`` section of the Jarvis config dict.  Expected keys:
        ``highpass_cutoff_hz``, ``sample_rate``, ``pre_gain``.

    Returns
    -------
    np.ndarray:
        Preprocessed float32 samples clipped to [-1, 1].
    """
    cutoff_hz: float = audio_cfg.get(
        "highpass_cutoff_hz", _DEFAULTS["highpass_cutoff_hz"]
    )
    sample_rate: int = audio_cfg.get("sample_rate", _DEFAULTS["sample_rate"])
    pre_gain: float = audio_cfg.get("pre_gain", _DEFAULTS["pre_gain"])

    filtered = _highpass_filter(chunk, cutoff_hz, sample_rate)
    boosted = np.clip(filtered * pre_gain, -1.0, 1.0).astype(np.float32)
    return boosted


def preprocess_segment(
    segment: np.ndarray,
    audio_cfg: dict[str, Any],
) -> np.ndarray:
    """Adaptive AGC normalization applied to a complete speech segment before Whisper.

    Normalizes the peak toward ``agc_target_peak`` so the whole utterance
    reaches the STT engine at a consistent level regardless of how softly
    or loudly the user spoke.  A floor (``agc_min_peak``) and max-gain cap
    protect against amplifying silence into noise.

    Parameters
    ----------
    segment:
        1-D float32 audio samples representing one complete utterance.
    audio_cfg:
        The ``audio`` section of the Jarvis config dict.  Expected keys:
        ``agc_target_peak``, ``agc_max_gain``, ``agc_min_peak``.

    Returns
    -------
    np.ndarray:
        Gain-adjusted float32 samples clipped to [-1, 1].
    """
    agc_target: float = audio_cfg.get("agc_target_peak", _DEFAULTS["agc_target_peak"])
    agc_max_gain: float = audio_cfg.get("agc_max_gain", _DEFAULTS["agc_max_gain"])
    agc_min_peak: float = audio_cfg.get("agc_min_peak", _DEFAULTS["agc_min_peak"])

    if segment.size == 0:
        return segment

    peak = float(np.max(np.abs(segment)))
    if peak < agc_min_peak:
        logger.debug(
            "Segment peak %.5f below floor %.5f — skipping AGC", peak, agc_min_peak
        )
        return segment

    gain = agc_target / peak
    if gain > agc_max_gain:
        gain = agc_max_gain

    logger.debug("AGC: peak=%.4f, gain=%.2f", peak, gain)
    boosted = segment * gain
    return np.clip(boosted, -1.0, 1.0).astype(np.float32)


def select_audio_device(
    audio_cfg: dict[str, Any],
) -> tuple[int | None, str]:
    """Pick the best input device, preferring modern host APIs.

    Returns ``(device_index, human_description)``.

    Resolution order:

    1. Environment variable ``JARVIS_AUDIO_DEVICE`` (int index or name substring).
    2. Config key ``audio.device_override`` (same semantics).
    3. Iterate host APIs in ``AUDIO_DEVICE_API_PRIORITY`` order and pick the
       first *real* (non-generic-alias) input device.
    4. Fall back to the system default input device.

    Parameters
    ----------
    audio_cfg:
        The ``audio`` section of the Jarvis config dict.  Expected key:
        ``device_override``.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    def _hostapi_name(idx: int) -> str:
        try:
            return hostapis[idx]["name"]
        except Exception:
            return ""

    # --- Resolve override: env var takes precedence over config ---
    override = os.environ.get(_ENV_DEVICE_OVERRIDE) or audio_cfg.get(
        "device_override", _DEFAULTS["device_override"]
    )

    if override is not None:
        override_str = str(override)
        try:
            idx = int(override_str)
            dev = devices[idx]
            desc = f"{dev['name']} [{_hostapi_name(dev['hostapi'])}] (override index)"
            logger.info("Audio device override (index): %s", desc)
            return idx, desc
        except (ValueError, IndexError):
            needle = override_str.lower()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
                    desc = f"{dev['name']} [{_hostapi_name(dev['hostapi'])}] (override match)"
                    logger.info("Audio device override (name): %s", desc)
                    return i, desc
            logger.warning(
                "Device override '%s' did not match any input device — falling back to auto",
                override_str,
            )

    # --- Auto-select: prefer input devices on modern host APIs ---
    for api_hint in AUDIO_DEVICE_API_PRIORITY:
        generic_match: tuple[int, str] | None = None
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue
            api_name = _hostapi_name(dev["hostapi"])
            if api_hint.lower() not in api_name.lower():
                continue

            description = f"{dev['name']} [{api_name}]"
            if dev["name"].strip().lower() in _GENERIC_INPUT_ALIASES:
                generic_match = (i, description)
                continue

            logger.info("Auto-selected audio device: %s", description)
            return i, description

        # If the only match for this API was a generic alias, use it as fallback.
        if generic_match is not None:
            logger.info(
                "Auto-selected audio device (generic alias): %s", generic_match[1]
            )
            return generic_match

    # --- Fallback: system default ---
    default_idx = sd.default.device[0]
    if default_idx is not None and default_idx >= 0:
        dev = devices[default_idx]
        desc = f"{dev['name']} [{_hostapi_name(dev['hostapi'])}] (default)"
        logger.info("Falling back to system default audio device: %s", desc)
        return default_idx, desc

    logger.warning("No input device found — returning None")
    return None, "unknown (system default)"


# ---------------------------------------------------------------------------
# AudioPipeline — streaming wake word + VAD class
# ---------------------------------------------------------------------------

import queue
import time
from typing import Generator

import torch
from silero_vad import load_silero_vad, VADIterator

try:
    from openwakeword.model import Model as OpenWakeWordModel

    _OPENWAKEWORD_AVAILABLE = True
except ImportError:
    _OPENWAKEWORD_AVAILABLE = False

from jarvis.models import AudioEvent, SegmentEvent, TickEvent, WakeEvent

# Silero VAD requires exactly 512 samples per window at 16 kHz.
_SILERO_WINDOW_SAMPLES = 512
# openWakeWord requires exactly 1280 samples (80 ms at 16 kHz).
_OW_FRAME_SAMPLES = 1280

# Defaults for VAD params when not supplied via config.
_VAD_DEFAULTS: dict[str, Any] = {
    "vad_threshold": 0.45,
    "vad_min_silence_ms": 600,
    "vad_speech_pad_ms": 150,
    "vad_min_segment_seconds": 0.3,
}


class AudioPipeline:
    """Streams audio events: WakeEvent when wake word detected, SegmentEvent when speech captured.

    Combines a sounddevice InputStream, openWakeWord inference (for wake
    word detection) and Silero VAD (for speech segmentation) into a single
    generator-based pipeline.  All parameters are config-driven.
    """

    def __init__(self, audio_cfg: dict[str, Any], wake_cfg: dict[str, Any]) -> None:
        """Initialize audio pipeline with config.

        Parameters
        ----------
        audio_cfg:
            The ``audio`` section of the Jarvis config.  Expected keys:
            ``sample_rate``, ``channels``, ``chunk_duration_ms``,
            ``device_override``, ``pre_gain``, ``highpass_cutoff_hz``,
            ``agc_target_peak``, ``agc_max_gain``, ``agc_min_peak``.
            Optional VAD keys: ``vad_threshold``, ``vad_min_silence_ms``,
            ``vad_speech_pad_ms``, ``vad_min_segment_seconds``.
        wake_cfg:
            The ``wake_word`` section of the Jarvis config.  Expected keys:
            ``engine``, ``model``, ``threshold``, ``consecutive_frames``,
            ``extra_gain``.
        """
        self._audio_cfg = audio_cfg
        self._wake_cfg = wake_cfg
        self._sample_rate: int = audio_cfg.get("sample_rate", _DEFAULTS["sample_rate"])
        self._closed = False
        self._last_audio_frame_at: float = 0.0
        self._recent_rms_peak: float = 0.0

        # Mute window — suppresses processing after TTS playback.
        self._mute_until: float = 0.0

        # Internal audio queue — the sounddevice callback pushes frames here.
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)

        # Last preprocessed chunk — exposed so the daemon can read audio
        # levels for UI waveform feedback without modifying the generator.
        self.last_chunk: np.ndarray | None = None

        # ── Load Silero VAD ──
        logger.info("Loading Silero VAD model…")
        self._vad_model = load_silero_vad()
        logger.info("Silero VAD loaded.")

        # ── Load openWakeWord ──
        self._ow_model: OpenWakeWordModel | None = None
        self._ow_model_key: str | None = None
        if (
            wake_cfg.get("engine", "openwakeword") == "openwakeword"
            and _OPENWAKEWORD_AVAILABLE
        ):
            model_name = wake_cfg.get("model", "hey_jarvis")
            try:
                logger.info("Loading openWakeWord model '%s'…", model_name)
                self._ow_model = OpenWakeWordModel(
                    wakeword_models=[model_name],
                    inference_framework="onnx",
                )
                loaded_models = getattr(self._ow_model, "models", {})
                if isinstance(loaded_models, dict) and loaded_models:
                    self._ow_model_key = str(next(iter(loaded_models.keys())))
                else:
                    self._ow_model_key = str(model_name)
                logger.info(
                    "openWakeWord loaded. Configured model='%s', score key='%s'",
                    model_name,
                    self._ow_model_key,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load openWakeWord: %s — wake word disabled", exc
                )
        elif not _OPENWAKEWORD_AVAILABLE:
            logger.warning("openwakeword package not installed — wake word disabled")

    # ------------------------------------------------------------------
    # sounddevice callback
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frame_count: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """sd.InputStream callback — pushes raw audio frames into the queue."""
        if status:
            logger.debug("Audio callback status: %s", status)
        try:
            self._queue.put_nowait(indata.copy())
            self._last_audio_frame_at = time.time()
            samples = indata.flatten().astype(np.float32)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples * samples)))
                if rms > self._recent_rms_peak:
                    self._recent_rms_peak = rms
        except queue.Full:
            pass  # Drop frames if processing is falling behind.

    # ------------------------------------------------------------------
    # Main generator
    # ------------------------------------------------------------------

    def stream_events(self) -> Generator[AudioEvent, None, None]:
        """Yield audio events from the microphone.

        Yields :class:`WakeEvent` when the wake word is detected (consecutive
        frames above threshold) and :class:`SegmentEvent` when VAD captures a
        complete speech segment.

        The **caller** is responsible for state management — this method just
        yields events.
        """
        # ── Config ──
        ow_threshold: float = self._wake_cfg.get("threshold", 0.45)
        ow_consecutive_required: int = self._wake_cfg.get("consecutive_frames", 4)
        ow_extra_gain: float = self._wake_cfg.get("extra_gain", 2.0)
        ow_wake_word: str = self._ow_model_key or str(
            self._wake_cfg.get("model", "hey_jarvis")
        )
        vad_threshold: float = self._audio_cfg.get(
            "vad_threshold", _VAD_DEFAULTS["vad_threshold"]
        )
        vad_min_silence_ms: int = self._audio_cfg.get(
            "vad_min_silence_ms", _VAD_DEFAULTS["vad_min_silence_ms"]
        )
        vad_speech_pad_ms: int = self._audio_cfg.get(
            "vad_speech_pad_ms", _VAD_DEFAULTS["vad_speech_pad_ms"]
        )
        vad_min_segment_sec: float = self._audio_cfg.get(
            "vad_min_segment_seconds", _VAD_DEFAULTS["vad_min_segment_seconds"]
        )
        min_segment_samples = int(self._sample_rate * vad_min_segment_sec)

        # ── VAD iterator ──
        vad = VADIterator(
            self._vad_model,
            threshold=vad_threshold,
            sampling_rate=self._sample_rate,
            min_silence_duration_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        )

        # ── State ──
        speech_buffer: list[np.ndarray] = []
        in_speech = False
        leftover = np.empty(0, dtype=np.float32)
        ow_buffer = np.empty(0, dtype=np.float32)
        ow_consecutive_hits = 0
        ow_cooldown_until = 0.0  # avoid double-firing wake word
        ow_debug_last_log = 0.0
        no_audio_last_log = 0.0
        ow_debug_max_score = 0.0

        # Reset highpass filter state for a clean start.
        reset_highpass()
        self._last_audio_frame_at = 0.0
        self._recent_rms_peak = 0.0

        # Drain stale audio from previous runs.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # ── Select device ──
        device_index, device_name = select_audio_device(self._audio_cfg)
        logger.info("Audio input: %s (index=%s)", device_name, device_index)

        # ── Open stream ──
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._audio_cfg.get("channels", 1),
            dtype="float32",
            blocksize=_SILERO_WINDOW_SAMPLES,
            callback=self._audio_callback,
            device=device_index,
        ):
            while not self._closed:
                # ── Read next chunk ──
                try:
                    chunk = self._queue.get(timeout=0.5)
                except queue.Empty:
                    now = time.time()
                    if (
                        self._last_audio_frame_at > 0
                        and (now - self._last_audio_frame_at) >= 5.0
                    ):
                        if (now - no_audio_last_log) >= 5.0:
                            logger.warning(
                                "No audio frames for %.1fs (device=%s, index=%s)",
                                now - self._last_audio_frame_at,
                                device_name,
                                device_index,
                            )
                            no_audio_last_log = now
                    # Emit a heartbeat so the daemon can keep checking
                    # silence timeout and refreshing UI state even when no
                    # wake/VAD event is produced.
                    yield TickEvent(timestamp=now)
                    continue

                # ── Mute window — skip processing after TTS ──
                if time.time() < self._mute_until:
                    continue

                samples = chunk.flatten().astype(np.float32)
                rms = float(np.sqrt(np.mean(samples**2)))
                if rms > 0.01:
                    logger.debug("Audio chunk: rms=%.4f, len=%d", rms, len(samples))

                # Chunk-level preprocessing: highpass + pre-gain.
                samples = preprocess_chunk(samples, self._audio_cfg)

                # Expose the latest preprocessed chunk for UI audio levels.
                self.last_chunk = samples

                if leftover.size:
                    samples = np.concatenate([leftover, samples])
                    leftover = np.empty(0, dtype=np.float32)

                # ── openWakeWord inference ──
                if self._ow_model is not None:
                    ow_buffer = np.concatenate([ow_buffer, samples])
                    while len(ow_buffer) >= _OW_FRAME_SAMPLES:
                        frame = ow_buffer[:_OW_FRAME_SAMPLES]
                        ow_buffer = ow_buffer[_OW_FRAME_SAMPLES:]

                        if time.time() <= ow_cooldown_until:
                            continue  # still in post-wake dedup window

                        try:
                            # Extra gain dedicated to openWakeWord (on top
                            # of the global preprocess_chunk pre-gain).
                            frame_boosted = np.clip(frame * ow_extra_gain, -1.0, 1.0)
                            frame_int16 = (frame_boosted * 32767).astype(np.int16)
                            predictions = self._ow_model.predict(frame_int16)
                            score = float(predictions.get(ow_wake_word, 0.0))

                            if score > ow_debug_max_score:
                                ow_debug_max_score = score
                            now_dbg = time.time()
                            if now_dbg - ow_debug_last_log > 3.0:
                                logger.debug(
                                    "OW rolling stats: score_peak=%.4f rms_peak_raw=%.4f key=%s",
                                    ow_debug_max_score,
                                    self._recent_rms_peak,
                                    ow_wake_word,
                                )
                                ow_debug_max_score = 0.0
                                self._recent_rms_peak = 0.0
                                ow_debug_last_log = now_dbg

                            if score > 0.05:  # Log any non-trivial score
                                logger.debug(
                                    "OW score: %.4f (threshold=%.2f, consecutive=%d/%d)",
                                    score,
                                    ow_threshold,
                                    ow_consecutive_hits,
                                    ow_consecutive_required,
                                )

                            if score > ow_threshold:
                                ow_consecutive_hits += 1
                                if ow_consecutive_hits < ow_consecutive_required:
                                    # Not enough consecutive frames yet.
                                    continue
                                logger.info(
                                    "openWakeWord '%s' score=%.2f (%d frames) — WAKE!",
                                    ow_wake_word,
                                    score,
                                    ow_consecutive_hits,
                                )
                                ow_consecutive_hits = 0
                                ow_cooldown_until = time.time() + 3.0
                                # Reset model internal state so the next
                                # detection starts fresh.
                                try:
                                    self._ow_model.reset()
                                except Exception:
                                    pass
                                yield WakeEvent(score=score)
                            else:
                                ow_consecutive_hits = 0
                        except Exception as exc:
                            logger.error("openWakeWord error: %s", exc)
                            break

                # ── Silero VAD inference ──
                num_windows = len(samples) // _SILERO_WINDOW_SAMPLES
                for i in range(num_windows):
                    start = i * _SILERO_WINDOW_SAMPLES
                    window = samples[start : start + _SILERO_WINDOW_SAMPLES]

                    event = vad(torch.from_numpy(window))

                    if in_speech:
                        speech_buffer.append(window)

                    if event:
                        if "start" in event:
                            in_speech = True
                            speech_buffer = [window]
                        elif "end" in event and speech_buffer:
                            in_speech = False
                            segment_audio = np.concatenate(speech_buffer)
                            # Drop segments shorter than the minimum
                            # duration — they are noise pops, not speech.
                            if segment_audio.size < min_segment_samples:
                                logger.debug(
                                    "Dropped short segment: %d samples (min %d)",
                                    segment_audio.size,
                                    min_segment_samples,
                                )
                                speech_buffer = []
                                continue
                            # Segment-level AGC normalization for STT.
                            segment_audio = preprocess_segment(
                                segment_audio, self._audio_cfg
                            )
                            duration = segment_audio.size / self._sample_rate
                            yield SegmentEvent(
                                audio=segment_audio,
                                duration_seconds=duration,
                            )
                            speech_buffer = []

                tail = num_windows * _SILERO_WINDOW_SAMPLES
                if tail < len(samples):
                    leftover = samples[tail:].copy()

    # ------------------------------------------------------------------
    # Public control methods
    # ------------------------------------------------------------------

    def set_mute_window(self, duration_seconds: float) -> None:
        """Mute audio processing for *duration_seconds* (to prevent TTS echo).

        While muted, chunks from the microphone are read and discarded so the
        sounddevice buffer does not overflow, but no events are generated.
        """
        self._mute_until = time.time() + duration_seconds
        logger.debug("Mute window set: %.1fs", duration_seconds)

    def close(self) -> None:
        """Signal the stream loop to stop and release resources."""
        self._closed = True
        logger.info("AudioPipeline closed.")
