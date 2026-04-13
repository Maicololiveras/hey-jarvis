"""Shared data models — contracts between Jarvis modules."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Union

import numpy as np


@dataclass(frozen=True, slots=True)
class WakeEvent:
    """Emitted when the wake word is detected."""
    score: float
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    """Emitted when a speech segment is captured after VAD."""
    audio: np.ndarray
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)


# Union type for the audio pipeline generator
AudioEvent = Union[WakeEvent, SegmentEvent]


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Result from a backend query (claude-p, local-qwen, etc.)."""
    ok: bool
    text: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    error: str = ""


@dataclass(frozen=True, slots=True)
class UICommand:
    """Command sent from daemon thread to UI thread via queue."""
    action: str  # "show", "hide", "update_waveform", "set_state"
    data: object = None  # np.ndarray for waveform, str for state name
