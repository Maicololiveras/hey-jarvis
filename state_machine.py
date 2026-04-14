"""Two-state state machine for Jarvis daemon."""

from __future__ import annotations

import logging
import time
from uuid import uuid4
from enum import Enum

log = logging.getLogger(__name__)


class State(Enum):
    DORMIDO = "DORMIDO"
    ACTIVO = "ACTIVO"


class StateMachine:
    """Manages Jarvis state transitions between DORMIDO and ACTIVO."""

    def __init__(self, silence_timeout: float = 5.0) -> None:
        self._state = State.DORMIDO
        self._session_id: str | None = None
        self._state_entered_at = time.time()
        self._last_audio_time = time.time()
        self._silence_timeout = silence_timeout
        log.info(
            "[Jarvis] StateMachine initialized — state=%s, timeout=%.1fs",
            self._state.value,
            silence_timeout,
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def time_in_state(self) -> float:
        return time.time() - self._state_entered_at

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def is_dormido(self) -> bool:
        return self._state is State.DORMIDO

    @property
    def is_activo(self) -> bool:
        return self._state is State.ACTIVO

    def activate(self) -> bool:
        """Transition DORMIDO -> ACTIVO. Returns True if transition was valid."""
        if self._state is not State.DORMIDO:
            log.warning("[Jarvis] Cannot activate — already in %s", self._state.value)
            return False
        self._state = State.ACTIVO
        self._session_id = str(uuid4())
        self._state_entered_at = time.time()
        self._last_audio_time = time.time()
        log.info(
            "[Jarvis] State: DORMIDO -> ACTIVO at %.3f (session=%s)",
            self._state_entered_at,
            self._session_id,
        )
        return True

    def deactivate(self) -> bool:
        """Transition ACTIVO -> DORMIDO. Returns True if transition was valid."""
        if self._state is not State.ACTIVO:
            log.warning("[Jarvis] Cannot deactivate — already in %s", self._state.value)
            return False
        self._state = State.DORMIDO
        self._state_entered_at = time.time()
        self._last_audio_time = self._state_entered_at
        log.info(
            "[Jarvis] State: ACTIVO -> DORMIDO at %.3f (session=%s)",
            self._state_entered_at,
            self._session_id,
        )
        self._session_id = None
        return True

    def record_audio_activity(self) -> None:
        """Call this when speech is detected to reset silence timer."""
        self._last_audio_time = time.time()

    def check_silence_timeout(self, tts_playing: bool = False) -> bool:
        """Check if silence timeout has been exceeded.

        Returns True if 5s of silence has passed (not counting TTS time).
        Only meaningful in ACTIVO state.
        """
        if self._state is not State.ACTIVO:
            return False
        if tts_playing:
            # Don't count silence during TTS — reset the timer
            self._last_audio_time = time.time()
            return False
        elapsed = time.time() - self._last_audio_time
        if elapsed >= self._silence_timeout:
            log.info(
                "[Jarvis] Silence timeout reached (%.1fs >= %.1fs)",
                elapsed,
                self._silence_timeout,
            )
            return True
        return False
