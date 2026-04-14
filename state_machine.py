"""Two-state state machine for Jarvis daemon."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class State(Enum):
    DORMIDO = "DORMIDO"
    ACTIVO = "ACTIVO"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class StateMachine:
    """Manages Jarvis state transitions between DORMIDO and ACTIVO."""

    def __init__(self, silence_timeout: float = 5.0) -> None:
        self._state = State.DORMIDO
        self._sub_state: State | None = None
        self._state_entered_at = time.time()
        self._last_audio_time = time.time()
        self._silence_timeout = silence_timeout
        self._conversation_turns = 0
        self._last_user_text: str | None = None
        self._last_response: str | None = None
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
    def is_dormido(self) -> bool:
        return self._state is State.DORMIDO

    @property
    def is_activo(self) -> bool:
        return self._state is State.ACTIVO

    @property
    def sub_state(self) -> State | None:
        return self._sub_state

    @property
    def conversation_turns(self) -> int:
        return self._conversation_turns

    @property
    def last_user_text(self) -> str | None:
        return self._last_user_text

    @property
    def last_response(self) -> str | None:
        return self._last_response

    @property
    def is_multi_turn(self) -> bool:
        return self._conversation_turns > 0

    def activate(self) -> bool:
        """Transition DORMIDO -> ACTIVO. Returns True if transition was valid."""
        if self._state is not State.DORMIDO:
            log.warning("[Jarvis] Cannot activate — already in %s", self._state.value)
            return False
        self._state = State.ACTIVO
        self._state_entered_at = time.time()
        self._last_audio_time = time.time()
        log.info("[Jarvis] State: DORMIDO -> ACTIVO at %.3f", self._state_entered_at)
        return True

    def deactivate(self) -> bool:
        """Transition ACTIVO -> DORMIDO. Returns True if transition was valid."""
        if self._state is not State.ACTIVO:
            log.warning("[Jarvis] Cannot deactivate — already in %s", self._state.value)
            return False
        self._state = State.DORMIDO
        self._state_entered_at = time.time()
        log.info("[Jarvis] State: ACTIVO -> DORMIDO at %.3f", self._state_entered_at)
        return True

    def record_audio_activity(self) -> None:
        """Call this when speech is detected to reset silence timer."""
        self._last_audio_time = time.time()

    def set_sub_state(self, sub_state: State) -> None:
        """Set a sub-state (LISTENING, PROCESSING, SPEAKING)."""
        self._sub_state = sub_state
        log.debug("[Jarvis] Sub-state set to %s", sub_state.value)

    def clear_sub_state(self) -> None:
        """Clear the sub-state."""
        self._sub_state = None

    def increment_turn(self) -> int:
        """Increment conversation turn counter and return new value."""
        self._conversation_turns += 1
        log.debug("[Jarvis] Conversation turn: %d", self._conversation_turns)
        return self._conversation_turns

    def set_context(self, user_text: str | None, response: str | None) -> None:
        """Store conversation context for multi-turn detection."""
        self._last_user_text = user_text
        self._last_response = response

    def clear_context(self) -> None:
        """Clear conversation context."""
        self._last_user_text = None
        self._last_response = None
        self._conversation_turns = 0

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
