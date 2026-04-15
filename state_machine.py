"""Canonical Jarvis state machine for daemon and MCP flows."""

from __future__ import annotations

import logging
from threading import RLock
import time
from collections.abc import Callable
from enum import Enum
from uuid import uuid4

log = logging.getLogger(__name__)


class State(Enum):
    DORMIDO = "DORMIDO"
    ACTIVO = "ACTIVO"
    PROCESANDO = "PROCESANDO"
    HABLANDO = "HABLANDO"
    ERROR = "ERROR"
    ESCUCHANDO = "ESCUCHANDO"


class StateMachine:
    """Manages canonical Jarvis state transitions and session ownership."""

    def __init__(
        self,
        silence_timeout: float = 5.0,
        error_timeout: float = 5.0,
        on_state_change: Callable[[State, State], None] | None = None,
    ) -> None:
        self._lock = RLock()
        self._state = State.DORMIDO
        self._session_id: str | None = None
        self._state_entered_at = time.time()
        self._last_audio_time = time.time()
        self._silence_timeout = silence_timeout
        self._error_timeout = error_timeout
        self._error_recover_at: float | None = None
        self._on_state_change = on_state_change
        log.info(
            "[Jarvis] StateMachine initialized — state=%s, silence=%.1fs, error=%.1fs",
            self._state.value,
            silence_timeout,
            error_timeout,
        )

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def time_in_state(self) -> float:
        with self._lock:
            return time.time() - self._state_entered_at

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def is_dormido(self) -> bool:
        return self.state is State.DORMIDO

    @property
    def is_activo(self) -> bool:
        return self.state is State.ACTIVO

    @property
    def is_error(self) -> bool:
        return self.state is State.ERROR

    @property
    def has_active_session(self) -> bool:
        return self.state is not State.DORMIDO

    def _emit_state_change(self, old: State, new: State) -> None:
        callback = self._on_state_change
        if callback is None:
            return
        try:
            callback(old, new)
        except Exception:
            log.exception(
                "[Jarvis] on_state_change callback failed (%s -> %s)",
                old.value,
                new.value,
            )

    def _set_state(self, new_state: State) -> bool:
        callback_args: tuple[State, State] | None = None
        with self._lock:
            old_state = self._state
            if old_state is new_state:
                return False
            self._state = new_state
            self._state_entered_at = time.time()
            self._error_recover_at = (
                self._state_entered_at + self._error_timeout
                if new_state is State.ERROR
                else None
            )
            log.info(
                "[Jarvis] State: %s -> %s at %.3f (session=%s)",
                old_state.value,
                new_state.value,
                self._state_entered_at,
                self._session_id,
            )
            callback_args = (old_state, new_state)

        self._emit_state_change(*callback_args)
        return True

    def activate(self) -> bool:
        """Transition DORMIDO -> ACTIVO and start a new session."""
        with self._lock:
            if self._state is not State.DORMIDO:
                log.warning(
                    "[Jarvis] Cannot activate — already in %s", self._state.value
                )
                return False
            self._session_id = str(uuid4())
            self._last_audio_time = time.time()
        return self._set_state(State.ACTIVO)

    def deactivate(self) -> bool:
        """Transition any active state -> DORMIDO and close the session."""
        with self._lock:
            if self._state is State.DORMIDO:
                log.warning(
                    "[Jarvis] Cannot deactivate — already in %s", self._state.value
                )
                return False
            self._last_audio_time = time.time()
        changed = self._set_state(State.DORMIDO)
        with self._lock:
            self._session_id = None
        return changed

    def start_processing(self) -> bool:
        """Transition ACTIVO/ESCUCHANDO -> PROCESANDO."""
        current = self.state
        if current not in {State.ACTIVO, State.ESCUCHANDO}:
            log.warning("[Jarvis] Cannot start processing from %s", current.value)
            return False
        return self._set_state(State.PROCESANDO)

    def start_speaking(self) -> bool:
        """Transition PROCESANDO/ACTIVO -> HABLANDO."""
        current = self.state
        if current not in {State.PROCESANDO, State.ACTIVO}:
            log.warning("[Jarvis] Cannot start speaking from %s", current.value)
            return False
        return self._set_state(State.HABLANDO)

    def finish_speaking(self) -> bool:
        """Transition HABLANDO -> ACTIVO."""
        if self.state is not State.HABLANDO:
            log.warning("[Jarvis] Cannot finish speaking from %s", self.state.value)
            return False
        self.record_audio_activity()
        return self._set_state(State.ACTIVO)

    def start_listening(self) -> bool:
        """Transition ACTIVO -> ESCUCHANDO for manual STT."""
        if self.state is not State.ACTIVO:
            log.warning("[Jarvis] Cannot start listening from %s", self.state.value)
            return False
        self.record_audio_activity()
        return self._set_state(State.ESCUCHANDO)

    def receive_text(self) -> bool:
        """Transition ESCUCHANDO -> PROCESANDO once text is captured."""
        if self.state is not State.ESCUCHANDO:
            log.warning("[Jarvis] Cannot receive text from %s", self.state.value)
            return False
        self.record_audio_activity()
        return self._set_state(State.PROCESANDO)

    def enter_error(self) -> bool:
        """Transition any state -> ERROR and schedule auto-recovery."""
        return self._set_state(State.ERROR)

    def recover_from_error(self) -> bool:
        """Transition ERROR -> ACTIVO after the cooldown window."""
        if self.state is not State.ERROR:
            return False
        self.record_audio_activity()
        return self._set_state(State.ACTIVO)

    def record_audio_activity(self) -> None:
        """Call this when speech is detected to reset silence timer."""
        with self._lock:
            self._last_audio_time = time.time()

    def check_error_timeout(self) -> bool:
        """Auto-recover from ERROR after the configured timeout."""
        with self._lock:
            should_recover = (
                self._state is State.ERROR
                and self._error_recover_at is not None
                and time.time() >= self._error_recover_at
            )
        if not should_recover:
            return False
        log.info("[Jarvis] Error timeout reached, returning to ACTIVO")
        return self.recover_from_error()

    def check_silence_timeout(self, tts_playing: bool = False) -> bool:
        """Check if silence timeout has been exceeded.

        Returns True if 5s of silence has passed (not counting TTS time).
        Only meaningful in ACTIVO state.
        """
        with self._lock:
            if self._state is not State.ACTIVO:
                return False
            if tts_playing:
                # Don't count silence during TTS — reset the timer
                self._last_audio_time = time.time()
                return False
            elapsed = time.time() - self._last_audio_time
            timeout = self._silence_timeout
        if elapsed >= timeout:
            log.info(
                "[Jarvis] Silence timeout reached (%.1fs >= %.1fs)",
                elapsed,
                timeout,
            )
            return True
        return False
