"""Latched operator stop — blocks all motion until explicitly cleared."""

from __future__ import annotations

import threading

from citrascope.safety.safety_monitor import SafetyAction, SafetyCheck


class OperatorStopCheck(SafetyCheck):
    """Latched operator stop — blocks all motion until explicitly cleared.

    Session-scoped (no persistence).  When active, ``check()`` returns
    EMERGENCY and ``check_proposed_action()`` blocks every action type.
    """

    def __init__(self) -> None:
        self._active = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "operator_stop"

    def check(self) -> SafetyAction:
        with self._lock:
            return SafetyAction.EMERGENCY if self._active else SafetyAction.SAFE

    def check_proposed_action(self, action_type: str, **kwargs) -> bool:
        with self._lock:
            return not self._active

    def get_status(self) -> dict:
        with self._lock:
            active = self._active
        return {
            "name": self.name,
            "active": active,
        }

    # -- public API for activate / clear / query -----------------------

    def activate(self) -> None:
        """Latch the stop state — all motion is blocked until cleared."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._last_action = SafetyAction.EMERGENCY

    def clear(self) -> None:
        """Release the stop state — motion may resume."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._last_action = SafetyAction.SAFE

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active
