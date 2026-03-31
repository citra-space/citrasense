"""Hardware safety check — bridges an external safety monitor into CitraScope.

Polls a callable that queries the hardware adapter's safety monitor device
(e.g. NINA's safety monitor reporting ceiling/weather status).  Returns
EMERGENCY when the device reports unsafe, WARN when the device state is
unknown, and SAFE when conditions are confirmed safe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from citrascope.safety.safety_monitor import SafetyAction, SafetyCheck


class HardwareSafetyCheck(SafetyCheck):
    """Monitors an external hardware safety device via a query callable.

    The callable returns True (safe), False (unsafe), or None (unknown/
    disconnected).  This check does not import any hardware module — the
    daemon wires the callable at construction time.
    """

    def __init__(self, logger: logging.Logger, query_fn: Callable[[], bool | None]) -> None:
        self._logger = logger
        self._query_fn = query_fn
        self._is_safe: bool | None = None
        self._prev_safe: bool | None = None

    @property
    def name(self) -> str:
        return "hardware_safety"

    def check(self) -> SafetyAction:
        self._prev_safe = self._is_safe
        try:
            self._is_safe = self._query_fn()
        except Exception:
            self._logger.warning("Hardware safety query failed — treating as WARN", exc_info=True)
            self._is_safe = None
            return SafetyAction.WARN

        if self._is_safe is None:
            return SafetyAction.WARN
        if not self._is_safe:
            if self._prev_safe is not False:
                self._logger.critical("Hardware safety monitor reports UNSAFE conditions")
            return SafetyAction.EMERGENCY
        return SafetyAction.SAFE

    def check_proposed_action(self, action_type: str, **kwargs: object) -> bool:
        if self._is_safe is False and action_type in ("slew", "capture"):
            self._logger.warning("Blocking %s — hardware safety monitor reports unsafe", action_type)
            return False
        return True

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "is_safe": self._is_safe,
        }
