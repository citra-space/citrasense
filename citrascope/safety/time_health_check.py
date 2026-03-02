"""Time health safety check — wraps the existing TimeMonitor as a SafetyCheck.

Reads ``TimeMonitor.get_current_health()`` and maps ``TimeStatus`` to
``SafetyAction``.  The actual NTP/Chrony monitoring logic stays in
``TimeMonitor``; this check just bridges it into the SafetyMonitor framework.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from citrascope.safety.safety_monitor import SafetyAction, SafetyCheck
from citrascope.time.time_health import TimeStatus

if TYPE_CHECKING:
    from citrascope.time.time_monitor import TimeMonitor


class TimeHealthCheck(SafetyCheck):
    """Maps TimeMonitor health status into SafetyAction levels."""

    def __init__(self, logger: logging.Logger, time_monitor: TimeMonitor) -> None:
        self._logger = logger
        self._time_monitor = time_monitor

    @property
    def name(self) -> str:
        return "time_health"

    def check(self) -> SafetyAction:
        health = self._time_monitor.get_current_health()
        if health is None:
            return SafetyAction.SAFE
        if health.status == TimeStatus.CRITICAL:
            return SafetyAction.QUEUE_STOP
        return SafetyAction.SAFE

    def check_proposed_action(self, action: str, **kwargs: object) -> bool:
        health = self._time_monitor.get_current_health()
        if health is not None and health.status == TimeStatus.CRITICAL and action in ("slew", "capture"):
            self._logger.warning("Blocking %s — time sync is CRITICAL (offset %.0f ms)", action, health.offset_ms)
            return False
        return True

    def get_status(self) -> dict:
        health = self._time_monitor.get_current_health()
        result: dict = {"name": self.name}
        if health:
            result["offset_ms"] = health.offset_ms
            result["source"] = health.source
            result["time_status"] = health.status.value
        return result
