"""SafetyMonitor framework — pluggable safety checks with watchdog thread.

Three layers of enforcement:
  1. **Watchdog thread** — polls checks every few seconds, fires abort on EMERGENCY
  2. **Pre-action gate** — ``is_action_safe()`` called before slews/sequences
  3. **Task-loop integration** — ``evaluate()`` called each cycle for cooperative fixes
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum


class SafetyAction(Enum):
    """Severity level returned by a SafetyCheck."""

    SAFE = "safe"
    WARN = "warn"
    QUEUE_STOP = "stop"
    EMERGENCY = "emergency"


_ACTION_SEVERITY = {
    SafetyAction.SAFE: 0,
    SafetyAction.WARN: 1,
    SafetyAction.QUEUE_STOP: 2,
    SafetyAction.EMERGENCY: 3,
}


class SafetyError(RuntimeError):
    """Raised when a pre-action gate blocks an unsafe operation."""


class SafetyCheck(ABC):
    """Abstract base for a single safety check."""

    _last_action: SafetyAction = SafetyAction.SAFE
    # Set by :meth:`SafetyMonitor.register_sensor_check` when the check is
    # scoped to a particular sensor.  ``None`` means a site-level check.
    sensor_id: str | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check(self) -> SafetyAction:
        """Assess current conditions and return a severity level."""

    def execute_action(self) -> None:  # noqa: B027
        """Perform corrective action (e.g. cable unwind). Optional."""

    def reset(self) -> None:  # noqa: B027
        """Clear state after a corrective action completes. Optional."""

    def check_proposed_action(self, action_type: str, **kwargs) -> bool:
        """Pre-flight gate: is this specific action safe to start?

        Returns True if safe, False to block.  Default allows everything.
        """
        return True

    def get_status(self) -> dict:
        """Return check-specific status for the web UI. Optional."""
        return {"name": self.name}


from citrasense.safety.operator_stop_check import OperatorStopCheck  # noqa: E402

__all__ = [
    "OperatorStopCheck",
    "SafetyAction",
    "SafetyCheck",
    "SafetyError",
    "SafetyMonitor",
]


class SafetyMonitor:
    """Orchestrates multiple SafetyCheck instances with a watchdog thread.

    An ``OperatorStopCheck`` is automatically prepended to the check list
    and exposed via ``self.operator_stop`` so callers can activate/clear
    the stop without reaching through ``get_check()``.
    """

    def __init__(
        self,
        logger: logging.Logger,
        checks: list[SafetyCheck],
        abort_callback: Callable[[], None] | None = None,
    ) -> None:
        self._logger = logger
        self._abort_callback = abort_callback

        self.operator_stop = OperatorStopCheck()
        self._checks: list[SafetyCheck] = [self.operator_stop, *checks]

        self._sensor_checks: dict[str, list[SafetyCheck]] = {}
        self._checks_lock = threading.Lock()

        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_interval: float = 1.0
        self._watchdog_last_heartbeat: float = 0.0
        self._last_watchdog_action: SafetyAction = SafetyAction.SAFE
        self._action_threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Per-sensor check registration
    # ------------------------------------------------------------------

    def register_sensor_check(self, sensor_id: str, check: SafetyCheck) -> None:
        """Register a safety check owned by a specific sensor.

        Duplicate registrations (same *sensor_id* + *check.name*) are
        silently ignored so callers don't need to track prior state.
        """
        with self._checks_lock:
            sensor_list = self._sensor_checks.setdefault(sensor_id, [])
            if any(existing.name == check.name for existing in sensor_list):
                self._logger.warning(
                    "Skipped duplicate sensor check %r for sensor %s",
                    check.name,
                    sensor_id,
                )
                return
            check.sensor_id = sensor_id
            sensor_list.append(check)
            self._checks.append(check)
        self._logger.info("Registered sensor check %r for sensor %s", check.name, sensor_id)

    def unregister_sensor_check(self, sensor_id: str, check_name: str) -> SafetyCheck | None:
        """Remove a sensor check by name. Returns the check or ``None``.

        Idempotent — safe to call even if the check was already removed.
        """
        with self._checks_lock:
            sensor_list = self._sensor_checks.get(sensor_id, [])
            for chk in sensor_list:
                if chk.name == check_name:
                    sensor_list.remove(chk)
                    try:
                        self._checks.remove(chk)
                    except ValueError:
                        pass
                    self._logger.info("Unregistered sensor check %r for sensor %s", check_name, sensor_id)
                    return chk
        return None

    def get_sensor_checks(self, sensor_id: str) -> list[SafetyCheck]:
        """Return a copy of checks registered for a given sensor."""
        with self._checks_lock:
            return list(self._sensor_checks.get(sensor_id, []))

    # ------------------------------------------------------------------
    # Operator stop  (convenience pass-throughs)
    # ------------------------------------------------------------------

    def activate_operator_stop(self) -> None:
        was_active = self.operator_stop.is_active
        self.operator_stop.activate()
        if not was_active and self.operator_stop.is_active:
            self._logger.warning("Operator STOP activated — all motion blocked")

    def clear_operator_stop(self) -> None:
        was_active = self.operator_stop.is_active
        self.operator_stop.clear()
        if was_active and not self.operator_stop.is_active:
            self._logger.info("Operator stop cleared")

    @property
    def is_operator_stopped(self) -> bool:
        return self.operator_stop.is_active

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _checks_for(self, sensor_id: str | None) -> list[SafetyCheck]:
        """Return a snapshot of checks relevant to the given sensor.

        ``sensor_id=None`` — site-wide evaluation, returns every registered
        check (watchdog path, site-level UI).

        ``sensor_id="..."`` — per-sensor evaluation, returns site-level
        checks (``chk.sensor_id is None``) plus checks owned by that
        sensor.  Checks owned by *other* sensors are filtered out so one
        scope's cable unwind (or any sensor-scoped ``QUEUE_STOP``) doesn't
        veto work on its siblings.
        """
        with self._checks_lock:
            if sensor_id is None:
                return list(self._checks)
            return [chk for chk in self._checks if chk.sensor_id in (None, sensor_id)]

    def evaluate(self, sensor_id: str | None = None) -> tuple[SafetyAction, SafetyCheck | None]:
        """Run checks and return the most severe action and its trigger.

        When ``sensor_id`` is given, only site-level checks plus that
        sensor's own checks participate — used by the per-runtime gate in
        :class:`citrasense.tasks.task_dispatcher.TaskDispatcher` so one
        sensor's ``QUEUE_STOP`` doesn't freeze another sensor's scheduling.
        The watchdog continues to call ``evaluate()`` with no argument so
        it still sees site-wide EMERGENCY conditions.

        Fail-closed: a check that raises is treated as QUEUE_STOP so new work
        is blocked until the check recovers.  We don't escalate to EMERGENCY
        because a code bug shouldn't trigger abort_slew.
        """
        worst_action = SafetyAction.SAFE
        worst_check: SafetyCheck | None = None

        for chk in self._checks_for(sensor_id):
            try:
                action = chk.check()
            except Exception:
                self._logger.error(
                    "Safety check %r raised an exception — treating as QUEUE_STOP", chk.name, exc_info=True
                )
                action = SafetyAction.QUEUE_STOP
            chk._last_action = action
            if _ACTION_SEVERITY[action] > _ACTION_SEVERITY[worst_action]:
                worst_action = action
                worst_check = chk
        return worst_action, worst_check

    def is_action_safe(self, action_type: str, sensor_id: str | None = None, **kwargs) -> bool:
        """Pre-action gate: ask the relevant checks whether *action_type* is safe.

        When ``sensor_id`` is passed, only site-level checks
        (``OperatorStopCheck``, ``DiskSpaceCheck``, ``TimeHealthCheck``,
        ...) plus that sensor's own checks get consulted.  Without it every
        registered check votes — matches the historical contract used by
        tests that never cared about scoping.

        Fail-closed: if a registered check raises, the action is blocked.
        """
        for chk in self._checks_for(sensor_id):
            try:
                if not chk.check_proposed_action(action_type, **kwargs):
                    self._logger.warning("Safety check %r blocked action %r", chk.name, action_type)
                    return False
            except Exception:
                self._logger.error(
                    "Safety check %r raised during pre-action gate — blocking %r", chk.name, action_type, exc_info=True
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Watchdog thread  (Layer 1)
    # ------------------------------------------------------------------

    def start_watchdog(self, interval_seconds: float = 1.0) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_interval = interval_seconds
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="safety-watchdog")
        self._watchdog_thread.start()
        self._logger.info("Safety watchdog started (interval=%.1fs)", interval_seconds)

    def stop_watchdog(self) -> None:
        if self._watchdog_thread is None:
            return
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=self._watchdog_interval + 2)
        self._watchdog_thread = None
        for t in self._action_threads:
            if t.is_alive():
                t.join(timeout=10.0)
        self._action_threads.clear()
        self._logger.info("Safety watchdog stopped")

    @property
    def watchdog_healthy(self) -> bool:
        """True if watchdog has checked in within 3x the poll interval."""
        if self._watchdog_last_heartbeat == 0.0:
            return False
        return (time.monotonic() - self._watchdog_last_heartbeat) < self._watchdog_interval * 3

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.is_set():
            try:
                self._watchdog_last_heartbeat = time.monotonic()
                action, triggered_check = self.evaluate()
                if action == SafetyAction.EMERGENCY:
                    is_new = self._last_watchdog_action != SafetyAction.EMERGENCY
                    if is_new and triggered_check:
                        self._logger.critical(
                            "SAFETY EMERGENCY from %r — aborting motion",
                            triggered_check.name,
                        )
                    if self._abort_callback:
                        try:
                            self._abort_callback()
                        except Exception:
                            self._logger.error("abort_callback raised", exc_info=True)
                    if triggered_check and is_new:
                        self._run_corrective_action(triggered_check)
                self._last_watchdog_action = action
            except Exception:
                self._logger.error("Watchdog cycle failed", exc_info=True)
            self._watchdog_stop.wait(self._watchdog_interval)

    def _run_corrective_action(self, check: SafetyCheck) -> None:
        """Fire a check's corrective action in a background thread.

        ``execute_action`` can block (e.g. cable unwind takes minutes),
        so we avoid stalling the watchdog loop.  The check itself is
        responsible for idempotency — duplicate calls are no-ops.
        """

        def _action() -> None:
            try:
                check.execute_action()
            except Exception:
                self._logger.error("Corrective action from %r failed", check.name, exc_info=True)

        t = threading.Thread(target=_action, daemon=True, name=f"safety-action-{check.name}")
        self._action_threads = [t for t in self._action_threads if t.is_alive()]
        t.start()
        self._action_threads.append(t)

    # ------------------------------------------------------------------
    # Check lookup
    # ------------------------------------------------------------------

    def get_check(self, name: str) -> SafetyCheck | None:
        """Find a registered check by name."""
        with self._checks_lock:
            checks_snapshot = list(self._checks)
        for chk in checks_snapshot:
            if chk.name == name:
                return chk
        return None

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return status of all checks for the web UI.

        Uses the cached ``_last_action`` from the most recent ``evaluate()``
        call instead of calling ``check()`` again, because some checks (e.g.
        CableWrapCheck) have side effects in ``check()``.

        Sensor-owned checks include a ``sensor_id`` key so the frontend can
        associate them with the correct sensor panel.
        """
        with self._checks_lock:
            checks_snapshot = list(self._checks)
            chk_to_sensor: dict[int, str] = {}
            for sid, chk_list in self._sensor_checks.items():
                for chk in chk_list:
                    chk_to_sensor[id(chk)] = sid
        check_statuses: list[dict] = []
        for chk in checks_snapshot:
            try:
                status = chk.get_status()
                status["action"] = chk._last_action.value
            except Exception:
                status = {"name": chk.name, "action": "error"}
            sensor_id = chk_to_sensor.get(id(chk))
            if sensor_id is not None:
                status["sensor_id"] = sensor_id
            check_statuses.append(status)
        return {
            "checks": check_statuses,
            "watchdog_alive": self.watchdog_healthy,
            "watchdog_last_heartbeat": self._watchdog_last_heartbeat,
        }
