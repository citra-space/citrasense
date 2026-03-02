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


from citrascope.safety.operator_stop_check import OperatorStopCheck  # noqa: E402

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

        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_interval: float = 1.0
        self._watchdog_last_heartbeat: float = 0.0
        self._last_watchdog_action: SafetyAction = SafetyAction.SAFE
        self._action_threads: list[threading.Thread] = []

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

    def evaluate(self) -> tuple[SafetyAction, SafetyCheck | None]:
        """Run all checks, return the most severe action and its trigger.

        Fail-closed: a check that raises is treated as QUEUE_STOP so new work
        is blocked until the check recovers.  We don't escalate to EMERGENCY
        because a code bug shouldn't trigger abort_slew.
        """
        worst_action = SafetyAction.SAFE
        worst_check: SafetyCheck | None = None

        for chk in self._checks:
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

    def is_action_safe(self, action_type: str, **kwargs) -> bool:
        """Pre-action gate: ask every check whether *action_type* is safe.

        Fail-closed: if a registered check raises, the action is blocked.
        """
        for chk in self._checks:
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
        for chk in self._checks:
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
        """
        check_statuses: list[dict] = []
        for chk in self._checks:
            try:
                status = chk.get_status()
                status["action"] = chk._last_action.value
            except Exception:
                status = {"name": chk.name, "action": "error"}
            check_statuses.append(status)
        return {
            "checks": check_statuses,
            "watchdog_alive": self.watchdog_healthy,
            "watchdog_last_heartbeat": self._watchdog_last_heartbeat,
        }
