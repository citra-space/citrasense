"""Tests for SafetyMonitor framework."""

import time
from unittest.mock import MagicMock

from citrascope.safety.safety_monitor import OperatorStopCheck, SafetyAction, SafetyCheck, SafetyMonitor


class _StubCheck(SafetyCheck):
    def __init__(self, name: str, action: SafetyAction = SafetyAction.SAFE):
        self._name = name
        self._action = action

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> SafetyAction:
        return self._action


class _BlockingCheck(SafetyCheck):
    """Check that blocks a specific action type."""

    @property
    def name(self) -> str:
        return "blocker"

    def check(self) -> SafetyAction:
        return SafetyAction.SAFE

    def check_proposed_action(self, action_type: str, **kwargs) -> bool:
        return action_type != "slew"


class _ExplodingCheck(SafetyCheck):
    """Check that raises on every call (both check and pre-action gate)."""

    @property
    def name(self) -> str:
        return "exploding"

    def check(self) -> SafetyAction:
        raise RuntimeError("boom")

    def check_proposed_action(self, action_type: str, **kwargs) -> bool:
        raise RuntimeError("boom")


class TestSafetyMonitorEvaluate:
    def test_empty_checks_is_safe(self):
        monitor = SafetyMonitor(MagicMock(), [])
        action, check = monitor.evaluate()
        assert action == SafetyAction.SAFE
        assert check is None

    def test_all_safe_returns_safe(self):
        checks = [_StubCheck("a"), _StubCheck("b")]
        monitor = SafetyMonitor(MagicMock(), checks)
        action, check = monitor.evaluate()
        assert action == SafetyAction.SAFE
        assert check is None

    def test_returns_most_severe(self):
        checks = [
            _StubCheck("safe", SafetyAction.SAFE),
            _StubCheck("warn", SafetyAction.WARN),
            _StubCheck("stop", SafetyAction.QUEUE_STOP),
        ]
        monitor = SafetyMonitor(MagicMock(), checks)
        action, check = monitor.evaluate()
        assert action == SafetyAction.QUEUE_STOP
        assert check is not None
        assert check.name == "stop"

    def test_emergency_trumps_queue_stop(self):
        checks = [
            _StubCheck("stop", SafetyAction.QUEUE_STOP),
            _StubCheck("emergency", SafetyAction.EMERGENCY),
        ]
        monitor = SafetyMonitor(MagicMock(), checks)
        action, check = monitor.evaluate()
        assert action == SafetyAction.EMERGENCY
        assert check.name == "emergency"

    def test_exploding_check_treated_as_queue_stop(self):
        """A check that raises is treated as QUEUE_STOP (fail-closed)."""
        checks = [_ExplodingCheck(), _StubCheck("ok", SafetyAction.WARN)]
        monitor = SafetyMonitor(MagicMock(), checks)
        action, _check = monitor.evaluate()
        assert action == SafetyAction.QUEUE_STOP

    def test_exploding_check_alone_is_queue_stop(self):
        checks = [_ExplodingCheck()]
        monitor = SafetyMonitor(MagicMock(), checks)
        action, _check = monitor.evaluate()
        assert action == SafetyAction.QUEUE_STOP


class TestSafetyMonitorPreActionGate:
    def test_all_safe_allows_action(self):
        checks = [_StubCheck("a"), _StubCheck("b")]
        monitor = SafetyMonitor(MagicMock(), checks)
        assert monitor.is_action_safe("slew", ra=10, dec=20) is True

    def test_blocker_rejects_action(self):
        checks = [_BlockingCheck()]
        monitor = SafetyMonitor(MagicMock(), checks)
        assert monitor.is_action_safe("slew") is False
        assert monitor.is_action_safe("capture") is True

    def test_exploding_check_blocks_action(self):
        """A check that raises during pre-action gate blocks the action (fail-closed)."""
        checks = [_ExplodingCheck()]
        monitor = SafetyMonitor(MagicMock(), checks)
        assert monitor.is_action_safe("slew") is False


class TestSafetyMonitorWatchdog:
    def test_watchdog_starts_and_stops(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("a")])
        monitor.start_watchdog(interval_seconds=0.1)
        time.sleep(0.3)
        assert monitor.watchdog_healthy is True
        monitor.stop_watchdog()

    def test_watchdog_fires_abort_on_emergency(self):
        abort = MagicMock()
        checks = [_StubCheck("crit", SafetyAction.EMERGENCY)]
        monitor = SafetyMonitor(MagicMock(), checks, abort_callback=abort)
        monitor.start_watchdog(interval_seconds=0.1)
        time.sleep(0.3)
        monitor.stop_watchdog()
        assert abort.called

    def test_watchdog_survives_exploding_check(self):
        checks = [_ExplodingCheck()]
        monitor = SafetyMonitor(MagicMock(), checks)
        monitor.start_watchdog(interval_seconds=0.1)
        time.sleep(0.3)
        assert monitor.watchdog_healthy is True
        monitor.stop_watchdog()

    def test_watchdog_healthy_false_before_start(self):
        monitor = SafetyMonitor(MagicMock(), [])
        assert monitor.watchdog_healthy is False


class TestSafetyMonitorGetStatus:
    def test_get_status_uses_cached_action(self):
        """get_status() must use _last_action from evaluate(), not call check() again."""
        call_count = 0

        class _CountingCheck(SafetyCheck):
            @property
            def name(self) -> str:
                return "counting"

            def check(self) -> SafetyAction:
                nonlocal call_count
                call_count += 1
                return SafetyAction.WARN

        checks = [_CountingCheck()]
        monitor = SafetyMonitor(MagicMock(), checks)
        monitor.evaluate()
        assert call_count == 1
        status = monitor.get_status()
        assert call_count == 1
        # checks[0] is the built-in operator_stop; registered checks follow
        assert status["checks"][1]["action"] == "warn"

    def test_get_status_before_evaluate_defaults_safe(self):
        checks = [_StubCheck("a", SafetyAction.EMERGENCY)]
        monitor = SafetyMonitor(MagicMock(), checks)
        status = monitor.get_status()
        # checks[0] is operator_stop (safe), registered checks follow
        assert status["checks"][0]["action"] == "safe"
        assert status["checks"][1]["action"] == "safe"

    def test_get_status_includes_operator_stop(self):
        monitor = SafetyMonitor(MagicMock(), [])
        status = monitor.get_status()
        op = status["checks"][0]
        assert op["name"] == "operator_stop"
        assert op["active"] is False
        assert op["action"] == "safe"

        monitor.activate_operator_stop()
        status = monitor.get_status()
        op = status["checks"][0]
        assert op["active"] is True
        assert op["action"] == "emergency"


class TestSafetyMonitorGetCheck:
    def test_get_check_found(self):
        checks = [_StubCheck("a"), _StubCheck("b")]
        monitor = SafetyMonitor(MagicMock(), checks)
        assert monitor.get_check("b") is checks[1]

    def test_get_check_not_found(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("a")])
        assert monitor.get_check("missing") is None


class TestOperatorStop:
    def test_initially_not_stopped(self):
        monitor = SafetyMonitor(MagicMock(), [])
        assert monitor.is_operator_stopped is False

    def test_activate_returns_emergency(self):
        monitor = SafetyMonitor(MagicMock(), [])
        monitor.activate_operator_stop()
        assert monitor.is_operator_stopped is True
        action, _check = monitor.evaluate()
        assert action == SafetyAction.EMERGENCY

    def test_activate_is_idempotent(self):
        logger = MagicMock()
        monitor = SafetyMonitor(logger, [])
        monitor.activate_operator_stop()
        monitor.activate_operator_stop()
        assert monitor.is_operator_stopped is True
        assert logger.warning.call_count == 1

    def test_clear_restores_safe(self):
        monitor = SafetyMonitor(MagicMock(), [])
        monitor.activate_operator_stop()
        monitor.clear_operator_stop()
        assert monitor.is_operator_stopped is False
        action, _check = monitor.evaluate()
        assert action == SafetyAction.SAFE

    def test_clear_is_idempotent(self):
        logger = MagicMock()
        monitor = SafetyMonitor(logger, [])
        monitor.clear_operator_stop()
        assert logger.info.call_count == 0

    def test_blocks_all_actions(self):
        monitor = SafetyMonitor(MagicMock(), [])
        assert monitor.is_action_safe("slew") is True
        monitor.activate_operator_stop()
        assert monitor.is_action_safe("slew") is False
        assert monitor.is_action_safe("home") is False
        assert monitor.is_action_safe("capture") is False
        assert monitor.is_action_safe("unwind") is False

    def test_triggers_abort_via_watchdog(self):
        abort = MagicMock()
        monitor = SafetyMonitor(MagicMock(), [], abort_callback=abort)
        monitor.activate_operator_stop()
        monitor.start_watchdog(interval_seconds=0.05)
        time.sleep(0.2)
        monitor.stop_watchdog()
        assert abort.called

    def test_takes_priority_over_check_emergency(self):
        """Operator stop wins even when a registered check is also EMERGENCY."""
        cable_emergency = _StubCheck("cable_wrap", SafetyAction.EMERGENCY)
        monitor = SafetyMonitor(MagicMock(), [cable_emergency])
        monitor.activate_operator_stop()
        action, triggered = monitor.evaluate()
        assert action == SafetyAction.EMERGENCY
        # OperatorStopCheck is first in the list so it's the worst_check when
        # both it and cable_wrap return EMERGENCY (ties keep the first).
        assert triggered is monitor.operator_stop

    def test_emergency_without_registered_checks(self):
        monitor = SafetyMonitor(MagicMock(), [])
        monitor.activate_operator_stop()
        action, triggered = monitor.evaluate()
        assert action == SafetyAction.EMERGENCY
        assert triggered is monitor.operator_stop


class TestOperatorStopCheck:
    """Isolated tests for OperatorStopCheck as a standalone SafetyCheck."""

    def test_defaults_to_safe(self):
        chk = OperatorStopCheck()
        assert chk.is_active is False
        assert chk.check() == SafetyAction.SAFE

    def test_activate_returns_emergency(self):
        chk = OperatorStopCheck()
        chk.activate()
        assert chk.is_active is True
        assert chk.check() == SafetyAction.EMERGENCY

    def test_clear_returns_safe(self):
        chk = OperatorStopCheck()
        chk.activate()
        chk.clear()
        assert chk.is_active is False
        assert chk.check() == SafetyAction.SAFE

    def test_activate_is_idempotent(self):
        chk = OperatorStopCheck()
        chk.activate()
        chk.activate()
        assert chk.is_active is True

    def test_clear_is_idempotent(self):
        chk = OperatorStopCheck()
        chk.clear()
        assert chk.is_active is False

    def test_blocks_all_proposed_actions_when_active(self):
        chk = OperatorStopCheck()
        assert chk.check_proposed_action("slew") is True
        assert chk.check_proposed_action("home") is True
        chk.activate()
        assert chk.check_proposed_action("slew") is False
        assert chk.check_proposed_action("home") is False
        assert chk.check_proposed_action("capture") is False
        assert chk.check_proposed_action("unwind") is False

    def test_name(self):
        chk = OperatorStopCheck()
        assert chk.name == "operator_stop"

    def test_get_status(self):
        chk = OperatorStopCheck()
        status = chk.get_status()
        assert status["name"] == "operator_stop"
        assert status["active"] is False

        chk.activate()
        status = chk.get_status()
        assert status["active"] is True

    def test_last_action_synced_on_activate(self):
        chk = OperatorStopCheck()
        assert chk._last_action == SafetyAction.SAFE
        chk.activate()
        assert chk._last_action == SafetyAction.EMERGENCY
        chk.clear()
        assert chk._last_action == SafetyAction.SAFE
