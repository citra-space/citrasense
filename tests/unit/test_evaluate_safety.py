"""Tests for TaskDispatcher._evaluate_safety — emergency imaging queue clear (#315)."""

from __future__ import annotations

from unittest.mock import MagicMock

from citrasense.safety.safety_monitor import SafetyAction, SafetyCheck, SafetyMonitor


class _StubCheck(SafetyCheck):
    def __init__(self, name: str, action: SafetyAction = SafetyAction.SAFE):
        self._name = name
        self._action = action

    @property
    def name(self) -> str:
        return self._name

    def check(self) -> SafetyAction:
        return self._action


def _make_task_dispatcher(safety_monitor):
    """Build a minimal TaskDispatcher with mocked dependencies."""
    from citrasense.tasks.task_dispatcher import TaskDispatcher

    td = TaskDispatcher(
        api_client=MagicMock(),
        logger=MagicMock(),
        settings=MagicMock(),
        hardware_adapter=MagicMock(),
        safety_monitor=safety_monitor,
    )

    runtime = MagicMock()
    runtime.sensor_id = "test-scope"
    runtime.sensor_type = "telescope"
    runtime.acquisition_queue = MagicMock()
    runtime.acquisition_queue.is_idle.return_value = True
    runtime.processing_queue = MagicMock()
    runtime.upload_queue = MagicMock()
    runtime.are_queues_idle.return_value = True
    td._runtimes["test-scope"] = runtime
    return td


class TestEvaluateSafetyEmergencyClear:
    def test_emergency_clears_imaging_queue_on_first_transition(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)

        result = td._evaluate_safety()

        assert result is True
        td.imaging_queue.clear.assert_called_once()

    def test_emergency_does_not_clear_on_subsequent_polls(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)

        td._evaluate_safety()
        td.imaging_queue.clear.reset_mock()

        td._evaluate_safety()
        td.imaging_queue.clear.assert_not_called()

    def test_safe_does_not_clear_imaging_queue(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.SAFE)])
        td = _make_task_dispatcher(monitor)

        result = td._evaluate_safety()

        assert result is False
        td.imaging_queue.clear.assert_not_called()

    def test_queue_stop_does_not_clear_imaging_queue(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.QUEUE_STOP)])
        td = _make_task_dispatcher(monitor)

        result = td._evaluate_safety()

        assert result is True
        td.imaging_queue.clear.assert_not_called()

    def test_emergency_recovery_then_re_emergency_clears_again(self):
        check = _StubCheck("hw", SafetyAction.EMERGENCY)
        monitor = SafetyMonitor(MagicMock(), [check])
        td = _make_task_dispatcher(monitor)

        td._evaluate_safety()
        td.imaging_queue.clear.assert_called_once()
        td.imaging_queue.clear.reset_mock()

        check._action = SafetyAction.SAFE
        td._evaluate_safety()

        check._action = SafetyAction.EMERGENCY
        td._evaluate_safety()
        td.imaging_queue.clear.assert_called_once()

    def test_emergency_calls_abort_slew(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)

        td._evaluate_safety()

        td.hardware_adapter.abort_slew.assert_called()

    def test_emergency_fires_toast_on_first_transition(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)
        td.on_toast = MagicMock()

        td._evaluate_safety()

        td.on_toast.assert_called_once()
        msg, toast_type, toast_id = td.on_toast.call_args[0]
        assert "hw" in msg
        assert toast_type == "danger"
        assert toast_id == "safety-emergency"

    def test_emergency_toast_not_fired_without_callback(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)
        assert td.on_toast is None

        td._evaluate_safety()

    def test_emergency_toast_not_fired_on_subsequent_polls(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)
        td.on_toast = MagicMock()

        td._evaluate_safety()
        td.on_toast.reset_mock()

        td._evaluate_safety()
        td.on_toast.assert_not_called()

    def test_emergency_clears_imaging_tasks_dict(self):
        monitor = SafetyMonitor(MagicMock(), [_StubCheck("hw", SafetyAction.EMERGENCY)])
        td = _make_task_dispatcher(monitor)
        td.imaging_tasks["fake-task-id"] = 1234567890.0

        td._evaluate_safety()

        assert td.imaging_tasks == {}
