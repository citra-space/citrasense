"""Tests for TaskDispatcher: routing, facade, runtime registration, queue
management, and safety evaluation."""

from __future__ import annotations

import heapq
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from dateutil import parser as dtparser

from citrasense.safety.safety_monitor import SafetyAction
from citrasense.tasks.task import Task
from citrasense.tasks.task_dispatcher import TaskDispatcher

# ── helpers ────────────────────────────────────────────────────────────────


def _mock_settings():
    s = MagicMock()
    s.task_processing_paused = False
    return s


def _make_dispatcher(**overrides) -> TaskDispatcher:
    defaults = {
        "api_client": MagicMock(),
        "logger": MagicMock(),
        "settings": _mock_settings(),
    }
    defaults.update(overrides)
    return TaskDispatcher(**defaults)


def _mock_runtime(sensor_id: str = "scope-0", sensor_type: str = "telescope"):
    rt = MagicMock()
    rt.sensor_id = sensor_id
    rt.sensor_type = sensor_type
    rt.acquisition_queue = MagicMock()
    rt.acquisition_queue.is_idle.return_value = True
    rt.processing_queue = MagicMock()
    rt.upload_queue = MagicMock()
    rt.autofocus_manager = MagicMock()
    rt.alignment_manager = MagicMock()
    rt.homing_manager = MagicMock()
    rt.calibration_manager = None
    rt.are_queues_idle.return_value = True
    return rt


def _mock_task(sensor_type="telescope", sensor_id=None):
    task = MagicMock()
    task.sensor_type = sensor_type
    task.sensor_id = sensor_id
    task.id = "task-001"
    return task


# ── Runtime registration ──────────────────────────────────────────────────


class TestRuntimeRegistration:
    def test_register_runtime_sets_dispatcher(self):
        td = _make_dispatcher()
        rt = _mock_runtime()
        td.register_runtime(rt)

        rt.set_dispatcher.assert_called_once_with(td)
        assert td._runtimes["scope-0"] is rt

    def test_registered_runtime_accessible(self):
        td = _make_dispatcher()
        rt = _mock_runtime()
        td.register_runtime(rt)

        assert td.get_runtime(rt.sensor_id) is rt

    def test_multiple_runtimes(self):
        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt2 = _mock_runtime("radar-0", "passive_radar")
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        assert len(td._runtimes) == 2


# ── Task routing ──────────────────────────────────────────────────────────


class TestTaskRouting:
    def test_routes_by_sensor_id(self):
        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt2 = _mock_runtime("radar-0", "passive_radar")
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        task = _mock_task(sensor_type="passive_radar", sensor_id="radar-0")
        assert td._runtime_for_task(task) is rt2

    def test_routes_by_sensor_type_fallback(self):
        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt2 = _mock_runtime("radar-0", "passive_radar")
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        task = _mock_task(sensor_type="passive_radar", sensor_id=None)
        assert td._runtime_for_task(task) is rt2

    def test_returns_none_for_unknown_sensor_type(self):
        td = _make_dispatcher()
        rt = _mock_runtime("scope-0", "telescope")
        td.register_runtime(rt)

        task = _mock_task(sensor_type="unknown", sensor_id=None)
        assert td._runtime_for_task(task) is None

    def test_rejects_task_with_no_runtimes(self):
        td = _make_dispatcher()
        task = _mock_task()
        assert td._runtime_for_task(task) is None


# ── Stage tracking ────────────────────────────────────────────────────────


class TestStageTracking:
    def test_update_task_stage_imaging(self):
        td = _make_dispatcher()
        td.update_task_stage("t1", "imaging")
        assert "t1" in td.imaging_tasks

    def test_update_task_stage_moves_between_stages(self):
        td = _make_dispatcher()
        td.update_task_stage("t1", "imaging")
        td.update_task_stage("t1", "processing")
        assert "t1" not in td.imaging_tasks
        assert "t1" in td.processing_tasks

    def test_remove_task_from_all_stages(self):
        td = _make_dispatcher()
        rt = _mock_runtime("scope-0")
        td.register_runtime(rt)
        td.update_task_stage("t1", "uploading")
        td._sensor_task_dicts["scope-0"]["t1"] = MagicMock()
        td.remove_task_from_all_stages("t1")
        assert "t1" not in td.uploading_tasks
        assert "t1" not in td._sensor_task_dicts["scope-0"]


# ── Stats ─────────────────────────────────────────────────────────────────


class TestStats:
    def test_lifetime_counters(self):
        td = _make_dispatcher()
        td.record_task_started()
        td.record_task_started()
        td.record_task_succeeded()
        td.record_task_failed()
        stats = td.get_task_stats()
        assert stats == {"started": 2, "succeeded": 1, "failed": 1}


# ── Drop scheduled task ──────────────────────────────────────────────────


class TestDropScheduledTask:
    def test_drop_removes_from_heap(self):
        td = _make_dispatcher()
        rt = _mock_runtime("scope-0")
        td.register_runtime(rt)
        task = MagicMock()
        heapq.heappush(td._sensor_heaps["scope-0"], (1000, 2000, "t1", task))
        td._sensor_task_ids["scope-0"].add("t1")
        td._sensor_task_dicts["scope-0"]["t1"] = task

        assert td.drop_scheduled_task("t1") is True
        assert "t1" not in td._sensor_task_ids["scope-0"]
        assert "t1" not in td._sensor_task_dicts["scope-0"]
        assert all(entry[2] != "t1" for entry in td._sensor_heaps["scope-0"])

    def test_drop_unknown_returns_false(self):
        td = _make_dispatcher()
        assert td.drop_scheduled_task("nope") is False


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_starts_runtimes(self):
        td = _make_dispatcher()
        rt = _mock_runtime()
        td.register_runtime(rt)

        td.start()

        rt.start.assert_called_once()

        td.stop()

        rt.stop.assert_called_once()


# ── Queue management (migrated from test_task_manager.py) ────────────────


def _create_test_task(task_id, status="Pending", start_offset_seconds=60):
    now = datetime.now(timezone.utc)
    start_time = now + timedelta(seconds=start_offset_seconds)
    stop_time = start_time + timedelta(seconds=300)
    return Task(
        id=task_id,
        type="observation",
        status=status,
        creationEpoch=now.isoformat(),
        updateEpoch=now.isoformat(),
        taskStart=start_time.isoformat(),
        taskStop=stop_time.isoformat(),
        userId="user-123",
        username="testuser",
        satelliteId="sat-456",
        satelliteName="Test Satellite",
        telescopeId="test-telescope-123",
        telescopeName="Test Telescope",
        groundStationId="test-gs-456",
        groundStationName="Test Ground Station",
    )


WIRED_SID = "test-telescope-123"


@pytest.fixture
def wired_dispatcher():
    """Create a TaskDispatcher with a registered runtime, for queue management tests."""
    api_client = MagicMock()
    api_client.get_telescope_tasks.return_value = []
    api_client.put_telescope_status.return_value = None
    settings = MagicMock()
    settings.keep_images = False
    settings.max_task_retries = 3
    settings.initial_retry_delay_seconds = 30
    settings.max_retry_delay_seconds = 300
    settings.task_processing_paused = False

    td = TaskDispatcher(
        api_client=api_client,
        logger=MagicMock(),
        settings=settings,
    )

    runtime = MagicMock()
    runtime.sensor_id = WIRED_SID
    runtime.sensor_type = "telescope"
    runtime.sensor.citra_record = {"id": WIRED_SID, "maxSlewRate": 5.0, "automatedScheduling": False}
    runtime.acquisition_queue = MagicMock()
    runtime.acquisition_queue.is_idle.return_value = True
    runtime.processing_queue = MagicMock()
    runtime.upload_queue = MagicMock()
    runtime.are_queues_idle.return_value = True
    td.register_runtime(runtime)
    return td, api_client


def test_poll_tasks_adds_new_tasks(wired_dispatcher):
    td, api_client = wired_dispatcher
    heap = td._sensor_heaps[WIRED_SID]
    ids = td._sensor_task_ids[WIRED_SID]
    dicts = td._sensor_task_dicts[WIRED_SID]
    task1 = _create_test_task("task-001", "Pending")
    task2 = _create_test_task("task-002", "Scheduled", start_offset_seconds=120)

    api_client.get_telescope_tasks.return_value = [task1.__dict__, task2.__dict__]

    with td.heap_lock:
        td._report_online()
        tasks = api_client.get_telescope_tasks(td.telescope_runtimes()[0].sensor.citra_record["id"])
        api_task_map = {}
        for task_dict in tasks:
            task = Task.from_dict(task_dict)
            tid = task.id
            if tid and task.status in ["Pending", "Scheduled"]:
                api_task_map[tid] = task

        now = int(time.time())
        active_ids = set(td.current_task_ids.values())
        for tid, task in api_task_map.items():
            if tid not in ids and tid not in active_ids:
                start_epoch = int(dtparser.isoparse(task.taskStart).timestamp())
                stop_epoch = int(dtparser.isoparse(task.taskStop).timestamp()) if task.taskStop else 0
                if not (stop_epoch and stop_epoch < now):
                    heapq.heappush(heap, (start_epoch, stop_epoch, tid, task))
                    ids.add(tid)
                    dicts[tid] = task

    assert len(heap) == 2
    assert "task-001" in ids
    assert "task-002" in ids


def test_poll_tasks_removes_cancelled_tasks(wired_dispatcher):
    td, api_client = wired_dispatcher
    heap = td._sensor_heaps[WIRED_SID]
    ids = td._sensor_task_ids[WIRED_SID]
    dicts = td._sensor_task_dicts[WIRED_SID]
    task1 = _create_test_task("task-001", "Pending")
    task2 = _create_test_task("task-002", "Pending", start_offset_seconds=120)

    start1 = int(dtparser.isoparse(task1.taskStart).timestamp())
    stop1 = int(dtparser.isoparse(task1.taskStop).timestamp())
    start2 = int(dtparser.isoparse(task2.taskStart).timestamp())
    stop2 = int(dtparser.isoparse(task2.taskStop).timestamp())

    with td.heap_lock:
        heapq.heappush(heap, (start1, stop1, "task-001", task1))
        heapq.heappush(heap, (start2, stop2, "task-002", task2))
        ids.update({"task-001", "task-002"})
        dicts.update({"task-001": task1, "task-002": task2})

    assert len(heap) == 2

    api_client.get_telescope_tasks.return_value = [task1.__dict__]

    with td.heap_lock:
        tasks = api_client.get_telescope_tasks(td.telescope_runtimes()[0].sensor.citra_record["id"])
        api_task_map = {}
        for task_dict in tasks:
            task = Task.from_dict(task_dict)
            tid = task.id
            if tid and task.status in ["Pending", "Scheduled"]:
                api_task_map[tid] = task

        active_ids = set(td.current_task_ids.values())
        new_heap = []
        removed = 0
        for se, so, tid, task in heap:
            if tid in active_ids or tid in api_task_map:
                new_heap.append((se, so, tid, task))
            else:
                ids.discard(tid)
                dicts.pop(tid, None)
                removed += 1
        if removed > 0:
            td._sensor_heaps[WIRED_SID] = new_heap
            heapq.heapify(td._sensor_heaps[WIRED_SID])

    assert len(td._sensor_heaps[WIRED_SID]) == 1
    assert "task-001" in ids
    assert "task-002" not in ids


def test_poll_tasks_removes_tasks_with_changed_status(wired_dispatcher):
    td, api_client = wired_dispatcher
    heap = td._sensor_heaps[WIRED_SID]
    ids = td._sensor_task_ids[WIRED_SID]
    dicts = td._sensor_task_dicts[WIRED_SID]
    task1 = _create_test_task("task-001", "Pending")
    se = int(dtparser.isoparse(task1.taskStart).timestamp())
    so = int(dtparser.isoparse(task1.taskStop).timestamp())

    with td.heap_lock:
        heapq.heappush(heap, (se, so, "task-001", task1))
        ids.add("task-001")
        dicts["task-001"] = task1

    cancelled = _create_test_task("task-001", "Cancelled")
    api_client.get_telescope_tasks.return_value = [cancelled.__dict__]

    with td.heap_lock:
        tasks = api_client.get_telescope_tasks(td.telescope_runtimes()[0].sensor.citra_record["id"])
        api_task_map = {}
        for td2 in tasks:
            t = Task.from_dict(td2)
            if t.id and t.status in ["Pending", "Scheduled"]:
                api_task_map[t.id] = t

        active_ids = set(td.current_task_ids.values())
        new_heap = []
        for se2, so2, tid, task in heap:
            if tid in active_ids or tid in api_task_map:
                new_heap.append((se2, so2, tid, task))
            else:
                ids.discard(tid)
                dicts.pop(tid, None)
        td._sensor_heaps[WIRED_SID] = new_heap
        heapq.heapify(td._sensor_heaps[WIRED_SID])

    assert len(td._sensor_heaps[WIRED_SID]) == 0
    assert "task-001" not in ids


def test_poll_tasks_does_not_remove_current_task(wired_dispatcher):
    td, api_client = wired_dispatcher
    heap = td._sensor_heaps[WIRED_SID]
    ids = td._sensor_task_ids[WIRED_SID]
    dicts = td._sensor_task_dicts[WIRED_SID]
    task1 = _create_test_task("task-001", "Pending")
    se = int(dtparser.isoparse(task1.taskStart).timestamp())
    so = int(dtparser.isoparse(task1.taskStop).timestamp())

    with td.heap_lock:
        heapq.heappush(heap, (se, so, "task-001", task1))
        ids.add("task-001")
        dicts["task-001"] = task1
        td._current_task_ids[WIRED_SID] = "task-001"

    api_client.get_telescope_tasks.return_value = []

    with td.heap_lock:
        tasks = api_client.get_telescope_tasks(td.telescope_runtimes()[0].sensor.citra_record["id"])
        api_task_map = {}
        for td2 in tasks:
            t = Task.from_dict(td2)
            if t.id and t.status in ["Pending", "Scheduled"]:
                api_task_map[t.id] = t

        active_ids = set(td.current_task_ids.values())
        new_heap = []
        for se2, so2, tid, task in heap:
            if tid in active_ids or tid in api_task_map:
                new_heap.append((se2, so2, tid, task))
            else:
                ids.discard(tid)
        td._sensor_heaps[WIRED_SID] = new_heap
        heapq.heapify(td._sensor_heaps[WIRED_SID])

    assert len(td._sensor_heaps[WIRED_SID]) == 1
    assert "task-001" in ids


# ── _evaluate_safety_for — cable wrap soft-lock regression (#239) ────────


class TestEvaluateSafetyQueueStop:
    """Verify QUEUE_STOP always attempts corrective action when the imaging
    queue is idle, even if the state transition already happened on a
    previous tick (regression for issue #239 soft-lock)."""

    def _call(self, td, *, queue_idle: bool, action: SafetyAction, triggered_check=None):
        mock_monitor = MagicMock()
        mock_monitor.evaluate.return_value = (action, triggered_check)
        td.safety_monitor = mock_monitor
        td.get_runtime("test-telescope-123").acquisition_queue.is_idle.return_value = queue_idle
        # _evaluate_safety_for() runs the per-runtime gate — same code
        # path as the task_runner loop now uses.  The triggered-check
        # (MagicMock) has no string ``sensor_id`` so the dispatcher
        # routes this as a site-level check, preserving the existing
        # "wait for every runtime idle" semantics these tests check.
        return td._evaluate_safety_for("test-telescope-123")

    def test_unwind_fires_after_queue_drains(self, wired_dispatcher):
        td, _ = wired_dispatcher
        check = MagicMock()
        check.name = "cable_wrap"

        result = self._call(td, queue_idle=False, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert result is True
        check.execute_action.assert_not_called()

        result = self._call(td, queue_idle=True, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert result is True
        check.execute_action.assert_called_once()

    def test_unwind_retried_after_failure(self, wired_dispatcher):
        td, _ = wired_dispatcher
        check = MagicMock()
        check.name = "cable_wrap"
        check.execute_action.side_effect = [RuntimeError("stall"), None]

        self._call(td, queue_idle=True, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert check.execute_action.call_count == 1
        td.logger.error.assert_called()

        self._call(td, queue_idle=True, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert check.execute_action.call_count == 2

    def test_no_action_when_queue_busy(self, wired_dispatcher):
        td, _ = wired_dispatcher
        check = MagicMock()
        check.name = "cable_wrap"
        self._call(td, queue_idle=False, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        check.execute_action.assert_not_called()

    def test_queue_stop_yields_task_loop(self, wired_dispatcher):
        td, _ = wired_dispatcher
        result = self._call(td, queue_idle=False, action=SafetyAction.QUEUE_STOP)
        assert result is True

    def test_safe_does_not_yield(self, wired_dispatcher):
        td, _ = wired_dispatcher
        result = self._call(td, queue_idle=True, action=SafetyAction.SAFE)
        assert result is False


# ── Multi-runtime coverage ─────────────────────────────────────────────────


class TestMultiRuntime:
    def test_are_queues_idle_all_runtimes(self):
        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt2 = _mock_runtime("scope-1", "telescope")
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        rt1.are_queues_idle.return_value = True
        rt2.are_queues_idle.return_value = True
        assert td.are_queues_idle() is True

        rt2.are_queues_idle.return_value = False
        assert td.are_queues_idle() is False

    def test_clear_pending_tasks_clears_all_runtimes(self):
        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt2 = _mock_runtime("scope-1", "telescope")
        rt1.acquisition_queue.clear.return_value = 2
        rt2.acquisition_queue.clear.return_value = 3
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        total = td.clear_pending_tasks()
        assert total == 5
        rt1.acquisition_queue.clear.assert_called_once()
        rt2.acquisition_queue.clear.assert_called_once()

    def test_report_online_reports_all_telescopes(self):
        api_client = MagicMock()
        td = _make_dispatcher(api_client=api_client)
        rt1 = _mock_runtime("scope-0", "telescope")
        rt1.sensor.citra_record = {"id": "api-scope-0"}
        rt2 = _mock_runtime("scope-1", "telescope")
        rt2.sensor.citra_record = {"id": "api-scope-1"}
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        td._report_online()
        api_client.put_telescope_status.assert_called_once()
        statuses = api_client.put_telescope_status.call_args[0][0]
        ids = {s["id"] for s in statuses}
        assert ids == {"api-scope-0", "api-scope-1"}

    def test_automated_scheduling_any_telescope(self):
        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt1.sensor.citra_record = {"id": "a", "automatedScheduling": False}
        rt2 = _mock_runtime("scope-1", "telescope")
        rt2.sensor.citra_record = {"id": "b", "automatedScheduling": True}
        td.register_runtime(rt1)
        td.register_runtime(rt2)

        assert td.automated_scheduling is True

    def test_queue_stop_waits_for_all_runtimes(self):
        monitor = MagicMock()
        check = MagicMock()
        check.name = "cable_wrap"
        monitor.evaluate.return_value = (SafetyAction.QUEUE_STOP, check)

        td = _make_dispatcher()
        rt1 = _mock_runtime("scope-0", "telescope")
        rt2 = _mock_runtime("scope-1", "telescope")
        td.register_runtime(rt1)
        td.register_runtime(rt2)
        td.safety_monitor = monitor

        rt1.acquisition_queue.is_idle.return_value = True
        rt2.acquisition_queue.is_idle.return_value = False
        # Site-level triggers (no string sensor_id on the mock) still wait
        # for every runtime to be idle before the corrective action fires.
        td._evaluate_safety_for("scope-0")
        check.execute_action.assert_not_called()

        rt2.acquisition_queue.is_idle.return_value = True
        td._evaluate_safety_for("scope-0")
        check.execute_action.assert_called()


class TestPerSensorSafetyGate:
    """Regressions for cross-sensor safety bleed.

    A cable unwind (or any sensor-scoped QUEUE_STOP / EMERGENCY) on
    telescope A must only freeze telescope A — siblings keep scheduling.
    """

    @staticmethod
    def _scoped_monitor(actions_by_sensor: dict[str, tuple[SafetyAction, object]]):
        """Build a fake SafetyMonitor whose evaluate() branches by sensor_id."""
        monitor = MagicMock()

        def _evaluate(sensor_id: str | None = None):
            return actions_by_sensor.get(sensor_id, (SafetyAction.SAFE, None))

        monitor.evaluate.side_effect = _evaluate
        return monitor

    def test_sensor_scoped_queue_stop_does_not_block_siblings(self):
        """Sensor A unwinding → A yields, B is free to continue."""
        td = _make_dispatcher()
        rt_a = _mock_runtime("scope-a", "telescope")
        rt_b = _mock_runtime("scope-b", "telescope")
        td.register_runtime(rt_a)
        td.register_runtime(rt_b)

        check_a = MagicMock()
        check_a.name = "cable_wrap"
        check_a.sensor_id = "scope-a"

        td.safety_monitor = self._scoped_monitor(
            {
                "scope-a": (SafetyAction.QUEUE_STOP, check_a),
                "scope-b": (SafetyAction.SAFE, None),
            }
        )

        assert td._evaluate_safety_for("scope-a") is True
        assert td._evaluate_safety_for("scope-b") is False

    def test_sensor_scoped_queue_stop_waits_only_for_owning_runtime(self):
        """Sensor A's corrective action fires once A is idle, even if B is busy."""
        td = _make_dispatcher()
        rt_a = _mock_runtime("scope-a", "telescope")
        rt_b = _mock_runtime("scope-b", "telescope")
        td.register_runtime(rt_a)
        td.register_runtime(rt_b)

        check_a = MagicMock()
        check_a.name = "cable_wrap"
        check_a.sensor_id = "scope-a"

        td.safety_monitor = self._scoped_monitor(
            {
                "scope-a": (SafetyAction.QUEUE_STOP, check_a),
                "scope-b": (SafetyAction.SAFE, None),
            }
        )

        rt_a.acquisition_queue.is_idle.return_value = True
        rt_b.acquisition_queue.is_idle.return_value = False
        td._evaluate_safety_for("scope-a")
        check_a.execute_action.assert_called_once()

    def test_sensor_scoped_emergency_only_aborts_triggering_adapter(self):
        """EMERGENCY on A must not call abort_slew on B's adapter."""
        td = _make_dispatcher()
        rt_a = _mock_runtime("scope-a", "telescope")
        rt_b = _mock_runtime("scope-b", "telescope")
        rt_a.hardware_adapter = MagicMock()
        rt_b.hardware_adapter = MagicMock()
        td.register_runtime(rt_a)
        td.register_runtime(rt_b)

        check_a = MagicMock()
        check_a.name = "cable_wrap"
        check_a.sensor_id = "scope-a"

        td.safety_monitor = self._scoped_monitor({"scope-a": (SafetyAction.EMERGENCY, check_a)})

        td._evaluate_safety_for("scope-a")

        rt_a.hardware_adapter.abort_slew.assert_called_once()
        rt_b.hardware_adapter.abort_slew.assert_not_called()

    def test_clear_pending_tasks_scoped_only_drains_owning_runtime(self):
        """clear_pending_tasks(sensor_id=X) only touches X's queue."""
        td = _make_dispatcher()
        rt_a = _mock_runtime("scope-a", "telescope")
        rt_b = _mock_runtime("scope-b", "telescope")
        rt_a.acquisition_queue.clear.return_value = 2
        rt_b.acquisition_queue.clear.return_value = 3
        td.register_runtime(rt_a)
        td.register_runtime(rt_b)

        total = td.clear_pending_tasks(sensor_id="scope-a")

        assert total == 2
        rt_a.acquisition_queue.clear.assert_called_once()
        rt_b.acquisition_queue.clear.assert_not_called()
