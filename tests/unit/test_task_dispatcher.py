"""Tests for TaskDispatcher: routing, facade, runtime registration."""

from __future__ import annotations

import heapq
from unittest.mock import MagicMock

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

    def test_default_runtime_returns_first(self):
        td = _make_dispatcher()
        rt = _mock_runtime()
        td.register_runtime(rt)

        assert td._default_runtime is rt

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

    def test_falls_back_to_default(self):
        td = _make_dispatcher()
        rt = _mock_runtime("scope-0", "telescope")
        td.register_runtime(rt)

        task = _mock_task(sensor_type="unknown", sensor_id=None)
        assert td._runtime_for_task(task) is rt

    def test_rejects_task_with_no_runtimes(self):
        td = _make_dispatcher()
        task = _mock_task()
        assert td._runtime_for_task(task) is None


# ── Web facade ────────────────────────────────────────────────────────────


class TestWebFacade:
    def _wired_dispatcher(self):
        td = _make_dispatcher()
        rt = _mock_runtime()
        td.register_runtime(rt)
        return td, rt

    def test_imaging_queue_delegates(self):
        td, rt = self._wired_dispatcher()
        assert td.imaging_queue is rt.acquisition_queue

    def test_processing_queue_delegates(self):
        td, rt = self._wired_dispatcher()
        assert td.processing_queue is rt.processing_queue

    def test_upload_queue_delegates(self):
        td, rt = self._wired_dispatcher()
        assert td.upload_queue is rt.upload_queue

    def test_autofocus_manager_delegates(self):
        td, rt = self._wired_dispatcher()
        assert td.autofocus_manager is rt.autofocus_manager

    def test_alignment_manager_delegates(self):
        td, rt = self._wired_dispatcher()
        assert td.alignment_manager is rt.alignment_manager

    def test_homing_manager_delegates(self):
        td, rt = self._wired_dispatcher()
        assert td.homing_manager is rt.homing_manager

    def test_calibration_manager_setter(self):
        td, rt = self._wired_dispatcher()
        mock_cal = MagicMock()
        td.calibration_manager = mock_cal
        assert rt.calibration_manager is mock_cal

    def test_are_queues_idle_delegates(self):
        td, rt = self._wired_dispatcher()
        rt.are_queues_idle.return_value = False
        assert td.are_queues_idle() is False


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
        td.update_task_stage("t1", "uploading")
        td.task_dict["t1"] = MagicMock()
        td.remove_task_from_all_stages("t1")
        assert "t1" not in td.uploading_tasks
        assert "t1" not in td.task_dict


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
        task = MagicMock()
        heapq.heappush(td.task_heap, (1000, 2000, "t1", task))
        td.task_ids.add("t1")
        td.task_dict["t1"] = task

        assert td.drop_scheduled_task("t1") is True
        assert "t1" not in td.task_ids
        assert "t1" not in td.task_dict
        assert all(entry[2] != "t1" for entry in td.task_heap)

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
