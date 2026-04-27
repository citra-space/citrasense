"""Tests for e-stop clear_pending_tasks() and expired-task guard (#101)."""

import heapq
import threading
import time
from unittest.mock import MagicMock

from citrasense.acquisition.acquisition_queue import AcquisitionQueue
from citrasense.tasks.task_dispatcher import TaskDispatcher

SID = "test-scope"


def _make_task_dispatcher() -> TaskDispatcher:
    """Build a TaskDispatcher with a mock runtime, without starting worker threads."""

    settings = MagicMock()
    settings.task_processing_paused = False
    settings.max_task_retries = 3
    settings.initial_retry_delay_seconds = 1
    settings.max_retry_delay_seconds = 10

    td = TaskDispatcher(
        api_client=MagicMock(),
        logger=MagicMock(),
        settings=settings,
    )

    runtime = MagicMock()
    runtime.sensor_id = SID
    runtime.sensor_type = "telescope"
    runtime.acquisition_queue = MagicMock()
    runtime.acquisition_queue.is_idle.return_value = True
    runtime.processing_queue = MagicMock()
    runtime.upload_queue = MagicMock()
    runtime.are_queues_idle.return_value = True
    td.register_runtime(runtime)
    return td


def _make_task(name="TestSat"):
    """Create a mock Task with the fields get_tasks_by_stage expects."""
    task = MagicMock()
    task.satelliteName = name
    task.get_status_info.return_value = ("Queued for imaging...", None, False)
    return task


# ---------------------------------------------------------------------------
# clear_pending_tasks preserves heap and task_dict
# ---------------------------------------------------------------------------


class TestClearPendingTasks:
    def test_preserves_heap(self):
        td = _make_task_dispatcher()
        task = _make_task()
        heapq.heappush(td._sensor_heaps[SID], (1000, 2000, "t1", task))
        td._sensor_task_ids[SID].add("t1")
        td._sensor_task_dicts[SID]["t1"] = task

        td.clear_pending_tasks()

        assert len(td._sensor_heaps[SID]) == 1
        assert "t1" in td._sensor_task_ids[SID]
        assert "t1" in td._sensor_task_dicts[SID]

    def test_clears_imaging_tasks(self):
        td = _make_task_dispatcher()
        td.imaging_tasks["t1"] = time.time()
        td.imaging_tasks["t2"] = time.time()

        td.clear_pending_tasks()

        assert len(td.imaging_tasks) == 0

    def test_does_not_clear_processing_or_upload_stages(self):
        td = _make_task_dispatcher()
        td.processing_tasks["t1"] = time.time()
        td.uploading_tasks["t2"] = time.time()

        td.clear_pending_tasks()

        assert "t1" in td.processing_tasks
        assert "t2" in td.uploading_tasks

    def test_calls_imaging_queue_clear(self):
        td = _make_task_dispatcher()
        rt = td.get_runtime("test-scope")
        rt.acquisition_queue.clear = MagicMock(return_value=2)

        count = td.clear_pending_tasks()

        rt.acquisition_queue.clear.assert_called_once()
        assert count == 2

    def test_no_orphan_guids_after_clear(self):
        """After clear, get_tasks_by_stage should not return bare-GUID entries."""
        td = _make_task_dispatcher()
        task = _make_task("MySat")
        heapq.heappush(td._sensor_heaps[SID], (1000, 2000, "t1", task))
        td._sensor_task_ids[SID].add("t1")
        td._sensor_task_dicts[SID]["t1"] = task
        td.imaging_tasks["t1"] = time.time()

        td.clear_pending_tasks()

        stages = td.get_tasks_by_stage()
        assert stages["imaging"] == []


# ---------------------------------------------------------------------------
# Expired task guard in task_runner
# ---------------------------------------------------------------------------


class TestExpiredTaskGuard:
    def test_skips_expired_task(self):
        """Tasks whose stop_epoch < now should be discarded, not executed."""
        td = _make_task_dispatcher()
        heap = td._sensor_heaps[SID]
        ids = td._sensor_task_ids[SID]
        dicts = td._sensor_task_dicts[SID]
        now = int(time.time())
        past_stop = now - 100
        task = _make_task("Expired")
        heapq.heappush(heap, (now - 200, past_stop, "t-expired", task))
        ids.add("t-expired")
        dicts["t-expired"] = task

        valid_task = _make_task("Valid")
        heapq.heappush(heap, (now - 100, now + 3600, "t-valid", valid_task))
        ids.add("t-valid")
        dicts["t-valid"] = valid_task

        popped = []
        while heap and heap[0][0] <= now:
            _start_epoch, stop_epoch, tid, _task = heap[0]
            if stop_epoch and stop_epoch < now:
                heapq.heappop(heap)
                ids.discard(tid)
                dicts.pop(tid, None)
                continue
            heapq.heappop(heap)
            ids.discard(tid)
            popped.append(tid)

        assert "t-expired" not in popped
        assert "t-expired" not in ids
        assert "t-expired" not in dicts
        assert "t-valid" in popped

    def test_zero_stop_epoch_means_no_expiry(self):
        """Tasks with stop_epoch=0 (no end time) should always execute."""
        td = _make_task_dispatcher()
        heap = td._sensor_heaps[SID]
        ids = td._sensor_task_ids[SID]
        dicts = td._sensor_task_dicts[SID]
        now = int(time.time())
        task = _make_task("NoExpiry")
        heapq.heappush(heap, (now - 100, 0, "t1", task))
        ids.add("t1")
        dicts["t1"] = task

        popped = []
        while heap and heap[0][0] <= now:
            _start_epoch, stop_epoch, tid, _task = heap[0]
            if stop_epoch and stop_epoch < now:
                heapq.heappop(heap)
                ids.discard(tid)
                dicts.pop(tid, None)
                continue
            heapq.heappop(heap)
            ids.discard(tid)
            popped.append(tid)

        assert "t1" in popped

    def test_future_stop_epoch_not_skipped(self):
        """Tasks whose stop_epoch is in the future should execute normally."""
        td = _make_task_dispatcher()
        heap = td._sensor_heaps[SID]
        ids = td._sensor_task_ids[SID]
        dicts = td._sensor_task_dicts[SID]
        now = int(time.time())
        task = _make_task("Future")
        heapq.heappush(heap, (now - 50, now + 3600, "t1", task))
        ids.add("t1")
        dicts["t1"] = task

        popped = []
        while heap and heap[0][0] <= now:
            _start_epoch, stop_epoch, tid, _task = heap[0]
            if stop_epoch and stop_epoch < now:
                heapq.heappop(heap)
                ids.discard(tid)
                dicts.pop(tid, None)
                continue
            heapq.heappop(heap)
            ids.discard(tid)
            popped.append(tid)

        assert "t1" in popped


# ---------------------------------------------------------------------------
# AcquisitionQueue._on_cancelled does NOT mark the task as failed on the API
# ---------------------------------------------------------------------------


class TestImagingCancelledNotFailed:
    """E-stop cancelled imaging tasks should be retryable, not killed."""

    def _make_imaging_queue(self):
        settings = MagicMock()
        settings.max_task_retries = 3
        settings.initial_retry_delay_seconds = 1
        settings.max_retry_delay_seconds = 10
        logger = MagicMock()
        api_client = MagicMock()
        runtime = MagicMock()
        iq = AcquisitionQueue(
            num_workers=0,
            settings=settings,
            logger=logger,
            api_client=api_client,
            runtime=runtime,
        )
        return iq, api_client, runtime

    def test_on_cancelled_does_not_mark_failed(self):
        """_on_cancelled should NOT call mark_task_failed on the API."""
        iq, api_client, runtime = self._make_imaging_queue()
        item = {
            "task_id": "t1",
            "task": _make_task("CancelMe"),
            "on_complete": MagicMock(),
        }

        iq._on_cancelled(item)

        api_client.mark_task_failed.assert_not_called()
        runtime.remove_task_from_all_stages.assert_called_once_with("t1")
        item["on_complete"].assert_called_once_with("t1", success=False)

    def test_on_permanent_failure_still_marks_failed(self):
        """Genuine permanent failures should still hit the API."""
        iq, api_client, runtime = self._make_imaging_queue()
        item = {
            "task_id": "t2",
            "task": _make_task("RealFail"),
            "on_complete": MagicMock(),
        }

        iq._on_permanent_failure(item)

        api_client.mark_task_failed.assert_called_once_with("t2")
        runtime.remove_task_from_all_stages.assert_called_once_with("t2")

    def test_epoch_mismatch_routes_to_on_cancelled(self):
        """When clear() bumps the epoch during execution, the worker should
        call _on_cancelled (not _on_permanent_failure)."""
        iq, api_client, _runtime = self._make_imaging_queue()

        completed = threading.Event()
        on_complete = MagicMock(side_effect=lambda *a, **kw: completed.set())

        iq._execute_work = MagicMock(side_effect=lambda item: (True, None))  # type: ignore[assignment]

        item = {
            "task_id": "t3",
            "task": _make_task("EpochTest"),
            "telescope_task_instance": MagicMock(),
            "on_complete": on_complete,
        }

        iq.running = True
        worker = threading.Thread(target=iq._worker_loop, daemon=True)
        worker.start()

        iq._clear_epoch += 1
        iq.work_queue.put(item)

        completed.wait(timeout=5)
        iq.running = False
        iq.work_queue.put(None)
        worker.join(timeout=2)

        api_client.mark_task_failed.assert_not_called()
