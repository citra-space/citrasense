"""Tests for e-stop clear_pending_tasks() and expired-task guard (#101)."""

import heapq
import threading
import time
from unittest.mock import MagicMock

from citrascope.tasks.imaging_queue import ImagingQueue
from citrascope.tasks.runner import TaskManager


def _make_task_manager() -> TaskManager:
    """Build a TaskManager with minimal mocks, without starting worker threads."""

    settings = MagicMock()
    settings.task_processing_paused = False
    settings.max_task_retries = 3
    settings.initial_retry_delay_seconds = 1
    settings.max_retry_delay_seconds = 10

    tm = TaskManager(
        api_client=MagicMock(),
        logger=MagicMock(),
        hardware_adapter=MagicMock(),
        settings=settings,
        processor_registry=MagicMock(),
        telescope_record={"id": "test-scope"},
    )
    return tm


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
        tm = _make_task_manager()
        task = _make_task()
        heapq.heappush(tm.task_heap, (1000, 2000, "t1", task))
        tm.task_ids.add("t1")
        tm.task_dict["t1"] = task

        tm.clear_pending_tasks()

        assert len(tm.task_heap) == 1
        assert "t1" in tm.task_ids
        assert "t1" in tm.task_dict

    def test_clears_imaging_tasks(self):
        tm = _make_task_manager()
        tm.imaging_tasks["t1"] = time.time()
        tm.imaging_tasks["t2"] = time.time()

        tm.clear_pending_tasks()

        assert len(tm.imaging_tasks) == 0

    def test_does_not_clear_processing_or_upload_stages(self):
        tm = _make_task_manager()
        tm.processing_tasks["t1"] = time.time()
        tm.uploading_tasks["t2"] = time.time()

        tm.clear_pending_tasks()

        assert "t1" in tm.processing_tasks
        assert "t2" in tm.uploading_tasks

    def test_calls_imaging_queue_clear(self):
        tm = _make_task_manager()
        tm.imaging_queue.clear = MagicMock(return_value=2)

        count = tm.clear_pending_tasks()

        tm.imaging_queue.clear.assert_called_once()
        assert count == 2

    def test_no_orphan_guids_after_clear(self):
        """After clear, get_tasks_by_stage should not return bare-GUID entries."""
        tm = _make_task_manager()
        task = _make_task("MySat")
        heapq.heappush(tm.task_heap, (1000, 2000, "t1", task))
        tm.task_ids.add("t1")
        tm.task_dict["t1"] = task
        tm.imaging_tasks["t1"] = time.time()

        tm.clear_pending_tasks()

        stages = tm.get_tasks_by_stage()
        assert stages["imaging"] == []


# ---------------------------------------------------------------------------
# Expired task guard in task_runner
# ---------------------------------------------------------------------------


class TestExpiredTaskGuard:
    def test_skips_expired_task(self):
        """Tasks whose stop_epoch < now should be discarded, not executed."""
        tm = _make_task_manager()
        now = int(time.time())
        past_stop = now - 100
        task = _make_task("Expired")
        heapq.heappush(tm.task_heap, (now - 200, past_stop, "t-expired", task))
        tm.task_ids.add("t-expired")
        tm.task_dict["t-expired"] = task

        # Also add a valid task after the expired one
        valid_task = _make_task("Valid")
        heapq.heappush(tm.task_heap, (now - 100, now + 3600, "t-valid", valid_task))
        tm.task_ids.add("t-valid")
        tm.task_dict["t-valid"] = valid_task

        # Run one iteration of the task runner's pop logic manually
        popped = []
        while tm.task_heap and tm.task_heap[0][0] <= now:
            _start_epoch, stop_epoch, tid, _task = tm.task_heap[0]
            if stop_epoch and stop_epoch < now:
                heapq.heappop(tm.task_heap)
                tm.task_ids.discard(tid)
                tm.task_dict.pop(tid, None)
                continue
            heapq.heappop(tm.task_heap)
            tm.task_ids.discard(tid)
            popped.append(tid)

        assert "t-expired" not in popped
        assert "t-expired" not in tm.task_ids
        assert "t-expired" not in tm.task_dict
        assert "t-valid" in popped

    def test_zero_stop_epoch_means_no_expiry(self):
        """Tasks with stop_epoch=0 (no end time) should always execute."""
        tm = _make_task_manager()
        now = int(time.time())
        task = _make_task("NoExpiry")
        heapq.heappush(tm.task_heap, (now - 100, 0, "t1", task))
        tm.task_ids.add("t1")
        tm.task_dict["t1"] = task

        popped = []
        while tm.task_heap and tm.task_heap[0][0] <= now:
            _start_epoch, stop_epoch, tid, _task = tm.task_heap[0]
            if stop_epoch and stop_epoch < now:
                heapq.heappop(tm.task_heap)
                tm.task_ids.discard(tid)
                tm.task_dict.pop(tid, None)
                continue
            heapq.heappop(tm.task_heap)
            tm.task_ids.discard(tid)
            popped.append(tid)

        assert "t1" in popped

    def test_future_stop_epoch_not_skipped(self):
        """Tasks whose stop_epoch is in the future should execute normally."""
        tm = _make_task_manager()
        now = int(time.time())
        task = _make_task("Future")
        heapq.heappush(tm.task_heap, (now - 50, now + 3600, "t1", task))
        tm.task_ids.add("t1")
        tm.task_dict["t1"] = task

        popped = []
        while tm.task_heap and tm.task_heap[0][0] <= now:
            _start_epoch, stop_epoch, tid, _task = tm.task_heap[0]
            if stop_epoch and stop_epoch < now:
                heapq.heappop(tm.task_heap)
                tm.task_ids.discard(tid)
                tm.task_dict.pop(tid, None)
                continue
            heapq.heappop(tm.task_heap)
            tm.task_ids.discard(tid)
            popped.append(tid)

        assert "t1" in popped


# ---------------------------------------------------------------------------
# ImagingQueue._on_cancelled does NOT mark the task as failed on the API
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
        task_manager = MagicMock()
        iq = ImagingQueue(
            num_workers=0,
            settings=settings,
            logger=logger,
            api_client=api_client,
            task_manager=task_manager,
        )
        return iq, api_client, task_manager

    def test_on_cancelled_does_not_mark_failed(self):
        """_on_cancelled should NOT call mark_task_failed on the API."""
        iq, api_client, task_manager = self._make_imaging_queue()
        item = {
            "task_id": "t1",
            "task": _make_task("CancelMe"),
            "on_complete": MagicMock(),
        }

        iq._on_cancelled(item)

        api_client.mark_task_failed.assert_not_called()
        task_manager.remove_task_from_all_stages.assert_called_once_with("t1")
        item["on_complete"].assert_called_once_with("t1", success=False)

    def test_on_permanent_failure_still_marks_failed(self):
        """Genuine permanent failures should still hit the API."""
        iq, api_client, task_manager = self._make_imaging_queue()
        item = {
            "task_id": "t2",
            "task": _make_task("RealFail"),
            "on_complete": MagicMock(),
        }

        iq._on_permanent_failure(item)

        api_client.mark_task_failed.assert_called_once_with("t2")
        task_manager.remove_task_from_all_stages.assert_called_once_with("t2")

    def test_epoch_mismatch_routes_to_on_cancelled(self):
        """When clear() bumps the epoch during execution, the worker should
        call _on_cancelled (not _on_permanent_failure)."""
        iq, api_client, _task_manager = self._make_imaging_queue()

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
