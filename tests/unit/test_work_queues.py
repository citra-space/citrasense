"""Unit tests for BaseWorkQueue, ImagingQueue, and ProcessingQueue."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from citrascope.tasks.base_work_queue import BaseWorkQueue

# ---------------------------------------------------------------------------
# Concrete subclass for testing BaseWorkQueue
# ---------------------------------------------------------------------------


class StubQueue(BaseWorkQueue):
    """Minimal concrete subclass for testing."""

    def __init__(self, execute_fn=None, **kwargs):
        settings = MagicMock()
        settings.max_task_retries = kwargs.get("max_retries", 3)
        settings.initial_retry_delay_seconds = kwargs.get("initial_delay", 1)
        settings.max_retry_delay_seconds = kwargs.get("max_delay", 10)
        super().__init__(num_workers=1, settings=settings, logger=MagicMock())
        self._execute_fn = execute_fn or (lambda item: (True, "ok"))
        self.success_items = []
        self.failure_items = []

    def _execute_work(self, item):
        return self._execute_fn(item)

    def _on_success(self, item, result):
        self.success_items.append(item["task_id"])

    def _on_permanent_failure(self, item):
        self.failure_items.append(item["task_id"])

    def _get_task_from_item(self, item):
        return item.get("task")


# ---------------------------------------------------------------------------
# BaseWorkQueue
# ---------------------------------------------------------------------------


def test_is_idle_initially():
    q = StubQueue()
    assert q.is_idle() is True


def test_get_stats_initial():
    q = StubQueue()
    stats = q.get_stats()
    assert stats["attempts"] == 0
    assert stats["successes"] == 0
    assert stats["permanent_failures"] == 0


def test_start_stop():
    q = StubQueue()
    q.start()
    assert q.running is True
    assert len(q.workers) == 1
    q.stop()
    assert q.running is False


def test_successful_work_item():
    q = StubQueue(execute_fn=lambda item: (True, "done"))
    q.start()
    q.work_queue.put({"task_id": "t1", "task": MagicMock()})
    time.sleep(0.5)
    q.stop()
    assert "t1" in q.success_items
    assert q.total_successes == 1
    assert q.total_attempts == 1


def test_failed_then_permanent_failure():
    q = StubQueue(execute_fn=lambda item: (False, None), max_retries=0)
    q.start()
    q.work_queue.put({"task_id": "t2", "task": MagicMock()})
    time.sleep(0.5)
    q.stop()
    assert "t2" in q.failure_items
    assert q.total_permanent_failures == 1


def test_exception_in_execute_triggers_retry():
    call_count = 0

    def flaky(item):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")
        return (True, "recovered")

    q = StubQueue(execute_fn=flaky, max_retries=2, initial_delay=0.1, max_delay=0.2)
    q.start()
    q.work_queue.put({"task_id": "t3", "task": MagicMock()})
    time.sleep(1.5)
    q.stop()
    assert "t3" in q.success_items


def test_calculate_backoff():
    q = StubQueue(initial_delay=10, max_delay=100)
    q.retry_counts["t1"] = 0
    assert q._calculate_backoff("t1") == 10
    q.retry_counts["t1"] = 3
    assert q._calculate_backoff("t1") == 80
    q.retry_counts["t1"] = 10
    assert q._calculate_backoff("t1") == 100


def test_should_retry():
    q = StubQueue(max_retries=3)
    assert q._should_retry("t1") is True
    q.retry_counts["t1"] = 3
    assert q._should_retry("t1") is False


def test_poison_pill_stops_worker():
    q = StubQueue()
    q.start()
    q.work_queue.put(None)
    time.sleep(0.3)
    q.stop()


def test_clear_cancels_pending_retry_timers():
    q = StubQueue(execute_fn=lambda item: (False, None), max_retries=3, initial_delay=60, max_delay=120)
    q.start()
    q.work_queue.put({"task_id": "t1", "task": MagicMock()})
    time.sleep(0.5)

    assert len(q._pending_timers) == 1
    timer = q._pending_timers[0]

    q.clear()
    # cancel() prevents the callback; timer thread may take a moment to exit
    assert timer.finished.is_set()
    assert len(q._pending_timers) == 0
    q.stop()


def test_clear_maxes_retry_counts():
    q = StubQueue(max_retries=5)
    q.retry_counts["t1"] = 1
    q.retry_counts["t2"] = 0
    q.clear()
    assert q.retry_counts["t1"] == 5
    assert q.retry_counts["t2"] == 5


def test_clear_epoch_discards_in_flight_result():
    """After clear(), an in-flight task's result is discarded via epoch check."""
    barrier = threading.Event()

    def slow_fail(item):
        barrier.wait(5)
        return (False, None)

    q = StubQueue(execute_fn=slow_fail, max_retries=3, initial_delay=0.1, max_delay=0.2)
    q.start()
    q.work_queue.put({"task_id": "t1", "task": MagicMock()})
    time.sleep(0.1)

    q.clear()
    barrier.set()
    time.sleep(0.5)
    q.stop()

    assert "t1" in q.failure_items
    assert len(q.success_items) == 0


# ---------------------------------------------------------------------------
# ImagingQueue
# ---------------------------------------------------------------------------


def test_imaging_queue_clear_cancels_in_flight_task():
    from citrascope.tasks.imaging_queue import ImagingQueue

    cancel_event = threading.Event()

    class FakeTelescopeTask:
        def execute(self):
            cancel_event.wait(5)
            return True

        def cancel(self):
            cancel_event.set()

    iq = ImagingQueue(
        num_workers=1,
        settings=MagicMock(max_task_retries=3, initial_retry_delay_seconds=1, max_retry_delay_seconds=10),
        logger=MagicMock(),
        api_client=MagicMock(),
        task_manager=MagicMock(),
    )
    iq.start()
    iq.submit("t1", MagicMock(), FakeTelescopeTask(), MagicMock())
    time.sleep(0.2)

    assert iq._current_item is not None
    iq.clear()
    assert cancel_event.is_set()
    time.sleep(0.3)
    iq.stop()


def test_imaging_queue_submit():
    from citrascope.tasks.imaging_queue import ImagingQueue

    iq = ImagingQueue(
        num_workers=1,
        settings=MagicMock(max_task_retries=3, initial_retry_delay_seconds=1, max_retry_delay_seconds=10),
        logger=MagicMock(),
        api_client=MagicMock(),
        task_manager=MagicMock(),
    )
    task = MagicMock()
    tele_task = MagicMock()
    cb = MagicMock()
    iq.submit("t1", task, tele_task, cb)
    assert not iq.is_idle()


def test_imaging_queue_success():
    from citrascope.tasks.imaging_queue import ImagingQueue

    mock_tm = MagicMock()
    iq = ImagingQueue(
        num_workers=1,
        settings=MagicMock(max_task_retries=3, initial_retry_delay_seconds=1, max_retry_delay_seconds=10),
        logger=MagicMock(),
        api_client=MagicMock(),
        task_manager=mock_tm,
    )
    task = MagicMock()
    tele_task = MagicMock()
    tele_task.execute.return_value = True
    cb = MagicMock()

    iq.start()
    iq.submit("t1", task, tele_task, cb)
    time.sleep(0.5)
    iq.stop()

    cb.assert_called_once_with("t1", success=True)


def test_imaging_queue_get_task_from_item():
    from citrascope.tasks.imaging_queue import ImagingQueue

    iq = ImagingQueue(
        num_workers=1,
        settings=MagicMock(),
        logger=MagicMock(),
        api_client=MagicMock(),
        task_manager=MagicMock(),
    )
    task = MagicMock()
    assert iq._get_task_from_item({"task": task}) is task


# ---------------------------------------------------------------------------
# ProcessingQueue
# ---------------------------------------------------------------------------


def test_processing_queue_get_working_dir_with_settings():
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(
        num_workers=1,
        settings=MagicMock(max_task_retries=3, initial_retry_delay_seconds=1, max_retry_delay_seconds=10),
        logger=MagicMock(),
    )
    mock_settings = MagicMock()
    mock_settings.get_images_dir.return_value = Path("/data/images")
    wd = pq._get_working_dir("task-123", mock_settings)
    assert wd == Path("/data/processing/task-123")


def test_processing_queue_get_working_dir_no_settings():
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    wd = pq._get_working_dir("task-123", None)
    assert "task-123" in str(wd)


def test_processing_queue_cleanup(tmp_path):
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    mock_settings = MagicMock()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    work_dir = tmp_path / "processing" / "task-1"
    work_dir.mkdir(parents=True)
    (work_dir / "file.txt").write_text("data")
    mock_settings.get_images_dir.return_value = images_dir
    pq._cleanup_working_dir("task-1", mock_settings)
    assert not work_dir.exists()


def test_processing_queue_get_task_from_item():
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    task = MagicMock()
    assert pq._get_task_from_item({"context": {"task": task}}) is task


def test_processing_queue_submit():
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    pq.submit("t1", Path("/img.fits"), {"task": MagicMock()}, MagicMock())
    assert not pq.is_idle()


def test_processing_queue_execute_success(tmp_path):
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(
        num_workers=1,
        settings=MagicMock(max_task_retries=3, initial_retry_delay_seconds=1, max_retry_delay_seconds=10),
        logger=MagicMock(),
    )
    mock_daemon = MagicMock()
    mock_result = MagicMock()
    mock_result.total_time = 1.5
    mock_daemon.processor_registry.process_all.return_value = mock_result

    mock_settings = MagicMock()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    mock_settings.get_images_dir.return_value = images_dir

    item = {
        "task_id": "t1",
        "image_path": str(tmp_path / "img.fits"),
        "context": {
            "task": MagicMock(),
            "settings": mock_settings,
            "daemon": mock_daemon,
            "telescope_record": {},
            "ground_station_record": {},
        },
        "on_complete": MagicMock(),
    }
    success, result = pq._execute_work(item)
    assert success is True
    assert result is mock_result


def test_processing_queue_execute_exception(tmp_path):
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    mock_daemon = MagicMock()
    mock_daemon.processor_registry.process_all.side_effect = Exception("boom")

    mock_settings = MagicMock()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    mock_settings.get_images_dir.return_value = images_dir

    item = {
        "task_id": "t1",
        "image_path": str(tmp_path / "img.fits"),
        "context": {"task": MagicMock(), "settings": mock_settings, "daemon": mock_daemon},
        "on_complete": MagicMock(),
    }
    success, _result = pq._execute_work(item)
    assert success is False


def test_processing_queue_on_success():
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    on_complete = MagicMock()
    task_obj = MagicMock()
    item = {
        "task_id": "t1",
        "context": {"task": task_obj, "settings": MagicMock()},
        "on_complete": on_complete,
    }
    with patch.object(pq, "_cleanup_working_dir"):
        pq._on_success(item, "result")
    task_obj.set_status_msg.assert_called_with("Processing complete")
    on_complete.assert_called_once_with("t1", "result")


def test_processing_queue_on_permanent_failure():
    from citrascope.tasks.processing_queue import ProcessingQueue

    pq = ProcessingQueue(num_workers=1, settings=MagicMock(), logger=MagicMock())
    on_complete = MagicMock()
    task_obj = MagicMock()
    item = {
        "task_id": "t1",
        "context": {"task": task_obj, "settings": MagicMock()},
        "on_complete": on_complete,
    }
    with patch.object(pq, "_cleanup_working_dir"):
        pq._on_permanent_failure(item)
    on_complete.assert_called_once_with("t1", None)
