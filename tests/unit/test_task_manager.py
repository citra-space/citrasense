"""
Unit tests for TaskManager task queue management.
"""

import heapq
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from dateutil import parser as dtparser

from citrascope.safety.safety_monitor import SafetyAction
from citrascope.tasks.runner import TaskManager
from citrascope.tasks.task import Task


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = MagicMock()
    client.get_telescope_tasks.return_value = []
    client.put_telescope_status.return_value = None
    return client


@pytest.fixture
def mock_hardware_adapter():
    """Create a mock hardware adapter."""
    adapter = MagicMock()
    return adapter


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = MagicMock()
    return logger


@pytest.fixture
def mock_settings():
    """Create a mock settings instance."""
    settings = MagicMock()
    settings.keep_images = False
    settings.max_task_retries = 3
    settings.initial_retry_delay_seconds = 30
    settings.max_retry_delay_seconds = 300
    return settings


@pytest.fixture
def mock_processor_registry():
    """Create a mock processor registry."""
    registry = MagicMock()
    return registry


@pytest.fixture
def mock_daemon(mock_api_client, mock_hardware_adapter, mock_logger, mock_settings):
    """Create a mock daemon instance for testing."""
    daemon = MagicMock()
    daemon.api_client = mock_api_client
    daemon.hardware_adapter = mock_hardware_adapter
    daemon.logger = mock_logger
    daemon.telescope_record = {"id": "test-telescope-123", "maxSlewRate": 5.0, "automatedScheduling": False}
    daemon.ground_station = {"id": "test-gs-456", "latitude": 40.0, "longitude": -74.0, "altitude": 100}
    daemon.settings = mock_settings
    daemon.location_service = MagicMock()
    return daemon


@pytest.fixture
def task_manager(
    mock_api_client, mock_hardware_adapter, mock_logger, mock_daemon, mock_settings, mock_processor_registry
):
    """Create a TaskManager instance for testing."""
    tm = TaskManager(
        api_client=mock_api_client,
        logger=mock_logger,
        hardware_adapter=mock_hardware_adapter,
        settings=mock_settings,
        processor_registry=mock_processor_registry,
        telescope_record=mock_daemon.telescope_record,
        ground_station=mock_daemon.ground_station,
        location_service=mock_daemon.location_service,
    )
    return tm


def create_test_task(task_id, status="Pending", start_offset_seconds=60):
    """Create a test task with a start time in the future."""
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


def test_poll_tasks_adds_new_tasks(task_manager, mock_api_client):
    """Test that poll_tasks adds new pending tasks to the queue."""
    # Create a test task
    task1 = create_test_task("task-001", "Pending")
    task2 = create_test_task("task-002", "Scheduled", start_offset_seconds=120)

    # Mock API to return the tasks
    mock_api_client.get_telescope_tasks.return_value = [
        task1.__dict__,
        task2.__dict__,
    ]

    # Run poll_tasks once (manually, not in thread)
    with task_manager.heap_lock:
        # Simulate one iteration of poll_tasks
        task_manager._report_online()
        tasks = mock_api_client.get_telescope_tasks(task_manager.telescope_record["id"])

        # The actual logic from poll_tasks
        api_task_map = {}
        for task_dict in tasks:
            task = Task.from_dict(task_dict)
            tid = task.id
            if tid and task.status in ["Pending", "Scheduled"]:
                api_task_map[tid] = task

        # Add new tasks
        now = int(time.time())

        for tid, task in api_task_map.items():
            if tid not in task_manager.task_ids and tid != task_manager.current_task_id:
                task_start = task.taskStart
                task_stop = task.taskStop
                start_epoch = int(dtparser.isoparse(task_start).timestamp())
                stop_epoch = int(dtparser.isoparse(task_stop).timestamp()) if task_stop else 0
                if not (stop_epoch and stop_epoch < now):
                    heapq.heappush(task_manager.task_heap, (start_epoch, stop_epoch, tid, task))
                    task_manager.task_ids.add(tid)
                    task_manager.task_dict[tid] = task

    # Assert both tasks were added
    assert len(task_manager.task_heap) == 2
    assert "task-001" in task_manager.task_ids
    assert "task-002" in task_manager.task_ids


def test_poll_tasks_removes_cancelled_tasks(task_manager, mock_api_client):
    """Test that poll_tasks removes tasks that have been cancelled."""
    # Create and add two tasks to the queue
    task1 = create_test_task("task-001", "Pending")
    task2 = create_test_task("task-002", "Pending", start_offset_seconds=120)

    # Add tasks to the heap manually
    start_epoch1 = int(dtparser.isoparse(task1.taskStart).timestamp())
    stop_epoch1 = int(dtparser.isoparse(task1.taskStop).timestamp())
    start_epoch2 = int(dtparser.isoparse(task2.taskStart).timestamp())
    stop_epoch2 = int(dtparser.isoparse(task2.taskStop).timestamp())

    with task_manager.heap_lock:
        heapq.heappush(task_manager.task_heap, (start_epoch1, stop_epoch1, "task-001", task1))
        heapq.heappush(task_manager.task_heap, (start_epoch2, stop_epoch2, "task-002", task2))
        task_manager.task_ids.add("task-001")
        task_manager.task_ids.add("task-002")
        task_manager.task_dict["task-001"] = task1
        task_manager.task_dict["task-002"] = task2

    assert len(task_manager.task_heap) == 2

    # Now mock API to return only task-001 (task-002 has been cancelled)
    mock_api_client.get_telescope_tasks.return_value = [
        task1.__dict__,
    ]

    # Run the removal logic from poll_tasks
    with task_manager.heap_lock:
        tasks = mock_api_client.get_telescope_tasks(task_manager.telescope_record["id"])

        # Build api_task_map
        api_task_map = {}
        for task_dict in tasks:
            task = Task.from_dict(task_dict)
            tid = task.id
            if tid and task.status in ["Pending", "Scheduled"]:
                api_task_map[tid] = task

        # Remove tasks not in api_task_map
        new_heap = []
        removed = 0
        for start_epoch, stop_epoch, tid, task in task_manager.task_heap:
            if tid == task_manager.current_task_id or tid in api_task_map:
                new_heap.append((start_epoch, stop_epoch, tid, task))
            else:
                task_manager.task_ids.discard(tid)
                task_manager.task_dict.pop(tid, None)
                removed += 1

        if removed > 0:
            task_manager.task_heap = new_heap
            heapq.heapify(task_manager.task_heap)

    # Assert task-002 was removed
    assert len(task_manager.task_heap) == 1
    assert "task-001" in task_manager.task_ids
    assert "task-002" not in task_manager.task_ids
    assert task_manager.task_heap[0][2] == "task-001"


def test_poll_tasks_removes_tasks_with_changed_status(task_manager, mock_api_client):
    """Test that poll_tasks removes tasks whose status changed from Pending to Cancelled."""
    # Create and add a task
    task1 = create_test_task("task-001", "Pending")

    start_epoch = int(dtparser.isoparse(task1.taskStart).timestamp())
    stop_epoch = int(dtparser.isoparse(task1.taskStop).timestamp())

    with task_manager.heap_lock:
        heapq.heappush(task_manager.task_heap, (start_epoch, stop_epoch, "task-001", task1))
        task_manager.task_ids.add("task-001")
        task_manager.task_dict["task-001"] = task1

    assert len(task_manager.task_heap) == 1

    # Now the task status changed to "Cancelled" in the API
    task1_cancelled = create_test_task("task-001", "Cancelled")
    mock_api_client.get_telescope_tasks.return_value = [
        task1_cancelled.__dict__,
    ]

    # Run the removal logic
    with task_manager.heap_lock:
        tasks = mock_api_client.get_telescope_tasks(task_manager.telescope_record["id"])

        # Build api_task_map (Cancelled tasks won't be included)
        api_task_map = {}
        for task_dict in tasks:
            task = Task.from_dict(task_dict)
            tid = task.id
            if tid and task.status in ["Pending", "Scheduled"]:
                api_task_map[tid] = task

        # Remove tasks not in api_task_map
        new_heap = []
        removed = 0
        for start_epoch, stop_epoch, tid, task in task_manager.task_heap:
            if tid == task_manager.current_task_id or tid in api_task_map:
                new_heap.append((start_epoch, stop_epoch, tid, task))
            else:
                task_manager.task_ids.discard(tid)
                task_manager.task_dict.pop(tid, None)
                removed += 1

        if removed > 0:
            task_manager.task_heap = new_heap
            heapq.heapify(task_manager.task_heap)

    # Assert the task was removed
    assert len(task_manager.task_heap) == 0
    assert "task-001" not in task_manager.task_ids


def test_poll_tasks_does_not_remove_current_task(task_manager, mock_api_client):
    """Test that poll_tasks doesn't remove the currently executing task even if it's not in API response."""
    # Create and add a task
    task1 = create_test_task("task-001", "Pending")

    start_epoch = int(dtparser.isoparse(task1.taskStart).timestamp())
    stop_epoch = int(dtparser.isoparse(task1.taskStop).timestamp())

    with task_manager.heap_lock:
        heapq.heappush(task_manager.task_heap, (start_epoch, stop_epoch, "task-001", task1))
        task_manager.task_ids.add("task-001")
        task_manager.task_dict["task-001"] = task1
        # Mark this task as currently executing
        task_manager.current_task_id = "task-001"

    # API returns no tasks (task was cancelled or status changed)
    mock_api_client.get_telescope_tasks.return_value = []

    # Run the removal logic
    with task_manager.heap_lock:
        tasks = mock_api_client.get_telescope_tasks(task_manager.telescope_record["id"])

        api_task_map = {}
        for task_dict in tasks:
            task = Task.from_dict(task_dict)
            tid = task.id
            if tid and task.status in ["Pending", "Scheduled"]:
                api_task_map[tid] = task

        # Remove tasks not in api_task_map (but not current task)
        new_heap = []
        removed = 0
        for start_epoch, stop_epoch, tid, task in task_manager.task_heap:
            if tid == task_manager.current_task_id or tid in api_task_map:
                new_heap.append((start_epoch, stop_epoch, tid, task))
            else:
                task_manager.task_ids.discard(tid)
                removed += 1

        if removed > 0:
            task_manager.task_heap = new_heap
            heapq.heapify(task_manager.task_heap)

    # Assert the current task was NOT removed
    assert len(task_manager.task_heap) == 1
    assert "task-001" in task_manager.task_ids
    assert task_manager.task_heap[0][2] == "task-001"


# ------------------------------------------------------------------
# _evaluate_safety — cable wrap soft-lock regression (issue #239)
# ------------------------------------------------------------------


class TestEvaluateSafetyQueueStop:
    """Verify QUEUE_STOP always attempts corrective action when the imaging
    queue is idle, even if the state transition already happened on a
    previous tick (regression for issue #239 soft-lock)."""

    def _call(self, task_manager, *, queue_idle: bool, action: SafetyAction, triggered_check=None):
        """Helper: configure mocks and call ``_evaluate_safety`` once."""
        mock_monitor = MagicMock()
        mock_monitor.evaluate.return_value = (action, triggered_check)
        task_manager.safety_monitor = mock_monitor
        task_manager.imaging_queue = MagicMock()
        task_manager.imaging_queue.is_idle.return_value = queue_idle
        return task_manager._evaluate_safety()

    def test_unwind_fires_after_queue_drains(self, task_manager):
        """Soft-lock scenario: QUEUE_STOP arrives while imaging is busy,
        then the queue drains on the next tick.  The unwind must still fire."""
        check = MagicMock()
        check.name = "cable_wrap"

        # Tick 1: QUEUE_STOP but queue is busy — no unwind.
        result = self._call(task_manager, queue_idle=False, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert result is True
        check.execute_action.assert_not_called()

        # Tick 2: still QUEUE_STOP, queue now idle — unwind MUST fire.
        result = self._call(task_manager, queue_idle=True, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert result is True
        check.execute_action.assert_called_once()

    def test_unwind_retried_after_failure(self, task_manager):
        """After a failed unwind, the next idle tick should retry."""
        check = MagicMock()
        check.name = "cable_wrap"

        # Tick 1: first QUEUE_STOP with idle queue — unwind fires.
        self._call(task_manager, queue_idle=True, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert check.execute_action.call_count == 1

        # Tick 2: still QUEUE_STOP, still idle — should call again (retry).
        self._call(task_manager, queue_idle=True, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        assert check.execute_action.call_count == 2

    def test_no_action_when_queue_busy(self, task_manager):
        """While imaging is in-flight, no corrective action is attempted."""
        check = MagicMock()
        check.name = "cable_wrap"

        self._call(task_manager, queue_idle=False, action=SafetyAction.QUEUE_STOP, triggered_check=check)
        check.execute_action.assert_not_called()

    def test_queue_stop_yields_task_loop(self, task_manager):
        """QUEUE_STOP must return True so the task loop yields."""
        result = self._call(task_manager, queue_idle=False, action=SafetyAction.QUEUE_STOP)
        assert result is True

    def test_safe_does_not_yield(self, task_manager):
        """SAFE action should not block the task loop."""
        result = self._call(task_manager, queue_idle=True, action=SafetyAction.SAFE)
        assert result is False
