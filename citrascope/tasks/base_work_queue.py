"""Base class for background work queues with retry logic."""

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any


class BaseWorkQueue(ABC):
    """Base class for background work queues with worker threads, retry logic, and exponential backoff."""

    def __init__(self, num_workers: int, settings, logger):
        """
        Initialize work queue.

        Args:
            num_workers: Number of concurrent worker threads
            settings: Settings instance with retry configuration
            logger: Logger instance
        """
        self.work_queue = queue.Queue()
        self.num_workers = num_workers
        self.settings = settings
        self.logger = logger
        self.workers = []
        self.running = False

        # Retry tracking (per-stage)
        self.retry_counts: dict[str, int] = {}
        self.last_failure: dict[str, float] = {}
        self._pending_timers: list[threading.Timer] = []
        self._timer_lock = threading.Lock()
        self._clear_epoch: int = 0

        # In-flight work item tracking (used by clear() to cancel active work)
        self._current_item_lock = threading.Lock()
        self._current_item: dict[str, Any] | None = None

        # Lifetime counters — increments are GIL-safe for simple int in CPython;
        # the lock exists to give atomic multi-field snapshots in get_stats().
        self._stats_lock = threading.Lock()
        self.total_attempts: int = 0
        self.total_successes: int = 0
        self.total_permanent_failures: int = 0

    @abstractmethod
    def _execute_work(self, item: dict[str, Any]) -> tuple[bool, Any]:
        """
        Execute stage-specific work. Must be implemented by subclasses.

        Args:
            item: Work item dictionary containing task_id and stage-specific data

        Returns:
            Tuple of (success: bool, result: Any)
        """
        pass

    @abstractmethod
    def _on_success(self, item: dict[str, Any], result: Any):
        """
        Handle successful work completion.

        Args:
            item: Work item dictionary
            result: Result from _execute_work
        """
        pass

    @abstractmethod
    def _on_permanent_failure(self, item: dict[str, Any]):
        """
        Handle permanent failure after max retries.

        Args:
            item: Work item dictionary
        """
        pass

    def _cancel_current_item(self, item: dict[str, Any]) -> None:  # noqa: B027
        """Cancel an in-flight work item during clear().

        Override in subclasses to perform item-specific cancellation
        (e.g. aborting a telescope task).  Default is a no-op.
        """

    def _update_retry_status(self, item: dict[str, Any], backoff: float, retry_count: int, max_retries: int):
        """Update task status message for retry."""
        task = self._get_task_from_item(item)
        if task:
            # Extract stage name from class name (e.g., "ImagingQueue" -> "Imaging")
            stage_name = self.__class__.__name__.replace("Queue", "")
            task.set_status_msg(
                f"{stage_name} failed (attempt {retry_count}/{max_retries}), retrying in {backoff:.0f}s..."
            )

    @abstractmethod
    def _get_task_from_item(self, item: dict[str, Any]):
        """
        Get the Task object from the work item dictionary.
        Must be implemented by subclasses since item structure varies.

        Args:
            item: Work item dictionary

        Returns:
            Task object or None if not found
        """
        pass

    def _set_retry_scheduled_time(self, item: dict[str, Any], scheduled_time: float | None = None):
        """Set the retry scheduled time on the task."""
        task = self._get_task_from_item(item)
        if task:
            task.set_retry_time(scheduled_time)

    def _update_status_on_resubmit(self, item: dict[str, Any]):
        """Update status when retry timer fires and task is resubmitted."""
        task = self._get_task_from_item(item)
        if task:
            # Extract stage name from class name (e.g., "ImagingQueue" -> "imaging")
            stage_name = self.__class__.__name__.replace("Queue", "").lower()
            task.set_status_msg(f"Retrying {stage_name}...")

    def _set_executing(self, item: dict[str, Any], executing: bool):
        """Set whether task is being actively executed."""
        task = self._get_task_from_item(item)
        if task:
            task.set_executing(executing)

    def _calculate_backoff(self, task_id: str) -> float:
        """Calculate exponential backoff delay."""
        retry_count = self.retry_counts.get(task_id, 0)
        initial = self.settings.initial_retry_delay_seconds
        max_delay = self.settings.max_retry_delay_seconds
        return min(initial * (2**retry_count), max_delay)

    def _should_retry(self, task_id: str) -> bool:
        """Check if task should be retried."""
        return self.retry_counts.get(task_id, 0) < self.settings.max_task_retries

    def _schedule_retry(self, item: dict[str, Any]):
        """Schedule a retry with exponential backoff."""
        task_id = item["task_id"]
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1
        self.last_failure[task_id] = time.time()

        backoff = self._calculate_backoff(task_id)
        retry_count = self.retry_counts[task_id]
        max_retries = self.settings.max_task_retries

        self.logger.warning(
            f"Task {task_id} failed (attempt {retry_count}/{max_retries}), " f"retrying in {backoff:.0f}s"
        )

        # Let subclass update task status message
        self._update_retry_status(item, backoff, retry_count, max_retries)

        # Set retry scheduled time
        scheduled_time = time.time() + backoff
        self._set_retry_scheduled_time(item, scheduled_time)

        def resubmit():
            with self._timer_lock:
                if timer in self._pending_timers:
                    self._pending_timers.remove(timer)
            self._set_retry_scheduled_time(item, None)
            self._update_status_on_resubmit(item)
            self.work_queue.put(item)

        timer = threading.Timer(backoff, resubmit)
        timer.daemon = True
        with self._timer_lock:
            self._pending_timers.append(timer)
        timer.start()

    def _worker_loop(self):
        """Worker thread main loop."""
        while self.running:
            try:
                item = self.work_queue.get(timeout=1)
                if item is None:  # Poison pill
                    break

                task_id = item["task_id"]

                with self._current_item_lock:
                    self._current_item = item
                try:
                    self._set_executing(item, True)
                    self.total_attempts += 1
                    epoch = self._clear_epoch

                    success, result = self._execute_work(item)

                    self._set_executing(item, False)

                    if self._clear_epoch != epoch:
                        self.logger.info(f"Task {task_id} result discarded (queue cleared during execution)")
                        self.retry_counts.pop(task_id, None)
                        self.last_failure.pop(task_id, None)
                        self._on_permanent_failure(item)
                    elif success:
                        self.total_successes += 1
                        self.retry_counts.pop(task_id, None)
                        self.last_failure.pop(task_id, None)
                        self._set_retry_scheduled_time(item, None)
                        self._on_success(item, result)
                    else:
                        if self._should_retry(task_id):
                            self._schedule_retry(item)
                        else:
                            self.total_permanent_failures += 1
                            self.logger.error(
                                f"Task {task_id} permanently failed after "
                                f"{self.retry_counts.get(task_id, 0)} retries"
                            )
                            self.retry_counts.pop(task_id, None)
                            self.last_failure.pop(task_id, None)
                            self._on_permanent_failure(item)

                except Exception as e:
                    self._set_executing(item, False)

                    if self._clear_epoch != epoch:
                        self.logger.info(f"Task {task_id} discarded after exception (queue cleared)")
                        self.retry_counts.pop(task_id, None)
                        self.last_failure.pop(task_id, None)
                        self._on_permanent_failure(item)
                    else:
                        self.logger.error(f"Worker error for {task_id}: {e}", exc_info=True)
                        if self._should_retry(task_id):
                            self._schedule_retry(item)
                        else:
                            self.total_permanent_failures += 1
                            self.retry_counts.pop(task_id, None)
                            self.last_failure.pop(task_id, None)
                            self._on_permanent_failure(item)

                finally:
                    with self._current_item_lock:
                        self._current_item = None
                    self.work_queue.task_done()

            except queue.Empty:
                continue

    def clear(self) -> int:
        """Cancel in-flight work, drain pending items, cancel retry timers,
        and suppress in-flight retries.

        After clear(), any currently-executing task that fails will go straight
        to _on_permanent_failure instead of scheduling another retry.
        """
        self._clear_epoch += 1

        with self._current_item_lock:
            if self._current_item is not None:
                self._cancel_current_item(self._current_item)

        with self._timer_lock:
            for t in self._pending_timers:
                t.cancel()
            self._pending_timers.clear()

        # Max out retry counts so _should_retry() returns False for any
        # in-flight task that completes after this point.
        max_retries = self.settings.max_task_retries
        for task_id in list(self.retry_counts):
            self.retry_counts[task_id] = max_retries

        count = 0
        while True:
            try:
                self.work_queue.get_nowait()
                self.work_queue.task_done()
                count += 1
            except queue.Empty:
                break
        return count

    def is_idle(self) -> bool:
        """Return True if no items are queued or being actively worked on."""
        return self.work_queue.unfinished_tasks == 0

    def get_stats(self) -> dict:
        """Return a consistent snapshot of lifetime counters."""
        with self._stats_lock:
            return {
                "attempts": self.total_attempts,
                "successes": self.total_successes,
                "permanent_failures": self.total_permanent_failures,
            }

    def start(self):
        """Start worker threads."""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop, name=f"{self.__class__.__name__}-Worker-{i}", daemon=True
            )
            worker.start()
            self.workers.append(worker)
            self.logger.info(f"Started {self.__class__.__name__} worker {i}")

    def stop(self):
        """Stop all workers gracefully."""
        self.logger.info(f"Stopping {self.__class__.__name__}...")
        self.running = False

        # Send poison pills
        for _ in range(self.num_workers):
            self.work_queue.put(None)

        # Wait for completion
        for worker in self.workers:
            worker.join(timeout=5)

        self.logger.info(f"{self.__class__.__name__} stopped")
