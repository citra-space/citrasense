import heapq
import threading
import time
from datetime import datetime, timezone

from dateutil import parser as dtparser

from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrascope.tasks.alignment_manager import AlignmentManager
from citrascope.tasks.autofocus_manager import AutofocusManager
from citrascope.tasks.homing_manager import HomingManager
from citrascope.tasks.scope.static_telescope_task import StaticTelescopeTask
from citrascope.tasks.task import Task

# Task polling interval in seconds
TASK_POLL_INTERVAL_SECONDS = 15


class TaskManager:
    def __init__(
        self,
        api_client,
        logger,
        hardware_adapter: AbstractAstroHardwareAdapter,
        daemon,
        settings,
        processor_registry,
    ):
        self.api_client = api_client
        self.logger = logger
        self.hardware_adapter = hardware_adapter
        self.daemon = daemon
        self.settings = settings
        self.processor_registry = processor_registry

        # Initialize work queues (TaskManager now owns these)
        from citrascope.tasks.imaging_queue import ImagingQueue
        from citrascope.tasks.processing_queue import ProcessingQueue
        from citrascope.tasks.upload_queue import UploadQueue

        self.imaging_queue = ImagingQueue(
            num_workers=1,
            settings=settings,
            logger=logger,
            api_client=api_client,
            task_manager=self,
        )
        self.processing_queue = ProcessingQueue(
            num_workers=1,
            settings=settings,
            logger=logger,
        )
        self.upload_queue = UploadQueue(
            num_workers=1,
            settings=settings,
            logger=logger,
            task_manager=self,
        )

        # Stage tracking (TaskManager now owns this)
        self._stage_lock = threading.Lock()
        self.imaging_tasks = {}  # task_id -> start_time (float)
        self.processing_tasks = {}  # task_id -> start_time (float)
        self.uploading_tasks = {}  # task_id -> start_time (float)

        self.task_heap = []  # min-heap by start time (scheduled future work only)
        self.task_ids = set()
        self.task_dict = {}  # task_id -> Task object for quick lookup
        self.heap_lock = threading.RLock()
        self._stop_event = threading.Event()
        self.current_task_id = None  # Track currently executing task
        # Task processing control (restored from settings)
        self._processing_active = not settings.task_processing_paused
        self._processing_lock = threading.Lock()
        self._last_safety_action: object = None
        self.autofocus_manager = AutofocusManager(
            self.logger,
            self.hardware_adapter,
            self.daemon,
            imaging_queue=self.imaging_queue,
        )
        self.alignment_manager = AlignmentManager(
            self.logger,
            self.hardware_adapter,
            self.daemon,
            imaging_queue=self.imaging_queue,
        )
        self.homing_manager = HomingManager(
            self.logger,
            self.hardware_adapter,
            imaging_queue=self.imaging_queue,
        )

        # Lifetime task counters — lock gives atomic multi-field snapshots in get_task_stats().
        self._task_stats_lock = threading.Lock()
        self.total_tasks_started: int = 0
        self.total_tasks_succeeded: int = 0
        self.total_tasks_failed: int = 0
        # Automated scheduling state (initialized from server on startup)
        self._automated_scheduling = (
            daemon.telescope_record.get("automatedScheduling", False) if daemon.telescope_record else False
        )

    def update_task_stage(self, task_id: str, stage: str):
        """Move task to specified stage. Stage is 'imaging', 'processing', or 'uploading'."""
        with self._stage_lock:
            # Remove from all stages first
            self.imaging_tasks.pop(task_id, None)
            self.processing_tasks.pop(task_id, None)
            self.uploading_tasks.pop(task_id, None)

            # Add to new stage
            if stage == "imaging":
                self.imaging_tasks[task_id] = time.time()
            elif stage == "processing":
                self.processing_tasks[task_id] = time.time()
            elif stage == "uploading":
                self.uploading_tasks[task_id] = time.time()

    def remove_task_from_all_stages(self, task_id: str):
        """Remove task from all stages and active tracking (when complete)."""
        with self._stage_lock:
            self.imaging_tasks.pop(task_id, None)
            self.processing_tasks.pop(task_id, None)
            self.uploading_tasks.pop(task_id, None)

        # Also remove from task_dict
        with self.heap_lock:
            self.task_dict.pop(task_id, None)

    def record_task_started(self) -> None:
        with self._task_stats_lock:
            self.total_tasks_started += 1

    def record_task_succeeded(self) -> None:
        with self._task_stats_lock:
            self.total_tasks_succeeded += 1

    def record_task_failed(self) -> None:
        with self._task_stats_lock:
            self.total_tasks_failed += 1

    def get_task_stats(self) -> dict:
        """Return a consistent snapshot of lifetime task counters."""
        with self._task_stats_lock:
            return {
                "started": self.total_tasks_started,
                "succeeded": self.total_tasks_succeeded,
                "failed": self.total_tasks_failed,
            }

    def get_tasks_by_stage(self) -> dict:
        """Get current tasks in each stage, enriched with task details."""
        with self._stage_lock:
            now = time.time()

            def enrich_task(task_id: str, start_time: float) -> dict:
                """Look up task details and create enriched dict."""
                result = {"task_id": task_id, "elapsed": now - start_time}
                # Look up task from task manager
                task = self.get_task_by_id(task_id)
                if task:
                    result["target_name"] = task.satelliteName
                    # Use thread-safe getters for status fields
                    status_msg, retry_time, is_executing = task.get_status_info()
                    result["status_msg"] = status_msg
                    result["retry_scheduled_time"] = retry_time
                    result["is_being_executed"] = is_executing
                return result

            def sort_tasks(tasks):
                """Sort tasks: active work first, queued next, retry-waiting last."""

                def sort_key(task):
                    retry_time = task.get("retry_scheduled_time")
                    is_executing = task.get("is_being_executed", False)

                    # Three-tier priority:
                    # Priority 0: Currently executing (highest priority)
                    # Priority 1: Queued and ready to execute
                    # Priority 2: Waiting for retry (lowest priority)
                    if retry_time is not None:
                        priority = 2
                        sort_value = retry_time  # Soonest retry first
                    elif is_executing:
                        priority = 0
                        sort_value = -task.get("elapsed", 0)  # Longest running first
                    else:
                        priority = 1
                        sort_value = -task.get("elapsed", 0)  # Longest waiting first

                    return (priority, sort_value)

                return sorted(tasks, key=sort_key)

            return {
                "imaging": sort_tasks([enrich_task(tid, start) for tid, start in self.imaging_tasks.items()]),
                "processing": sort_tasks([enrich_task(tid, start) for tid, start in self.processing_tasks.items()]),
                "uploading": sort_tasks([enrich_task(tid, start) for tid, start in self.uploading_tasks.items()]),
            }

    def poll_tasks(self):
        while not self._stop_event.is_set():
            try:
                # Refresh elset hot list when stale (for satellite matcher)
                if getattr(self.daemon, "elset_cache", None) and self.daemon.telescope_record:
                    interval_hours = getattr(self.daemon.settings, "elset_refresh_interval_hours", 6)
                    self.daemon.elset_cache.refresh_if_stale(
                        self.api_client, self.logger, interval_hours=interval_hours
                    )
                self._report_online()
                tasks = self.api_client.get_telescope_tasks(self.daemon.telescope_record["id"])

                # If API call failed (timeout, network error, etc.), wait before retrying
                if tasks is None:
                    self._stop_event.wait(TASK_POLL_INTERVAL_SECONDS)
                    continue

                added = 0
                removed = 0
                now = int(time.time())
                # Snapshot in-flight task IDs before acquiring heap_lock to avoid
                # lock inversion (remove_task_from_all_stages holds _stage_lock then heap_lock).
                with self._stage_lock:
                    inflight_ids = set(self.imaging_tasks) | set(self.processing_tasks) | set(self.uploading_tasks)
                with self.heap_lock:
                    # Build a map of current valid tasks from the API
                    api_task_map = {}
                    for task_dict in tasks:
                        try:
                            task = Task.from_dict(task_dict)
                            tid = task.id
                            if tid and task.status in ["Pending", "Scheduled"]:
                                api_task_map[tid] = task
                        except Exception as e:
                            self.logger.error(f"Error parsing task from API: {e}", exc_info=True)

                    # Remove tasks from heap that are no longer valid (cancelled, completed, or not in API response)
                    new_heap = []
                    for start_epoch, stop_epoch, tid, task in self.task_heap:
                        # Keep task if it's still in the API response with a valid status
                        # Don't remove currently executing task
                        if tid == self.current_task_id or tid in api_task_map:
                            new_heap.append((start_epoch, stop_epoch, tid, task))
                        else:
                            self.logger.info(f"Removing task {tid} from queue (cancelled or status changed)")
                            self.task_ids.discard(tid)
                            self.task_dict.pop(tid, None)
                            removed += 1

                    # Rebuild heap if we removed anything
                    if removed > 0:
                        self.task_heap = new_heap
                        heapq.heapify(self.task_heap)

                    # Add new tasks that aren't already in the heap
                    for tid, task in api_task_map.items():
                        # Skip if task is in heap, currently executing, or already in-flight
                        # through the imaging → processing → upload pipeline.
                        if tid not in self.task_ids and tid != self.current_task_id and tid not in inflight_ids:
                            task_start = task.taskStart
                            task_stop = task.taskStop
                            try:
                                start_epoch = int(dtparser.isoparse(task_start).timestamp())
                                stop_epoch = int(dtparser.isoparse(task_stop).timestamp()) if task_stop else 0
                            except Exception:
                                self.logger.error(f"Could not parse taskStart/taskStop for task {tid}")
                                continue
                            if stop_epoch and stop_epoch < now:
                                self.logger.debug(f"Skipping past task {tid} that ended at {task_stop}")
                                continue  # Skip tasks whose end date has passed
                            heapq.heappush(self.task_heap, (start_epoch, stop_epoch, tid, task))
                            self.task_ids.add(tid)
                            self.task_dict[tid] = task  # Store for quick lookup
                            added += 1

                    if added > 0 or removed > 0:
                        action = []
                        if added > 0:
                            action.append(f"Added {added}")
                        if removed > 0:
                            action.append(f"Removed {removed}")
                        self.logger.info(self._heap_summary(f"{', '.join(action)} tasks"))
                    # self.logger.info(self._heap_summary("Polled tasks"))
            except Exception as e:
                self.logger.error(f"Exception in poll_tasks loop: {e}", exc_info=True)
                time.sleep(5)  # avoid tight error loop
            self._stop_event.wait(TASK_POLL_INTERVAL_SECONDS)

    def _report_online(self):
        """
        PUT to /telescopes to report this telescope as online.
        """
        telescope_id = self.daemon.telescope_record["id"]
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        self.api_client.put_telescope_status([{"id": telescope_id, "last_connection_epoch": iso_timestamp}])
        self.logger.debug(f"Reported online status for telescope {telescope_id} at {iso_timestamp}")

    def task_runner(self):
        while not self._stop_event.is_set():
            # Safety evaluation runs first — before any managers or tasks.
            if self._evaluate_safety():
                self._stop_event.wait(1)
                continue

            # Operator-requested maintenance runs regardless of pause state.
            self.homing_manager.check_and_execute()
            self.alignment_manager.check_and_execute()
            self.autofocus_manager.check_and_execute()

            # Defer tasks while homing is active — mount is physically moving.
            if self.homing_manager.is_running() or self.homing_manager.is_requested():
                self._stop_event.wait(1)
                continue

            with self._processing_lock:
                is_paused = not self._processing_active

            if not is_paused:
                try:
                    now = int(time.time())
                    completed = 0
                    while True:
                        with self._processing_lock:
                            if not self._processing_active:
                                break

                        with self.heap_lock:
                            if not (self.task_heap and self.task_heap[0][0] <= now):
                                break
                            _, _, tid, task = heapq.heappop(self.task_heap)
                            self.task_ids.discard(tid)

                            self.logger.info(f"Starting task {tid} at {datetime.now().isoformat()}: {task}")
                            self.current_task_id = tid

                        self.update_task_stage(tid, "imaging")
                        task.set_status_msg("Queued for imaging...")
                        telescope_task = self._create_telescope_task(task)

                        def on_imaging_complete(task_id, success):
                            """Callback when imaging completes or permanently fails."""
                            if success:
                                with self.heap_lock:
                                    self.current_task_id = None
                                self.logger.info(f"Completed imaging task {task_id} successfully.")
                            else:
                                self.logger.error(f"Imaging task {task_id} permanently failed.")
                                with self.heap_lock:
                                    self.current_task_id = None
                                self.record_task_failed()
                                self.remove_task_from_all_stages(task_id)

                        self.imaging_queue.submit(tid, task, telescope_task, on_imaging_complete)
                        completed += 1

                    if completed > 0:
                        self.logger.info(self._heap_summary("Completed tasks"))
                except Exception as e:
                    self.logger.error(f"Exception in task_runner loop: {e}", exc_info=True)
                    time.sleep(5)

            self._stop_event.wait(1)

    def _evaluate_safety(self) -> bool:
        """Run safety monitor evaluation; return True if the loop should yield."""
        safety_monitor = getattr(self.daemon, "safety_monitor", None)
        if not safety_monitor:
            return False

        from citrascope.safety.safety_monitor import SafetyAction

        try:
            action, triggered_check = safety_monitor.evaluate()
        except Exception:
            self.logger.error("Safety monitor evaluation failed — yielding task loop", exc_info=True)
            return True

        if action == SafetyAction.EMERGENCY:
            try:
                self.hardware_adapter.abort_slew()
            except Exception:
                pass
            if triggered_check:
                is_new = self._last_safety_action != SafetyAction.EMERGENCY
                if is_new:
                    self.logger.critical("Executing safety action from %r", triggered_check.name)
                    try:
                        triggered_check.execute_action()
                    except Exception:
                        self.logger.error("Safety corrective action failed", exc_info=True)
            self._last_safety_action = action
            return True

        if action == SafetyAction.QUEUE_STOP:
            if self.imaging_queue.is_idle():
                if triggered_check:
                    is_new = self._last_safety_action != SafetyAction.QUEUE_STOP
                    if is_new:
                        self.logger.warning("Executing safety action from %r (queue idle)", triggered_check.name)
                        try:
                            triggered_check.execute_action()
                        except Exception:
                            self.logger.error("Safety corrective action failed", exc_info=True)
            self._last_safety_action = action
            return True

        self._last_safety_action = action
        return False

    def _create_telescope_task(self, task: Task):
        """Create appropriate telescope task instance for the given task."""
        # For now, use StaticTelescopeTask
        # Future: could choose between Static and Tracking based on task type
        return StaticTelescopeTask(
            self.api_client,
            self.hardware_adapter,
            self.logger,
            task,
            self.daemon,
        )

    def get_task_by_id(self, task_id: str):
        """Get a task by ID. Thread-safe."""
        with self.heap_lock:
            return self.task_dict.get(task_id)

    def _heap_summary(self, event):
        with self.heap_lock:
            summary = f"{event}: {len(self.task_heap)} tasks in queue. "
            next_tasks = []
            if self.task_heap:
                next_tasks = [
                    f"{tid} at {datetime.fromtimestamp(start).isoformat()}"
                    for start, stop, tid, _ in self.task_heap[:3]
                ]
                if len(self.task_heap) > 3:
                    next_tasks.append(f"... ({len(self.task_heap)-3} more)")
            if self.current_task_id is not None:
                # Show the current in-flight task at the front
                summary += f"Current: {self.current_task_id}. "
            if not next_tasks:
                summary += "No tasks scheduled."
            return summary

    def clear_pending_tasks(self) -> int:
        """Drain the task heap and imaging queue. Returns total items removed."""
        with self.heap_lock:
            count = len(self.task_heap)
            self.task_heap.clear()
            self.task_ids.clear()
            self.task_dict.clear()
        count += self.imaging_queue.clear()
        return count

    def pause(self) -> bool:
        """Pause task processing. Returns new state (False)."""
        with self._processing_lock:
            self._processing_active = False
            self.logger.info("Task processing paused")
            return self._processing_active

    def resume(self) -> bool:
        """Resume task processing. Returns new state (True)."""
        with self._processing_lock:
            self._processing_active = True
            self.logger.info("Task processing resumed")
            return self._processing_active

    def is_processing_active(self) -> bool:
        """Check if task processing is currently active."""
        with self._processing_lock:
            return self._processing_active

    def start(self):
        self._stop_event.clear()

        # Start work queues
        self.logger.info("Starting work queues...")
        self.imaging_queue.start()
        self.processing_queue.start()
        self.upload_queue.start()

        # Start task management threads
        self.poll_thread = threading.Thread(target=self.poll_tasks, daemon=True)
        self.runner_thread = threading.Thread(target=self.task_runner, daemon=True)
        self.poll_thread.start()
        self.runner_thread.start()

    def stop(self):
        self._stop_event.set()

        # Stop work queues
        self.logger.info("Stopping work queues...")
        self.imaging_queue.stop()
        self.processing_queue.stop()
        self.upload_queue.stop()

        # Stop task management threads
        self.poll_thread.join()
        self.runner_thread.join()
