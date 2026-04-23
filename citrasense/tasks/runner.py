from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dateutil import parser as dtparser

from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrasense.preview_bus import PreviewBus
from citrasense.safety.safety_monitor import SafetyAction
from citrasense.tasks.alignment_manager import AlignmentManager
from citrasense.tasks.autofocus_manager import AutofocusManager
from citrasense.tasks.calibration_manager import CalibrationManager
from citrasense.tasks.homing_manager import HomingManager
from citrasense.tasks.scope.sidereal_telescope_task import SiderealTelescopeTask
from citrasense.tasks.scope.tracking_telescope_task import TrackingTelescopeTask
from citrasense.tasks.task import Task

if TYPE_CHECKING:
    from citrasense.tasks.observing_session import ObservingSessionManager
    from citrasense.tasks.self_tasking_manager import SelfTaskingManager

# Task polling interval in seconds
TASK_POLL_INTERVAL_SECONDS = 15


class TaskManager:
    def __init__(
        self,
        api_client,
        logger,
        hardware_adapter: AbstractAstroHardwareAdapter,
        settings,
        processor_registry,
        elset_cache=None,
        apass_catalog=None,
        safety_monitor=None,
        location_service=None,
        telescope_record: dict | None = None,
        ground_station: dict | None = None,
        on_annotated_image=None,
        preview_bus: PreviewBus | None = None,
        task_index=None,
    ):
        self.api_client = api_client
        self.logger = logger.getChild(type(self).__name__)
        self.hardware_adapter = hardware_adapter
        self.settings = settings
        self.processor_registry = processor_registry
        self.elset_cache = elset_cache
        self.apass_catalog = apass_catalog
        self.safety_monitor = safety_monitor
        self.location_service = location_service
        self.telescope_record = telescope_record
        self.ground_station = ground_station
        self._on_annotated_image = on_annotated_image
        self._preview_bus = preview_bus
        self.task_index = task_index
        self.on_toast: Callable[[str, str, str | None], None] | None = None

        # Initialize work queues (TaskManager now owns these)
        from citrasense.tasks.imaging_queue import ImagingQueue
        from citrasense.tasks.processing_queue import ProcessingQueue
        from citrasense.tasks.upload_queue import UploadQueue

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
            self.settings,
            imaging_queue=self.imaging_queue,
            location_service=self.location_service,
            preview_bus=self._preview_bus,
        )
        self.alignment_manager = AlignmentManager(
            self.logger,
            self.hardware_adapter,
            self.settings,
            imaging_queue=self.imaging_queue,
            safety_monitor=self.safety_monitor,
            location_service=self.location_service,
            preview_bus=self._preview_bus,
        )
        self.homing_manager = HomingManager(
            self.logger,
            self.hardware_adapter,
            imaging_queue=self.imaging_queue,
        )
        self.calibration_manager: CalibrationManager | None = None

        # Lifetime task counters — lock gives atomic multi-field snapshots in get_task_stats().
        self._task_stats_lock = threading.Lock()
        self.total_tasks_started: int = 0
        self.total_tasks_succeeded: int = 0
        self.total_tasks_failed: int = 0
        # Automated scheduling state (initialized from server on startup)
        self._automated_scheduling = (
            self.telescope_record.get("automatedScheduling", False) if self.telescope_record else False
        )

        # Session managers (set via set_session_managers after construction)
        self._observing_session_manager: ObservingSessionManager | None = None
        self._self_tasking_manager: SelfTaskingManager | None = None

    def set_session_managers(
        self,
        observing_session_manager: ObservingSessionManager,
        self_tasking_manager: SelfTaskingManager,
    ) -> None:
        """Wire session managers after construction (resolves circular dependency with AutofocusManager)."""
        self._observing_session_manager = observing_session_manager
        self._self_tasking_manager = self_tasking_manager

    @property
    def observing_session_manager(self) -> ObservingSessionManager | None:
        return self._observing_session_manager

    @property
    def self_tasking_manager(self) -> SelfTaskingManager | None:
        return self._self_tasking_manager

    def are_queues_idle(self) -> bool:
        """Check if all work queues are idle (no in-flight work)."""
        return self.imaging_queue.is_idle() and self.processing_queue.is_idle() and self.upload_queue.is_idle()

    @property
    def automated_scheduling(self) -> bool:
        return self._automated_scheduling

    @automated_scheduling.setter
    def automated_scheduling(self, value: bool) -> None:
        self._automated_scheduling = value

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

    def drop_scheduled_task(self, task_id: str) -> bool:
        """Fully evict a scheduled (not-yet-running) task.

        Unlike :meth:`remove_task_from_all_stages` (which assumes the task
        has already been ``heappop``-ed off the heap when imaging started),
        this method also rebuilds the schedule heap so a UI-driven cancel
        is reflected immediately in the next ``get_tasks_snapshot`` call
        without waiting for the next API poll.

        Returns True when something was actually removed.
        """
        with self._stage_lock:
            self.imaging_tasks.pop(task_id, None)
            self.processing_tasks.pop(task_id, None)
            self.uploading_tasks.pop(task_id, None)

        with self.heap_lock:
            removed = task_id in self.task_ids
            if removed:
                self.task_ids.discard(task_id)
                self.task_dict.pop(task_id, None)
                # Rebuild the heap without the cancelled entry. O(n) is fine
                # here: the queue is small and cancels are rare.
                self.task_heap = [entry for entry in self.task_heap if entry[2] != task_id]
                heapq.heapify(self.task_heap)
        return removed

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
                    result["target_name"] = task.satelliteName if task.sensor_type == "telescope" else task.sensor_id
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
        if self.telescope_record is None:
            self.logger.error("poll_tasks called without telescope_record; cannot poll for tasks")
            return

        while not self._stop_event.is_set():
            try:
                # Drive session state machine and self-tasking before polling tasks
                if self._observing_session_manager:
                    self._observing_session_manager.update()
                if self._self_tasking_manager:
                    self._self_tasking_manager.check_and_request()

                if self.elset_cache:
                    interval_hours = self.settings.elset_refresh_interval_hours
                    self.elset_cache.refresh_if_stale(self.api_client, self.logger, interval_hours=interval_hours)
                self._report_online()
                now_iso = datetime.now(timezone.utc).isoformat()
                tasks = self.api_client.get_telescope_tasks(
                    self.telescope_record["id"],
                    statuses=["Pending", "Scheduled"],
                    task_stop_after=now_iso,
                )

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
        if self.telescope_record is None:
            return
        telescope_id = self.telescope_record["id"]
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
            if self.calibration_manager:
                self.calibration_manager.check_and_execute()

            # Defer tasks while homing or calibration is active.
            cal_busy = self.calibration_manager and (
                self.calibration_manager.is_running() or self.calibration_manager.is_requested()
            )
            if self.homing_manager.is_running() or self.homing_manager.is_requested() or cal_busy:
                self._stop_event.wait(1)
                continue

            with self._processing_lock:
                is_paused = not self._processing_active

            if not is_paused:
                # Don't start new imaging during session shutdown — let current work drain
                winding_down = (
                    self._observing_session_manager is not None and self._observing_session_manager.is_winding_down()
                )

                try:
                    now = int(time.time())
                    completed = 0
                    while not winding_down:
                        with self._processing_lock:
                            if not self._processing_active:
                                break

                        # Only submit one task at a time so maintenance (autofocus,
                        # alignment) and safety checks run between every task.
                        if not self.imaging_queue.is_idle():
                            break

                        if (
                            self.autofocus_manager.is_requested()
                            or self.autofocus_manager.is_running()
                            or self.alignment_manager.is_requested()
                            or self.alignment_manager.is_running()
                        ):
                            break

                        if self._last_safety_action in (SafetyAction.EMERGENCY, SafetyAction.QUEUE_STOP):
                            break

                        with self.heap_lock:
                            if not (self.task_heap and self.task_heap[0][0] <= now):
                                break
                            _start_epoch, stop_epoch, tid, task = self.task_heap[0]
                            if stop_epoch and stop_epoch < now:
                                heapq.heappop(self.task_heap)
                                self.task_ids.discard(tid)
                                self.task_dict.pop(tid, None)
                                self.logger.info(f"Skipping expired task {tid} (window closed)")
                                continue
                            heapq.heappop(self.task_heap)
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
        safety_monitor = self.safety_monitor
        if not safety_monitor:
            return False

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
            is_new = self._last_safety_action != SafetyAction.EMERGENCY
            if is_new:
                self.clear_pending_tasks()
                trigger_name = triggered_check.name if triggered_check else "unknown"
                self.logger.critical(
                    "EMERGENCY — cancelled in-flight imaging (trigger: %s)",
                    trigger_name,
                )
                if self.on_toast:
                    self.on_toast(
                        f"Safety EMERGENCY: {trigger_name} — imaging cancelled",
                        "danger",
                        "safety-emergency",
                    )
            if triggered_check and is_new:
                try:
                    triggered_check.execute_action()
                except Exception:
                    self.logger.error("Safety corrective action failed", exc_info=True)
            self._last_safety_action = action
            return True

        if action == SafetyAction.QUEUE_STOP:
            if self.imaging_queue.is_idle() and triggered_check:
                is_new = self._last_safety_action != SafetyAction.QUEUE_STOP
                if is_new:
                    self.logger.warning("Executing safety action from %r (queue idle)", triggered_check.name)
                try:
                    triggered_check.execute_action()
                except Exception:
                    if is_new:
                        self.logger.error("Safety corrective action failed", exc_info=True)
            self._last_safety_action = action
            return True

        self._last_safety_action = action
        return False

    def _create_telescope_task(self, task: Task):
        """Create appropriate telescope task instance for the given task.

        Selection depends on the ``observation_mode`` setting:
        - "auto": use TrackingTelescopeTask if the adapter reports
          ``supports_custom_tracking``, otherwise SiderealTelescopeTask.
        - "tracking": always TrackingTelescopeTask.
        - "sidereal": always SiderealTelescopeTask.
        """
        mode = self.settings.observation_mode

        use_tracking = False
        if mode == "tracking":
            use_tracking = True
        elif mode == "auto":
            use_tracking = self.hardware_adapter.supports_custom_tracking

        if use_tracking:
            self.logger.info("Using TrackingTelescopeTask (mode=%s)", mode)
        else:
            self.logger.info("Using SiderealTelescopeTask (mode=%s)", mode)

        cls = TrackingTelescopeTask if use_tracking else SiderealTelescopeTask
        return cls(
            self.api_client,
            self.hardware_adapter,
            self.logger,
            task,
            settings=self.settings,
            task_manager=self,
            location_service=self.location_service,
            telescope_record=self.telescope_record,
            ground_station=self.ground_station,
            elset_cache=self.elset_cache,
            apass_catalog=self.apass_catalog,
            processor_registry=self.processor_registry,
            on_annotated_image=self._set_latest_annotated_image,
            task_index=self.task_index,
        )

    @property
    def pending_task_count(self) -> int:
        """Return the number of tasks on the scheduling heap (thread-safe)."""
        with self.heap_lock:
            return len(self.task_heap)

    def get_tasks_snapshot(self, exclude_active: bool = False) -> list[Task]:
        """Return a thread-safe snapshot of tasks on the scheduling heap.

        If exclude_active is True, omits tasks currently in imaging/processing/upload stages.
        """
        if exclude_active:
            with self._stage_lock:
                active_ids = set(self.imaging_tasks) | set(self.processing_tasks) | set(self.uploading_tasks)
        else:
            active_ids = set()

        with self.heap_lock:
            return [task for _start, _stop, task_id, task in self.task_heap if task_id not in active_ids]

    def _set_latest_annotated_image(self, path: str) -> None:
        """Forward annotated image path to daemon for web UI display."""
        if self._on_annotated_image:
            self._on_annotated_image(path)

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
        """Cancel in-flight imaging and clear stage tracking.

        Returns the number of queued imaging items drained.  Any in-flight
        imaging task is also cancelled and will be marked as failed by the
        imaging worker's normal failure path.

        The task heap, task_ids, and task_dict are intentionally preserved —
        tm.pause() already prevents the runner from popping new tasks, and
        on resume the heap re-enters the imaging pipeline naturally.
        """
        with self._stage_lock:
            cleared = self.imaging_queue.clear()
            self.imaging_tasks.clear()
        return cleared

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
