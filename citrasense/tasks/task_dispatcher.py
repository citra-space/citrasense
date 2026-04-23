"""Site-level task orchestration.

``TaskDispatcher`` is the single, site-level component that polls the API for
new tasks, manages the scheduling heap, routes tasks to the appropriate
:class:`~citrasense.sensors.sensor_runtime.SensorRuntime`, tracks task stages,
handles safety evaluation, pause/resume, lifetime stats, and session managers.

It exposes a backward-compatible facade so existing web routes (which access
``daemon.task_manager.imaging_queue``, ``daemon.task_manager.autofocus_manager``,
etc.) continue to work without modification.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from dateutil import parser as dtparser

from citrasense.safety.safety_monitor import SafetyAction
from citrasense.tasks.task import Task

if TYPE_CHECKING:
    from citrasense.sensors.sensor_runtime import SensorRuntime
    from citrasense.tasks.observing_session import ObservingSessionManager
    from citrasense.tasks.self_tasking_manager import SelfTaskingManager

TASK_POLL_INTERVAL_SECONDS = 15


class TaskDispatcher:
    """Site-level task orchestration: poll, schedule, route, track."""

    def __init__(
        self,
        api_client: Any,
        logger: Any,
        settings: Any,
        *,
        hardware_adapter: Any = None,
        safety_monitor: Any = None,
        telescope_record: dict | None = None,
        elset_cache: Any = None,
    ) -> None:
        self.api_client = api_client
        self.logger = logger.getChild(type(self).__name__)
        self.settings = settings
        self.hardware_adapter = hardware_adapter
        self.safety_monitor = safety_monitor
        self.telescope_record = telescope_record
        self.elset_cache = elset_cache
        self.on_toast: Callable[[str, str, str | None], None] | None = None

        # Registered runtimes keyed by sensor_id
        self._runtimes: dict[str, SensorRuntime] = {}

        # ── Stage tracking (site-wide) ─────────────────────────────────
        self._stage_lock = threading.Lock()
        self.imaging_tasks: dict[str, float] = {}
        self.processing_tasks: dict[str, float] = {}
        self.uploading_tasks: dict[str, float] = {}

        # ── Task heap ──────────────────────────────────────────────────
        self.task_heap: list = []
        self.task_ids: set[str] = set()
        self.task_dict: dict[str, Task] = {}
        self.heap_lock = threading.RLock()
        self._stop_event = threading.Event()
        self.current_task_id: str | None = None

        # ── Pause / resume ─────────────────────────────────────────────
        self._processing_active = not settings.task_processing_paused
        self._processing_lock = threading.Lock()
        self._last_safety_action: object = None

        # ── Lifetime stats ─────────────────────────────────────────────
        self._task_stats_lock = threading.Lock()
        self.total_tasks_started: int = 0
        self.total_tasks_succeeded: int = 0
        self.total_tasks_failed: int = 0

        # ── Automated scheduling ───────────────────────────────────────
        self._automated_scheduling = (
            self.telescope_record.get("automatedScheduling", False) if self.telescope_record else False
        )

        # ── Session managers (set after construction) ──────────────────
        self._observing_session_manager: ObservingSessionManager | None = None
        self._self_tasking_manager: SelfTaskingManager | None = None

    # ── Runtime registration ───────────────────────────────────────────

    def register_runtime(self, runtime: SensorRuntime) -> None:
        """Add a SensorRuntime and wire its dispatcher back-reference."""
        self._runtimes[runtime.sensor_id] = runtime
        runtime.set_dispatcher(self)

    @property
    def _default_runtime(self) -> SensorRuntime:
        """Phase 1: the single registered runtime."""
        return next(iter(self._runtimes.values()))

    def _runtime_for_task(self, task: Task) -> SensorRuntime | None:
        """Resolve the runtime responsible for this task."""
        if task.sensor_id and task.sensor_id in self._runtimes:
            return self._runtimes[task.sensor_id]
        for rt in self._runtimes.values():
            if rt.sensor_type == task.sensor_type:
                return rt
        return self._default_runtime if self._runtimes else None

    # ── Web-compat facade ──────────────────────────────────────────────
    # Existing web routes access daemon.task_manager.<property>.
    # These properties delegate to the default (phase 1: only) runtime.

    @property
    def imaging_queue(self):
        return self._default_runtime.acquisition_queue

    @property
    def processing_queue(self):
        return self._default_runtime.processing_queue

    @property
    def upload_queue(self):
        return self._default_runtime.upload_queue

    @property
    def autofocus_manager(self):
        return self._default_runtime.autofocus_manager

    @property
    def alignment_manager(self):
        return self._default_runtime.alignment_manager

    @property
    def homing_manager(self):
        return self._default_runtime.homing_manager

    @property
    def calibration_manager(self):
        return self._default_runtime.calibration_manager

    @calibration_manager.setter
    def calibration_manager(self, value: Any) -> None:
        self._default_runtime.calibration_manager = value

    # ── Session managers ───────────────────────────────────────────────

    def set_session_managers(
        self,
        observing_session_manager: ObservingSessionManager,
        self_tasking_manager: SelfTaskingManager,
    ) -> None:
        """Wire session managers after construction (resolves circular dependency)."""
        self._observing_session_manager = observing_session_manager
        self._self_tasking_manager = self_tasking_manager

    @property
    def observing_session_manager(self) -> ObservingSessionManager | None:
        return self._observing_session_manager

    @property
    def self_tasking_manager(self) -> SelfTaskingManager | None:
        return self._self_tasking_manager

    @property
    def automated_scheduling(self) -> bool:
        return self._automated_scheduling

    @automated_scheduling.setter
    def automated_scheduling(self, value: bool) -> None:
        self._automated_scheduling = value

    # ── Queue helpers (facade) ─────────────────────────────────────────

    def are_queues_idle(self) -> bool:
        return self._default_runtime.are_queues_idle()

    # ── Stage tracking ─────────────────────────────────────────────────

    def update_task_stage(self, task_id: str, stage: str) -> None:
        with self._stage_lock:
            self.imaging_tasks.pop(task_id, None)
            self.processing_tasks.pop(task_id, None)
            self.uploading_tasks.pop(task_id, None)

            if stage == "imaging":
                self.imaging_tasks[task_id] = time.time()
            elif stage == "processing":
                self.processing_tasks[task_id] = time.time()
            elif stage == "uploading":
                self.uploading_tasks[task_id] = time.time()

    def remove_task_from_all_stages(self, task_id: str) -> None:
        with self._stage_lock:
            self.imaging_tasks.pop(task_id, None)
            self.processing_tasks.pop(task_id, None)
            self.uploading_tasks.pop(task_id, None)

        with self.heap_lock:
            self.task_dict.pop(task_id, None)

    def drop_scheduled_task(self, task_id: str) -> bool:
        """Fully evict a scheduled (not-yet-running) task.

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
                self.task_heap = [entry for entry in self.task_heap if entry[2] != task_id]
                heapq.heapify(self.task_heap)
        return removed

    # ── Stats ──────────────────────────────────────────────────────────

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
        with self._task_stats_lock:
            return {
                "started": self.total_tasks_started,
                "succeeded": self.total_tasks_succeeded,
                "failed": self.total_tasks_failed,
            }

    # ── Task queries ───────────────────────────────────────────────────

    def get_tasks_by_stage(self) -> dict:
        with self._stage_lock:
            now = time.time()

            def enrich_task(task_id: str, start_time: float) -> dict:
                result: dict[str, Any] = {"task_id": task_id, "elapsed": now - start_time}
                task = self.get_task_by_id(task_id)
                if task:
                    result["target_name"] = task.satelliteName if task.sensor_type == "telescope" else task.sensor_id
                    status_msg, retry_time, is_executing = task.get_status_info()
                    result["status_msg"] = status_msg
                    result["retry_scheduled_time"] = retry_time
                    result["is_being_executed"] = is_executing
                return result

            def sort_tasks(tasks: list) -> list:
                def sort_key(task: dict) -> tuple:
                    retry_time = task.get("retry_scheduled_time")
                    is_executing = task.get("is_being_executed", False)
                    if retry_time is not None:
                        return (2, retry_time)
                    if is_executing:
                        return (0, -task.get("elapsed", 0))
                    return (1, -task.get("elapsed", 0))

                return sorted(tasks, key=sort_key)

            return {
                "imaging": sort_tasks([enrich_task(tid, start) for tid, start in self.imaging_tasks.items()]),
                "processing": sort_tasks([enrich_task(tid, start) for tid, start in self.processing_tasks.items()]),
                "uploading": sort_tasks([enrich_task(tid, start) for tid, start in self.uploading_tasks.items()]),
            }

    @property
    def pending_task_count(self) -> int:
        with self.heap_lock:
            return len(self.task_heap)

    def get_tasks_snapshot(self, exclude_active: bool = False) -> list[Task]:
        if exclude_active:
            with self._stage_lock:
                active_ids = set(self.imaging_tasks) | set(self.processing_tasks) | set(self.uploading_tasks)
        else:
            active_ids = set()

        with self.heap_lock:
            return [task for _start, _stop, task_id, task in self.task_heap if task_id not in active_ids]

    def get_task_by_id(self, task_id: str) -> Task | None:
        with self.heap_lock:
            return self.task_dict.get(task_id)

    # ── Poll loop ──────────────────────────────────────────────────────

    def poll_tasks(self) -> None:
        if self.telescope_record is None:
            self.logger.error("poll_tasks called without telescope_record; cannot poll for tasks")
            return

        while not self._stop_event.is_set():
            try:
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

                if tasks is None:
                    self._stop_event.wait(TASK_POLL_INTERVAL_SECONDS)
                    continue

                added = 0
                removed = 0
                now = int(time.time())
                with self._stage_lock:
                    inflight_ids = set(self.imaging_tasks) | set(self.processing_tasks) | set(self.uploading_tasks)
                with self.heap_lock:
                    api_task_map: dict[str, Task] = {}
                    for task_dict in tasks:
                        try:
                            task = Task.from_dict(task_dict)
                            tid = task.id
                            if tid and task.status in ["Pending", "Scheduled"]:
                                api_task_map[tid] = task
                        except Exception as e:
                            self.logger.error("Error parsing task from API: %s", e, exc_info=True)

                    new_heap = []
                    for start_epoch, stop_epoch, tid, task in self.task_heap:
                        if tid == self.current_task_id or tid in api_task_map:
                            new_heap.append((start_epoch, stop_epoch, tid, task))
                        else:
                            self.logger.info("Removing task %s from queue (cancelled or status changed)", tid)
                            self.task_ids.discard(tid)
                            self.task_dict.pop(tid, None)
                            removed += 1

                    if removed > 0:
                        self.task_heap = new_heap
                        heapq.heapify(self.task_heap)

                    for tid, task in api_task_map.items():
                        if tid not in self.task_ids and tid != self.current_task_id and tid not in inflight_ids:
                            task_start = task.taskStart
                            task_stop = task.taskStop
                            try:
                                start_epoch = int(dtparser.isoparse(task_start).timestamp())
                                stop_epoch = int(dtparser.isoparse(task_stop).timestamp()) if task_stop else 0
                            except Exception:
                                self.logger.error("Could not parse taskStart/taskStop for task %s", tid)
                                continue
                            if stop_epoch and stop_epoch < now:
                                self.logger.debug("Skipping past task %s that ended at %s", tid, task_stop)
                                continue
                            heapq.heappush(self.task_heap, (start_epoch, stop_epoch, tid, task))
                            self.task_ids.add(tid)
                            self.task_dict[tid] = task
                            added += 1

                    if added > 0 or removed > 0:
                        action_parts = []
                        if added > 0:
                            action_parts.append(f"Added {added}")
                        if removed > 0:
                            action_parts.append(f"Removed {removed}")
                        self.logger.info(self._heap_summary(f"{', '.join(action_parts)} tasks"))
            except Exception as e:
                self.logger.error("Exception in poll_tasks loop: %s", e, exc_info=True)
                time.sleep(5)
            self._stop_event.wait(TASK_POLL_INTERVAL_SECONDS)

    def _report_online(self) -> None:
        if self.telescope_record is None:
            return
        telescope_id = self.telescope_record["id"]
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        self.api_client.put_telescope_status([{"id": telescope_id, "last_connection_epoch": iso_timestamp}])
        self.logger.debug("Reported online status for telescope %s at %s", telescope_id, iso_timestamp)

    # ── Task runner ────────────────────────────────────────────────────

    def task_runner(self) -> None:
        while not self._stop_event.is_set():
            if self._evaluate_safety():
                self._stop_event.wait(1)
                continue

            # Run maintenance on all runtimes
            for rt in self._runtimes.values():
                rt.check_maintenance()

            # Defer tasks while maintenance is active
            if any(rt.is_maintenance_blocking() for rt in self._runtimes.values()):
                self._stop_event.wait(1)
                continue

            with self._processing_lock:
                is_paused = not self._processing_active

            if not is_paused:
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

                        # Check if the default runtime can accept work
                        rt = self._default_runtime
                        if not rt.acquisition_queue.is_idle():
                            break
                        if rt.is_focus_or_alignment_active():
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
                                self.logger.info("Skipping expired task %s (window closed)", tid)
                                continue
                            heapq.heappop(self.task_heap)
                            self.task_ids.discard(tid)

                            self.logger.info("Starting task %s at %s: %s", tid, datetime.now().isoformat(), task)
                            self.current_task_id = tid

                        # Route to the appropriate runtime
                        runtime = self._runtime_for_task(task)
                        if runtime is None:
                            self.logger.warning(
                                "No runtime for task %s (sensor_type=%s) — dropping",
                                tid,
                                task.sensor_type,
                            )
                            self.remove_task_from_all_stages(tid)
                            with self.heap_lock:
                                self.current_task_id = None
                            continue

                        self.update_task_stage(tid, "imaging")
                        task.set_status_msg("Queued for imaging...")

                        def on_imaging_complete(task_id: str, success: bool) -> None:
                            if success:
                                with self.heap_lock:
                                    self.current_task_id = None
                                self.logger.info("Completed imaging task %s successfully.", task_id)
                            else:
                                self.logger.error("Imaging task %s permanently failed.", task_id)
                                with self.heap_lock:
                                    self.current_task_id = None
                                self.record_task_failed()
                                self.remove_task_from_all_stages(task_id)

                        runtime.submit_task(task, on_imaging_complete)
                        completed += 1

                    if completed > 0:
                        self.logger.info(self._heap_summary("Completed tasks"))
                except Exception as e:
                    self.logger.error("Exception in task_runner loop: %s", e, exc_info=True)
                    time.sleep(5)

            self._stop_event.wait(1)

    # ── Safety ─────────────────────────────────────────────────────────

    def _evaluate_safety(self) -> bool:
        """Run safety monitor evaluation; return True if the loop should yield."""
        if not self.safety_monitor:
            return False

        try:
            action, triggered_check = self.safety_monitor.evaluate()
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

    # ── Pending task management ────────────────────────────────────────

    def clear_pending_tasks(self) -> int:
        """Cancel in-flight imaging and clear stage tracking.

        Returns the number of queued imaging items drained.
        """
        with self._stage_lock:
            cleared = self._default_runtime.acquisition_queue.clear()
            self.imaging_tasks.clear()
        return cleared

    def pause(self) -> bool:
        with self._processing_lock:
            self._processing_active = False
            self.logger.info("Task processing paused")
            return self._processing_active

    def resume(self) -> bool:
        with self._processing_lock:
            self._processing_active = True
            self.logger.info("Task processing resumed")
            return self._processing_active

    def is_processing_active(self) -> bool:
        with self._processing_lock:
            return self._processing_active

    # ── Diagnostics ────────────────────────────────────────────────────

    def _heap_summary(self, event: str) -> str:
        with self.heap_lock:
            summary = f"{event}: {len(self.task_heap)} tasks in queue. "
            next_tasks = []
            if self.task_heap:
                next_tasks = [
                    f"{tid} at {datetime.fromtimestamp(start).isoformat()}"
                    for start, stop, tid, _ in self.task_heap[:3]
                ]
                if len(self.task_heap) > 3:
                    next_tasks.append(f"... ({len(self.task_heap) - 3} more)")
            if self.current_task_id is not None:
                summary += f"Current: {self.current_task_id}. "
            if not next_tasks:
                summary += "No tasks scheduled."
            return summary

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()

        self.logger.info("Starting sensor runtimes...")
        for rt in self._runtimes.values():
            rt.start()

        self.poll_thread = threading.Thread(target=self.poll_tasks, daemon=True)
        self.runner_thread = threading.Thread(target=self.task_runner, daemon=True)
        self.poll_thread.start()
        self.runner_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        self.logger.info("Stopping sensor runtimes...")
        for rt in self._runtimes.values():
            rt.stop()

        self.poll_thread.join()
        self.runner_thread.join()
