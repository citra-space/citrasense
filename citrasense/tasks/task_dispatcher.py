"""Site-level task orchestration.

``TaskDispatcher`` is the single, site-level component that polls the API for
new tasks, manages the scheduling heap, routes tasks to the appropriate
:class:`~citrasense.sensors.sensor_runtime.SensorRuntime`, tracks task stages,
handles safety evaluation, pause/resume, lifetime stats, and session managers.

Web routes and the daemon resolve per-sensor hardware through
``SensorManager`` / ``SensorRuntime`` directly; TaskDispatcher no longer
exposes per-sensor facade properties.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from dateutil import parser as dtparser

from citrasense.safety.safety_monitor import SafetyAction
from citrasense.tasks.task import Task

if TYPE_CHECKING:
    from citrasense.sensors.sensor_runtime import SensorRuntime
    from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor

TASK_POLL_INTERVAL_SECONDS = 15

# Sentinel key used in ``_last_safety_action_by_key`` to record the
# last action reported by a site-level safety check (one with no
# ``sensor_id``).  Per-sensor checks key off their own ``sensor_id``.
_SITE_SAFETY_KEY = "__site__"


class TaskDispatcher:
    """Site-level task orchestration: poll, schedule, route, track."""

    def __init__(
        self,
        api_client: Any,
        logger: Any,
        settings: Any,
        *,
        safety_monitor: Any = None,
        elset_cache: Any = None,
    ) -> None:
        self.api_client = api_client
        self.logger = logger.getChild(type(self).__name__)
        self.settings = settings
        self.safety_monitor = safety_monitor
        self.elset_cache = elset_cache
        self.on_toast: Callable[[str, str, str | None], None] | None = None

        # Registered runtimes keyed by sensor_id
        self._runtimes: dict[str, SensorRuntime] = {}

        # ── Stage tracking (site-wide) ─────────────────────────────────
        self._stage_lock = threading.Lock()
        self.imaging_tasks: dict[str, float] = {}
        self.processing_tasks: dict[str, float] = {}
        self.uploading_tasks: dict[str, float] = {}

        # ── Per-sensor task heaps ─────────────────────────────────────
        self._sensor_heaps: dict[str, list] = {}
        self._sensor_task_ids: dict[str, set[str]] = {}
        self._sensor_task_dicts: dict[str, dict[str, Task]] = {}
        self._current_task_ids: dict[str, str | None] = {}
        self.heap_lock = threading.RLock()
        self._stop_event = threading.Event()

        # ── Safety / last action ───────────────────────────────────────
        #
        # Per-sensor edge detection so that a QUEUE_STOP on sensor A
        # doesn't suppress the "first-time" log / toast / corrective-action
        # firing for a later QUEUE_STOP on sensor B.  Keyed by sensor_id
        # for sensor-scoped checks and the ``_SITE_SAFETY_KEY`` sentinel
        # for site-level checks.
        self._last_safety_action_by_key: dict[str, SafetyAction] = {}

        # ── Lifetime stats ─────────────────────────────────────────────
        self._task_stats_lock = threading.Lock()
        self.total_tasks_started: int = 0
        self.total_tasks_succeeded: int = 0
        self.total_tasks_failed: int = 0

        # Session managers now live on SensorRuntime (per-sensor).

    # ── Runtime registration ───────────────────────────────────────────

    def register_runtime(self, runtime: SensorRuntime) -> None:
        """Add a SensorRuntime and wire its dispatcher back-reference."""
        sid = runtime.sensor_id
        self._runtimes[sid] = runtime
        runtime.set_dispatcher(self)
        with self.heap_lock:
            self._sensor_heaps.setdefault(sid, [])
            self._sensor_task_ids.setdefault(sid, set())
            self._sensor_task_dicts.setdefault(sid, {})
            self._current_task_ids.setdefault(sid, None)

    def get_runtime(self, sensor_id: str) -> SensorRuntime | None:
        """Look up a registered runtime by *sensor_id*."""
        return self._runtimes.get(sensor_id)

    def iter_runtimes(self) -> list[SensorRuntime]:
        """Public snapshot of all registered runtimes.

        Returns a freshly copied list so callers can iterate safely without
        reaching into :attr:`_runtimes`.
        """
        return list(self._runtimes.values())

    def telescope_runtimes(self) -> list[SensorRuntime]:
        """All telescope runtimes that have a ``citra_record`` wired up."""
        return [
            rt
            for rt in self._runtimes.values()
            if rt.sensor_type == "telescope" and getattr(rt.sensor, "citra_record", None)
        ]

    def _runtime_for_task(self, task: Task) -> SensorRuntime | None:
        """Resolve the runtime responsible for this task.

        Matches by config-level ``sensor_id`` first, then by API-side
        telescope ID (from ``citra_record.id``). When ``task.sensor_id`` is
        unset and only one runtime of the requested type exists, that
        single runtime is used — this keeps single-sensor deployments and
        migrations from breaking without silently misrouting on multi-rig
        sites.

        Returns ``None`` (task is dropped) if no unambiguous match exists.
        """
        sid = task.sensor_id
        tid = getattr(task, "id", "?")
        if sid:
            if sid in self._runtimes:
                rt = self._runtimes[sid]
                # Defensive: if the local sensor_id happens to collide with
                # a Citra-side id of a different modality, refuse the route
                # rather than handing a telescope task to a streaming sensor
                # whose ``submit_task`` raises NotImplementedError.  Today
                # only telescope tasks flow through here; future modalities
                # will widen this comparison.
                if rt.sensor_type == task.sensor_type:
                    return rt
                self.logger.warning(
                    "Task %s claims sensor_id=%r but that runtime is type %r, not %r; dropping",
                    tid,
                    sid,
                    rt.sensor_type,
                    task.sensor_type,
                )
                self._notify_dropped_task(
                    f"Task dropped — sensor_id {sid!r} resolves to a "
                    f"{rt.sensor_type} runtime but the task is for {task.sensor_type}",
                    toast_id=f"task-drop:type-mismatch:{sid}",
                )
                return None
            for rt in self._runtimes.values():
                rec = getattr(rt.sensor, "citra_record", None)
                if rec and rec.get("id") == sid:
                    return rt
            self.logger.warning(
                "Task %s has sensor_id=%r but no matching runtime; dropping",
                tid,
                sid,
            )
            self._notify_dropped_task(
                f"Task dropped — sensor_id {sid!r} has no matching runtime",
                toast_id=f"task-drop:unknown-sensor:{sid}",
            )
            return None

        # No sensor_id on the task. Fall back to the single runtime of the
        # requested type if (and only if) exactly one is registered.
        candidates = [rt for rt in self._runtimes.values() if rt.sensor_type == task.sensor_type]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            self.logger.warning(
                "Task %s has no sensor_id but %d %s runtimes are registered; dropping",
                tid,
                len(candidates),
                task.sensor_type,
            )
            self._notify_dropped_task(
                f"Task dropped — no sensor_id on {task.sensor_type} task but "
                f"{len(candidates)} candidates exist (check Citra task payload)",
                toast_id=f"task-drop:missing-sensor:{task.sensor_type}",
            )
        else:
            # No candidates: either nothing of this type is registered or
            # the site has no runtimes at all.  Either way, the task can't
            # be routed — make that visible in the log so operators can
            # correlate dropped tasks with config mistakes (wrong sensor
            # type on the API side, no telescope registered, etc.).
            self.logger.warning(
                "Task %s (sensor_type=%r) has no sensor_id and no matching runtime; dropping",
                tid,
                task.sensor_type,
            )
            self._notify_dropped_task(
                f"Task dropped — no {task.sensor_type} sensor registered to run it",
                toast_id=f"task-drop:no-runtime:{task.sensor_type}",
            )
        return None

    def _notify_dropped_task(self, message: str, *, toast_id: str | None = None) -> None:
        """Best-effort operator toast for dropped/unroutable tasks."""
        if not self.on_toast:
            return
        try:
            self.on_toast(message, "warning", toast_id)
        except Exception:
            self.logger.debug("Dropped-task toast failed", exc_info=True)

    @property
    def automated_scheduling(self) -> bool:
        """True if *any* telescope has automated scheduling enabled.

        ``automatedScheduling`` is a **per-sensor** flag on the Citra-side
        telescope record — there is no site-level counterpart. This
        property is an explicit "any of them" summary used by the
        self-tasking gate and the monitoring UI; callers that need a
        specific rig's state should read
        ``runtime.sensor.citra_record["automatedScheduling"]`` directly
        and toggles go through
        ``PATCH /api/sensors/{sensor_id}/scheduling`` (which updates
        exactly one rig at a time).
        """
        for rt in self._runtimes.values():
            if rt.sensor_type == "telescope":
                rec = getattr(rt.sensor, "citra_record", None)
                if rec and rec.get("automatedScheduling", False):
                    return True
        return False

    # ── Queue helpers (facade) ─────────────────────────────────────────

    def are_queues_idle(self) -> bool:
        return all(rt.are_queues_idle() for rt in self._runtimes.values())

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
            for td in self._sensor_task_dicts.values():
                td.pop(task_id, None)

    def drop_scheduled_task(self, task_id: str) -> bool:
        """Fully evict a scheduled (not-yet-running) task.

        Returns True when something was actually removed.
        """
        with self._stage_lock:
            self.imaging_tasks.pop(task_id, None)
            self.processing_tasks.pop(task_id, None)
            self.uploading_tasks.pop(task_id, None)

        with self.heap_lock:
            removed = False
            for sid in list(self._sensor_heaps):
                ids = self._sensor_task_ids[sid]
                if task_id in ids:
                    removed = True
                    ids.discard(task_id)
                    self._sensor_task_dicts[sid].pop(task_id, None)
                    self._sensor_heaps[sid] = [e for e in self._sensor_heaps[sid] if e[2] != task_id]
                    heapq.heapify(self._sensor_heaps[sid])
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
                    result["sensor_type"] = task.sensor_type
                    result["sensor_id"] = getattr(task, "sensor_id", None)
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
            return sum(len(h) for h in self._sensor_heaps.values())

    def get_tasks_snapshot(self, exclude_active: bool = False) -> list[Task]:
        if exclude_active:
            with self._stage_lock:
                active_ids = set(self.imaging_tasks) | set(self.processing_tasks) | set(self.uploading_tasks)
        else:
            active_ids = set()

        with self.heap_lock:
            tasks = []
            for heap in self._sensor_heaps.values():
                tasks.extend(task for _start, _stop, task_id, task in heap if task_id not in active_ids)
            return tasks

    def get_task_by_id(self, task_id: str) -> Task | None:
        with self.heap_lock:
            for td in self._sensor_task_dicts.values():
                if task_id in td:
                    return td[task_id]
            return None

    @property
    def current_task_ids(self) -> dict[str, str]:
        """Mapping of ``sensor_id -> currently-executing task id``.

        Excludes sensors that are idle. Used by the web layer to decide
        per-row ``isActive`` flags and to reject cancel requests for any
        task currently executing on *any* sensor.
        """
        return {sid: tid for sid, tid in self._current_task_ids.items() if tid}

    @property
    def task_dict(self) -> dict[str, Task]:
        """Flat aggregate of all per-sensor task dicts (read-only view for snapshots)."""
        merged: dict[str, Task] = {}
        for td in self._sensor_task_dicts.values():
            merged.update(td)
        return merged

    def restore_task_dict(self, tasks: dict[str, Task]) -> None:
        """Re-insert tasks into the correct per-sensor heaps (used on config reload)."""
        with self.heap_lock:
            for task_id, task in tasks.items():
                rt = self._runtime_for_task(task)
                if rt is None:
                    continue
                sid = rt.sensor_id
                if task_id in self._sensor_task_ids.get(sid, set()):
                    continue
                self._sensor_task_dicts.setdefault(sid, {})[task_id] = task
                self._sensor_task_ids.setdefault(sid, set()).add(task_id)
                start = getattr(task, "taskStart", "") or ""
                stop = getattr(task, "taskStop", "") or ""
                heapq.heappush(
                    self._sensor_heaps.setdefault(sid, []),
                    (start, stop, task_id, task),
                )

    # ── Poll loop ──────────────────────────────────────────────────────

    def _all_task_ids(self) -> set[str]:
        """Union of all task IDs across all sensor heaps."""
        ids: set[str] = set()
        for s in self._sensor_task_ids.values():
            ids |= s
        return ids

    def poll_tasks(self) -> None:
        # Telescope sensors drive Citra's task API; streaming sensors
        # (passive_radar today, RF tomorrow) don't participate in the
        # poll.  On a radar-only site there are legitimately no
        # telescope runtimes — return early instead of logging an
        # error and letting the thread die silently.
        if not self.telescope_runtimes():
            self.logger.info(
                "poll_tasks: no telescope (on-demand) runtimes — radar-only deployment, skipping poll loop"
            )
            return

        while not self._stop_event.is_set():
            try:
                for rt in self._runtimes.values():
                    # Observing-session and self-tasking both reach into
                    # the adapter (location service, autofocus, etc.) —
                    # don't fire them for sensors whose connect hasn't
                    # completed yet.
                    if not rt.is_ready:
                        continue
                    if rt.observing_session_manager:
                        rt.observing_session_manager.update()
                    if rt.self_tasking_manager:
                        rt.self_tasking_manager.check_and_request()

                if self.elset_cache:
                    interval_hours = self.settings.elset_refresh_interval_hours
                    self.elset_cache.refresh_if_stale(self.api_client, self.logger, interval_hours=interval_hours)
                self._report_online()
                now_iso = datetime.now(timezone.utc).isoformat()
                telescope_rts = self.telescope_runtimes()
                if not telescope_rts:
                    self._stop_event.wait(TASK_POLL_INTERVAL_SECONDS)
                    continue

                tasks: list = []
                for trt in telescope_rts:
                    rec = cast("TelescopeSensor", trt.sensor).citra_record
                    assert rec is not None
                    batch = self.api_client.get_telescope_tasks(
                        rec["id"],
                        statuses=["Pending", "Scheduled"],
                        task_stop_after=now_iso,
                    )
                    if batch:
                        tasks.extend(batch)

                if not tasks:
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

                    current_ids = {v for v in self._current_task_ids.values() if v}
                    for sid in list(self._sensor_heaps):
                        heap = self._sensor_heaps[sid]
                        ids = self._sensor_task_ids[sid]
                        dicts = self._sensor_task_dicts[sid]
                        new_heap = []
                        for entry in heap:
                            start_epoch, stop_epoch, tid, task = entry
                            if tid in current_ids or tid in api_task_map:
                                new_heap.append(entry)
                            else:
                                self.logger.info("Removing task %s from queue (cancelled or status changed)", tid)
                                ids.discard(tid)
                                dicts.pop(tid, None)
                                removed += 1
                        if len(new_heap) != len(heap):
                            self._sensor_heaps[sid] = new_heap
                            heapq.heapify(new_heap)

                    all_known = self._all_task_ids()
                    for tid, task in api_task_map.items():
                        if tid in all_known or tid in current_ids or tid in inflight_ids:
                            continue
                        rt = self._runtime_for_task(task)
                        if rt is None:
                            self.logger.warning(
                                "Dropping task %s — no matching runtime (sensor_id=%s)", tid, task.sensor_id
                            )
                            continue
                        target_sid = rt.sensor_id
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
                        heapq.heappush(self._sensor_heaps[target_sid], (start_epoch, stop_epoch, tid, task))
                        self._sensor_task_ids[target_sid].add(tid)
                        self._sensor_task_dicts[target_sid][tid] = task
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
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        statuses = []
        for trt in self.telescope_runtimes():
            rec = cast("TelescopeSensor", trt.sensor).citra_record
            assert rec is not None
            statuses.append({"id": rec["id"], "last_connection_epoch": iso_timestamp})
        if statuses:
            self.api_client.put_telescope_status(statuses)
            self.logger.debug("Reported online status for %d telescope(s) at %s", len(statuses), iso_timestamp)

    # ── Task runner ────────────────────────────────────────────────────

    def task_runner(self) -> None:
        while not self._stop_event.is_set():
            for rt in self._runtimes.values():
                # Maintenance managers (autofocus, alignment, homing,
                # calibration) all touch the hardware adapter; skip them
                # for sensors whose adapter hasn't connected yet.
                if rt.is_ready:
                    rt.check_maintenance()

            try:
                now = int(time.time())
                completed = 0
                for sid, rt in self._runtimes.items():
                    # Skip sensors whose hardware-init pipeline hasn't
                    # reached ``connected`` yet (still pending /
                    # connecting / failed / timed_out).  The dispatcher
                    # registers every runtime up front so per-sensor
                    # status surfaces, but a runtime without a live
                    # adapter can't run tasks.
                    if not rt.is_ready:
                        continue
                    # Per-runtime safety gate.  Evaluating scope-by-scope
                    # means one sensor's cable unwind (or any other
                    # sensor-scoped QUEUE_STOP/EMERGENCY) only freezes
                    # itself — siblings keep pulling from their heaps.
                    # Site-level checks still veto every sensor because
                    # ``_checks_for(sensor_id)`` always includes them.
                    if self._evaluate_safety_for(sid):
                        continue
                    if rt.paused:
                        continue
                    if rt.is_maintenance_blocking():
                        continue
                    if rt.observing_session_manager and rt.observing_session_manager.is_winding_down():
                        continue
                    if not rt.acquisition_queue.is_idle():
                        continue
                    if rt.is_focus_or_alignment_active():
                        continue

                    with self.heap_lock:
                        heap = self._sensor_heaps.get(sid)
                        if not heap or heap[0][0] > now:
                            continue
                        _start_epoch, stop_epoch, tid, task = heap[0]
                        if stop_epoch and stop_epoch < now:
                            heapq.heappop(heap)
                            self._sensor_task_ids[sid].discard(tid)
                            self._sensor_task_dicts[sid].pop(tid, None)
                            self.logger.info("Skipping expired task %s (window closed)", tid)
                            continue
                        heapq.heappop(heap)
                        self._sensor_task_ids[sid].discard(tid)
                        self.logger.info("Starting task %s at %s: %s", tid, datetime.now().isoformat(), task)
                        self._current_task_ids[sid] = tid

                    self.update_task_stage(tid, "imaging")
                    task.set_status_msg("Queued for imaging...")

                    target_sid = sid

                    def on_imaging_complete(task_id: str, success: bool, _sid: str = target_sid) -> None:
                        if success:
                            with self.heap_lock:
                                self._current_task_ids[_sid] = None
                            self.logger.info("Completed imaging task %s successfully.", task_id)
                        else:
                            self.logger.error("Imaging task %s permanently failed.", task_id)
                            with self.heap_lock:
                                self._current_task_ids[_sid] = None
                            self.record_task_failed()
                            self.remove_task_from_all_stages(task_id)

                    rt.submit_task(task, on_imaging_complete)
                    completed += 1

                if completed > 0:
                    self.logger.info(self._heap_summary("Started tasks"))
            except Exception as e:
                self.logger.error("Exception in task_runner loop: %s", e, exc_info=True)
                time.sleep(5)

            self._stop_event.wait(1)

    # ── Safety ─────────────────────────────────────────────────────────

    def _evaluate_safety_for(self, sensor_id: str) -> bool:
        """Per-runtime safety gate.  Returns True if *sensor_id* should yield.

        Only site-level checks plus the one sensor's own checks participate,
        so a ``QUEUE_STOP`` triggered on a different sensor (e.g. a cable
        unwind in progress on a sibling scope) doesn't freeze this runtime.
        Site-level checks (``DiskSpaceCheck``, ``TimeHealthCheck``,
        ``OperatorStopCheck``) still veto every sensor because they carry
        ``sensor_id=None`` and always pass the filter.
        """
        if not self.safety_monitor:
            return False

        try:
            action, triggered_check = self.safety_monitor.evaluate(sensor_id=sensor_id)
        except Exception:
            self.logger.error("Safety monitor evaluation failed for %s — yielding", sensor_id, exc_info=True)
            return True

        scope_key = self._safety_scope_key(triggered_check)
        if action == SafetyAction.EMERGENCY:
            self._handle_emergency(triggered_check, scope_key)
        elif action == SafetyAction.QUEUE_STOP:
            self._handle_queue_stop(triggered_check, scope_key)

        self._last_safety_action_by_key[scope_key] = action
        return action in (SafetyAction.EMERGENCY, SafetyAction.QUEUE_STOP)

    def _safety_scope_key(self, triggered_check: Any) -> str:
        """Resolve the edge-detection key for a triggered check.

        Sensor-scoped checks key off their ``sensor_id``; site-level checks
        (and the "no trigger" ``SAFE`` case) share ``_SITE_SAFETY_KEY`` so
        per-scope edge tracking stays meaningful without leaking into the
        site bucket.
        """
        if triggered_check is None:
            return _SITE_SAFETY_KEY
        raw = getattr(triggered_check, "sensor_id", None)
        if isinstance(raw, str) and raw:
            return raw
        return _SITE_SAFETY_KEY

    def _handle_emergency(self, triggered_check: Any, scope_key: str) -> None:
        """Abort in-flight imaging and fire corrective action on EMERGENCY.

        Sensor-scoped triggers only abort and clear the triggering sensor;
        site-level triggers keep the historical "blast every adapter"
        behavior.  The corrective action and the operator toast only fire
        on the edge (first cycle in which EMERGENCY is reported for this
        scope) so repeated evaluation passes don't spam logs or retry the
        action.
        """
        trigger_sid = scope_key if scope_key != _SITE_SAFETY_KEY else None

        # Edge-detect *before* any side effect.  Without this guard, a
        # persistent EMERGENCY fires ``abort_slew()`` every poll cycle —
        # and because ``_evaluate_safety_for`` is now called once per
        # runtime per cycle, a site-level EMERGENCY in a fleet of N
        # sensors would rain N×N abort calls per cycle on every adapter.
        # Everything below (abort, queue-clear, logger.critical, toast,
        # execute_action) belongs to the "we just transitioned into
        # EMERGENCY" bucket, so it all sits under the same guard.
        is_new = self._last_safety_action_by_key.get(scope_key) != SafetyAction.EMERGENCY
        if not is_new:
            return

        if trigger_sid is not None:
            rt = self._runtimes.get(trigger_sid)
            runtimes_to_abort = [rt] if rt is not None else []
        else:
            runtimes_to_abort = list(self._runtimes.values())
        for rt in runtimes_to_abort:
            adapter = getattr(rt, "hardware_adapter", None)
            if adapter:
                try:
                    adapter.abort_slew()
                except Exception:
                    pass

        self.clear_pending_tasks(sensor_id=trigger_sid)
        trigger_name = triggered_check.name if triggered_check else "unknown"
        self.logger.critical(
            "EMERGENCY — cancelled in-flight imaging (trigger: %s, sensor: %s)",
            trigger_name,
            trigger_sid or "site",
        )
        if self.on_toast:
            where = f" on {trigger_sid}" if trigger_sid else ""
            toast_id = f"safety-emergency:{trigger_sid}" if trigger_sid else "safety-emergency"
            self.on_toast(
                f"Safety EMERGENCY: {trigger_name}{where} — imaging cancelled",
                "danger",
                toast_id,
            )
        if triggered_check:
            try:
                triggered_check.execute_action()
            except Exception:
                self.logger.error("Safety corrective action failed", exc_info=True)

    def _handle_queue_stop(self, triggered_check: Any, scope_key: str) -> None:
        """Fire the triggered check's corrective action once its queue is idle.

        Per-sensor isolation: a sensor-scoped check only waits for *its*
        acquisition queue to go idle before running its corrective action.
        Site-level checks still gate on every runtime being idle so they
        don't yank hardware out from under another rig mid-exposure.
        """
        if triggered_check is None:
            return

        scoped_sid = scope_key if scope_key != _SITE_SAFETY_KEY else None
        if scoped_sid is not None:
            rt = self._runtimes.get(scoped_sid)
            idle = rt is not None and rt.acquisition_queue.is_idle()
        else:
            idle = all(rt.acquisition_queue.is_idle() for rt in self._runtimes.values())
        if not idle:
            return

        is_new = self._last_safety_action_by_key.get(scope_key) != SafetyAction.QUEUE_STOP
        if is_new:
            self.logger.warning(
                "Executing safety action from %r (queue idle, scope=%s)",
                triggered_check.name,
                scoped_sid or "site",
            )
        try:
            triggered_check.execute_action()
        except Exception:
            if is_new:
                self.logger.error("Safety corrective action failed", exc_info=True)

    # ── Pending task management ────────────────────────────────────────

    def clear_pending_tasks(self, sensor_id: str | None = None) -> int:
        """Cancel in-flight imaging and clear stage tracking.

        Without a ``sensor_id``, every runtime's acquisition queue is
        drained and ``self.imaging_tasks`` is wiped (site-level EMERGENCY
        path).  With one, only that sensor's queue is drained and only its
        known in-flight / scheduled task ids are pruned from
        ``imaging_tasks`` so sibling sensors keep running.

        Returns the number of queued imaging items drained.
        """
        with self._stage_lock:
            cleared = 0
            if sensor_id is None:
                for rt in self._runtimes.values():
                    cleared += rt.acquisition_queue.clear()
                self.imaging_tasks.clear()
                return cleared
            rt = self._runtimes.get(sensor_id)
            if rt is not None:
                cleared += rt.acquisition_queue.clear()
            owned_ids: set[str] = set()
            current_tid = self._current_task_ids.get(sensor_id)
            if current_tid:
                owned_ids.add(current_tid)
            owned_ids.update(self._sensor_task_dicts.get(sensor_id, {}).keys())
            for tid in owned_ids:
                self.imaging_tasks.pop(tid, None)
            return cleared

    def pause_sensor(self, sensor_id: str | None = None) -> None:
        """Pause one sensor (by id) or all sensors."""
        for sid, rt in self._runtimes.items():
            if sensor_id is None or sid == sensor_id:
                rt.set_paused(True)
                self.logger.info("Task processing paused for sensor %s", sid)

    def resume_sensor(self, sensor_id: str | None = None) -> None:
        """Resume one sensor (by id) or all sensors."""
        for sid, rt in self._runtimes.items():
            if sensor_id is None or sid == sensor_id:
                rt.set_paused(False)
                self.logger.info("Task processing resumed for sensor %s", sid)

    def is_processing_active(self) -> bool:
        """True if any sensor is not paused."""
        return any(not rt.paused for rt in self._runtimes.values())

    # ── Diagnostics ────────────────────────────────────────────────────

    def _heap_summary(self, event: str) -> str:
        with self.heap_lock:
            total = sum(len(h) for h in self._sensor_heaps.values())
            summary = f"{event}: {total} tasks in queue. "
            all_entries = []
            for heap in self._sensor_heaps.values():
                all_entries.extend(heap[:3])
            all_entries.sort()
            next_tasks = [
                f"{tid} at {datetime.fromtimestamp(start).isoformat()}" for start, stop, tid, _ in all_entries[:3]
            ]
            if total > 3:
                next_tasks.append(f"... ({total - 3} more)")
            active = [tid for tid in self._current_task_ids.values() if tid]
            if active:
                summary += f"Current: {', '.join(active)}. "
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
        """Stop runtimes and worker threads.

        Tolerant of a never-started dispatcher: ``_initialize_components``
        constructs a ``TaskDispatcher`` *before* calling ``.start()``, so
        if init fails in between (e.g. a sensor adapter blowup), the next
        ``reload_configuration`` will hit this method on a dispatcher
        whose ``poll_thread`` / ``runner_thread`` attributes were never
        created. Silently no-op instead of raising ``AttributeError``,
        which would otherwise mask the real init error behind an obscure
        cleanup crash.
        """
        self._stop_event.set()

        self.logger.info("Stopping sensor runtimes...")
        for rt in self._runtimes.values():
            rt.stop()

        poll_thread = getattr(self, "poll_thread", None)
        if poll_thread is not None:
            poll_thread.join()
        runner_thread = getattr(self, "runner_thread", None)
        if runner_thread is not None:
            runner_thread.join()
