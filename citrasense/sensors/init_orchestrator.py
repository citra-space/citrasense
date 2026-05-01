"""Site-level coordinator for parallel/async sensor initialization.

This module owns the orchestration plumbing that grew out of issue
#339: a per-sensor watchdog connect, a thread-pool that fans out
``adapter.connect()`` calls, and the per-sensor reconnect / disconnect
API the web layer calls into.

The orchestrator is intentionally generic — it never imports any
modality-specific builder.  Instead it holds an opaque
:class:`~citrasense.sensors.runtime_builder.RuntimeBuilder` per sensor
and asks each builder to run its own ``connect_post_wiring`` step
after the watchdog confirms the adapter is live.

Threading model:

* One **init thread** (``sensor-init-orchestrator``, daemon) drains the
  initial fan-out futures so log lines like "Sensor init fan-out
  complete: N/M connected" appear once everyone has resolved.
* One **executor** (``ThreadPoolExecutor``, named ``sensor-init``) runs
  per-sensor connect workers.  Reconnects re-use the same pool.
* Per-sensor connect calls live inside a tiny **nested executor** so
  ``Future.result(timeout=...)`` can bound a hung ``connect()``.

A reconnect request that lands while a previous connect is still in
flight is rejected with ``(False, "Reconnect already in flight ...")``
so two adapter inits never race.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

    from citrasense.api.citra_api_client import AbstractCitraApiClient
    from citrasense.location import LocationService
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.sensors.runtime_builder import RuntimeBuilder
    from citrasense.sensors.sensor_manager import SensorManager
    from citrasense.sensors.sensor_runtime import SensorRuntime
    from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor
    from citrasense.settings.citrasense_settings import CitraSenseSettings
    from citrasense.tasks.task_dispatcher import TaskDispatcher
    from citrasense.web.server import CitraSenseWebServer


def resolve_canonical_ground_station(
    *,
    sensor_manager: SensorManager,
    settings: CitraSenseSettings,
    api_client: AbstractCitraApiClient,
    logger: logging.Logger,
    location_service: LocationService | None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Pre-flight: pick the canonical ground_station deterministically.

    Walks every telescope sensor in lexical ``sensor_id`` order,
    fetches its ``citra_record`` + ``ground_station`` from the API,
    stamps the record onto the sensor, and selects the first
    successfully-fetched ground station as the site's canonical one.
    Subsequent sensors that report a *different* ``groundStationId``
    cause the boot to fail with the same message the old in-loop
    check used.

    Stamping ``citra_record`` here means the dispatcher can poll for
    tasks for a sensor whose adapter is still mid-connect — the
    scheduler doesn't need a live adapter to filter by API id.

    Returns a tuple ``(ok, error, ground_station)``.  When ``ok`` is
    True, ``ground_station`` may still be ``None`` for non-telescope
    deployments (radar-only, allsky-only).
    """
    telescopes = sorted(
        sensor_manager.iter_by_type("telescope"),
        key=lambda s: s.sensor_id,
    )
    if not telescopes:
        return True, None, None

    ground_station: dict[str, Any] | None = None
    for ts in telescopes:
        sensor_cfg = settings.get_sensor_config(ts.sensor_id)
        if sensor_cfg is None:
            err = f"No SensorConfig for telescope sensor {ts.sensor_id!r}"
            logger.error(err)
            return False, err, None
        citra_sensor_id = sensor_cfg.citra_sensor_id
        try:
            citra_record = api_client.get_telescope(citra_sensor_id)
        except Exception as exc:
            err = f"API error fetching telescope record for {citra_sensor_id!r}: {exc}"
            logger.error(err)
            return False, err, None
        if not citra_record:
            err = f"Sensor ID '{citra_sensor_id}' is not valid on the server."
            logger.error(err)
            return False, err, None
        cast("TelescopeSensor", ts).citra_record = citra_record

        try:
            gs = api_client.get_ground_station(citra_record["groundStationId"])
        except Exception as exc:
            err = f"API error fetching ground station for sensor {ts.sensor_id!r}: {exc}"
            logger.error(err)
            return False, err, None
        if not gs:
            err = "Could not get ground station info from the server."
            logger.error(err)
            return False, err, None

        if ground_station is None:
            ground_station = gs
        elif ground_station.get("id") != gs.get("id"):
            err = (
                f"Sensor '{citra_sensor_id}' is registered to ground station "
                f"{gs.get('id')!r} ({gs.get('name', '?')}), "
                f"but the daemon is already serving ground station "
                f"{ground_station.get('id')!r} "
                f"({ground_station.get('name', '?')}). "
                "Multi-ground-station deployments are not supported — run a "
                "separate CitraSense daemon per site."
            )
            logger.error(err)
            return False, err, None

    if ground_station is not None and location_service:
        location_service.set_ground_station(ground_station)
    return True, None, ground_station


class SensorInitOrchestrator:
    """Drives the parallel/async sensor init pipeline.

    See module docstring for the threading model and lifecycle.

    The orchestrator holds:

    * An executor that runs per-sensor connect workers.
    * A futures map keyed by ``sensor_id`` for in-flight connects.
    * A ``builders`` map keyed by ``sensor_id`` so reconnect requests
      can re-run the same modality builder's
      :meth:`~citrasense.sensors.runtime_builder.RuntimeBuilder.connect_post_wiring`.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        web_server: CitraSenseWebServer | None,
        sensor_manager: SensorManager,
        settings: CitraSenseSettings,
        task_dispatcher: TaskDispatcher,
    ) -> None:
        self.logger = logger
        self.web_server = web_server
        self.sensor_manager = sensor_manager
        self.settings = settings
        self.task_dispatcher = task_dispatcher

        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future[None]] = {}
        self._futures_lock = threading.Lock()
        self._init_thread: threading.Thread | None = None

        # ``RuntimeBuilder`` instances keyed by sensor_id.  Populated by
        # :meth:`fan_out`; reconnect lookups re-use them so the same
        # builder runs both the initial connect and any later
        # reconnects.
        self._builders: dict[str, RuntimeBuilder] = {}

    # ── Toast callback factory ───────────────────────────────────────

    def make_init_state_toast_callback(self, sensor_id: str) -> Callable[[str, str | None], None] | None:
        """Return a callback that pushes init-state toasts to the web UI.

        Closes over ``sensor_id`` (and the orchestrator's web server)
        so the runtime doesn't need to know anything about toasts.
        Toasts are deduped per-sensor with
        ``id="sensor-init-{sensor_id}"`` so repeated reconnect attempts
        replace the previous toast instead of stacking.

        - ``connecting`` → info toast ("Reconnecting <sensor>...")
        - ``connected`` → green success toast ("<sensor> connected")
        - ``failed`` → persistent danger toast with the error string
        - ``timed_out`` → persistent danger toast with the timeout
        - ``pending`` is silent (no transition the operator cares about)

        Returns ``None`` when there's no web server configured (headless
        tests / pre-init bring-up) so the runtime can skip the wiring
        entirely instead of installing a no-op callback.
        """
        if self.web_server is None:
            return None

        def _callback(state: str, error: str | None) -> None:
            ws = self.web_server
            if ws is None:
                return
            sensor_label = self._sensor_display_name(sensor_id)
            toast_id = f"sensor-init-{sensor_id}"
            try:
                if state == "connected":
                    ws.send_toast(f"{sensor_label} connected", "success", toast_id)
                elif state == "connecting":
                    ws.send_toast(f"Connecting {sensor_label}…", "info", toast_id)
                elif state == "failed":
                    ws.send_toast(
                        f"{sensor_label} connect failed: {error or 'unknown error'}",
                        "danger",
                        toast_id,
                    )
                elif state == "timed_out":
                    ws.send_toast(
                        f"{sensor_label} connect timed out: {error or 'no response'}",
                        "danger",
                        toast_id,
                    )
            except Exception:
                self.logger.debug("init-state toast failed for %s", sensor_id, exc_info=True)

        return _callback

    def _sensor_display_name(self, sensor_id: str) -> str:
        """Best-effort human-readable name for toasts/log lines."""
        s = self.sensor_manager.get_sensor(sensor_id)
        if s is None:
            return sensor_id
        return getattr(s, "name", None) or sensor_id

    # ── Lifecycle ────────────────────────────────────────────────────

    def fan_out(self, pairs: list[tuple[SensorRuntime, RuntimeBuilder]]) -> None:
        """Submit one connect future per ``(runtime, builder)`` pair.

        Idempotent — safe to call again on reload because the previous
        executor is shut down before each ``_initialize_components``
        cycle.

        Returns immediately; the per-sensor connects keep running on
        the executor's worker threads.  Spawns a daemon orchestrator
        thread that joins on each future just to log when the fan-out
        completes.
        """
        if not pairs:
            self.logger.info("No sensor runtimes registered; skipping connect fan-out")
            return

        # Cap the pool: per-sensor parallelism above ~8 doesn't help
        # since most sensors block on serial / USB / HTTP I/O, and each
        # adapter probe holds onto a few file descriptors.  Use one
        # worker per sensor up to 8.
        max_workers = max(1, min(len(pairs), 8))
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sensor-init")

        with self._futures_lock:
            self._futures.clear()
            self._builders.clear()
            for rt, builder in pairs:
                self._builders[rt.sensor_id] = builder
                fut = self._executor.submit(self._run_sensor_init, rt, builder)
                self._futures[rt.sensor_id] = fut

        self._init_thread = threading.Thread(
            target=self._await_fan_out,
            name="sensor-init-orchestrator",
            daemon=True,
        )
        self._init_thread.start()

    def _await_fan_out(self) -> None:
        """Drain the initial fan-out futures and log the summary.

        Runs on the ``sensor-init-orchestrator`` daemon thread.  The
        per-sensor watchdog timeout lives inside ``_run_sensor_init``,
        not here — this method only joins.
        """
        with self._futures_lock:
            futures = list(self._futures.values())
        for fut in futures:
            try:
                fut.result()
            except Exception:
                # _run_sensor_init swallows its own exceptions and
                # surfaces them via init_state; anything bubbling up
                # here is a genuine programming error worth logging.
                self.logger.error("sensor-init worker raised unexpectedly", exc_info=True)
        # Don't tear the executor down here: per-sensor reconnects
        # submit fresh futures into the same pool.  shutdown() and
        # the next fan_out cycle handle teardown.
        connected = sum(1 for rt in self.task_dispatcher.iter_runtimes() if rt.is_ready)
        total = len(futures)
        self.logger.info("Sensor init fan-out complete: %d/%d sensor(s) connected", connected, total)

    def shutdown(self) -> None:
        """Tear down the executor.  Idempotent.

        ``cancel_futures=True`` only cancels queued work; in-flight
        connects keep running on their worker threads but their results
        are discarded.  Daemon threads in the pool die with the
        process anyway.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        with self._futures_lock:
            self._futures.clear()

    # ── Per-sensor connect/disconnect API (web layer entry points) ───

    def request_reconnect(self, sensor_id: str) -> tuple[bool, str | None]:
        """Submit a per-sensor reconnect to the same async init worker.

        Returns ``(True, None)`` once the reconnect is queued — the HTTP
        layer can immediately return ``202 Accepted`` and the operator
        watches the toast / status badge for the result.  Returns
        ``(False, msg)`` when the reconnect can't be queued (sensor
        unknown, no builder registered, or another reconnect already
        in flight).
        """
        runtime = self.task_dispatcher.get_runtime(sensor_id)
        if runtime is None:
            return False, f"Unknown sensor: {sensor_id}"
        builder = self._builders.get(sensor_id)
        if builder is None:
            return False, f"No runtime builder registered for {sensor_id}"

        with self._futures_lock:
            existing = self._futures.get(sensor_id)
            if existing is not None and not existing.done():
                return False, f"Reconnect already in flight for {sensor_id}"

            if self._executor is None:
                # No live executor (rare: someone hit reconnect before
                # fan_out finished, or after shutdown).  Spin one up
                # just for this reconnect.
                self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sensor-init")

            # Best-effort disconnect first so the reconnect picks up a
            # fresh adapter state.  Ignore errors — we're about to
            # connect anyway, and a stale adapter that won't disconnect
            # cleanly will surface in the connect step.
            try:
                runtime.sensor.disconnect()
            except Exception:
                self.logger.debug("Pre-reconnect disconnect of %s raised", sensor_id, exc_info=True)
            runtime.mark_disconnected()

            fut = self._executor.submit(self._run_sensor_init, runtime, builder)
            self._futures[sensor_id] = fut

        return True, None

    def request_disconnect(self, sensor_id: str) -> tuple[bool, str | None]:
        """Disconnect a sensor's hardware adapter and reset its init_state.

        Synchronous in spirit (a clean disconnect should be quick), but
        guarded against blocking — we still log + return cleanly even
        if the adapter's ``disconnect()`` takes a moment.
        """
        runtime = self.task_dispatcher.get_runtime(sensor_id)
        if runtime is None:
            return False, f"Unknown sensor: {sensor_id}"
        try:
            runtime.sensor.disconnect()
        except Exception as exc:
            self.logger.error("Disconnect failed for %s: %s", sensor_id, exc, exc_info=True)
            return False, str(exc)
        runtime.mark_disconnected()
        return True, None

    # ── Worker ───────────────────────────────────────────────────────

    def _run_sensor_init(self, runtime: SensorRuntime, builder: RuntimeBuilder) -> None:
        """Worker: drive one sensor's hardware connect with a watchdog timeout.

        Runs on a ``sensor-init`` thread-pool worker.  Wraps the
        adapter's ``connect()`` in :class:`Future.result(timeout=N)`
        via a *nested* executor so a hung adapter blows the deadline
        cleanly without dragging down the orchestrator thread.

        State transitions:

        * ``pending`` → ``connecting`` (immediate, before connect).
        * ``connected`` on success.
        * ``failed`` if ``connect()`` returned False or raised.
        * ``timed_out`` if the watchdog deadline was exceeded.

        This method swallows all exceptions so a buggy
        post-connect-wiring step (filter discovery, calibration manager
        wiring) doesn't crash the worker pool.  The error is captured
        on the runtime instead.
        """
        sensor = runtime.sensor
        sensor_id = runtime.sensor_id
        sensor_cfg = self.settings.get_sensor_config(sensor_id)
        timeout = sensor_cfg.connect_timeout_seconds if sensor_cfg else 60.0

        runtime.mark_connecting()

        # Run adapter.connect() in a tiny nested executor so we can
        # bound it cleanly with Future.result(timeout=...).  Calling
        # sensor.connect() directly here would just block this worker
        # for ``timeout`` seconds — fine for a single hung sensor but
        # we'd lose the ability to detect the hang for status display.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"connect-{sensor_id}") as inner:
            connect_future = inner.submit(self._safe_connect, sensor)
            try:
                ok = connect_future.result(timeout=timeout)
            except FuturesTimeout:
                err = f"connect timed out after {timeout:.0f}s"
                self.logger.warning("Sensor %s: %s", sensor_id, err)
                runtime.mark_timed_out(err)
                # Don't wait for the inner executor to drain — the hung
                # connect may genuinely never return.  Daemon threads
                # are reclaimed by the OS on shutdown.
                inner.shutdown(wait=False, cancel_futures=False)
                return
            except Exception as exc:
                err = f"connect raised: {exc}"
                self.logger.error("Sensor %s: %s", sensor_id, err, exc_info=True)
                runtime.mark_failed(err)
                return

        if not ok:
            err = f"adapter.connect() returned False ({type(getattr(sensor, 'adapter', sensor)).__name__})"
            self.logger.error("Sensor %s: %s", sensor_id, err)
            runtime.mark_failed(err)
            return

        try:
            builder.connect_post_wiring(runtime)
        except Exception as exc:
            err = f"post-connect wiring failed: {exc}"
            self.logger.error("Sensor %s: %s", sensor_id, err, exc_info=True)
            runtime.mark_failed(err)
            return

        runtime.mark_connected()

    @staticmethod
    def _safe_connect(sensor: AbstractSensor) -> bool:
        """Adapter for ``sensor.connect()`` that turns exceptions into False+log.

        The watchdog wrapper in :meth:`_run_sensor_init` distinguishes
        timeouts from raises by catching ``FuturesTimeout`` separately,
        so we don't normalize raises to False here — exceptions are
        surfaced verbatim to the caller for richer error messages.
        """
        return bool(sensor.connect())
