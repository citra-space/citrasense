from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from citrasense.sensors.sensor_runtime import SensorRuntime
    from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor

from citrasense.analysis.retention import cleanup_previews, cleanup_processing_output
from citrasense.analysis.task_index import TaskIndex
from citrasense.api.citra_api_client import AbstractCitraApiClient, CitraApiClient
from citrasense.api.dummy_api_client import DummyApiClient
from citrasense.astro.elset_cache import ElsetCache
from citrasense.catalogs.apass_catalog import ApassCatalog
from citrasense.hardware.filter_sync import sync_filters_to_backend
from citrasense.location import LocationService
from citrasense.logging import CITRASENSE_LOGGER
from citrasense.logging._citrasense_logger import setup_file_logging
from citrasense.pipelines.common.pipeline_registry import PipelineRegistry
from citrasense.sensors.bus import InProcessBus
from citrasense.sensors.init_orchestrator import (
    SensorInitOrchestrator,
    resolve_canonical_ground_station,
)
from citrasense.sensors.preview_bus import PreviewBus
from citrasense.sensors.runtime_builder import BuildContext, build_for_sensor
from citrasense.sensors.sensor_manager import SensorManager
from citrasense.settings.citrasense_settings import CitraSenseSettings
from citrasense.startup_checks import check_processor_runtime_deps
from citrasense.tasks.task_dispatcher import TaskDispatcher
from citrasense.time.time_monitor import TimeMonitor
from citrasense.web.server import CitraSenseWebServer


class CitraSenseDaemon:
    def __init__(
        self,
        settings: CitraSenseSettings,
        api_client: AbstractCitraApiClient | None = None,
    ):
        self.settings = settings
        CITRASENSE_LOGGER.setLevel(self.settings.log_level)

        # Setup file logging if enabled
        if self.settings.file_logging_enabled:
            self.settings.directories.ensure_log_directory()
            log_path = self.settings.directories.current_log_path()
            setup_file_logging(log_path, backup_count=self.settings.log_retention_days)
            CITRASENSE_LOGGER.info(f"Logging to file: {log_path}")

        from citrasense.version import format_version_log, get_version_info

        ver_info = get_version_info()
        ver = format_version_log(ver_info)
        CITRASENSE_LOGGER.info("=" * 60)
        CITRASENSE_LOGGER.info(f"CitraSense {ver} starting (PID {os.getpid()})")

        self.api_client = api_client
        self.web_server = None
        self.task_dispatcher = None
        self.time_monitor = None
        self.location_service = None
        # Site-level ground station record returned by the Citra API.  One
        # per daemon instance: CitraSense treats one ground station = one
        # deployed site, and all configured sensors belong to it.  The
        # per-sensor citra_sensor_id on each SensorConfig then selects
        # which API sensor slot that local sensor drives.
        self.ground_station = None
        self.safety_monitor = None
        self.configuration_error: str | None = None
        self.latest_annotated_image_paths: dict[str, str] = {}
        self.preview_bus = PreviewBus()

        self.sensor_bus: InProcessBus = InProcessBus()
        self.sensor_manager: SensorManager | None = None
        self._stop_requested = False
        self._shutdown_done = False

        # ── Sensor init pipeline ───────────────────────────────────────
        # The orchestrator owns the ThreadPoolExecutor + per-sensor
        # watchdog that fans out ``adapter.connect()`` calls in the
        # background (issue #339 — a hung adapter on one sensor no
        # longer wedges every other sensor's startup).  Constructed
        # in ``_initialize_telescopes`` once the dispatcher exists.
        self._init_orchestrator: SensorInitOrchestrator | None = None

        # Cached list of banner-shape dicts for processor-layer missing
        # deps; populated once per startup in _initialize_components.
        self._processor_dep_issues: list[dict] = []

        # Site-level processor registry — metadata only (schema for
        # ``/api/processors``, cosmetic ``status.active_processors`` list).
        # Execution and per-sensor stats live on ``SensorRuntime.processor_registry``.
        self.processor_registry = PipelineRegistry(settings=self.settings, logger=CITRASENSE_LOGGER)

        # Elset cache for satellite matcher (file-backed; warm-start from disk, full refresh at init)
        self.elset_cache = ElsetCache(cache_path=self.settings.directories.elset_cache_path)

        # APASS catalog for local photometry (file-backed; downloaded on first authenticated startup)
        self.apass_catalog = ApassCatalog(
            db_path=self.settings.directories.catalogs_dir / "apass_dr10.db",
            logger=CITRASENSE_LOGGER,
        )

        # Local analysis index — persists pipeline metrics across restarts
        db_path = self.settings.directories.analysis_dir / "task_index.db"
        self.task_index = TaskIndex(db_path)
        # Stamp sensor_id onto any legacy rows recorded before multi-sensor
        # analysis landed.  Idempotent and cheap after convergence (early
        # short-circuit when no NULL rows remain), so safe on every start.
        self.task_index.backfill_sensor_ids(self.settings.directories.processing_dir)
        self._retention_timer: threading.Timer | None = None

        # Note: Work queues and stage tracking now managed by TaskDispatcher + SensorRuntime

        # Create web server instance (always enabled)
        self.web_server = CitraSenseWebServer(daemon=self, host="0.0.0.0", port=self.settings.web_port)

    def _on_annotated_image(self, path: str, sensor_id: str) -> None:
        """Handle a new annotated task image: store path and notify preview bus via URL.

        Uses a lightweight URL notification instead of base64-encoding the
        full image through the WebSocket, keeping the socket clear for
        status/log/task updates on bandwidth-constrained links.

        ``sensor_id`` is required — every callsite must supply the id of
        the sensor that produced the image so per-sensor preview slots
        don't alias under an empty-string key in multi-sensor
        deployments.  Callers that don't know the id (e.g. future
        site-wide captures) should pass the explicit site sentinel, not
        rely on a default.
        """
        if not sensor_id:
            CITRASENSE_LOGGER.warning(
                "Annotated image %s published without sensor_id; dropping to avoid empty-key aliasing",
                path,
            )
            return
        self.latest_annotated_image_paths[sensor_id] = path
        try:
            from urllib.parse import quote

            mtime_ns = Path(path).stat().st_mtime_ns
            url = f"/api/task-preview/latest?sensor_id={quote(sensor_id, safe='')}&t={mtime_ns}"
            self.preview_bus.push_url(url, "task", sensor_id=sensor_id)
        except Exception as e:
            CITRASENSE_LOGGER.warning("Failed to publish annotated image preview for %s: %s", path, e)

    def _refresh_elset_cache_with_retry(self, max_attempts: int = 3) -> None:
        """Force-refresh the elset cache at startup, retrying on failure."""
        assert self.api_client is not None
        for attempt in range(1, max_attempts + 1):
            if self.elset_cache.refresh(self.api_client, logger=CITRASENSE_LOGGER):
                return
            if attempt < max_attempts:
                backoff = 2**attempt
                CITRASENSE_LOGGER.warning(
                    "ElsetCache: refresh attempt %d/%d failed, retrying in %ds",
                    attempt,
                    max_attempts,
                    backoff,
                )
                time.sleep(backoff)
        CITRASENSE_LOGGER.warning(
            "ElsetCache: all %d refresh attempts failed — using cached data if available",
            max_attempts,
        )

    def _initialize_sensors(self) -> None:
        """Build sensors from ``self.settings.sensors`` via the sensor manager."""
        self.settings.directories.ensure_data_directories()
        images_dir = self.settings.directories.images_dir

        try:
            self.sensor_manager = SensorManager.from_configs(
                self.settings.sensors,
                logger=CITRASENSE_LOGGER,
                images_dir=images_dir,
                data_dir=self.settings.directories.data_dir,
                cache_dir=self.settings.directories.cache_dir,
            )
        except ImportError as e:
            adapter_keys = sorted({s.adapter for s in self.settings.sensors}) or ["unknown"]
            adapter_list = ", ".join(adapter_keys)
            CITRASENSE_LOGGER.error(
                "Configured adapter(s) [%s] requested but dependencies not available. Error: %s",
                adapter_list,
                e,
            )
            raise RuntimeError(
                f"Configured adapter(s) [{adapter_list}] require additional dependencies. "
                "Check documentation for installation instructions."
            ) from e

        if len(self.sensor_manager) == 0:
            raise RuntimeError("No sensors configured — at least one sensor is required")

    def _initialize_components(self, reload_settings: bool = False) -> tuple[bool, str | None]:
        """Initialize or reinitialize all components.

        Args:
            reload_settings: If True, reload settings from disk before initializing

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if reload_settings:
                CITRASENSE_LOGGER.info("\u2500" * 60)
                CITRASENSE_LOGGER.info("Configuration reload requested")
                CITRASENSE_LOGGER.info("\u2500" * 60)
                # Reload settings from file (preserving web_port and base_dir)
                new_settings = CitraSenseSettings.load(web_port=self.settings.web_port, base_dir=self.settings.base_dir)
                self.settings = new_settings
                CITRASENSE_LOGGER.setLevel(self.settings.log_level)

                # Ensure web log handler is still attached after logger changes
                if self.web_server:
                    self.web_server.ensure_log_handler()

                # Re-setup file logging if enabled
                if self.settings.file_logging_enabled:
                    self.settings.directories.ensure_log_directory()
                    log_path = self.settings.directories.current_log_path()
                    setup_file_logging(log_path, backup_count=self.settings.log_retention_days)

            # Preserve task metadata across reload
            old_task_dict = {}
            old_imaging_tasks = {}
            old_processing_tasks = {}
            old_uploading_tasks = {}

            # Cleanup existing resources
            # Tear the init executor down first so a reload doesn't pile
            # up zombie connect futures for the previous dispatcher's
            # runtimes.  ``cancel_futures=True`` only cancels queued
            # work; in-flight connects keep running on their thread pool
            # but their results are discarded.
            if self._init_orchestrator is not None:
                self._init_orchestrator.shutdown()
                self._init_orchestrator = None

            if self.task_dispatcher:
                CITRASENSE_LOGGER.info("Stopping existing task manager...")
                # Capture task metadata before destruction
                old_task_dict = dict(self.task_dispatcher.task_dict)
                old_imaging_tasks = dict(self.task_dispatcher.imaging_tasks)
                old_processing_tasks = dict(self.task_dispatcher.processing_tasks)
                old_uploading_tasks = dict(self.task_dispatcher.uploading_tasks)
                self.task_dispatcher.stop()
                self.task_dispatcher = None

            if self.safety_monitor:
                if self.sensor_manager:
                    for s in self.sensor_manager.iter_by_type("telescope"):
                        cast("TelescopeSensor", s).unregister_safety_checks(self.safety_monitor)
                self.safety_monitor.stop_watchdog()
                self.safety_monitor = None

            if self.time_monitor:
                self.time_monitor.stop()
                self.time_monitor = None

            if self.location_service:
                self.location_service.stop()
                self.location_service = None

            if self.sensor_manager:
                self.sensor_manager.disconnect_all()
                self.sensor_manager = None

            # Check if configuration is complete
            if not self.settings.is_configured():
                error_msg = "Configuration incomplete. Please set access token, sensor ID, and hardware adapter."
                CITRASENSE_LOGGER.warning(error_msg)
                self.configuration_error = error_msg
                return False, error_msg

            # Initialize API client
            if self.settings.use_dummy_api:
                CITRASENSE_LOGGER.info("Using DummyApiClient for local testing")
                self.api_client = DummyApiClient(
                    logger=CITRASENSE_LOGGER,
                    cache_path=self.settings.directories.cache_dir / "dummy_tle_cache.json",
                )
            else:
                self.api_client = CitraApiClient(
                    self.settings.host,
                    self.settings.personal_access_token,
                    self.settings.use_ssl,
                    CITRASENSE_LOGGER,
                )

            # Warm-start elset cache from disk (source-aware: discards if API source changed)
            self.elset_cache.load_from_file(expected_source=self.api_client.cache_source_key)
            elset_thread = getattr(self, "_elset_refresh_thread", None)
            if elset_thread is None or not elset_thread.is_alive():
                self._elset_refresh_thread = threading.Thread(
                    target=self._refresh_elset_cache_with_retry,
                    name="elset-refresh",
                    daemon=True,
                )
                self._elset_refresh_thread.start()

            if self.settings.use_local_apass_catalog:
                self.apass_catalog.start_background_download(self.api_client)

            # Initialize sensors via the sensor manager.
            self._initialize_sensors()

            # Check for missing dependencies (non-fatal, just warn)
            assert self.sensor_manager is not None
            for ts in self.sensor_manager.iter_by_type("telescope"):
                adapter = getattr(ts, "adapter", None)
                if adapter is None:
                    continue
                missing_deps = adapter.get_missing_dependencies()
                for dep in missing_deps:
                    CITRASENSE_LOGGER.warning(
                        f"{dep['device_type']} '{dep['device_name']}' missing dependencies: "
                        f"{dep['missing_packages']}. Install with: {dep['install_cmd']}"
                    )

            # Check processor-layer runtime deps (astropy_healpix, solve-field, sex).
            # Cached on self so the status collector reads it without re-probing.
            self._processor_dep_issues = check_processor_runtime_deps(self.settings)
            for dep in self._processor_dep_issues:
                CITRASENSE_LOGGER.warning(
                    f"processor '{dep['device_name']}' missing dependencies: "
                    f"{dep['missing_packages']}. Install with: {dep['install_cmd']}"
                )

            # Initialize location service (manages GPS internally)
            self.location_service = LocationService(
                api_client=self.api_client,
                settings=self.settings,
            )

            # Initialize time monitor with GPS reference from location service
            self.time_monitor = TimeMonitor(
                check_interval_minutes=self.settings.time_check_interval_minutes,
                pause_threshold_ms=self.settings.time_offset_pause_ms,
                gps_monitor=(
                    self.location_service.gps_monitor
                    if self.location_service and self.location_service.gps_monitor_started
                    else None
                ),
            )
            self.time_monitor.start()
            CITRASENSE_LOGGER.info("Time synchronization monitoring started")

            # Initialize every configured sensor (historically telescope-only,
            # now multi-sensor) + TaskDispatcher + polling.
            success, error = self._initialize_telescopes(
                old_task_dict=old_task_dict,
                old_imaging_tasks=old_imaging_tasks,
                old_processing_tasks=old_processing_tasks,
                old_uploading_tasks=old_uploading_tasks,
            )

            if success:
                self.configuration_error = None
                CITRASENSE_LOGGER.info("Components initialized successfully!")
                return True, None
            else:
                self.configuration_error = error
                return False, error

        except Exception as e:
            error_msg = f"Failed to initialize components: {e!s}"
            CITRASENSE_LOGGER.error(error_msg, exc_info=True)
            self.configuration_error = error_msg
            return False, error_msg

    def reload_configuration(self) -> tuple[bool, str | None]:
        """Reload configuration from file and reinitialize all components."""
        return self._initialize_components(reload_settings=True)

    def _initialize_telescopes(
        self,
        old_task_dict: dict | None = None,
        old_imaging_tasks: dict | None = None,
        old_processing_tasks: dict | None = None,
        old_uploading_tasks: dict | None = None,
    ) -> tuple[bool, str | None]:
        """Build site-level state, register every runtime in ``pending``, fan out per-sensor connect.

        Site-level orchestration core for parallel/async sensor init
        (issue #339).  The heavy lifting lives in
        :class:`citrasense.sensors.init_orchestrator.SensorInitOrchestrator`
        and the per-modality
        :class:`citrasense.sensors.runtime_builder.RuntimeBuilder`
        implementations.  This method is just the glue:

        1. **Synchronous site-level setup** — API auth, safety monitor,
           TaskDispatcher, duplicate-id check, deterministic
           ground-station resolution.
        2. **Synchronous runtime build** — each sensor gets a runtime
           registered with the dispatcher in ``init_state="pending"``
           via :func:`build_for_sensor`.  No ``adapter.connect()`` calls
           happen here.
        3. **Async connect fan-out** — the orchestrator submits one
           ``connect()`` per sensor to a thread-pool executor and
           updates each runtime's ``init_state``
           (``connected`` / ``failed`` / ``timed_out``) as the futures
           resolve.  A hung adapter only affects its own sensor.
        4. **Dispatcher start** — the poll/runner threads run
           immediately; they gate per-sensor task dispatch on
           ``runtime.is_ready`` so unconnected sensors are simply
           skipped until their connect future resolves.

        Returns:
            Tuple of (success, error_message). ``success=True`` means
            the daemon is ready to operate; individual sensor connects
            may still be in flight in the background.
        """
        old_task_dict = old_task_dict or {}
        old_imaging_tasks = old_imaging_tasks or {}
        old_processing_tasks = old_processing_tasks or {}
        old_uploading_tasks = old_uploading_tasks or {}
        assert self.api_client is not None

        try:
            CITRASENSE_LOGGER.info(f"CitraAPISettings host is {self.settings.host}")

            if not self.api_client.does_api_server_accept_key():
                error_msg = "Could not authenticate with Citra API. Check your access token."
                CITRASENSE_LOGGER.error(error_msg)
                return False, error_msg

            self._initialize_safety_monitor()

            self.task_dispatcher = TaskDispatcher(
                self.api_client,
                CITRASENSE_LOGGER,
                self.settings,
                safety_monitor=self.safety_monitor,
                elset_cache=self.elset_cache,
            )

            if self.web_server:
                self.task_dispatcher.on_toast = self.web_server.send_toast

            # Pre-flight: reject duplicate citra_sensor_id across sensors.
            # TaskDispatcher._runtime_for_task resolves API ids by walking
            # the runtimes and returning the *first* match, so two sensors
            # with the same backend id silently starve one scope.  Fail
            # fast with a clear message instead.
            dupes = CitraSenseSettings.find_duplicate_citra_sensor_ids(self.settings.sensors)
            if dupes:
                detail = "; ".join(
                    f"{api_id!r} claimed by {', '.join(local_ids)}" for api_id, local_ids in dupes.items()
                )
                error_msg = (
                    "Duplicate citra_sensor_id detected — two local sensors cannot share "
                    f"one backend telescope record: {detail}. Edit the config so each "
                    "local sensor points at a distinct Citra telescope id."
                )
                CITRASENSE_LOGGER.critical(error_msg)
                if self.web_server:
                    self.web_server.send_toast(
                        "Duplicate citra_sensor_id — one scope would be starved of tasks. Check config.",
                        "danger",
                        "duplicate-citra-sensor-id",
                    )
                return False, error_msg

            assert self.sensor_manager is not None
            assert self.safety_monitor is not None

            # ── Pre-flight: deterministic ground-station resolution ──
            # Iterate sensors in lexical id order and take the first
            # one whose API record returns a usable ground station;
            # divergent groundStationIds across sensors are rejected
            # here.
            ok, err, ground_station = resolve_canonical_ground_station(
                sensor_manager=self.sensor_manager,
                settings=self.settings,
                api_client=self.api_client,
                logger=CITRASENSE_LOGGER,
                location_service=self.location_service,
            )
            if not ok:
                return False, err
            self.ground_station = ground_station

            self._init_orchestrator = SensorInitOrchestrator(
                logger=CITRASENSE_LOGGER,
                web_server=self.web_server,
                sensor_manager=self.sensor_manager,
                settings=self.settings,
                task_dispatcher=self.task_dispatcher,
            )

            build_ctx = BuildContext(
                api_client=self.api_client,
                task_dispatcher=self.task_dispatcher,
                safety_monitor=self.safety_monitor,
                settings=self.settings,
                location_service=self.location_service,
                elset_cache=self.elset_cache,
                apass_catalog=self.apass_catalog,
                ground_station=self.ground_station,
                preview_bus=self.preview_bus,
                task_index=self.task_index,
                sensor_bus=self.sensor_bus,
                web_server=self.web_server,
                on_annotated_image=self._on_annotated_image,
                save_filter_config=self.save_filter_config,
                sync_filters_to_backend=self.sync_filters_to_backend,
                logger=CITRASENSE_LOGGER,
                init_state_callback_factory=self._init_orchestrator.make_init_state_toast_callback,
            )

            # ── Build every runtime up front in init_state="pending" ──
            # Telescopes first (in lexical order so multi-sensor logs
            # group consistently), then radar, then allsky.
            sensors_in_build_order: list = []
            sensors_in_build_order.extend(
                sorted(self.sensor_manager.iter_by_type("telescope"), key=lambda s: s.sensor_id)
            )
            sensors_in_build_order.extend(self.sensor_manager.iter_by_type("passive_radar"))
            sensors_in_build_order.extend(self.sensor_manager.iter_by_type("allsky"))

            built: list = []
            for sensor in sensors_in_build_order:
                try:
                    runtime, builder = build_for_sensor(sensor, build_ctx)
                    built.append((runtime, builder))
                except Exception as exc:
                    # A failure to *build* the runtime (bad config,
                    # missing SensorConfig) is different from a failed
                    # adapter.connect(): the runtime can't even be
                    # registered, so we surface it as a per-sensor
                    # startup error.  Other sensors still get their
                    # chance via the loop's continue.
                    CITRASENSE_LOGGER.error(
                        "Failed to build runtime for sensor %s: %s",
                        sensor.sensor_id,
                        exc,
                        exc_info=True,
                    )
                    if self.web_server:
                        self.web_server.send_toast(
                            f"Could not initialize {sensor.sensor_id}: {exc}",
                            "danger",
                            f"sensor-init-{sensor.sensor_id}",
                        )

            # Restore preserved task metadata BEFORE starting the dispatcher
            # so the poll loop sees the same heap shape as before reload.
            if old_task_dict:
                CITRASENSE_LOGGER.info(f"Restoring {len(old_task_dict)} task(s) from previous TaskDispatcher")
                self.task_dispatcher.restore_task_dict(old_task_dict)
            if old_imaging_tasks:
                CITRASENSE_LOGGER.info(f"Restoring {len(old_imaging_tasks)} imaging task(s)")
                self.task_dispatcher.imaging_tasks.update(old_imaging_tasks)

            dropped = len(old_processing_tasks) + len(old_uploading_tasks)
            if dropped:
                CITRASENSE_LOGGER.info(
                    f"Dropping {dropped} in-flight processing/uploading task(s) — will be re-queued on next poll"
                )

            # Dispatcher starts NOW.  Pending sensors are gated out of
            # task dispatch by ``runtime.is_ready``, so the poll/runner
            # loops just skip them until the orchestrator's connect
            # workers flip them to ``connected``.
            self.task_dispatcher.start()
            self._start_retention_timer()

            # Async fan-out: submit every sensor's connect() to the
            # orchestrator's executor.  Returns immediately — any
            # sensor that connects quickly starts taking tasks within
            # seconds; a hung sensor times out under its own watchdog
            # without blocking the rest of the site.
            self._init_orchestrator.fan_out(built)

            CITRASENSE_LOGGER.info("Site initialized; %d sensor(s) connecting in background", len(built))
            return True, None

        except Exception as e:
            error_msg = f"Error initializing telescopes: {e!s}"
            CITRASENSE_LOGGER.error(error_msg, exc_info=True)
            return False, error_msg

    def _start_retention_timer(self) -> None:
        """Run retention cleanup once, then schedule the next run in 1 hour."""
        try:
            retention = self.settings.processing_output_retention_hours
            cleanup_processing_output(self.settings.directories.processing_dir, retention)
            cleanup_previews(self.settings.directories.analysis_previews_dir)
        except Exception as e:
            CITRASENSE_LOGGER.warning("Retention cleanup error: %s", e)

        if self._stop_requested:
            return
        self._retention_timer = threading.Timer(3600, self._start_retention_timer)
        self._retention_timer.daemon = True
        self._retention_timer.start()

    def request_sensor_connect(self, sensor_id: str) -> tuple[bool, str | None]:
        """Thin facade over :meth:`SensorInitOrchestrator.request_connect`.

        Connect-or-noop: returns ``(True, "already connected")`` when
        the sensor is already up, otherwise queues a ``connect()`` on
        the async init worker and returns ``(True, None)``.  Used by
        ``POST /api/sensors/{id}/connect``.
        """
        if self._init_orchestrator is None:
            return False, "Daemon not initialized"
        return self._init_orchestrator.request_connect(sensor_id)

    def request_sensor_reconnect(self, sensor_id: str) -> tuple[bool, str | None]:
        """Thin facade over :meth:`SensorInitOrchestrator.request_reconnect`.

        Kept on the daemon so the FastAPI route handlers can keep
        calling ``ctx.daemon.request_sensor_reconnect(...)`` — the web
        layer stays oblivious to the orchestrator object's lifecycle.
        """
        if self._init_orchestrator is None:
            return False, "Daemon not initialized"
        return self._init_orchestrator.request_reconnect(sensor_id)

    def request_sensor_disconnect(self, sensor_id: str) -> tuple[bool, str | None]:
        """Thin facade over :meth:`SensorInitOrchestrator.request_disconnect`."""
        if self._init_orchestrator is None:
            return False, "Daemon not initialized"
        return self._init_orchestrator.request_disconnect(sensor_id)

    def _initialize_safety_monitor(self) -> None:
        """Create SafetyMonitor with site-level checks and wire to hardware.

        Sensor-specific checks (e.g. cable wrap) are registered separately
        by each sensor via ``safety_monitor.register_sensor_check()``.
        """
        from citrasense.safety.disk_space_check import DiskSpaceCheck
        from citrasense.safety.safety_monitor import SafetyMonitor
        from citrasense.safety.time_health_check import TimeHealthCheck

        assert self.sensor_manager is not None

        checks: list = []

        # Disk space check (site-level — all sensors share the images dir)
        images_dir = self.settings.directories.images_dir
        checks.append(DiskSpaceCheck(CITRASENSE_LOGGER, images_dir))

        # Time health check
        if self.time_monitor:
            checks.append(TimeHealthCheck(CITRASENSE_LOGGER, self.time_monitor))

        def abort_callback() -> None:
            if not self.sensor_manager:
                return
            for sensor in self.sensor_manager:
                mount = getattr(getattr(sensor, "adapter", None), "mount", None)
                if not mount:
                    continue
                try:
                    mount.abort_slew()
                    mount.stop_tracking()
                    for d in ("north", "south", "east", "west"):
                        mount.stop_move(d)
                except Exception:
                    pass

        self.safety_monitor = SafetyMonitor(CITRASENSE_LOGGER, checks, abort_callback=abort_callback)

        # Hardware safety check — polls each adapter's external safety monitor.
        # Registered per-sensor so `get_sensor_checks(sensor_id)` can disambiguate
        # when multiple telescopes each have their own hardware safety device.
        if self.settings and self.settings.hardware_safety_check_enabled:
            wired_hw_safety = False
            for s in self.sensor_manager.iter_by_type("telescope"):
                adapter = getattr(s, "adapter", None)
                if adapter and getattr(adapter, "supports_hardware_safety_monitor", False):
                    from citrasense.safety.hardware_safety_check import HardwareSafetyCheck

                    self.safety_monitor.register_sensor_check(
                        s.sensor_id,
                        HardwareSafetyCheck(CITRASENSE_LOGGER, adapter.query_hardware_safety),
                    )
                    CITRASENSE_LOGGER.info("Hardware safety check enabled for %s", s.sensor_id)
                    wired_hw_safety = True
            if not wired_hw_safety:
                CITRASENSE_LOGGER.info(
                    "Hardware safety check enabled in settings but no adapter supports it — skipping"
                )

        if self.sensor_manager:
            for sensor in self.sensor_manager:
                adapter = getattr(sensor, "adapter", None)
                if adapter and hasattr(adapter, "set_safety_monitor"):
                    adapter.set_safety_monitor(self.safety_monitor)

        self.safety_monitor.start_watchdog()
        CITRASENSE_LOGGER.info("Safety monitor started with %d site-level check(s)", len(checks))

    def save_filter_config(self, sensor: TelescopeSensor):
        """Save filter configuration from adapter to settings if supported."""
        ts = sensor
        if not ts:
            return
        adapter = ts.adapter
        if not adapter.supports_filter_management():
            return

        try:
            filter_config = adapter.get_filter_config()
            if filter_config:
                sensor_cfg = self.settings.get_sensor_config(ts.sensor_id)
                if sensor_cfg:
                    sensor_cfg.adapter_settings["filters"] = filter_config
                self.settings.save()
                CITRASENSE_LOGGER.info(f"Saved filter configuration with {len(filter_config)} filters")
        except Exception as e:
            CITRASENSE_LOGGER.warning(f"Failed to save filter configuration: {e}")

    def sync_filters_to_backend(self, sensor: TelescopeSensor):
        """Sync enabled filters to backend API."""
        ts = sensor
        if not ts or not self.api_client or not ts.citra_record:
            return

        try:
            filter_config = ts.adapter.get_filter_config()
            sync_filters_to_backend(self.api_client, ts.citra_record["id"], filter_config, CITRASENSE_LOGGER)
        except Exception as e:
            CITRASENSE_LOGGER.warning(f"Failed to sync filters to backend: {e}", exc_info=True)

    def _resolve_runtime(self, sensor_id: str) -> SensorRuntime | None:
        """Resolve a telescope runtime by sensor_id."""
        if not self.task_dispatcher:
            return None
        return self.task_dispatcher.get_runtime(sensor_id)

    def trigger_autofocus(self, sensor_id: str) -> tuple[bool, str | None]:
        """Request autofocus to run at next safe point between tasks."""
        rt = self._resolve_runtime(sensor_id)
        if rt is None:
            return False, "Telescope runtime not initialized"
        if not rt.hardware_adapter or not rt.hardware_adapter.supports_filter_management():
            return False, "Hardware adapter does not support filter management"
        rt.autofocus_manager.request()
        return True, None

    def cancel_autofocus(self, sensor_id: str) -> bool:
        """Cancel autofocus whether it is queued or actively running."""
        rt = self._resolve_runtime(sensor_id)
        if rt is None:
            return False
        return rt.autofocus_manager.cancel()

    def is_autofocus_requested(self, sensor_id: str) -> bool:
        """Check if autofocus is currently queued."""
        rt = self._resolve_runtime(sensor_id)
        if rt is None:
            return False
        return rt.autofocus_manager.is_requested()

    def trigger_calibration(self, params: dict, *, sensor_id: str) -> tuple[bool, str | None]:
        """Request calibration capture at next safe point between tasks."""
        rt = self._resolve_runtime(sensor_id)
        if rt is None or not rt.calibration_manager:
            return False, "Calibration not available (no direct camera control)"
        ok = rt.calibration_manager.request(params)
        if not ok:
            return False, "Calibration already in progress"
        return True, None

    def trigger_calibration_suite(self, jobs: list[dict], *, sensor_id: str) -> tuple[bool, str | None]:
        """Request a batch calibration suite at next safe point between tasks."""
        rt = self._resolve_runtime(sensor_id)
        if rt is None or not rt.calibration_manager:
            return False, "Calibration not available (no direct camera control)"
        if not jobs:
            return False, "No calibration jobs specified"
        ok = rt.calibration_manager.request_suite(jobs)
        if not ok:
            return False, "Calibration already in progress"
        return True, None

    def cancel_calibration(self, sensor_id: str) -> bool:
        """Cancel calibration whether queued or actively running."""
        rt = self._resolve_runtime(sensor_id)
        if rt is None or not rt.calibration_manager:
            return False
        return rt.calibration_manager.cancel()

    def start_headless(self) -> None:
        """Start the web server and initialize components without signal handling.

        Intended for E2E tests and embedded usage where the caller manages
        the lifecycle (e.g. running in a background thread).
        """
        assert self.web_server is not None
        self.web_server.start()
        self._initialize_components()

    def request_stop(self) -> None:
        """Request a graceful shutdown of the daemon loop."""
        self._stop_requested = True

    def run(self):
        assert self.web_server is not None
        # atexit ensures cleanup runs even if the process is killed abruptly
        # (e.g. debugger Stop button). _shutdown is idempotent so it's safe
        # to fire from both atexit and the finally block.
        atexit.register(self._shutdown)

        # Start web server FIRST, so users can monitor/configure
        # The web interface will remain available even if configuration is incomplete
        self.web_server.start()
        CITRASENSE_LOGGER.info(f"Web interface available at http://{self.web_server.host}:{self.web_server.port}")

        try:
            # Try to initialize components
            success, error = self._initialize_components()
            if success:
                CITRASENSE_LOGGER.info("=" * 60)
                CITRASENSE_LOGGER.info("CitraSense ready \u2014 watching for tasks")
            else:
                CITRASENSE_LOGGER.warning(
                    f"Could not start telescope operations: {error}. "
                    f"Configure via web interface at http://{self.web_server.host}:{self.web_server.port}"
                )
            self._keep_running()
        finally:
            self._shutdown()

    def _keep_running(self):
        """Keep the daemon running until interrupted by SIGINT/SIGTERM."""
        self._stop_requested = False

        def _signal_handler(signum, _frame):
            sig_name = signal.Signals(signum).name
            CITRASENSE_LOGGER.info(f"Received {sig_name}, shutting down daemon.")
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            while not self._stop_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            CITRASENSE_LOGGER.info("Shutting down daemon.")

    def _shutdown(self):
        """Clean up resources on shutdown.  Idempotent — safe to call multiple
        times (e.g. from both the ``finally`` block and ``atexit``)."""
        if self._shutdown_done:
            return
        self._shutdown_done = True

        CITRASENSE_LOGGER.info("Shutting down...")

        # 0. Cancel retention timer — set _stop_requested first to prevent reschedule
        self._stop_requested = True
        if self._retention_timer:
            self._retention_timer.cancel()

        # 0b. Tear down the sensor-init executor.  Daemon threads in the
        # pool die with the process anyway, but explicitly shutting it
        # down stops new reconnect submissions from the web layer (e.g.
        # if a request lands during shutdown).
        if self._init_orchestrator is not None:
            self._init_orchestrator.shutdown()

        # 1. Stop sources of new motion
        if self.task_dispatcher:
            self.task_dispatcher.stop()
        if self.time_monitor:
            self.time_monitor.stop()

        # 2. Abort any residual motion on all telescopes
        if self.sensor_manager:
            for ts in self.sensor_manager.iter_by_type("telescope"):
                try:
                    m = cast("TelescopeSensor", ts).adapter.mount
                    if m:
                        m.abort_slew()
                except Exception:
                    pass

        # 3. Stop safety (watchdog last — it was guarding steps 1-2)
        if self.safety_monitor:
            if self.sensor_manager:
                for ts in self.sensor_manager.iter_by_type("telescope"):
                    cast("TelescopeSensor", ts).unregister_safety_checks(self.safety_monitor)
            self.safety_monitor.stop_watchdog()

        # 4. Disconnect hardware via sensor manager
        if self.sensor_manager:
            CITRASENSE_LOGGER.info("Disconnecting sensors...")
            self.sensor_manager.disconnect_all()

        # 5. Close analysis index
        if self.task_index:
            try:
                self.task_index.close()
            except Exception:
                pass
            self.task_index = None

        CITRASENSE_LOGGER.info("Shutdown complete.")

        # 6. Stop web server (tears down log handler — must be last)
        if self.web_server:
            if self.web_server.web_log_handler:
                CITRASENSE_LOGGER.removeHandler(self.web_server.web_log_handler)
