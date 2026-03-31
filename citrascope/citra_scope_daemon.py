from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrascope.calibration.calibration_library import CalibrationLibrary

from citrascope.api.citra_api_client import AbstractCitraApiClient, CitraApiClient
from citrascope.api.dummy_api_client import DummyApiClient
from citrascope.catalogs.apass_catalog import ApassCatalog
from citrascope.elset_cache import ElsetCache
from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrascope.hardware.adapter_registry import get_adapter_class
from citrascope.hardware.filter_sync import sync_filters_to_backend
from citrascope.location import LocationService
from citrascope.logging import CITRASCOPE_LOGGER
from citrascope.logging._citrascope_logger import setup_file_logging
from citrascope.processors.processor_registry import ProcessorRegistry
from citrascope.settings.citrascope_settings import CitraScopeSettings
from citrascope.tasks.runner import TaskManager
from citrascope.time.time_monitor import TimeMonitor
from citrascope.web.server import CitraScopeWebServer


class CitraScopeDaemon:
    def __init__(
        self,
        settings: CitraScopeSettings,
        api_client: AbstractCitraApiClient | None = None,
        hardware_adapter: AbstractAstroHardwareAdapter | None = None,
    ):
        self.settings = settings
        CITRASCOPE_LOGGER.setLevel(self.settings.log_level)

        # Setup file logging if enabled
        if self.settings.file_logging_enabled:
            self.settings.directories.ensure_log_directory()
            log_path = self.settings.directories.current_log_path()
            setup_file_logging(log_path, backup_count=self.settings.log_retention_days)
            CITRASCOPE_LOGGER.info(f"Logging to file: {log_path}")

        try:
            ver = pkg_version("citrascope")
        except PackageNotFoundError:
            ver = "dev"
        CITRASCOPE_LOGGER.info("=" * 60)
        CITRASCOPE_LOGGER.info(f"CitraScope {ver} starting (PID {os.getpid()})")

        self.api_client = api_client
        self.hardware_adapter = hardware_adapter
        self.web_server = None
        self.task_manager = None
        self.time_monitor = None
        self.location_service = None
        self.ground_station = None
        self.telescope_record = None
        self.safety_monitor = None
        self.configuration_error: str | None = None
        self.latest_annotated_image_path: str | None = None
        self.calibration_library: CalibrationLibrary | None = None
        self._stop_requested = False
        self._shutdown_done = False

        # Initialize processor registry
        self.processor_registry = ProcessorRegistry(settings=self.settings, logger=CITRASCOPE_LOGGER)

        # Elset cache for satellite matcher (file-backed; warm-start from disk, full refresh at init)
        self.elset_cache = ElsetCache()

        # APASS catalog for local photometry (file-backed; downloaded on first authenticated startup)
        self.apass_catalog = ApassCatalog(logger=CITRASCOPE_LOGGER)

        # Note: Work queues and stage tracking now managed by TaskManager

        # Create web server instance (always enabled)
        self.web_server = CitraScopeWebServer(daemon=self, host="0.0.0.0", port=self.settings.web_port)

    def _refresh_elset_cache_with_retry(self, max_attempts: int = 3) -> None:
        """Force-refresh the elset cache at startup, retrying on failure."""
        assert self.api_client is not None
        for attempt in range(1, max_attempts + 1):
            if self.elset_cache.refresh(self.api_client, logger=CITRASCOPE_LOGGER):
                return
            if attempt < max_attempts:
                backoff = 2**attempt
                CITRASCOPE_LOGGER.warning(
                    "ElsetCache: refresh attempt %d/%d failed, retrying in %ds",
                    attempt,
                    max_attempts,
                    backoff,
                )
                time.sleep(backoff)
        CITRASCOPE_LOGGER.warning(
            "ElsetCache: all %d refresh attempts failed — using cached data if available",
            max_attempts,
        )

    def _create_hardware_adapter(self) -> AbstractAstroHardwareAdapter:
        """Factory method to create the appropriate hardware adapter based on settings."""
        try:
            adapter_class = get_adapter_class(self.settings.hardware_adapter)
            self.settings.directories.ensure_data_directories()
            images_dir = self.settings.directories.images_dir
            return adapter_class(logger=CITRASCOPE_LOGGER, images_dir=images_dir, **self.settings.adapter_settings)
        except ImportError as e:
            CITRASCOPE_LOGGER.error(
                f"{self.settings.hardware_adapter} adapter requested but dependencies not available. " f"Error: {e}"
            )
            raise RuntimeError(
                f"{self.settings.hardware_adapter} adapter requires additional dependencies. "
                f"Check documentation for installation instructions."
            ) from e

    def _initialize_components(self, reload_settings: bool = False) -> tuple[bool, str | None]:
        """Initialize or reinitialize all components.

        Args:
            reload_settings: If True, reload settings from disk before initializing

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if reload_settings:
                CITRASCOPE_LOGGER.info("\u2500" * 60)
                CITRASCOPE_LOGGER.info("Configuration reload requested")
                CITRASCOPE_LOGGER.info("\u2500" * 60)
                # Reload settings from file (preserving web_port)
                new_settings = CitraScopeSettings.load(web_port=self.settings.web_port)
                self.settings = new_settings
                CITRASCOPE_LOGGER.setLevel(self.settings.log_level)

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
            if self.task_manager:
                CITRASCOPE_LOGGER.info("Stopping existing task manager...")
                # Capture task metadata before destruction
                old_task_dict = dict(self.task_manager.task_dict)
                old_imaging_tasks = dict(self.task_manager.imaging_tasks)
                old_processing_tasks = dict(self.task_manager.processing_tasks)
                old_uploading_tasks = dict(self.task_manager.uploading_tasks)
                self.task_manager.stop()
                self.task_manager = None

            if self.safety_monitor:
                self.safety_monitor.stop_watchdog()
                self.safety_monitor = None

            if self.time_monitor:
                self.time_monitor.stop()
                self.time_monitor = None

            if self.location_service:
                self.location_service.stop()
                self.location_service = None

            if self.hardware_adapter:
                try:
                    self.hardware_adapter.disconnect()
                except Exception as e:
                    CITRASCOPE_LOGGER.warning(f"Error disconnecting hardware: {e}")
                self.hardware_adapter = None

            # Check if configuration is complete
            if not self.settings.is_configured():
                error_msg = "Configuration incomplete. Please set access token, telescope ID, and hardware adapter."
                CITRASCOPE_LOGGER.warning(error_msg)
                self.configuration_error = error_msg
                return False, error_msg

            # Initialize API client
            if self.settings.use_dummy_api:
                CITRASCOPE_LOGGER.info("Using DummyApiClient for local testing")
                self.api_client = DummyApiClient(logger=CITRASCOPE_LOGGER)
            else:
                self.api_client = CitraApiClient(
                    self.settings.host,
                    self.settings.personal_access_token,
                    self.settings.use_ssl,
                    CITRASCOPE_LOGGER,
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

            # Initialize hardware adapter
            self.hardware_adapter = self._create_hardware_adapter()

            # Check for missing dependencies (non-fatal, just warn)
            missing_deps = self.hardware_adapter.get_missing_dependencies()
            if missing_deps:
                for dep in missing_deps:
                    CITRASCOPE_LOGGER.warning(
                        f"{dep['device_type']} '{dep['device_name']}' missing dependencies: "
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
                gps_monitor=self.location_service.gps_monitor if self.location_service else None,
            )
            self.time_monitor.start()
            CITRASCOPE_LOGGER.info("Time synchronization monitoring started")

            # Initialize telescope
            success, error = self._initialize_telescope(
                old_task_dict=old_task_dict,
                old_imaging_tasks=old_imaging_tasks,
                old_processing_tasks=old_processing_tasks,
                old_uploading_tasks=old_uploading_tasks,
            )

            if success:
                self.configuration_error = None
                CITRASCOPE_LOGGER.info("Components initialized successfully!")
                return True, None
            else:
                self.configuration_error = error
                return False, error

        except Exception as e:
            error_msg = f"Failed to initialize components: {e!s}"
            CITRASCOPE_LOGGER.error(error_msg, exc_info=True)
            self.configuration_error = error_msg
            return False, error_msg

    def reload_configuration(self) -> tuple[bool, str | None]:
        """Reload configuration from file and reinitialize all components."""
        return self._initialize_components(reload_settings=True)

    def retry_connection(self) -> tuple[bool, str | None]:
        """Retry hardware connection using current in-memory settings."""
        CITRASCOPE_LOGGER.info("\u2500" * 60)
        CITRASCOPE_LOGGER.info("Hardware reconnect requested")
        CITRASCOPE_LOGGER.info("\u2500" * 60)
        success, error = self._initialize_components(reload_settings=False)
        if success:
            CITRASCOPE_LOGGER.info("Hardware reconnect completed successfully")
        else:
            CITRASCOPE_LOGGER.warning(f"Hardware reconnect failed: {error}")
        return success, error

    def _initialize_telescope(
        self,
        old_task_dict: dict | None = None,
        old_imaging_tasks: dict | None = None,
        old_processing_tasks: dict | None = None,
        old_uploading_tasks: dict | None = None,
    ) -> tuple[bool, str | None]:
        """Initialize telescope connection and task manager.

        Args:
            old_task_dict: Preserved task_dict from previous TaskManager (for config reload)
            old_imaging_tasks: Preserved imaging_tasks from previous TaskManager (for config reload)
            old_processing_tasks: Preserved processing_tasks from previous TaskManager (for config reload)
            old_uploading_tasks: Preserved uploading_tasks from previous TaskManager (for config reload)

        Returns:
            Tuple of (success, error_message)
        """
        old_task_dict = old_task_dict or {}
        old_imaging_tasks = old_imaging_tasks or {}
        old_processing_tasks = old_processing_tasks or {}
        old_uploading_tasks = old_uploading_tasks or {}
        assert self.api_client is not None
        assert self.hardware_adapter is not None
        try:
            CITRASCOPE_LOGGER.info(f"CitraAPISettings host is {self.settings.host}")
            CITRASCOPE_LOGGER.info(f"CitraAPISettings telescope_id is {self.settings.telescope_id}")

            # check api for valid key, telescope and ground station
            if not self.api_client.does_api_server_accept_key():
                error_msg = "Could not authenticate with Citra API. Check your access token."
                CITRASCOPE_LOGGER.error(error_msg)
                return False, error_msg

            citra_telescope_record = self.api_client.get_telescope(self.settings.telescope_id)
            if not citra_telescope_record:
                error_msg = f"Telescope ID '{self.settings.telescope_id}' is not valid on the server."
                CITRASCOPE_LOGGER.error(error_msg)
                return False, error_msg
            self.telescope_record = citra_telescope_record

            ground_station = self.api_client.get_ground_station(citra_telescope_record["groundStationId"])
            if not ground_station:
                error_msg = "Could not get ground station info from the server."
                CITRASCOPE_LOGGER.error(error_msg)
                return False, error_msg
            self.ground_station = ground_station

            # Update location service with ground station reference
            if self.location_service:
                self.location_service.set_ground_station(self.ground_station)

            # Provide location service so the adapter can sync site coordinates to the mount
            if self.location_service:
                self.hardware_adapter.set_location_service(self.location_service)

            # connect to hardware server (serial handshake + config — no motion)
            CITRASCOPE_LOGGER.info(f"Connecting to hardware with {type(self.hardware_adapter).__name__}...")
            if not self.hardware_adapter.connect():
                error_msg = f"Failed to connect to hardware adapter: {type(self.hardware_adapter).__name__}"
                CITRASCOPE_LOGGER.error(error_msg)
                return False, error_msg

            self.hardware_adapter.scope_slew_rate_degrees_per_second = citra_telescope_record["maxSlewRate"]
            self.hardware_adapter.telescope_record = citra_telescope_record
            CITRASCOPE_LOGGER.info(
                f"Hardware connected. Slew rate: {self.hardware_adapter.scope_slew_rate_degrees_per_second} deg/sec"
            )

            # Save filter configuration if adapter supports it
            self.save_filter_config()
            # Sync discovered filters to backend on startup
            self.sync_filters_to_backend()

            adapter_name = type(self.hardware_adapter).__name__
            slew_rate = self.hardware_adapter.scope_slew_rate_degrees_per_second
            filter_cfg = self.hardware_adapter.get_filter_config()
            enabled = sum(1 for f in filter_cfg.values() if f.get("enabled", False)) if filter_cfg else 0
            gs_name = self.ground_station.get("name", "?") if self.ground_station else "?"
            scope_name = citra_telescope_record.get("name", "?")
            CITRASCOPE_LOGGER.info(
                f"Hardware ready: adapter={adapter_name}, slew={slew_rate} deg/s, "
                f"filters={enabled}/{len(filter_cfg)}, "
                f"station={gs_name}, telescope={scope_name}"
            )

            # Safety monitor MUST be online before any mount motion.
            # connect() above establishes the serial link and syncs site/time
            # but does NOT home.  The operator can home manually via the web
            # UI (/api/mount/home) when ready.
            if not self.hardware_adapter.is_mount_homed():
                CITRASCOPE_LOGGER.info("Mount is not at home position — home via web UI if GoTo fails")

            self._initialize_safety_monitor()

            # Create TaskManager (now owns all queues and stage tracking)
            self.task_manager = TaskManager(
                self.api_client,
                CITRASCOPE_LOGGER,
                self.hardware_adapter,
                self.settings,
                self.processor_registry,
                elset_cache=self.elset_cache,
                apass_catalog=self.apass_catalog,
                safety_monitor=self.safety_monitor,
                location_service=self.location_service,
                telescope_record=self.telescope_record,
                ground_station=self.ground_station,
                on_annotated_image=lambda path: setattr(self, "latest_annotated_image_path", path),
            )

            # Restore preserved task metadata
            if old_task_dict:
                CITRASCOPE_LOGGER.info(f"Restoring {len(old_task_dict)} task(s) from previous TaskManager")
                self.task_manager.task_dict.update(old_task_dict)
            if old_imaging_tasks:
                CITRASCOPE_LOGGER.info(f"Restoring {len(old_imaging_tasks)} imaging task(s)")
                self.task_manager.imaging_tasks.update(old_imaging_tasks)

            # Don't restore processing_tasks or uploading_tasks — those represent
            # in-flight work that died with the old queues.  The tasks still exist
            # in task_dict and on the API, so the next poll cycle will see them as
            # unassigned and re-queue them from scratch.
            dropped = len(old_processing_tasks) + len(old_uploading_tasks)
            if dropped:
                CITRASCOPE_LOGGER.info(
                    f"Dropping {dropped} in-flight processing/uploading task(s) — will be re-queued on next poll"
                )

            # Initialize CalibrationManager if direct camera control is available
            if self.hardware_adapter.supports_direct_camera_control():
                from citrascope.calibration.calibration_library import CalibrationLibrary
                from citrascope.processors.builtin.calibration_processor import CalibrationProcessor
                from citrascope.tasks.calibration_manager import CalibrationManager

                self.calibration_library = CalibrationLibrary()
                self.task_manager.calibration_manager = CalibrationManager(
                    CITRASCOPE_LOGGER,
                    self.hardware_adapter,
                    self.calibration_library,
                    imaging_queue=self.task_manager.imaging_queue,
                )
                # Inject the library into the CalibrationProcessor
                for proc in self.processor_registry.processors:
                    if isinstance(proc, CalibrationProcessor):
                        proc.library = self.calibration_library
                        break

            self.task_manager.start()

            if self.settings and self.settings.align_on_startup:
                CITRASCOPE_LOGGER.info("Requesting startup alignment (align_on_startup=True)")
                self.task_manager.alignment_manager.request()

            CITRASCOPE_LOGGER.info("Telescope initialized successfully!")
            return True, None

        except Exception as e:
            error_msg = f"Error initializing telescope: {e!s}"
            CITRASCOPE_LOGGER.error(error_msg, exc_info=True)
            return False, error_msg

    def _initialize_safety_monitor(self) -> None:
        """Create SafetyMonitor with applicable checks and wire to hardware."""
        from citrascope.safety.cable_wrap_check import CableWrapCheck
        from citrascope.safety.disk_space_check import DiskSpaceCheck
        from citrascope.safety.safety_monitor import SafetyMonitor
        from citrascope.safety.time_health_check import TimeHealthCheck

        assert self.hardware_adapter is not None

        checks: list = []

        # Cable wrap check — only for adapters with a direct mount
        cable_check: CableWrapCheck | None = None
        needs_startup_unwind = False
        mount = self.hardware_adapter.mount
        if mount is not None:
            import platformdirs

            data_dir = Path(platformdirs.user_data_dir("citrascope", appauthor="citrascope"))
            state_file = data_dir / "cable_wrap_state.json"
            cable_check = CableWrapCheck(CITRASCOPE_LOGGER, mount, state_file=state_file)
            cable_check.start()
            mount.register_sync_listener(cable_check.notify_sync)

            if cable_check.needs_startup_unwind():
                CITRASCOPE_LOGGER.warning(
                    "Persisted cable wrap at %.1f° exceeds hard limit — will unwind after safety gate is wired",
                    cable_check.cumulative_deg,
                )
                needs_startup_unwind = True

            checks.append(cable_check)

        # Disk space check
        checks.append(DiskSpaceCheck(CITRASCOPE_LOGGER, self.hardware_adapter.images_dir))

        # Time health check
        if self.time_monitor:
            checks.append(TimeHealthCheck(CITRASCOPE_LOGGER, self.time_monitor))

        # Hardware safety check — polls the adapter's external safety monitor device
        if self.settings and self.settings.hardware_safety_check_enabled:
            if self.hardware_adapter.supports_hardware_safety_monitor:
                from citrascope.safety.hardware_safety_check import HardwareSafetyCheck

                checks.append(HardwareSafetyCheck(CITRASCOPE_LOGGER, self.hardware_adapter.query_hardware_safety))
                CITRASCOPE_LOGGER.info("Hardware safety check enabled")
            else:
                CITRASCOPE_LOGGER.info(
                    "Hardware safety check enabled in settings but adapter %s does not support it — skipping",
                    type(self.hardware_adapter).__name__,
                )

        def abort_callback() -> None:
            try:
                m = self.hardware_adapter.mount if self.hardware_adapter else None
                if not m:
                    return
                m.abort_slew()
                m.stop_tracking()
                for d in ("north", "south", "east", "west"):
                    m.stop_move(d)
            except Exception:
                pass

        self.safety_monitor = SafetyMonitor(CITRASCOPE_LOGGER, checks, abort_callback=abort_callback)
        self.hardware_adapter.set_safety_monitor(self.safety_monitor)

        # Wire safety gate so cable unwind respects operator stop.
        # Must check operator_stop directly — is_action_safe() would ask
        # cable_wrap itself, which returns False while _unwinding is True.
        # IMPORTANT: this must happen BEFORE any execute_action() call so
        # the unwind can be interrupted by operator stop.
        op_stop = self.safety_monitor.operator_stop
        if cable_check is not None:
            cable_check.safety_gate = lambda: not op_stop.is_active

        self.safety_monitor.start_watchdog()
        CITRASCOPE_LOGGER.info("Safety monitor started with %d check(s)", len(checks))

        # Now that the safety gate is wired, attempt the startup unwind.
        if needs_startup_unwind and cable_check is not None:
            CITRASCOPE_LOGGER.info("Starting deferred cable unwind (safety gate active)")
            cable_check.execute_action()
            if cable_check.did_last_unwind_fail():
                cable_check.mark_intervention_required()
                CITRASCOPE_LOGGER.critical(
                    "Startup unwind did not converge (%.1f° remaining) — "
                    "manual intervention required before the system can "
                    "operate. Use web UI to reset after physically "
                    "verifying cables.",
                    cable_check.cumulative_deg,
                )

    def save_filter_config(self):
        """Save filter configuration from adapter to settings if supported.

        This method is called:
        - After hardware initialization to save discovered filters
        - After autofocus to save updated focus positions
        - After manual filter focus updates via web API

        Note: This only saves locally. Call sync_filters_to_backend() separately
        when enabled filters change to update the backend.

        Thread safety: This modifies self.settings and writes to disk.
        Should be called from main daemon thread or properly synchronized.
        """
        if not self.hardware_adapter or not self.hardware_adapter.supports_filter_management():
            return

        try:
            filter_config = self.hardware_adapter.get_filter_config()
            if filter_config:
                self.settings.adapter_settings["filters"] = filter_config
                self.settings.save()
                CITRASCOPE_LOGGER.info(f"Saved filter configuration with {len(filter_config)} filters")
        except Exception as e:
            CITRASCOPE_LOGGER.warning(f"Failed to save filter configuration: {e}")

    def sync_filters_to_backend(self):
        """Sync enabled filters to backend API.

        Extracts enabled filter names from hardware adapter, expands them via
        the filter library API, then updates the telescope's spectral_config.
        Logs warnings on failure without blocking daemon operations.
        """
        if not self.hardware_adapter or not self.api_client or not self.telescope_record:
            return

        try:
            filter_config = self.hardware_adapter.get_filter_config()
            sync_filters_to_backend(self.api_client, self.telescope_record["id"], filter_config, CITRASCOPE_LOGGER)
        except Exception as e:
            CITRASCOPE_LOGGER.warning(f"Failed to sync filters to backend: {e}", exc_info=True)

    def trigger_autofocus(self) -> tuple[bool, str | None]:
        """Request autofocus to run at next safe point between tasks.

        Returns:
            Tuple of (success, error_message)
        """
        if not self.hardware_adapter:
            return False, "No hardware adapter initialized"

        if not self.hardware_adapter.supports_filter_management():
            return False, "Hardware adapter does not support filter management"

        if not self.task_manager:
            return False, "Task manager not initialized"

        # Request autofocus - will run between tasks
        self.task_manager.autofocus_manager.request()
        return True, None

    def cancel_autofocus(self) -> bool:
        """Cancel autofocus whether it is queued or actively running.

        Returns:
            bool: True if something was cancelled, False if nothing to cancel.
        """
        if not self.task_manager:
            return False
        return self.task_manager.autofocus_manager.cancel()

    def is_autofocus_requested(self) -> bool:
        """Check if autofocus is currently queued.

        Returns:
            bool: True if autofocus is queued, False otherwise.
        """
        if not self.task_manager:
            return False
        return self.task_manager.autofocus_manager.is_requested()

    def trigger_calibration(self, params: dict) -> tuple[bool, str | None]:
        """Request calibration capture at next safe point between tasks."""
        if not self.task_manager or not self.task_manager.calibration_manager:
            return False, "Calibration not available (no direct camera control)"
        ok = self.task_manager.calibration_manager.request(params)
        if not ok:
            return False, "Calibration already in progress"
        return True, None

    def trigger_calibration_suite(self, jobs: list[dict]) -> tuple[bool, str | None]:
        """Request a batch calibration suite at next safe point between tasks."""
        if not self.task_manager or not self.task_manager.calibration_manager:
            return False, "Calibration not available (no direct camera control)"
        if not jobs:
            return False, "No calibration jobs specified"
        ok = self.task_manager.calibration_manager.request_suite(jobs)
        if not ok:
            return False, "Calibration already in progress"
        return True, None

    def cancel_calibration(self) -> bool:
        """Cancel calibration whether queued or actively running."""
        if not self.task_manager or not self.task_manager.calibration_manager:
            return False
        return self.task_manager.calibration_manager.cancel()

    def run(self):
        assert self.web_server is not None
        # atexit ensures cleanup runs even if the process is killed abruptly
        # (e.g. debugger Stop button). _shutdown is idempotent so it's safe
        # to fire from both atexit and the finally block.
        atexit.register(self._shutdown)

        # Start web server FIRST, so users can monitor/configure
        # The web interface will remain available even if configuration is incomplete
        self.web_server.start()
        CITRASCOPE_LOGGER.info(f"Web interface available at http://{self.web_server.host}:{self.web_server.port}")

        try:
            # Try to initialize components
            success, error = self._initialize_components()
            if success:
                CITRASCOPE_LOGGER.info("=" * 60)
                CITRASCOPE_LOGGER.info("CitraScope ready \u2014 watching for tasks")
            else:
                CITRASCOPE_LOGGER.warning(
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
            CITRASCOPE_LOGGER.info(f"Received {sig_name}, shutting down daemon.")
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            while not self._stop_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            CITRASCOPE_LOGGER.info("Shutting down daemon.")

    def _shutdown(self):
        """Clean up resources on shutdown.  Idempotent — safe to call multiple
        times (e.g. from both the ``finally`` block and ``atexit``)."""
        if self._shutdown_done:
            return
        self._shutdown_done = True

        CITRASCOPE_LOGGER.info("Shutting down...")

        # 1. Stop sources of new motion
        if self.task_manager:
            self.task_manager.stop()
        if self.time_monitor:
            self.time_monitor.stop()

        # 2. Abort any residual motion
        if self.hardware_adapter:
            try:
                m = self.hardware_adapter.mount
                if m:
                    m.abort_slew()
            except Exception:
                pass

        # 3. Stop safety (watchdog last — it was guarding steps 1-2)
        if self.safety_monitor:
            from citrascope.safety.cable_wrap_check import CableWrapCheck

            cable_check = self.safety_monitor.get_check("cable_wrap")
            if isinstance(cable_check, CableWrapCheck):
                cable_check.join_unwind(timeout=10.0)
                cable_check.stop()
            self.safety_monitor.stop_watchdog()

        # 4. Disconnect hardware
        if self.hardware_adapter:
            try:
                CITRASCOPE_LOGGER.info("Disconnecting hardware...")
                self.hardware_adapter.disconnect()
            except Exception as e:
                CITRASCOPE_LOGGER.warning(f"Error disconnecting hardware: {e}")

        CITRASCOPE_LOGGER.info("Shutdown complete.")

        # 5. Stop web server (tears down log handler — must be last)
        if self.web_server:
            if self.web_server.web_log_handler:
                CITRASCOPE_LOGGER.removeHandler(self.web_server.web_log_handler)
