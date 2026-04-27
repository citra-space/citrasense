"""Per-sensor execution context.

A ``SensorRuntime`` is the execution silo for a single sensor.  It owns the
three-stage work-queue pipeline (acquisition -> processing -> upload),
hardware-specific maintenance managers, and streaming-event ingestion.

Stage tracking and lifetime statistics are delegated to the parent
:class:`~citrasense.tasks.task_dispatcher.TaskDispatcher` via thin proxy
methods -- this keeps the source of truth site-wide while letting
``base_telescope_task`` (and future per-sensor task classes) talk to a
single object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from citrasense.acquisition.acquisition_queue import AcquisitionQueue
from citrasense.acquisition.processing_queue import ProcessingQueue
from citrasense.acquisition.upload_queue import UploadQueue
from citrasense.logging.sensor_logger import get_sensor_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.preview_bus import PreviewBus
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.sensors.bus import SensorBus, Subscription
    from citrasense.sensors.telescope.observing_session import ObservingSessionManager
    from citrasense.sensors.telescope.self_tasking_manager import SelfTaskingManager
    from citrasense.settings.citrasense_settings import CitraSenseSettings
    from citrasense.tasks.task import Task
    from citrasense.tasks.task_dispatcher import TaskDispatcher


class SensorRuntime:
    """Execution silo for a single sensor.

    Owns the queue trio, hardware-specific managers, and streaming ingestion.
    Proxies stage tracking and stats back to the parent TaskDispatcher.
    """

    def __init__(
        self,
        sensor: AbstractSensor,
        *,
        logger: logging.Logger,
        settings: CitraSenseSettings,
        api_client: Any,
        hardware_adapter: AbstractAstroHardwareAdapter | None = None,
        processor_registry: Any = None,
        elset_cache: Any = None,
        apass_catalog: Any = None,
        location_service: Any = None,
        telescope_record: dict | None = None,
        ground_station: dict | None = None,
        on_annotated_image: Callable[[str], None] | None = None,
        preview_bus: PreviewBus | None = None,
        task_index: Any = None,
        safety_monitor: Any = None,
        sensor_bus: SensorBus | None = None,
    ) -> None:
        self.sensor = sensor
        self.sensor_id = sensor.sensor_id
        self.sensor_type = sensor.sensor_type
        # Wrap the runtime logger in a sensor-scoped adapter so every record
        # it emits carries ``extra={'sensor_id': ...}``.  WebLogHandler
        # forwards that onto the WebSocket payload for per-sensor filtering
        # in the log panel.
        self.logger = get_sensor_logger(
            logger.getChild(f"SensorRuntime[{sensor.sensor_id}]"),
            sensor.sensor_id,
        )
        self.settings = settings
        self.api_client = api_client
        if sensor.sensor_type == "telescope" and hardware_adapter is None:
            raise ValueError(f"SensorRuntime for telescope sensor {sensor.sensor_id!r} " "requires a hardware_adapter")
        self.hardware_adapter = hardware_adapter
        # Per-sensor pipeline registry. Callers may still inject a registry
        # (older unit tests do), but production code builds one here so two
        # runtimes don't share a single CalibrationProcessor instance.
        if processor_registry is None and sensor.sensor_type == "telescope":
            from citrasense.pipelines.common.pipeline_registry import PipelineRegistry

            processor_registry = PipelineRegistry(settings=settings, logger=self.logger)
        self.processor_registry = processor_registry
        # Per-sensor calibration library — assigned later by the daemon when
        # the adapter is known to support direct camera control. The library
        # is then wired into this runtime's CalibrationProcessor via
        # ``attach_calibration_library``.
        self.calibration_library: Any = None
        self.elset_cache = elset_cache
        self.apass_catalog = apass_catalog
        self.location_service = location_service
        self.telescope_record = telescope_record
        self.ground_station = ground_station
        self._on_annotated_image = on_annotated_image
        self.task_index = task_index
        self._sensor_bus = sensor_bus
        self._preview_bus = preview_bus

        self._dispatcher: TaskDispatcher | None = None
        self._streaming_sub: Subscription | None = None

        # Per-sensor config (observation, processing tuning, autofocus, etc.)
        self._sensor_config = settings.get_sensor_config(sensor.sensor_id)

        # ── Queue trio ─────────────────────────────────────────────────
        # Pass the sensor-scoped logger (``self.logger``) so every record
        # emitted by the queues and their downstream pipeline contexts
        # carries ``extra={'sensor_id': ...}`` for the web log filter.
        self.acquisition_queue = AcquisitionQueue(
            num_workers=1,
            settings=settings,
            logger=self.logger,
            api_client=api_client,
            runtime=self,
        )
        self.processing_queue = ProcessingQueue(
            num_workers=1,
            settings=settings,
            logger=self.logger,
        )
        self.upload_queue = UploadQueue(
            num_workers=1,
            settings=settings,
            logger=self.logger,
        )

        # ── Hardware managers (telescope-specific) ─────────────────────
        self.autofocus_manager: Any = None
        self.alignment_manager: Any = None
        self.homing_manager: Any = None
        self.calibration_manager: Any = None
        self.observing_session_manager: ObservingSessionManager | None = None
        self.self_tasking_manager: SelfTaskingManager | None = None

        if sensor.sensor_type == "telescope" and hardware_adapter is not None:
            from citrasense.sensors.telescope.managers.alignment_manager import AlignmentManager
            from citrasense.sensors.telescope.managers.autofocus_manager import AutofocusManager
            from citrasense.sensors.telescope.managers.homing_manager import HomingManager

            self.autofocus_manager = AutofocusManager(
                self.logger,
                hardware_adapter,
                settings,
                imaging_queue=self.acquisition_queue,
                location_service=location_service,
                preview_bus=preview_bus,
            )
            self.autofocus_manager._sensor_id = self.sensor_id
            self.autofocus_manager._sensor_config = self._sensor_config
            self.alignment_manager = AlignmentManager(
                self.logger,
                hardware_adapter,
                settings,
                imaging_queue=self.acquisition_queue,
                safety_monitor=safety_monitor,
                location_service=location_service,
                preview_bus=preview_bus,
            )
            self.alignment_manager._sensor_id = self.sensor_id
            self.alignment_manager._sensor_config = self._sensor_config
            self.homing_manager = HomingManager(
                self.logger,
                hardware_adapter,
                imaging_queue=self.acquisition_queue,
                sensor_id=self.sensor_id,
            )

    def set_dispatcher(self, dispatcher: TaskDispatcher) -> None:
        """Wire the parent dispatcher after construction (resolves circular dependency)."""
        self._dispatcher = dispatcher

    def attach_calibration_library(self, library: Any) -> None:
        """Bind a CalibrationLibrary to this runtime and its CalibrationProcessor.

        Each runtime gets its own library so two telescopes don't end up
        sharing a single processor whose ``library`` attribute reflects
        whichever sensor connected last.
        """
        self.calibration_library = library
        if self.processor_registry is None:
            return
        from citrasense.pipelines.optical.calibration_processor import CalibrationProcessor

        for proc in self.processor_registry.processors:
            if isinstance(proc, CalibrationProcessor):
                proc.library = library
                break

    # ── Stage tracking proxies ─────────────────────────────────────────

    def update_task_stage(self, task_id: str, stage: str) -> None:
        assert self._dispatcher is not None
        self._dispatcher.update_task_stage(task_id, stage)

    def remove_task_from_all_stages(self, task_id: str) -> None:
        assert self._dispatcher is not None
        self._dispatcher.remove_task_from_all_stages(task_id)

    # ── Stats proxies ──────────────────────────────────────────────────

    def record_task_started(self) -> None:
        assert self._dispatcher is not None
        self._dispatcher.record_task_started()

    def record_task_succeeded(self) -> None:
        assert self._dispatcher is not None
        self._dispatcher.record_task_succeeded()

    def record_task_failed(self) -> None:
        assert self._dispatcher is not None
        self._dispatcher.record_task_failed()

    # ── Task lookup proxy ──────────────────────────────────────────────

    def get_task_by_id(self, task_id: str) -> Task | None:
        assert self._dispatcher is not None
        return self._dispatcher.get_task_by_id(task_id)

    # ── Queue helpers ──────────────────────────────────────────────────

    def are_queues_idle(self) -> bool:
        return self.acquisition_queue.is_idle() and self.processing_queue.is_idle() and self.upload_queue.is_idle()

    def busy_reason(self) -> str:
        """Return a human-readable reason this sensor is busy, or ``""`` when idle.

        Used by the web layer to gate per-sensor manual hardware calls
        (preview, jog, filter move, focuser move, etc.) — we only want to
        return 409 when *this* sensor is busy, not when some other sensor
        in the site is imaging.
        """
        reasons: list[str] = []
        if not self.acquisition_queue.is_idle():
            reasons.append("imaging")
        af = self.autofocus_manager
        if af and af.is_running():
            reasons.append("autofocus")
        al = self.alignment_manager
        if al and al.is_running():
            reasons.append("alignment")
        if al and al.is_calibrating():
            reasons.append("pointing calibration")
        hm = self.homing_manager
        if hm and (hm.is_running() or hm.is_requested()):
            reasons.append("homing")
        cm = self.calibration_manager
        if cm and (cm.is_running() or cm.is_requested()):
            reasons.append("calibration capture")
        return ", ".join(reasons)

    @property
    def sensor_config(self):
        """The SensorConfig for this runtime's sensor."""
        return self._sensor_config or self.settings.get_sensor_config(self.sensor_id)

    @property
    def paused(self) -> bool:
        """Whether task processing is paused for this sensor."""
        sc = self.sensor_config
        return sc.task_processing_paused if sc else False

    def set_paused(self, value: bool) -> None:
        """Set paused state and persist to config."""
        sc = self.sensor_config
        if sc:
            sc.task_processing_paused = value
            self.settings.save()

    # ── Task submission ────────────────────────────────────────────────

    def submit_task(self, task: Task, on_complete: Callable) -> None:
        """Create the appropriate task object and submit to the acquisition queue."""
        if self.sensor_type == "telescope":
            telescope_task = self._create_telescope_task(task)
            self.acquisition_queue.submit(task.id, task, telescope_task, on_complete)
        else:
            raise NotImplementedError(f"submit_task not implemented for sensor_type={self.sensor_type!r}")

    def _create_telescope_task(self, task: Task) -> Any:
        """Create appropriate telescope task instance.

        Selection depends on the ``observation_mode`` setting:
        - "auto": use TrackingTelescopeTask if the adapter reports
          ``supports_custom_tracking``, otherwise SiderealTelescopeTask.
        - "tracking": always TrackingTelescopeTask.
        - "sidereal": always SiderealTelescopeTask.
        """
        from citrasense.sensors.telescope.tasks.sidereal_telescope_task import SiderealTelescopeTask
        from citrasense.sensors.telescope.tasks.tracking_telescope_task import TrackingTelescopeTask

        sc = self.sensor_config
        mode = sc.observation_mode if sc else "auto"

        use_tracking = False
        if mode == "tracking":
            use_tracking = True
        elif mode == "auto":
            use_tracking = self.hardware_adapter.supports_custom_tracking if self.hardware_adapter else False

        if use_tracking:
            self.logger.info("Using TrackingTelescopeTask (mode=%s)", mode)
        else:
            self.logger.info("Using SiderealTelescopeTask (mode=%s)", mode)

        cls = TrackingTelescopeTask if use_tracking else SiderealTelescopeTask
        assert self.hardware_adapter is not None
        return cls(
            self.api_client,
            self.hardware_adapter,
            self.logger,
            task,
            settings=self.settings,
            runtime=self,
            location_service=self.location_service,
            telescope_record=self.telescope_record,
            ground_station=self.ground_station,
            elset_cache=self.elset_cache,
            apass_catalog=self.apass_catalog,
            processor_registry=self.processor_registry,
            on_annotated_image=self._set_latest_annotated_image,
            task_index=self.task_index,
        )

    def _set_latest_annotated_image(self, path: str) -> None:
        if self._on_annotated_image:
            self._on_annotated_image(path)

    # ── Maintenance ────────────────────────────────────────────────────

    def check_maintenance(self) -> None:
        """Run maintenance manager check-and-execute (telescope-specific)."""
        if self.homing_manager:
            self.homing_manager.check_and_execute()
        if self.alignment_manager:
            self.alignment_manager.check_and_execute()
        if self.autofocus_manager:
            self.autofocus_manager.check_and_execute()
        if self.calibration_manager:
            self.calibration_manager.check_and_execute()

    def is_maintenance_blocking(self) -> bool:
        """Return True if a maintenance operation prevents task dispatch."""
        if self.homing_manager and (self.homing_manager.is_running() or self.homing_manager.is_requested()):
            return True
        cal = self.calibration_manager
        if cal and (cal.is_running() or cal.is_requested()):
            return True
        return False

    def is_focus_or_alignment_active(self) -> bool:
        """Return True if autofocus or alignment is requested/running."""
        af = self.autofocus_manager
        al = self.alignment_manager
        if af and (af.is_requested() or af.is_running()):
            return True
        if al and (al.is_requested() or al.is_running()):
            return True
        return False

    # ── Streaming ingestion ────────────────────────────────────────────

    def _subscribe_streaming(self) -> None:
        """Subscribe to streaming events for this sensor on the SensorBus."""
        if not self._sensor_bus:
            return
        caps = self.sensor.get_capabilities()
        from citrasense.sensors.abstract_sensor import SensorAcquisitionMode

        if caps.acquisition_mode == SensorAcquisitionMode.STREAMING:
            pattern = f"sensors.{self.sensor_id}.events.acquisition"
            self._streaming_sub = self._sensor_bus.subscribe(pattern, self._on_streaming_event)
            self.logger.info("Subscribed to streaming events: %s", pattern)

    def _on_streaming_event(self, subject: str, event: Any) -> None:
        """Handle a streaming acquisition event (placeholder for future sensors)."""
        self.logger.debug("Streaming event on %s: %s", subject, type(event).__name__)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        self.acquisition_queue.start()
        self.processing_queue.start()
        self.upload_queue.start()
        self._subscribe_streaming()

    def stop(self) -> None:
        self.acquisition_queue.stop()
        self.processing_queue.stop()
        self.upload_queue.stop()
        if self._streaming_sub:
            self._streaming_sub.unsubscribe()
            self._streaming_sub = None
