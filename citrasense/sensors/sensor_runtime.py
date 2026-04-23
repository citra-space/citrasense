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

if TYPE_CHECKING:
    from collections.abc import Callable

    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.preview_bus import PreviewBus
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.sensors.bus import SensorBus, Subscription
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
        self.logger = logger.getChild(f"SensorRuntime[{sensor.sensor_id}]")
        self.settings = settings
        self.api_client = api_client
        self.hardware_adapter = hardware_adapter
        self.processor_registry = processor_registry
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

        # ── Queue trio ─────────────────────────────────────────────────
        self.acquisition_queue = AcquisitionQueue(
            num_workers=1,
            settings=settings,
            logger=logger,
            api_client=api_client,
            runtime=self,
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

        # ── Hardware managers (telescope-specific) ─────────────────────
        self.autofocus_manager: Any = None
        self.alignment_manager: Any = None
        self.homing_manager: Any = None
        self.calibration_manager: Any = None

        if sensor.sensor_type == "telescope" and hardware_adapter is not None:
            from citrasense.tasks.alignment_manager import AlignmentManager
            from citrasense.tasks.autofocus_manager import AutofocusManager
            from citrasense.tasks.homing_manager import HomingManager

            self.autofocus_manager = AutofocusManager(
                self.logger,
                hardware_adapter,
                settings,
                imaging_queue=self.acquisition_queue,
                location_service=location_service,
                preview_bus=preview_bus,
            )
            self.alignment_manager = AlignmentManager(
                self.logger,
                hardware_adapter,
                settings,
                imaging_queue=self.acquisition_queue,
                safety_monitor=safety_monitor,
                location_service=location_service,
                preview_bus=preview_bus,
            )
            self.homing_manager = HomingManager(
                self.logger,
                hardware_adapter,
                imaging_queue=self.acquisition_queue,
            )

    def set_dispatcher(self, dispatcher: TaskDispatcher) -> None:
        """Wire the parent dispatcher after construction (resolves circular dependency)."""
        self._dispatcher = dispatcher

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
        from citrasense.tasks.scope.sidereal_telescope_task import SiderealTelescopeTask
        from citrasense.tasks.scope.tracking_telescope_task import TrackingTelescopeTask

        mode = self.settings.observation_mode

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
