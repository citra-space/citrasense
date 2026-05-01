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
import threading
from typing import TYPE_CHECKING, Any, Literal

from citrasense.acquisition.acquisition_queue import AcquisitionQueue
from citrasense.acquisition.processing_queue import ProcessingQueue
from citrasense.acquisition.upload_queue import UploadQueue
from citrasense.logging.sensor_logger import get_sensor_logger

#: Lifecycle state of a sensor's hardware-init pipeline.
#:
#: - ``pending``: registered with the dispatcher but ``connect()`` has not
#:   been attempted yet.
#: - ``connecting``: a worker is currently inside ``connect()`` (or about
#:   to call it).  Manual reconnect endpoints flip back to this state.
#: - ``connected``: ``connect()`` returned True.  The dispatcher routes
#:   tasks here.
#: - ``failed``: ``connect()`` returned False or raised.  ``init_error``
#:   carries the operator-visible reason.
#: - ``timed_out``: the worker future blew its deadline.  The adapter
#:   thread may still be alive but the daemon has moved on.
SensorInitState = Literal["pending", "connecting", "connected", "failed", "timed_out"]

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from pydantic import BaseModel

    from citrasense.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrasense.pipelines.radar.radar_pipeline import RadarPipeline
    from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.sensors.bus import SensorBus, Subscription
    from citrasense.sensors.preview_bus import PreviewBus
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

        # ── Init-state machine ─────────────────────────────────────────
        # Lifecycle for the sensor's hardware-init pipeline. Lives on the
        # runtime (not the sensor) because runtimes are created up front
        # by the daemon — even before connect() is attempted — so the
        # dispatcher can register them in ``pending`` and gate routing
        # until they flip to ``connected``.
        self._init_state: SensorInitState = "pending"
        self._init_error: str | None = None
        self._init_state_lock = threading.Lock()
        # Optional callback for init-state transitions. Wired by the daemon
        # so toasts fire from one place regardless of whether the
        # transition came from startup, manual reconnect, or a future
        # auto-retry. Signature: ``(state, error_message_or_None)``.
        self.on_init_state_change: Callable[[SensorInitState, str | None], None] | None = None
        # The queue trio is started exactly once after the first
        # successful connect(); reconnects don't re-start the queues
        # (they're idempotent state machines).
        self._queues_started = False
        self._queues_started_lock = threading.Lock()

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

        # ── Radar pipeline (passive_radar only) ────────────────────────
        self._radar_pipeline: RadarPipeline | None = None
        if sensor.sensor_type == "passive_radar":
            from citrasense.pipelines.radar import build_radar_pipeline

            self._radar_pipeline = build_radar_pipeline()

        if sensor.sensor_type == "telescope" and hardware_adapter is not None:
            from citrasense.sensors.telescope.managers.alignment_manager import AlignmentManager
            from citrasense.sensors.telescope.managers.autofocus_manager import AutofocusManager
            from citrasense.sensors.telescope.managers.homing_manager import HomingManager

            if self._sensor_config is None:
                raise RuntimeError(
                    f"Telescope sensor {self.sensor_id!r} has no SensorConfig in settings; "
                    "cannot build AutofocusManager / AlignmentManager"
                )

            self.autofocus_manager = AutofocusManager(
                self.logger,
                hardware_adapter,
                settings,
                sensor_id=self.sensor_id,
                sensor_config=self._sensor_config,
                imaging_queue=self.acquisition_queue,
                location_service=location_service,
                preview_bus=preview_bus,
            )
            self.alignment_manager = AlignmentManager(
                self.logger,
                hardware_adapter,
                settings,
                sensor_id=self.sensor_id,
                sensor_config=self._sensor_config,
                imaging_queue=self.acquisition_queue,
                safety_monitor=safety_monitor,
                location_service=location_service,
                preview_bus=preview_bus,
            )
            self.homing_manager = HomingManager(
                self.logger,
                hardware_adapter,
                imaging_queue=self.acquisition_queue,
                sensor_id=self.sensor_id,
            )

    def set_dispatcher(self, dispatcher: TaskDispatcher) -> None:
        """Wire the parent dispatcher after construction (resolves circular dependency)."""
        self._dispatcher = dispatcher

    # ── Init-state ─────────────────────────────────────────────────────

    @property
    def init_state(self) -> SensorInitState:
        """Current hardware-init lifecycle state (thread-safe read)."""
        with self._init_state_lock:
            return self._init_state

    @property
    def init_error(self) -> str | None:
        """Last init-error message, or ``None`` when healthy (thread-safe read)."""
        with self._init_state_lock:
            return self._init_error

    @property
    def is_ready(self) -> bool:
        """True when the dispatcher should route tasks here.

        Equivalent to ``init_state == 'connected'``; named to read
        naturally in dispatcher gates (``if not rt.is_ready: continue``).
        """
        return self.init_state == "connected"

    def _set_init_state(self, state: SensorInitState, error: str | None = None) -> None:
        """Atomically transition init state and fire the on-change callback.

        The callback is invoked **outside** the state lock so listeners
        (toasts, status broadcasts) can call back into the runtime
        without risking deadlock.  Errors raised by the callback are
        logged and swallowed — a flaky toast handler must never break
        the connect pipeline.
        """
        with self._init_state_lock:
            prev = self._init_state
            self._init_state = state
            self._init_error = error if state in ("failed", "timed_out") else None
        if prev != state:
            self.logger.debug("init_state: %s -> %s%s", prev, state, f" ({error})" if error else "")
        cb = self.on_init_state_change
        if cb is None:
            return
        try:
            cb(state, error)
        except Exception:
            self.logger.warning("on_init_state_change callback failed", exc_info=True)

    def mark_connecting(self) -> None:
        """Flip to ``connecting`` (called by the init worker before ``connect()``)."""
        self._set_init_state("connecting", None)

    def mark_connected(self) -> None:
        """Flip to ``connected`` (called by the init worker on success)."""
        self._set_init_state("connected", None)
        self._ensure_queues_started()

    def mark_failed(self, error: str) -> None:
        """Flip to ``failed`` with an operator-visible error string."""
        self._set_init_state("failed", error)

    def mark_timed_out(self, error: str) -> None:
        """Flip to ``timed_out`` (the watchdog gave up; adapter thread may still be alive)."""
        self._set_init_state("timed_out", error)

    def mark_disconnected(self) -> None:
        """Reset to ``pending`` (called when the operator hits Disconnect)."""
        self._set_init_state("pending", None)

    def _ensure_queues_started(self) -> None:
        """Start the queue trio + streaming subscription on first ``connected``.

        Idempotent — every reconnect calls this, but the queues / bus
        subscription are only spun up once.  Reconnects keep the same
        worker threads alive, so an in-flight imaging task isn't dropped
        when the adapter blips.
        """
        with self._queues_started_lock:
            if self._queues_started:
                return
            self._queues_started = True
        self.acquisition_queue.start()
        self.processing_queue.start()
        self.upload_queue.start()
        self._subscribe_streaming()
        self._start_streaming_sensor()

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

    def _on_streaming_event(self, subject: str, event: BaseModel) -> None:
        """Route an acquisition event into the correct modality pipeline.

        Today only ``passive_radar`` emits on
        ``sensors.{sensor_id}.events.acquisition``.  Future streaming
        modalities can plug in here without touching the sensor itself.
        """
        del subject
        if self.sensor_type == "passive_radar":
            self._dispatch_radar_event(event)
            return
        self.logger.debug("Streaming event from non-radar sensor: %s", type(event).__name__)

    def _dispatch_radar_event(self, event: BaseModel) -> None:
        """Build a RadarProcessingContext and hand it off to the processing queue."""
        if self._radar_pipeline is None:
            return
        from citrasense.pipelines.radar import radar_artifact_dir
        from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext

        # Event is a RadarObservationEvent (pydantic BaseModel).
        payload = getattr(event, "payload", None)
        timestamp: datetime | None = getattr(event, "timestamp", None)
        if payload is None or timestamp is None:
            self.logger.warning("Unexpected event shape on acquisition bus: %s", type(event).__name__)
            return

        sensor = self.sensor
        antenna_id = getattr(sensor, "citra_antenna_id", "") or ""
        detection_min_snr_db = float(getattr(sensor, "_detection_min_snr_db", 0.0))
        forward_only_tasked = bool(getattr(sensor, "_forward_only_tasked", False))

        base_dir = self.settings.directories.processing_dir if self.settings.directories else None
        artifact_dir = radar_artifact_dir(base_dir, self.sensor_id) if base_dir else None

        ctx = RadarProcessingContext(
            sensor_id=self.sensor_id,
            event=event,  # type: ignore[arg-type]
            antenna_id=antenna_id,
            api_client=self.api_client,
            artifact_dir=artifact_dir or self.settings.directories.data_dir / "radar" / self.sensor_id,
            detection_min_snr_db=detection_min_snr_db,
            forward_only_tasked_satellites=forward_only_tasked,
            task_index=self._dispatcher,
            logger=self.logger,
        )

        self.processing_queue.submit_radar_event(ctx, self._radar_pipeline, self._on_radar_processed)

    def _on_radar_processed(self, ctx: RadarProcessingContext, upload_ready: bool) -> None:
        """Callback from the processing queue after a radar observation finishes.

        On success (``upload_ready``), hand the formatter's payload off
        to the upload queue.  On drop, just log and move on — the
        artifact writer has already persisted the raw observation.
        """
        if not upload_ready:
            if ctx.drop_reason:
                self.logger.debug("Dropped radar observation for %s: %s", ctx.sensor_id, ctx.drop_reason)
            return
        if ctx.upload_payload is None:
            self.logger.warning("Radar pipeline reported upload_ready without a payload for %s", ctx.sensor_id)
            return
        self.upload_queue.submit_radar_observation(
            sensor_id=ctx.sensor_id,
            payload=ctx.upload_payload,
            api_client=self.api_client,
            settings=self.settings,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Bring the runtime online from the dispatcher.

        Historically this started the queue trio + streaming
        subscription unconditionally.  Under the parallel/async-init
        model the dispatcher now starts every runtime up front in
        ``pending`` state, *before* the per-sensor ``connect()`` worker
        has run, so we can't blindly start the queues here — the
        adapter may not be alive yet.  Instead, the queue trio is
        started lazily by :meth:`mark_connected` (see
        :meth:`_ensure_queues_started`).  If the sensor is already
        connected by the time the dispatcher starts (e.g. tests that
        flip the state synchronously), the call below is the no-op
        idempotent path.
        """
        if self.is_ready:
            self._ensure_queues_started()

    def stop(self) -> None:
        self._stop_streaming_sensor()
        self.acquisition_queue.stop()
        self.processing_queue.stop()
        self.upload_queue.stop()
        if self._streaming_sub:
            self._streaming_sub.unsubscribe()
            self._streaming_sub = None
        with self._queues_started_lock:
            self._queues_started = False

    def _start_streaming_sensor(self) -> None:
        """Ask a STREAMING sensor to begin publishing to the bus.

        Called after :meth:`_subscribe_streaming` so the runtime is
        ready to receive events the moment the sensor starts pushing
        them.  Safe to call on ON_DEMAND sensors — they raise
        :class:`NotImplementedError` on ``start_stream`` and we treat
        that as the expected signal that there's nothing to do.
        """
        if not self._sensor_bus:
            return
        from citrasense.sensors.abstract_sensor import AcquisitionContext, SensorAcquisitionMode

        caps = self.sensor.get_capabilities()
        if caps.acquisition_mode != SensorAcquisitionMode.STREAMING:
            return
        try:
            self.sensor.start_stream(self._sensor_bus, AcquisitionContext())
        except NotImplementedError:
            pass
        except Exception as exc:
            self.logger.warning("start_stream raised for %s: %s", self.sensor_id, exc)

    def _stop_streaming_sensor(self) -> None:
        from citrasense.sensors.abstract_sensor import SensorAcquisitionMode

        try:
            caps = self.sensor.get_capabilities()
        except Exception:
            return
        if caps.acquisition_mode != SensorAcquisitionMode.STREAMING:
            return
        try:
            self.sensor.stop_stream()
        except NotImplementedError:
            pass
        except Exception as exc:
            self.logger.debug("stop_stream raised for %s: %s", self.sensor_id, exc)
