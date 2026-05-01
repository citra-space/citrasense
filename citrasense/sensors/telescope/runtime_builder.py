"""Telescope-specific runtime builder.

Owns the build-and-wire steps that used to live on
``CitraSenseDaemon._build_telescope_runtime`` and
``CitraSenseDaemon._connect_telescope_runtime``.  Co-locating with
:mod:`citrasense.sensors.telescope.telescope_sensor` means the
sensor type, its session managers, and its construction logic all
live in the same directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from citrasense.sensors.runtime_builder import BuildContext
from citrasense.sensors.sensor_runtime import SensorRuntime

if TYPE_CHECKING:
    from citrasense.sensors.abstract_sensor import AbstractSensor
    from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor


class TelescopeRuntimeBuilder:
    """Build + post-connect-wire a telescope :class:`SensorRuntime`.

    This is the heaviest builder by far — telescope sensors have
    pointing models, calibration backends, observing-session
    machinery, and cable-wrap safety state to migrate.  The split
    between :meth:`build` (synchronous, runs before connect) and
    :meth:`connect_post_wiring` (runs on the init worker after
    connect) is what makes the parallel/async init from issue #339
    safe — anything that depends on a *live* adapter must wait until
    after ``adapter.connect()`` succeeds.
    """

    def __init__(self, ctx: BuildContext) -> None:
        self.ctx = ctx

    def build(self, sensor: AbstractSensor) -> SensorRuntime:
        """Construct a telescope SensorRuntime in ``init_state="pending"``.

        Synchronous and side-effect-light: builds the runtime, wires
        site managers (observing session, self-tasking, calibration),
        registers it with the dispatcher, and installs the
        on_init_state_change toast callback.  Does **not** call
        ``adapter.connect()`` — that runs later in
        :meth:`connect_post_wiring` on the init executor.

        ``citra_record`` is already stamped onto the sensor by the
        ground-station resolution step in
        :func:`citrasense.sensors.init_orchestrator.resolve_canonical_ground_station`,
        so the API record / groundStationId rejection rules have
        already been applied by the time this method runs.
        """
        ctx = self.ctx
        telescope_sensor = cast("TelescopeSensor", sensor)

        adapter = telescope_sensor.adapter
        sensor_cfg = ctx.settings.get_sensor_config(telescope_sensor.sensor_id)
        assert sensor_cfg is not None, f"No SensorConfig for {telescope_sensor.sensor_id}"

        citra_telescope_record = telescope_sensor.citra_record
        assert citra_telescope_record is not None, (
            f"TelescopeRuntimeBuilder.build called before resolve_canonical_ground_station "
            f"stamped citra_record for {telescope_sensor.sensor_id}"
        )
        ground_station = ctx.ground_station
        assert ground_station is not None, "ground_station must be resolved before building telescope runtimes"

        ctx.logger.info(
            "Building runtime for telescope sensor %s (API id: %s)",
            telescope_sensor.sensor_id,
            sensor_cfg.citra_sensor_id,
        )

        if ctx.location_service:
            adapter.set_location_service(ctx.location_service)

        # Stamp slew rate / telescope_record / elset_cache on the adapter
        # eagerly so the connect path doesn't need to redo this work; the
        # adapter doesn't *use* them until after connect, but having them
        # set early means the web UI can read them before connect finishes.
        adapter.scope_slew_rate_degrees_per_second = citra_telescope_record["maxSlewRate"]
        adapter.telescope_record = citra_telescope_record
        adapter.elset_cache = ctx.elset_cache

        # Create SensorRuntime. Each runtime builds its own PipelineRegistry
        # (and CalibrationProcessor) so two telescopes don't share a single
        # processor whose calibration_library gets clobbered on reconnect.
        sensor_id = telescope_sensor.sensor_id
        on_annotated = ctx.on_annotated_image
        telescope_runtime = SensorRuntime(
            telescope_sensor,
            logger=ctx.logger,
            settings=ctx.settings,
            api_client=ctx.api_client,
            hardware_adapter=adapter,
            elset_cache=ctx.elset_cache,
            apass_catalog=ctx.apass_catalog,
            location_service=ctx.location_service,
            telescope_record=citra_telescope_record,
            ground_station=ground_station,
            on_annotated_image=lambda path, _sid=sensor_id: on_annotated(path, _sid),
            preview_bus=ctx.preview_bus,
            task_index=ctx.task_index,
            safety_monitor=ctx.safety_monitor,
            sensor_bus=ctx.sensor_bus,
        )

        if ctx.init_state_callback_factory is not None:
            telescope_runtime.on_init_state_change = ctx.init_state_callback_factory(sensor_id)

        ctx.task_dispatcher.register_runtime(telescope_runtime)

        if ctx.web_server and telescope_runtime.autofocus_manager:
            telescope_runtime.autofocus_manager.on_toast = ctx.web_server.send_toast

        from citrasense.sensors.telescope.observing_session import ObservingSessionManager
        from citrasense.sensors.telescope.self_tasking_manager import SelfTaskingManager

        def _get_location_tuple() -> tuple[float, float] | None:
            if not ctx.location_service:
                return None
            loc = ctx.location_service.get_current_location()
            if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                return loc["latitude"], loc["longitude"]
            return None

        can_park = adapter.supports_park()
        alignment_mgr = telescope_runtime.alignment_manager
        rt_logger = telescope_runtime.logger
        osm = ObservingSessionManager(
            sensor_config=sensor_cfg,
            logger=rt_logger,
            get_location=_get_location_tuple,
            request_autofocus=telescope_runtime.autofocus_manager.request,
            is_autofocus_running=telescope_runtime.autofocus_manager.is_running,
            is_imaging_idle=telescope_runtime.acquisition_queue.is_idle,
            are_queues_idle=telescope_runtime.are_queues_idle,
            park_mount=adapter.park_mount if can_park else None,
            unpark_mount=adapter.unpark_mount if can_park else None,
            request_pointing_calibration=alignment_mgr.request_calibration,
            is_pointing_calibration_running=alignment_mgr.is_calibrating,
        )
        stm = SelfTaskingManager(
            api_client=ctx.api_client,
            sensor_config=sensor_cfg,
            logger=rt_logger,
            ground_station_id=ground_station["id"],
            sensor_id=citra_telescope_record["id"],
            get_session_state=lambda osm=osm: osm.state,
            get_observing_window=lambda osm=osm: osm.observing_window,
        )
        telescope_runtime.observing_session_manager = osm
        telescope_runtime.self_tasking_manager = stm

        # Calibration library is independent of adapter state — bind it
        # now so the CalibrationProcessor has a library before any
        # observation is processed.
        from citrasense.calibration.calibration_library import CalibrationLibrary

        library = CalibrationLibrary(root=ctx.settings.directories.calibration_dir)
        telescope_runtime.attach_calibration_library(library)

        return telescope_runtime

    def connect_post_wiring(self, runtime: SensorRuntime) -> None:
        """Run the post-connect hardware wiring for a telescope.

        Called from the init-executor worker AFTER ``adapter.connect()``
        returns True.  Anything here may safely assume the adapter is
        live: filter discovery, pointing-model wiring, calibration
        backend selection, safety-check registration.

        Errors raised here propagate up to the orchestrator which
        flips the runtime to ``failed`` — so a transient issue in
        e.g. filter sync doesn't silently leave the sensor in an
        inconsistent ``connected`` state.
        """
        ctx = self.ctx
        adapter = runtime.hardware_adapter
        assert adapter is not None
        telescope_sensor = cast("TelescopeSensor", runtime.sensor)
        sensor_cfg = ctx.settings.get_sensor_config(telescope_sensor.sensor_id)
        assert sensor_cfg is not None
        assert ctx.safety_monitor is not None

        ctx.logger.info(
            "Hardware connected for %s. Slew rate: %s deg/sec",
            telescope_sensor.sensor_id,
            adapter.scope_slew_rate_degrees_per_second,
        )

        if ctx.location_service:
            ctx.location_service.set_hardware_adapter_gps_provider(
                adapter.get_gps_location, sensor_id=telescope_sensor.sensor_id
            )

        ctx.save_filter_config(telescope_sensor)
        ctx.sync_filters_to_backend(telescope_sensor)

        adapter_name = type(adapter).__name__
        slew_rate = adapter.scope_slew_rate_degrees_per_second
        filter_cfg = adapter.get_filter_config()
        enabled = sum(1 for f in filter_cfg.values() if f.get("enabled", False)) if filter_cfg else 0
        gs_name = (ctx.ground_station or {}).get("name", "?")
        scope_name = (telescope_sensor.citra_record or {}).get("name", "?")
        ctx.logger.info(
            f"Hardware ready ({telescope_sensor.sensor_id}): adapter={adapter_name}, "
            f"slew={slew_rate} deg/s, filters={enabled}/{len(filter_cfg)}, "
            f"station={gs_name}, telescope={scope_name}"
        )

        if not adapter.is_mount_homed():
            ctx.logger.info(
                "Mount %s is not at home position — home via web UI if GoTo fails",
                telescope_sensor.sensor_id,
            )

        # Register telescope-specific safety checks (cable wrap).
        # Cable-wrap state lives next to the rest of the daemon's data so
        # custom data directories are honored.
        data_dir = ctx.settings.directories.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        state_file = data_dir / f"cable_wrap_state_{telescope_sensor.sensor_id}.json"

        # Legacy single-telescope state file lived in platformdirs.user_data_dir;
        # migrate it into the current data_dir the first time we boot.
        # Two legacy locations exist because earlier versions mixed
        # ``appauthor="citrasense"`` and the canonical
        # ``APP_AUTHOR="citra-space"`` from ``citrasense.constants``.
        import platformdirs

        from citrasense.constants import APP_AUTHOR, APP_NAME

        legacy_candidates = [
            data_dir / "cable_wrap_state.json",
            Path(platformdirs.user_data_dir(APP_NAME, appauthor=APP_AUTHOR)) / "cable_wrap_state.json",
            Path(platformdirs.user_data_dir("citrasense", appauthor="citrasense")) / "cable_wrap_state.json",
        ]
        for legacy_state in legacy_candidates:
            if legacy_state.exists() and not state_file.exists():
                try:
                    legacy_state.rename(state_file)
                except OSError as exc:
                    ctx.logger.warning(
                        "Failed to migrate cable wrap state file %s → %s: %s",
                        legacy_state,
                        state_file,
                        exc,
                    )
                else:
                    ctx.logger.info("Migrated cable wrap state file → %s", state_file)
                break

        telescope_sensor.register_safety_checks(ctx.safety_monitor, logger=ctx.logger, state_file=state_file)

        if adapter.pointing_model and runtime.alignment_manager:
            runtime.alignment_manager.set_pointing_model(adapter.pointing_model)

        # CalibrationManager wiring depends on adapter capabilities probed
        # post-connect (camera presence, flat-automation support).  The
        # CalibrationLibrary itself was attached eagerly in
        # ``build()`` so the processor can apply existing
        # masters from the moment the runtime exists.
        from citrasense.calibration.flat_capture_backend import (
            DirectCameraFlatBackend,
            FlatCaptureBackend,
        )

        library = runtime.calibration_library
        flat_backend: FlatCaptureBackend | None = None
        if adapter.supports_direct_camera_control() and adapter.camera is not None:
            flat_backend = DirectCameraFlatBackend(adapter.camera)
        elif adapter.supports_flat_automation():
            try:
                from citrasense.calibration.nina_trained_flat_backend import NinaTrainedFlatBackend

                flat_backend = NinaTrainedFlatBackend(adapter)  # type: ignore[arg-type]
            except Exception as e:
                ctx.logger.warning(
                    "Could not wire NinaTrainedFlatBackend for %s: %s",
                    telescope_sensor.sensor_id,
                    e,
                )

        needs_manager = (
            adapter.supports_direct_camera_control() and adapter.camera is not None
        ) or flat_backend is not None
        if needs_manager:
            from citrasense.sensors.telescope.managers.calibration_manager import CalibrationManager

            loc_svc = ctx.location_service

            def _loc_provider() -> dict[str, float] | None:
                if loc_svc is None:
                    return None
                try:
                    return loc_svc.get_current_location()
                except Exception:
                    return None

            runtime.calibration_manager = CalibrationManager(
                runtime.logger,
                adapter,
                library,
                imaging_queue=runtime.acquisition_queue,
                flat_backend=flat_backend,
                settings=ctx.settings,
                sensor_id=telescope_sensor.sensor_id,
                location_provider=_loc_provider,
            )

        ctx.logger.info("Telescope sensor %s initialized successfully!", telescope_sensor.sensor_id)
