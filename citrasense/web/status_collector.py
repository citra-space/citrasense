"""Collects runtime state from the daemon and populates a SystemStatus in place.

Extracted from CitraSenseWebApp to keep the web app a thin container.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from citrasense.constants import DEV_APP_URL, PROD_APP_URL
from citrasense.logging import CITRASENSE_LOGGER
from citrasense.web.helpers import _gps_fix_to_dict, _resolve_autofocus_target_name
from citrasense.web.models import SystemStatus


class StatusCollector:
    """Populate a SystemStatus in place from the current daemon state."""

    def __init__(self, daemon: Any) -> None:
        self.daemon = daemon

    def collect(self, status: SystemStatus) -> None:
        """Update `status` in-place from the daemon's current state."""
        if not self.daemon:
            return

        _t0 = time.perf_counter()
        _prev = _t0
        _breakdown: dict[str, float] = {}

        def _mark(name: str) -> None:
            nonlocal _prev
            now = time.perf_counter()
            _breakdown[name] = round((now - _prev) * 1000, 2)
            _prev = now

        try:
            sm = getattr(self.daemon, "sensor_manager", None)
            td = getattr(self.daemon, "task_dispatcher", None)

            # Build per-sensor skeleton for ALL sensors.
            #
            # The ``SystemStatus`` instance lives for the lifetime of the web
            # app (see ``CitraSenseWebApp.__init__``), so entries in
            # ``status.sensors`` would otherwise accumulate forever — each
            # poll only writes keys for *currently* registered sensors, but
            # never deletes keys for ones that disappeared on a config
            # reload.  Removing a telescope from the config and hitting
            # "Save & Reload" would tear down the runtime on the backend
            # but leave a ghost entry in ``status.sensors`` that the
            # monitoring UI kept rendering.  Reseed the dict against the
            # current sensor manager before re-populating.
            if sm:
                live_ids = {s.sensor_id for s in sm}
                for stale_id in list(status.sensors.keys() - live_ids):
                    del status.sensors[stale_id]
                for s in sm:
                    status.sensors[s.sensor_id] = {
                        "type": s.sensor_type,
                        "connected": s.is_connected(),
                        "name": getattr(s, "name", s.sensor_id),
                        "adapter_key": getattr(s, "adapter_key", None),
                    }

            # Populate site-wide tasks_by_stage BEFORE enrichment so each
            # sensor can filter the site map down to its own tasks.
            if td:
                status.tasks_by_stage = td.get_tasks_by_stage()
            else:
                status.tasks_by_stage = None

            # Enrich each telescope sensor with hw/manager/session state
            self._enrich_sensors(status, sm, td)
            _mark("sensor_enrichment")

            # Site-level: task dispatcher aggregate.  No scalar
            # ``current_task`` — consumers use ``current_task_ids`` (keyed by
            # sensor) and the per-sensor ``sensors[sid].current_task``
            # populated by ``_enrich_sensors``.
            if td:
                status.current_task_ids = dict(td.current_task_ids)
                status.tasks_pending = td.pending_task_count
                status.processing_active = td.is_processing_active()

            # Per-sensor and aggregate ``system_busy`` flags.  The site-wide
            # aggregate is retained for the top-level status bar, but
            # per-sensor manual-control gating (preview, jog, filter move
            # etc.) now reads ``status.sensors[sid].system_busy`` so one
            # sensor imaging doesn't freeze the other sensor's card.
            busy_reasons: list[str] = []
            if td and td.is_processing_active():
                busy_reasons.append("processing")
            for sd in status.sensors.values():
                if sd.get("type") != "telescope":
                    sd["system_busy"] = False
                    sd["system_busy_reason"] = ""
                    continue
                sensor_reasons: list[str] = []
                if not sd.get("acquisition_idle", True):
                    sensor_reasons.append("imaging")
                if sd.get("alignment_running"):
                    sensor_reasons.append("alignment")
                if sd.get("autofocus_running"):
                    sensor_reasons.append("autofocus")
                if sd.get("pointing_calibration_running"):
                    sensor_reasons.append("calibration")
                if sd.get("mount_homing"):
                    sensor_reasons.append("homing")
                sd["system_busy"] = bool(sensor_reasons)
                sd["system_busy_reason"] = ", ".join(sensor_reasons)
                busy_reasons.extend(sensor_reasons)
            status.system_busy = bool(busy_reasons)
            status.system_busy_reason = ", ".join(busy_reasons)
            _mark("task_dispatcher")

            # Per-sensor autofocus/alignment timing lives in
            # ``status.sensors[sid]`` (populated by ``_enrich_sensors`` →
            # ``_add_autofocus_alignment_fields``). No site-level
            # "representative sensor" copy — site-level duplicates were
            # dropped in the multi-sensor cleanup.
            _mark("autofocus")

            # Time sync status
            if hasattr(self.daemon, "time_monitor") and self.daemon.time_monitor:
                health = self.daemon.time_monitor.get_current_health()
                status.time_health = health.to_dict() if health else None
            else:
                status.time_health = None

            # GPS status
            if hasattr(self.daemon, "location_service") and self.daemon.location_service:
                gpsd_fix = self.daemon.location_service.get_gpsd_fix(allow_blocking=False)
                status.gpsd_fix = _gps_fix_to_dict(gpsd_fix) if gpsd_fix else None

                adapter_fix = self.daemon.location_service.get_equipment_gps()
                status.adapter_gps = _gps_fix_to_dict(adapter_fix) if adapter_fix else None
            else:
                status.gpsd_fix = None
                status.adapter_gps = None

            # Ground station
            if hasattr(self.daemon, "ground_station") and self.daemon.ground_station:
                gs_record = self.daemon.ground_station
                gs_id = gs_record.get("id")
                gs_name = gs_record.get("name", "Unknown")

                api_host = self.daemon.settings.host
                base_url = DEV_APP_URL if "dev." in api_host else PROD_APP_URL

                status.ground_station_id = gs_id
                status.ground_station_name = gs_name
                status.ground_station_url = f"{base_url}/ground-stations/{gs_id}" if gs_id else None
                status.ground_station_latitude = gs_record.get("latitude")
                status.ground_station_longitude = gs_record.get("longitude")
                status.ground_station_altitude = gs_record.get("altitude")

            # Resolve active operating location
            best_gps = status.gpsd_fix if status.gpsd_fix and status.gpsd_fix.get("is_strong") else None
            if not best_gps and status.adapter_gps and status.adapter_gps.get("is_strong"):
                best_gps = status.adapter_gps
            gps = best_gps
            if gps and gps.get("latitude") is not None and gps.get("longitude") is not None:
                status.location_source = "gps"
                status.location_latitude = gps["latitude"]
                status.location_longitude = gps["longitude"]
                status.location_altitude = gps.get("altitude")
            elif status.ground_station_latitude is not None and status.ground_station_longitude is not None:
                status.location_source = "ground_station"
                status.location_latitude = status.ground_station_latitude
                status.location_longitude = status.ground_station_longitude
                status.location_altitude = status.ground_station_altitude
            else:
                status.location_source = None
                status.location_latitude = None
                status.location_longitude = None
                status.location_altitude = None

            _mark("time_gps_location")

            # Aggregate missing dependencies from all telescope adapters
            status.missing_dependencies = []
            if sm:
                for s in sm.iter_by_type("telescope"):
                    adapter = getattr(s, "adapter", None)
                    if adapter:
                        try:
                            status.missing_dependencies.extend(adapter.get_missing_dependencies())
                        except Exception as e:
                            CITRASENSE_LOGGER.debug(f"Could not read hardware missing dependencies: {e}")
            status.missing_dependencies.extend(getattr(self.daemon, "_processor_dep_issues", []) or [])

            # Active processors (site-level)
            if hasattr(self.daemon, "processor_registry") and self.daemon.processor_registry:
                status.active_processors = [p.name for p in self.daemon.processor_registry.processors]
            else:
                status.active_processors = []

            # Aggregate pipeline stats across all runtimes
            if td:
                agg_imaging: dict[str, int] = {}
                agg_processing: dict[str, int] = {}
                agg_uploading: dict[str, int] = {}
                for rt in td.iter_runtimes():
                    for k, v in rt.acquisition_queue.get_stats().items():
                        agg_imaging[k] = agg_imaging.get(k, 0) + v
                    for k, v in rt.processing_queue.get_stats().items():
                        agg_processing[k] = agg_processing.get(k, 0) + v
                    for k, v in rt.upload_queue.get_stats().items():
                        agg_uploading[k] = agg_uploading.get(k, 0) + v
                status.pipeline_stats = {
                    "imaging": agg_imaging,
                    "processing": agg_processing,
                    "uploading": agg_uploading,
                    "tasks": td.get_task_stats(),
                }
            else:
                status.pipeline_stats = None
            if td:
                agg_proc_stats: dict[str, dict[str, Any]] = {}
                for rt in td.iter_runtimes():
                    # Telescope modality: PipelineRegistry lives on
                    # ``processor_registry``.  Radar modality: the
                    # purpose-built :class:`RadarPipeline` lives on
                    # ``_radar_pipeline`` and exposes the same
                    # ``get_processor_stats()`` shape so we can treat
                    # the two uniformly here — keeps the UI's
                    # ``status.pipeline_stats.processors`` single-
                    # shape regardless of modality.
                    sources: list[Any] = []
                    reg = getattr(rt, "processor_registry", None)
                    if reg is not None:
                        sources.append(reg)
                    radar_pipeline = getattr(rt, "_radar_pipeline", None)
                    if radar_pipeline is not None:
                        sources.append(radar_pipeline)
                    for src in sources:
                        for name, stats in src.get_processor_stats().items():
                            cur = agg_proc_stats.setdefault(
                                name, {"runs": 0, "failures": 0, "last_failure_reason": None}
                            )
                            cur["runs"] += stats.get("runs", 0)
                            cur["failures"] += stats.get("failures", 0)
                            if stats.get("last_failure_reason"):
                                cur["last_failure_reason"] = stats["last_failure_reason"]
                if agg_proc_stats:
                    if status.pipeline_stats is None:
                        status.pipeline_stats = {}
                    status.pipeline_stats["processors"] = agg_proc_stats

            _mark("pipeline")

            # Safety monitor status (site-level)
            if hasattr(self.daemon, "safety_monitor") and self.daemon.safety_monitor:
                try:
                    status.safety_status = self.daemon.safety_monitor.get_status()
                except Exception:
                    status.safety_status = None
            else:
                status.safety_status = None

            # Elset cache health (site-level)
            if hasattr(self.daemon, "elset_cache") and self.daemon.elset_cache:
                status.elset_health = self.daemon.elset_cache.get_health()
            else:
                status.elset_health = None

            _mark("safety_elset")

            # The "latest annotated image" used to be aggregated into a
            # single site-level URL here.  In the multi-sensor world that
            # concept doesn't make sense, so the per-sensor URL lives on
            # ``status.sensors[sid]['latest_task_image_url']`` (populated by
            # :meth:`_collect_telescope_optics`) and the fullscreen preview
            # modal picks the one for the sensor the user clicked.

            _mark("optics_calibration")

            status.last_update = datetime.now().isoformat()
            status.status_collection_ms = round((time.perf_counter() - _t0) * 1000, 2)
            status.status_collection_breakdown = _breakdown

        except Exception:
            CITRASENSE_LOGGER.exception("Error updating status")

    # ── Per-sensor enrichment ─────────────────────────────────────────

    def _enrich_sensors(
        self,
        status: SystemStatus,
        sm: Any,
        td: Any,
    ) -> None:
        """Independently collect hardware, manager, and session state for every sensor."""
        if not status.sensors:
            return

        site_tasks = status.tasks_by_stage or {}

        for sensor_id, sd in status.sensors.items():
            s_runtime = td.get_runtime(sensor_id) if td else None

            # Per-sensor task state
            if td:
                sd["current_task"] = td.current_task_ids.get(sensor_id)
                sd["processing_active"] = not s_runtime.paused if s_runtime else False
            else:
                sd["current_task"] = None
                sd["processing_active"] = False

            # Per-runtime pipeline stats.  For radar sensors we also
            # flatten the :class:`RadarPipeline`'s per-processor stats
            # into the same ``processors`` slot telescope sensors use —
            # the detail template can then render filter/formatter/
            # writer success bars with the same component as
            # calibration/plate_solver/photometry.
            if s_runtime:
                sd["pipeline_stats"] = {
                    "imaging": s_runtime.acquisition_queue.get_stats(),
                    "processing": s_runtime.processing_queue.get_stats(),
                    "uploading": s_runtime.upload_queue.get_stats(),
                }
                sensor_proc_stats: dict[str, dict[str, Any]] = {}
                reg = getattr(s_runtime, "processor_registry", None)
                if reg is not None:
                    sensor_proc_stats.update(reg.get_processor_stats())
                radar_pipeline = getattr(s_runtime, "_radar_pipeline", None)
                if radar_pipeline is not None:
                    sensor_proc_stats.update(radar_pipeline.get_processor_stats())
                if sensor_proc_stats:
                    sd["pipeline_stats"]["processors"] = sensor_proc_stats
                sd["acquisition_idle"] = s_runtime.acquisition_queue.is_idle()
            else:
                sd["pipeline_stats"] = None
                sd["acquisition_idle"] = True

            if sd.get("type") == "passive_radar":
                try:
                    sensor = sm.get(sensor_id) if sm else None
                except KeyError:
                    sensor = None
                if sensor is not None:
                    # Use the status tick as a rising edge for staleness
                    # toasts — avoids a second daemon timer just for this.
                    if hasattr(sensor, "poll_staleness"):
                        try:
                            sensor.poll_staleness()
                        except Exception:
                            pass
                    if hasattr(sensor, "get_live_status"):
                        try:
                            sd["radar"] = sensor.get_live_status()
                        except Exception:
                            sd["radar"] = None
                continue

            if sd.get("type") != "telescope":
                continue

            # Resolve the actual sensor and adapter objects
            try:
                sensor = sm.get(sensor_id) if sm else None
            except KeyError:
                sensor = None
            if sensor is None:
                continue
            adapter = getattr(sensor, "adapter", None)
            if adapter is None:
                continue

            # Attach Citra API record ID for task matching
            tr = getattr(sensor, "citra_record", None)
            if tr:
                sd["api_id"] = tr.get("id")

            # Per-task-stage filtering
            api_id = sd.get("api_id")
            sd["tasks_by_stage"] = {
                stage: [t for t in tasks if t.get("sensor_id") in (sensor_id, api_id)]
                for stage, tasks in site_tasks.items()
            }

            self._collect_telescope_hardware(sd, adapter)
            self._collect_telescope_managers(sd, s_runtime)
            self._collect_telescope_autofocus_timing(sd, s_runtime)
            self._collect_telescope_session(sd, sensor_id, s_runtime, td)
            self._collect_telescope_optics(sd, sensor_id, sensor, adapter, s_runtime)

    def _collect_telescope_hardware(self, sd: dict, adapter: Any) -> None:
        """Populate hardware connection and telemetry fields for one telescope."""
        mount = adapter.mount

        try:
            sd["telescope_connected"] = adapter.is_telescope_connected()
            if sd["telescope_connected"]:
                snap = mount.cached_state if mount is not None else None
                if snap is not None:
                    sd["telescope_ra"] = snap.ra_deg
                    sd["telescope_dec"] = snap.dec_deg
                    sd["telescope_az"] = snap.az_deg
                    sd["telescope_alt"] = snap.alt_deg
                    sd["mount_tracking"] = snap.is_tracking
                    sd["mount_slewing"] = snap.is_slewing
                else:
                    ra, dec = adapter.get_telescope_direction()
                    sd["telescope_ra"] = ra
                    sd["telescope_dec"] = dec
                    sd["mount_tracking"] = False
                    sd["mount_slewing"] = False
            else:
                sd["mount_tracking"] = False
                sd["mount_slewing"] = False
        except Exception:
            sd["telescope_connected"] = False
            sd["mount_tracking"] = False
            sd["mount_slewing"] = False

        try:
            sd["camera_connected"] = adapter.is_camera_connected()
            camera = adapter.camera
            if sd["camera_connected"] and camera is not None:
                sd["camera_temperature"] = camera.get_temperature()
            else:
                sd["camera_temperature"] = None
        except Exception:
            sd["camera_connected"] = False
            sd["camera_temperature"] = None

        try:
            sd["supports_direct_camera_control"] = adapter.supports_direct_camera_control()
        except Exception:
            sd["supports_direct_camera_control"] = False

        sd["supports_direct_mount_control"] = mount is not None and sd.get("telescope_connected", False)
        sd["supports_autofocus"] = adapter.supports_autofocus()
        sd["supports_hardware_safety_monitor"] = adapter.supports_hardware_safety_monitor

        try:
            pos = adapter.get_filter_position()
            sd["current_filter_position"] = pos
            if pos is not None and pos in adapter.filter_map:
                sd["current_filter_name"] = adapter.filter_map[pos].get("name")
            else:
                sd["current_filter_name"] = None
        except Exception:
            sd["current_filter_position"] = None
            sd["current_filter_name"] = None

        focuser = adapter.focuser
        if focuser is not None and focuser.is_connected():
            sd["focuser_connected"] = True
            try:
                sd["focuser_position"] = focuser.get_position()
            except Exception:
                sd["focuser_position"] = None
            try:
                sd["focuser_max_position"] = focuser.get_max_position()
            except Exception:
                sd["focuser_max_position"] = None
            try:
                sd["focuser_temperature"] = focuser.get_temperature()
            except Exception:
                sd["focuser_temperature"] = None
            try:
                sd["focuser_moving"] = focuser.is_moving()
            except Exception:
                sd["focuser_moving"] = False
        else:
            sd["focuser_connected"] = False
            sd["focuser_position"] = None
            sd["focuser_max_position"] = None
            sd["focuser_temperature"] = None
            sd["focuser_moving"] = False

        sd["supports_alignment"] = sd.get("camera_connected", False) and mount is not None

        if mount is not None and mount.cached_state is not None:
            sd["supports_manual_sync"] = mount.cached_mount_info.get("supports_sync", False)
            sd["mount_at_home"] = mount.cached_state.is_at_home
            try:
                h_limit, o_limit = mount.cached_limits
                sd["mount_horizon_limit"] = h_limit
                sd["mount_overhead_limit"] = o_limit
            except (TypeError, ValueError):
                sd["mount_horizon_limit"] = None
                sd["mount_overhead_limit"] = None
        else:
            sd.setdefault("supports_manual_sync", False)
            sd.setdefault("mount_at_home", False)
            sd.setdefault("mount_horizon_limit", None)
            sd.setdefault("mount_overhead_limit", None)

    def _collect_telescope_managers(self, sd: dict, runtime: Any) -> None:
        """Populate autofocus/alignment/homing manager state for one telescope."""
        if not runtime:
            return

        sd["mount_homing"] = runtime.homing_manager.is_running() or runtime.homing_manager.is_requested()
        sd["autofocus_requested"] = runtime.autofocus_manager.is_requested()
        sd["autofocus_running"] = runtime.autofocus_manager.is_running()
        sd["autofocus_progress"] = runtime.autofocus_manager.progress
        sd["autofocus_points"] = [{"pos": p, "hfr": h, "filter": f} for p, h, f in runtime.autofocus_manager.points]
        sd["autofocus_filter_results"] = runtime.autofocus_manager.filter_results
        sd["autofocus_last_result"] = runtime.autofocus_manager.last_result
        hfr_hist = runtime.autofocus_manager.hfr_history
        sd["hfr_history"] = [{"hfr": h, "ts": t, "filter": f} for h, t, f in hfr_hist]
        sd["last_hfr_median"] = hfr_hist[-1][0] if hfr_hist else None

        # Per-sensor HFR baseline (typed field on SensorConfig as of v8).
        sensor_cfg = (
            self.daemon.settings.get_sensor_config(runtime.sensor_id) if self.daemon and self.daemon.settings else None
        )
        if sensor_cfg:
            sd["hfr_baseline"] = sensor_cfg.hfr_baseline
        else:
            sd["hfr_baseline"] = None

        if sensor_cfg:
            sd["hfr_increase_percent"] = sensor_cfg.autofocus_hfr_increase_percent
            sd["hfr_refocus_enabled"] = sensor_cfg.autofocus_on_hfr_increase_enabled
            sd["hfr_sample_window"] = sensor_cfg.autofocus_hfr_sample_window

        sd["alignment_requested"] = runtime.alignment_manager.is_requested()
        sd["alignment_running"] = runtime.alignment_manager.is_running()
        sd["alignment_progress"] = runtime.alignment_manager.progress
        sd["pointing_calibration_running"] = runtime.alignment_manager.is_calibrating()
        sd["pointing_calibration_progress"] = runtime.alignment_manager.calibration_progress

    def _collect_telescope_autofocus_timing(self, sd: dict, runtime: Any) -> None:
        """Populate autofocus timing fields for one telescope.

        Resolves this runtime's ``SensorConfig`` strictly by ``sensor_id`` — no
        first-sensor fallback, which would report the wrong telescope's
        autofocus timestamps in a multi-sensor deployment.
        """
        if not self.daemon or not self.daemon.settings:
            return
        settings = self.daemon.settings
        sensor_id = getattr(runtime, "sensor_id", None)
        sc = settings.get_sensor_config(sensor_id) if sensor_id else None
        if sc is None:
            return
        afs = sc
        sd["last_autofocus_timestamp"] = getattr(afs, "last_autofocus_timestamp", None)
        sd["last_alignment_timestamp"] = getattr(afs, "last_alignment_timestamp", None)
        sd["autofocus_target_name"] = _resolve_autofocus_target_name(afs)

        if runtime and hasattr(runtime, "autofocus_manager"):
            sd["next_autofocus_minutes"] = runtime.autofocus_manager.get_next_autofocus_minutes()
        elif afs.scheduled_autofocus_enabled:
            last_ts = getattr(afs, "last_autofocus_timestamp", None)
            interval_minutes = afs.autofocus_interval_minutes
            if last_ts is not None:
                elapsed_minutes = (int(time.time()) - last_ts) / 60
                remaining = max(0, interval_minutes - elapsed_minutes)
                sd["next_autofocus_minutes"] = int(remaining)
            else:
                sd["next_autofocus_minutes"] = 0
        else:
            sd["next_autofocus_minutes"] = None

    def _collect_telescope_session(self, sd: dict, sensor_id: str, runtime: Any, td: Any) -> None:
        """Populate observing session, self-tasking, and processing-toggle state
        for one telescope.

        Templates read these fields off the per-sensor ``status.sensors[sid]``
        object so they never have to reach into the root ``config`` (where
        these fields no longer live after the per-sensor config migration).
        """
        sensor_cfg = self.daemon.settings.get_sensor_config(sensor_id) if self.daemon and self.daemon.settings else None
        sd["task_processing_paused"] = sensor_cfg.task_processing_paused if sensor_cfg else False
        sd["observing_session_enabled"] = sensor_cfg.observing_session_enabled if sensor_cfg else False
        sd["self_tasking_enabled"] = sensor_cfg.self_tasking_enabled if sensor_cfg else False
        sd["processors_enabled"] = sensor_cfg.processors_enabled if sensor_cfg else True
        sd["self_tasking_collection_type"] = sensor_cfg.self_tasking_collection_type if sensor_cfg else "Track"
        sd["self_tasking_include_orbit_regimes"] = list(
            sensor_cfg.self_tasking_include_orbit_regimes if sensor_cfg else []
        )
        sd["self_tasking_exclude_object_types"] = list(
            sensor_cfg.self_tasking_exclude_object_types if sensor_cfg else []
        )

        if runtime:
            osm = runtime.observing_session_manager
            stm = runtime.self_tasking_manager
            if osm:
                osm_dict = osm.status_dict()
                sd["observing_session_state"] = osm_dict.get("observing_session_state", "daytime")
                sd["session_activity"] = osm_dict.get("session_activity")
                sd["observing_session_threshold"] = osm_dict.get("observing_session_threshold", -12.0)
                sd["sun_altitude"] = osm_dict.get("sun_altitude")
                sd["dark_window_start"] = osm_dict.get("dark_window_start")
                sd["dark_window_end"] = osm_dict.get("dark_window_end")
            if stm:
                stm_dict = stm.status_dict()
                sd["last_batch_request"] = stm_dict.get("last_batch_request")
                sd["last_batch_created"] = stm_dict.get("last_batch_created")
                sd["next_request_seconds"] = stm_dict.get("next_request_seconds")

        # Automated scheduling from the Citra API record
        sm = getattr(self.daemon, "sensor_manager", None)
        if sm:
            try:
                s = sm.get(sensor_id)
                tr = getattr(s, "citra_record", None)
                sd["automated_scheduling"] = (tr or {}).get("automatedScheduling", False)
            except KeyError:
                sd["automated_scheduling"] = False

    def _collect_telescope_optics(self, sd: dict, sensor_id: str, sensor: Any, adapter: Any, runtime: Any) -> None:
        """Populate calibration, pointing model, and config health for one telescope."""
        sd["calibration_status"] = self._build_calibration_status(adapter, runtime)
        sd["pointing_model"] = adapter.get_pointing_model_status() if adapter else None
        sd["fov_short_deg"] = adapter.observed_fov_short_deg if adapter else None

        tr = getattr(sensor, "citra_record", None)
        sd["telescope_min_elevation"] = None
        if tr:
            min_el = tr.get("minElevation")
            if min_el is not None:
                try:
                    sd["telescope_min_elevation"] = float(min_el)
                except (TypeError, ValueError):
                    pass

        if tr and adapter:
            try:
                from citrasense.hardware.config_health import assess_config_health

                camera_info = adapter.get_camera_info()
                binning = adapter.get_current_binning()
                health = assess_config_health(
                    telescope_record=tr,
                    camera_info=camera_info,
                    binning=binning,
                    observed_pixel_scale=adapter.observed_pixel_scale_arcsec,
                    observed_fov_w=adapter.observed_fov_w_deg,
                    observed_fov_h=adapter.observed_fov_h_deg,
                    observed_slew_rate=adapter.observed_slew_rate_deg_per_s,
                    slew_rate_samples=adapter.slew_rate_tracker.count,
                )
                sd["config_health"] = health.to_dict()
            except Exception:
                sd["config_health"] = None
        else:
            sd["config_health"] = None

        paths = getattr(self.daemon, "latest_annotated_image_paths", {})
        ann_path = paths.get(sensor_id)
        if ann_path and Path(ann_path).exists():
            mtime_ns = Path(ann_path).stat().st_mtime_ns
            sd["latest_task_image_url"] = f"/api/task-preview/latest?sensor_id={sensor_id}&t={mtime_ns}"
        else:
            sd["latest_task_image_url"] = None

        # Per-sensor missing dependencies
        sd["missing_dependencies"] = []
        try:
            sd["missing_dependencies"].extend(adapter.get_missing_dependencies())
        except Exception:
            pass

    def _build_calibration_status(self, adapter: Any, runtime: Any) -> dict[str, Any] | None:
        """Build calibration status dict for SystemStatus."""
        if not self.daemon:
            return None
        lib = getattr(runtime, "calibration_library", None) if runtime else None
        if not lib or not adapter or not adapter.supports_direct_camera_control():
            return None

        camera = adapter.camera
        if not camera:
            return None

        profile = camera.get_calibration_profile()
        if not profile.calibration_applicable:
            return None

        cam_id = profile.camera_id
        gain = profile.current_gain or 0
        binning = profile.current_binning
        temperature = profile.current_temperature
        read_mode = profile.read_mode

        filter_name = ""
        filter_pos = adapter.get_filter_position() if hasattr(adapter, "get_filter_position") else None
        if filter_pos is not None and hasattr(adapter, "filter_map"):
            fdata = adapter.filter_map.get(filter_pos, {})
            filter_name = fdata.get("name", "")

        has_bias = lib.get_master_bias(cam_id, gain, binning, read_mode) is not None
        has_dark = (
            lib.get_master_dark(cam_id, gain, binning, temperature or 0.0, read_mode) is not None
            if temperature is not None
            else False
        )
        has_flat = (
            lib.get_master_flat(cam_id, gain, binning, filter_name, read_mode) is not None if filter_name else True
        )

        missing: list[str] = []
        if not has_bias:
            missing.append(f"bias (gain {gain}, bin {binning})")
        if not has_dark:
            temp_str = f"{temperature:.1f}°C" if temperature is not None else "unknown"
            missing.append(f"dark (at {temp_str})")
        if filter_name and not has_flat:
            missing.append(f"flat ({filter_name})")

        cal_mgr = runtime.calibration_manager if runtime else None
        capture_running = cal_mgr.is_running() if cal_mgr else False
        capture_requested = cal_mgr.is_requested() if cal_mgr else False
        capture_progress = cal_mgr.get_progress() if cal_mgr else {}

        return {
            "has_bias": has_bias,
            "has_dark": has_dark,
            "has_flat": has_flat,
            "missing": missing,
            "missing_summary": ", ".join(missing) if missing else "",
            "capture_running": capture_running,
            "capture_requested": capture_requested,
            "capture_progress": capture_progress,
            "calibration_applicable": True,
            "has_mechanical_shutter": profile.has_mechanical_shutter,
            "has_cooling": profile.has_cooling,
            "camera_id": cam_id,
            "current_gain": gain,
            "current_binning": binning,
            "current_temperature": temperature,
        }
