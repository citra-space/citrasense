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
            status.hardware_adapter = self.daemon.settings.hardware_adapter

            if hasattr(self.daemon, "hardware_adapter") and self.daemon.hardware_adapter:
                adapter = self.daemon.hardware_adapter
                mount = adapter.mount

                # Check telescope connection status
                try:
                    status.telescope_connected = adapter.is_telescope_connected()
                    if status.telescope_connected:
                        snap = mount.cached_state if mount is not None else None
                        if snap is not None:
                            status.telescope_ra = snap.ra_deg
                            status.telescope_dec = snap.dec_deg
                            status.telescope_az = snap.az_deg
                            status.telescope_alt = snap.alt_deg
                            status.mount_tracking = snap.is_tracking
                            status.mount_slewing = snap.is_slewing
                        else:
                            ra, dec = adapter.get_telescope_direction()
                            status.telescope_ra = ra
                            status.telescope_dec = dec
                            status.mount_tracking = False
                            status.mount_slewing = False
                    else:
                        status.mount_tracking = False
                        status.mount_slewing = False
                except Exception:
                    status.telescope_connected = False
                    status.mount_tracking = False
                    status.mount_slewing = False
                _mark("hw.telescope")

                # Check camera connection status
                try:
                    status.camera_connected = adapter.is_camera_connected()
                    camera = adapter.camera
                    if status.camera_connected and camera is not None:
                        status.camera_temperature = camera.get_temperature()
                    else:
                        status.camera_temperature = None
                except Exception:
                    status.camera_connected = False
                    status.camera_temperature = None
                _mark("hw.camera")

                # Check adapter capabilities
                try:
                    status.supports_direct_camera_control = adapter.supports_direct_camera_control()
                except Exception:
                    status.supports_direct_camera_control = False

                status.supports_direct_mount_control = mount is not None and status.telescope_connected

                status.supports_autofocus = adapter.supports_autofocus()
                status.supports_hardware_safety_monitor = adapter.supports_hardware_safety_monitor
                _mark("hw.capabilities")

                try:
                    pos = adapter.get_filter_position()
                    status.current_filter_position = pos
                    if pos is not None and pos in adapter.filter_map:
                        status.current_filter_name = adapter.filter_map[pos].get("name")
                    else:
                        status.current_filter_name = None
                except Exception:
                    status.current_filter_position = None
                    status.current_filter_name = None
                _mark("hw.filter")

                # Check focuser status
                focuser = adapter.focuser
                if focuser is not None and focuser.is_connected():
                    status.focuser_connected = True
                    try:
                        status.focuser_position = focuser.get_position()
                    except Exception:
                        status.focuser_position = None
                    try:
                        status.focuser_max_position = focuser.get_max_position()
                    except Exception:
                        status.focuser_max_position = None
                    try:
                        status.focuser_temperature = focuser.get_temperature()
                    except Exception:
                        status.focuser_temperature = None
                    try:
                        status.focuser_moving = focuser.is_moving()
                    except Exception:
                        status.focuser_moving = False
                else:
                    status.focuser_connected = False
                    status.focuser_position = None
                    status.focuser_max_position = None
                    status.focuser_temperature = None
                    status.focuser_moving = False
                _mark("hw.focuser")

                status.supports_alignment = status.camera_connected and mount is not None

                if mount is not None and mount.cached_state is not None:
                    status.supports_manual_sync = mount.cached_mount_info.get("supports_sync", False)
                    status.mount_at_home = mount.cached_state.is_at_home
                    h_limit, o_limit = mount.cached_limits
                    status.mount_horizon_limit = h_limit
                    status.mount_overhead_limit = o_limit
                _mark("hw.mount_state")

            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                task_manager = self.daemon.task_manager
                status.current_task = task_manager.current_task_id
                status.mount_homing = (
                    task_manager.homing_manager.is_running() or task_manager.homing_manager.is_requested()
                )
                status.autofocus_requested = task_manager.autofocus_manager.is_requested()
                status.autofocus_running = task_manager.autofocus_manager.is_running()
                status.autofocus_progress = task_manager.autofocus_manager.progress
                status.autofocus_points = [
                    {"pos": p, "hfr": h, "filter": f} for p, h, f in task_manager.autofocus_manager.points
                ]
                status.autofocus_filter_results = task_manager.autofocus_manager.filter_results
                status.autofocus_last_result = task_manager.autofocus_manager.last_result
                hfr_hist = task_manager.autofocus_manager.hfr_history
                status.hfr_history = [{"hfr": h, "ts": t, "filter": f} for h, t, f in hfr_hist]
                status.last_hfr_median = hfr_hist[-1][0] if hfr_hist else None
                if self.daemon.settings:
                    status.hfr_baseline = self.daemon.settings.adapter_settings.get("hfr_baseline")
                    status.hfr_increase_percent = self.daemon.settings.autofocus_hfr_increase_percent
                    status.hfr_refocus_enabled = self.daemon.settings.autofocus_on_hfr_increase_enabled
                    status.hfr_sample_window = self.daemon.settings.autofocus_hfr_sample_window
                status.alignment_requested = task_manager.alignment_manager.is_requested()
                status.alignment_running = task_manager.alignment_manager.is_running()
                status.alignment_progress = task_manager.alignment_manager.progress
                status.pointing_calibration_running = task_manager.alignment_manager.is_calibrating()
                status.pointing_calibration_progress = task_manager.alignment_manager.calibration_progress
                status.tasks_pending = task_manager.pending_task_count

                busy_reasons: list[str] = []
                if task_manager.is_processing_active():
                    busy_reasons.append("processing")
                if not task_manager.imaging_queue.is_idle():
                    busy_reasons.append("imaging")
                if status.alignment_running:
                    busy_reasons.append("alignment")
                if status.autofocus_running:
                    busy_reasons.append("autofocus")
                if status.pointing_calibration_running:
                    busy_reasons.append("calibration")
                if status.mount_homing:
                    busy_reasons.append("homing")
                status.system_busy = bool(busy_reasons)
                status.system_busy_reason = ", ".join(busy_reasons)

            _mark("task_manager")

            # Get autofocus timing information
            if self.daemon.settings:
                settings = self.daemon.settings
                status.last_autofocus_timestamp = settings.last_autofocus_timestamp
                status.last_alignment_timestamp = settings.last_alignment_timestamp
                status.autofocus_target_name = _resolve_autofocus_target_name(settings)

                # Calculate next autofocus time (delegates mode-aware logic to AutofocusManager)
                if (
                    hasattr(self.daemon, "task_manager")
                    and self.daemon.task_manager
                    and hasattr(self.daemon.task_manager, "autofocus_manager")
                ):
                    status.next_autofocus_minutes = (
                        self.daemon.task_manager.autofocus_manager.get_next_autofocus_minutes()
                    )
                elif settings.scheduled_autofocus_enabled:
                    last_ts = settings.last_autofocus_timestamp
                    interval_minutes = settings.autofocus_interval_minutes
                    if last_ts is not None:
                        elapsed_minutes = (int(time.time()) - last_ts) / 60
                        remaining = max(0, interval_minutes - elapsed_minutes)
                        status.next_autofocus_minutes = int(remaining)
                    else:
                        status.next_autofocus_minutes = 0
                else:
                    status.next_autofocus_minutes = None

            _mark("autofocus")

            # Get time sync status from time monitor
            if hasattr(self.daemon, "time_monitor") and self.daemon.time_monitor:
                health = self.daemon.time_monitor.get_current_health()
                status.time_health = health.to_dict() if health else None
            else:
                # Time monitoring not initialized yet
                status.time_health = None

            # Get GPS status from both sources separately
            # Use allow_blocking=False to prevent blocking the async event loop
            if hasattr(self.daemon, "location_service") and self.daemon.location_service:
                gpsd_fix = self.daemon.location_service.get_gpsd_fix(allow_blocking=False)
                status.gpsd_fix = _gps_fix_to_dict(gpsd_fix) if gpsd_fix else None

                adapter_fix = self.daemon.location_service.get_equipment_gps()
                status.adapter_gps = _gps_fix_to_dict(adapter_fix) if adapter_fix else None
            else:
                status.gpsd_fix = None
                status.adapter_gps = None

            # Get ground station information from daemon (available after API validation)
            if hasattr(self.daemon, "ground_station") and self.daemon.ground_station:
                gs_record = self.daemon.ground_station
                gs_id = gs_record.get("id")
                gs_name = gs_record.get("name", "Unknown")

                # Build the URL based on the API host (dev vs prod)
                api_host = self.daemon.settings.host
                base_url = DEV_APP_URL if "dev." in api_host else PROD_APP_URL

                status.ground_station_id = gs_id
                status.ground_station_name = gs_name
                status.ground_station_url = f"{base_url}/ground-stations/{gs_id}" if gs_id else None
                status.ground_station_latitude = gs_record.get("latitude")
                status.ground_station_longitude = gs_record.get("longitude")
                status.ground_station_altitude = gs_record.get("altitude")

            # Resolve active operating location from data already fetched above
            # (avoids calling get_current_location() which can block on subprocess)
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

            # Update task processing state
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                status.processing_active = self.daemon.task_manager.is_processing_active()
                status.automated_scheduling = self.daemon.task_manager.automated_scheduling

                # Observing session / self-tasking status
                osm = self.daemon.task_manager.observing_session_manager
                stm = self.daemon.task_manager.self_tasking_manager
                if osm:
                    sd = osm.status_dict()
                    status.observing_session_state = sd.get("observing_session_state", "daytime")
                    status.session_activity = sd.get("session_activity")
                    status.observing_session_threshold = sd.get("observing_session_threshold", -12.0)
                    status.sun_altitude = sd.get("sun_altitude")
                    status.dark_window_start = sd.get("dark_window_start")
                    status.dark_window_end = sd.get("dark_window_end")
                if stm:
                    st = stm.status_dict()
                    status.last_batch_request = st.get("last_batch_request")
                    status.last_batch_created = st.get("last_batch_created")
                    status.next_request_seconds = st.get("next_request_seconds")

            status.observing_session_enabled = self.daemon.settings.observing_session_enabled
            status.self_tasking_enabled = self.daemon.settings.self_tasking_enabled

            _mark("session")

            # Merge hardware + processor missing-dep issues into one list.
            # Hardware is declarative per-device; processors are a small fixed
            # cast checked imperatively at daemon startup (see startup_checks.py).
            status.missing_dependencies = []
            if getattr(self.daemon, "hardware_adapter", None):
                try:
                    status.missing_dependencies.extend(self.daemon.hardware_adapter.get_missing_dependencies())
                except Exception as e:
                    CITRASENSE_LOGGER.debug(f"Could not read hardware missing dependencies: {e}")
            status.missing_dependencies.extend(getattr(self.daemon, "_processor_dep_issues", []) or [])

            # Get list of active processors
            if hasattr(self.daemon, "processor_registry") and self.daemon.processor_registry:
                status.active_processors = [p.name for p in self.daemon.processor_registry.processors]
            else:
                status.active_processors = []

            # Get tasks by pipeline stage
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                status.tasks_by_stage = self.daemon.task_manager.get_tasks_by_stage()
            else:
                status.tasks_by_stage = None

            # Collect lifetime pipeline stats from queues, processor registry, and task manager
            if hasattr(self.daemon, "task_manager") and self.daemon.task_manager:
                tm = self.daemon.task_manager
                status.pipeline_stats = {
                    "imaging": tm.imaging_queue.get_stats(),
                    "processing": tm.processing_queue.get_stats(),
                    "uploading": tm.upload_queue.get_stats(),
                    "tasks": tm.get_task_stats(),
                }
            else:
                status.pipeline_stats = None
            if hasattr(self.daemon, "processor_registry") and self.daemon.processor_registry:
                if status.pipeline_stats is None:
                    status.pipeline_stats = {}
                status.pipeline_stats["processors"] = self.daemon.processor_registry.get_processor_stats()

            _mark("pipeline")

            # Safety monitor status
            if hasattr(self.daemon, "safety_monitor") and self.daemon.safety_monitor:
                try:
                    status.safety_status = self.daemon.safety_monitor.get_status()
                except Exception:
                    status.safety_status = None
            else:
                status.safety_status = None

            # Elset cache health for satellite matching status
            if hasattr(self.daemon, "elset_cache") and self.daemon.elset_cache:
                status.elset_health = self.daemon.elset_cache.get_health()
            else:
                status.elset_health = None

            _mark("safety_elset")

            # Latest annotated task image for the Optics pane
            ann_path = getattr(self.daemon, "latest_annotated_image_path", None)
            if ann_path and Path(ann_path).exists():
                mtime_ns = Path(ann_path).stat().st_mtime_ns
                status.latest_task_image_url = f"/api/task-preview/latest?t={mtime_ns}"
            else:
                status.latest_task_image_url = None

            # Calibration status
            status.calibration_status = self._build_calibration_status()

            # Pointing model status
            adapter = self.daemon.hardware_adapter
            status.pointing_model = adapter.get_pointing_model_status() if adapter else None
            status.fov_short_deg = adapter.observed_fov_short_deg if adapter else None

            # User-defined "won't observe below this" elevation from the
            # telescope record on the Citra backend.  Different from the mount's
            # hardware altitude limit; both can be set independently.  Used by
            # the Sky compass on the monitoring page to color targets that fall
            # under the operator's preferred floor.
            #
            # Always overwrite so a disappearing telescope_record clears the
            # previous value rather than leaving the UI to color against a
            # stale floor.  Parse defensively: the outer try/except would
            # otherwise abort the whole collect() cycle on a single bad field.
            status.telescope_min_elevation = None
            if self.daemon.telescope_record:
                min_el = self.daemon.telescope_record.get("minElevation")
                if min_el is not None:
                    try:
                        status.telescope_min_elevation = float(min_el)
                    except (TypeError, ValueError):
                        pass

            # Config health: compare server telescope record vs hardware + plate solve
            if self.daemon.telescope_record and adapter:
                from citrasense.hardware.config_health import assess_config_health

                camera_info = adapter.get_camera_info()
                binning = adapter.get_current_binning()
                health = assess_config_health(
                    telescope_record=self.daemon.telescope_record,
                    camera_info=camera_info,
                    binning=binning,
                    observed_pixel_scale=adapter.observed_pixel_scale_arcsec,
                    observed_fov_w=adapter.observed_fov_w_deg,
                    observed_fov_h=adapter.observed_fov_h_deg,
                    observed_slew_rate=adapter.observed_slew_rate_deg_per_s,
                    slew_rate_samples=adapter.slew_rate_tracker.count,
                )
                status.config_health = health.to_dict()
            else:
                status.config_health = None

            _mark("optics_calibration")

            status.last_update = datetime.now().isoformat()
            status.status_collection_ms = round((time.perf_counter() - _t0) * 1000, 2)
            status.status_collection_breakdown = _breakdown

        except Exception:
            CITRASENSE_LOGGER.exception("Error updating status")

    def _build_calibration_status(self) -> dict[str, Any] | None:
        """Build calibration status dict for SystemStatus."""
        if not self.daemon:
            return None
        lib = getattr(self.daemon, "calibration_library", None)
        hw = self.daemon.hardware_adapter
        if not lib or not hw or not hw.supports_direct_camera_control():
            return None

        camera = hw.camera
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
        filter_pos = hw.get_filter_position() if hasattr(hw, "get_filter_position") else None
        if filter_pos is not None and hasattr(hw, "filter_map"):
            fdata = hw.filter_map.get(filter_pos, {})
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

        # CalibrationManager state
        tm = self.daemon.task_manager
        cal_mgr = tm.calibration_manager if tm else None
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
