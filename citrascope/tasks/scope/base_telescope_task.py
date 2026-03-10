import math
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from dateutil import parser as dtparser
from skyfield.api import EarthSatellite, load, wgs84

from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrascope.tasks.fits_enrichment import enrich_fits_metadata

_DEFAULT_SLEW_ACCELERATION_DEG_PER_S2 = 2.0
_DEFAULT_SETTLE_TIME_S = 0.5
_DEFAULT_CONVERGENCE_THRESHOLD_DEG = 0.3
_FOV_CONVERGENCE_FRACTION = 0.35

_MIN_MOTION_TIME_S = 0.2
_MIN_SLEW_DISTANCE_DEG = 0.05
_MIN_OBSERVED_RATE_DEG_PER_S = 0.1
_MAX_OBSERVED_RATE_DEG_PER_S = 50.0
_RATE_DIVERGENCE_WARNING_THRESHOLD = 0.3
_SLEW_RATE_EMA_ALPHA = 0.4


def estimate_slew_time(
    distance_deg: float,
    max_rate: float,
    acceleration: float = _DEFAULT_SLEW_ACCELERATION_DEG_PER_S2,
    settle_time: float = _DEFAULT_SETTLE_TIME_S,
) -> float:
    """Estimate slew time using a trapezoidal velocity profile.

    For short slews the mount never reaches max speed (triangle profile).
    For long slews it accelerates, cruises, then decelerates (trapezoid).
    Settle time (vibration damping after stop) is always added.
    """
    if distance_deg <= 0 or max_rate <= 0 or acceleration <= 0:
        return settle_time

    # Distance threshold where the mount just barely reaches max_rate
    d_transition = max_rate**2 / acceleration

    if distance_deg < d_transition:
        motion_time = 2.0 * math.sqrt(distance_deg / acceleration)
    else:
        motion_time = distance_deg / max_rate + max_rate / acceleration

    return motion_time + settle_time


class AbstractBaseTelescopeTask(ABC):
    def __init__(
        self,
        api_client,
        hardware_adapter: AbstractAstroHardwareAdapter,
        logger,
        task,
        daemon,
    ):
        self.api_client = api_client
        self.hardware_adapter: AbstractAstroHardwareAdapter = hardware_adapter
        self.logger = logger
        self.task = task
        self.daemon = daemon
        self._cancelled = threading.Event()

        # Multi-image completion tracking (reset per upload_image_and_mark_complete call)
        self._completion_lock = threading.Lock()
        self._pending_images: int = 0
        self._completed_images: int = 0
        self._any_upload_succeeded: bool = False
        self._finalized: bool = False

    def cancel(self) -> None:
        """Signal this task to abort at the next safe point."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def fetch_satellite(self) -> dict | None:
        satellite_data = self.api_client.get_satellite(self.task.satelliteId)
        if not satellite_data:
            self.logger.error(f"Could not fetch satellite data for {self.task.satelliteId}")
            return None
        elsets = satellite_data.get("elsets", [])
        if not elsets:
            self.logger.error(f"No elsets found for satellite {self.task.satelliteId}")
            return None
        satellite_data["most_recent_elset"] = self._get_most_recent_elset(satellite_data)
        return satellite_data

    def _get_most_recent_elset(self, satellite_data) -> dict | None:
        if "most_recent_elset" in satellite_data:
            return satellite_data["most_recent_elset"]

        elsets = satellite_data.get("elsets", [])
        if not elsets:
            self.logger.error(f"No elsets found for satellite {self.task.satelliteId}")
            return None
        most_recent_elset = max(
            elsets,
            key=lambda e: (
                dtparser.isoparse(e["creationEpoch"])
                if e.get("creationEpoch")
                else dtparser.isoparse("1970-01-01T00:00:00Z")
            ),
        )
        return most_recent_elset

    def upload_image_and_mark_complete(
        self,
        filepath: str | list[str],
        satellite_data: dict | None = None,
        pointing_report: dict | None = None,
    ) -> bool:
        """
        Image captured. Queue for background processing and return immediately.
        Telescope is now free to start next task.
        """
        # Handle list input
        if isinstance(filepath, str):
            filepaths = [filepath]
        else:
            filepaths = filepath

        # Reset multi-image completion tracking
        with self._completion_lock:
            self._pending_images = len(filepaths)
            self._completed_images = 0
            self._any_upload_succeeded = False
            self._finalized = False

        # Count this as a started task (one increment regardless of image count)
        self.daemon.task_manager.record_task_started()

        # Update status message and stage ONCE before processing loop
        if self.daemon.settings.processors_enabled:
            self.task.set_status_msg("Queued for processing...")
            self.daemon.task_manager.update_task_stage(self.task.id, "processing")

        for image_path in filepaths:
            # 1. Enrich FITS metadata (quick, keep synchronous)
            try:
                enrich_fits_metadata(
                    image_path,
                    task=self.task,
                    daemon=self.daemon,
                )
            except Exception as e:
                self.logger.warning(f"Failed to enrich FITS metadata for {image_path}: {e}")

            # 2. Queue for background processing
            if self.daemon.settings.processors_enabled:
                self.daemon.task_manager.processing_queue.submit(
                    task_id=self.task.id,
                    image_path=Path(image_path),
                    context={
                        "task": self.task,
                        "telescope_record": self.daemon.telescope_record,
                        "ground_station_record": self.daemon.ground_station,
                        "settings": self.daemon.settings,
                        "daemon": self.daemon,
                        "satellite_data": satellite_data,
                        "pointing_report": pointing_report,
                    },
                    on_complete=lambda tid, result, fp=image_path: self._on_processing_complete(fp, tid, result),
                )
            else:
                # No processing, go straight to upload
                self._queue_for_upload(image_path, processing_result=None)

        # Return immediately - telescope is free
        return True

    def _on_processing_complete(self, filepath: str, task_id: str, result):
        """Called by background worker when processing finishes."""
        self.logger.info(f"Processing complete for task {task_id}")

        # Check if processors rejected upload — count as completed image
        if result and not result.should_upload:
            self.logger.info(f"Skipping upload per processor: {result.skip_reason}")
            self._on_image_done(task_id, success=True)
            return

        # Log processing summary (human-readable, not the raw dict)
        if result and result.extracted_data:
            self.logger.info(self._format_processing_summary(task_id, result.extracted_data))

        # Feed plate solve result to hardware adapter so mount model can update (e.g. alignment offsets)
        if result and result.extracted_data and self.hardware_adapter:
            ra = result.extracted_data.get("plate_solver.ra_center")
            dec = result.extracted_data.get("plate_solver.dec_center")
            if ra is not None and dec is not None:
                expected_ra = getattr(self.task, "target_ra_deg", None)
                expected_dec = getattr(self.task, "target_dec_deg", None)
                self.hardware_adapter.update_from_plate_solve(
                    float(ra),
                    float(dec),
                    expected_ra_deg=expected_ra,
                    expected_dec_deg=expected_dec,
                )

            self._update_observed_fov_from_plate_solve(result.extracted_data)

        # Queue for upload
        self._queue_for_upload(filepath, processing_result=result)

    def _queue_for_upload(self, filepath: str, processing_result):
        """Queue image for background upload."""
        # Capture sensor location now (GPS-enhanced if available) so the upload worker
        # can attach it to optical observations without accessing the daemon later.
        try:
            sensor_location = self.daemon.location_service.get_current_location()
        except Exception:
            sensor_location = None

        # Clear previous status message and set upload message
        self.task.set_status_msg("Queued for upload...")
        self.daemon.task_manager.update_task_stage(self.task.id, "uploading")
        self.daemon.task_manager.upload_queue.submit(
            task_id=self.task.id,
            task=self.task,
            image_path=filepath,
            processing_result=processing_result,
            api_client=self.api_client,
            telescope_record=self.daemon.telescope_record,
            sensor_location=sensor_location,
            settings=self.daemon.settings,
            on_complete=self._on_image_done,
        )

    def _on_image_done(self, task_id: str, success: bool):
        """Called when a single image finishes (upload success/failure or processor skip).

        Decrements the pending-image counter. When the last image for this task
        finishes, marks the task complete on the server (if any image succeeded),
        records stats once, and removes the task from stage tracking.

        A ``_finalized`` flag ensures the completion path runs exactly once, even
        if callbacks fire more times than ``_pending_images`` (defensive guard).
        """
        with self._completion_lock:
            self._completed_images += 1
            if success:
                self._any_upload_succeeded = True
            should_finalize = (
                self._completed_images >= self._pending_images and self._pending_images > 0 and not self._finalized
            )
            if should_finalize:
                self._finalized = True
            remaining = self._pending_images - self._completed_images

        if not should_finalize:
            if remaining > 0:
                self.logger.debug(
                    f"Task {task_id}: image {self._completed_images}/{self._pending_images} done "
                    f"({'ok' if success else 'FAILED'}), {remaining} remaining"
                )
            else:
                self.logger.warning(f"Task {task_id}: spurious _on_image_done after finalization")
            return

        # All images for this task have finished — finalize exactly once
        if self._any_upload_succeeded:
            marked = self._mark_complete_with_retry(task_id)
            if marked:
                self.daemon.task_manager.record_task_succeeded()
                self.logger.info(f"Task {task_id} fully complete ({self._completed_images} image(s) processed)")
            else:
                self.daemon.task_manager.record_task_failed()
                self.logger.error(f"Task {task_id}: data uploaded but failed to mark complete after retries")
        else:
            self.daemon.task_manager.record_task_failed()
            self.logger.error(f"Task {task_id} failed — all {self._completed_images} image(s) failed to upload")

        self.daemon.task_manager.remove_task_from_all_stages(task_id)

    def _mark_complete_with_retry(self, task_id: str, attempts: int = 3, delay: float = 2.0) -> bool:
        """Try to mark a task complete on the server, retrying on transient failures."""
        for attempt in range(1, attempts + 1):
            if self.api_client.mark_task_complete(task_id):
                return True
            if attempt < attempts:
                self.logger.warning(
                    f"mark_task_complete failed for {task_id} (attempt {attempt}/{attempts}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
        return False

    @staticmethod
    def _format_processing_summary(task_id: str, data: dict) -> str:
        parts = [f"Processing results for {task_id[:8]}:"]

        solved = data.get("plate_solver.plate_solved")
        if solved is not None:
            parts.append(f"solved={'yes' if solved else 'NO'}")
        ra = data.get("plate_solver.ra_center")
        dec = data.get("plate_solver.dec_center")
        if ra is not None and dec is not None:
            parts.append(f"RA={ra:.4f}° DEC={dec:.4f}°")
        scale = data.get("plate_solver.pixel_scale")
        if scale is not None:
            parts.append(f'scale={scale:.2f}"/px')
        fw = data.get("plate_solver.field_width_deg")
        fh = data.get("plate_solver.field_height_deg")
        if fw is not None and fh is not None:
            parts.append(f"FOV={fw:.2f}\u00d7{fh:.2f}\u00b0")

        sources = data.get("source_extractor.num_sources")
        if sources is not None:
            parts.append(f"{sources} sources")

        zp = data.get("photometry.zero_point")
        cal = data.get("photometry.num_calibration_stars")
        if zp is not None:
            zp_str = f"ZP={zp:.2f}"
            if cal is not None:
                zp_str += f" ({cal} cal stars)"
            parts.append(zp_str)

        sats = data.get("satellite_matcher.num_satellites_detected")
        if sats is not None:
            parts.append(f"{sats} satellite{'s' if sats != 1 else ''}")

        return ", ".join(parts)

    def set_filter_for_task(self) -> None:
        """Resolve the assigned filter for this task and command the hardware to switch.

        No-op if the adapter has no filter wheel or the task has no filter assignment.
        Raises RuntimeError if the assigned filter is missing/disabled or the wheel fails to move.
        """
        if not self.hardware_adapter.filter_map:
            return

        filters_to_use = self.hardware_adapter.select_filters_for_task(self.task, allow_no_filter=True)
        if filters_to_use is None:
            return

        filter_id = next(iter(filters_to_use))
        filter_name = filters_to_use[filter_id]["name"]

        if not self.hardware_adapter.set_filter(filter_id):
            raise RuntimeError(f"Failed to set filter '{filter_name}' (position {filter_id})")

        self.logger.info(f"Filter set to '{filter_name}' (position {filter_id}) for task {self.task.id}")

    def verify_pointing(self, target_ra_deg: float, target_dec_deg: float) -> bool:
        """Plate-solve to confirm and correct pointing after slewing.

        Calls the adapter's ``perform_alignment`` which captures, solves, and
        syncs the mount.  Returns True on success.
        """
        self.task.set_status_msg("Verifying pointing (plate solve)...")
        success = self.hardware_adapter.perform_alignment(target_ra_deg, target_dec_deg)
        if not success:
            self.logger.warning("Post-slew plate solve failed — proceeding with unverified pointing")
        return success

    @abstractmethod
    def execute(self):
        pass

    def _get_skyfield_ground_station_and_satellite(self, satellite_data):
        """
        Returns (ground_station, satellite, ts) Skyfield objects for the given satellite and elset.
        Uses GPS-enhanced location if available, otherwise falls back to ground station record.
        """
        ts = load.timescale()
        most_recent_elset = self._get_most_recent_elset(satellite_data)
        if most_recent_elset is None:
            raise ValueError("No valid elset available for satellite.")

        # Get current location from location service (GPS preferred, ground station fallback)
        # Defensive check against race condition during component reinitialization
        if not self.daemon.location_service:
            raise ValueError("Location service not available (system may be reinitializing)")

        location = self.daemon.location_service.get_current_location()
        if not location:
            raise ValueError("No location available from location service")

        ground_station = wgs84.latlon(
            location["latitude"],
            location["longitude"],
            elevation_m=location["altitude"],
        )
        satellite = EarthSatellite(most_recent_elset["tle"][0], most_recent_elset["tle"][1], satellite_data["name"], ts)
        return ground_station, satellite, ts

    def get_target_radec_and_rates(self, satellite_data, seconds_from_now: float = 0.0):
        ground_station, satellite, ts = self._get_skyfield_ground_station_and_satellite(satellite_data)
        difference = satellite - ground_station
        days_to_add = seconds_from_now / (24 * 60 * 60)  # Skyfield uses days
        topocentric = difference.at(ts.now() + days_to_add)
        target_ra, target_dec, _ = topocentric.radec()

        # determine ra/dec travel rates
        rates = topocentric.frame_latlon_and_rates(
            ground_station
        )  # TODO can this be collapsed with .radec() call above?
        target_dec_rate = rates[4]
        target_ra_rate = rates[3]

        return target_ra, target_dec, target_ra_rate, target_dec_rate

    def predict_slew_time_seconds(
        self, satellite_data, seconds_from_now: float = 0.0, max_rate: float | None = None
    ) -> float:
        current_scope_ra, current_scope_dec = self.hardware_adapter.get_telescope_direction()
        current_target_ra, current_target_dec, _, _ = self.get_target_radec_and_rates(satellite_data, seconds_from_now)

        distance_deg = self.hardware_adapter.angular_distance(
            current_scope_ra,
            current_scope_dec,
            current_target_ra.degrees,  # type: ignore
            current_target_dec.degrees,  # type: ignore
        )

        rate = max_rate if max_rate is not None else self.hardware_adapter.scope_slew_rate_degrees_per_second
        return estimate_slew_time(distance_deg, rate)

    def point_to_lead_position(self, satellite_data, extra_lead_seconds: float = 0.0) -> dict:
        """Iteratively slew the telescope toward the satellite's predicted position.

        Args:
            satellite_data: Satellite data dict with TLE and metadata.
            extra_lead_seconds: Additional seconds to lead beyond the slew time.

        Returns a pointing report dict with convergence telemetry for diagnostics.
        """
        self.logger.debug(f"Using TLE {satellite_data['most_recent_elset']['tle']}")

        max_angular_distance_deg = self._compute_convergence_threshold()
        self.logger.info(f"Convergence threshold: {max_angular_distance_deg:.3f}°")

        effective_rate = self.hardware_adapter.observed_slew_rate_deg_per_s
        rate_warning_logged = False
        attempts = 0
        max_attempts = 10

        iteration_log: list[dict] = []
        report: dict = {
            "convergence_threshold_deg": round(max_angular_distance_deg, 4),
            "max_attempts": max_attempts,
            "configured_slew_rate_deg_per_s": self.hardware_adapter.scope_slew_rate_degrees_per_second,
        }

        while attempts < max_attempts:
            if self.is_cancelled:
                raise RuntimeError("Task cancelled")

            attempts += 1

            pre_slew_ra, pre_slew_dec = self.hardware_adapter.get_telescope_direction()

            lead_ra, lead_dec, est_slew_time = self.estimate_lead_position(
                satellite_data, max_rate=effective_rate, extra_lead_seconds=extra_lead_seconds
            )
            self.logger.info(
                f"Pointing ahead to RA: {lead_ra.degrees:.4f}°, DEC: {lead_dec.degrees:.4f}°, "
                f"estimated slew time: {est_slew_time:.1f}s"
            )

            slew_start_time = time.time()
            self.hardware_adapter.point_telescope(lead_ra.degrees, lead_dec.degrees)  # type: ignore
            while self.hardware_adapter.telescope_is_moving():
                if self.is_cancelled:
                    raise RuntimeError("Task cancelled")
                time.sleep(0.1)

            slew_duration = time.time() - slew_start_time

            post_slew_ra, post_slew_dec = self.hardware_adapter.get_telescope_direction()
            slewed_distance = self.hardware_adapter.angular_distance(
                pre_slew_ra, pre_slew_dec, post_slew_ra, post_slew_dec
            )

            # Adaptive rate: learn from observed slew performance.
            # slew_duration is GoTo time only (mount reports done); settle is post-motion
            # vibration damping and is NOT included, so no subtraction needed.
            iter_observed_rate: float | None = None
            if slew_duration > _MIN_MOTION_TIME_S and slewed_distance > _MIN_SLEW_DISTANCE_DEG:
                observed_rate = slewed_distance / slew_duration
                observed_rate = max(_MIN_OBSERVED_RATE_DEG_PER_S, min(_MAX_OBSERVED_RATE_DEG_PER_S, observed_rate))

                if effective_rate is not None:
                    effective_rate = _SLEW_RATE_EMA_ALPHA * observed_rate + (1 - _SLEW_RATE_EMA_ALPHA) * effective_rate
                else:
                    effective_rate = observed_rate

                self.hardware_adapter.observed_slew_rate_deg_per_s = effective_rate
                iter_observed_rate = round(effective_rate, 2)

                if not rate_warning_logged:
                    api_rate = self.hardware_adapter.scope_slew_rate_degrees_per_second
                    if api_rate > 0 and abs(effective_rate - api_rate) / api_rate > _RATE_DIVERGENCE_WARNING_THRESHOLD:
                        self.logger.warning(
                            f"Observed slew rate ({effective_rate:.1f} deg/s) differs from "
                            f"configured maxSlewRate ({api_rate:.1f} deg/s) by "
                            f"{abs(effective_rate - api_rate) / api_rate * 100:.0f}% — "
                            f"using observed rate for predictions. Consider updating "
                            f"maxSlewRate in your Citra telescope settings."
                        )
                    rate_warning_logged = True

            self.logger.info(
                f"Telescope slew done, took {slew_duration:.1f} sec, "
                f"off by {abs(slew_duration - est_slew_time):.1f} sec."
            )

            # Convergence: did we arrive at our intended target?
            target_lead_ra_deg = float(lead_ra.degrees)  # type: ignore[union-attr]
            target_lead_dec_deg = float(lead_dec.degrees)  # type: ignore[union-attr]
            target_distance_deg = self.hardware_adapter.angular_distance(
                post_slew_ra, post_slew_dec, target_lead_ra_deg, target_lead_dec_deg
            )

            # Satellite's current position (diagnostic only — may differ from
            # target when extra_lead_seconds > 0 because we're ahead of it)
            current_satellite_position = self.get_target_radec_and_rates(satellite_data)
            sat_ra_deg = float(current_satellite_position[0].degrees)  # type: ignore[union-attr]
            sat_dec_deg = float(current_satellite_position[1].degrees)  # type: ignore[union-attr]
            current_angular_distance_deg = self.hardware_adapter.angular_distance(
                post_slew_ra, post_slew_dec, sat_ra_deg, sat_dec_deg
            )

            self.logger.info(
                f"Distance to target: {target_distance_deg:.3f}°, "
                f"distance to satellite: {current_angular_distance_deg:.3f}°"
            )

            converged = target_distance_deg <= max_angular_distance_deg
            iteration_log.append(
                {
                    "attempt": attempts,
                    "pre_slew_ra_deg": round(pre_slew_ra, 4),
                    "pre_slew_dec_deg": round(pre_slew_dec, 4),
                    "target_lead_ra_deg": round(target_lead_ra_deg, 4),
                    "target_lead_dec_deg": round(target_lead_dec_deg, 4),
                    "estimated_slew_time_s": round(est_slew_time, 2),
                    "actual_slew_time_s": round(slew_duration, 2),
                    "post_slew_ra_deg": round(post_slew_ra, 4),
                    "post_slew_dec_deg": round(post_slew_dec, 4),
                    "slewed_distance_deg": round(slewed_distance, 4),
                    "target_distance_deg": round(target_distance_deg, 4),
                    "satellite_ra_deg": round(sat_ra_deg, 4),
                    "satellite_dec_deg": round(sat_dec_deg, 4),
                    "angular_distance_to_satellite_deg": round(current_angular_distance_deg, 4),
                    "observed_slew_rate_deg_per_s": iter_observed_rate,
                    "converged": converged,
                }
            )

            if converged:
                self.logger.info("Telescope is within acceptable range of target.")
                break

        report["attempts"] = attempts
        report["iterations"] = iteration_log
        report["converged"] = bool(iteration_log and iteration_log[-1].get("converged"))
        if iteration_log:
            report["final_angular_distance_deg"] = iteration_log[-1]["angular_distance_to_satellite_deg"]
            report["final_telescope_ra_deg"] = iteration_log[-1]["post_slew_ra_deg"]
            report["final_telescope_dec_deg"] = iteration_log[-1]["post_slew_dec_deg"]
        return report

    def _compute_convergence_threshold(self) -> float:
        """Derive pointing convergence threshold from FOV.

        Prefers plate-solved FOV (from previous solves in this session), falls back
        to nominal FOV from telescope record, then to a hardcoded default.
        """
        observed_fov = self.hardware_adapter.observed_fov_short_deg
        if observed_fov and observed_fov > 0:
            return max((observed_fov / 2) * _FOV_CONVERGENCE_FRACTION, 0.1)

        tr = self.hardware_adapter.telescope_record
        if tr:
            try:
                pixel_scale_arcsec = float(tr["pixelSize"]) / float(tr["focalLength"]) * 206.265
                short_axis_px = min(int(tr["horizontalPixelCount"]), int(tr["verticalPixelCount"]))
                half_fov_deg = (short_axis_px * pixel_scale_arcsec / 3600) / 2
                return max(half_fov_deg * _FOV_CONVERGENCE_FRACTION, 0.1)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass

        return _DEFAULT_CONVERGENCE_THRESHOLD_DEG

    def _update_observed_fov_from_plate_solve(self, extracted_data: dict) -> None:
        """Update the adapter's observed FOV from plate-solve results.

        Called once after the first successful plate solve in a session. If the
        solved pixel scale diverges from the telescope record by >10%, logs a
        warning so the operator knows their config may be stale.
        """
        adapter = self.hardware_adapter
        if adapter is None or adapter.observed_fov_short_deg:
            return

        field_w = extracted_data.get("plate_solver.field_width_deg")
        field_h = extracted_data.get("plate_solver.field_height_deg")
        if not field_w or not field_h:
            return

        adapter.observed_fov_short_deg = min(field_w, field_h)

        tr = adapter.telescope_record
        if tr:
            try:
                nominal_scale = float(tr["pixelSize"]) / float(tr["focalLength"]) * 206.265
                solved_scale = extracted_data.get("plate_solver.pixel_scale")
                if solved_scale and nominal_scale > 0:
                    pct_diff = abs(solved_scale - nominal_scale) / nominal_scale
                    if pct_diff > 0.1:
                        self.logger.warning(
                            f'Plate-solved pixel scale ({solved_scale:.2f}"/px) differs from '
                            f'telescope record ({nominal_scale:.2f}"/px) by {pct_diff * 100:.0f}% — '
                            f"using plate-solved FOV for pointing threshold"
                        )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass

    def _get_fov_radius_deg(self) -> float:
        """Return half the short-axis FOV in degrees.

        Prefers plate-solved FOV, falls back to telescope record, then 0.5 deg.
        Unlike ``_compute_convergence_threshold`` this returns the raw half-FOV
        without the convergence fraction multiplier.
        """
        observed_fov = self.hardware_adapter.observed_fov_short_deg
        if observed_fov and observed_fov > 0:
            return observed_fov / 2.0

        tr = self.hardware_adapter.telescope_record
        if tr:
            try:
                pixel_scale_arcsec = float(tr["pixelSize"]) / float(tr["focalLength"]) * 206.265
                short_axis_px = min(int(tr["horizontalPixelCount"]), int(tr["verticalPixelCount"]))
                return (short_axis_px * pixel_scale_arcsec / 3600) / 2
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass

        return 0.5

    def compute_angular_rate(self, satellite_data: dict) -> float:
        """Total angular rate of the satellite on the sky (deg/s).

        Combines RA and Dec rates with cos(dec) correction for RA projection.
        Uses ``.arcseconds.per_second`` (the same accessor the tracking task uses)
        to avoid ambiguity about what ``.degrees`` returns for Rate objects.
        """
        _, dec, ra_rate, dec_rate = self.get_target_radec_and_rates(satellite_data)
        ra_arcsec_s: float = ra_rate.arcseconds.per_second  # type: ignore[union-attr]
        dec_arcsec_s: float = dec_rate.arcseconds.per_second  # type: ignore[union-attr]
        ra_deg_s = (ra_arcsec_s / 3600) * math.cos(math.radians(dec.degrees))  # type: ignore[union-attr]
        dec_deg_s = dec_arcsec_s / 3600
        return math.sqrt(ra_deg_s**2 + dec_deg_s**2)

    def compute_satellite_timing(self, satellite_data: dict) -> dict:
        """Compute real-time timing for satellite FOV crossing.

        Uses current confirmed telescope pointing and live Skyfield ephemeris.
        The closure rate is computed numerically from positions 1s apart, which
        naturally handles arbitrary trajectories without needing rate decomposition.
        """
        scope_ra, scope_dec = self.hardware_adapter.get_telescope_direction()

        sat_now = self.get_target_radec_and_rates(satellite_data, 0.0)
        dist_now = self.hardware_adapter.angular_distance(
            scope_ra, scope_dec, sat_now[0].degrees, sat_now[1].degrees  # type: ignore[union-attr]
        )

        sat_1s = self.get_target_radec_and_rates(satellite_data, 1.0)
        dist_1s = self.hardware_adapter.angular_distance(
            scope_ra, scope_dec, sat_1s[0].degrees, sat_1s[1].degrees  # type: ignore[union-attr]
        )
        closure_rate = dist_now - dist_1s  # positive = approaching

        fov_radius = self._get_fov_radius_deg()

        if closure_rate <= 0:
            return {
                "angular_distance_deg": dist_now,
                "closure_rate_deg_per_s": closure_rate,
                "time_to_center_s": 0.0,
                "fov_radius_deg": fov_radius,
                "time_to_fov_entry_s": 0.0,
            }

        time_to_center = dist_now / closure_rate
        fov_entry_offset = fov_radius / closure_rate
        time_to_fov_entry = max(0.0, time_to_center - fov_entry_offset)

        return {
            "angular_distance_deg": dist_now,
            "closure_rate_deg_per_s": closure_rate,
            "time_to_center_s": time_to_center,
            "fov_radius_deg": fov_radius,
            "time_to_fov_entry_s": time_to_fov_entry,
        }

    def estimate_lead_position(
        self,
        satellite_data: dict,
        max_iterations: int = 5,
        tolerance: float = 0.1,
        max_rate: float | None = None,
        extra_lead_seconds: float = 0.0,
    ):
        """Iteratively estimate the future RA/Dec where the satellite will be
        when the telescope finishes slewing, plus optional extra lead time.

        Args:
            satellite_data: Satellite data dict.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence threshold in seconds.
            max_rate: Override slew rate (deg/s) from adaptive measurement.
            extra_lead_seconds: Additional seconds to lead beyond the slew time
                (e.g. to account for plate-solve time + half the imaging window).

        Returns:
            Tuple of (target_ra, target_dec, total_lead_seconds)
        """
        est_slew_time = self.predict_slew_time_seconds(satellite_data, max_rate=max_rate)
        for _ in range(max_iterations):
            future_radec = self.get_target_radec_and_rates(satellite_data, est_slew_time)
            new_slew_time = self.predict_slew_time_seconds(satellite_data, est_slew_time, max_rate=max_rate)
            if abs(new_slew_time - est_slew_time) < tolerance:
                break
            est_slew_time = new_slew_time
        total_lead = est_slew_time + extra_lead_seconds
        if extra_lead_seconds > 0:
            future_radec = self.get_target_radec_and_rates(satellite_data, total_lead)
        return future_radec[0], future_radec[1], total_lead
