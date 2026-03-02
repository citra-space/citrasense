import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from dateutil import parser as dtparser
from skyfield.api import EarthSatellite, load, wgs84

from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
from citrascope.tasks.fits_enrichment import enrich_fits_metadata


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

    def upload_image_and_mark_complete(self, filepath: str | list[str]) -> bool:
        """
        Image captured. Queue for background processing and return immediately.
        Telescope is now free to start next task.
        """
        # Handle list input
        if isinstance(filepath, str):
            filepaths = [filepath]
        else:
            filepaths = filepath

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

        # Check if processors rejected upload
        if result and not result.should_upload:
            self.logger.info(f"Skipping upload per processor: {result.skip_reason}")
            self.api_client.mark_task_complete(task_id)
            self.daemon.task_manager.remove_task_from_all_stages(task_id)
            return

        # Log extracted data
        if result and result.extracted_data:
            self.logger.info(f"Extracted data: {result.extracted_data}")

        # Feed plate solve result to hardware adapter so mount model can update (e.g. alignment offsets)
        if result and result.extracted_data and self.daemon.hardware_adapter:
            ra = result.extracted_data.get("plate_solver.ra_center")
            dec = result.extracted_data.get("plate_solver.dec_center")
            if ra is not None and dec is not None:
                expected_ra = getattr(self.task, "target_ra_deg", None)
                expected_dec = getattr(self.task, "target_dec_deg", None)
                self.daemon.hardware_adapter.update_from_plate_solve(
                    float(ra),
                    float(dec),
                    expected_ra_deg=expected_ra,
                    expected_dec_deg=expected_dec,
                )

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
            on_complete=self._on_upload_complete,
        )

    def _on_upload_complete(self, task_id: str, success: bool):
        """Called by background worker when upload finishes."""
        if success:
            self.daemon.task_manager.record_task_succeeded()
            self.logger.info(f"Task {task_id} fully complete (uploaded)")
        else:
            self.daemon.task_manager.record_task_failed()
            self.logger.error(f"Task {task_id} upload failed - not retrying")
        self.daemon.task_manager.remove_task_from_all_stages(task_id)

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

    def predict_slew_time_seconds(self, satellite_data, seconds_from_now: float = 0.0) -> float:
        current_scope_ra, current_scope_dec = self.hardware_adapter.get_telescope_direction()
        current_target_ra, current_target_dec, _, _ = self.get_target_radec_and_rates(satellite_data, seconds_from_now)

        ra_diff_deg = abs(current_target_ra.degrees - current_scope_ra)  # type: ignore
        dec_diff_deg = abs(current_target_dec.degrees - current_scope_dec)  # type: ignore

        if ra_diff_deg > dec_diff_deg:
            return ra_diff_deg / self.hardware_adapter.scope_slew_rate_degrees_per_second
        else:
            return dec_diff_deg / self.hardware_adapter.scope_slew_rate_degrees_per_second

    def point_to_lead_position(self, satellite_data):

        self.logger.debug(f"Using TLE {satellite_data['most_recent_elset']['tle']}")

        max_angular_distance_deg = 0.3
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            if self.is_cancelled:
                raise RuntimeError("Task cancelled")

            attempts += 1
            lead_ra, lead_dec, est_slew_time = self.estimate_lead_position(satellite_data)
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
            self.logger.info(
                f"Telescope slew done, took {slew_duration:.1f} sec, "
                f"off by {abs(slew_duration - est_slew_time):.1f} sec."
            )

            current_scope_ra, current_scope_dec = self.hardware_adapter.get_telescope_direction()
            current_satellite_position = self.get_target_radec_and_rates(satellite_data)
            current_angular_distance_deg = self.hardware_adapter.angular_distance(
                current_scope_ra,
                current_scope_dec,
                current_satellite_position[0].degrees,  # type: ignore
                current_satellite_position[1].degrees,  # type: ignore
            )
            self.logger.info(f"Current angular distance to satellite is {current_angular_distance_deg:.3f} degrees.")
            if current_angular_distance_deg <= max_angular_distance_deg:
                self.logger.info("Telescope is within acceptable range of target.")
                break

    def estimate_lead_position(
        self,
        satellite_data: dict,
        max_iterations: int = 5,
        tolerance: float = 0.1,
    ):
        """
        Iteratively estimate the future RA/Dec where the satellite will be when the telescope finishes slewing.

        Args:
            satellite_data: Satellite data dict.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence threshold in seconds.

        Returns:
            Tuple of (target_ra, target_dec, estimated_slew_time)
        """
        # Get initial estimate
        est_slew_time = self.predict_slew_time_seconds(satellite_data)
        for _ in range(max_iterations):
            future_radec = self.get_target_radec_and_rates(satellite_data, est_slew_time)
            try:
                new_slew_time = self.predict_slew_time_seconds(satellite_data, est_slew_time)
            except TypeError:
                # Fallback for legacy predict_slew_time_seconds signature
                new_slew_time = self.predict_slew_time_seconds(satellite_data)
            if abs(new_slew_time - est_slew_time) < tolerance:
                break
            est_slew_time = new_slew_time
        return future_radec[0], future_radec[1], est_slew_time
