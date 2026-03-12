import time

from citrascope.hardware.abstract_astro_hardware_adapter import ObservationStrategy
from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask


class StaticTelescopeTask(AbstractBaseTelescopeTask):
    def execute(self):

        self.task.set_status_msg("Fetching satellite data...")
        satellite_data = self.fetch_satellite()
        if not satellite_data or satellite_data.get("most_recent_elset") is None:
            raise ValueError("Could not fetch valid satellite data or TLE.")

        self.pointing_report: dict | None = None
        try:
            strategy = self.hardware_adapter.get_observation_strategy()

            if strategy == ObservationStrategy.MANUAL:
                filepaths = self._execute_manual(satellite_data)

            elif strategy == ObservationStrategy.SEQUENCE_TO_CONTROLLER:
                target_ra, target_dec, _, _ = self.get_target_radec_and_rates(satellite_data)
                satellite_data["ra"] = target_ra.degrees
                satellite_data["dec"] = target_dec.degrees

                self.task.set_status_msg("Running observation sequence...")
                filepaths = self.hardware_adapter.perform_observation_sequence(self.task, satellite_data)

            else:
                raise RuntimeError(f"Unsupported observation strategy: {strategy}")

        except Exception as e:
            self.logger.error(f"Observation failed for task {self.task.id}: {e}")
            return False

        return self.upload_image_and_mark_complete(
            filepaths, satellite_data=satellite_data, pointing_report=self.pointing_report
        )

    def _execute_manual(self, satellite_data: dict) -> list[str]:
        """Adaptive slew-ahead imaging strategy.

        Slow movers (GEO): point directly at the target and shoot.
        Fast movers (LEO): slew ahead, plate-solve, wait for the satellite
        to enter the FOV, then burst-capture.
        """
        exposure = self.settings.exposure_seconds
        num_exposures = self.settings.num_exposures
        plate_solve = self.settings.plate_solve_after_slew
        imaging_window = num_exposures * exposure

        # -- Slow-mover detection --
        angular_rate = self.compute_angular_rate(satellite_data)
        fov_radius = self._get_fov_radius_deg()
        satellite_travel = angular_rate * imaging_window
        is_slow_mover = satellite_travel < fov_radius

        if is_slow_mover:
            self.logger.info(
                "Slow mover (%.4f\u00b0/s, travels %.3f\u00b0 in %.1fs window vs %.2f\u00b0 FOV radius) "
                "\u2014 pointing directly at target",
                angular_rate,
                satellite_travel,
                imaging_window,
                fov_radius,
            )
            extra_lead = 0.0
        else:
            plate_solve_budget = 15.0 if plate_solve else 0.0
            extra_lead = plate_solve_budget + imaging_window / 2.0
            self.logger.info(
                "Fast mover (%.4f\u00b0/s, travels %.3f\u00b0 in %.1fs window) " "\u2014 leading by %.1fs extra",
                angular_rate,
                satellite_travel,
                imaging_window,
                extra_lead,
            )

        # -- Filter --
        self.task.set_status_msg("Setting filter...")
        self.set_filter_for_task()

        # -- Slew (ahead for fast movers, directly for slow movers) --
        self.task.set_status_msg("Slewing to target...")
        self.pointing_report = self.point_to_lead_position(satellite_data, extra_lead_seconds=extra_lead)
        if self.is_cancelled:
            raise RuntimeError("Task cancelled")

        # Attach slew-ahead strategy data to pointing report for artifact dumping
        self.pointing_report["slew_ahead"] = {
            "is_slow_mover": is_slow_mover,
            "angular_rate_deg_per_s": round(angular_rate, 6),
            "satellite_travel_deg": round(satellite_travel, 4),
            "fov_radius_deg": round(fov_radius, 4),
            "imaging_window_s": round(imaging_window, 2),
            "extra_lead_seconds": round(extra_lead, 2),
            "num_exposures": num_exposures,
            "exposure_seconds": exposure,
            "plate_solve_after_slew": plate_solve,
        }

        # -- Verify pointing --
        if plate_solve:
            lead_ra = self.pointing_report["final_telescope_ra_deg"]
            lead_dec = self.pointing_report["final_telescope_dec_deg"]
            verified = self.verify_pointing(lead_ra, lead_dec)
            self.pointing_report["slew_ahead"]["plate_solve_succeeded"] = verified

        # -- Real-time timing gate (fast movers only) --
        if not is_slow_mover:
            timing = self.compute_satellite_timing(satellite_data)
            self.logger.info(
                "Satellite timing: %.2f\u00b0 away, closing at %.3f\u00b0/s, "
                "FOV entry in %.1fs, center crossing in %.1fs",
                timing["angular_distance_deg"],
                timing["closure_rate_deg_per_s"],
                timing["time_to_fov_entry_s"],
                timing["time_to_center_s"],
            )

            _MAX_TIMING_WAIT_S = 120.0
            wait = timing["time_to_fov_entry_s"] - exposure
            if wait > _MAX_TIMING_WAIT_S:
                self.logger.warning(
                    "Timing gate wants %.1fs wait — capping at %.0fs (check FOV/lead calculation)",
                    wait,
                    _MAX_TIMING_WAIT_S,
                )
                wait = _MAX_TIMING_WAIT_S
            if wait > 0:
                self.logger.info("Timing gate: waiting %.1fs for satellite to approach FOV", wait)
                _poll_interval = 0.1
                _status_interval = 1.0
                _waited = 0.0
                _last_status = -_status_interval
                while _waited < wait:
                    if self.is_cancelled:
                        raise RuntimeError("Task cancelled")
                    if _waited - _last_status >= _status_interval:
                        remaining = wait - _waited
                        self.task.set_status_msg(f"Waiting {remaining:.0f}s for satellite...")
                        _last_status = _waited
                    time.sleep(min(_poll_interval, wait - _waited))
                    _waited += _poll_interval
            elif timing["closure_rate_deg_per_s"] <= 0:
                self.logger.warning(
                    "Satellite is not approaching our pointing (closure rate %.3f\u00b0/s) "
                    "\u2014 imaging immediately",
                    timing["closure_rate_deg_per_s"],
                )

            timing["wait_applied_s"] = round(max(0.0, wait), 2)
            self.pointing_report["slew_ahead"]["satellite_timing"] = timing

        # -- Burst capture --
        filepaths: list[str] = []
        for i in range(num_exposures):
            if self.is_cancelled:
                break
            self.task.set_status_msg(f"Exposing {i + 1}/{num_exposures} ({exposure}s)...")
            filepaths.append(self.hardware_adapter.take_image(self.task.id, exposure))

        if self.is_cancelled and not filepaths:
            raise RuntimeError("Task cancelled before any exposures completed")

        return filepaths
