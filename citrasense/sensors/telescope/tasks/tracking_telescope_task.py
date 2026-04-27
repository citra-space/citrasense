from citrasense.sensors.telescope.tasks.base_telescope_task import AbstractBaseTelescopeTask


class TrackingTelescopeTask(AbstractBaseTelescopeTask):
    @property
    def tracking_mode(self) -> str:
        return "rate"

    def execute(self):

        self.task.set_status_msg("Fetching satellite data...")
        satellite_data = self.fetch_satellite()
        if not satellite_data or satellite_data.get("most_recent_elset") is None:
            raise ValueError("Could not fetch valid satellite data or TLE.")

        self.task.set_status_msg("Setting filter...")
        try:
            self.set_filter_for_task()
        except RuntimeError as e:
            self.logger.error(f"Filter change failed for task {self.task.id}: {e}")
            return False

        self.task.set_status_msg("Slewing to target...")
        self.timing_info.stamp_now("slew_started_at")
        try:
            self.pointing_report = self.point_to_lead_position(satellite_data)
        except RuntimeError as e:
            self.logger.error(f"Observation failed for task {self.task.id}: {e}")
            return False

        if self.is_cancelled:
            return False

        _, _, target_ra_rate, target_dec_rate = self.get_target_radec_and_rates(satellite_data)

        try:
            self.task.set_status_msg("Setting tracking rates...")
            tracking_set = self.hardware_adapter.set_custom_tracking_rate(
                target_ra_rate * 3600.0,
                target_dec_rate * 3600.0,
            )
            if not tracking_set:
                self.logger.error("Failed to set tracking rates on telescope.")
                return False

            if self.is_cancelled:
                return False

            sc = getattr(self.runtime, "sensor_config", None) or self.settings
            exposure = sc.exposure_seconds
            self.task.set_status_msg(f"Exposing image ({exposure}s)...")
            self.timing_info.stamp_now("imaging_started_at")
            filepath = self.hardware_adapter.take_image(self.task.id, exposure)
            self.timing_info.stamp_now("imaging_finished_at")
            return self.upload_image_and_mark_complete(
                [filepath], satellite_data=satellite_data, pointing_report=self.pointing_report
            )
        finally:
            try:
                self.hardware_adapter.reset_tracking_rates()
            except Exception as e:
                self.logger.error("Failed to reset tracking rates for task %s: %s", self.task.id, e)
