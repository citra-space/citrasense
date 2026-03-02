from citrascope.hardware.abstract_astro_hardware_adapter import ObservationStrategy
from citrascope.tasks.scope.base_telescope_task import AbstractBaseTelescopeTask


class StaticTelescopeTask(AbstractBaseTelescopeTask):
    def execute(self):

        self.task.set_status_msg("Fetching satellite data...")
        satellite_data = self.fetch_satellite()
        if not satellite_data or satellite_data.get("most_recent_elset") is None:
            raise ValueError("Could not fetch valid satellite data or TLE.")

        try:
            strategy = self.hardware_adapter.get_observation_strategy()

            if strategy == ObservationStrategy.MANUAL:
                self.task.set_status_msg("Slewing to target...")
                self.point_to_lead_position(satellite_data)
                if self.is_cancelled:
                    return False
                self.task.set_status_msg("Exposing image (2s)...")
                filepaths = self.hardware_adapter.take_image(self.task.id, 2.0)

            elif strategy == ObservationStrategy.SEQUENCE_TO_CONTROLLER:
                target_ra, target_dec, _, _ = self.get_target_radec_and_rates(satellite_data)
                satellite_data["ra"] = target_ra.degrees
                satellite_data["dec"] = target_dec.degrees

                self.task.set_status_msg("Running observation sequence...")
                filepaths = self.hardware_adapter.perform_observation_sequence(self.task, satellite_data)

            else:
                raise RuntimeError(f"Unsupported observation strategy: {strategy}")

        except RuntimeError as e:
            self.logger.error(f"Observation failed for task {self.task.id}: {e}")
            return False

        return self.upload_image_and_mark_complete(filepaths)
