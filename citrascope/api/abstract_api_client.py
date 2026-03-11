from abc import ABC, abstractmethod


class AbstractCitraApiClient(ABC):
    @property
    @abstractmethod
    def cache_source_key(self) -> str:
        """Unique key identifying this API source for cache invalidation.

        Used by ElsetCache to detect when the data source changes (e.g. switching
        between dummy and real API, or between dev and prod hosts).
        """
        ...

    @abstractmethod
    def does_api_server_accept_key(self):
        pass

    @abstractmethod
    def get_telescope(self, telescope_id):
        pass

    @abstractmethod
    def get_satellite(self, satellite_id):
        pass

    @abstractmethod
    def get_telescope_tasks(self, telescope_id):
        pass

    @abstractmethod
    def get_ground_station(self, ground_station_id):
        pass

    @abstractmethod
    def put_telescope_status(self, body):
        """
        PUT to /telescopes to report online status.
        """
        pass

    @abstractmethod
    def expand_filters(self, filter_names):
        """
        POST to /filters/expand to expand filter names to spectral specs.
        """
        pass

    @abstractmethod
    def update_telescope_spectral_config(self, telescope_id, spectral_config):
        """
        PATCH to /telescopes to update telescope's spectral configuration.
        """
        pass

    @abstractmethod
    def update_ground_station_location(self, ground_station_id, latitude, longitude, altitude):
        """
        PATCH to /ground-stations to update ground station's GPS location.
        """
        pass

    @abstractmethod
    def get_elsets_latest(self, days: int = 14):
        """
        GET /elsets/latest - fetch all latest elsets (for satellite matching hot list).
        Returns list of dicts with satelliteId, satelliteName, tle, etc., or None on failure.
        """
        pass

    @abstractmethod
    def update_telescope_automated_scheduling(self, telescope_id: str, enabled: bool) -> bool:
        """PATCH /telescopes to toggle automated scheduling on/off."""
        pass

    @abstractmethod
    def upload_optical_observations(
        self,
        observations: list,
        telescope_record: dict,
        sensor_location: dict,
        task_id: str | None = None,
    ) -> bool:
        """
        POST /observations/optical - submit satellite observations extracted from an image.

        Args:
            observations: list of observation dicts (from satellite_matcher.satellite_observations)
            telescope_record: full telescope dict from the API (provides angularNoise, spectral
                wavelength bounds, and telescope UUID)
            sensor_location: dict with latitude, longitude, altitude keys (metres)
            task_id: optional task UUID to attach to each observation

        Returns True on success, False otherwise.
        """
        pass
