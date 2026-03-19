"""
Utility classes and functions for testing.

This module contains helper classes and mock implementations to facilitate testing.

Classes:
    DummyLogger: A simple logger implementation for capturing log messages during tests.
    MockCitraApiClient: A mock implementation of AbstractCitraApiClient for testing purposes.
"""

from citrascope.api.citra_api_client import AbstractCitraApiClient


class DummyLogger:
    """
    A simple logger implementation for capturing log messages during tests.

    Attributes:
        infos (list): Captures info-level log messages.
        errors (list): Captures error-level log messages.
        debugs (list): Captures debug-level log messages.
    """

    def __init__(self):
        self.infos = []
        self.errors = []
        self.debugs = []

    def info(self, msg):
        """Log an info-level message."""
        self.infos.append(msg)

    def error(self, msg):
        """Log an error-level message."""
        self.errors.append(msg)

    def debug(self, msg):
        """Log a debug-level message."""
        self.debugs.append(msg)


class MockCitraApiClient(AbstractCitraApiClient):
    """
    A mock implementation of AbstractCitraApiClient for testing purposes.

    This class provides mock responses for API client methods, allowing tests to run
    without making actual HTTP requests.
    """

    @property
    def cache_source_key(self) -> str:
        return "MockCitraApiClient"

    def does_api_server_accept_key(self):
        """Simulate API key validation."""
        return True

    def get_telescope(self, telescope_id):
        """Simulate fetching a telescope by ID."""
        return {"id": telescope_id, "name": "Mock Telescope"}

    def get_satellite(self, satellite_id):
        """Simulate fetching a satellite by ID."""
        return {"id": satellite_id, "name": "Mock Satellite"}

    def get_best_elset(self, satellite_id) -> dict | None:
        """Simulate fetching the best elset for a satellite."""
        return {
            "tle": [
                "1 00000U 00000A   25001.00000000  .00000000  00000-0  00000-0 0    09",
                "2 00000   0.0000   0.0000 0000000   0.0000   0.0000  1.00000000    07",
            ],
            "epoch": "2025-01-01T00:00:00Z",
            "creationEpoch": "2025-01-01T00:00:00Z",
        }

    def get_telescope_tasks(self, telescope_id):
        """Simulate fetching tasks for a telescope."""
        return [{"task_id": 1, "description": "Mock Task"}]

    def get_ground_station(self, ground_station_id):
        """Simulate fetching a ground station by ID."""
        return {"id": ground_station_id, "name": "Mock Ground Station"}

    def put_telescope_status(self, body):
        """Mock PUT to /telescopes for online status reporting."""
        return {"status": "ok", "body": body}

    def expand_filters(self, filter_names):
        """Simulate expanding filter names to spectral specifications."""
        return [{"name": name, "wavelength_nm": 550, "bandwidth_nm": 100} for name in filter_names]

    def update_telescope_spectral_config(self, telescope_id, spectral_config):
        """Simulate updating telescope spectral configuration."""
        return {"status": "ok", "telescope_id": telescope_id, "spectral_config": spectral_config}

    def update_ground_station_location(self, ground_station_id, latitude, longitude, altitude):
        """Simulate updating ground station GPS location."""
        return {
            "status": "ok",
            "ground_station_id": ground_station_id,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
        }

    def get_elsets_latest(self, days=14):
        """Simulate fetching latest elsets (same shape as real API for normalization)."""
        return [
            {
                "satelliteId": "12345",
                "satelliteName": "Mock Sat",
                "tle": [
                    "1 12345U 98067A  12345.67890123  .00012345  00000-0  12345-3 0  1234",
                    "2 12345  51.6400 123.4567 0001234  0.0000  0.0000 15.12345678901234",
                ],
            },
        ]

    def update_telescope_automated_scheduling(self, telescope_id, enabled):
        return True

    def upload_optical_observations(self, observations, telescope_record, sensor_location, task_id=None):
        """Simulate uploading optical observations — always succeeds."""
        return True
