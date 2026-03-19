import os

import httpx
from keplemon.elements import TopocentricElements
from keplemon.enums import TimeSystem
from keplemon.time import Epoch

from .abstract_api_client import AbstractCitraApiClient


class CitraApiClient(AbstractCitraApiClient):
    @property
    def cache_source_key(self) -> str:
        return self.base_url

    def put_telescope_status(self, body):
        """
        PUT to /telescopes to report online status.
        """
        try:
            response = self._request("PUT", "/telescopes", json=body)
            if self.logger:
                self.logger.debug(f"PUT /telescopes: {response}")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed PUT /telescopes: {e}")
            return None

    def __init__(self, host: str, token: str, use_ssl: bool = True, logger=None):
        self.base_url = ("https" if use_ssl else "http") + "://" + host
        self.token = token
        self.logger = logger
        self.client = httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {self.token}"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _request(self, method: str, endpoint: str, **kwargs):
        try:
            resp = self.client.request(method, endpoint, **kwargs)
            if self.logger:
                self.logger.debug(f"{method} {endpoint}: {resp.status_code} {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if self.logger:
                # Check if response is HTML (e.g., Cloudflare error pages)
                content_type = e.response.headers.get("content-type", "")
                response_text = e.response.text

                if "text/html" in content_type or response_text.strip().startswith("<"):
                    # Log only status and a brief message for HTML responses, sometimes we get Cloudflare error pages
                    self.logger.error(
                        f"HTTP error: {e.response.status_code} - "
                        f"Received HTML error page (likely Cloudflare or server error) for {method} {endpoint}"
                    )
                else:
                    # Log full response for non-HTML errors (JSON, plain text, etc.)
                    self.logger.error(f"HTTP error: {e.response.status_code} {response_text}")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Request error: {e}")
            return None

    def does_api_server_accept_key(self):
        """Check if the API key is valid."""
        response = self._request("GET", "/auth/personal-access-tokens")
        return response is not None

    def get_telescope(self, telescope_id):
        """Check if the telescope ID is valid."""
        return self._request("GET", f"/telescopes/{telescope_id}")

    def get_satellite(self, satellite_id):
        """Fetch satellite details from /satellites/{satellite_id}"""
        return self._request("GET", f"/satellites/{satellite_id}")

    def get_best_elset(self, satellite_id) -> dict | None:
        """Fetch the server-canonical best elset for a satellite."""
        result = self._request("GET", f"/satellites/{satellite_id}/elsets?limit=1")
        if not result or not isinstance(result, list) or len(result) == 0:
            return None
        return result[0]

    def get_telescope_tasks(self, telescope_id):
        """Fetch tasks for a given telescope."""
        return self._request("GET", f"/telescopes/{telescope_id}/tasks")

    def get_ground_station(self, ground_station_id):
        """Fetch ground station details from /ground-stations/{ground_station_id}"""
        return self._request("GET", f"/ground-stations/{ground_station_id}")

    def get_elsets_latest(self, days: int = 14):
        """Fetch all latest elsets from GET /elsets/latest for satellite matching hot list.

        Uses a long timeout because the response can be ~26MB.
        """
        return self._request("GET", f"/elsets/latest?days={days}", timeout=300.0)

    def update_telescope_automated_scheduling(self, telescope_id: str, enabled: bool) -> bool:
        payload = [{"id": telescope_id, "automatedScheduling": enabled}]
        response = self._request("PATCH", "/telescopes", json=payload)
        return response is not None

    def upload_optical_observations(
        self,
        observations: list,
        telescope_record: dict,
        sensor_location: dict,
        task_id: str | None = None,
    ) -> bool:
        """POST /observations/optical with satellite observations extracted from an image.

        Maps each observation dict (from satellite_matcher) to the OpticalObservationCreate schema.
        angularNoise and spectral wavelength bounds come from the telescope record; sensor position
        from sensor_location. Returns True on success, False otherwise.
        """
        telescope_id = telescope_record.get("id")
        angular_noise = telescope_record.get("angularNoise")
        min_wavelength = telescope_record.get("spectralMinWavelengthNm")
        max_wavelength = telescope_record.get("spectralMaxWavelengthNm")

        altitude_km = sensor_location.get("altitude", 0.0) / 1000.0

        payload = []
        for obs in observations:
            # Convert J2000/ICRS astrometric RA/Dec to topocentric TEME (server convention)
            obs_epoch = Epoch.from_iso(obs["timestamp"], time_system=TimeSystem.UTC)
            topo = TopocentricElements.from_j2000(obs_epoch, obs["ra"], obs["dec"])

            entry = {
                "satelliteId": obs["norad_id"],
                "telescopeId": telescope_id,
                "epoch": obs["timestamp"],
                "rightAscension": topo.right_ascension,
                "declination": topo.declination,
                "sensorLatitude": sensor_location["latitude"],
                "sensorLongitude": sensor_location["longitude"],
                "sensorAltitude": altitude_km,
                "angularNoise": angular_noise,
            }
            if obs.get("mag") is not None:
                entry["visualMagnitude"] = obs["mag"]
            if task_id is not None:
                entry["taskId"] = task_id
            if min_wavelength is not None:
                entry["minWavelength"] = min_wavelength
            if max_wavelength is not None:
                entry["maxWavelength"] = max_wavelength
            payload.append(entry)

        if not payload:
            if self.logger:
                self.logger.warning("upload_optical_observations: no observations to upload")
            return False

        result = self._request("POST", "/observations/optical", json=payload)
        if result is None:
            if self.logger:
                self.logger.error("upload_optical_observations: POST /observations/optical failed")
            return False

        if self.logger:
            self.logger.info(f"upload_optical_observations: uploaded {len(payload)} observation(s) for task {task_id}")
        return True

    def upload_image(self, task_id, telescope_id, filepath):
        """Upload an image file for a given task."""
        file_size = os.path.getsize(filepath)
        signed_url_response = self._request(
            "POST",
            f"/my/images?filename=citra_task_{task_id}_image.fits&telescope_id={telescope_id}&task_id={task_id}&file_size={file_size}",
        )
        if not signed_url_response or "uploadUrl" not in signed_url_response:
            if self.logger:
                self.logger.error("Failed to get signed URL for image upload.")
            return None

        upload_url = signed_url_response["uploadUrl"]
        fields = signed_url_response["fields"]

        # Prepare the multipart form data
        files = {"file": (os.path.basename(filepath), open(filepath, "rb"), "application/fits")}
        data = fields  # Fields provided in the signed URL response

        # Perform the POST request to upload the file
        try:
            response = httpx.post(upload_url, data=data, files=files)
            if self.logger:
                self.logger.debug(f"Image upload response: {response.status_code} {response.text}")
            response.raise_for_status()
            return signed_url_response.get("resultsUrl")  # Return the results URL if needed
        except httpx.RequestError as e:
            if self.logger:
                self.logger.error(f"Failed to upload image: {e}")
            return None
        finally:
            # Ensure the file is closed after the upload
            files["file"][1].close()

    def mark_task_complete(self, task_id):
        """Mark a task as complete using the API."""
        try:
            body = {"status": "Succeeded"}
            response = self._request("PUT", f"/tasks/{task_id}", json=body)
            if self.logger:
                self.logger.debug(f"Marked task {task_id} as complete: {response}")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to mark task {task_id} as complete: {e}")
            return None

    def mark_task_failed(self, task_id):
        """Mark a task as failed using the API."""
        try:
            body = {"status": "Failed"}
            response = self._request("PUT", f"/tasks/{task_id}", json=body)
            if self.logger:
                self.logger.debug(f"Marked task {task_id} as failed: {response}")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to mark task {task_id} as failed: {e}")
            return None

    def expand_filters(self, filter_names):
        """Expand filter names to full spectral specifications.

        Args:
            filter_names: List of filter name strings (e.g., ["Red", "Ha", "Clear"])

        Returns:
            Response dict with 'filters' array, or None on error
        """
        try:
            body = {"filter_names": filter_names}
            response = self._request("POST", "/filters/expand", json=body)
            if self.logger:
                self.logger.debug(f"POST /filters/expand: {response}")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to expand filters: {e}")
            return None

    def update_telescope_spectral_config(self, telescope_id, spectral_config):
        """Update telescope's spectral configuration.

        Args:
            telescope_id: Telescope UUID string
            spectral_config: Dict with spectral configuration (discrete filters, etc.)

        Returns:
            Response from PATCH request, or None on error
        """
        try:
            body = [{"id": telescope_id, "spectralConfig": spectral_config}]
            response = self._request("PATCH", "/telescopes", json=body)
            if self.logger:
                self.logger.debug(f"PATCH /telescopes spectral_config: {response}")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update telescope spectral config: {e}")
            return None

    def update_ground_station_location(self, ground_station_id, latitude, longitude, altitude):
        """Update ground station's GPS location (for mobile stations).

        Args:
            ground_station_id: Ground station UUID string
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters

        Returns:
            Response from PUT request, or None on error
        """
        try:
            body = {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
            }
            response = self._request("PUT", f"/ground-stations/{ground_station_id}", json=body)
            if self.logger:
                self.logger.debug(f"PUT /ground-stations/{ground_station_id} location: {response}")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update ground station location: {e}")
            return None
