"""Location service for CitraScope.

Manages ground station location from GPS or API sources with intelligent fallback.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from citrascope.location.gps_monitor import GPSFix, GPSMonitor
from citrascope.logging import CITRASCOPE_LOGGER

if TYPE_CHECKING:
    from citrascope.api.abstract_api_client import AbstractCitraApiClient
    from citrascope.settings.citrascope_settings import CitraScopeSettings


class LocationService:
    """
    Location service that coordinates GPS updates and provides current location.

    Architecture note: Ground stations have locations (lat/long/alt), not telescopes.
    Telescopes are physically located AT ground stations. For mobile ground stations
    with GPS, location updates dynamically. For fixed ground stations, location is
    static configuration.

    The service manages GPS monitoring internally, starting it if available and
    handling all GPS lifecycle operations.  If gpsd is not yet responsive at
    startup (common with systemd socket activation), the service retries
    periodically until a deadline.
    """

    # GPS polling interval - how often to check GPS for fresh coordinates
    GPS_CHECK_INTERVAL_SECONDS = 10

    # Maximum age for GPS fix before falling back to ground station (5 minutes)
    GPS_MAX_AGE_SECONDS = 300

    # How long to retry gpsd availability after a failed initial check
    GPS_RETRY_TIMEOUT_SECONDS = 300

    # Minimum interval between retry attempts (avoid subprocess spam)
    GPS_RETRY_INTERVAL_SECONDS = 10

    def __init__(
        self,
        api_client: AbstractCitraApiClient | None = None,
        settings: CitraScopeSettings | None = None,
    ):
        """
        Initialize location service with GPS monitoring.

        Args:
            api_client: API client for updating ground station location
            settings: Settings object for configuration
        """
        self.api_client = api_client
        self.settings = settings
        self._ground_station_ref: dict | None = None
        self._lock = threading.Lock()  # Protect _ground_station_ref access
        self._last_server_update = 0.0  # Track last server update for rate limiting
        self._hardware_gps_provider: Callable[[], dict | None] | None = None

        # Initialize GPS monitor with frequent polling (for UI freshness)
        # Server updates are rate-limited in the callback to gps_update_interval_minutes
        self.gps_monitor = GPSMonitor(
            check_interval_minutes=self.GPS_CHECK_INTERVAL_SECONDS / 60,
            fix_callback=self.on_gps_fix_changed,
        )

        self._gps_started = False

        if not self._try_start_gps():
            # gpsd wasn't ready at init — retry in background so we never
            # block the async web event loop with subprocess calls later.
            self._retry_thread = threading.Thread(target=self._gps_retry_loop, daemon=True)
            self._retry_thread.start()

    def _try_start_gps(self) -> bool:
        """Attempt to start GPS monitoring if gpsd is available.

        Calls ``is_available()`` which spawns a short-lived subprocess,
        so this must only be called from synchronous / background contexts.

        Returns:
            True if the GPS monitor is (now) running.
        """
        if self._gps_started:
            return True

        if self.gps_monitor.is_available():
            self.gps_monitor.start()
            self._gps_started = True
            CITRASCOPE_LOGGER.info("GPS monitoring started by location service")
            return True

        return False

    def _gps_retry_loop(self) -> None:
        """Background thread that retries GPS availability until deadline.

        Runs only when gpsd wasn't responsive at init.  Exits on first
        success or after GPS_RETRY_TIMEOUT_SECONDS.
        """
        deadline = time.time() + self.GPS_RETRY_TIMEOUT_SECONDS
        while not self._gps_started and time.time() < deadline:
            time.sleep(self.GPS_RETRY_INTERVAL_SECONDS)
            if self._try_start_gps():
                return
        if not self._gps_started:
            CITRASCOPE_LOGGER.info("GPS not available after retries — location service using API-only mode")

    def get_gps_fix(self, allow_blocking: bool = True) -> GPSFix | None:
        """Get GPS fix data, never blocking on gpsd availability probes.

        Args:
            allow_blocking: If False, never blocks on a subprocess call.
                Use False from async contexts.

        Returns:
            Current GPS fix, or None if GPS is unavailable.
        """
        if self._gps_started:
            return self.gps_monitor.get_current_fix(allow_blocking=allow_blocking)
        return None

    def stop(self) -> None:
        """Stop GPS monitoring."""
        if self._gps_started:
            self.gps_monitor.stop()
            self._gps_started = False

    def set_ground_station(self, ground_station: dict) -> None:
        """
        Set the ground station record reference from API.

        Args:
            ground_station: Ground station record from API (will be kept as reference)
        """
        with self._lock:
            self._ground_station_ref = ground_station

    def set_hardware_gps_provider(self, provider: Callable[[], dict | None]) -> None:
        """Set a hardware GPS provider as a fallback location source.

        The provider is a callable returning a dict with latitude, longitude,
        altitude_msl, satellites, and fix keys — or None.  Used when gpsd
        has no fix (e.g. dead receiver) but a camera has built-in GPS.
        """
        self._hardware_gps_provider = provider
        CITRASCOPE_LOGGER.info("Hardware GPS provider registered as fallback location source")

    def _query_hardware_gps(self) -> dict | None:
        """Try the hardware GPS provider, returning location dict or None."""
        if self._hardware_gps_provider is None:
            return None
        try:
            data = self._hardware_gps_provider()
            if data and data.get("fix"):
                return data
        except Exception:
            pass
        return None

    def on_gps_fix_changed(self, fix: GPSFix) -> None:
        """
        Callback invoked when GPS fix is checked by background thread (every 10 seconds).

        Updates ground station location on Citra API when strong fix is available,
        but rate-limited to gps_update_interval_minutes to avoid API spam.

        Args:
            fix: Current GPS fix information
        """
        if not self.settings or not self.settings.gps_location_updates_enabled:
            return

        lat: float | None = None
        lon: float | None = None
        alt: float | None = None
        source = "gps"

        if fix.is_strong_fix and fix.latitude is not None and fix.longitude is not None and fix.altitude is not None:
            lat, lon, alt = fix.latitude, fix.longitude, fix.altitude
        else:
            hw = self._query_hardware_gps()
            if hw:
                lat, lon, alt = hw["latitude"], hw["longitude"], hw.get("altitude_msl")
                source = "camera_gps"

        if lat is None or lon is None:
            return

        current_time = time.time()
        update_interval_seconds = self.settings.gps_update_interval_minutes * 60
        if current_time - self._last_server_update < update_interval_seconds:
            return

        with self._lock:
            if self.api_client and self._ground_station_ref:
                ground_station_id = self._ground_station_ref["id"]
                result = self.api_client.update_ground_station_location(
                    ground_station_id,
                    lat,
                    lon,
                    alt,
                )
                if result:
                    self._ground_station_ref["latitude"] = lat
                    self._ground_station_ref["longitude"] = lon
                    self._ground_station_ref["altitude"] = alt
                    self._last_server_update = current_time

                    CITRASCOPE_LOGGER.info(
                        "Updated ground station location from %s: lat=%.6f, lon=%.6f, alt=%.1fm",
                        source,
                        lat,
                        lon,
                        alt or 0.0,
                    )

    def get_current_location(self) -> dict | None:
        """Return best available location.

        Priority:
        1. gpsd GPS (strong fix, fresh) — live location for mobile stations
        2. Hardware GPS (camera GPS with fix) — fallback when gpsd is down
        3. Ground station (from API) — static fallback for fixed stations
        """
        if self.settings and self.settings.gps_location_updates_enabled:
            fix = self.get_gps_fix()
            if fix and fix.is_strong_fix:
                age_seconds = time.time() - fix.timestamp
                if age_seconds < self.GPS_MAX_AGE_SECONDS:
                    return {
                        "latitude": fix.latitude,
                        "longitude": fix.longitude,
                        "altitude": fix.altitude,
                        "source": "gps",
                    }
                else:
                    CITRASCOPE_LOGGER.warning(f"GPS fix is stale ({age_seconds:.0f}s old), falling back")

        hw = self._query_hardware_gps()
        if hw:
            return {
                "latitude": hw["latitude"],
                "longitude": hw["longitude"],
                "altitude": hw.get("altitude_msl"),
                "source": "camera_gps",
            }

        with self._lock:
            if self._ground_station_ref:
                return {
                    "latitude": self._ground_station_ref["latitude"],
                    "longitude": self._ground_station_ref["longitude"],
                    "altitude": self._ground_station_ref["altitude"],
                    "source": "ground_station",
                }

        return None
