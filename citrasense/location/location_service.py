"""Location service for CitraSense.

Manages ground station location from GPS or API sources with intelligent fallback.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from citrasense.location.gps_fix import GPSFix
from citrasense.location.gps_monitor import GPSMonitor
from citrasense.logging import CITRASENSE_LOGGER

if TYPE_CHECKING:
    from citrasense.api.abstract_api_client import AbstractCitraApiClient
    from citrasense.settings.citrasense_settings import CitraSenseSettings


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
        settings: CitraSenseSettings | None = None,
    ):
        """
        Initialize location service with GPS monitoring.

        Args:
            api_client: API client for updating ground station location
            settings: Settings object for configuration
        """
        self.logger = CITRASENSE_LOGGER.getChild(type(self).__name__)
        self.api_client = api_client
        self.settings = settings
        self._ground_station_ref: dict | None = None
        self._lock = threading.Lock()  # Protect _ground_station_ref access
        self._last_server_update = 0.0  # Track last server update for rate limiting
        self._hardware_adapter_gps_provider: Callable[[], GPSFix | None] | None = None
        self._hardware_adapter_gps_cache: GPSFix | None = None
        self._hardware_adapter_gps_cache_time: float = 0.0
        self._equipment_poll_started = False

        # Initialize GPS monitor with frequent polling (for UI freshness)
        # Server updates are rate-limited in the callback to gps_update_interval_minutes
        self.gps_monitor = GPSMonitor(
            check_interval_minutes=self.GPS_CHECK_INTERVAL_SECONDS / 60,
            fix_callback=self.on_gps_fix_changed,
        )

        self._gps_started = False

        gps_enabled = settings.gps_monitoring_enabled if settings else True
        if gps_enabled:
            if not self._try_start_gps():
                # gpsd wasn't ready at init — retry in background so we never
                # block the async web event loop with subprocess calls later.
                self._retry_thread = threading.Thread(target=self._gps_retry_loop, daemon=True)
                self._retry_thread.start()
        else:
            self.logger.info("Computer GPS (gpsd) monitoring disabled by configuration")

    @property
    def gps_monitor_started(self) -> bool:
        """Whether the gpsd monitor is actively running."""
        return self._gps_started

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
            self.logger.info("GPS monitoring started by location service")
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
            self.logger.info("GPS not available after retries — location service using API-only mode")

    def get_gpsd_fix(self, allow_blocking: bool = True) -> GPSFix | None:
        """Get GPS fix from the gpsd daemon, never blocking on availability probes.

        Args:
            allow_blocking: If False, never blocks on a subprocess call.
                Use False from async contexts.

        Returns:
            Current gpsd fix, or None if gpsd is unavailable.
        """
        if self._gps_started:
            return self.gps_monitor.get_current_fix(allow_blocking=allow_blocking)
        return None

    def get_best_gps_fix(self, allow_blocking: bool = True) -> GPSFix | None:
        """Return the best available GPS fix: gpsd first, then hardware adapter."""
        fix = self.get_gpsd_fix(allow_blocking=allow_blocking)
        if fix and fix.latitude is not None and fix.longitude is not None:
            return fix
        return self._query_hardware_adapter_gps()

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

    def set_hardware_adapter_gps_provider(self, provider: Callable[[], GPSFix | None]) -> None:
        """Register the hardware adapter's GPS as a fallback location source.

        The provider is a callable returning a GPSFix or None.  Used when gpsd
        has no fix (e.g. dead receiver) but a device like the Moravian camera
        has a built-in GPS receiver.
        """
        self._hardware_adapter_gps_provider = provider
        self.logger.info("Hardware adapter GPS provider registered as fallback location source")

        if not self._gps_started and not self._equipment_poll_started:
            self._equipment_poll_started = True
            threading.Thread(target=self._equipment_gps_poll_loop, daemon=True).start()

    def get_equipment_gps(self) -> GPSFix | None:
        """Return cached equipment GPS fix, or None. Never blocks on hardware I/O."""
        return self._hardware_adapter_gps_cache

    def _query_hardware_adapter_gps(self) -> GPSFix | None:
        """Try the hardware adapter's GPS, returning cached GPSFix or None.

        May return a GPSFix without coordinates (e.g. device detected but still
        acquiring satellites).  Callers needing position data must check
        ``latitude``/``longitude`` before use.

        Results are cached for 30s (matching GPSMonitor's cache TTL) so the
        1-second web status broadcast doesn't hammer the camera's USB bus.
        """
        if self._hardware_adapter_gps_provider is None:
            return self._hardware_adapter_gps_cache

        now = time.time()
        if now - self._hardware_adapter_gps_cache_time < GPSMonitor.CACHE_TTL_SECONDS:
            return self._hardware_adapter_gps_cache

        try:
            fix = self._hardware_adapter_gps_provider()
            self._hardware_adapter_gps_cache = fix
            self._hardware_adapter_gps_cache_time = now
            return fix
        except Exception:
            self._hardware_adapter_gps_cache = None
            self._hardware_adapter_gps_cache_time = now
            return None

    def on_gps_fix_changed(self, fix: GPSFix | None) -> None:
        """
        Callback invoked on every GPS poll cycle (every 10 seconds).

        Performs two jobs:
        1. Location health monitoring — warns only when NO source has a location.
        2. API updates — pushes location to Citra API when a strong fix is available.

        Args:
            fix: Current GPS fix information, or None if gpsd returned nothing.
        """
        gpsd_has_position = fix is not None and fix.latitude is not None and fix.longitude is not None

        if not gpsd_has_position:
            if self.settings and not self.settings.gps_monitoring_enabled:
                return
            adapter_fix = self._query_hardware_adapter_gps()
            has_adapter = (
                adapter_fix is not None and adapter_fix.latitude is not None and adapter_fix.longitude is not None
            )
            with self._lock:
                has_ground_station = self._ground_station_ref is not None
            if has_adapter or has_ground_station:
                self.logger.debug("gpsd has no position (system located via other source)")
            else:
                self.logger.warning("No location available from any source (gpsd, hardware GPS, or ground station)")

        if not self.settings or not self.settings.gps_location_updates_enabled:
            return

        if fix is None:
            return

        lat: float | None = None
        lon: float | None = None
        alt: float | None = None
        source = "gps"

        if fix.is_strong_fix and fix.latitude is not None and fix.longitude is not None and fix.altitude is not None:
            lat, lon, alt = fix.latitude, fix.longitude, fix.altitude
        else:
            adapter_fix = self._query_hardware_adapter_gps()
            if adapter_fix:
                lat, lon, alt = adapter_fix.latitude, adapter_fix.longitude, adapter_fix.altitude
                source = "hardware_adapter_gps"

        if lat is not None and lon is not None:
            self._push_location_update(lat, lon, alt, source)

    def _equipment_gps_poll_loop(self) -> None:
        """Poll equipment GPS for server updates when gpsd is disabled."""
        while True:
            time.sleep(self.GPS_CHECK_INTERVAL_SECONDS)
            fix = self._query_hardware_adapter_gps()
            if fix and fix.latitude is not None and fix.longitude is not None:
                self._push_location_update(fix.latitude, fix.longitude, fix.altitude, "hardware_adapter_gps")

    def _push_location_update(self, lat: float, lon: float, alt: float | None, source: str) -> None:
        """Rate-limited push of a location update to the Citra API."""
        if not self.settings or not self.settings.gps_location_updates_enabled:
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

                    self.logger.info(
                        "Updated ground station location from %s: lat=%.6f, lon=%.6f, alt=%.1fm",
                        source,
                        lat,
                        lon,
                        alt or 0.0,
                    )

    def get_current_location(self) -> dict | None:
        """Return best available location.

        Priority:
        1. gpsd (strong fix, fresh) — live location for mobile stations
        2. Hardware adapter GPS (camera GPS with fix) — fallback when gpsd is down
        3. Ground station (from API) — static fallback for fixed stations
        """
        if self.settings and self.settings.gps_location_updates_enabled:
            fix = self.get_gpsd_fix()
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
                    self.logger.warning(f"GPS fix is stale ({age_seconds:.0f}s old), falling back")

        adapter_fix = self._query_hardware_adapter_gps()
        if adapter_fix and adapter_fix.latitude is not None and adapter_fix.longitude is not None:
            return {
                "latitude": adapter_fix.latitude,
                "longitude": adapter_fix.longitude,
                "altitude": adapter_fix.altitude,
                "source": "hardware_adapter_gps",
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
