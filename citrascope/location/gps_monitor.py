"""GPS monitoring for CitraScope.

Monitors GPS receiver via gpsd/gpspipe to provide location and fix quality information.
"""

import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from citrascope.logging import CITRASCOPE_LOGGER


@dataclass
class GPSFix:
    """GPS fix information.

    An instance may carry only gpsd diagnostics (``gpsd_version``,
    ``device_path``) without position data — ``latitude`` and ``longitude``
    will be ``None`` in that case.  Always check coordinates before using
    them for position calculations.
    """

    latitude: float | None = None  # degrees
    longitude: float | None = None  # degrees
    altitude: float | None = None  # meters
    fix_mode: int = 0  # 0=no fix, 2=2D, 3=3D
    satellites: int = 0  # number of satellites used
    timestamp: float = 0.0  # time.time() when fix was obtained
    eph: float | None = None  # estimated horizontal position error (meters)
    sep: float | None = None  # spherical error probable (meters)

    # gpsd subsystem diagnostics (from VERSION and DEVICES messages)
    gpsd_version: str | None = None  # e.g. "3.25"
    device_path: str | None = None  # e.g. "/dev/ttyACM0"
    device_driver: str | None = None  # e.g. "u-blox"

    @property
    def is_strong_fix(self) -> bool:
        """Check if this is a strong GPS fix (3D with 4+ satellites and valid coordinates)."""
        return (
            self.fix_mode >= 3
            and self.satellites >= 4
            and self.latitude is not None
            and self.longitude is not None
            and self.altitude is not None
        )


class GPSMonitor:
    """
    GPS monitoring with on-demand queries and caching.

    Architecture:
    - get_current_fix(): Returns cached GPS data (30s TTL) or queries gpsd if stale
    - Background thread: Monitors fix quality changes for server update callbacks

    Caching prevents repeated expensive subprocess calls during burst captures.
    GPS coordinates change slowly enough that 30-second caching is safe.
    """

    # Cache TTL for GPS fixes (coordinates don't change rapidly)
    CACHE_TTL_SECONDS = 30

    def __init__(
        self,
        check_interval_minutes: float = 5,
        fix_callback: Callable[[GPSFix], None] | None = None,
    ):
        """
        Initialize GPS monitor.

        Args:
            check_interval_minutes: Minutes between GPS checks
            fix_callback: Optional callback function called with GPSFix when fix quality changes
        """
        self.check_interval_minutes = check_interval_minutes
        self.fix_callback = fix_callback

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Current fix status
        self._current_fix: GPSFix | None = None
        self._last_fix_mode = 0

        # Cache for get_current_fix() to avoid repeated subprocess calls
        self._cached_fix: GPSFix | None = None
        self._cache_timestamp: float = 0.0

    def is_available(self) -> bool:
        """
        Check if GPS is available (gpsd running and responsive).

        Returns:
            True if gpsd is running and responsive, False otherwise.
        """
        try:
            # Try to actually query gpsd with minimal request
            # We just check if gpsd responds, not if it has a fix yet
            result = subprocess.run(
                ["gpspipe", "-w", "-n", "1"],
                capture_output=True,
                timeout=2,
                text=True,
            )
            # Success if command runs (gpsd is responding)
            # Don't require output - GPS might not have a fix yet at boot
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False
        except Exception:
            return False

    def start(self) -> None:
        """Start the GPS monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            CITRASCOPE_LOGGER.warning("GPS monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        # Log interval in human-readable format
        if self.check_interval_minutes < 1:
            interval_seconds = int(self.check_interval_minutes * 60)
            CITRASCOPE_LOGGER.info(f"GPS monitor started (check interval: {interval_seconds} seconds)")
        else:
            CITRASCOPE_LOGGER.info(f"GPS monitor started (check interval: {self.check_interval_minutes} minutes)")

    def stop(self) -> None:
        """Stop the GPS monitoring thread."""
        if self._thread is None:
            return

        CITRASCOPE_LOGGER.info("Stopping GPS monitor...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        CITRASCOPE_LOGGER.info("GPS monitor stopped")

    def get_current_fix(self, allow_blocking: bool = True) -> GPSFix | None:
        """
        Get current GPS fix with smart caching.

        Returns cached fix if less than 30 seconds old, otherwise:
        - If allow_blocking=True: Queries gpsd for fresh data (may block up to 5 seconds)
        - If allow_blocking=False: Returns last known fix from background thread (may be up to 5 min old)

        Args:
            allow_blocking: If False, returns last known fix instead of blocking on subprocess.
                           Use False when calling from async contexts to avoid blocking event loop.

        Returns:
            Current GPS fix, or None if unavailable
        """
        with self._lock:
            now = time.time()
            # Return fresh cache if available (recent query within 30s)
            if self._cached_fix and (now - self._cache_timestamp) < self.CACHE_TTL_SECONDS:
                return self._cached_fix

            # Cache is stale - if non-blocking, return last known fix from background thread
            if not allow_blocking:
                return self._current_fix

        # Cache stale and blocking allowed - query fresh data (may block up to 5 seconds)
        fix = self._query_gpsd()

        # Update cache if we got valid data
        if fix:
            with self._lock:
                self._cached_fix = fix
                self._cache_timestamp = time.time()

        return fix

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        # Perform initial check immediately
        self._check_gps()

        # Then check periodically
        interval_seconds = self.check_interval_minutes * 60

        while not self._stop_event.is_set():
            # Wait for interval or stop signal
            if self._stop_event.wait(timeout=interval_seconds):
                break

            self._check_gps()

    def _check_gps(self) -> None:
        """Perform a single GPS check."""
        try:
            fix = self._query_gpsd()

            # Store current fix and update cache (thread-safe)
            # This ensures non-blocking callers (web app) always get recent data
            with self._lock:
                self._current_fix = fix
                if fix:
                    self._cached_fix = fix
                    self._cache_timestamp = time.time()

            # Log based on fix quality
            if fix:
                # Check if fix mode changed for logging purposes
                fix_mode_changed = fix.fix_mode != self._last_fix_mode
                self._log_fix_status(fix, fix_mode_changed)

                # Track fix mode for logging changes
                if fix_mode_changed:
                    self._last_fix_mode = fix.fix_mode

                # Always notify callback (rate limiting happens in callback)
                if self.fix_callback:
                    try:
                        self.fix_callback(fix)
                    except Exception as e:
                        CITRASCOPE_LOGGER.error(f"GPS fix callback failed: {e}", exc_info=True)
            else:
                CITRASCOPE_LOGGER.warning("GPS fix unavailable")
                self._last_fix_mode = 0

        except Exception as e:
            CITRASCOPE_LOGGER.error(f"GPS check failed: {e}", exc_info=True)
            with self._lock:
                self._current_fix = None

    def _query_gpsd(self) -> GPSFix | None:
        """
        Query gpsd for GPS fix information using gpspipe.

        Returns:
            GPSFix object with location and fix quality, or None if unavailable.
        """
        try:
            # Use gpspipe to get JSON output from gpsd
            # Request 10 messages to ensure we get both TPV (position) and SKY (satellites)
            result = subprocess.run(
                ["gpspipe", "-w", "-n", "10"],
                capture_output=True,
                timeout=5,
                text=True,
            )

            if result.returncode != 0:
                return None

            # Parse JSON lines to extract data
            fix = GPSFix(timestamp=time.time())

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    msg_class = data.get("class")

                    if msg_class == "VERSION":
                        fix.gpsd_version = data.get("release")

                    elif msg_class == "DEVICES":
                        devices = data.get("devices", [])
                        if devices:
                            dev = devices[0]
                            fix.device_path = dev.get("path")
                            fix.device_driver = dev.get("driver")

                    elif msg_class == "TPV":
                        if "mode" in data:
                            fix.fix_mode = data["mode"]
                        if "lat" in data:
                            fix.latitude = data["lat"]
                        if "lon" in data:
                            fix.longitude = data["lon"]
                        if "alt" in data:
                            fix.altitude = data["alt"]
                        if "eph" in data:
                            fix.eph = data["eph"]
                        if "sep" in data:
                            fix.sep = data["sep"]

                    elif msg_class == "SKY":
                        # Prefer uSat (used satellites) if available
                        if "uSat" in data:
                            fix.satellites = data["uSat"]
                        # Fall back to counting satellites array
                        elif "satellites" in data:
                            fix.satellites = len([s for s in data["satellites"] if s.get("used", False)])

                except json.JSONDecodeError:
                    continue

            # Return fix if we got position data OR gpsd subsystem info
            if fix.latitude is not None and fix.longitude is not None:
                return fix
            elif fix.gpsd_version is not None or fix.device_path is not None:
                return fix
            else:
                return None

        except (FileNotFoundError, OSError):
            # gpspipe not available or gpsd not running
            return None
        except Exception as e:
            CITRASCOPE_LOGGER.debug(f"Could not query gpsd: {e}")
            return None

    def _log_fix_status(self, fix: GPSFix, fix_mode_changed: bool = False) -> None:
        """Log GPS fix status at appropriate level.

        Args:
            fix: GPS fix information
            fix_mode_changed: True if fix mode changed since last check (logs at INFO)
        """
        if fix.latitude is None or fix.longitude is None:
            CITRASCOPE_LOGGER.warning("GPS position unavailable")
            return

        fix_type = ["no fix", "no fix", "2D", "3D"][min(fix.fix_mode, 3)]
        location_str = f"lat={fix.latitude:.6f}°, lon={fix.longitude:.6f}°"
        if fix.altitude is not None:
            location_str += f", alt={fix.altitude:.1f}m"

        if fix.is_strong_fix:
            # Log at INFO only when fix mode changes, otherwise DEBUG for routine checks
            if fix_mode_changed:
                CITRASCOPE_LOGGER.info(f"GPS strong fix: {location_str} ({fix.satellites} sats, {fix_type})")
            else:
                CITRASCOPE_LOGGER.debug(f"GPS strong fix: {location_str} ({fix.satellites} sats, {fix_type})")
        elif fix.fix_mode >= 2:
            # Weak fix: log at INFO only when fix mode changes
            if fix_mode_changed:
                CITRASCOPE_LOGGER.info(f"GPS weak fix: {location_str} ({fix.satellites} sats, {fix_type})")
            else:
                CITRASCOPE_LOGGER.debug(f"GPS weak fix: {location_str} ({fix.satellites} sats, {fix_type})")
        else:
            # No fix is always a problem - keep at WARNING level
            CITRASCOPE_LOGGER.warning(f"GPS no fix: {fix.satellites} sats")
