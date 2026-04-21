"""Time synchronization monitoring thread for CitraSense."""

import threading
from typing import TYPE_CHECKING, Optional

from citrasense.logging import CITRASENSE_LOGGER
from citrasense.time.time_health import TimeHealth, TimeStatus
from citrasense.time.time_sources import AbstractTimeSource, ChronyTimeSource, NTPTimeSource

if TYPE_CHECKING:
    from citrasense.location.gps_monitor import GPSMonitor


class TimeMonitor:
    """
    Background thread that monitors system clock synchronization.

    Periodically checks clock offset against NTP servers and logs
    warnings/errors based on drift severity.  Enforcement (pausing
    task processing, blocking slews) is handled by ``TimeHealthCheck``
    in the SafetyMonitor framework.
    """

    def __init__(
        self,
        check_interval_minutes: int = 5,
        pause_threshold_ms: float = 500.0,
        gps_monitor: Optional["GPSMonitor"] = None,
    ):
        """
        Initialize time monitor.

        Args:
            check_interval_minutes: Minutes between time sync checks
            pause_threshold_ms: Threshold in ms that determines CRITICAL status
            gps_monitor: Optional GPS monitor to get GPS metadata from
        """
        self.check_interval_minutes = check_interval_minutes
        self.pause_threshold_ms = pause_threshold_ms
        self.gps_monitor = gps_monitor

        # Detect and initialize best available time source
        self.time_source: AbstractTimeSource = self._detect_best_source()

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Current health status
        self._current_health: TimeHealth | None = None

    def start(self) -> None:
        """Start the time monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            CITRASENSE_LOGGER.warning("Time monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        CITRASENSE_LOGGER.info(f"Time monitor started (check interval: {self.check_interval_minutes} minutes)")

    def stop(self) -> None:
        """Stop the time monitoring thread."""
        if self._thread is None:
            return

        CITRASENSE_LOGGER.info("Stopping time monitor...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        CITRASENSE_LOGGER.info("Time monitor stopped")

    def get_current_health(self) -> TimeHealth | None:
        """Get the current time health status (thread-safe)."""
        with self._lock:
            return self._current_health

    def _detect_best_source(self) -> AbstractTimeSource:
        """
        Detect and return the best available time source.

        Priority order: Chrony > NTP

        Returns:
            The best available time source instance.
        """
        # Try ChronyTimeSource
        chrony_source = ChronyTimeSource()
        if chrony_source.is_available():
            CITRASENSE_LOGGER.info("Time monitor initialized with Chrony source")
            return chrony_source

        # Fall back to NTP
        CITRASENSE_LOGGER.info("Time monitor initialized with NTP source")
        return NTPTimeSource()

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        # Perform initial check immediately
        self._check_time_sync()

        # Then check periodically
        interval_seconds = self.check_interval_minutes * 60

        while not self._stop_event.is_set():
            # Wait for interval or stop signal
            if self._stop_event.wait(timeout=interval_seconds):
                break

            self._check_time_sync()

    def _check_time_sync(self) -> None:
        """Perform a single time synchronization check."""
        try:
            # Query time source for offset
            offset_ms = self.time_source.get_offset_ms()

            # Get metadata from GPS monitor if available
            metadata = None
            if self.gps_monitor:
                fix = self.gps_monitor.get_current_fix()
                if fix and fix.fix_mode > 0:
                    metadata = {
                        "satellites": fix.satellites,
                        "fix_mode": fix.fix_mode,
                    }

            # Determine source name (GPS if available, otherwise time source name)
            source_name = self.time_source.get_source_name()
            if metadata and self.time_source.get_source_name() == "chrony":
                # If we have GPS metadata and using chrony, report as GPS
                source_name = "gps"

            # Calculate health status
            health = TimeHealth.from_offset(
                offset_ms=offset_ms,
                source=source_name,
                pause_threshold=self.pause_threshold_ms,
                metadata=metadata,
            )

            # Store current health (thread-safe)
            with self._lock:
                self._current_health = health

            # Log based on status
            self._log_health_status(health)

        except Exception as e:
            CITRASENSE_LOGGER.error(f"Time sync check failed: {e}", exc_info=True)
            # Create unknown status on error
            health = TimeHealth.from_offset(
                offset_ms=None,
                source="unknown",
                pause_threshold=self.pause_threshold_ms,
                message=f"Check failed: {e}",
            )
            with self._lock:
                self._current_health = health

    def _log_health_status(self, health: TimeHealth) -> None:
        """Log time health status at appropriate level."""
        if health.offset_ms is None:
            CITRASENSE_LOGGER.warning("Time sync check failed - offset unknown")
            return

        offset_str = f"{health.offset_ms:+.1f}ms"

        if health.status == TimeStatus.OK:
            CITRASENSE_LOGGER.info(f"Time sync OK: {offset_str}")
        elif health.status == TimeStatus.CRITICAL:
            CITRASENSE_LOGGER.critical(
                f"CRITICAL time drift: offset {offset_str} exceeds {self.pause_threshold_ms}ms threshold. "
                "Task processing will be paused."
            )
