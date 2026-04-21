"""Time source implementations for CitraSense."""

import subprocess
from abc import ABC, abstractmethod

import ntplib


class AbstractTimeSource(ABC):
    """Abstract base class for time sources."""

    @abstractmethod
    def get_offset_ms(self) -> float | None:
        """
        Get the clock offset in milliseconds.

        Returns:
            Clock offset in milliseconds (positive = system ahead), or None if unavailable.
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this time source."""
        pass

    def get_metadata(self) -> dict | None:
        """
        Get optional metadata about the time source.

        Returns:
            Dictionary with metadata, or None if not applicable.
        """
        return None


class NTPTimeSource(AbstractTimeSource):
    """NTP-based time source using pool.ntp.org."""

    def __init__(self, ntp_server: str = "pool.ntp.org", timeout: int = 5):
        """
        Initialize NTP time source.

        Args:
            ntp_server: NTP server hostname (default: pool.ntp.org)
            timeout: Query timeout in seconds
        """
        self.ntp_server = ntp_server
        self.timeout = timeout
        self.client = ntplib.NTPClient()

    def get_offset_ms(self) -> float | None:
        """
        Query NTP server for clock offset.

        Returns:
            Clock offset in milliseconds, or None if query fails.
        """
        try:
            response = self.client.request(self.ntp_server, version=3, timeout=self.timeout)
            # NTP offset is in seconds, convert to milliseconds
            offset_ms = response.offset * 1000.0
            return offset_ms
        except Exception:
            # Query failed - network issue, timeout, etc.
            return None

    def get_source_name(self) -> str:
        """Get the name of this time source."""
        return "ntp"


class ChronyTimeSource(AbstractTimeSource):
    """Chrony-based time source."""

    def __init__(self, timeout: int = 5):
        """
        Initialize Chrony time source.

        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout

    def is_available(self) -> bool:
        """
        Check if chrony is available and running.

        Returns:
            True if chronyc command succeeds, False otherwise.
        """
        try:
            result = subprocess.run(
                ["chronyc", "-c", "tracking"],
                capture_output=True,
                timeout=self.timeout,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_offset_ms(self) -> float | None:
        """
        Query chrony for clock offset.

        Returns:
            Clock offset in milliseconds, or None if query fails.
        """
        try:
            # Get tracking info for offset
            tracking_result = subprocess.run(
                ["chronyc", "-c", "tracking"],
                capture_output=True,
                timeout=self.timeout,
                text=True,
                check=True,
            )

            # Parse CSV output: field index 4 is "System time" in seconds
            tracking_fields = tracking_result.stdout.strip().split(",")
            if len(tracking_fields) > 4:
                offset_seconds = float(tracking_fields[4])
                offset_ms = offset_seconds * 1000.0
                return offset_ms
            else:
                return None

        except Exception:
            return None

    def get_source_name(self) -> str:
        """Get the name of this time source."""
        return "chrony"
