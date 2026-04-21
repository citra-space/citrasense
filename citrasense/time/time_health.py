"""Time health status calculation and monitoring."""

from dataclasses import dataclass
from enum import Enum


class TimeStatus(str, Enum):
    """Time synchronization status levels."""

    OK = "ok"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class TimeHealth:
    """Time synchronization health status."""

    offset_ms: float | None
    """Clock offset in milliseconds (positive = system clock ahead)."""

    status: TimeStatus
    """Current time sync status level."""

    source: str
    """Time source used (ntp, gps, chrony, unknown)."""

    message: str | None = None
    """Optional status message or error description."""

    metadata: dict | None = None
    """Optional metadata (e.g., GPS satellite count, fix mode)."""

    @staticmethod
    def calculate_status(
        offset_ms: float | None,
        pause_threshold: float,
    ) -> TimeStatus:
        """
        Calculate time status based on offset and pause threshold.

        Args:
            offset_ms: Clock offset in milliseconds (None if check failed)
            pause_threshold: Threshold in milliseconds that triggers task pause

        Returns:
            TimeStatus level (OK, CRITICAL, or UNKNOWN)
        """
        if offset_ms is None:
            return TimeStatus.UNKNOWN

        abs_offset = abs(offset_ms)

        if abs_offset < pause_threshold:
            return TimeStatus.OK
        else:
            return TimeStatus.CRITICAL

    @classmethod
    def from_offset(
        cls,
        offset_ms: float | None,
        source: str,
        pause_threshold: float,
        message: str | None = None,
        metadata: dict | None = None,
    ) -> "TimeHealth":
        """
        Create TimeHealth from offset and pause threshold.

        Args:
            offset_ms: Clock offset in milliseconds
            source: Time source identifier
            pause_threshold: Threshold that triggers task pause
            message: Optional status message
            metadata: Optional metadata dict

        Returns:
            TimeHealth instance
        """
        status = cls.calculate_status(offset_ms, pause_threshold)
        return cls(
            offset_ms=offset_ms,
            status=status,
            source=source,
            message=message,
            metadata=metadata,
        )

    def should_pause_observations(self) -> bool:
        """Check if observations should be paused due to time sync issues."""
        return self.status == TimeStatus.CRITICAL

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "offset_ms": self.offset_ms,
            "status": self.status.value,
            "source": self.source,
            "message": self.message,
            "metadata": self.metadata,
        }
