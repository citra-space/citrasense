"""Abstract mount device interface."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from citrascope.hardware.devices.abstract_hardware_device import AbstractHardwareDevice

if TYPE_CHECKING:
    from citrascope.hardware.devices.mount.mount_state_cache import MountSnapshot


class AbstractMount(AbstractHardwareDevice):
    """Abstract base class for telescope mount devices.

    Provides a common interface for controlling equatorial and alt-az mounts.
    All RA/Dec coordinates are in **degrees** (project convention).

    When a ``MountStateCache`` is attached (via ``_state_cache``), the
    ``cached_state``, ``cached_mount_info``, and ``cached_limits`` properties
    expose its data.  The adapter creates and attaches the cache — consumers
    never see it directly.
    """

    # ------------------------------------------------------------------
    # Core abstract methods — every mount must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def slew_to_radec(self, ra: float, dec: float) -> bool:
        """Slew the mount to specified RA/Dec coordinates.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees

        Returns:
            True if slew initiated successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_slewing(self) -> bool:
        """Check if mount is currently slewing.

        Returns:
            True if slewing, False if stationary or tracking
        """
        pass

    @abstractmethod
    def abort_slew(self):
        """Stop the current slew operation."""
        pass

    @abstractmethod
    def get_radec(self) -> tuple[float, float]:
        """Get current mount RA/Dec position.

        Returns:
            Tuple of (RA in degrees, Dec in degrees)
        """
        pass

    @abstractmethod
    def start_tracking(self, rate: str | None = "sidereal") -> bool:
        """Start tracking at specified rate.

        Args:
            rate: Tracking rate - "sidereal", "lunar", "solar", or device-specific

        Returns:
            True if tracking started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_tracking(self) -> bool:
        """Stop tracking.

        Returns:
            True if tracking stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def is_tracking(self) -> bool:
        """Check if mount is currently tracking.

        Returns:
            True if tracking, False otherwise
        """
        pass

    @abstractmethod
    def park(self) -> bool:
        """Park the mount to its home position.

        Returns:
            True if park initiated successfully, False otherwise
        """
        pass

    @abstractmethod
    def unpark(self) -> bool:
        """Unpark the mount from its home position.

        Returns:
            True if unpark successful, False otherwise
        """
        pass

    @abstractmethod
    def is_parked(self) -> bool:
        """Check if mount is parked.

        Returns:
            True if parked, False otherwise
        """
        pass

    @abstractmethod
    def get_mount_info(self) -> dict:
        """Get mount capabilities and information.

        Returns:
            Dictionary containing mount specs and capabilities
        """
        pass

    # ------------------------------------------------------------------
    # Optional capability methods — concrete defaults so subclasses only
    # override what they support.
    # ------------------------------------------------------------------

    def sync_to_radec(self, ra: float, dec: float) -> bool:
        """Sync the mount's internal model to the given coordinates.

        Tells the mount that it is currently pointing at (ra, dec).
        Used after plate-solving to correct pointing errors.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees

        Returns:
            True if sync accepted, False otherwise
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sync")

    def set_custom_tracking_rates(self, ra_rate: float, dec_rate: float) -> bool:
        """Set custom tracking rates for satellite or non-sidereal tracking.

        Args:
            ra_rate: RA tracking rate offset in arcseconds per second
            dec_rate: Dec tracking rate offset in arcseconds per second

        Returns:
            True if rates accepted, False if unsupported
        """
        return False

    def guide_pulse(self, direction: str, duration_ms: int) -> bool:
        """Send an autoguiding correction pulse.

        Args:
            direction: One of "north", "south", "east", "west"
            duration_ms: Pulse duration in milliseconds (typically 0-9999)

        Returns:
            True if pulse sent, False if unsupported
        """
        return False

    def set_site_location(self, latitude: float, longitude: float, altitude: float) -> bool:
        """Set the observing site location on the mount.

        Args:
            latitude: Latitude in decimal degrees (positive = North)
            longitude: Longitude in decimal degrees (positive = East)
            altitude: Altitude in metres above sea level

        Returns:
            True if accepted, False if unsupported
        """
        return False

    def get_site_location(self) -> tuple[float, float, float] | None:
        """Get the observing site location stored on the mount.

        Returns:
            (latitude, longitude, altitude) or None if unsupported
        """
        return None

    def sync_datetime(self) -> bool:
        """Sync the system clock to the mount's internal clock.

        Pushes the current UTC date/time so the mount can compute
        sidereal time, horizon limits, and meridian flips.

        Returns:
            True if accepted, False if unsupported
        """
        return False

    def get_limits(self) -> tuple[int | None, int | None]:
        """Get the mount's altitude limits.

        Returns:
            (horizon_limit, overhead_limit) in integer degrees.
            Either value may be None if the mount doesn't report it.
        """
        return None, None

    def set_horizon_limit(self, degrees: int) -> bool:
        """Set the minimum altitude (horizon limit) the mount will slew to.

        Args:
            degrees: Minimum altitude in degrees (typically -30 to +30)

        Returns:
            True if accepted, False if unsupported
        """
        return False

    def set_overhead_limit(self, degrees: int) -> bool:
        """Set the maximum altitude (overhead limit) the mount will slew to.

        Args:
            degrees: Maximum altitude in degrees (typically 60 to 90)

        Returns:
            True if accepted, False if unsupported
        """
        return False

    def get_meridian_auto_flip(self) -> bool | None:
        """Query whether the mount will auto-flip at the meridian.

        Returns:
            True if flip enabled, False if stop-at-meridian, None if unsupported.
        """
        return None

    def set_meridian_auto_flip(self, enabled: bool) -> bool:
        """Enable or disable automatic meridian flip.

        Returns:
            True if accepted, False if unsupported.
        """
        return False

    def set_equatorial_mode(self) -> bool:
        """Switch the mount to equatorial (German EQ / polar-aligned) mode.

        Some mounts boot in alt-az mode and need an explicit command to
        switch to equatorial for proper RA/Dec tracking and meridian flips.

        Returns:
            True if accepted, False if unsupported
        """
        return False

    def get_mount_mode(self) -> str:
        """Get the mount's current operating mode.

        Returns:
            ``"equatorial"``, ``"altaz"``, or ``"unknown"``
        """
        return "unknown"

    def get_altitude_limits_enabled(self) -> bool | None:
        """Query whether altitude limits are enforced.

        Returns:
            True/False, or None if unsupported.
        """
        return None

    def set_altitude_limits_enabled(self, enable: bool) -> None:
        """Enable or disable altitude limit enforcement."""
        pass

    def find_home(self) -> bool:
        """Initiate the mount's homing routine to establish absolute encoder position.

        Homing searches for encoder index marks so the mount knows its
        physical orientation.  This is typically required once after
        power-on before GoTo commands will work.

        Returns:
            True if homing initiated successfully, False if unsupported
        """
        return False

    def is_home(self) -> bool:
        """Check whether the mount has been homed (knows its absolute position).

        Returns:
            True if the mount is at (or has found) its home position
        """
        return False

    # ------------------------------------------------------------------
    # Safety-related optional methods — used by SafetyMonitor for cable
    # wrap protection.  Mounts that don't support these are silently
    # excluded from azimuth-based safety checks.
    # ------------------------------------------------------------------

    def get_azimuth(self) -> float | None:
        """Get current mount azimuth in degrees (0-360).

        Used by CableWrapCheck to track cumulative azimuth rotation
        in alt-az mode.

        Returns:
            Azimuth in degrees, or None if unsupported.
        """
        return None

    def get_altitude(self) -> float | None:
        """Get current mount altitude in degrees.

        Returns:
            Altitude in degrees (0-90 above horizon), or None if unsupported.
        """
        return None

    def start_move(self, direction: str, rate: int = 7) -> bool:
        """Start continuous motion in a cardinal direction.

        Used for directional cable unwinding where explicit CW/CCW
        control is required (GoTo takes shortest path which may worsen wrap).

        Args:
            direction: One of ``"north"``, ``"south"``, ``"east"``, ``"west"``
            rate: Slew rate 0-9 (0 slowest, 9 fastest). Default 7 is moderate.

        Returns:
            True if motion started, False if unsupported.
        """
        return False

    def stop_move(self, direction: str) -> bool:
        """Stop continuous motion in a cardinal direction.

        Args:
            direction: One of ``"north"``, ``"south"``, ``"east"``, ``"west"``

        Returns:
            True if stop issued, False if unsupported.
        """
        return False

    # ------------------------------------------------------------------
    # Cached state — populated by MountStateCache (attached by adapter)
    # ------------------------------------------------------------------

    @property
    def cached_state(self) -> MountSnapshot | None:
        """Latest cached position/status snapshot, or None if no cache."""
        cache = getattr(self, "_state_cache", None)
        return cache.snapshot if cache is not None else None

    @property
    def cached_mount_info(self) -> dict:
        """Cached mount info (model, capabilities). Empty dict if no cache."""
        cache = getattr(self, "_state_cache", None)
        return cache.mount_info if cache is not None else {}

    @property
    def cached_limits(self) -> tuple[int | None, int | None]:
        """Cached altitude limits. (None, None) if no cache."""
        cache = getattr(self, "_state_cache", None)
        return cache.limits if cache is not None else (None, None)
