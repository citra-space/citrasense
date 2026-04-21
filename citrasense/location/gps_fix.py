"""GPSFix — shared GPS fix dataclass used across CitraSense.

Produced by GPSMonitor (gpsd), camera GPS modules (Moravian), and
any future GPS source.  Consumers should check coordinates before
using them — an instance may carry only device diagnostics without
position data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPSFix:
    """GPS fix information.

    An instance may carry only device diagnostics (``device_path``,
    ``device_driver``) without position data — ``latitude`` and
    ``longitude`` will be ``None`` in that case.  Always check
    coordinates before using them for position calculations.
    """

    latitude: float | None = None  # degrees
    longitude: float | None = None  # degrees
    altitude: float | None = None  # meters
    fix_mode: int = 0  # 0=no fix, 2=2D, 3=3D
    satellites: int = 0  # number of satellites used
    timestamp: float = 0.0  # time.time() when fix was obtained
    eph: float | None = None  # estimated horizontal position error (meters)
    sep: float | None = None  # spherical error probable (meters)

    gpsd_version: str | None = None  # e.g. "3.25"
    device_path: str | None = None  # e.g. "/dev/ttyACM0", "camera"
    device_driver: str | None = None  # e.g. "u-blox", "moravian"

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
