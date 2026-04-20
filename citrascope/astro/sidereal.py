"""Shared sidereal-time and Observatory helpers.

Two pieces live here because they were being duplicated (and in one case,
silently miscomputed) across the hardware, web, and api layers:

1. :func:`gast_degrees` — Greenwich Apparent Sidereal Time in degrees.
   Computed via astropy/ERFA's IAU 2006/2000A model so it includes the
   equation of equinoxes (the precession + nutation correction that turns
   GMST into GAST). Keplemon's ``Epoch.to_fk5_greenwich_angle()`` and
   siblings all return GMST despite their names — they omit the equation
   of equinoxes, which is a 4-10 arcsec seasonal oscillation. Using GAST
   here preserves parity with the pre-keplemon Skyfield behaviour.

2. :func:`make_observatory` — builds a keplemon ``Observatory`` from
   meters-altitude input and converts to the km-altitude the constructor
   expects. Centralising the conversion prevents the "did they remember
   to divide by 1000?" review burden at every call site.

The :data:`SIDEREAL_RATE_DEG_PER_S` constant is the IAU sidereal rotation
rate in deg/s, used to convert between J2000-inertial and Earth-fixed
RA rates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from astropy.time import Time

if TYPE_CHECKING:
    from keplemon.bodies import Observatory


# Earth's sidereal rotation rate in deg/s. 15.04106864 arcsec/s is the IAU
# value, derived from one sidereal day = 86164.0905 s. Used to convert a
# J2000-inertial RA rate (dRA/dt relative to the star field) into the
# Earth-fixed RA rate that mount tracking-offset APIs expect.
SIDEREAL_RATE_DEG_PER_S = 15.04106864 / 3600.0


def gast_degrees(when: datetime | None = None) -> float:
    """Greenwich Apparent Sidereal Time in degrees (IAU 2006/2000A via astropy/ERFA).

    Apparent (not mean) — includes the equation of equinoxes. Matches the
    pre-keplemon Skyfield behaviour to sub-arcsec.

    The name is accurate on purpose: keplemon's ``to_fk5_greenwich_angle``
    returns GMST, not GAST, which is why we route through astropy here
    rather than using keplemon directly. See ``citrascope/astro/sidereal.py``
    docstring for the full story.

    Args:
        when: UTC datetime to evaluate at. Defaults to ``datetime.now(timezone.utc)``.

    Returns:
        GAST in degrees in the range [0, 360).
    """
    dt = when if when is not None else datetime.now(timezone.utc)
    # ``.deg`` on astropy Angle is typed as ``UnitScale | NDArray[...]`` by
    # the stubs even when the input is a scalar Time, because astropy
    # supports vectorized inputs. Our input is always a scalar datetime,
    # so the return is always a scalar float; the ignore is the documented
    # pattern in this codebase for astropy stub gaps.
    return float(Time(dt, scale="utc").sidereal_time("apparent", "greenwich").deg)  # type: ignore[arg-type]


def make_observatory(lat_deg: float, lon_deg: float, alt_m: float) -> Observatory:
    """Build a keplemon ``Observatory`` from meters-altitude, converting to km.

    Keplemon's ``Observatory`` constructor takes altitude in kilometres; all
    of citrascope's other layers (location service, config, FITS) use metres.
    This helper owns the single unit conversion so a missed or double
    ``/ 1000.0`` can't drift into production.
    """
    # Imported inside the function so importing ``citrascope.astro.sidereal``
    # doesn't require keplemon at collection time (matches the pattern used
    # elsewhere in the codebase for optional-at-import-time heavy deps).
    from keplemon.bodies import Observatory

    return Observatory(float(lat_deg), float(lon_deg), float(alt_m) / 1000.0)
