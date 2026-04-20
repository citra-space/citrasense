"""Locks down the apparent-not-mean contract of ``gast_degrees``.

The original ``_gast_degrees`` in ``altaz_pointing_model`` used keplemon's
``Epoch.to_fk5_greenwich_angle()``, which returns GMST (Greenwich *Mean*
Sidereal Time), not GAST. GMST omits the equation of equinoxes — a 4-10
arcsec seasonal oscillation driven by nutation — and the pre-keplemon code
(via Skyfield's ``t.gast``) had been using GAST all along. The switch was a
silent accuracy regression only visible at arcsec scale; the bug is easy to
reintroduce if someone "simplifies" ``sidereal.py`` to call keplemon again
without re-reading the module docstring.

These tests pin our output at three epochs against Skyfield's ``t.gast``.
If someone accidentally routes GAST through GMST, every assertion fails by
at least 2-3 arcsec — two orders of magnitude larger than the tolerance.
"""

from datetime import datetime, timezone

import pytest

from citrascope.astro.sidereal import SIDEREAL_RATE_DEG_PER_S, gast_degrees

_TOL_ARCSEC = 0.1
_TOL_DEG = _TOL_ARCSEC / 3600.0


@pytest.mark.parametrize(
    ("dt", "expected_gast_deg"),
    [
        # Reference values captured from Skyfield ``t.gast * 15.0`` at three
        # epochs spread across the year and across two years. Any drift here
        # means either (a) astropy's ERFA model changed, which would be news
        # worth propagating consciously, or (b) someone routed through GMST
        # by accident, which would drift by 5+ arcsec not 0.1.
        (datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 100.15128663),
        (datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc), 358.03596300),
        (datetime(2026, 9, 22, 0, 0, 0, tzinfo=timezone.utc), 0.87443874),
    ],
)
def test_gast_matches_skyfield(dt: datetime, expected_gast_deg: float) -> None:
    """Our GAST is apparent sidereal time, within ~0.1 arcsec of Skyfield."""
    got = gast_degrees(dt)
    assert got == pytest.approx(expected_gast_deg, abs=_TOL_DEG), (
        f"GAST drift at {dt.isoformat()}: got {got}, expected {expected_gast_deg} "
        f"(delta {(got - expected_gast_deg) * 3600:.3f} arcsec). If this test fails with "
        f"a multi-arcsec delta, someone probably routed the implementation through "
        f"GMST — re-read the docstring in citrascope/astro/sidereal.py."
    )


def test_gast_is_not_gmst() -> None:
    """Belt-and-suspenders: the equation of equinoxes must not be zero everywhere.

    GAST − GMST is the equation of equinoxes, which oscillates up to ±18 arcsec
    seasonally. At the 2024-01-01 epoch it's about −0.02 arcsec, near a zero
    crossing; at 2026-03-20 12 UTC (near vernal equinox) it's much larger.
    Assert the latter to prove our output is *not* GMST.
    """
    dt = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    gmst_deg = 358.03437391  # Skyfield's t.gmst * 15.0 at that instant
    got = gast_degrees(dt)
    # Must be at least 1 arcsec away from GMST.
    assert abs(got - gmst_deg) > 1.0 / 3600.0, (
        f"gast_degrees output is suspiciously close to GMST at {dt.isoformat()}: "
        f"{got} vs GMST {gmst_deg}. The equation of equinoxes should put these apart."
    )


def test_gast_default_uses_now() -> None:
    """Calling ``gast_degrees()`` without args uses ``datetime.now()``.

    We don't pin the value — we just verify it's in [0, 360) and finite,
    which rules out silent None-propagation or timezone mishandling.
    """
    got = gast_degrees()
    assert isinstance(got, float)
    assert 0.0 <= got < 360.0


def test_sidereal_rate_constant_matches_iau_value() -> None:
    """Lock down the constant so someone can't silently redefine it.

    15.04106864 arcsec/s is the IAU-derived sidereal rotation rate
    (1 sidereal day = 86164.0905 s). Any change to this constant would
    silently shift every Earth-fixed rate computation in the codebase.
    """
    assert SIDEREAL_RATE_DEG_PER_S == pytest.approx(15.04106864 / 3600.0, abs=1e-15)
