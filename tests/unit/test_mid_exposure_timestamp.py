"""Tests for the mid-exposure timestamp offset in the satellite matcher.

Validates that DATE-OBS + EXPTIME/2 produces the correct mid-exposure epoch
and ISO timestamp string, matching the logic in SatelliteMatcherProcessor._match_satellites().

Uses the real _parse_fits_timestamp -> Epoch -> to_datetime() path so these
tests break if the production parsing changes.
"""

from datetime import datetime, timedelta, timezone

from citrasense.pipelines.optical.satellite_matcher_processor import SatelliteMatcherProcessor


def _compute_mid_exposure(timestamp_str: str, exptime: float) -> tuple[datetime, str]:
    """Reproduce the mid-exposure logic from satellite_matcher_processor.py.

    Uses the real _parse_fits_timestamp to stay in sync with the production path.
    """
    proc = SatelliteMatcherProcessor()
    epoch = proc._parse_fits_timestamp(timestamp_str)
    dt = epoch.to_datetime().replace(tzinfo=timezone.utc)
    if exptime > 0:
        dt = dt + timedelta(seconds=exptime / 2.0)
    return dt, dt.isoformat()


def test_mid_exposure_offset_10s():
    mid_dt, mid_str = _compute_mid_exposure("2026-03-10T02:11:32.202000", 10.0)
    assert mid_dt == datetime(2026, 3, 10, 2, 11, 37, 202000, tzinfo=timezone.utc)
    assert mid_str == "2026-03-10T02:11:37.202000+00:00"


def test_mid_exposure_offset_zero():
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.000000", 0.0)
    assert mid_dt == datetime(2026, 3, 10, 2, 11, 32, tzinfo=timezone.utc)


def test_mid_exposure_offset_fractional():
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.000000", 3.0)
    assert mid_dt == datetime(2026, 3, 10, 2, 11, 33, 500000, tzinfo=timezone.utc)


def test_mid_exposure_with_z_suffix():
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.202000Z", 10.0)
    assert mid_dt == datetime(2026, 3, 10, 2, 11, 37, 202000, tzinfo=timezone.utc)


def test_mid_exposure_nina_7digit_timestamp():
    """NINA writes 7 fractional digits; _parse_fits_timestamp normalizes them."""
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.1054519", 4.0)
    expected = datetime(2026, 3, 10, 2, 11, 34, 105000, tzinfo=timezone.utc)
    assert abs((mid_dt - expected).total_seconds()) < 0.001


def test_mid_exposure_no_fractional_seconds():
    mid_dt, mid_str = _compute_mid_exposure("2026-03-10T02:11:32", 6.0)
    assert mid_dt == datetime(2026, 3, 10, 2, 11, 35, tzinfo=timezone.utc)
    assert "2026-03-10T02:11:35" in mid_str


def test_offset_is_half_exptime_not_full():
    """Guard against accidentally using EXPTIME instead of EXPTIME/2."""
    mid_dt, _ = _compute_mid_exposure("2026-03-10T00:00:00.000000", 20.0)
    assert mid_dt == datetime(2026, 3, 10, 0, 0, 10, tzinfo=timezone.utc)


def test_roundtrip_preserves_utc():
    """Epoch.to_datetime() returns naive — verify we tag UTC and it survives from_datetime()."""
    proc = SatelliteMatcherProcessor()
    epoch = proc._parse_fits_timestamp("2026-06-15T12:00:00.000000")
    dt = epoch.to_datetime().replace(tzinfo=timezone.utc)
    mid = dt + timedelta(seconds=5.0)
    from keplemon.time import Epoch as KepEpoch

    roundtripped = KepEpoch.from_datetime(mid)
    rt_dt = roundtripped.to_datetime().replace(tzinfo=timezone.utc)
    assert (
        abs((rt_dt - mid).total_seconds()) < 0.001
    ), f"Roundtrip diverged: {mid} -> {rt_dt} (diff={abs((rt_dt - mid).total_seconds())}s)"
