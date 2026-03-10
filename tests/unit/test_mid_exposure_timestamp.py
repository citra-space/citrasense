"""Tests for the mid-exposure timestamp offset in the satellite matcher.

Validates that DATE-OBS + EXPTIME/2 produces the correct mid-exposure epoch
and ISO timestamp string, matching the logic in SatelliteMatcherProcessor._match_satellites().
"""

from datetime import datetime, timedelta, timezone

from citrascope.processors.builtin.processor_dependencies import normalize_fits_timestamp


def _compute_mid_exposure(timestamp_str: str, exptime: float) -> tuple[datetime, str]:
    """Reproduce the mid-exposure logic from satellite_matcher_processor.py."""
    dt = datetime.fromisoformat(normalize_fits_timestamp(timestamp_str).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if exptime > 0:
        dt = dt + timedelta(seconds=exptime / 2.0)
    return dt, dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def test_mid_exposure_offset_10s():
    mid_dt, mid_str = _compute_mid_exposure("2026-03-10T02:11:32.202015", 10.0)
    expected = datetime(2026, 3, 10, 2, 11, 37, 202015, tzinfo=timezone.utc)
    assert mid_dt == expected
    assert mid_str == "2026-03-10T02:11:37.202015"


def test_mid_exposure_offset_zero():
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.000000", 0.0)
    expected = datetime(2026, 3, 10, 2, 11, 32, 0, tzinfo=timezone.utc)
    assert mid_dt == expected


def test_mid_exposure_offset_fractional():
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.000000", 3.0)
    expected = datetime(2026, 3, 10, 2, 11, 33, 500000, tzinfo=timezone.utc)
    assert mid_dt == expected


def test_mid_exposure_with_z_suffix():
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.202015Z", 10.0)
    expected = datetime(2026, 3, 10, 2, 11, 37, 202015, tzinfo=timezone.utc)
    assert mid_dt == expected


def test_mid_exposure_nina_7digit_timestamp():
    """NINA writes 7 fractional digits; normalize_fits_timestamp truncates to 6."""
    mid_dt, _ = _compute_mid_exposure("2026-03-10T02:11:32.1054519", 4.0)
    expected = datetime(2026, 3, 10, 2, 11, 34, 105451, tzinfo=timezone.utc)
    assert mid_dt == expected


def test_mid_exposure_no_fractional_seconds():
    mid_dt, mid_str = _compute_mid_exposure("2026-03-10T02:11:32", 6.0)
    expected = datetime(2026, 3, 10, 2, 11, 35, 0, tzinfo=timezone.utc)
    assert mid_dt == expected
    assert mid_str == "2026-03-10T02:11:35.000000"
