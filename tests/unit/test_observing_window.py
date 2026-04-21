"""Tests for compute_observing_window() and ObservingWindow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from citrasense.location.twilight import NAUTICAL_DEG, ObservingWindow, compute_observing_window


class _FakeAltitude:
    def __init__(self, deg: float):
        self.degrees = deg


class _FakeAltaz:
    def __init__(self, alt_deg: float):
        self._alt = _FakeAltitude(alt_deg)

    def __getitem__(self, idx: int):
        if idx == 0:
            return self._alt
        raise IndexError


def _mock_skyfield_env(sun_alt, crossing_times=None, crossing_events=None):
    """Return context-manager patches for Skyfield objects used by compute_observing_window."""
    mock_ts = MagicMock()
    mock_eph = MagicMock()

    mock_observer = MagicMock()
    mock_apparent = MagicMock()
    mock_apparent.altaz.return_value = _FakeAltaz(sun_alt)
    mock_observe = MagicMock()
    mock_observe.apparent.return_value = mock_apparent
    mock_observer.at.return_value.observe.return_value = mock_observe

    earth = MagicMock()
    earth.__add__ = MagicMock(return_value=mock_observer)

    def eph_getitem(self_or_key, key=None):
        actual_key = key if key is not None else self_or_key
        if actual_key == "earth":
            return earth
        return MagicMock()

    mock_eph.__getitem__ = MagicMock(side_effect=lambda key: earth if key == "earth" else MagicMock())

    mock_almanac = MagicMock()
    if crossing_times is None:
        crossing_times = []
        crossing_events = []
    mock_almanac.find_discrete.return_value = (crossing_times, crossing_events)
    mock_almanac.risings_and_settings.return_value = MagicMock()

    mock_wgs84 = MagicMock()

    return mock_ts, mock_eph, mock_almanac, mock_wgs84


def test_observing_window_daytime():
    """When the sun is above the threshold, is_dark should be False."""
    mock_ts, mock_eph, mock_almanac, mock_wgs84 = _mock_skyfield_env(sun_alt=10.0)

    with (
        patch("citrasense.location.twilight._get_skyfield_objects", return_value=(mock_ts, mock_eph)),
        patch("skyfield.almanac.find_discrete", mock_almanac.find_discrete),
        patch("skyfield.almanac.risings_and_settings", mock_almanac.risings_and_settings),
        patch("skyfield.api.wgs84", mock_wgs84),
    ):
        result = compute_observing_window(38.0, -105.0)

    assert isinstance(result, ObservingWindow)
    assert result.is_dark is False
    assert result.current_sun_altitude == 10.0
    assert result.dark_start is None
    assert result.dark_end is None


def test_observing_window_dark():
    """When the sun is below the threshold, is_dark should be True with window bounds."""
    now = datetime.now(timezone.utc)
    t_set = MagicMock()
    t_set.utc_datetime.return_value = now - timedelta(hours=2)
    t_set.utc_iso.return_value = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

    t_rise = MagicMock()
    t_rise.utc_datetime.return_value = now + timedelta(hours=6)
    t_rise.utc_iso.return_value = (now + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    mock_ts, mock_eph, mock_almanac, mock_wgs84 = _mock_skyfield_env(
        sun_alt=-20.0,
        crossing_times=[t_set, t_rise],
        crossing_events=[False, True],  # set, then rise
    )

    with (
        patch("citrasense.location.twilight._get_skyfield_objects", return_value=(mock_ts, mock_eph)),
        patch("skyfield.almanac.find_discrete", mock_almanac.find_discrete),
        patch("skyfield.almanac.risings_and_settings", mock_almanac.risings_and_settings),
        patch("skyfield.api.wgs84", mock_wgs84),
    ):
        result = compute_observing_window(38.0, -105.0)

    assert result.is_dark is True
    assert result.current_sun_altitude == -20.0
    assert result.dark_start is not None
    assert result.dark_end is not None


def test_observing_window_daytime_returns_next_dark_start():
    """During daytime, dark_start should be the next setting time (sun dropping below threshold)."""
    now = datetime.now(timezone.utc)
    t_next_set = MagicMock()
    t_next_set.utc_datetime.return_value = now + timedelta(hours=4)
    t_next_set.utc_iso.return_value = (now + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")

    t_next_rise = MagicMock()
    t_next_rise.utc_datetime.return_value = now + timedelta(hours=14)
    t_next_rise.utc_iso.return_value = (now + timedelta(hours=14)).strftime("%Y-%m-%dT%H:%M:%SZ")

    mock_ts, mock_eph, mock_almanac, mock_wgs84 = _mock_skyfield_env(
        sun_alt=10.0,
        crossing_times=[t_next_set, t_next_rise],
        crossing_events=[False, True],  # set (future), then rise (future)
    )

    with (
        patch("citrasense.location.twilight._get_skyfield_objects", return_value=(mock_ts, mock_eph)),
        patch("skyfield.almanac.find_discrete", mock_almanac.find_discrete),
        patch("skyfield.almanac.risings_and_settings", mock_almanac.risings_and_settings),
        patch("skyfield.api.wgs84", mock_wgs84),
    ):
        result = compute_observing_window(38.0, -105.0)

    assert result.is_dark is False
    assert result.dark_start is not None
    assert result.dark_end is None


def test_observing_window_threshold_boundary():
    """At exactly -12 degrees, is_dark should be False (< not <=)."""
    mock_ts, mock_eph, mock_almanac, mock_wgs84 = _mock_skyfield_env(sun_alt=-12.0)

    with (
        patch("citrasense.location.twilight._get_skyfield_objects", return_value=(mock_ts, mock_eph)),
        patch("skyfield.almanac.find_discrete", mock_almanac.find_discrete),
        patch("skyfield.almanac.risings_and_settings", mock_almanac.risings_and_settings),
        patch("skyfield.api.wgs84", mock_wgs84),
    ):
        result = compute_observing_window(38.0, -105.0, sun_altitude_threshold=NAUTICAL_DEG)

    # At exactly -12.0, not strictly less than -12.0
    assert result.is_dark is False
