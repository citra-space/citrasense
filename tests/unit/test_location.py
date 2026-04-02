"""Unit tests for GPSMonitor and LocationService."""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from citrascope.location.gps_fix import GPSFix
from citrascope.location.gps_monitor import GPSMonitor
from citrascope.location.location_service import LocationService

# ---------------------------------------------------------------------------
# GPSFix dataclass
# ---------------------------------------------------------------------------


def test_gps_fix_defaults():
    fix = GPSFix()
    assert fix.latitude is None
    assert fix.longitude is None
    assert fix.altitude is None
    assert fix.fix_mode == 0
    assert fix.satellites == 0


def test_gps_fix_strong_fix():
    fix = GPSFix(latitude=40.0, longitude=-74.0, altitude=100.0, fix_mode=3, satellites=8)
    assert fix.is_strong_fix is True


def test_gps_fix_weak_2d():
    fix = GPSFix(latitude=40.0, longitude=-74.0, altitude=None, fix_mode=2, satellites=3)
    assert fix.is_strong_fix is False


def test_gps_fix_no_coords():
    fix = GPSFix(fix_mode=3, satellites=8)
    assert fix.is_strong_fix is False


def test_gps_fix_few_sats():
    fix = GPSFix(latitude=40.0, longitude=-74.0, altitude=100.0, fix_mode=3, satellites=2)
    assert fix.is_strong_fix is False


# ---------------------------------------------------------------------------
# GPSMonitor
# ---------------------------------------------------------------------------


def test_gps_monitor_is_available_false():
    gm = GPSMonitor()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert gm.is_available() is False


def test_gps_monitor_is_available_true():
    gm = GPSMonitor()
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        assert gm.is_available() is True


def test_gps_monitor_query_gpsd_parses_tpv():
    gm = GPSMonitor()
    gpsd_output = (
        '{"class":"VERSION"}\n'
        '{"class":"TPV","mode":3,"lat":40.123,"lon":-74.456,"alt":50.0,"eph":5.0,"sep":8.0}\n'
        '{"class":"SKY","uSat":10}\n'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = gpsd_output

    with patch("subprocess.run", return_value=mock_result):
        fix = gm._query_gpsd()

    assert fix is not None
    assert fix.latitude == pytest.approx(40.123)
    assert fix.longitude == pytest.approx(-74.456)
    assert fix.altitude == pytest.approx(50.0)
    assert fix.satellites == 10
    assert fix.fix_mode == 3
    assert fix.eph == pytest.approx(5.0)


def test_gps_monitor_query_gpsd_no_position():
    gm = GPSMonitor()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '{"class":"VERSION"}\n'

    with patch("subprocess.run", return_value=mock_result):
        assert gm._query_gpsd() is None


def test_gps_fix_metadata_only_not_strong():
    fix = GPSFix(gpsd_version="3.25")
    assert fix.is_strong_fix is False
    assert fix.latitude is None


def test_gps_monitor_query_gpsd_metadata_only():
    gm = GPSMonitor()
    gpsd_output = (
        '{"class":"VERSION","release":"3.25","rev":"release-3.25"}\n'
        '{"class":"DEVICES","devices":[{"path":"/dev/ttyACM0","driver":"u-blox"}]}\n'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = gpsd_output

    with patch("subprocess.run", return_value=mock_result):
        fix = gm._query_gpsd()

    assert fix is not None
    assert fix.gpsd_version == "3.25"
    assert fix.device_path == "/dev/ttyACM0"
    assert fix.device_driver == "u-blox"
    assert fix.latitude is None
    assert fix.longitude is None


def test_gps_monitor_query_gpsd_subprocess_error():
    gm = GPSMonitor()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert gm._query_gpsd() is None


def test_gps_monitor_get_current_fix_cache():
    gm = GPSMonitor()
    cached = GPSFix(latitude=1.0, longitude=2.0, timestamp=time.time())
    gm._cached_fix = cached
    gm._cache_timestamp = time.time()
    assert gm.get_current_fix() is cached


def test_gps_monitor_get_current_fix_stale_nonblocking():
    gm = GPSMonitor()
    old_fix = GPSFix(latitude=1.0, longitude=2.0, timestamp=time.time() - 600)
    gm._current_fix = old_fix
    gm._cached_fix = old_fix
    gm._cache_timestamp = time.time() - 60

    result = gm.get_current_fix(allow_blocking=False)
    assert result is old_fix


def test_gps_monitor_check_gps_calls_callback():
    callback = MagicMock()
    gm = GPSMonitor(fix_callback=callback)
    fix = GPSFix(latitude=1.0, longitude=2.0, altitude=100.0, fix_mode=3, satellites=8, timestamp=time.time())
    with patch.object(gm, "_query_gpsd", return_value=fix):
        gm._check_gps()
    callback.assert_called_once_with(fix)


def test_gps_monitor_check_gps_no_fix():
    gm = GPSMonitor()
    with patch.object(gm, "_query_gpsd", return_value=None):
        gm._check_gps()
    assert gm._current_fix is None


def test_gps_monitor_log_fix_strong(caplog):
    gm = GPSMonitor()
    fix = GPSFix(latitude=40.0, longitude=-74.0, altitude=50.0, fix_mode=3, satellites=8)
    gm._log_fix_status(fix, fix_mode_changed=True)


def test_gps_monitor_log_fix_weak(caplog):
    gm = GPSMonitor()
    fix = GPSFix(latitude=40.0, longitude=-74.0, altitude=50.0, fix_mode=2, satellites=3)
    gm._log_fix_status(fix, fix_mode_changed=False)


def test_gps_monitor_log_no_fix():
    gm = GPSMonitor()
    fix = GPSFix(fix_mode=0, satellites=0)
    gm._log_fix_status(fix)


def test_gps_monitor_satellite_count_fallback():
    gm = GPSMonitor()
    gpsd_output = (
        '{"class":"TPV","mode":3,"lat":40.0,"lon":-74.0,"alt":50.0}\n'
        '{"class":"SKY","satellites":[{"used":true},{"used":true},{"used":false}]}\n'
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = gpsd_output

    with patch("subprocess.run", return_value=mock_result):
        fix = gm._query_gpsd()

    assert fix.satellites == 2


def test_gps_monitor_query_gpsd_timeout_parses_partial():
    """TimeoutExpired should still parse whatever output was captured."""
    gm = GPSMonitor()
    partial_output = (
        '{"class":"VERSION","release":"3.25"}\n'
        '{"class":"DEVICES","devices":[{"path":"/dev/ttyAMA0","driver":"MTK-3301"}]}\n'
        '{"class":"TPV","mode":1}\n'
    )
    timeout_exc = subprocess.TimeoutExpired(cmd=["gpspipe"], timeout=5, output=partial_output)

    with patch("subprocess.run", side_effect=timeout_exc):
        fix = gm._query_gpsd()

    assert fix is not None
    assert fix.gpsd_version == "3.25"
    assert fix.device_path == "/dev/ttyAMA0"
    assert fix.device_driver == "MTK-3301"
    assert fix.fix_mode == 1
    assert fix.latitude is None


def test_gps_monitor_query_gpsd_timeout_empty():
    """TimeoutExpired with no captured output should return None."""
    gm = GPSMonitor()
    timeout_exc = subprocess.TimeoutExpired(cmd=["gpspipe"], timeout=5, output=None)

    with patch("subprocess.run", side_effect=timeout_exc):
        assert gm._query_gpsd() is None


# ---------------------------------------------------------------------------
# LocationService
# ---------------------------------------------------------------------------


def test_location_service_gps_unavailable_keeps_monitor():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()
    assert ls.gps_monitor is not None
    assert ls._gps_started is False


def test_location_service_gps_available_at_init():
    with patch.object(GPSMonitor, "is_available", return_value=True), patch.object(GPSMonitor, "start"):
        ls = LocationService()
    assert ls._gps_started is True


def test_location_service_retry_loop_starts_gps():
    """Background retry thread should start GPS when gpsd becomes available."""
    call_count = 0

    def availability_sequence():
        nonlocal call_count
        call_count += 1
        return call_count >= 2

    with (
        patch.object(GPSMonitor, "is_available", side_effect=availability_sequence),
        patch.object(GPSMonitor, "start"),
        patch.object(LocationService, "GPS_RETRY_INTERVAL_SECONDS", 0.01),
        patch.object(LocationService, "GPS_RETRY_TIMEOUT_SECONDS", 5),
    ):
        ls = LocationService()
        assert ls._gps_started is False
        ls._retry_thread.join(timeout=2)

    assert ls._gps_started is True


def test_location_service_retry_loop_gives_up():
    """Background retry thread should stop after deadline."""
    with (
        patch.object(GPSMonitor, "is_available", return_value=False),
        patch.object(LocationService, "GPS_RETRY_INTERVAL_SECONDS", 0.01),
        patch.object(LocationService, "GPS_RETRY_TIMEOUT_SECONDS", 0.03),
    ):
        ls = LocationService()
        ls._retry_thread.join(timeout=2)

    assert ls._gps_started is False


def test_location_service_get_gpsd_fix_never_blocks_on_retry():
    """get_gpsd_fix() must not call _try_start_gps() — retry is background only."""
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    assert ls.get_gpsd_fix(allow_blocking=False) is None
    assert ls.get_gpsd_fix(allow_blocking=True) is None
    assert ls._gps_started is False


def test_location_service_get_gpsd_fix_returns_data_when_started():
    with patch.object(GPSMonitor, "is_available", return_value=True), patch.object(GPSMonitor, "start"):
        ls = LocationService()

    fix_obj = GPSFix(latitude=40.0, longitude=-74.0, altitude=100.0, fix_mode=3, satellites=8, timestamp=time.time())
    with patch.object(GPSMonitor, "get_current_fix", return_value=fix_obj):
        result = ls.get_gpsd_fix(allow_blocking=False)

    assert result is fix_obj


def test_location_service_get_current_location_from_ground_station():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()
    ls.set_ground_station({"id": "gs1", "latitude": 40.0, "longitude": -74.0, "altitude": 100.0})
    loc = ls.get_current_location()
    assert loc["source"] == "ground_station"
    assert loc["latitude"] == 40.0


def test_location_service_get_current_location_none():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()
    assert ls.get_current_location() is None


def test_location_service_on_gps_fix_no_settings():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()
    ls.settings = None
    fix = GPSFix(latitude=40.0, longitude=-74.0, altitude=100.0, fix_mode=3, satellites=8)
    ls.on_gps_fix_changed(fix)


def test_location_service_on_gps_fix_updates_server():
    mock_api = MagicMock()
    mock_api.update_ground_station_location.return_value = True
    mock_settings = MagicMock()
    mock_settings.gps_location_updates_enabled = True
    mock_settings.gps_update_interval_minutes = 0

    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService(api_client=mock_api, settings=mock_settings)

    ls.set_ground_station({"id": "gs1", "latitude": 0.0, "longitude": 0.0, "altitude": 0.0})
    fix = GPSFix(latitude=41.0, longitude=-75.0, altitude=200.0, fix_mode=3, satellites=8)
    ls.on_gps_fix_changed(fix)

    mock_api.update_ground_station_location.assert_called_once_with("gs1", 41.0, -75.0, 200.0)


def test_location_service_on_gps_fix_rate_limited():
    mock_api = MagicMock()
    mock_settings = MagicMock()
    mock_settings.gps_location_updates_enabled = True
    mock_settings.gps_update_interval_minutes = 999

    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService(api_client=mock_api, settings=mock_settings)

    ls.set_ground_station({"id": "gs1", "latitude": 0.0, "longitude": 0.0, "altitude": 0.0})
    ls._last_server_update = time.time()
    fix = GPSFix(latitude=41.0, longitude=-75.0, altitude=200.0, fix_mode=3, satellites=8)
    ls.on_gps_fix_changed(fix)

    mock_api.update_ground_station_location.assert_not_called()


def test_location_service_stop_started():
    with patch.object(GPSMonitor, "is_available", return_value=True), patch.object(GPSMonitor, "start"):
        ls = LocationService()
    assert ls._gps_started is True

    with patch.object(GPSMonitor, "stop"):
        ls.stop()
    assert ls._gps_started is False


def test_location_service_stop_not_started():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()
    ls.stop()
    assert ls._gps_started is False


# ---------------------------------------------------------------------------
# Hardware adapter GPS provider and get_best_gps_fix
# ---------------------------------------------------------------------------


def test_location_service_hardware_adapter_gps_provider():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    adapter_fix = GPSFix(
        latitude=38.82,
        longitude=-104.87,
        altitude=1660.0,
        fix_mode=3,
        satellites=5,
        timestamp=time.time(),
        device_path="camera",
        device_driver="moravian",
    )
    ls.set_hardware_adapter_gps_provider(lambda: adapter_fix)

    result = ls._query_hardware_adapter_gps()
    assert result is adapter_fix
    assert result.device_driver == "moravian"


def test_location_service_hardware_adapter_gps_provider_no_coords():
    """Partial fix (device detected, no coordinates) is returned for UI status."""
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    no_fix = GPSFix(fix_mode=0, satellites=3, timestamp=time.time(), device_path="camera", device_driver="moravian")
    ls.set_hardware_adapter_gps_provider(lambda: no_fix)

    result = ls._query_hardware_adapter_gps()
    assert result is no_fix
    assert result.latitude is None
    assert result.device_driver == "moravian"
    assert result.satellites == 3


def test_location_service_hardware_adapter_gps_provider_none():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    ls.set_hardware_adapter_gps_provider(lambda: None)
    assert ls._query_hardware_adapter_gps() is None


def test_location_service_get_best_gps_fix_prefers_gpsd():
    with patch.object(GPSMonitor, "is_available", return_value=True), patch.object(GPSMonitor, "start"):
        ls = LocationService()

    gpsd_fix = GPSFix(
        latitude=40.0,
        longitude=-74.0,
        altitude=100.0,
        fix_mode=3,
        satellites=8,
        timestamp=time.time(),
        device_path="/dev/ttyACM0",
        device_driver="u-blox",
    )
    adapter_fix = GPSFix(
        latitude=38.82,
        longitude=-104.87,
        altitude=1660.0,
        fix_mode=3,
        satellites=5,
        timestamp=time.time(),
        device_path="camera",
        device_driver="moravian",
    )
    ls.set_hardware_adapter_gps_provider(lambda: adapter_fix)

    with patch.object(GPSMonitor, "get_current_fix", return_value=gpsd_fix):
        result = ls.get_best_gps_fix(allow_blocking=False)

    assert result is gpsd_fix


def test_location_service_get_best_gps_fix_falls_back_to_hardware_adapter():
    with patch.object(GPSMonitor, "is_available", return_value=True), patch.object(GPSMonitor, "start"):
        ls = LocationService()

    adapter_fix = GPSFix(
        latitude=38.82,
        longitude=-104.87,
        altitude=1660.0,
        fix_mode=3,
        satellites=5,
        timestamp=time.time(),
        device_path="camera",
        device_driver="moravian",
    )
    ls.set_hardware_adapter_gps_provider(lambda: adapter_fix)

    # gpsd returns metadata only (no position)
    gpsd_meta = GPSFix(gpsd_version="3.25", device_path="/dev/ttyACM0")
    with patch.object(GPSMonitor, "get_current_fix", return_value=gpsd_meta):
        result = ls.get_best_gps_fix(allow_blocking=False)

    assert result is adapter_fix
    assert result.device_driver == "moravian"


def test_location_service_get_best_gps_fix_none():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    result = ls.get_best_gps_fix(allow_blocking=False)
    assert result is None


def test_location_service_get_current_location_from_hardware_adapter_gps():
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    adapter_fix = GPSFix(
        latitude=38.82,
        longitude=-104.87,
        altitude=1660.0,
        fix_mode=3,
        satellites=5,
        timestamp=time.time(),
        device_path="camera",
        device_driver="moravian",
    )
    ls.set_hardware_adapter_gps_provider(lambda: adapter_fix)

    loc = ls.get_current_location()
    assert loc is not None
    assert loc["source"] == "hardware_adapter_gps"
    assert loc["latitude"] == pytest.approx(38.82)
    assert loc["longitude"] == pytest.approx(-104.87)
    assert loc["altitude"] == pytest.approx(1660.0)


def test_location_service_get_current_location_skips_partial_adapter_fix():
    """Partial adapter fix (no coords) shouldn't be used for location resolution."""
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    partial = GPSFix(fix_mode=0, satellites=3, device_path="camera", device_driver="moravian")
    ls.set_hardware_adapter_gps_provider(lambda: partial)
    ls.set_ground_station({"latitude": 38.82, "longitude": -104.87, "altitude": 1899.0})

    loc = ls.get_current_location()
    assert loc is not None
    assert loc["source"] == "ground_station"


def test_location_service_on_gps_fix_falls_back_to_hardware_adapter():
    mock_api = MagicMock()
    mock_api.update_ground_station_location.return_value = True
    mock_settings = MagicMock()
    mock_settings.gps_location_updates_enabled = True
    mock_settings.gps_update_interval_minutes = 0

    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService(api_client=mock_api, settings=mock_settings)

    ls.set_ground_station({"id": "gs1", "latitude": 0.0, "longitude": 0.0, "altitude": 0.0})

    adapter_fix = GPSFix(
        latitude=38.82,
        longitude=-104.87,
        altitude=1660.0,
        fix_mode=3,
        satellites=5,
        timestamp=time.time(),
        device_path="camera",
        device_driver="moravian",
    )
    ls.set_hardware_adapter_gps_provider(lambda: adapter_fix)

    # Weak gpsd fix (no coordinates) triggers hardware adapter fallback
    weak_fix = GPSFix(fix_mode=0, satellites=0)
    ls.on_gps_fix_changed(weak_fix)

    mock_api.update_ground_station_location.assert_called_once_with("gs1", 38.82, -104.87, 1660.0)


def test_location_service_hardware_adapter_gps_cached():
    """Repeated calls within 30s return cached fix without re-querying the provider."""
    with patch.object(GPSMonitor, "is_available", return_value=False):
        ls = LocationService()

    call_count = 0

    def counting_provider() -> GPSFix:
        nonlocal call_count
        call_count += 1
        return GPSFix(
            latitude=38.82,
            longitude=-104.87,
            altitude=1660.0,
            fix_mode=3,
            satellites=5,
            timestamp=time.time(),
            device_path="camera",
            device_driver="moravian",
        )

    ls.set_hardware_adapter_gps_provider(counting_provider)

    # First call queries the provider
    fix1 = ls._query_hardware_adapter_gps()
    assert fix1 is not None
    assert call_count == 1

    # Second call within TTL returns cache — provider not called again
    fix2 = ls._query_hardware_adapter_gps()
    assert fix2 is fix1
    assert call_count == 1

    # Expire the cache and verify provider is called again
    ls._hardware_adapter_gps_cache_time = time.time() - 60
    fix3 = ls._query_hardware_adapter_gps()
    assert fix3 is not None
    assert call_count == 2
