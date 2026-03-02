"""Unit tests for time health, time monitor, and time sources."""

from unittest.mock import MagicMock, patch

import pytest

from citrascope.time.time_health import TimeHealth, TimeStatus

# ---------------------------------------------------------------------------
# TimeStatus
# ---------------------------------------------------------------------------


def test_time_status_values():
    assert TimeStatus.OK == "ok"
    assert TimeStatus.CRITICAL == "critical"
    assert TimeStatus.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# TimeHealth
# ---------------------------------------------------------------------------


def test_calculate_status_ok():
    assert TimeHealth.calculate_status(10.0, 500.0) == TimeStatus.OK


def test_calculate_status_critical():
    assert TimeHealth.calculate_status(600.0, 500.0) == TimeStatus.CRITICAL


def test_calculate_status_unknown():
    assert TimeHealth.calculate_status(None, 500.0) == TimeStatus.UNKNOWN


def test_calculate_status_negative_offset():
    assert TimeHealth.calculate_status(-100.0, 500.0) == TimeStatus.OK


def test_calculate_status_exactly_at_threshold():
    assert TimeHealth.calculate_status(500.0, 500.0) == TimeStatus.CRITICAL


def test_from_offset():
    h = TimeHealth.from_offset(offset_ms=42.0, source="ntp", pause_threshold=500.0)
    assert h.offset_ms == 42.0
    assert h.source == "ntp"
    assert h.status == TimeStatus.OK


def test_from_offset_critical():
    h = TimeHealth.from_offset(offset_ms=1000.0, source="ntp", pause_threshold=500.0)
    assert h.status == TimeStatus.CRITICAL
    assert h.should_pause_observations() is True


def test_from_offset_with_metadata():
    h = TimeHealth.from_offset(
        offset_ms=5.0,
        source="gps",
        pause_threshold=500.0,
        message="GPS lock",
        metadata={"satellites": 8},
    )
    assert h.metadata["satellites"] == 8
    assert h.message == "GPS lock"


def test_should_pause_observations_ok():
    h = TimeHealth(offset_ms=10.0, status=TimeStatus.OK, source="ntp")
    assert h.should_pause_observations() is False


def test_should_pause_observations_unknown():
    h = TimeHealth(offset_ms=None, status=TimeStatus.UNKNOWN, source="unknown")
    assert h.should_pause_observations() is False


def test_to_dict():
    h = TimeHealth(offset_ms=42.5, status=TimeStatus.OK, source="ntp", message="fine")
    d = h.to_dict()
    assert d["offset_ms"] == 42.5
    assert d["status"] == "ok"
    assert d["source"] == "ntp"
    assert d["message"] == "fine"


# ---------------------------------------------------------------------------
# NTPTimeSource
# ---------------------------------------------------------------------------


def test_ntp_source_name():
    from citrascope.time.time_sources import NTPTimeSource

    src = NTPTimeSource()
    assert src.get_source_name() == "ntp"
    assert src.get_metadata() is None


def test_ntp_get_offset_success():
    from citrascope.time.time_sources import NTPTimeSource

    src = NTPTimeSource()
    mock_resp = MagicMock()
    mock_resp.offset = 0.05
    with patch.object(src.client, "request", return_value=mock_resp):
        offset = src.get_offset_ms()
    assert offset == pytest.approx(50.0)


def test_ntp_get_offset_failure():
    from citrascope.time.time_sources import NTPTimeSource

    src = NTPTimeSource()
    with patch.object(src.client, "request", side_effect=Exception("timeout")):
        assert src.get_offset_ms() is None


# ---------------------------------------------------------------------------
# ChronyTimeSource
# ---------------------------------------------------------------------------


def test_chrony_source_name():
    from citrascope.time.time_sources import ChronyTimeSource

    src = ChronyTimeSource()
    assert src.get_source_name() == "chrony"


def test_chrony_is_available_false():
    from citrascope.time.time_sources import ChronyTimeSource

    src = ChronyTimeSource()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert src.is_available() is False


def test_chrony_get_offset_success():
    from citrascope.time.time_sources import ChronyTimeSource

    src = ChronyTimeSource()
    mock_result = MagicMock()
    mock_result.stdout = "ref,stratum,ref_time,sys_time,0.000025,freq,resid_freq,skew,root_delay,root_disp"
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        offset = src.get_offset_ms()
    assert offset == pytest.approx(0.025)


def test_chrony_get_offset_failure():
    from citrascope.time.time_sources import ChronyTimeSource

    src = ChronyTimeSource()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert src.get_offset_ms() is None


# ---------------------------------------------------------------------------
# TimeMonitor
# ---------------------------------------------------------------------------


def test_time_monitor_get_current_health_initially_none():
    from citrascope.time.time_sources import NTPTimeSource

    with patch.object(NTPTimeSource, "__init__", lambda self, **kw: None):
        with patch("citrascope.time.time_monitor.ChronyTimeSource") as MockChrony:
            MockChrony.return_value.is_available.return_value = False
            with patch("citrascope.time.time_monitor.NTPTimeSource") as MockNTP:
                MockNTP.return_value = MagicMock()
                from citrascope.time.time_monitor import TimeMonitor

                tm = TimeMonitor(check_interval_minutes=60)
    assert tm.get_current_health() is None


def test_time_monitor_check_time_sync():
    from citrascope.time.time_monitor import TimeMonitor

    mock_source = MagicMock()
    mock_source.get_offset_ms.return_value = 25.0
    mock_source.get_source_name.return_value = "ntp"

    with patch.object(TimeMonitor, "_detect_best_source", return_value=mock_source):
        tm = TimeMonitor(check_interval_minutes=60, pause_threshold_ms=500.0)
        tm._check_time_sync()

    health = tm.get_current_health()
    assert health is not None
    assert health.offset_ms == 25.0
    assert health.status == TimeStatus.OK


def test_time_monitor_critical_sets_status():
    from citrascope.time.time_monitor import TimeMonitor

    mock_source = MagicMock()
    mock_source.get_offset_ms.return_value = 9999.0
    mock_source.get_source_name.return_value = "ntp"

    with patch.object(TimeMonitor, "_detect_best_source", return_value=mock_source):
        tm = TimeMonitor(check_interval_minutes=60, pause_threshold_ms=500.0)
        tm._check_time_sync()

    health = tm.get_current_health()
    assert health is not None
    assert health.status == TimeStatus.CRITICAL


def test_time_monitor_check_failure_sets_unknown():
    from citrascope.time.time_monitor import TimeMonitor

    mock_source = MagicMock()
    mock_source.get_offset_ms.side_effect = RuntimeError("boom")

    with patch.object(TimeMonitor, "_detect_best_source", return_value=mock_source):
        tm = TimeMonitor(check_interval_minutes=60)
        tm._check_time_sync()

    health = tm.get_current_health()
    assert health.status == TimeStatus.UNKNOWN
