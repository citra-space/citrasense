"""Tests for TimeHealthCheck — maps TimeMonitor health to SafetyAction."""

from unittest.mock import MagicMock

from citrascope.safety.safety_monitor import SafetyAction
from citrascope.safety.time_health_check import TimeHealthCheck
from citrascope.time.time_health import TimeHealth, TimeStatus


def _make_monitor(health: TimeHealth | None = None):
    monitor = MagicMock()
    monitor.get_current_health.return_value = health
    return monitor


class TestTimeHealthCheck:
    def test_none_health_returns_safe(self):
        check = TimeHealthCheck(MagicMock(), _make_monitor(None))
        assert check.check() == SafetyAction.SAFE

    def test_ok_status_returns_safe(self):
        health = TimeHealth(offset_ms=5.0, status=TimeStatus.OK, source="ntp")
        check = TimeHealthCheck(MagicMock(), _make_monitor(health))
        assert check.check() == SafetyAction.SAFE

    def test_critical_status_returns_queue_stop(self):
        health = TimeHealth(offset_ms=500.0, status=TimeStatus.CRITICAL, source="ntp")
        check = TimeHealthCheck(MagicMock(), _make_monitor(health))
        assert check.check() == SafetyAction.QUEUE_STOP

    def test_get_status_with_health(self):
        health = TimeHealth(offset_ms=12.3, status=TimeStatus.OK, source="gps")
        check = TimeHealthCheck(MagicMock(), _make_monitor(health))
        status = check.get_status()
        assert status["name"] == "time_health"
        assert status["offset_ms"] == 12.3
        assert status["source"] == "gps"
        assert status["time_status"] == "ok"

    def test_get_status_without_health(self):
        check = TimeHealthCheck(MagicMock(), _make_monitor(None))
        status = check.get_status()
        assert status["name"] == "time_health"
        assert "offset_ms" not in status
        assert "source" not in status
