"""Tests for DiskSpaceCheck."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from citrascope.safety.disk_space_check import CRITICAL_BYTES, WARN_BYTES, DiskSpaceCheck
from citrascope.safety.safety_monitor import SafetyAction


def _usage(free: int):
    """Create a mock disk_usage result."""
    return type("Usage", (), {"total": 500_000_000_000, "used": 500_000_000_000 - free, "free": free})()


class TestDiskSpaceCheck:
    def test_plenty_of_space_is_safe(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(10_000_000_000)):
            assert check.check() == SafetyAction.SAFE

    def test_low_space_warns(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(WARN_BYTES - 1)):
            assert check.check() == SafetyAction.WARN

    def test_critical_space_stops(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(CRITICAL_BYTES - 1)):
            assert check.check() == SafetyAction.QUEUE_STOP

    def test_proposed_capture_blocked_when_critical(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(CRITICAL_BYTES - 1)):
            check.check()
        assert check.check_proposed_action("capture") is False

    def test_proposed_capture_allowed_when_ok(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(10_000_000_000)):
            check.check()
        assert check.check_proposed_action("capture") is True

    def test_proposed_slew_always_allowed(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(0)):
            check.check()
        assert check.check_proposed_action("slew") is True

    def test_disk_read_failure_warns(self):
        """If disk_usage() raises, check returns WARN (fail-closed)."""
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", side_effect=OSError("permission denied")):
            assert check.check() == SafetyAction.WARN

    def test_capture_blocked_when_disk_unknown(self):
        """Captures are blocked when disk state is unknown (fail-closed)."""
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        assert check._free_bytes is None
        assert check.check_proposed_action("capture") is False

    def test_get_status(self):
        check = DiskSpaceCheck(MagicMock(), Path("/tmp"))
        with patch("shutil.disk_usage", return_value=_usage(5_000_000_000)):
            check.check()
        status = check.get_status()
        assert status["name"] == "disk_space"
        assert status["free_mb"] == 5000
