"""Tests for ZwoAmMount device — methods that require a mocked transport."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from citrasense.hardware.devices.mount.zwo_am_mount import ZwoAmMount
from citrasense.hardware.devices.mount.zwo_am_transport import ZwoAmTransport


@pytest.fixture
def mount():
    """Create a ZwoAmMount with a mocked transport (no hardware needed)."""
    m = ZwoAmMount(logger=MagicMock(), connection_type="serial", port="/dev/null")
    m._transport = MagicMock()
    return m


class TestSyncDatetime:
    def test_sends_tz_time_then_date_last(self, mount: ZwoAmMount):
        mount._transport.send_command_bool_with_retry.return_value = True

        frozen = datetime(2026, 2, 24, 19, 45, 30, tzinfo=timezone.utc)
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = frozen
        with patch("citrasense.hardware.devices.mount.zwo_am_mount.datetime", mock_dt):
            result = mount.sync_datetime()

        assert result is True

        calls = mount._transport.send_command_bool_with_retry.call_args_list
        assert len(calls) == 3

        assert calls[0].args[0] == ":SG+00#"
        assert calls[1].args[0] == ":SL19:45:30#"
        assert calls[2].args[0] == ":SC02/24/26#"

        # :SC goes last; verify buffer is drained for subsequent callers
        mount._transport._clear_input.assert_called_once()

    def test_returns_false_on_partial_failure(self, mount: ZwoAmMount):
        mount._transport.send_command_bool_with_retry.side_effect = [True, False, True]

        result = mount.sync_datetime()

        assert result is False

    def test_returns_false_when_all_fail(self, mount: ZwoAmMount):
        mount._transport.send_command_bool_with_retry.return_value = False

        result = mount.sync_datetime()

        assert result is False


class TestCustomTrackingRates:
    def test_set_custom_tracking_rates_sends_commands(self, mount: ZwoAmMount):
        mount._transport.send_command_with_retry.return_value = "NG#"  # tracking=True
        result = mount.set_custom_tracking_rates(50.0, -10.0)

        assert result is True
        no_resp_calls = [c.args[0] for c in mount._transport.send_command_no_response.call_args_list]
        assert ":RA+050.000000#" in no_resp_calls
        assert ":RE-010.000000#" in no_resp_calls

    def test_set_custom_tracking_rates_starts_tracking_if_not_active(self, mount: ZwoAmMount):
        mount._transport.send_command_with_retry.return_value = "nN#"  # tracking=False
        mount.set_custom_tracking_rates(1.0, 2.0)

        no_resp_calls = [c.args[0] for c in mount._transport.send_command_no_response.call_args_list]
        assert ":TQ#" in no_resp_calls  # sidereal rate set
        assert ":Te#" in no_resp_calls  # tracking enabled

    def test_reset_tracking_rates_zeros_offsets(self, mount: ZwoAmMount):
        mount.reset_tracking_rates()

        no_resp_calls = [c.args[0] for c in mount._transport.send_command_no_response.call_args_list]
        assert ":RA+000.000000#" in no_resp_calls
        assert ":RE+000.000000#" in no_resp_calls

    def test_get_mount_info_supports_custom_tracking(self, mount: ZwoAmMount):
        mount._transport.send_command_with_retry.return_value = "NG#"
        info = mount.get_mount_info()
        assert info["supports_custom_tracking"] is False


class _FakeTransport(ZwoAmTransport):
    """In-memory transport that records command order for concurrency testing."""

    def __init__(self) -> None:
        super().__init__(timeout_s=1.0, retry_count=1)
        self.log: list[str] = []
        self._log_lock = threading.Lock()

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def is_open(self) -> bool:
        return True

    def _write(self, data: bytes) -> None:
        cmd = data.decode("ascii")
        with self._log_lock:
            self.log.append(f"TX:{cmd}")
        time.sleep(0.05)

    def _read_until_hash(self) -> str:
        time.sleep(0.05)
        with self._log_lock:
            self.log.append("RX")
        return "ok#"

    def _clear_input(self) -> None:
        pass

    def _try_read_one(self) -> str | None:
        return None


class TestTransportThreadSafety:
    def test_concurrent_commands_do_not_interleave(self):
        transport = _FakeTransport()
        errors: list[Exception] = []

        def send_a():
            try:
                transport.send_command(":AAA#")
            except Exception as e:
                errors.append(e)

        def send_b():
            try:
                transport.send_command(":BBB#")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=send_a)
        t2 = threading.Thread(target=send_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors

        # With the lock, we expect TX/RX pairs to be atomic (never interleaved).
        # Either [TX:A, RX, TX:B, RX] or [TX:B, RX, TX:A, RX].
        assert len(transport.log) == 4
        assert transport.log[0].startswith("TX:")
        assert transport.log[1] == "RX"
        assert transport.log[2].startswith("TX:")
        assert transport.log[3] == "RX"
        # The two TX commands should be different (not mixed)
        assert transport.log[0] != transport.log[2]
