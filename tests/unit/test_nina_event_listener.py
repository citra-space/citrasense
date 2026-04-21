"""Unit tests for NinaEventListener."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from citrasense.hardware.nina.nina_event_listener import NinaEventListener, derive_ws_url

# ---------- derive_ws_url ----------


class TestDeriveWsUrl:
    def test_http_to_ws(self):
        assert derive_ws_url("http://nina:1888/v2/api") == "ws://nina:1888/v2/socket"

    def test_https_to_wss(self):
        assert derive_ws_url("https://nina:1888/v2/api") == "wss://nina:1888/v2/socket"

    def test_trailing_slash(self):
        assert derive_ws_url("http://nina:1888/v2/api/") == "ws://nina:1888/v2/socket"

    def test_localhost(self):
        assert derive_ws_url("http://localhost:1888/v2/api") == "ws://localhost:1888/v2/socket"

    def test_ip_address(self):
        assert derive_ws_url("http://10.211.55.3:1888/v2/api") == "ws://10.211.55.3:1888/v2/socket"

    def test_no_api_suffix(self):
        assert derive_ws_url("http://nina:1888/v2") == "ws://nina:1888/v2/socket"


# ---------- NinaEventListener ----------


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def listener(logger):
    """Create a listener without starting its background thread."""
    return NinaEventListener("ws://nina:1888/v2/socket", logger)


class TestEventDispatch:
    """Test that _dispatch correctly routes NINA WebSocket messages."""

    def test_sequence_finished(self, listener):
        listener.sequence_finished.clear()
        listener._dispatch(
            {
                "Response": {"Event": "SEQUENCE-FINISHED"},
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.sequence_finished.is_set()

    def test_sequence_entity_failed(self, listener):
        listener.sequence_failed.clear()
        listener._dispatch(
            {
                "Response": {
                    "Event": "SEQUENCE-ENTITY-FAILED",
                    "Entity": "Dew Heater",
                    "Error": "Camera not connected",
                },
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.sequence_failed.is_set()
        assert listener.last_sequence_error["Entity"] == "Dew Heater"
        assert listener.last_sequence_error["Error"] == "Camera not connected"

    def test_autofocus_finished(self, listener):
        listener.autofocus_finished.clear()
        listener._dispatch(
            {
                "Response": {"Event": "AUTOFOCUS-FINISHED"},
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.autofocus_finished.is_set()

    def test_autofocus_error(self, listener):
        listener.autofocus_error.clear()
        listener._dispatch(
            {
                "Response": {"Event": "ERROR-AF"},
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.autofocus_error.is_set()

    def test_autofocus_point_added_with_callback(self, listener):
        callback = MagicMock()
        listener.on_af_point = callback
        listener._dispatch(
            {
                "Response": {
                    "Event": "AUTOFOCUS-POINT-ADDED",
                    "ImageStatistics": {"Position": 8500, "HFR": 2.31},
                },
                "Success": True,
                "Type": "Socket",
            }
        )
        callback.assert_called_once_with(8500, 2.31)

    def test_autofocus_point_added_no_callback(self, listener):
        listener.on_af_point = None
        listener._dispatch(
            {
                "Response": {
                    "Event": "AUTOFOCUS-POINT-ADDED",
                    "ImageStatistics": {"Position": 8500, "HFR": 2.31},
                },
                "Success": True,
                "Type": "Socket",
            }
        )

    def test_autofocus_point_callback_error_is_swallowed(self, listener):
        callback = MagicMock(side_effect=ValueError("boom"))
        listener.on_af_point = callback
        listener._dispatch(
            {
                "Response": {
                    "Event": "AUTOFOCUS-POINT-ADDED",
                    "ImageStatistics": {"Position": 8500, "HFR": 2.31},
                },
                "Success": True,
                "Type": "Socket",
            }
        )
        callback.assert_called_once()

    def test_filterwheel_changed(self, listener):
        listener.filter_changed.clear()
        listener._dispatch(
            {
                "Response": {
                    "Event": "FILTERWHEEL-CHANGED",
                    "Previous": {"Name": "Luminance", "Id": 0},
                    "New": {"Name": "Red", "Id": 1},
                },
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.filter_changed.is_set()
        assert listener.last_filter_change["New"]["Name"] == "Red"

    def test_image_save(self, listener):
        listener.image_saved.clear()
        callback = MagicMock()
        listener.on_image_save = callback
        listener._dispatch(
            {
                "Response": {
                    "Event": "IMAGE-SAVE",
                    "ImageStatistics": {
                        "Filename": "/path/to/image.fits",
                        "Filter": "Luminance",
                        "HFR": 1.8,
                        "Stars": 42,
                    },
                },
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.image_saved.is_set()
        assert listener.last_image_save["Filename"] == "/path/to/image.fits"
        callback.assert_called_once()

    def test_unknown_event_ignored(self, listener):
        listener.sequence_finished.clear()
        listener._dispatch(
            {
                "Response": {"Event": "DOME-PARKED"},
                "Success": True,
                "Type": "Socket",
            }
        )
        assert not listener.sequence_finished.is_set()

    def test_non_dict_response_ignored(self, listener):
        listener._dispatch({"Response": "just a string", "Success": True})

    def test_no_event_key_ignored(self, listener):
        listener._dispatch({"Response": {"SomeOther": "data"}, "Success": True})

    def test_empty_message_ignored(self, listener):
        listener._dispatch({})


class TestEventWaitPatterns:
    """Test the clear-then-wait pattern used by the adapter."""

    def test_clear_wait_set_pattern(self, listener):
        listener.filter_changed.clear()
        assert not listener.filter_changed.is_set()

        def set_after_delay():
            time.sleep(0.05)
            listener._dispatch(
                {
                    "Response": {
                        "Event": "FILTERWHEEL-CHANGED",
                        "Previous": {"Name": "L", "Id": 0},
                        "New": {"Name": "R", "Id": 1},
                    },
                    "Success": True,
                    "Type": "Socket",
                }
            )

        t = threading.Thread(target=set_after_delay)
        t.start()
        result = listener.filter_changed.wait(timeout=2.0)
        t.join()
        assert result is True

    def test_wait_timeout(self, listener):
        listener.autofocus_finished.clear()
        result = listener.autofocus_finished.wait(timeout=0.05)
        assert result is False

    def test_multiple_events_before_wait(self, listener):
        """Events set before wait() should be immediately available."""
        listener.sequence_finished.clear()
        listener._dispatch(
            {
                "Response": {"Event": "SEQUENCE-FINISHED"},
                "Success": True,
                "Type": "Socket",
            }
        )
        assert listener.sequence_finished.wait(timeout=0.01) is True


class TestThreadSafety:
    """Verify that concurrent access to last-event data is safe."""

    def test_concurrent_filter_changes(self, listener):
        errors = []

        def dispatch_many():
            for i in range(100):
                listener._dispatch(
                    {
                        "Response": {
                            "Event": "FILTERWHEEL-CHANGED",
                            "Previous": {"Name": f"F{i}", "Id": i},
                            "New": {"Name": f"F{i+1}", "Id": i + 1},
                        },
                        "Success": True,
                        "Type": "Socket",
                    }
                )

        def read_many():
            for _ in range(100):
                data = listener.last_filter_change
                if data is not None and "New" not in data:
                    errors.append("Corrupt data read")

        t1 = threading.Thread(target=dispatch_many)
        t2 = threading.Thread(target=read_many)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []


class TestLifecycle:
    """Test start/stop without a real WebSocket server."""

    @patch("citrasense.hardware.nina.nina_event_listener.ws_connect")
    def test_start_and_stop(self, mock_ws_connect, listener):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = TimeoutError
        mock_ws.__enter__ = MagicMock(return_value=mock_ws)
        mock_ws.__exit__ = MagicMock(return_value=False)
        mock_ws_connect.return_value = mock_ws

        listener.start()
        assert listener.connected
        time.sleep(0.15)

        listener.stop()
        assert not listener._running

    def test_stop_without_start(self, listener):
        listener.stop()

    def test_double_start(self, listener):
        """Starting twice should not create two threads."""
        with patch("citrasense.hardware.nina.nina_event_listener.ws_connect") as mock:
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = TimeoutError
            mock_ws.__enter__ = MagicMock(return_value=mock_ws)
            mock_ws.__exit__ = MagicMock(return_value=False)
            mock.return_value = mock_ws

            listener.start()
            first_thread = listener._thread
            listener.start()
            assert listener._thread is first_thread
            listener.stop()
