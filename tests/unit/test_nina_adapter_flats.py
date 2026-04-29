"""Unit tests for NINA adapter's trained-flat REST wrappers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from citrasense.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter

API = "http://nina:1888/v2/api"


def _ok(response=None) -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"Success": True, "Response": response or {}}
    m.raise_for_status.return_value = None
    return m


def _fail(error: str = "nope") -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"Success": False, "Error": error}
    m.raise_for_status.return_value = None
    return m


@pytest.fixture
def adapter() -> NinaAdvancedHttpAdapter:
    return NinaAdvancedHttpAdapter(
        logger=MagicMock(),
        images_dir=Path("/tmp"),
        nina_api_path=API,
    )


def test_supports_flat_automation_true_when_flat_device_connected(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.side_effect = [
            _ok({"Connected": True}),  # camera info
            _ok({"Connected": True, "Name": "Flatbox"}),  # flatdevice info
        ]
        assert adapter.supports_flat_automation() is True


def test_supports_flat_automation_false_without_flat_device(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.side_effect = [
            _ok({"Connected": True}),  # camera info
            _ok({"Connected": False}),  # flatdevice info — no panel
        ]
        assert adapter.supports_flat_automation() is False


def test_supports_flat_automation_false_without_camera(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.side_effect = [
            _ok({"Connected": False}),  # camera info
        ]
        assert adapter.supports_flat_automation() is False


def test_run_trained_flat_sends_correct_params(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _ok({"State": "Running"})
        adapter.run_trained_flat(filter_id=2, count=10, gain=100, binning=2)
        assert mock_get.called
        url = mock_get.call_args[0][0]
        params = mock_get.call_args.kwargs["params"]
        assert url == f"{API}/flats/trained-flat"
        assert params["count"] == 10
        assert params["filterId"] == 2
        assert params["gain"] == 100
        assert params["binning"] == "2x2"


def test_run_trained_flat_raises_on_nina_failure(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _fail("trained profile missing")
        with pytest.raises(RuntimeError, match="trained profile missing"):
            adapter.run_trained_flat(filter_id=0, count=5)


def test_run_trained_flat_raises_on_http_error(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("boom")
        with pytest.raises(RuntimeError, match="trained-flat request failed"):
            adapter.run_trained_flat(filter_id=0, count=5)


def test_poll_flat_status_returns_response_dict(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _ok({"State": "Finished", "TotalImageCount": 10})
        status = adapter.poll_flat_status()
        assert status["State"] == "Finished"
        assert status["TotalImageCount"] == 10


def test_poll_flat_status_returns_empty_on_error(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("boom")
        assert adapter.poll_flat_status() == {}


def test_stop_flats_returns_true_on_success(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _ok()
        assert adapter.stop_flats() is True


def test_stop_flats_returns_false_on_failure(adapter):
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _fail("nothing running")
        assert adapter.stop_flats() is False


def test_list_recent_flat_images_filters_by_type_and_date(adapter):
    rows = [
        {"ImageType": "LIGHT", "Date": "2026-04-29T18:00:00", "Filename": "light.fits"},
        {"ImageType": "FLAT", "Date": "2026-04-29T17:59:00", "Filename": "old_flat.fits"},
        {"ImageType": "FLAT", "Date": "2026-04-29T18:05:00", "Filename": "new_flat_a.fits"},
        {"ImageType": "FLAT", "Date": "2026-04-29T18:06:00", "Filename": "new_flat_b.fits"},
    ]
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _ok(rows)
        result = adapter.list_recent_flat_images(since_iso="2026-04-29T18:00:00")
        assert len(result) == 2
        names = {r["Filename"] for r in result}
        assert names == {"new_flat_a.fits", "new_flat_b.fits"}
        # Indexes preserved from the original ordering (LIGHT at 0, old FLAT at 1)
        indices = {r["_index"] for r in result}
        assert indices == {2, 3}


def test_list_recent_flat_images_without_since_returns_all_flats(adapter):
    rows = [
        {"ImageType": "FLAT", "Date": "2020-01-01T00:00:00", "Filename": "a.fits"},
        {"ImageType": "LIGHT", "Date": "2025-01-01T00:00:00", "Filename": "b.fits"},
    ]
    with patch("citrasense.hardware.nina.nina_adapter.requests.get") as mock_get:
        mock_get.return_value = _ok(rows)
        result = adapter.list_recent_flat_images()
        assert len(result) == 1
        assert result[0]["Filename"] == "a.fits"
