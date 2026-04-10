"""Tests for create_batch_collection_requests on API clients."""

from __future__ import annotations

import logging
from unittest.mock import patch

from citrascope.api.dummy_api_client import DummyApiClient


def test_dummy_client_batch_returns_ok():
    client = DummyApiClient(logger=logging.getLogger("test"))
    result = client.create_batch_collection_requests(
        window_start="2025-01-01T00:00:00Z",
        window_stop="2025-01-01T06:00:00Z",
        ground_station_id="gs-001",
        sensor_id="sensor-001",
    )
    assert result is not None
    assert result["status"] == "ok"


def test_dummy_client_batch_with_filters():
    client = DummyApiClient(logger=logging.getLogger("test"))
    result = client.create_batch_collection_requests(
        window_start="2025-01-01T00:00:00Z",
        window_stop="2025-01-01T06:00:00Z",
        ground_station_id="gs-001",
        sensor_id="sensor-001",
        satellite_group_ids=["group-1"],
        exclude_types=["Debris"],
        include_orbit_regimes=["LEO"],
    )
    assert result is not None


def test_citra_client_batch_constructs_payload():
    """Verify the real client constructs the correct POST payload."""
    from citrascope.api.citra_api_client import CitraApiClient

    client = CitraApiClient(host="test.api.citra.space", token="fake", logger=logging.getLogger("test"))

    with patch.object(client, "_request", return_value={"status": "ok"}) as mock_req:
        client.create_batch_collection_requests(
            window_start="2025-01-01T00:00:00Z",
            window_stop="2025-01-01T06:00:00Z",
            ground_station_id="gs-001",
            sensor_id="sensor-001",
            discover_visible=True,
            exclude_types=["Debris"],
            include_orbit_regimes=["LEO"],
        )

    mock_req.assert_called_once()
    call_args = mock_req.call_args
    assert call_args[0] == ("POST", "/collection-requests/batch")
    body = call_args[1]["json"]
    assert body["windowStart"] == "2025-01-01T00:00:00Z"
    assert body["windowStop"] == "2025-01-01T06:00:00Z"
    assert body["params"]["ground_station_ids"] == ["gs-001"]
    assert body["params"]["sensor_selections"] == {"gs-001": "sensor-001"}
    assert body["discoverVisible"] is True
    assert body["excludeTypes"] == ["Debris"]
    assert body["includeOrbitRegimes"] == ["LEO"]
    assert "satelliteGroupIds" not in body  # Empty list not sent


def test_citra_client_batch_characterization_type():
    """Verify request_type='Characterization' lands in the payload as type."""
    from citrascope.api.citra_api_client import CitraApiClient

    client = CitraApiClient(host="test.api.citra.space", token="fake", logger=logging.getLogger("test"))

    with patch.object(client, "_request", return_value={"status": "ok"}) as mock_req:
        client.create_batch_collection_requests(
            window_start="2025-01-01T00:00:00Z",
            window_stop="2025-01-01T06:00:00Z",
            ground_station_id="gs-001",
            sensor_id="sensor-001",
            request_type="Characterization",
        )

    body = mock_req.call_args[1]["json"]
    assert body["type"] == "Characterization"
