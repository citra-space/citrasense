"""Unit tests for CitraApiClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from citrascope.api.citra_api_client import CitraApiClient


@pytest.fixture
def client():
    return CitraApiClient(host="api.test.com", token="tok_123", use_ssl=True, logger=MagicMock())


@pytest.fixture
def mock_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '{"ok": true}'
    resp.json.return_value = {"ok": True}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Init and context manager
# ---------------------------------------------------------------------------


def test_init_builds_base_url():
    c = CitraApiClient(host="api.citra.space", token="t", use_ssl=True)
    assert c.base_url == "https://api.citra.space"


def test_init_http():
    c = CitraApiClient(host="localhost:8000", token="t", use_ssl=False)
    assert c.base_url == "http://localhost:8000"


def test_context_manager():
    with CitraApiClient(host="api.test.com", token="t") as c:
        assert c is not None


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------


def test_request_success(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        result = client._request("GET", "/test")
    assert result == {"ok": True}


def test_request_http_error_json(client):
    error_response = MagicMock()
    error_response.status_code = 400
    error_response.text = '{"error": "bad"}'
    error_response.headers = {"content-type": "application/json"}
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad", request=MagicMock(), response=error_response
    )
    with patch.object(client.client, "request", return_value=error_response):
        result = client._request("GET", "/fail")
    assert result is None


def test_request_http_error_html(client):
    error_response = MagicMock()
    error_response.status_code = 503
    error_response.text = "<html>Cloudflare</html>"
    error_response.headers = {"content-type": "text/html"}
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad", request=MagicMock(), response=error_response
    )
    with patch.object(client.client, "request", return_value=error_response):
        result = client._request("GET", "/cloudflare")
    assert result is None


def test_request_network_error(client):
    with patch.object(client.client, "request", side_effect=Exception("timeout")):
        result = client._request("GET", "/dead")
    assert result is None


# ---------------------------------------------------------------------------
# API methods
# ---------------------------------------------------------------------------


def test_does_api_server_accept_key(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        assert client.does_api_server_accept_key() is True


def test_does_api_server_reject_key(client):
    err_resp = MagicMock()
    err_resp.status_code = 401
    err_resp.text = "unauthorized"
    err_resp.headers = {"content-type": "text/plain"}
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError("", request=MagicMock(), response=err_resp)
    with patch.object(client.client, "request", return_value=err_resp):
        assert client.does_api_server_accept_key() is False


def test_get_telescope(client, mock_response):
    mock_response.json.return_value = {"id": "tel-1", "name": "CDK14"}
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.get_telescope("tel-1")
    assert result["name"] == "CDK14"


def test_get_satellite(client, mock_response):
    mock_response.json.return_value = {"id": "sat-1"}
    with patch.object(client.client, "request", return_value=mock_response):
        assert client.get_satellite("sat-1")["id"] == "sat-1"


def test_get_telescope_tasks(client, mock_response):
    mock_response.json.return_value = [{"id": "t1"}]
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.get_telescope_tasks("tel-1")
    assert len(result) == 1


def test_get_ground_station(client, mock_response):
    mock_response.json.return_value = {"id": "gs-1"}
    with patch.object(client.client, "request", return_value=mock_response):
        assert client.get_ground_station("gs-1")["id"] == "gs-1"


def test_put_telescope_status(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.put_telescope_status({"status": "online"})
    assert result is not None


def test_put_telescope_status_error(client):
    with patch.object(client.client, "request", side_effect=Exception("fail")):
        assert client.put_telescope_status({}) is None


def test_mark_task_complete(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.mark_task_complete("t1")
    assert result is not None


def test_mark_task_complete_error(client):
    with patch.object(client.client, "request", side_effect=Exception("fail")):
        assert client.mark_task_complete("t1") is None


def test_mark_task_failed(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.mark_task_failed("t1")
    assert result is not None


def test_mark_task_failed_error(client):
    with patch.object(client.client, "request", side_effect=Exception("fail")):
        assert client.mark_task_failed("t1") is None


def test_expand_filters(client, mock_response):
    mock_response.json.return_value = {"filters": [{"name": "Red"}]}
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.expand_filters(["Red"])
    assert "filters" in result


def test_expand_filters_error(client):
    with patch.object(client.client, "request", side_effect=Exception("fail")):
        assert client.expand_filters(["Red"]) is None


def test_update_telescope_spectral_config(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.update_telescope_spectral_config("tel-1", {"discreteFilters": []})
    assert result is not None


def test_update_telescope_spectral_config_error(client):
    with patch.object(client.client, "request", side_effect=Exception("fail")):
        assert client.update_telescope_spectral_config("tel-1", {}) is None


def test_update_ground_station_location(client, mock_response):
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.update_ground_station_location("gs-1", 40.0, -74.0, 100.0)
    assert result is not None


def test_update_ground_station_location_error(client):
    with patch.object(client.client, "request", side_effect=Exception("fail")):
        assert client.update_ground_station_location("gs-1", 40.0, -74.0, 100.0) is None


def test_get_elsets_latest(client, mock_response):
    mock_response.json.return_value = [{"satelliteId": "25544"}]
    with patch.object(client.client, "request", return_value=mock_response):
        result = client.get_elsets_latest(days=7)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# upload_optical_observations
# ---------------------------------------------------------------------------


def test_upload_optical_observations_success(client, mock_response):
    mock_response.json.return_value = {"created": 1}
    obs = [{"norad_id": "25544", "timestamp": "2026-01-01T00:00:00Z", "ra": 180.0, "dec": 45.0, "mag": 2.5}]
    tel = {"id": "tel-1", "angularNoise": 0.1, "spectralMinWavelengthNm": 400, "spectralMaxWavelengthNm": 700}
    loc = {"latitude": 34.0, "longitude": -118.0, "altitude": 500.0}
    with patch.object(client.client, "request", return_value=mock_response):
        assert client.upload_optical_observations(obs, tel, loc, task_id="t1") is True


def test_upload_optical_observations_empty(client):
    assert client.upload_optical_observations([], {"id": "t"}, {"latitude": 0, "longitude": 0}) is False


def test_upload_optical_observations_api_failure(client):
    err = MagicMock()
    err.status_code = 500
    err.text = "error"
    err.headers = {"content-type": "application/json"}
    err.raise_for_status.side_effect = httpx.HTTPStatusError("", request=MagicMock(), response=err)
    obs = [{"norad_id": "25544", "timestamp": "2026-01-01T00:00:00Z", "ra": 0, "dec": 0}]
    with patch.object(client.client, "request", return_value=err):
        assert client.upload_optical_observations(obs, {"id": "t"}, {"latitude": 0, "longitude": 0}) is False
