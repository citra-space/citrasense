"""Unit tests for CitraApiClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from citrascope.api.citra_api_client import CitraApiClient, _build_filter_wavelength_lookup


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


# ---------------------------------------------------------------------------
# _build_filter_wavelength_lookup
# ---------------------------------------------------------------------------


def test_build_filter_wavelength_lookup_discrete():
    tel = {
        "spectralConfig": {
            "type": "discrete",
            "filters": [
                {"name": "r", "central_wavelength_nm": 623.0, "bandwidth_nm": 137.0},
                {"name": "g", "central_wavelength_nm": 477.0, "bandwidth_nm": 137.0},
            ],
        }
    }
    lookup = _build_filter_wavelength_lookup(tel)
    assert lookup["r"] == pytest.approx((623.0 - 137.0 / 2, 623.0 + 137.0 / 2))
    assert lookup["g"] == pytest.approx((477.0 - 137.0 / 2, 477.0 + 137.0 / 2))


def test_build_filter_wavelength_lookup_no_spectral_config():
    assert _build_filter_wavelength_lookup({}) == {}
    assert _build_filter_wavelength_lookup({"spectralConfig": None}) == {}


def test_build_filter_wavelength_lookup_non_discrete():
    tel = {"spectralConfig": {"type": "tunable", "min_wavelength_nm": 400, "max_wavelength_nm": 900}}
    assert _build_filter_wavelength_lookup(tel) == {}


def test_build_filter_wavelength_lookup_skips_incomplete_filters():
    tel = {
        "spectralConfig": {
            "type": "discrete",
            "filters": [
                {"name": "r", "central_wavelength_nm": 623.0, "bandwidth_nm": 137.0},
                {"name": "bad", "central_wavelength_nm": None, "bandwidth_nm": 100.0},
                {"name": None, "central_wavelength_nm": 500.0, "bandwidth_nm": 100.0},
            ],
        }
    }
    lookup = _build_filter_wavelength_lookup(tel)
    assert len(lookup) == 1
    assert "r" in lookup


# ---------------------------------------------------------------------------
# upload_optical_observations — per-filter wavelength
# ---------------------------------------------------------------------------


_OBS_BASE = {"norad_id": "25544", "timestamp": "2026-01-01T00:00:00Z", "ra": 180.0, "dec": 45.0, "mag": 2.5}
_LOC = {"latitude": 34.0, "longitude": -118.0, "altitude": 500.0}


def _tel_discrete(**overrides):
    """Build a telescope record with discrete filters (null static bounds)."""
    tel = {
        "id": "tel-1",
        "angularNoise": 0.1,
        "spectralMinWavelengthNm": None,
        "spectralMaxWavelengthNm": None,
        "spectralConfig": {
            "type": "discrete",
            "filters": [
                {"name": "r", "central_wavelength_nm": 623.0, "bandwidth_nm": 137.0},
                {"name": "g", "central_wavelength_nm": 477.0, "bandwidth_nm": 137.0},
            ],
        },
    }
    tel.update(overrides)
    return tel


def _capture_payload(client, mock_response, obs_list, tel, loc=_LOC, task_id="t1"):
    """Run upload and return the JSON payload that was POSTed."""
    mock_response.json.return_value = {"created": len(obs_list)}
    with patch.object(client.client, "request", return_value=mock_response) as mock_req:
        result = client.upload_optical_observations(obs_list, tel, loc, task_id=task_id)
    posted = mock_req.call_args
    return result, posted.kwargs["json"] if posted else None


def test_discrete_filter_sets_wavelength(client, mock_response):
    obs = [{**_OBS_BASE, "filter": "r"}]
    ok, payload = _capture_payload(client, mock_response, obs, _tel_discrete())
    assert ok is True
    assert payload[0]["minWavelength"] == pytest.approx(623.0 - 137.0 / 2)
    assert payload[0]["maxWavelength"] == pytest.approx(623.0 + 137.0 / 2)


def test_discrete_filter_unmatched_omits_wavelength(client, mock_response):
    obs = [{**_OBS_BASE, "filter": "z"}]
    ok, payload = _capture_payload(client, mock_response, obs, _tel_discrete())
    assert ok is True
    assert "minWavelength" not in payload[0]
    assert "maxWavelength" not in payload[0]


def test_discrete_filter_no_filter_key_omits_wavelength(client, mock_response):
    obs = [{**_OBS_BASE}]  # no "filter" key
    ok, payload = _capture_payload(client, mock_response, obs, _tel_discrete())
    assert ok is True
    assert "minWavelength" not in payload[0]
    assert "maxWavelength" not in payload[0]


def test_static_bounds_take_priority(client, mock_response):
    tel = {
        "id": "tel-1",
        "angularNoise": 0.1,
        "spectralMinWavelengthNm": 400.0,
        "spectralMaxWavelengthNm": 700.0,
        "spectralConfig": {
            "type": "discrete",
            "filters": [{"name": "r", "central_wavelength_nm": 623.0, "bandwidth_nm": 137.0}],
        },
    }
    obs = [{**_OBS_BASE, "filter": "r"}]
    ok, payload = _capture_payload(client, mock_response, obs, tel)
    assert ok is True
    assert payload[0]["minWavelength"] == 400.0
    assert payload[0]["maxWavelength"] == 700.0


def test_no_spectral_config_omits_wavelength(client, mock_response):
    tel = {"id": "tel-1", "angularNoise": 0.1, "spectralMinWavelengthNm": None, "spectralMaxWavelengthNm": None}
    obs = [{**_OBS_BASE, "filter": "r"}]
    ok, payload = _capture_payload(client, mock_response, obs, tel)
    assert ok is True
    assert "minWavelength" not in payload[0]
    assert "maxWavelength" not in payload[0]


def test_multiple_obs_different_filters(client, mock_response):
    obs = [
        {**_OBS_BASE, "norad_id": "25544", "filter": "r"},
        {**_OBS_BASE, "norad_id": "25545", "filter": "g"},
    ]
    ok, payload = _capture_payload(client, mock_response, obs, _tel_discrete())
    assert ok is True
    assert payload[0]["minWavelength"] == pytest.approx(623.0 - 137.0 / 2)
    assert payload[1]["minWavelength"] == pytest.approx(477.0 - 137.0 / 2)
