"""Unit tests for ElsetCache."""

import json
from unittest.mock import MagicMock

from citrascope.elset_cache import ElsetCache, _normalize_api_response

# ---------------------------------------------------------------------------
# _normalize_api_response
# ---------------------------------------------------------------------------


def test_normalize_basic():
    raw = [
        {
            "satelliteId": "25544",
            "satelliteName": "ISS",
            "tle": [
                "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
                "2 25544  51.6416 208.5340 0001234 123.4567 236.5433 15.50000000999999",
            ],
        }
    ]
    result = _normalize_api_response(raw)
    assert len(result) == 1
    assert result[0]["satellite_id"] == "25544"
    assert result[0]["name"] == "ISS"
    assert len(result[0]["tle"]) == 2


def test_normalize_missing_fields():
    raw = [
        {
            "tle": [
                "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
                "2 25544  51.6416 208.5340 0001234 123.4567 236.5433 15.50000000999999",
            ],
        }
    ]
    result = _normalize_api_response(raw)
    assert len(result) == 1
    assert result[0]["satellite_id"] == "25544"


def test_normalize_bad_tle():
    raw = [{"satelliteId": "1", "tle": ["only one line"]}]
    assert _normalize_api_response(raw) == []


def test_normalize_non_dict_items():
    assert _normalize_api_response([42, "str", None]) == []


def test_normalize_none():
    assert _normalize_api_response(None) == []


# ---------------------------------------------------------------------------
# ElsetCache lifecycle
# ---------------------------------------------------------------------------


def test_cache_empty_initially(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    assert cache.get_elsets() == []


def test_cache_refresh_success(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    mock_api = MagicMock()
    mock_api.get_elsets_latest.return_value = [
        {
            "satelliteId": "25544",
            "satelliteName": "ISS",
            "tle": ["1 25544U ...", "2 25544 ..."],
        }
    ]
    assert cache.refresh(mock_api, logger=MagicMock()) is True
    assert len(cache.get_elsets()) == 1
    assert (tmp_path / "elsets.json").exists()


def test_cache_refresh_failure(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    mock_api = MagicMock()
    mock_api.get_elsets_latest.return_value = None
    assert cache.refresh(mock_api, logger=MagicMock()) is False


def test_cache_load_from_file(tmp_path):
    data = [{"satellite_id": "25544", "name": "ISS", "tle": ["l1", "l2"]}]
    cache_path = tmp_path / "elsets.json"
    cache_path.write_text(json.dumps(data))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file()
    assert len(cache.get_elsets()) == 1


def test_cache_load_sets_refresh_epoch_from_mtime(tmp_path):
    """Loading from file should set _last_refresh_epoch to the file's mtime,
    so refresh_if_stale() doesn't immediately re-download."""
    data = [{"satellite_id": "25544", "name": "ISS", "tle": ["l1", "l2"]}]
    cache_path = tmp_path / "elsets.json"
    cache_path.write_text(json.dumps(data))

    cache = ElsetCache(cache_path=cache_path)
    assert cache._last_refresh_epoch == 0.0
    cache.load_from_file()
    assert cache._last_refresh_epoch > 0.0
    assert cache._last_refresh_epoch == cache_path.stat().st_mtime

    mock_api = MagicMock()
    mock_api.get_elsets_latest.return_value = []
    cache.refresh_if_stale(mock_api, interval_hours=24)
    mock_api.get_elsets_latest.assert_not_called()


def test_cache_load_from_missing_file(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "nope.json")
    cache.load_from_file()
    assert cache.get_elsets() == []


def test_cache_refresh_if_stale(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    mock_api = MagicMock()
    mock_api.get_elsets_latest.return_value = []

    assert cache.refresh_if_stale(mock_api, interval_hours=0.001) is True
    mock_api.get_elsets_latest.assert_called_once()

    mock_api.reset_mock()
    assert cache.refresh_if_stale(mock_api, interval_hours=24) is False
    mock_api.get_elsets_latest.assert_not_called()
