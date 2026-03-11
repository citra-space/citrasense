"""Unit tests for ElsetCache."""

import json
import time
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
    mock_api.cache_source_key = "https://api.citra.space"
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


def test_cache_refresh_if_stale(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    mock_api = MagicMock()
    mock_api.cache_source_key = "https://api.citra.space"
    mock_api.get_elsets_latest.return_value = []

    assert cache.refresh_if_stale(mock_api, interval_hours=0.001) is True
    mock_api.get_elsets_latest.assert_called_once()

    mock_api.reset_mock()
    assert cache.refresh_if_stale(mock_api, interval_hours=24) is False
    mock_api.get_elsets_latest.assert_not_called()


# ---------------------------------------------------------------------------
# Tagged format and source validation
# ---------------------------------------------------------------------------


def test_refresh_writes_tagged_format(tmp_path):
    """refresh() should write the new tagged JSON format with source + elsets."""
    cache_path = tmp_path / "elsets.json"
    cache = ElsetCache(cache_path=cache_path)
    mock_api = MagicMock()
    mock_api.cache_source_key = "https://dev.api.citra.space"
    mock_api.get_elsets_latest.return_value = [
        {"satelliteId": "25544", "satelliteName": "ISS", "tle": ["1 25544U ...", "2 25544 ..."]}
    ]
    cache.refresh(mock_api, logger=MagicMock())

    data = json.loads(cache_path.read_text())
    assert isinstance(data, dict)
    assert data["source"] == "https://dev.api.citra.space"
    assert "refreshed_at" in data
    assert isinstance(data["elsets"], list)
    assert len(data["elsets"]) == 1


def test_load_tagged_format_matching_source(tmp_path):
    """load_from_file() accepts tagged data when source matches."""
    cache_path = tmp_path / "elsets.json"
    wrapper = {
        "source": "https://api.citra.space",
        "refreshed_at": "2026-03-09T00:00:00+00:00",
        "elsets": [{"satellite_id": "25544", "name": "ISS", "tle": ["l1", "l2"]}],
    }
    cache_path.write_text(json.dumps(wrapper))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file(expected_source="https://api.citra.space")
    assert len(cache.get_elsets()) == 1


def test_load_tagged_format_mismatched_source(tmp_path):
    """load_from_file() discards data when source doesn't match."""
    cache_path = tmp_path / "elsets.json"
    wrapper = {
        "source": "DummyApiClient",
        "refreshed_at": "2026-03-09T00:00:00+00:00",
        "elsets": [{"satellite_id": "1", "name": "SAT", "tle": ["l1", "l2"]}],
    }
    cache_path.write_text(json.dumps(wrapper))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file(expected_source="https://api.citra.space")
    assert cache.get_elsets() == []


def test_load_tagged_format_dev_vs_prod(tmp_path):
    """Switching between dev and prod API hosts invalidates the cache."""
    cache_path = tmp_path / "elsets.json"
    wrapper = {
        "source": "https://dev.api.citra.space",
        "refreshed_at": "2026-03-09T00:00:00+00:00",
        "elsets": [{"satellite_id": "1", "name": "SAT", "tle": ["l1", "l2"]}],
    }
    cache_path.write_text(json.dumps(wrapper))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file(expected_source="https://api.citra.space")
    assert cache.get_elsets() == []


def test_load_legacy_bare_list_discarded_with_source(tmp_path):
    """Legacy bare-list format is discarded when an expected_source is provided."""
    cache_path = tmp_path / "elsets.json"
    cache_path.write_text(json.dumps([{"satellite_id": "25544", "name": "ISS", "tle": ["l1", "l2"]}]))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file(expected_source="https://api.citra.space")
    assert cache.get_elsets() == []


def test_load_legacy_bare_list_accepted_without_source(tmp_path):
    """Legacy bare-list format is accepted when no expected_source is given."""
    cache_path = tmp_path / "elsets.json"
    cache_path.write_text(json.dumps([{"satellite_id": "25544", "name": "ISS", "tle": ["l1", "l2"]}]))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file()
    assert len(cache.get_elsets()) == 1


def test_load_from_missing_file(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "nope.json")
    cache.load_from_file()
    assert cache.get_elsets() == []


def test_load_sets_refresh_epoch_from_mtime(tmp_path):
    """Loading from file should set _last_refresh_epoch to the file's mtime."""
    cache_path = tmp_path / "elsets.json"
    wrapper = {
        "source": "https://api.citra.space",
        "elsets": [{"satellite_id": "25544", "name": "ISS", "tle": ["l1", "l2"]}],
    }
    cache_path.write_text(json.dumps(wrapper))

    cache = ElsetCache(cache_path=cache_path)
    assert cache._last_refresh_epoch == 0.0
    cache.load_from_file(expected_source="https://api.citra.space")
    assert cache._last_refresh_epoch > 0.0
    assert cache._last_refresh_epoch == cache_path.stat().st_mtime


# ---------------------------------------------------------------------------
# get_health()
# ---------------------------------------------------------------------------


def test_get_health_empty(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    health = cache.get_health()
    assert health["elset_count"] == 0
    assert health["last_refresh"] == 0.0
    assert health["source"] == ""


def test_get_health_after_refresh(tmp_path):
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    mock_api = MagicMock()
    mock_api.cache_source_key = "https://dev.api.citra.space"
    mock_api.get_elsets_latest.return_value = [
        {"satelliteId": "25544", "satelliteName": "ISS", "tle": ["1 25544U ...", "2 25544 ..."]}
    ]

    before = time.time()
    cache.refresh(mock_api, logger=MagicMock())
    after = time.time()

    health = cache.get_health()
    assert health["elset_count"] == 1
    assert health["source"] == "https://dev.api.citra.space"
    assert before <= health["last_refresh"] <= after


def test_get_health_after_load(tmp_path):
    cache_path = tmp_path / "elsets.json"
    wrapper = {
        "source": "https://api.citra.space",
        "elsets": [{"satellite_id": "1", "name": "S", "tle": ["l1", "l2"]}] * 3,
    }
    cache_path.write_text(json.dumps(wrapper))

    cache = ElsetCache(cache_path=cache_path)
    cache.load_from_file(expected_source="https://api.citra.space")

    health = cache.get_health()
    assert health["elset_count"] == 3
    assert health["source"] == "https://api.citra.space"


# ---------------------------------------------------------------------------
# Low-count warning
# ---------------------------------------------------------------------------


def test_refresh_logs_warning_for_low_count(tmp_path):
    """refresh() should log a warning when count < 25K."""
    cache = ElsetCache(cache_path=tmp_path / "elsets.json")
    mock_api = MagicMock()
    mock_api.cache_source_key = "test"
    mock_api.get_elsets_latest.return_value = [{"satelliteId": "1", "satelliteName": "S", "tle": ["1 line", "2 line"]}]
    mock_logger = MagicMock()

    cache.refresh(mock_api, logger=mock_logger)

    warning_calls = [c for c in mock_logger.warning.call_args_list if "only" in str(c) and "elsets" in str(c)]
    assert len(warning_calls) == 1
