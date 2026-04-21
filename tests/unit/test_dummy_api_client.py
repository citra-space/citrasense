"""Tests for DummyApiClient — verifies pass-based task scheduling with live TLEs."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from skyfield.api import EarthSatellite, load, wgs84

from citrasense.api.dummy_api_client import (
    DummyApiClient,
    _parse_3le_text,
)

_SAMPLE_3LE = """\
DIRECTV 1 (DBS 1)
1 22930U 93078A   26055.49034455 -.00000036  00000+0  00000+0 0  9993
2 22930  11.9265  34.8070 0008996 218.2353 346.4629  0.98983377124904
DIRECTV 1R
1 25937U 99056A   26055.62504652 -.00000075  00000+0  00000+0 0  9994
2 25937  10.4213  49.5410 0006826 255.8956 271.2738  0.98961757 96102
"""


class TestParse3le:
    def test_parses_valid_3le(self):
        result = _parse_3le_text(_SAMPLE_3LE)
        assert len(result) == 2
        assert result[0]["name"] == "DIRECTV 1 (DBS 1)"
        assert result[0]["tle_line1"].startswith("1 22930")
        assert result[1]["name"] == "DIRECTV 1R"


@pytest.fixture
def client():
    """Create a DummyApiClient with a small mock catalog (no network)."""
    catalog = _parse_3le_text(_SAMPLE_3LE)
    mock_catalog = {}
    for entry in catalog:
        norad_id = entry["tle_line1"].split()[1].rstrip("U")
        mock_catalog[f"sat-{norad_id}"] = entry

    with patch("citrasense.api.dummy_api_client.load_satellite_catalog", return_value=mock_catalog):
        return DummyApiClient()


class TestDummyPassScheduling:
    def test_generated_tasks_are_above_horizon(self, client: DummyApiClient):
        """Every auto-generated task should have the satellite above the horizon at mid-pass."""
        tasks = client.get_telescope_tasks("dummy-telescope-001")

        if not tasks:
            pytest.skip("No visible passes found in search window (location/time dependent)")

        gs = client.data["ground_station"]
        ts = load.timescale()
        observer = wgs84.latlon(gs["latitude"], gs["longitude"], elevation_m=gs["altitude"])

        for task in tasks:
            sat_id = task["satelliteId"]
            cat = client._satellite_catalog[sat_id]
            sat = EarthSatellite(cat["tle_line1"], cat["tle_line2"], cat["name"], ts)

            start = datetime.fromisoformat(task["taskStart"])
            stop = datetime.fromisoformat(task["taskStop"])
            mid = start + (stop - start) / 2
            t = ts.from_datetime(mid)

            topo = (sat - observer).at(t)
            alt_deg, _, _ = topo.altaz()
            elevation: float = alt_deg.degrees  # type: ignore[assignment]
            assert elevation > 0, f"{cat['name']} at {mid.isoformat()} is at {elevation:.1f}° (expected above horizon)"

    def test_no_tasks_with_stale_pass(self, client: DummyApiClient):
        tasks = client.get_telescope_tasks("dummy-telescope-001")
        now = datetime.now(timezone.utc)
        for task in tasks:
            stop = datetime.fromisoformat(task["taskStop"])
            assert stop > now, f"Task {task['id']} has already expired (stop={stop})"

    def test_satellite_lookup_works(self, client: DummyApiClient):
        """get_satellite should return data for any satellite in the catalog."""
        for sat_id in client._satellite_catalog:
            sat_data = client.get_satellite(sat_id)
            assert sat_data is not None
            assert len(sat_data["elsets"]) > 0
            assert len(sat_data["elsets"][0]["tle"]) == 2

    def test_get_elsets_latest_returns_all_catalog_satellites(self, client: DummyApiClient):
        elsets = client.get_elsets_latest()
        returned_ids = {e["satelliteId"] for e in elsets}
        assert returned_ids == set(client._satellite_catalog.keys())


class TestDummyCancelTask:
    def test_cancel_task_marks_pending_canceled(self, client: DummyApiClient):
        tasks = client.get_telescope_tasks("dummy-telescope-001")
        if not tasks:
            pytest.skip("No tasks generated to cancel")

        target = tasks[0]
        assert target["status"] in ("Pending", "Scheduled")

        assert client.cancel_task(target["id"]) is True

        # The task is still in the underlying store, but flipped to Canceled.
        stored = next(t for t in client.data["tasks"] if t["id"] == target["id"])
        assert stored["status"] == "Canceled"

        # Subsequent fetches (Pending/Scheduled only) shouldn't return it.
        remaining = client.get_telescope_tasks("dummy-telescope-001")
        assert all(t["id"] != target["id"] for t in remaining)

    def test_cancel_unknown_task_returns_false(self, client: DummyApiClient):
        assert client.cancel_task("does-not-exist") is False

    def test_cancel_terminal_task_returns_false(self, client: DummyApiClient):
        # Seed a task in a terminal state directly into the store.
        client.data.setdefault("tasks", []).append({"id": "done-1", "status": "Succeeded"})
        assert client.cancel_task("done-1") is False
        stored = next(t for t in client.data["tasks"] if t["id"] == "done-1")
        assert stored["status"] == "Succeeded"
