"""Dummy API client for local testing without real server.

On first init, fetches fresh TLEs from CelesTrak for a mix of LEO (Starlink, ISS)
and GEO (DirecTV) satellites. Results are cached to disk so subsequent runs work
offline. The task scheduler uses Skyfield pass prediction to only generate tasks
when satellites are actually visible from the ground station.
"""

import json
import logging
import random
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import platformdirs
import requests
from skyfield.api import EarthSatellite, load, wgs84

from .abstract_api_client import AbstractCitraApiClient

logger = logging.getLogger(__name__)

_APP_NAME = "citrascope"
_APP_AUTHOR = "citrascope"
_CACHE_FILE = "dummy_tle_cache.json"
_CACHE_MAX_AGE_HOURS = 24

# CelesTrak endpoints
_CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"
_TLE_SOURCES: list[dict[str, str | int]] = [
    {"url": f"{_CELESTRAK_BASE}?NAME=DIRECTV&FORMAT=tle", "label": "DirecTV", "limit": 0},
]

MIN_ELEVATION_DEG = 15.0
PASS_SEARCH_HOURS = 12

_DEFAULT_STATION_LAT = 38.8409
_DEFAULT_STATION_LON = -105.0423
_DEFAULT_STATION_ALT = 4302  # metres, summit of Pikes Peak


def _parse_3le_text(text: str) -> list[dict[str, str]]:
    """Parse CelesTrak 3LE format (-name / line1 / line2) into dicts."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    results: list[dict[str, str]] = []
    i = 0
    while i + 2 < len(lines):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            results.append({"name": lines[i].strip(), "tle_line1": lines[i + 1], "tle_line2": lines[i + 2]})
            i += 3
        else:
            i += 1
    return results


def _get_cache_path() -> Path:
    return Path(platformdirs.user_data_dir(_APP_NAME, appauthor=_APP_AUTHOR)) / _CACHE_FILE


def _fetch_tles_from_celestrak() -> dict[str, dict[str, str]]:
    """Fetch TLEs from CelesTrak, returning a sat_id -> {name, tle_line1, tle_line2} dict."""
    catalog: dict[str, dict[str, str]] = {}
    for source in _TLE_SOURCES:
        try:
            resp = requests.get(str(source["url"]), timeout=15)
            resp.raise_for_status()
            parsed = _parse_3le_text(resp.text)
            limit = int(source.get("limit", 0))
            if limit > 0:
                parsed = parsed[:limit]
            for entry in parsed:
                norad_id = entry["tle_line1"].split()[1].rstrip("U")
                sat_id = f"sat-{norad_id}"
                catalog[sat_id] = entry
            logger.info("CelesTrak: fetched %d %s TLEs", len(parsed), source["label"])
        except Exception as exc:
            logger.warning("CelesTrak fetch failed for %s: %s", source["label"], exc)
    return catalog


def _load_cached_tles() -> dict[str, dict[str, str]] | None:
    """Load TLEs from disk cache. Returns None if stale or missing."""
    cache_path = _get_cache_path()
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        if (datetime.now(timezone.utc) - cached_at).total_seconds() > _CACHE_MAX_AGE_HOURS * 3600:
            logger.info("TLE cache is stale (older than %dh), will re-fetch", _CACHE_MAX_AGE_HOURS)
            return None
        return data["satellites"]
    except Exception as exc:
        logger.warning("Failed to load TLE cache: %s", exc)
        return None


def _save_cached_tles(catalog: dict[str, dict[str, str]]) -> None:
    cache_path = _get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"cached_at": datetime.now(timezone.utc).isoformat(), "satellites": catalog}, indent=2)
    )
    logger.info("Cached %d TLEs to %s", len(catalog), cache_path)


def _load_forced_cache() -> dict[str, dict[str, str]] | None:
    """Last-resort: load cache ignoring staleness."""
    cache_path = _get_cache_path()
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        logger.info("Using stale TLE cache as fallback (%d sats)", len(data["satellites"]))
        return data["satellites"]
    except Exception:
        return None


def load_satellite_catalog() -> dict[str, dict[str, str]]:
    """Load satellite catalog: fresh from CelesTrak > disk cache > empty."""
    cached = _load_cached_tles()
    if cached:
        logger.info("Loaded %d TLEs from cache", len(cached))
        return cached

    fetched = _fetch_tles_from_celestrak()
    if fetched:
        _save_cached_tles(fetched)
        return fetched

    stale = _load_forced_cache()
    if stale:
        return stale

    logger.warning("No TLE data available (network down, no cache). DummyApiClient will have no satellites.")
    return {}


class DummyApiClient(AbstractCitraApiClient):
    """
    Dummy API client that keeps all data in memory.

    Perfect for testing the task pipeline without needing the real API server.
    Automatically maintains ~10 upcoming tasks at all times.
    """

    # Simulated failure rate for testing retry logic (30% chance of upload failure)
    UPLOAD_FAILURE_RATE = 0.1

    def __init__(self, logger=None):
        """Initialize dummy API client with in-memory data."""
        self.logger = logger

        # Thread-safe data access
        self._data_lock = threading.Lock()

        # Initialize in-memory data structures
        self._initialize_data()

        if self.logger:
            self.logger.info("DummyApiClient initialized (in-memory mode)")

    def _initialize_data(self):
        """Initialize in-memory data structures with fresh TLEs from CelesTrak."""
        self._satellite_catalog = load_satellite_catalog()
        now_iso = datetime.now(timezone.utc).isoformat()

        satellites: dict[str, dict] = {}
        for sat_id, cat in self._satellite_catalog.items():
            satellites[sat_id] = {
                "id": sat_id,
                "name": cat["name"],
                "elsets": [
                    {
                        "tle": [cat["tle_line1"], cat["tle_line2"]],
                        "tle_line1": cat["tle_line1"],
                        "tle_line2": cat["tle_line2"],
                        "creationEpoch": now_iso,
                    }
                ],
            }

        self.data: dict = {
            "telescope": {
                "id": "dummy-telescope-001",
                "name": "Dummy Telescope",
                "groundStationId": "dummy-gs-001",
                "automatedScheduling": True,
                "maxSlewRate": 5.0,
                "pixelSize": 5.86,
                "focalLength": 200.0,
                "focalRatio": 3.4,
                "horizontalPixelCount": 1024,
                "verticalPixelCount": 1024,
                "imageCircleDiameter": None,
                "angularNoise": 2.0,
                "spectralMinWavelengthNm": None,
                "spectralMaxWavelengthNm": None,
            },
            "ground_station": {
                "id": "dummy-gs-001",
                "name": "Pikes Peak Observatory",
                "latitude": _DEFAULT_STATION_LAT,
                "longitude": _DEFAULT_STATION_LON,
                "altitude": _DEFAULT_STATION_ALT,
            },
            "tasks": [],
            "satellites": satellites,
        }

        self._ts = load.timescale()
        self._skyfield_sats: dict[str, EarthSatellite] = {}
        for sat_id, cat in self._satellite_catalog.items():
            self._skyfield_sats[sat_id] = EarthSatellite(cat["tle_line1"], cat["tle_line2"], cat["name"], self._ts)

    # Abstract methods implementation

    def does_api_server_accept_key(self):
        """Check if the API key is valid (always True for dummy)."""
        if self.logger:
            self.logger.debug("DummyApiClient: API key check (always valid)")
        return True

    def get_telescope(self, telescope_id):
        """Get telescope details."""
        with self._data_lock:
            telescope = self.data.get("telescope", {})

            # Ensure required fields exist
            if "maxSlewRate" not in telescope:
                telescope["maxSlewRate"] = 5.0
                if self.logger:
                    self.logger.info("DummyApiClient: Added missing maxSlewRate to telescope data")

            if self.logger:
                self.logger.debug(f"DummyApiClient: get_telescope({telescope_id})")
            return telescope

    def get_satellite(self, satellite_id):
        """Fetch satellite details including TLE.

        Auto-populates missing satellites from the fetched catalog.
        """
        with self._data_lock:
            satellites = self.data.get("satellites", {})

            if satellite_id not in satellites:
                if satellite_id not in self._satellite_catalog:
                    if self.logger:
                        self.logger.warning(f"DummyApiClient: Unknown satellite {satellite_id}")
                    return None

                cat = self._satellite_catalog[satellite_id]
                line1, line2 = cat["tle_line1"], cat["tle_line2"]
                satellite = {
                    "id": satellite_id,
                    "name": cat["name"],
                    "elsets": [
                        {
                            "tle": [line1, line2],
                            "tle_line1": line1,
                            "tle_line2": line2,
                            "creationEpoch": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                }
                satellites[satellite_id] = satellite
                if self.logger:
                    self.logger.info(f"DummyApiClient: Auto-populated satellite {satellite_id}")
            else:
                satellite = satellites[satellite_id]

            if self.logger:
                self.logger.debug(f"DummyApiClient: get_satellite({satellite_id})")
            return satellite

    def get_telescope_tasks(self, telescope_id):
        """Fetch tasks for telescope (returns only Pending/Scheduled).

        Automatically maintains ~10 upcoming tasks at all times for easy testing.
        """
        with self._data_lock:
            tasks = self.data.get("tasks", [])

            # Clean up old completed/failed tasks (keep last 20 for history)
            now = datetime.now(timezone.utc)
            active_tasks = []
            completed_tasks = []

            for task in tasks:
                status = task.get("status")
                if status in ["Pending", "Scheduled"]:
                    # Check if task is too old (expired)
                    try:
                        task_stop = datetime.fromisoformat(task.get("taskStop", "").replace("Z", "+00:00"))
                        if task_stop < now:
                            # Task expired, mark as failed
                            task["status"] = "Failed"
                            completed_tasks.append(task)
                        else:
                            active_tasks.append(task)
                    except Exception:
                        active_tasks.append(task)
                else:
                    completed_tasks.append(task)

            completed_tasks = completed_tasks[-20:] if len(completed_tasks) > 20 else completed_tasks

            telescope = self.data.get("telescope", {})
            scheduling_enabled = telescope.get("automatedScheduling", True)
            target_task_count = 10
            if scheduling_enabled and len(active_tasks) < target_task_count:
                ground_station = self.data.get("ground_station", {})
                new_tasks = self._find_upcoming_passes(
                    ground_station, telescope, target_task_count - len(active_tasks), active_tasks
                )
                active_tasks.extend(new_tasks)
                self.data["tasks"] = active_tasks + completed_tasks

                if new_tasks and self.logger:
                    self.logger.info(f"DummyApiClient: Auto-generated {len(new_tasks)} new tasks")

            if self.logger:
                self.logger.debug(f"DummyApiClient: get_telescope_tasks({telescope_id}) -> {len(active_tasks)} tasks")
            return active_tasks

    def _make_task(
        self,
        sat_id: str,
        start: datetime,
        stop: datetime,
        telescope: dict,
        ground_station: dict,
    ) -> dict:
        cat = self._satellite_catalog.get(sat_id, {})
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "id": str(uuid.uuid4()),
            "type": "Track",
            "status": "Pending",
            "creationEpoch": now_iso,
            "updateEpoch": now_iso,
            "satelliteId": sat_id,
            "satelliteName": cat.get("name", sat_id),
            "taskStart": start.isoformat(),
            "taskStop": stop.isoformat(),
            "telescopeId": telescope.get("id", "dummy-telescope-001"),
            "telescopeName": telescope.get("name", "Dummy Telescope"),
            "groundStationId": ground_station.get("id", "dummy-gs-001"),
            "groundStationName": ground_station.get("name", "Test Ground Station"),
            "userId": "dummy-user",
            "username": "Test User",
        }

    def _find_upcoming_passes(
        self,
        ground_station: dict,
        telescope: dict,
        max_tasks: int,
        existing_tasks: list[dict],
    ) -> list[dict]:
        """Find satellites visible now and upcoming passes, prioritizing immediate targets."""
        observer = wgs84.latlon(
            ground_station.get("latitude", _DEFAULT_STATION_LAT),
            ground_station.get("longitude", _DEFAULT_STATION_LON),
            elevation_m=ground_station.get("altitude", _DEFAULT_STATION_ALT),
        )

        now = datetime.now(timezone.utc)
        t_now = self._ts.from_datetime(now)
        t_end = self._ts.from_datetime(now + timedelta(hours=PASS_SEARCH_HOURS))

        scheduled_sat_ids = {t.get("satelliteId") for t in existing_tasks}
        immediate_sats: list[tuple[str, float]] = []
        future_passes: list[tuple[datetime, str, datetime, datetime]] = []

        for sat_id, sat in self._skyfield_sats.items():
            if sat_id in scheduled_sat_ids:
                continue

            cat = self._satellite_catalog.get(sat_id, {})

            # Check if satellite is visible RIGHT NOW (handles GEO + mid-pass LEO)
            try:
                topo = (sat - observer).at(t_now)
                alt_deg = float(topo.altaz()[0].degrees)  # type: ignore[arg-type]
            except Exception:
                alt_deg = -1.0

            if alt_deg > MIN_ELEVATION_DEG:
                immediate_sats.append((sat_id, alt_deg))
                scheduled_sat_ids.add(sat_id)
                continue

            # Not visible now — find next pass
            try:
                t_events, events = sat.find_events(observer, t_now, t_end, altitude_degrees=MIN_ELEVATION_DEG)
            except Exception:
                continue

            for i, event_type in enumerate(events):
                if event_type != 1:
                    continue

                culm_dt = t_events[i].utc_datetime()
                if culm_dt < now + timedelta(seconds=10):
                    continue

                rise_dt = culm_dt - timedelta(seconds=30)
                set_dt = culm_dt + timedelta(seconds=30)
                for j in range(i - 1, -1, -1):
                    if events[j] == 0:
                        rise_dt = t_events[j].utc_datetime()
                        break
                for j in range(i + 1, len(events)):
                    if events[j] == 2:
                        set_dt = t_events[j].utc_datetime()
                        break

                future_passes.append((rise_dt, sat_id, rise_dt, set_dt))
                break

        # Build immediate tasks staggered 15s apart for steady throughput
        immediate_tasks: list[dict] = []
        _FIRST_TASK_DELAY_S = 10
        _TASK_STAGGER_S = 15
        for idx, (sat_id, alt_deg) in enumerate(immediate_sats):
            task_start = now + timedelta(seconds=_FIRST_TASK_DELAY_S + _TASK_STAGGER_S * idx)
            task_stop = task_start + timedelta(seconds=60)
            immediate_tasks.append(self._make_task(sat_id, task_start, task_stop, telescope, ground_station))
            cat = self._satellite_catalog.get(sat_id, {})
            if self.logger:
                self.logger.info(
                    "DummyApiClient: %s visible now at %.1f° — task in %ds",
                    cat.get("name", sat_id),
                    alt_deg,
                    _FIRST_TASK_DELAY_S + _TASK_STAGGER_S * idx,
                )

        # Immediate tasks first, then future passes sorted by start time
        future_passes.sort(key=lambda x: x[0])
        new_tasks = immediate_tasks[:max_tasks]
        remaining = max_tasks - len(new_tasks)

        for _, sat_id, rise_dt, set_dt in future_passes[:remaining]:
            cat = self._satellite_catalog.get(sat_id, {})
            new_tasks.append(self._make_task(sat_id, rise_dt, set_dt, telescope, ground_station))
            if self.logger:
                duration = (set_dt - rise_dt).total_seconds()
                self.logger.info(
                    "DummyApiClient: Scheduled %s pass at %s (%.0fs, above %d°)",
                    cat.get("name", sat_id),
                    rise_dt.strftime("%H:%M:%S UTC"),
                    duration,
                    MIN_ELEVATION_DEG,
                )

        return new_tasks

    def get_ground_station(self, ground_station_id):
        """Fetch ground station details."""
        with self._data_lock:
            ground_station = self.data.get("ground_station", {})
            if self.logger:
                self.logger.debug(f"DummyApiClient: get_ground_station({ground_station_id})")
            return ground_station

    def put_telescope_status(self, body):
        """Report telescope online status."""
        if self.logger:
            self.logger.debug(f"DummyApiClient: put_telescope_status({body})")
        return {"status": "ok"}

    def expand_filters(self, filter_names):
        """Expand filter names to spectral specs."""
        if self.logger:
            self.logger.debug(f"DummyApiClient: expand_filters({filter_names})")
        known_filters = {
            "Red": (635.0, 120.0),
            "Green": (530.0, 100.0),
            "Blue": (450.0, 100.0),
            "Clear": (550.0, 300.0),
            "Luminance": (550.0, 300.0),
            "Ha": (656.3, 7.0),
            "OIII": (500.7, 7.0),
            "SII": (671.6, 7.0),
        }
        filters = []
        for name in filter_names:
            wavelength, bandwidth = known_filters.get(name, (550.0, 100.0))
            filters.append(
                {
                    "name": name,
                    "central_wavelength_nm": wavelength,
                    "bandwidth_nm": bandwidth,
                    "is_known": name in known_filters,
                }
            )
        return {"filters": filters}

    def update_telescope_spectral_config(self, telescope_id, spectral_config):
        """Update telescope spectral configuration."""
        if self.logger:
            self.logger.debug(f"DummyApiClient: update_telescope_spectral_config({telescope_id})")
        return {"status": "ok"}

    def update_ground_station_location(self, ground_station_id, latitude, longitude, altitude):
        """Update ground station GPS location."""
        with self._data_lock:
            if self.logger:
                self.logger.debug(
                    f"DummyApiClient: update_ground_station_location({ground_station_id}, "
                    f"lat={latitude}, lon={longitude}, alt={altitude})"
                )
            # Update in-memory data
            if "ground_station" in self.data:
                self.data["ground_station"]["latitude"] = latitude
                self.data["ground_station"]["longitude"] = longitude
                self.data["ground_station"]["altitude"] = altitude
            return {"status": "ok"}

    def get_elsets_latest(self, days: int = 14):
        """Return stub list of elsets (same shape as real API: satelliteId, satelliteName, tle)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        result = []
        for sat_id, cat in self._satellite_catalog.items():
            result.append(
                {
                    "satelliteId": sat_id,
                    "satelliteName": cat["name"],
                    "tle": [str(cat["tle_line1"]), str(cat["tle_line2"])],
                    "creationEpoch": now_iso,
                }
            )
        if self.logger:
            self.logger.debug(f"DummyApiClient: get_elsets_latest(days={days}) -> {len(result)} items")
        return result

    def update_telescope_automated_scheduling(self, telescope_id: str, enabled: bool) -> bool:
        with self._data_lock:
            telescope = self.data.get("telescope", {})
            telescope["automatedScheduling"] = enabled
        if self.logger:
            self.logger.info(f"DummyApiClient: automated scheduling {'enabled' if enabled else 'disabled'}")
        return True

    def upload_optical_observations(
        self,
        observations: list,
        telescope_record: dict,
        sensor_location: dict,
        task_id: str | None = None,
    ) -> bool:
        """Stub: log and return True (no real upload in dummy mode)."""
        if self.logger:
            self.logger.info(f"DummyApiClient: upload_optical_observations({len(observations)} obs, task={task_id})")
        return True

    # Additional methods used by the system

    def upload_image(self, task_id, telescope_id, filepath):
        """Fake image upload with simulated random failures for testing retry logic."""
        if self.logger:
            self.logger.info(f"DummyApiClient: Fake upload for task {task_id}: {filepath}")

        # Simulate upload delay (5 seconds)
        time.sleep(5)

        # Randomly fail to test retry logic
        if random.random() < self.UPLOAD_FAILURE_RATE:
            if self.logger:
                self.logger.warning(f"DummyApiClient: Simulated upload failure for task {task_id}")
            return None  # Indicate upload failure

        # Return a fake results URL on success
        return f"https://dummy-server/results/{task_id}"

    def mark_task_complete(self, task_id):
        """Mark a task as complete with simulated random failures for testing retry logic."""
        # Randomly fail to test retry logic
        if random.random() < self.UPLOAD_FAILURE_RATE:
            if self.logger:
                self.logger.warning(f"DummyApiClient: Simulated mark_complete failure for task {task_id}")
            return False  # Indicate failure to mark complete

        with self._data_lock:
            tasks = self.data.get("tasks", [])

            for task in tasks:
                if task.get("id") == task_id:
                    task["status"] = "Succeeded"
                    task["updateEpoch"] = datetime.now(timezone.utc).isoformat()
                    if self.logger:
                        self.logger.info(f"DummyApiClient: Marked task {task_id} as Succeeded")
                    break

        return True  # Success

    def mark_task_failed(self, task_id):
        """Mark a task as failed (updates in-memory)."""
        with self._data_lock:
            tasks = self.data.get("tasks", [])

            for task in tasks:
                if task.get("id") == task_id:
                    task["status"] = "Failed"
                    task["updateEpoch"] = datetime.now(timezone.utc).isoformat()
                    break

            if self.logger:
                self.logger.info(f"DummyApiClient: Marked task {task_id} as Failed")

            return {"status": "Failed"}
