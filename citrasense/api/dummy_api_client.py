"""Dummy API client for local testing without real server.

On first init, fetches fresh TLEs from CelesTrak for a mix of LEO (Starlink, ISS)
and GEO (DirecTV) satellites. Results are cached to disk so subsequent runs work
offline. The task scheduler uses keplemon pass prediction to only generate tasks
when satellites are actually visible from the ground station.
"""

import json
import logging
import random
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import platformdirs
import requests
from keplemon import time as ktime
from keplemon.bodies import Satellite
from keplemon.elements import TLE
from keplemon.enums import ReferenceFrame
from keplemon.time import TimeSpan

from citrasense.astro.sidereal import gast_degrees, make_observatory
from citrasense.hardware.devices.mount.altaz_pointing_model import radec_to_altaz

from .abstract_api_client import AbstractCitraApiClient

logger = logging.getLogger("citrasense.DummyApiClient")

_APP_NAME = "citrasense"
_APP_AUTHOR = "citrasense"
_CACHE_FILE = "dummy_tle_cache.json"
_CACHE_MAX_AGE_HOURS = 24

# CelesTrak endpoints
_CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"
_TLE_SOURCES: list[dict[str, str | int]] = [
    {"url": f"{_CELESTRAK_BASE}?NAME=DIRECTV&FORMAT=tle", "label": "DirecTV", "limit": 0},
]

MIN_ELEVATION_DEG = 0.0  # dummy is a UI testing stub; horizon-only filter is enough
# keplemon's access report hangs when min_duration is literally zero (every
# instant satisfies the predicate). One second is effectively "any pass" for
# a horizon-only filter while still being a defined minimum.
#
# TODO(keplemon): remove this workaround once an upstream fix lands for the
# zero-duration access-report hang. No issue filed yet; this comment is the
# placeholder so the workaround doesn't outlive the bug.
_MIN_PASS_DURATION_SECONDS = 1.0
PASS_SEARCH_HOURS = 12
# Don't anchor the queue on a future pass that's more than this far out.  The
# real Citra scheduler doesn't give you tasks 5 hours away when nothing is
# happening sooner, and a far-future anchor cascades the whole back-to-back
# stack out with it.
MAX_FUTURE_RISE_MINUTES = 10.0

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
    """Dummy API client that keeps all data in memory.

    Multi-sensor aware: each ``telescope_id`` gets its own telescope record
    and independent task pool, auto-created on first access.  Shared state
    (ground station, satellite catalog, keplemon objects) is site-level.
    """

    UPLOAD_FAILURE_RATE = 0.1

    _TELESCOPE_TEMPLATE: ClassVar[dict] = {
        "groundStationId": "dummy-gs-001",
        "automatedScheduling": True,
        "maxSlewRate": 5.0,
        "pixelSize": 5.86,
        "focalLength": 200.0,
        "focalRatio": 3.4,
        "horizontalPixelCount": 1280,
        "verticalPixelCount": 1024,
        "imageCircleDiameter": None,
        "angularNoise": 2.0,
        "spectralMinWavelengthNm": None,
        "spectralMaxWavelengthNm": None,
        "spectralConfig": {
            "type": "discrete",
            "filters": [
                {"name": "Luminance", "central_wavelength_nm": 550.0, "bandwidth_nm": 300.0},
                {"name": "Red", "central_wavelength_nm": 658.0, "bandwidth_nm": 138.0},
                {"name": "Green", "central_wavelength_nm": 551.0, "bandwidth_nm": 88.0},
                {"name": "Blue", "central_wavelength_nm": 445.0, "bandwidth_nm": 94.0},
            ],
        },
    }

    @property
    def cache_source_key(self) -> str:
        return "DummyApiClient"

    def __init__(self, logger=None):
        """Initialize dummy API client with in-memory data."""
        self.logger = logger.getChild(type(self).__name__) if logger else None

        self._data_lock = threading.Lock()

        self._initialize_data()

        if self.logger:
            self.logger.info("DummyApiClient initialized (in-memory mode)")

    # ── Data initialisation ────────────────────────────────────────────

    def _initialize_data(self):
        """Initialize in-memory data structures with fresh TLEs from CelesTrak."""
        self._satellite_catalog = load_satellite_catalog()
        now_iso = datetime.now(timezone.utc).isoformat()

        self._satellites: dict[str, dict] = {}
        for sat_id, cat in self._satellite_catalog.items():
            self._satellites[sat_id] = {
                "id": sat_id,
                "name": cat["name"],
                "elsets": [
                    {
                        "tle": [cat["tle_line1"], cat["tle_line2"]],
                        "tle_line1": cat["tle_line1"],
                        "tle_line2": cat["tle_line2"],
                        "epoch": now_iso,
                        "creationEpoch": now_iso,
                    }
                ],
            }

        self._ground_station: dict = {
            "id": "dummy-gs-001",
            "name": "Pikes Peak Observatory",
            "latitude": _DEFAULT_STATION_LAT,
            "longitude": _DEFAULT_STATION_LON,
            "altitude": _DEFAULT_STATION_ALT,
        }

        # Per-telescope records and task pools, keyed by telescope_id.
        # Auto-created on first access via _ensure_telescope().
        self._telescopes: dict[str, dict] = {}
        self._tasks: dict[str, list[dict]] = {}
        self._telescope_counter = 0

        self._keplemon_sats: dict[str, Satellite] = {}
        for sat_id, cat in self._satellite_catalog.items():
            try:
                self._keplemon_sats[sat_id] = Satellite.from_tle(TLE.from_lines(cat["tle_line1"], cat["tle_line2"]))
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "DummyApiClient: skipping %s (%s) — TLE parse failed: %s",
                        cat.get("name", sat_id),
                        sat_id,
                        exc,
                    )

    def _ensure_telescope(self, telescope_id: str) -> dict:
        """Return the telescope record for *telescope_id*, creating one on first access."""
        if telescope_id not in self._telescopes:
            self._telescope_counter += 1
            ordinal = self._telescope_counter
            rec = dict(self._TELESCOPE_TEMPLATE)
            rec["spectralConfig"] = {
                k: (list(v) if isinstance(v, list) else v)
                for k, v in self._TELESCOPE_TEMPLATE["spectralConfig"].items()
            }
            rec["spectralConfig"]["filters"] = [dict(f) for f in self._TELESCOPE_TEMPLATE["spectralConfig"]["filters"]]
            rec["id"] = telescope_id
            rec["name"] = f"Dummy Telescope {ordinal}"
            self._telescopes[telescope_id] = rec
            self._tasks[telescope_id] = []
            if self.logger:
                self.logger.info(
                    "DummyApiClient: auto-created telescope record '%s' (%s)",
                    telescope_id,
                    rec["name"],
                )
        return self._telescopes[telescope_id]

    # ── Abstract methods implementation ──────────────────────────────

    def does_api_server_accept_key(self):
        """Check if the API key is valid (always True for dummy)."""
        if self.logger:
            self.logger.debug("DummyApiClient: API key check (always valid)")
        return True

    def get_telescope(self, telescope_id):
        """Get telescope details.  Auto-creates a record on first access."""
        with self._data_lock:
            telescope = self._ensure_telescope(telescope_id)
            if self.logger:
                self.logger.debug(f"DummyApiClient: get_telescope({telescope_id})")
            return telescope

    def get_satellite(self, satellite_id):
        """Fetch satellite details including TLE.

        Auto-populates missing satellites from the fetched catalog.
        """
        with self._data_lock:
            if satellite_id not in self._satellites:
                if satellite_id not in self._satellite_catalog:
                    if self.logger:
                        self.logger.warning(f"DummyApiClient: Unknown satellite {satellite_id}")
                    return None

                cat = self._satellite_catalog[satellite_id]
                line1, line2 = cat["tle_line1"], cat["tle_line2"]
                now_iso = datetime.now(timezone.utc).isoformat()
                self._satellites[satellite_id] = {
                    "id": satellite_id,
                    "name": cat["name"],
                    "elsets": [
                        {
                            "tle": [line1, line2],
                            "tle_line1": line1,
                            "tle_line2": line2,
                            "epoch": now_iso,
                            "creationEpoch": now_iso,
                        }
                    ],
                }
                if self.logger:
                    self.logger.info(f"DummyApiClient: Auto-populated satellite {satellite_id}")

            if self.logger:
                self.logger.debug(f"DummyApiClient: get_satellite({satellite_id})")
            return self._satellites[satellite_id]

    def get_best_elset(self, satellite_id, types: Sequence[str] | None = None) -> dict | None:
        """Return the single elset for a dummy satellite (mimics server's best-elset logic).

        The ``types`` argument is accepted for parity with the real client. The
        dummy cache only carries classic-SGP4 TLEs, so the filter is a no-op
        here — every cached entry already matches ``CLASSIC_SGP4_ELSET_TYPES``.
        """
        del types  # dummy data is always classic SGP4
        with self._data_lock:
            satellite = self._satellites.get(satellite_id)
            if not satellite:
                if satellite_id in self._satellite_catalog:
                    cat = self._satellite_catalog[satellite_id]
                    now_iso = datetime.now(timezone.utc).isoformat()
                    return {
                        "tle": [cat["tle_line1"], cat["tle_line2"]],
                        "epoch": now_iso,
                        "creationEpoch": now_iso,
                    }
                return None
            elsets = satellite.get("elsets", [])
            if not elsets:
                return None
            return elsets[0]

    def get_telescope_tasks(self, telescope_id, statuses=None, task_stop_after=None):
        """Fetch tasks for a specific telescope.

        Each telescope_id maintains its own independent task pool.
        Automatically generates ~10 upcoming tasks when the pool runs low.
        """
        with self._data_lock:
            telescope = self._ensure_telescope(telescope_id)
            tasks = self._tasks[telescope_id]

            now = datetime.now(timezone.utc)
            active_tasks = []
            completed_tasks = []

            for task in tasks:
                status = task.get("status")
                if status in ["Pending", "Scheduled"]:
                    try:
                        task_stop = datetime.fromisoformat(task.get("taskStop", "").replace("Z", "+00:00"))
                        if task_stop < now:
                            task["status"] = "Failed"
                            completed_tasks.append(task)
                        else:
                            active_tasks.append(task)
                    except Exception:
                        active_tasks.append(task)
                else:
                    completed_tasks.append(task)

            completed_tasks = completed_tasks[-20:]

            scheduling_enabled = telescope.get("automatedScheduling", True)
            target_task_count = 10
            if scheduling_enabled and len(active_tasks) < target_task_count:
                new_tasks = self._find_upcoming_passes(
                    self._ground_station, telescope, target_task_count - len(active_tasks), active_tasks
                )
                active_tasks.extend(new_tasks)
                self._tasks[telescope_id] = active_tasks + completed_tasks

                if new_tasks and self.logger:
                    self.logger.debug(
                        "DummyApiClient: Auto-generated %d new tasks for %s", len(new_tasks), telescope_id
                    )

            if self.logger:
                self.logger.debug(f"DummyApiClient: get_telescope_tasks({telescope_id}) -> {len(active_tasks)} tasks")
            return active_tasks

    def _find_task_across_pools(self, task_id: str) -> dict | None:
        """Find a task by ID across all telescope task pools.  Caller must hold _data_lock."""
        for pool in self._tasks.values():
            for task in pool:
                if task.get("id") == task_id:
                    return task
        return None

    def cancel_task(self, task_id) -> bool:
        """Cancel a task in the dummy store by setting status=Canceled.

        Returns True if the task was found and cancellable (i.e. not already
        in a terminal state), False otherwise.
        """
        with self._data_lock:
            task = self._find_task_across_pools(task_id)
            if task is not None:
                if task.get("status") in ("Canceled", "Failed", "Succeeded"):
                    if self.logger:
                        self.logger.warning(
                            f"DummyApiClient: cancel_task({task_id}) refused — already {task['status']}"
                        )
                    return False
                task["status"] = "Canceled"
                if self.logger:
                    self.logger.info(f"DummyApiClient: cancel_task({task_id}) -> Canceled")
                return True
        if self.logger:
            self.logger.warning(f"DummyApiClient: cancel_task({task_id}) — task not found")
        return False

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
            "assignedFilterName": self._random_filter_name(telescope),
        }

    def _random_filter_name(self, telescope: dict) -> str | None:
        """Pick a random filter from the telescope's spectralConfig, if any."""
        spec = telescope.get("spectralConfig", {})
        filters = spec.get("filters", [])
        if not filters:
            return None
        return random.choice(filters)["name"]

    def _find_upcoming_passes(
        self,
        ground_station: dict,
        telescope: dict,
        max_tasks: int,
        existing_tasks: list[dict],
    ) -> list[dict]:
        """Find satellites visible now and upcoming passes, prioritizing immediate targets."""
        lat_deg = float(ground_station.get("latitude", _DEFAULT_STATION_LAT))
        lon_deg = float(ground_station.get("longitude", _DEFAULT_STATION_LON))
        alt_m = float(ground_station.get("altitude", _DEFAULT_STATION_ALT))
        observer = make_observatory(lat_deg, lon_deg, alt_m)

        now = datetime.now(timezone.utc)
        now_epoch = ktime.Epoch.from_datetime(now)
        end_epoch = ktime.Epoch.from_datetime(now + timedelta(hours=PASS_SEARCH_HOURS))
        min_span = TimeSpan.from_seconds(_MIN_PASS_DURATION_SECONDS)

        scheduled_sat_ids = {t.get("satelliteId") for t in existing_tasks}

        # Seed the scheduling cursor from the tail of whatever's already queued
        # so refills can't lay new tasks on top of existing ones.  Without this,
        # every refill would re-anchor the immediate batch at "now + 10" and
        # collide with the back end of the queue from the previous call.
        latest_existing_stop: datetime | None = None
        for t in existing_tasks:
            try:
                stop = datetime.fromisoformat(str(t.get("taskStop", "")).replace("Z", "+00:00"))
                if latest_existing_stop is None or stop > latest_existing_stop:
                    latest_existing_stop = stop
            except Exception:
                continue

        immediate_sats: list[tuple[str, float]] = []
        future_passes: list[tuple[datetime, str, datetime, datetime]] = []

        for sat_id, sat in self._keplemon_sats.items():
            if sat_id in scheduled_sat_ids:
                continue

            # Check if satellite is visible RIGHT NOW (handles GEO + mid-pass LEO).
            # keplemon's access report needs min_duration > 0 or it hangs, so
            # for the point-in-time "visible now" check we convert J2000
            # topocentric RA/Dec to alt/az via the shared helper.
            #
            # Pass ``_gast_override`` anchored to the same ``now`` used for
            # propagation. Today both sides are "now" so the outcome is
            # sub-arcsec identical to letting ``radec_to_altaz`` fall back to
            # wall-clock GAST, but keeping the anchoring explicit makes the
            # pattern uniform with ``sky_enrichment._propagate_static`` and
            # prevents a silent bug if anyone ever refactors this into a
            # "visible at window-start" check (see PR #301 Copilot comment).
            try:
                gast_deg = gast_degrees(now)
                topo = observer.get_topocentric_to_satellite(now_epoch, sat, ReferenceFrame.J2000)
                _, alt_deg = radec_to_altaz(
                    float(topo.right_ascension),
                    float(topo.declination),
                    lat_deg,
                    lon_deg,
                    _gast_override=gast_deg,
                )
            except Exception:
                alt_deg = -1.0

            if alt_deg > MIN_ELEVATION_DEG:
                immediate_sats.append((sat_id, alt_deg))
                scheduled_sat_ids.add(sat_id)
                continue

            # Not visible now — find the next pass via keplemon's access report.
            try:
                report = sat.get_observatory_access_report(
                    [observer], now_epoch, end_epoch, MIN_ELEVATION_DEG, min_span
                )
            except Exception:
                continue
            if report is None:
                continue

            for access in report.accesses:
                try:
                    rise_dt = access.start.epoch.to_datetime().replace(tzinfo=timezone.utc)
                    set_dt = access.end.epoch.to_datetime().replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if set_dt < now + timedelta(seconds=10):
                    continue
                # First pass that starts after "right now" is the one we want.
                future_passes.append((rise_dt, sat_id, rise_dt, set_dt))
                break

        # All new tasks (immediate + future passes) are stacked against a single
        # scheduling cursor so nothing in the dummy queue ever overlaps.  The
        # real Citra scheduler doesn't hand out overlapping windows; emulating
        # that here keeps the lateness-attribution logic in the Analysis
        # dashboard from generating false-positive "inherited" delays.
        #
        # Three sources used to overlap before this was unified:
        #   1. Refills re-anchored "immediate" tasks at now+10 regardless of the
        #      tail of the existing queue.
        #   2. Two simultaneously-visible satellites both became immediate tasks
        #      with hand-stacked offsets, but ignored each other across calls.
        #   3. Future-pass rise/set windows from the pass-finder were appended raw,
        #      with no overlap check against immediates or each other.
        _FIRST_TASK_DELAY_S = 10
        _TASK_WINDOW_S = 30
        _INTER_TASK_GAP_S = 0  # back-to-back, like the real scheduler
        earliest_start = now + timedelta(seconds=_FIRST_TASK_DELAY_S)
        if latest_existing_stop is not None and latest_existing_stop > earliest_start:
            cursor = latest_existing_stop + timedelta(seconds=_INTER_TASK_GAP_S)
        else:
            cursor = earliest_start

        new_tasks: list[dict] = []

        for sat_id, alt_deg in immediate_sats:
            if len(new_tasks) >= max_tasks:
                break
            task_start = cursor
            task_stop = task_start + timedelta(seconds=_TASK_WINDOW_S)
            new_tasks.append(self._make_task(sat_id, task_start, task_stop, telescope, ground_station))
            cursor = task_stop + timedelta(seconds=_INTER_TASK_GAP_S)
            cat = self._satellite_catalog.get(sat_id, {})
            if self.logger:
                self.logger.debug(
                    "DummyApiClient: %s visible now at %.1f° — task at %s",
                    cat.get("name", sat_id),
                    alt_deg,
                    task_start.strftime("%H:%M:%S UTC"),
                )

        # Only honor future passes whose natural rise time both (a) doesn't
        # overlap with what we've already placed and (b) is reasonably soon.
        # Relocating a pass to the cursor was the source of the "queue anchored
        # 5 hours out" bug — once any far-future pass became the cursor, every
        # subsequent satellite got stacked behind it back-to-back.  A future
        # pass means "this satellite is visible at this real time"; if we can't
        # honor that, we drop it rather than lie about visibility.
        max_future_start = now + timedelta(minutes=MAX_FUTURE_RISE_MINUTES)
        future_passes.sort(key=lambda x: x[0])
        for _, sat_id, rise_dt, set_dt in future_passes:
            if len(new_tasks) >= max_tasks:
                break
            if rise_dt < cursor or rise_dt > max_future_start:
                continue
            natural_duration = (set_dt - rise_dt).total_seconds()
            duration_s = max(_TASK_WINDOW_S, min(natural_duration, _TASK_WINDOW_S * 4))
            task_start = rise_dt
            task_stop = task_start + timedelta(seconds=duration_s)
            new_tasks.append(self._make_task(sat_id, task_start, task_stop, telescope, ground_station))
            cursor = task_stop + timedelta(seconds=_INTER_TASK_GAP_S)
            cat = self._satellite_catalog.get(sat_id, {})
            if self.logger:
                self.logger.info(
                    "DummyApiClient: Scheduled %s pass at %s (%.0fs)",
                    cat.get("name", sat_id),
                    task_start.strftime("%H:%M:%S UTC"),
                    duration_s,
                )

        return new_tasks

    def get_ground_station(self, ground_station_id):
        """Fetch ground station details."""
        with self._data_lock:
            if self.logger:
                self.logger.debug(f"DummyApiClient: get_ground_station({ground_station_id})")
            return self._ground_station

    def put_telescope_status(self, body):
        """Report telescope online status."""
        if self.logger:
            self.logger.debug(f"DummyApiClient: put_telescope_status({body})")
        return {"status": "ok"}

    def expand_filters(self, filter_names):
        """Expand filter names to spectral specs."""
        if self.logger:
            self.logger.debug(f"DummyApiClient: expand_filters({filter_names})")
        # Wavelengths match soicat/models/citra/_filter_library.py exactly
        known_filters = {
            "Red": (630.0, 100.0),
            "Green": (530.0, 100.0),
            "Blue": (470.0, 100.0),
            "Clear": (550.0, 300.0),
            "Luminance": (550.0, 300.0),
            "Ha": (656.3, 7.0),
            "Hb": (486.1, 10.0),
            "OIII": (500.7, 10.0),
            "SII": (672.4, 10.0),
            "U": (365.0, 66.0),
            "B": (445.0, 94.0),
            "V": (551.0, 88.0),
            "R": (658.0, 138.0),
            "I": (806.0, 149.0),
            "sloan_u": (354.0, 57.0),
            "sloan_g": (477.0, 137.0),
            "sloan_r": (623.0, 137.0),
            "sloan_i": (763.0, 153.0),
            "sloan_z": (913.0, 95.0),
        }
        filters = []
        for name in filter_names:
            # Case-insensitive lookup to match real API behavior
            match = known_filters.get(name)
            if not match:
                match = next((v for k, v in known_filters.items() if k.lower() == name.lower()), None)
            wavelength, bandwidth = match if match else (550.0, 100.0)
            filters.append(
                {
                    "name": name,
                    "central_wavelength_nm": wavelength,
                    "bandwidth_nm": bandwidth,
                    "is_known": match is not None,
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
            self._ground_station["latitude"] = latitude
            self._ground_station["longitude"] = longitude
            self._ground_station["altitude"] = altitude
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
            telescope = self._ensure_telescope(telescope_id)
            telescope["automatedScheduling"] = enabled
        if self.logger:
            self.logger.info(
                "DummyApiClient: automated scheduling %s for %s",
                "enabled" if enabled else "disabled",
                telescope_id,
            )
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

    def get_catalog_download_url(self, catalog_name: str) -> dict | None:
        """Dummy: no catalog download in test mode."""
        if self.logger:
            self.logger.debug(f"DummyApiClient: get_catalog_download_url({catalog_name}) -> None")
        return None

    def create_batch_collection_requests(
        self,
        window_start: str,
        window_stop: str,
        ground_station_id: str,
        sensor_id: str,
        discover_visible: bool = True,
        satellite_group_ids: list[str] | None = None,
        request_type: str = "Track",
        priority: int = 5,
        exclude_types: list[str] | None = None,
        include_orbit_regimes: list[str] | None = None,
    ) -> dict | None:
        """Stub: log the batch request. Existing auto-task generation handles work in dummy mode."""
        if self.logger:
            self.logger.info(
                f"DummyApiClient: create_batch_collection_requests("
                f"window={window_start} -> {window_stop}, gs={ground_station_id}, "
                f"sensor={sensor_id}, discover={discover_visible}, groups={satellite_group_ids})"
            )
        return {"status": "ok", "created": 0}

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
        if random.random() < self.UPLOAD_FAILURE_RATE:
            if self.logger:
                self.logger.warning(f"DummyApiClient: Simulated mark_complete failure for task {task_id}")
            return False

        with self._data_lock:
            task = self._find_task_across_pools(task_id)
            if task is not None:
                task["status"] = "Succeeded"
                task["updateEpoch"] = datetime.now(timezone.utc).isoformat()
                if self.logger:
                    self.logger.info(f"DummyApiClient: Marked task {task_id} as Succeeded")

        return True

    def mark_task_failed(self, task_id):
        """Mark a task as failed (updates in-memory)."""
        with self._data_lock:
            task = self._find_task_across_pools(task_id)
            if task is not None:
                task["status"] = "Failed"
                task["updateEpoch"] = datetime.now(timezone.utc).isoformat()

            if self.logger:
                self.logger.info(f"DummyApiClient: Marked task {task_id} as Failed")

            return {"status": "Failed"}
