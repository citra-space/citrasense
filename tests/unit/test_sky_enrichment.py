"""Unit tests for sky enrichment of web/wire task payloads.

Tests target two layers:

1. ``_enrich_tasks`` -- the worker that mutates task dicts in place.  Covers
   the happy path, all degraded paths (missing TLE / observer / scope
   pointing / empty queue), and the memo behavior that lets static fields
   be computed once per task.id and reused on every emission.

2. ``get_web_tasks`` -- the single public emitter.  Verified end-to-end with
   a stub daemon to confirm both call sites (HTTP route + WebSocket
   broadcaster) receive identical, fully enriched dicts.

We use a real (historical) ISS TLE so Skyfield actually propagates.  Exact
alt/az numbers aren't asserted (they drift with Earth orientation
parameters), only structural validity and invariants like "max_alt >= alt".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from citrasense.web.sky_enrichment import (
    _SKY_MEMO,
    _angular_distance_deg,
    _clear_memo_for_tests,
    _compass_16pt,
    _enrich_tasks,
    get_web_tasks,
)

# A historical ISS TLE.  Old enough that Skyfield will warn about staleness on
# propagation, but still produces a valid Earth-orbiting position -- which is
# all we need to assert that enrichment fields are populated.
_ISS_TLE = [
    "1 25544U 98067A   24001.50000000  .00012000  00000-0  22000-3 0  9991",
    "2 25544  51.6400 297.7500 0006000  90.0000 270.0000 15.50000000400000",
]

# Pikes Peak, the dummy adapter's default ground station -- gives the ISS a
# realistic chance of being above the horizon at any moment.
_OBSERVER = {"latitude": 38.8409, "longitude": -105.0423, "altitude": 4302}
_OBSERVER_ALT = {"latitude": 32.0, "longitude": -110.0, "altitude": 800}


@pytest.fixture(autouse=True)
def _reset_memo():
    """Each test gets a clean memo so cached state doesn't leak across cases."""
    _clear_memo_for_tests()
    yield
    _clear_memo_for_tests()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubElsetCache:
    """Minimal stand-in for ElsetCache that returns a fixed elset list.

    Counts ``get_elsets`` calls so memo-hit tests can assert that we don't
    keep paying for Skyfield work on tasks we've already seen.
    """

    def __init__(self, elsets: list[dict]) -> None:
        self._elsets = elsets
        self.calls = 0

    def get_elsets(self) -> list[dict]:
        self.calls += 1
        return list(self._elsets)


class _StubLocationService:
    def __init__(self, location: dict | None) -> None:
        self._location = location

    def get_current_location(self) -> dict | None:
        return self._location


def _task(sat_id: str, start_offset_s: int = 60, duration_s: int = 30, *, task_id: str | None = None) -> dict:
    """Build a minimal task dict matching the /api/tasks payload shape."""
    now = datetime.now(timezone.utc)
    start = now + timedelta(seconds=start_offset_s)
    stop = start + timedelta(seconds=duration_s)
    return {
        "id": task_id or ("task-" + sat_id),
        "satelliteId": sat_id,
        "target": "Test Sat",
        "start_time": start.isoformat(),
        "stop_time": stop.isoformat(),
        "status": "Pending",
        "filter": "Red",
    }


def _make_daemon(*, tasks: list[Any], location: dict | None, elsets: list[dict]) -> Any:
    """Stub daemon exposing just the surface get_web_tasks reads."""
    cache = _StubElsetCache(elsets)
    task_manager = SimpleNamespace(get_tasks_snapshot=lambda exclude_active=False: list(tasks))
    return SimpleNamespace(
        task_manager=task_manager,
        elset_cache=cache,
        location_service=_StubLocationService(location),
    )


def _make_status(scope_ra: float | None = 180.0, scope_dec: float | None = 45.0) -> Any:
    return SimpleNamespace(telescope_ra=scope_ra, telescope_dec=scope_dec)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestCompass16pt:
    def test_cardinal_directions(self):
        assert _compass_16pt(0) == "N"
        assert _compass_16pt(90) == "E"
        assert _compass_16pt(180) == "S"
        assert _compass_16pt(270) == "W"

    def test_intermediate(self):
        assert _compass_16pt(202.5) == "SSW"
        assert _compass_16pt(45) == "NE"

    def test_wraparound(self):
        assert _compass_16pt(359.9) == "N"
        assert _compass_16pt(360.0) == "N"


class TestAngularDistance:
    def test_zero_distance_to_self(self):
        assert _angular_distance_deg(180.0, 30.0, 180.0, 30.0) == 0.0

    def test_one_degree_along_equator(self):
        d = _angular_distance_deg(180.0, 0.0, 181.0, 0.0)
        assert abs(d - 1.0) < 1e-6

    def test_antipodal_points(self):
        d = _angular_distance_deg(0.0, 0.0, 180.0, 0.0)
        assert abs(d - 180.0) < 1e-6


# ---------------------------------------------------------------------------
# _enrich_tasks -- happy path
# ---------------------------------------------------------------------------


def test_happy_path_populates_all_sky_fields():
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    tasks = [_task("sat-25544")]

    _enrich_tasks(tasks, daemon=daemon, status=_make_status())

    t = tasks[0]
    assert isinstance(t["sky_alt_deg"], float)
    assert -90.0 <= t["sky_alt_deg"] <= 90.0
    assert isinstance(t["sky_az_deg"], float)
    assert 0.0 <= t["sky_az_deg"] <= 360.0
    assert t["sky_compass"] in {
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    }
    assert t["sky_trend"] in {"rising", "setting", "flat"}
    assert isinstance(t["sky_max_alt_deg"], float)
    assert t["sky_max_alt_deg"] >= t["sky_alt_deg"] - 0.01
    assert isinstance(t["slew_from_current_deg"], float)
    assert 0.0 <= t["slew_from_current_deg"] <= 180.0


def test_omits_slew_when_scope_pointing_unknown():
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    tasks = [_task("sat-25544")]

    _enrich_tasks(tasks, daemon=daemon, status=_make_status(scope_ra=None, scope_dec=None))

    assert "sky_alt_deg" in tasks[0]
    assert "slew_from_current_deg" not in tasks[0]


# ---------------------------------------------------------------------------
# _enrich_tasks -- degraded paths
# ---------------------------------------------------------------------------


def test_missing_tle_for_satellite_is_silent_skip():
    daemon = _make_daemon(tasks=[], location=_OBSERVER, elsets=[])
    tasks = [_task("sat-99999")]

    _enrich_tasks(tasks, daemon=daemon, status=_make_status())

    assert "sky_alt_deg" not in tasks[0]
    assert "sky_az_deg" not in tasks[0]


def test_missing_location_is_no_op():
    daemon = _make_daemon(
        tasks=[],
        location=None,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    tasks = [_task("sat-25544")]

    _enrich_tasks(tasks, daemon=daemon, status=_make_status())

    assert "sky_alt_deg" not in tasks[0]


def test_empty_task_list_is_no_op_and_clears_memo():
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    # Seed the memo with one entry, then call with [] -- everything should drop.
    tasks = [_task("sat-25544")]
    _enrich_tasks(tasks, daemon=daemon, status=_make_status())
    assert _SKY_MEMO

    _enrich_tasks([], daemon=daemon, status=_make_status())
    assert not _SKY_MEMO


def test_task_without_satellite_id_is_skipped():
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    tasks = [{"id": "x", "start_time": datetime.now(timezone.utc).isoformat(), "stop_time": None}]

    _enrich_tasks(tasks, daemon=daemon, status=_make_status(scope_ra=None, scope_dec=None))

    assert "sky_alt_deg" not in tasks[0]


# ---------------------------------------------------------------------------
# Memo behavior -- the whole point of this refactor
# ---------------------------------------------------------------------------


def test_second_emission_does_not_repropagate():
    """A task seen twice should hit the memo on the second pass."""
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )

    tasks_a = [_task("sat-25544", task_id="task-iss")]
    _enrich_tasks(tasks_a, daemon=daemon, status=_make_status())
    first_calls = daemon.elset_cache.calls
    cached_alt = tasks_a[0]["sky_alt_deg"]

    tasks_b = [_task("sat-25544", task_id="task-iss")]
    # Mutate one of the times to confirm we DON'T just rebuild.  (Same signature
    # because we re-use _task() which generates fresh times -- so we manually
    # paste the original times to exercise the memo hit.)
    tasks_b[0]["start_time"] = tasks_a[0]["start_time"]
    tasks_b[0]["stop_time"] = tasks_a[0]["stop_time"]

    _enrich_tasks(tasks_b, daemon=daemon, status=_make_status())

    # No additional elset_cache hit -- propagation skipped entirely.
    assert daemon.elset_cache.calls == first_calls
    # And the cached static fields were pasted onto the new dict.
    assert tasks_b[0]["sky_alt_deg"] == cached_alt


def test_signature_change_invalidates_memo():
    """Re-scheduling a task (start_time changes) should force recompute."""
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )

    t1 = _task("sat-25544", task_id="task-iss", start_offset_s=60)
    _enrich_tasks([t1], daemon=daemon, status=_make_status())
    calls_after_first = daemon.elset_cache.calls

    # Same task.id, different start time.
    t2 = _task("sat-25544", task_id="task-iss", start_offset_s=120)
    _enrich_tasks([t2], daemon=daemon, status=_make_status())

    # Memo miss -> elset cache touched again.
    assert daemon.elset_cache.calls > calls_after_first


def test_observer_change_invalidates_entire_memo():
    """Moving the ground station should drop all cached entries -- topocentric
    coordinates are observer-relative."""
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    t = _task("sat-25544", task_id="task-iss")
    _enrich_tasks([t], daemon=daemon, status=_make_status())
    assert _SKY_MEMO

    # Same task, but the daemon now reports a new observer location.
    daemon.location_service = _StubLocationService(_OBSERVER_ALT)
    t2 = dict(t)  # same signature
    _enrich_tasks([t2], daemon=daemon, status=_make_status())

    # Memo was wiped on observer change, then repopulated for this task.
    assert len(_SKY_MEMO) == 1


def test_slew_updates_even_when_static_fields_cached():
    """The whole point of the memo split: slew is recomputed every call."""
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    t1 = _task("sat-25544", task_id="task-iss")
    _enrich_tasks([t1], daemon=daemon, status=_make_status(scope_ra=180.0, scope_dec=45.0))
    slew_a = t1["slew_from_current_deg"]

    # Same task, different scope pointing -> slew must change while static
    # fields stay identical (they came from the memo).
    t2 = dict(t1)  # preserves signature
    for k in ("sky_alt_deg", "sky_az_deg", "sky_compass", "sky_trend", "sky_max_alt_deg", "slew_from_current_deg"):
        t2.pop(k, None)
    _enrich_tasks([t2], daemon=daemon, status=_make_status(scope_ra=0.0, scope_dec=0.0))

    assert t2["sky_alt_deg"] == t1["sky_alt_deg"]  # static cached
    assert t2["slew_from_current_deg"] != slew_a  # live recomputed


def test_evicts_memo_for_tasks_no_longer_in_queue():
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[
            {"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE},
        ],
    )
    t_a = _task("sat-25544", task_id="task-a")
    t_b = _task("sat-25544", task_id="task-b")
    _enrich_tasks([t_a, t_b], daemon=daemon, status=_make_status())
    assert {"task-a", "task-b"}.issubset(_SKY_MEMO.keys())

    # Next snapshot drops task-a -- its memo entry should go too.
    _enrich_tasks([t_b], daemon=daemon, status=_make_status())
    assert "task-a" not in _SKY_MEMO
    assert "task-b" in _SKY_MEMO


# ---------------------------------------------------------------------------
# get_web_tasks -- end-to-end with a stub daemon
# ---------------------------------------------------------------------------


def test_get_web_tasks_returns_empty_when_daemon_not_ready():
    assert get_web_tasks(None, _make_status()) == []
    assert get_web_tasks(SimpleNamespace(task_manager=None), _make_status()) == []


def test_propagate_static_passes_per_sample_gast_override(monkeypatch):
    """Regression for PR #301 Copilot comment: ``radec_to_altaz`` must be called
    with ``_gast_override`` derived from the same epoch used for propagation,
    not ``datetime.now()``.

    With the bug, ``radec_to_altaz`` is called without ``_gast_override``; its
    default path reads wall-clock "now" for GAST regardless of ``sample_dt``.
    For tasks hours in the future, that's ~15°/hr of alt/az rotation wrong.

    We spy on the helper at the module boundary and check two things:

    1. Every call passes ``_gast_override`` (i.e. the kwarg is present and
       not None).
    2. The override values span a range consistent with sampling across the
       task window — i.e. they are per-sample, not frozen to one moment.
    """
    import citrasense.web.sky_enrichment as se

    captured_kwargs: list[dict] = []
    real = se.radec_to_altaz

    def spy(ra, dec, lat, lon, *, _gast_override=None):
        captured_kwargs.append({"_gast_override": _gast_override})
        return real(ra, dec, lat, lon, _gast_override=_gast_override)

    monkeypatch.setattr(se, "radec_to_altaz", spy)

    # Task far enough in the future, with a wide enough window, that the
    # per-sample GAST values differ noticeably (Earth turns at ~0.0042°/s,
    # so a 30-minute window gives ~7.5° of spread — well above any
    # numerical noise).
    daemon = _make_daemon(
        tasks=[],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )
    now = datetime.now(timezone.utc)
    task = {
        "id": "task-future",
        "satelliteId": "sat-25544",
        "target": "ISS future",
        "start_time": (now + timedelta(hours=6)).isoformat(),
        "stop_time": (now + timedelta(hours=6, minutes=30)).isoformat(),
        "status": "Pending",
        "filter": "Red",
    }

    _enrich_tasks([task], daemon=daemon, status=_make_status())

    assert captured_kwargs, "_propagate_static never called radec_to_altaz"
    # Every call must pass a concrete GAST; None would mean we fell back to
    # datetime.now() inside gast_degrees().
    for call in captured_kwargs:
        override = call["_gast_override"]
        assert override is not None, "radec_to_altaz called without _gast_override (bug)"
        assert isinstance(override, float)

    # Per-sample: the override values must span the task window, not all be
    # identical. 30 minutes of Earth rotation is ~7.5° of GAST drift; require
    # at least 1° of spread to comfortably exclude a single-value bug.
    overrides = [c["_gast_override"] for c in captured_kwargs]
    assert max(overrides) - min(overrides) > 1.0, (
        f"_gast_override values barely changed across samples ({overrides!r}) — "
        "this is exactly the PR #301 bug where GAST was frozen at 'now'."
    )


def test_get_web_tasks_emits_dicts_with_sky_fields():
    """End-to-end: stub daemon -> get_web_tasks -> sky-enriched dicts.

    Acts as a regression test for the bug where the WebSocket broadcaster
    bypassed enrichment.  Both call sites now go through this path, so if
    this test passes, both sites produce the same shape.
    """
    # Task domain object that satisfies _task_to_dict's attribute reads.
    now = datetime.now(timezone.utc)
    fake_task = SimpleNamespace(
        id="task-iss",
        satelliteId="sat-25544",
        taskStart=(now + timedelta(seconds=60)).isoformat(),
        taskStop=(now + timedelta(seconds=90)).isoformat(),
        status="Pending",
        satelliteName="ISS",
        assigned_filter_name="Red",
    )
    daemon = _make_daemon(
        tasks=[fake_task],
        location=_OBSERVER,
        elsets=[{"satellite_id": "sat-25544", "name": "ISS", "tle": _ISS_TLE}],
    )

    result = get_web_tasks(daemon, _make_status())

    assert len(result) == 1
    t = result[0]
    assert t["id"] == "task-iss"
    assert t["satelliteId"] == "sat-25544"
    assert "sky_alt_deg" in t
    assert "sky_compass" in t
    assert "slew_from_current_deg" in t
