"""Single source of task dicts for the web/wire layer, with sky enrichment.

The monitoring page renders tasks with derived sky-position fields (alt, az,
compass label, rising/setting trend, peak altitude over the window, plus the
slew distance from the scope's current pointing).  Two things motivated this
module ending up the way it has:

1. **Single emitter.**  Both ``GET /api/tasks`` and the periodic WebSocket
   broadcast in :class:`citrascope.web.app.CitraScopeWebApp` need the same
   wire format.  When they each built their own list of dicts, one of them
   forgot to enrich and only the first batch of tasks (from the HTTP fetch)
   showed sky data on the client.  All emitters now go through
   :func:`get_web_tasks`, so the wire format can never drift again.

2. **The static fields are time-invariant.**  ``sky_alt_deg`` etc are a pure
   function of ``(satellite, observer, task.start_time)``.  Start-time is
   fixed on the task, the observer doesn't move, and the satellite's TLE is
   stable for hours.  Recomputing them on every emission was correct but
   wasteful; we now memoize them per ``task.id`` and only recompute when
   the task's signature changes (re-scheduled) or the observer moves.  Only
   :data:`slew_from_current_deg` is genuinely time-varying (the scope moves)
   and is cheap enough to recompute on every emission as a great-circle on
   cached topocentric coordinates -- no propagator call required.

Propagation uses keplemon (SGP4 + SGP4-XP).  Alt/az comes from converting
the J2000 topocentric RA/Dec via :func:`radec_to_altaz`, which is a pure
math helper shared with the mount pointing model.

Best-effort throughout.  Missing TLE, observer, or scope pointing simply
omits the relevant fields rather than raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import asin, cos, degrees, radians, sin
from typing import Any

from keplemon import time as ktime
from keplemon.bodies import Observatory, Satellite
from keplemon.elements import TLE
from keplemon.enums import ReferenceFrame

from citrascope.astro.sidereal import gast_degrees, make_observatory
from citrascope.hardware.devices.mount.altaz_pointing_model import radec_to_altaz
from citrascope.web.helpers import _task_to_dict

_logger = logging.getLogger("citrascope.SkyEnrichment")

# Number of altitude samples drawn across [task_start, task_stop] when
# computing peak altitude and the rising/setting trend.  Twelve is dense
# enough to find the peak of a LEO arc reliably without slowing the cold
# path appreciably.
_TRACK_SAMPLES = 12

# 16-point compass labels, indexed by floor((az + 11.25) / 22.5) % 16.
_COMPASS_16 = [
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
]

# Trend classification threshold (degrees of altitude change between the
# first two samples).  Smaller deltas -- GEOs, near-zenith passes -- are
# reported as "flat".
_TREND_DEADBAND_DEG = 0.05

# Static field names emitted onto each task dict.  Kept here so the memo and
# the consumers can never drift.
_STATIC_FIELDS = ("sky_alt_deg", "sky_az_deg", "sky_compass", "sky_trend", "sky_max_alt_deg")


# ---------------------------------------------------------------------------
# Memo state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StaticSky:
    """Cached time-invariant sky data for a single task.

    ``signature`` lets us detect when a task has been re-scheduled (start or
    stop time changed) so we recompute instead of serving stale numbers.
    ``target_ra_deg`` / ``target_dec_deg`` are the topocentric RA/Dec of the
    satellite at ``task.start_time``; we keep them so slew distance can be
    recomputed via a pure great-circle without re-entering the propagator.
    """

    signature: tuple[str, str, str | None]
    fields: dict[str, Any]
    target_ra_deg: float | None
    target_dec_deg: float | None


# Per-task cache of the invariant fields.  Bounded by the queue length;
# evicted on every call for tasks no longer in the snapshot.
_SKY_MEMO: dict[str, _StaticSky] = {}

# Observer signature the memo was built against.  When the ground station
# moves (or the location service first becomes available), the entire memo
# is invalidated -- topocentric coords are observer-relative.
_OBSERVER_SIG: tuple[float, float, float] | None = None


def _clear_memo_for_tests() -> None:
    """Reset module state.  Test-only entry point."""
    global _OBSERVER_SIG
    _SKY_MEMO.clear()
    _OBSERVER_SIG = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _compass_16pt(az_deg: float) -> str:
    """Map azimuth (0..360, North=0) to a 16-point compass label."""
    idx = int((az_deg + 11.25) % 360 / 22.5)
    return _COMPASS_16[idx]


def _angular_distance_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """Great-circle distance between two RA/Dec points, in degrees."""
    ra1, dec1 = radians(ra1_deg), radians(dec1_deg)
    ra2, dec2 = radians(ra2_deg), radians(dec2_deg)
    dlat = dec2 - dec1
    dlon = ra2 - ra1
    a = sin(dlat / 2) ** 2 + cos(dec1) * cos(dec2) * sin(dlon / 2) ** 2
    return degrees(2 * asin(min(1.0, a**0.5)))


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _observer_signature(location: dict) -> tuple[float, float, float] | None:
    try:
        return (
            float(location["latitude"]),
            float(location["longitude"]),
            float(location.get("altitude") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _build_tle_lookup(elset_cache: Any, wanted_ids: set[str]) -> dict[str, list[str]]:
    """Pull just the TLEs we actually need out of the elset cache.

    The cache typically holds tens of thousands of elsets, but the scheduled
    queue is ~10 tasks.  Filtering up front keeps the per-call work bounded.
    """
    if not wanted_ids:
        return {}
    try:
        elsets = elset_cache.get_elsets()
    except Exception as exc:
        _logger.debug("ElsetCache.get_elsets() failed: %s", exc)
        return {}

    out: dict[str, list[str]] = {}
    for entry in elsets:
        sat_id = entry.get("satellite_id")
        if sat_id in wanted_ids and "tle" in entry and len(entry["tle"]) >= 2:
            out[sat_id] = entry["tle"]
    return out


def _propagate_static(
    satellite: Satellite,
    observer: Observatory,
    obs_lat_deg: float,
    obs_lon_deg: float,
    start_dt: datetime,
    stop_dt: datetime | None,
) -> tuple[dict[str, Any], float, float] | None:
    """Propagate once for a task and return ``(static_fields, ra_deg, dec_deg)``.

    RA/Dec comes from keplemon in J2000; alt/az is derived via the shared
    :func:`radec_to_altaz` helper.  Returns ``None`` on any propagation
    failure -- callers treat that as "skip this task" rather than letting it
    raise into the route handler.
    """
    end_dt = stop_dt if (stop_dt is not None and stop_dt > start_dt) else start_dt
    span_s = max(0.0, (end_dt - start_dt).total_seconds())

    try:
        alts: list[float] = []
        azs: list[float] = []
        target_ra_deg: float | None = None
        target_dec_deg: float | None = None
        n = _TRACK_SAMPLES if span_s > 0 else 1
        for i in range(n):
            frac = i / (n - 1) if n > 1 else 0.0
            sample_dt = start_dt + timedelta(seconds=span_s * frac)
            epoch = ktime.Epoch.from_datetime(sample_dt)
            topo = observer.get_topocentric_to_satellite(epoch, satellite, ReferenceFrame.J2000)
            ra = float(topo.right_ascension)
            dec = float(topo.declination)
            if i == 0:
                target_ra_deg = ra
                target_dec_deg = dec
            # GAST must match the propagation epoch, not wall-clock now —
            # scheduled tasks can be hours in the future, and without this
            # the alt/az would be rotated by ~15°/hr (i.e. a task 4 hours
            # out looks horizontal-mirrored if we silently use "now"-GAST).
            gast_deg = gast_degrees(sample_dt)
            az, alt = radec_to_altaz(ra, dec, obs_lat_deg, obs_lon_deg, _gast_override=gast_deg)
            alts.append(alt)
            azs.append(az)
    except Exception as exc:
        _logger.debug("Sky propagation failed: %s", exc)
        return None

    if target_ra_deg is None or target_dec_deg is None:
        return None

    if not alts:
        return None

    if len(alts) >= 2:
        delta = alts[1] - alts[0]
        if delta > _TREND_DEADBAND_DEG:
            trend = "rising"
        elif delta < -_TREND_DEADBAND_DEG:
            trend = "setting"
        else:
            trend = "flat"
    else:
        trend = "flat"

    fields: dict[str, Any] = {
        "sky_alt_deg": round(alts[0], 2),
        "sky_az_deg": round(azs[0], 2),
        "sky_compass": _compass_16pt(azs[0]),
        "sky_trend": trend,
        "sky_max_alt_deg": round(max(alts), 2),
    }
    return fields, target_ra_deg, target_dec_deg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_web_tasks(
    daemon: Any,
    status: Any,
    *,
    exclude_active: bool = False,
) -> list[dict]:
    """The single source of task dicts for the web/wire layer.

    Both ``GET /api/tasks`` and ``WebApp.broadcast_tasks`` go through here,
    so the shape of a task on the wire is defined in exactly one place.

    Sky enrichment uses a per-task memo keyed by ``task.id``; the static
    fields (alt/az/compass/trend/peak) are computed once on first sight and
    reused on every subsequent emission.  Only the slew-from-current-pointing
    is recomputed each time, since the scope moves.  Memo entries for tasks
    no longer in the queue are evicted at the top of every call.

    Returns ``[]`` when the daemon or task manager isn't yet ready.
    """
    if not daemon or not getattr(daemon, "task_manager", None):
        return []

    tasks = [_task_to_dict(t) for t in daemon.task_manager.get_tasks_snapshot(exclude_active=exclude_active)]
    _enrich_tasks(tasks, daemon=daemon, status=status)
    return tasks


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gather_location(daemon: Any) -> dict | None:
    """Read the observer's location from the daemon, swallowing failures."""
    if not getattr(daemon, "location_service", None):
        return None
    try:
        return daemon.location_service.get_current_location()
    except Exception as exc:
        _logger.debug("location_service.get_current_location failed: %s", exc)
        return None


def _enrich_tasks(tasks: list[dict], *, daemon: Any, status: Any) -> None:
    """Mutate each task dict with sky fields, using the per-task memo.

    Three phases:
      1. Drop memo entries for tasks no longer in the snapshot.
      2. For each task, paste cached static fields if the memo is fresh,
         otherwise propagate via keplemon and seed the memo.
      3. For each task with cached topocentric coords AND a known scope
         pointing, compute ``slew_from_current_deg`` via a great-circle
         (no propagation needed -- microseconds).
    """
    global _OBSERVER_SIG

    if not tasks:
        # Nothing to do, but still GC the memo so it doesn't leak across
        # the (unlikely) edge of "queue empties out".
        _SKY_MEMO.clear()
        return

    location = _gather_location(daemon)
    if not location:
        return

    obs_sig = _observer_signature(location)
    if obs_sig is None:
        return

    if obs_sig != _OBSERVER_SIG:
        _SKY_MEMO.clear()
        _OBSERVER_SIG = obs_sig

    # Phase 1 -- evict stale memo entries.
    live_ids = {t.get("id") for t in tasks if t.get("id")}
    for stale_id in [tid for tid in _SKY_MEMO if tid not in live_ids]:
        del _SKY_MEMO[stale_id]

    # Phase 2 -- paste from memo where possible, otherwise propagate via keplemon.
    needs_compute = [t for t in tasks if not _try_paste_from_memo(t)]

    if needs_compute:
        elset_cache = getattr(daemon, "elset_cache", None)
        if elset_cache is not None:
            try:
                observer = make_observatory(obs_sig[0], obs_sig[1], obs_sig[2])
            except (TypeError, ValueError) as exc:
                _logger.debug("Observatory construction failed for %s: %s", obs_sig, exc)
                observer = None

            if observer is not None:
                _populate_via_propagation(
                    needs_compute,
                    observer=observer,
                    obs_lat_deg=obs_sig[0],
                    obs_lon_deg=obs_sig[1],
                    elset_cache=elset_cache,
                )

    # Phase 3 -- recompute slew distance from cached target coords.
    scope_ra = getattr(status, "telescope_ra", None)
    scope_dec = getattr(status, "telescope_dec", None)
    if scope_ra is not None and scope_dec is not None:
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            entry = _SKY_MEMO.get(task_id)
            if entry and entry.target_ra_deg is not None and entry.target_dec_deg is not None:
                task["slew_from_current_deg"] = round(
                    _angular_distance_deg(scope_ra, scope_dec, entry.target_ra_deg, entry.target_dec_deg),
                    2,
                )


def _try_paste_from_memo(task: dict) -> bool:
    """If the memo has fresh static fields for this task, paste them on."""
    task_id = task.get("id")
    if not task_id:
        return False
    entry = _SKY_MEMO.get(task_id)
    if entry is None:
        return False
    if entry.signature != _task_signature(task):
        return False
    task.update(entry.fields)
    return True


def _task_signature(task: dict) -> tuple[str, str, str | None]:
    """Signature used to detect when a task's scheduling has changed.

    Includes ``satelliteId`` so a task being repointed (rare, but possible
    via re-scheduling) invalidates the cache.
    """
    return (
        str(task.get("satelliteId") or ""),
        str(task.get("start_time") or ""),
        str(task.get("stop_time")) if task.get("stop_time") else None,
    )


def _populate_via_propagation(
    tasks: list[dict],
    *,
    observer: Observatory,
    obs_lat_deg: float,
    obs_lon_deg: float,
    elset_cache: Any,
) -> None:
    """Fill static fields for tasks that missed the memo, seeding the memo."""
    wanted_ids = {str(t.get("satelliteId")) for t in tasks if t.get("satelliteId")}
    tle_by_id = _build_tle_lookup(elset_cache, wanted_ids)
    if not tle_by_id:
        return

    sat_cache: dict[str, Satellite] = {}
    for task in tasks:
        sat_id = task.get("satelliteId")
        if not sat_id or sat_id not in tle_by_id:
            continue

        start_dt = _parse_iso(task.get("start_time"))
        if start_dt is None:
            continue
        stop_dt = _parse_iso(task.get("stop_time"))

        try:
            sat = sat_cache.get(sat_id)
            if sat is None:
                tle = tle_by_id[sat_id]
                sat = Satellite.from_tle(TLE.from_lines(tle[0], tle[1]))
                sat_cache[sat_id] = sat
        except Exception as exc:
            _logger.debug("Satellite construction failed for %s: %s", sat_id, exc)
            continue

        result = _propagate_static(sat, observer, obs_lat_deg, obs_lon_deg, start_dt, stop_dt)
        if result is None:
            continue
        fields, target_ra, target_dec = result
        task.update(fields)

        task_id = task.get("id")
        if task_id:
            _SKY_MEMO[task_id] = _StaticSky(
                signature=_task_signature(task),
                fields=fields,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
            )
