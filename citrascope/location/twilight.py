"""Twilight computation for CitraScope.

Provides two capabilities:

1. **Flat-field windows** — nautical twilight windows (Sun between -6 and -12 deg)
   where sky illumination is suitable for flat-field calibration frames.
2. **Observing windows** — periods when the Sun is below a configurable threshold,
   used by the self-tasking session manager to drive autonomous night operations.

Skyfield timescale and ephemeris objects are cached as module-level singletons
so disk I/O (and a potential first-time download) happens only once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

_skyfield_ts: Any = None
_skyfield_eph: Any = None

CIVIL_DEG = -6.0
NAUTICAL_DEG = -12.0
ASTRONOMICAL_DEG = -18.0


@dataclass(frozen=True)
class FlatWindow:
    """A single twilight flat-field window."""

    start: str
    end: str
    type: str
    remaining_minutes: float | None = None


@dataclass(frozen=True)
class TwilightInfo:
    """Result of a twilight flat-window computation."""

    current_sun_altitude: float
    in_flat_window: bool
    flat_window: FlatWindow | None = None
    next_flat_window: FlatWindow | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (includes ``location_available: True``)."""
        d = asdict(self)
        d["location_available"] = True
        return d


def _get_skyfield_objects() -> tuple[Any, Any]:
    """Return cached Skyfield timescale and ephemeris (lazy-loaded once)."""
    global _skyfield_ts, _skyfield_eph
    if _skyfield_ts is None or _skyfield_eph is None:
        from skyfield.api import load

        _skyfield_ts = load.timescale()
        _skyfield_eph = load("de421.bsp")
    return _skyfield_ts, _skyfield_eph


def compute_sunset_utc(latitude: float, longitude: float) -> datetime | None:
    """Return the most recent or next upcoming sunset as a UTC datetime.

    Searches a 36-hour window centred on 12 hours ago (to catch a sunset
    that already happened this evening).  Returns the sunset closest to
    *now* that is in the past, or the next future one if none has occurred
    yet.  Returns ``None`` if no sunset is found (e.g. polar day).

    Uses the standard refraction-corrected horizon (-0.8333 deg).
    """
    from skyfield import almanac
    from skyfield.api import wgs84

    ts, eph = _get_skyfield_objects()
    topos = wgs84.latlon(latitude, longitude)

    now_utc = datetime.now(timezone.utc)
    t_start = ts.from_datetime(now_utc - timedelta(hours=12))
    t_end = ts.from_datetime(now_utc + timedelta(hours=24))

    times, events = almanac.find_discrete(
        t_start,
        t_end,
        almanac.risings_and_settings(eph, eph["sun"], topos, horizon_degrees=-0.8333),
    )

    sunsets = [t for t, ev in zip(times, events, strict=True) if not ev]
    if not sunsets:
        return None

    now_t = ts.from_datetime(now_utc)
    past = [s for s in sunsets if s.tt <= now_t.tt]
    if past:
        return past[-1].utc_datetime()
    return sunsets[0].utc_datetime()


def compute_twilight(latitude: float, longitude: float) -> TwilightInfo:
    """Compute twilight flat windows for the given observatory location.

    This function is synchronous and potentially expensive (Skyfield almanac
    search over a 36-hour window).  Call via ``asyncio.to_thread`` from async
    contexts to avoid blocking the event loop.

    Args:
        latitude: Observatory latitude in degrees.
        longitude: Observatory longitude in degrees.

    Returns:
        A ``TwilightInfo`` with current sun altitude, whether the flat
        window is active, and the current/next flat windows if any.
    """
    from skyfield import almanac
    from skyfield.api import wgs84

    ts, eph = _get_skyfield_objects()
    topos = wgs84.latlon(latitude, longitude)
    observer = eph["earth"] + topos

    now_utc = datetime.now(timezone.utc)
    t_now = ts.from_datetime(now_utc)
    t_end = ts.from_datetime(now_utc + timedelta(hours=36))

    current_alt = observer.at(t_now).observe(eph["sun"]).apparent().altaz()[0].degrees

    civil_times, civil_events = almanac.find_discrete(
        t_now,
        t_end,
        almanac.risings_and_settings(eph, eph["sun"], topos, horizon_degrees=CIVIL_DEG),
    )
    nautical_times, nautical_events = almanac.find_discrete(
        t_now,
        t_end,
        almanac.risings_and_settings(eph, eph["sun"], topos, horizon_degrees=NAUTICAL_DEG),
    )

    #   Evening: civil set (-6 deg down) → nautical set (-12 deg down)
    #   Morning: nautical rise (-12 deg up) → civil rise (-6 deg up)
    civil_sets = [t.utc_iso() for t, ev in zip(civil_times, civil_events, strict=True) if not ev]
    civil_rises = [t.utc_iso() for t, ev in zip(civil_times, civil_events, strict=True) if ev]
    nautical_sets = [t.utc_iso() for t, ev in zip(nautical_times, nautical_events, strict=True) if not ev]
    nautical_rises = [t.utc_iso() for t, ev in zip(nautical_times, nautical_events, strict=True) if ev]

    raw_windows: list[FlatWindow] = []
    for cs in civil_sets:
        for ns in nautical_sets:
            if ns > cs:
                raw_windows.append(FlatWindow(start=cs, end=ns, type="evening"))
                break
    for nr in nautical_rises:
        for cr in civil_rises:
            if cr > nr:
                raw_windows.append(FlatWindow(start=nr, end=cr, type="morning"))
                break

    raw_windows.sort(key=lambda w: w.start)

    in_flat_window = bool(NAUTICAL_DEG <= current_alt <= CIVIL_DEG)
    current_window: FlatWindow | None = None
    next_window: FlatWindow | None = None
    now_iso = t_now.utc_iso()

    for w in raw_windows:
        if w.start <= now_iso <= w.end:
            end_dt = datetime.fromisoformat(w.end.replace("Z", "+00:00"))
            remaining = (end_dt - now_utc).total_seconds() / 60
            current_window = FlatWindow(
                start=w.start,
                end=w.end,
                type=w.type,
                remaining_minutes=round(max(remaining, 0), 1),
            )
        elif w.start > now_iso and next_window is None:
            next_window = w

    return TwilightInfo(
        current_sun_altitude=round(float(current_alt), 1),
        in_flat_window=in_flat_window,
        flat_window=current_window,
        next_flat_window=next_window,
    )


@dataclass(frozen=True)
class ObservingWindow:
    """Result of an observing-window computation.

    Used by the self-tasking session manager to determine whether it is
    currently dark enough to observe and to provide the dark-period bounds
    as ``windowStart``/``windowStop`` for the batch collection-request API.
    """

    is_dark: bool
    current_sun_altitude: float
    dark_start: str | None = None  # ISO-8601 UTC
    dark_end: str | None = None  # ISO-8601 UTC


def compute_observing_window(
    latitude: float,
    longitude: float,
    sun_altitude_threshold: float = NAUTICAL_DEG,
) -> ObservingWindow:
    """Compute whether it is dark and the bounds of the current dark period.

    "Dark" means the Sun is below *sun_altitude_threshold* (default -12 deg,
    nautical twilight — the standard for satellite tracking).

    The function searches a 36-hour window centred on 12 hours ago so it can
    find the threshold crossing that started the current night even if it
    already happened.

    Args:
        latitude: Observatory latitude in degrees.
        longitude: Observatory longitude in degrees.
        sun_altitude_threshold: Sun altitude in degrees below which it is
            considered dark.  Standard values: -6 (civil), -12 (nautical),
            -18 (astronomical).

    Returns:
        An ``ObservingWindow`` with the current sun altitude, whether it is
        dark, and the start/end of the dark period (if applicable).
    """
    from skyfield import almanac
    from skyfield.api import wgs84

    ts, eph = _get_skyfield_objects()
    topos = wgs84.latlon(latitude, longitude)
    observer = eph["earth"] + topos

    now_utc = datetime.now(timezone.utc)
    t_now = ts.from_datetime(now_utc)

    current_alt = float(observer.at(t_now).observe(eph["sun"]).apparent().altaz()[0].degrees)
    is_dark = current_alt < sun_altitude_threshold

    # Search a wide window to find the threshold crossings bounding "now".
    t_start = ts.from_datetime(now_utc - timedelta(hours=12))
    t_end = ts.from_datetime(now_utc + timedelta(hours=24))

    times, events = almanac.find_discrete(
        t_start,
        t_end,
        almanac.risings_and_settings(eph, eph["sun"], topos, horizon_degrees=sun_altitude_threshold),
    )

    if not is_dark:
        next_dark_start: str | None = None
        for t, ev in zip(times, events, strict=True):
            if not ev and t.utc_datetime() > now_utc:
                next_dark_start = t.utc_iso()
                break
        return ObservingWindow(
            is_dark=False,
            current_sun_altitude=round(current_alt, 2),
            dark_start=next_dark_start,
        )

    # Sun is below threshold — find the bounding set (start) and rise (end).
    dark_start: str | None = None
    dark_end: str | None = None

    for t, ev in zip(times, events, strict=True):
        t_dt = t.utc_datetime()
        if not ev and t_dt <= now_utc:
            # "setting" (sun dropping below threshold) before now — latest wins
            dark_start = t.utc_iso()
        elif ev and t_dt > now_utc and dark_end is None:
            # "rising" (sun climbing above threshold) after now — first wins
            dark_end = t.utc_iso()

    return ObservingWindow(
        is_dark=True,
        current_sun_altitude=round(current_alt, 2),
        dark_start=dark_start,
        dark_end=dark_end,
    )
