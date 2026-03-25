"""Reconstruct a ProcessingContext from a saved debug directory.

When ``keep_processing_output`` is enabled in citrascope settings, each
processed task retains a directory under ``processing/<task_id>/`` containing
JSON artifacts and FITS images that fully describe the pipeline inputs.

This module reads those artifacts and rebuilds a :class:`ProcessingContext`
that can be fed back into :meth:`ProcessorRegistry.process_all` — allowing
pipeline developers to iterate on processor code and re-run against real
captured data without needing live hardware.

Debug directory layout (files read by this loader)
---------------------------------------------------
Required:
    ``task.json``                      Task metadata (id, type, satellite, filters, etc.)
    ``observer_location.json``         Lat/lon/alt used for TLE propagation
    ``telescope_record.json``          Telescope hardware config from the Citra API
    A ``.fits`` file                   The captured image (see FITS discovery below)

Optional (gracefully skipped if absent):
    ``elset_cache_snapshot.json``      TLE catalog at processing time
    ``target_satellite.json``          Satellite record from the Citra API
    ``pointing_report.json``           Iterative slew convergence telemetry
    ``satellite_matcher_debug.json``   Satellite matcher diagnostics (source of tracking_mode)

FITS image discovery priority:
    1. ``original_*.fits``   — raw backup (best for full pipeline replay)
    2. Any ``.fits`` that is NOT ``calibrated.fits`` or ``*_wcs.fits``
    3. ``calibrated.fits`` or any remaining ``.fits``

The selected FITS is **copied** into the output working directory so that
processors (especially the plate solver, which writes ``*_wcs.fits`` next to
the input) never mutate the original debug capture.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from citrascope.elset_cache import ElsetCache
from citrascope.processors.processor_result import ProcessingContext
from citrascope.tasks.task import Task

logger = logging.getLogger("citrascope.context_loader")


class FixedLocationService:
    """Stub LocationService that returns a static location from a saved artifact."""

    def __init__(self, location: dict[str, Any]):
        self._location = location

    def get_current_location(self) -> dict[str, Any]:
        return self._location


def _load_json(path: Path) -> Any:
    """Load and return parsed JSON from *path*, or None if missing/corrupt."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path.name, exc)
        return None


def _discover_fits(debug_dir: Path) -> Path | None:
    """Find the best FITS file in the debug directory for reprocessing.

    Priority:
        1. ``original_*.fits`` (raw backup from ProcessingQueue)
        2. Any .fits that isn't calibrated.fits or *_wcs.fits
        3. calibrated.fits or any remaining .fits
    """
    all_fits = sorted(debug_dir.glob("*.fits"))
    if not all_fits:
        return None

    # Tier 1: raw backup
    originals = [f for f in all_fits if f.name.startswith("original_")]
    if originals:
        return originals[0]

    # Tier 2: non-calibrated, non-WCS
    middle = [f for f in all_fits if f.name != "calibrated.fits" and not f.name.endswith("_wcs.fits")]
    if middle:
        return middle[0]

    # Tier 3: anything
    return all_fits[0]


def _task_from_saved_dict(data: dict[str, Any]) -> Task:
    """Construct a Task from the field names used by artifact_writer._task_to_dict.

    The saved JSON uses dataclass field names (e.g. ``assigned_filter_name``),
    not the camelCase API keys that ``Task.from_dict()`` expects.
    """
    return Task(
        id=str(data.get("id", "")),
        type=data.get("type", ""),
        status=str(data.get("status", "")),
        creationEpoch=data.get("creationEpoch", ""),
        updateEpoch=data.get("updateEpoch", ""),
        taskStart=data.get("taskStart", ""),
        taskStop=data.get("taskStop", ""),
        userId=data.get("userId", ""),
        username=data.get("username", ""),
        satelliteId=data.get("satelliteId", ""),
        satelliteName=data.get("satelliteName", ""),
        telescopeId=data.get("telescopeId", ""),
        telescopeName=data.get("telescopeName", ""),
        groundStationId=data.get("groundStationId", ""),
        groundStationName=data.get("groundStationName", ""),
        assigned_filter_name=data.get("assigned_filter_name"),
    )


def load_context_from_debug_dir(
    debug_dir: Path,
    output_dir: Path,
    settings: Any,
    log: logging.Logger | None = None,
    image_override: Path | None = None,
) -> ProcessingContext:
    """Reconstruct a ProcessingContext from a retained debug directory.

    Args:
        debug_dir: Path to the saved ``processing/<task_id>/`` directory.
        output_dir: Fresh working directory for reprocessing output.
            Created if it does not exist. The original debug directory is
            never modified.
        settings: CitraScopeSettings instance (used by processors for
            ``enabled_processors`` and other config).
        log: Logger instance.  Falls back to module-level logger.
        image_override: If provided, use this FITS path instead of
            auto-discovering one in *debug_dir*.

    Returns:
        A fully populated ProcessingContext ready for
        ``ProcessorRegistry.process_all()``.

    Raises:
        FileNotFoundError: If *debug_dir* doesn't exist or no FITS file
            can be found.
        ValueError: If required artifacts (task.json, observer_location.json,
            telescope_record.json) are missing or unreadable.
    """
    log = log or logger

    if not debug_dir.is_dir():
        raise FileNotFoundError(f"Debug directory does not exist: {debug_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.iterdir())
    if existing:
        log.warning("Output directory is not empty (%d items) — stale artifacts may contaminate results", len(existing))

    # --- FITS image ---
    if image_override:
        source_fits = image_override
        if not source_fits.exists():
            raise FileNotFoundError(f"Image override does not exist: {source_fits}")
    else:
        source_fits = _discover_fits(debug_dir)
        if source_fits is None:
            raise FileNotFoundError(f"No FITS files found in {debug_dir}")

    dest_fits = output_dir / source_fits.name
    shutil.copy2(source_fits, dest_fits)
    log.info("Using FITS image: %s (copied to %s)", source_fits.name, output_dir)

    # --- Required artifacts ---
    task_data = _load_json(debug_dir / "task.json")
    if not task_data:
        raise ValueError(f"Missing or unreadable task.json in {debug_dir}")
    task = _task_from_saved_dict(task_data)

    observer_location = _load_json(debug_dir / "observer_location.json")
    if not observer_location:
        raise ValueError(f"Missing or unreadable observer_location.json in {debug_dir}")

    telescope_record = _load_json(debug_dir / "telescope_record.json")
    if not telescope_record:
        raise ValueError(f"Missing or unreadable telescope_record.json in {debug_dir}")

    # --- Optional artifacts ---
    elset_snapshot = _load_json(debug_dir / "elset_cache_snapshot.json") or []
    elset_cache = ElsetCache.from_snapshot(elset_snapshot)
    log.info("Loaded %d elsets from snapshot", len(elset_snapshot))

    satellite_data = _load_json(debug_dir / "target_satellite.json")
    pointing_report = _load_json(debug_dir / "pointing_report.json")

    tracking_mode: str | None = None
    sat_debug = _load_json(debug_dir / "satellite_matcher_debug.json")
    if isinstance(sat_debug, dict):
        tracking_mode = sat_debug.get("tracking_mode")

    location_service = FixedLocationService(observer_location)

    return ProcessingContext(
        image_path=dest_fits,
        working_image_path=dest_fits,
        working_dir=output_dir,
        image_data=None,
        task=task,
        telescope_record=telescope_record,
        ground_station_record=None,
        settings=settings,
        location_service=location_service,
        elset_cache=elset_cache,
        satellite_data=satellite_data,
        pointing_report=pointing_report,
        tracking_mode=tracking_mode,
        logger=log,
    )
