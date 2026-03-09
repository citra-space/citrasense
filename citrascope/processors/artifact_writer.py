"""Best-effort artifact dumping for processing pipeline diagnostics.

All public functions catch exceptions internally and log warnings — they never
block or fail the processing pipeline.  Artifacts are always written to the
per-task working directory (``processing/{task_id}/``); the existing
``keep_processing_output`` setting controls whether the directory is retained
after processing completes.

Artifact schema reference
=========================

Context artifacts (written before processors run)
--------------------------------------------------
``task.json``
    Task dataclass fields: id, type, status, satelliteId, satelliteName,
    taskStart/Stop, assigned_filter_name, etc.  Excludes internal fields
    (locks, retry state).

``elset_cache_snapshot.json``
    Full list of TLEs available at processing time.  Each entry has
    satellite_id, name, tle (two-line array).  This is the "what were we
    searching?" artifact — compare against ``satellite_matcher_debug.json``
    to see which TLEs were propagated and which were in-field.

``observer_location.json``
    GPS/ground-station location used for TLE propagation: latitude,
    longitude, altitude, source.

``telescope_record.json``
    Telescope hardware config from the Citra API: focalLength, pixelSize,
    horizontalPixelCount, spectralConfig, etc.

``fits_header.json``
    Key FITS header fields from the captured image: DATE-OBS, CRVAL1/2,
    NAXIS1/2, XBINNING/YBINNING, EXPTIME, FILTER, SITELAT/SITELONG/SITEALT,
    INSTRUME, GAIN, TASKID, etc.

``target_satellite.json`` *(only present when satellite_data was available)*
    Full satellite record fetched from the Citra API during task execution,
    including all elsets and the ``most_recent_elset`` with the TLE that
    was actually used to point the telescope.  This is the authoritative
    "what TLE did we aim at?" artifact — compare its TLE against the
    cache TLE (see ``target_satellite`` section in
    ``satellite_matcher_debug.json``) to detect staleness.

Per-processor result artifacts
------------------------------
Each ``*_result.json`` contains: processor_name, confidence (0–1),
reason (human-readable), processing_time_seconds, should_upload, and
extracted_data (the processor's output dict).

``plate_solver_result.json``
    extracted_data includes: plate_solved (bool), ra_center, dec_center,
    pixel_scale (arcsec/px), field_width_deg, field_height_deg,
    wcs_image_path.

``source_extractor_result.json``
    extracted_data includes: num_sources, sources_catalog (path).
    Extra top-level fields: fwhm_min, fwhm_max, fwhm_median, fwhm_mean,
    count_fwhm_lt_1_5, count_fwhm_gte_1_5.  The 1.5-pixel FWHM boundary
    is the satellite-vs-star classification threshold.

``photometry_result.json``
    extracted_data includes: zero_point, num_calibration_stars, filter.

``photometry_apass_catalog.csv``
    Raw APASS DR10 catalog stars returned for the field (all columns from
    the AAVSO query).

``photometry_crossmatch.csv``
    Cross-matched sources (detected + catalog), with instrumental mag and
    catalog mag columns used for zero-point calculation.

``satellite_matcher_result.json``
    extracted_data includes: num_satellites_detected,
    satellite_observations (list of matched satellite dicts).

``satellite_matcher_debug.json``
    Comprehensive diagnostic bundle.  Top-level keys:

    - ``tracking_mode``: "rate" or "static" (FWHM classification direction)
    - ``fwhm_threshold``: 1.5 (pixel boundary for star/satellite split)
    - ``field_radius_deg``: 2.0 (TLE in-field filter radius)
    - ``match_radius_arcmin``: 1.0 (KDTree match acceptance radius)
    - ``source_classification``: total_sources, satellite_candidate_count,
      star_like_count, fwhm_min/max/median/mean
    - ``field_center``: {ra_deg, dec_deg} from FITS CRVAL1/2
    - ``epoch``: DATE-OBS timestamp string
    - ``elset_count``: number of TLEs searched
    - ``elset_source``: "cache" or "task_fallback"
    - ``target_satellite``: comparison of the TLE used for pointing vs the
      cache TLE for the task's target satellite.  Keys: satellite_id,
      pointing_tle, pointing_elset_epoch, cache_tle, tle_match (bool —
      True if both TLEs are identical, None if either is missing).
      If satellite_data wasn't threaded through, pointing_tle is None
      with an explanatory note.
    - ``predictions_all``: every TLE propagation attempt — satellite_id,
      name, predicted_ra_deg, predicted_dec_deg, distance_from_center_deg,
      in_field (bool), phase_angle (if in-field), propagation_error (if
      failed)
    - ``predictions_in_field``: subset of predictions_all with in_field=True
    - ``predictions_in_field_count``: count of successful in-field predictions
    - ``match_details``: for each satellite-candidate source, the nearest
      prediction: source_ra/dec/fwhm/mag,
      nearest_prediction_distance_arcmin (always populated, even beyond
      match radius), nearest_satellite_id/name, within_match_radius (bool),
      matched (bool, present only when True)
    - ``reverse_match``: for each in-field prediction, the nearest detected
      source: satellite_id, name, predicted_ra/dec,
      nearest_source_distance_arcmin (always populated),
      within_match_radius, nearest_source_ra/dec/fwhm/mag
    - ``satellite_observations``: final matched observations (same payload
      as uploaded to the API)

Post-processing summary
-----------------------
``processing_summary.json``
    Aggregated pipeline result: should_upload (bool), skip_reason,
    total_time, per-processor entries (name, confidence, reason,
    processing_time, extracted_data_keys), and the merged extracted_data
    dict.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.io import fits

if TYPE_CHECKING:
    import pandas as pd

    from citrascope.processors.processor_result import AggregatedResult, ProcessingContext

logger = logging.getLogger("citrascope.artifact_writer")

_FITS_HEADER_KEYS = (
    "DATE-OBS",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "NAXIS1",
    "NAXIS2",
    "XBINNING",
    "YBINNING",
    "EXPTIME",
    "FILTER",
    "OBJECT",
    "SITELAT",
    "SITELONG",
    "SITEALT",
    "TELESCOP",
    "INSTRUME",
    "GAIN",
    "OBSERVER",
    "TASKID",
    "ORIGIN",
)


def _safe_value(v: Any) -> Any:
    """Coerce a value to something JSON-serialisable."""
    if isinstance(v, float) and (v != v):  # NaN
        return None
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _sanitize(obj: Any) -> Any:
    """Recursively make a data structure JSON-safe (NaN -> None, Path -> str)."""
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def dump_json(working_dir: Path, filename: str, data: Any) -> None:
    """Write *data* as pretty-printed JSON to *working_dir/filename*.

    Best-effort: exceptions are logged and swallowed.
    """
    try:
        path = working_dir / filename
        with open(path, "w") as f:
            json.dump(_sanitize(data), f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to write artifact %s: %s", filename, exc)


def dump_csv(working_dir: Path, filename: str, dataframe: pd.DataFrame) -> None:
    """Write a pandas DataFrame as CSV to *working_dir/filename*.

    Best-effort: exceptions are logged and swallowed.
    """
    try:
        path = working_dir / filename
        dataframe.to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Failed to write artifact %s: %s", filename, exc)


def _task_to_dict(task: Any) -> dict:
    """Serialise a Task dataclass to a plain dict, skipping non-serialisable fields."""
    if task is None:
        return {}
    if dataclasses.is_dataclass(task) and not isinstance(task, type):
        d = {}
        for field in dataclasses.fields(task):
            if field.name.startswith("_"):
                continue
            d[field.name] = _safe_value(getattr(task, field.name))
        return d
    return {"repr": repr(task)}


def _read_fits_header(image_path: Path) -> dict:
    """Extract diagnostic FITS header fields."""
    result: dict[str, Any] = {}
    try:
        with fits.open(image_path) as hdul:
            primary = hdul[0]
            assert isinstance(primary, fits.PrimaryHDU)
            header = primary.header
            for key in _FITS_HEADER_KEYS:
                val = header.get(key)
                if val is not None:
                    result[key] = _safe_value(val)
    except Exception as exc:
        result["_error"] = str(exc)
    return result


def dump_context_artifacts(context: ProcessingContext) -> None:
    """Write pre-processing context artifacts to the working directory.

    Writes: task.json, elset_cache_snapshot.json, observer_location.json,
    telescope_record.json, fits_header.json.
    """
    wd = context.working_dir
    try:
        dump_json(wd, "task.json", _task_to_dict(context.task))

        elsets: list[dict] = []
        if context.elset_cache:
            try:
                elsets = context.elset_cache.get_elsets()
            except Exception:
                pass
        dump_json(wd, "elset_cache_snapshot.json", elsets)

        location: dict[str, Any] = {}
        if context.location_service:
            try:
                location = context.location_service.get_current_location() or {}
            except Exception:
                pass
        dump_json(wd, "observer_location.json", location)

        dump_json(wd, "telescope_record.json", context.telescope_record or {})

        dump_json(wd, "fits_header.json", _read_fits_header(context.working_image_path))

        if context.satellite_data:
            dump_json(wd, "target_satellite.json", context.satellite_data)
    except Exception as exc:
        logger.warning("Failed to dump context artifacts: %s", exc)


def dump_processor_result(working_dir: Path, filename: str, result: Any, extra: dict | None = None) -> None:
    """Write a ProcessorResult's key fields (plus optional extras) to JSON."""
    data: dict[str, Any] = {
        "processor_name": result.processor_name,
        "confidence": result.confidence,
        "reason": result.reason,
        "processing_time_seconds": result.processing_time_seconds,
        "should_upload": result.should_upload,
        "extracted_data": {k: _safe_value(v) for k, v in result.extracted_data.items()},
    }
    if extra:
        data.update(extra)
    dump_json(working_dir, filename, data)


def dump_processing_summary(working_dir: Path, aggregated: AggregatedResult) -> None:
    """Write the full aggregated processing result to processing_summary.json."""
    processors = []
    for r in aggregated.all_results:
        processors.append(
            {
                "processor_name": r.processor_name,
                "confidence": r.confidence,
                "reason": r.reason,
                "processing_time_seconds": r.processing_time_seconds,
                "should_upload": r.should_upload,
                "extracted_data_keys": list(r.extracted_data.keys()),
            }
        )
    data = {
        "should_upload": aggregated.should_upload,
        "skip_reason": aggregated.skip_reason,
        "total_time": aggregated.total_time,
        "processors": processors,
        "extracted_data": {k: _safe_value(v) for k, v in aggregated.extracted_data.items()},
    }
    dump_json(working_dir, "processing_summary.json", data)
