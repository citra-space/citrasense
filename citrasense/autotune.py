"""SExtractor auto-tuning: score parameter combinations against debug bundles.

Runs SExtractor with many different (detect_thresh, detect_minarea, filter_name)
combinations against retained debug bundles and scores each one by:
  - Satellite detection (elongation-aware, proximity-scaled)
  - APASS catalog cross-match count (photometric depth via real KDTree matching)
  - Signal purity (ratio of real stars to total sources)
  - Source quality (FWHM anchored to APASS-matched stars)
  - False-positive penalty (random match probability given source density)

Scoring reuses production pipeline artifacts and utilities rather than
reimplementing matching logic.  Only SExtractor is re-run per parameter
combination — everything else is read from the existing debug bundle.

Usage (CLI)::

    uv run python -m citrasense.autotune /path/to/processing/ --num-bundles 5

Usage (library)::

    from citrasense.autotune import autotune_extraction, score_extraction
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree

from citrasense.pipelines.optical.photometry_processor import cross_match_catalogs
from citrasense.pipelines.optical.satellite_matcher_processor import (
    _ELONGATION_THRESHOLD,
    _MATCH_RADIUS_DEG,
)
from citrasense.pipelines.optical.source_extractor_processor import SourceExtractorProcessor
from citrasense.settings.citrasense_settings import CitraSenseSettings

logger = logging.getLogger("citrasense.Autotune")

SEXTRACTOR_CONFIG_DIR = Path(__file__).parent / "pipelines" / "optical" / "sextractor_configs"

PARAM_GRID: dict[str, list] = {
    "detect_thresh": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0],
    "detect_minarea": [2, 3, 5, 7, 10],
    "filter_name": ["default", "gauss_2.5_5x5", "tophat_3.0_3x3"],
}

# Scoring weights — must sum to ~1.0 (penalty is subtracted)
W_SATELLITE = 0.35
W_DEPTH = 0.25
W_PURITY = 0.20
W_QUALITY = 0.10
W_FP_PENALTY = 0.10


@dataclass
class ExtractionScore:
    """Result of scoring one SExtractor parameter combination."""

    detect_thresh: float
    detect_minarea: int
    filter_name: str
    num_sources: int = 0
    satellite_detected: bool = False
    sat_match_distance_arcmin: float | None = None
    sat_candidates: int = 0
    num_calibration_stars: int = 0
    source_quality_ratio: float = 0.0
    signal_purity: float = 0.0
    false_positive_rate: float = 0.0
    score: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "detect_thresh": self.detect_thresh,
            "detect_minarea": self.detect_minarea,
            "filter_name": self.filter_name,
            "num_sources": self.num_sources,
            "satellite_detected": self.satellite_detected,
            "sat_match_distance_arcmin": self.sat_match_distance_arcmin,
            "sat_candidates": self.sat_candidates,
            "num_calibration_stars": self.num_calibration_stars,
            "source_quality_ratio": round(self.source_quality_ratio, 3),
            "signal_purity": round(self.signal_purity, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "score": round(self.score, 4),
            "error": self.error,
        }


@dataclass
class _BundleContext:
    """Pre-loaded data from a debug bundle's existing pipeline artifacts.

    All fields are read from files already written by a previous pipeline
    run — no pipeline logic is reimplemented here.
    """

    image_path: Path
    working_dir: Path
    predicted_ra: float | None = None
    predicted_dec: float | None = None
    tracking_mode: str = "sidereal"
    field_area_sq_deg: float = 4.0
    apass_catalog: pd.DataFrame | None = None


def _load_bundle_context(debug_dir: Path, apass_catalog: Any | None = None) -> _BundleContext | None:
    """Load scoring context from existing pipeline artifacts in a debug bundle.

    Reads satellite predictions from ``satellite_matcher_debug.json``, APASS
    catalog from ``photometry_apass_catalog.csv``, tracking mode from
    ``task.json``, and field geometry from the FITS WCS header.

    Args:
        debug_dir: Path to a processing debug bundle.
        apass_catalog: Optional live ApassCatalog instance for fallback when the
            bundle doesn't contain a saved ``photometry_apass_catalog.csv``.
    """
    # Find WCS-solved FITS (same priority as context_loader)
    fits_files = sorted(debug_dir.glob("*.fits"))
    wcs_files = [f for f in fits_files if f.name.endswith("_wcs.fits")]
    calibrated = [f for f in fits_files if f.name == "calibrated.fits"]

    image_path = None
    if wcs_files:
        image_path = wcs_files[0]
    elif calibrated:
        image_path = calibrated[0]
    elif fits_files:
        image_path = fits_files[0]

    if not image_path:
        return None

    # Read field geometry from FITS header
    field_center_ra: float | None = None
    field_center_dec: float | None = None
    field_area = 4.0

    try:
        with fits.open(image_path) as hdul:
            primary = hdul[0]
            assert isinstance(primary, fits.PrimaryHDU)
            if "CRVAL1" not in primary.header:
                return None
            field_center_ra = float(primary.header["CRVAL1"])  # type: ignore[arg-type]
            field_center_dec = float(primary.header["CRVAL2"])  # type: ignore[arg-type]
            header = primary.header
            naxis1 = float(header.get("NAXIS1", 0))  # type: ignore[arg-type]
            naxis2 = float(header.get("NAXIS2", 0))  # type: ignore[arg-type]
            cdelt1 = header.get("CDELT1") or header.get("CD1_1")
            cdelt2 = header.get("CDELT2") or header.get("CD2_2")
            if cdelt1 is not None and cdelt2 is not None:
                field_area = abs(float(cdelt1) * naxis1 * float(cdelt2) * naxis2)  # type: ignore[arg-type]
    except Exception:
        return None

    ctx = _BundleContext(image_path=image_path, working_dir=debug_dir, field_area_sq_deg=field_area)

    # Read tracking mode from task.json
    task_path = debug_dir / "task.json"
    task_data: dict = {}
    if task_path.exists():
        try:
            task_data = json.loads(task_path.read_text())
            ctx.tracking_mode = task_data.get("trackingMode", "sidereal")
        except Exception:
            pass

    # Read satellite predictions from existing satellite_matcher_debug.json
    sat_debug_path = debug_dir / "satellite_matcher_debug.json"
    if sat_debug_path.exists():
        try:
            sat_debug = json.loads(sat_debug_path.read_text())
            preds = sat_debug.get("predictions_in_field", [])
            target_sat_id = task_data.get("satelliteId", "")
            if target_sat_id:
                target_sat_id = str(target_sat_id).replace("sat-", "")

            if target_sat_id:
                for pred in preds:
                    if str(pred.get("satellite_id", "")) == target_sat_id:
                        ctx.predicted_ra = pred.get("predicted_ra_deg")
                        ctx.predicted_dec = pred.get("predicted_dec_deg")
                        break
            if ctx.predicted_ra is None and preds:
                ctx.predicted_ra = preds[0].get("predicted_ra_deg")
                ctx.predicted_dec = preds[0].get("predicted_dec_deg")
        except Exception:
            pass

    # Read APASS catalog from existing photometry_apass_catalog.csv
    apass_path = debug_dir / "photometry_apass_catalog.csv"
    if apass_path.exists():
        try:
            ctx.apass_catalog = pd.read_csv(apass_path)
        except Exception:
            pass

    # Fall back to live catalog query if bundle doesn't have the CSV
    if ctx.apass_catalog is None and apass_catalog is not None and field_center_ra is not None:
        try:
            if apass_catalog.is_available():
                ctx.apass_catalog = apass_catalog.cone_search(field_center_ra, field_center_dec, radius=2.0)
        except Exception:
            pass

    return ctx


def score_extraction(
    bundle: _BundleContext,
    detect_thresh: float,
    detect_minarea: int,
    filter_name: str,
) -> ExtractionScore:
    """Score a single SExtractor parameter combination against a bundle.

    Runs SExtractor with the given parameters, then scores the extracted
    sources against the bundle's existing pipeline artifacts:

    1. Satellite detection — elongation-filtered candidates matched to
       predicted position from ``satellite_matcher_debug.json``
    2. APASS cross-match — uses :func:`cross_match_catalogs` (the same
       KDTree implementation used by :class:`PhotometryProcessor`)
    3. Signal purity — fraction of sources matched to APASS
    4. Source quality — FWHM anchored to APASS-matched stars
    5. False-positive penalty — random match probability
    """
    result = ExtractionScore(
        detect_thresh=detect_thresh,
        detect_minarea=detect_minarea,
        filter_name=filter_name,
    )

    extractor = SourceExtractorProcessor()
    try:
        sources = extractor._extract_sources(
            image_path=bundle.image_path,
            config_dir=SEXTRACTOR_CONFIG_DIR,
            working_dir=bundle.working_dir,
            detect_thresh=detect_thresh,
            detect_minarea=detect_minarea,
            filter_name=filter_name if filter_name != "default" else None,
        )
    except Exception as exc:
        result.error = str(exc)
        return result

    result.num_sources = len(sources)
    if result.num_sources == 0:
        return result

    # 1. Elongation-aware satellite detection using constants from satellite_matcher_processor
    if bundle.predicted_ra is not None and bundle.predicted_dec is not None:
        if bundle.tracking_mode == "rate":
            candidates = sources[sources["elongation"] < _ELONGATION_THRESHOLD]
        else:
            candidates = sources[sources["elongation"] >= _ELONGATION_THRESHOLD]
        result.sat_candidates = len(candidates)

        if not candidates.empty:
            cand_coords = np.asarray(candidates[["ra", "dec"]])
            tree = KDTree(cand_coords)
            dist, _ = tree.query([[bundle.predicted_ra, bundle.predicted_dec]])
            min_dist = float(dist[0])

            if min_dist < _MATCH_RADIUS_DEG:
                result.satellite_detected = True
                result.sat_match_distance_arcmin = round(min_dist * 60.0, 4)

    # 2. APASS cross-match using the production cross_match_catalogs() utility
    num_matched = 0
    matched_source_indices: set[int] = set()
    if bundle.apass_catalog is not None and len(bundle.apass_catalog) > 0:
        apass = bundle.apass_catalog
        if "radeg" in apass.columns and "decdeg" in apass.columns:
            matched_df = cross_match_catalogs(sources, apass, max_separation=5.0 / 3600.0)
            num_matched = len(matched_df)
            if num_matched > 0:
                source_coords = np.asarray(sources[["ra", "dec"]])
                source_tree = KDTree(source_coords)
                matched_coords = np.asarray(matched_df[["ra", "dec"]])
                _, idx = source_tree.query(matched_coords)
                matched_source_indices = {int(i) for i in np.asarray(idx).flat}
    result.num_calibration_stars = num_matched

    # 3. Signal purity: fraction of sources that are real (matched to APASS)
    if result.num_calibration_stars > 0 and result.num_sources > 0:
        result.signal_purity = result.num_calibration_stars / result.num_sources

    # 4. Source quality — anchor FWHM to APASS-matched sources
    if "fwhm" in sources.columns and "elongation" in sources.columns:
        if matched_source_indices:
            matched_fwhm = sources.loc[list(matched_source_indices), "fwhm"]
            anchor_fwhm = float(matched_fwhm.median()) if len(matched_fwhm) > 0 else float(sources["fwhm"].median())
        else:
            anchor_fwhm = float(sources["fwhm"].median())

        if anchor_fwhm > 0:
            good_mask = (sources["fwhm"] > 0) & (sources["fwhm"] < 2.5 * anchor_fwhm) & (sources["elongation"] < 3.0)
            result.source_quality_ratio = float(good_mask.sum()) / len(sources)

    # 5. False-positive rate: probability of random source landing within match radius
    if bundle.field_area_sq_deg > 0 and result.num_sources > 0:
        match_area = math.pi * _MATCH_RADIUS_DEG**2
        result.false_positive_rate = min(1.0, result.num_sources * match_area / bundle.field_area_sq_deg)

    # Composite score
    if result.satellite_detected and result.sat_match_distance_arcmin is not None:
        sat_score = max(0.0, 1.0 - (result.sat_match_distance_arcmin / 60.0 / _MATCH_RADIUS_DEG))
    else:
        sat_score = 0.0

    depth_score = min(1.0, result.num_calibration_stars / 20.0) if result.num_calibration_stars > 0 else 0.0

    fp_penalty = max(0.0, result.false_positive_rate - 0.5) * 2.0

    result.score = (
        W_SATELLITE * sat_score
        + W_DEPTH * depth_score
        + W_PURITY * result.signal_purity
        + W_QUALITY * result.source_quality_ratio
        - W_FP_PENALTY * fp_penalty
    )

    return result


def autotune_extraction(
    debug_dirs: list[Path],
    settings: CitraSenseSettings | None = None,
    log: logging.Logger | None = None,
    grid: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[dict]:
    """Sweep SExtractor parameters across debug bundles and return ranked results.

    Args:
        debug_dirs: Paths to debug bundle directories.
        settings: Not currently used but reserved for future grid derivation.
        log: Logger.
        grid: Parameter grid override.  Defaults to ``PARAM_GRID``.
        on_progress: Called with ``(completed_combos, total_combos)``.
        is_cancelled: If provided, checked before each evaluation. Returns
            partial results gathered so far when cancellation is detected.

    Returns:
        List of averaged ``ExtractionScore.to_dict()`` dicts, sorted best-first.
    """
    log = log or logger
    grid = grid or PARAM_GRID

    thresholds = grid.get("detect_thresh", PARAM_GRID["detect_thresh"])
    minareas = grid.get("detect_minarea", PARAM_GRID["detect_minarea"])
    filters = grid.get("filter_name", PARAM_GRID["filter_name"])

    combos = [(t, m, f) for t in thresholds for m in minareas for f in filters]
    total = len(combos) * len(debug_dirs)

    log.info("Auto-tune: %d combos × %d bundles = %d evaluations", len(combos), len(debug_dirs), total)

    # Try to load the live APASS catalog for bundles missing the CSV
    live_apass = None
    try:
        from citrasense.catalogs.apass_catalog import ApassCatalog

        live_apass = ApassCatalog()
        if not live_apass.is_available():
            log.info("APASS catalog not installed — bundles without saved CSV will score without cross-match")
            live_apass = None
        else:
            log.info("APASS catalog available at %s — will use as fallback", live_apass.db_path)
    except Exception:
        pass

    bundles: list[_BundleContext] = []
    for d in debug_dirs:
        ctx = _load_bundle_context(d, apass_catalog=live_apass)
        if ctx:
            bundles.append(ctx)
        else:
            log.warning("Skipping %s: could not load bundle context", d.name)

    if not bundles:
        log.error("No valid bundles to tune against")
        return []

    scores_by_combo: dict[tuple, list[ExtractionScore]] = {c: [] for c in combos}
    done = 0
    cancelled = False

    for combo in combos:
        if cancelled:
            break
        thresh, minarea, fname = combo
        for bundle in bundles:
            if is_cancelled and is_cancelled():
                log.info("Auto-tune cancelled at %d/%d evaluations", done, total)
                cancelled = True
                break
            s = score_extraction(bundle, thresh, minarea, fname)
            scores_by_combo[combo].append(s)
            done += 1
            if on_progress:
                on_progress(done, total)

    averaged: list[dict] = []
    for combo, scores in scores_by_combo.items():
        valid = [s for s in scores if s.error is None]
        if not valid:
            continue
        avg: dict[str, Any] = {
            "detect_thresh": combo[0],
            "detect_minarea": combo[1],
            "filter_name": combo[2],
            "avg_score": round(float(np.mean([s.score for s in valid])), 4),
            "avg_sources": round(float(np.mean([s.num_sources for s in valid])), 1),
            "satellite_detection_rate": round(sum(1 for s in valid if s.satellite_detected) / len(valid), 3),
            "avg_calibration_stars": round(float(np.mean([s.num_calibration_stars for s in valid])), 1),
            "avg_quality_ratio": round(float(np.mean([s.source_quality_ratio for s in valid])), 3),
            "avg_purity": round(float(np.mean([s.signal_purity for s in valid])), 4),
            "avg_fp_rate": round(float(np.mean([s.false_positive_rate for s in valid])), 4),
            "bundles_evaluated": len(valid),
        }
        averaged.append(avg)

    averaged.sort(key=lambda x: x["avg_score"], reverse=True)
    return averaged


def _discover_bundles(base_dir: Path, max_bundles: int = 10) -> list[Path]:
    """Find debug bundle directories under *base_dir* that have WCS-solved FITS.

    Supports the legacy flat layout (``base_dir/<task_id>``) and the
    multi-sensor layout (``base_dir/<sensor_id>/<task_id>``).  Bundles
    across both layouts are combined and returned sorted by mtime
    (newest first).
    """
    candidates: list[Path] = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "task.json").exists():
            candidates.append(d)
        else:
            # Possible sensor dir — walk one level deeper.
            for nested in d.iterdir():
                if nested.is_dir() and (nested / "task.json").exists():
                    candidates.append(nested)

    bundles = []
    for d in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        fits_files = list(d.glob("*.fits"))
        has_wcs = any(f.name.endswith("_wcs.fits") or f.name == "calibrated.fits" for f in fits_files)
        if not has_wcs:
            continue
        bundles.append(d)
        if len(bundles) >= max_bundles:
            break
    return bundles


@click.command()
@click.argument("processing_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--num-bundles", default=5, help="Max bundles to evaluate against.")
@click.option("--apply", "apply_settings", is_flag=True, help="Write best settings to config.json.")
@click.option("--top", default=10, help="Number of top results to display.")
@click.option(
    "--sensor-id",
    "sensor_id",
    default=None,
    help=(
        "Sensor ID to apply tuning to.  Required when --apply is used and the "
        "site has more than one sensor configured (so we don't silently write "
        "tuning for the wrong rig)."
    ),
)
def cli(
    processing_dir: Path,
    num_bundles: int,
    apply_settings: bool,
    top: int,
    sensor_id: str | None,
) -> None:
    """Auto-tune SExtractor parameters against retained debug bundles."""
    log = logging.getLogger("citrasense.Autotune")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
        log.addHandler(handler)

    bundles = _discover_bundles(processing_dir, max_bundles=num_bundles)
    if not bundles:
        click.echo("No debug bundles found with WCS-solved FITS")
        return

    click.echo(f"Found {len(bundles)} bundle(s). Running parameter sweep...")
    start = time.time()

    def _progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            click.echo(f"  [{done}/{total}]")

    results = autotune_extraction(bundles, log=log, on_progress=_progress)
    elapsed = time.time() - start

    click.echo()
    click.echo(f"Completed in {elapsed:.1f}s. Top {min(top, len(results))} configs:")
    click.echo()
    hdr = (
        f"{'Rank':<5} {'Score':<8} {'Thresh':<8} {'MinArea':<9} {'Filter':<18}"
        f" {'Sources':<9} {'SatDet%':<9} {'CalStars':<10} {'Purity':<9} {'FP Risk':<9}"
    )
    click.echo(hdr)
    click.echo("-" * 104)
    for i, r in enumerate(results[:top]):
        click.echo(
            f"{i + 1:<5} {r['avg_score']:<8.4f} {r['detect_thresh']:<8} {r['detect_minarea']:<9} "
            f"{r['filter_name']:<18} {r['avg_sources']:<9.1f} "
            f"{r['satellite_detection_rate'] * 100:<9.1f} "
            f"{r['avg_calibration_stars']:<10.1f} {r['avg_purity']:<9.4f} {r['avg_fp_rate']:<9.4f}"
        )

    if apply_settings and results:
        best = results[0]
        settings = CitraSenseSettings.load()
        if not settings.sensors:
            click.echo("No sensors configured; cannot apply tuning.", err=True)
            sys.exit(1)

        # Pick the target sensor.  Refuse to silently default to sensors[0]
        # when more than one rig is configured — the operator must pick.
        target = None
        if sensor_id:
            target = next((sc for sc in settings.sensors if sc.id == sensor_id), None)
            if target is None:
                ids = ", ".join(sc.id for sc in settings.sensors)
                click.echo(
                    f"Sensor id {sensor_id!r} not found in config. Available: {ids}",
                    err=True,
                )
                sys.exit(2)
        elif len(settings.sensors) == 1:
            target = settings.sensors[0]
        else:
            ids = ", ".join(sc.id for sc in settings.sensors)
            click.echo(
                f"Multiple sensors configured ({ids}); pass --sensor-id to pick one.",
                err=True,
            )
            sys.exit(2)

        target.sextractor_detect_thresh = best["detect_thresh"]
        target.sextractor_detect_minarea = best["detect_minarea"]
        target.sextractor_filter_name = best["filter_name"]
        settings.save()
        click.echo()
        click.echo(
            f"Applied best config to sensor {target.id!r}: "
            f"thresh={best['detect_thresh']}, "
            f"minarea={best['detect_minarea']}, filter={best['filter_name']}"
        )


if __name__ == "__main__":
    cli()
