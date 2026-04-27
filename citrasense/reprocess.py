"""Replay a saved processing debug directory through the current pipeline.

This is a standalone developer tool for iterating on the citrasense image
processing pipeline.  It reconstructs a ProcessingContext from the JSON
artifacts and FITS image in a retained debug directory, then runs the full
processor chain against it.  No daemon, web server, or hardware is started.

Generating debug directories
-----------------------------
Enable **Keep Processing Output** in the citrasense web UI (Settings), or set
``keep_processing_output: true`` in your config JSON.  After each task
completes, the processing artifacts will be retained at::

    <data_dir>/processing/<task_id>/

The directory typically contains:

    task.json                       Task metadata
    observer_location.json          Observatory lat/lon/alt
    telescope_record.json           Telescope hardware config
    elset_cache_snapshot.json       TLE catalog at processing time
    target_satellite.json           Target satellite record (if available)
    pointing_report.json            Slew convergence telemetry (if available)
    satellite_matcher_debug.json    Satellite matcher diagnostics
    fits_header.json                FITS header snapshot
    original_<name>.fits            Raw captured image
    plate_solver_result.json        Plate solver output
    photometry_result.json          Photometry output
    satellite_matcher_result.json   Satellite matcher output
    processing_summary.json         Aggregated pipeline result
    annotated.png                   Annotated overlay image

Usage
-----
::

    python -m citrasense.reprocess /path/to/processing/<task_id>/

    # Override output location:
    python -m citrasense.reprocess /path/to/processing/<task_id>/ --output-dir /tmp/rerun

    # Use a specific FITS file instead of auto-discovery:
    python -m citrasense.reprocess /path/to/processing/<task_id>/ --image /path/to/raw.fits

Output is written to a new directory (default: ``<debug_dir>_reprocessed_<timestamp>``)
so the original debug capture is never modified.
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from citrasense.pipelines.common.context_loader import load_context_from_debug_dir
from citrasense.pipelines.common.pipeline_registry import PipelineRegistry
from citrasense.pipelines.common.processor_result import AggregatedResult
from citrasense.settings.citrasense_settings import CitraSenseSettings


def _setup_logging() -> logging.Logger:
    """Configure console logging for the reprocessing tool."""
    log = logging.getLogger("citrasense.Reprocess")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
        log.addHandler(handler)
    return log


def _default_output_dir(debug_dir: Path) -> Path:
    """Generate a timestamped output directory path next to the debug directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return debug_dir.parent / f"{debug_dir.name}_reprocessed_{stamp}"


def _resolve_sensor_id_from_bundle(debug_dir: Path) -> str | None:
    """Best-effort: infer ``sensor_id`` from bundle metadata.

    Looks in this order:

    1. ``task.json``'s ``sensor_id`` field (if present and truthy).
    2. ``processing_summary.json``'s ``sensor_id`` field — written by
       :func:`dump_processing_summary` for every run.
    3. The debug dir's parent name, when the bundle lives under the
       sensor-scoped layout ``processing/<sensor_id>/<task_id>``.
    """
    import json

    for name in ("task.json", "processing_summary.json"):
        path = debug_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        sid = data.get("sensor_id")
        if sid:
            return str(sid)

    parent = debug_dir.parent
    if parent.name and parent.name != "processing":
        # Heuristic: if the parent isn't the top-level "processing" root
        # it's probably a sensor namespace.  Return its name so callers
        # can validate it against the loaded settings.
        return parent.name
    return None


def reprocess_bundle(
    debug_dir: Path,
    output_dir: Path | None = None,
    settings_overrides: dict | None = None,
    image_override: Path | None = None,
    logger: logging.Logger | None = None,
    sensor_id: str | None = None,
) -> tuple[AggregatedResult, Path]:
    """Run the processing pipeline against a saved debug bundle.

    Args:
        debug_dir: Path to the saved ``processing/<task_id>/`` directory.
        output_dir: Where to write reprocessed output.  Defaults to
            ``<debug_dir>_reprocessed_<timestamp>``.
        settings_overrides: Dict of setting field names to values
            (e.g. ``{"sextractor_detect_thresh": 3.0}``).  Applied to a
            *copy* of the current settings so live config is never mutated.
        image_override: Use this FITS path instead of auto-discovering one.
        logger: Logger instance.  Falls back to a console logger.
        sensor_id: Which sensor's :class:`SensorConfig` to target with
            per-sensor overrides.  When omitted the sensor is inferred
            from the bundle (``task.json`` → ``processing_summary.json``
            → parent directory name).  Raises :class:`ValueError` when
            inference fails on a multi-sensor config so we never write
            tuning to the wrong rig.

    Returns:
        ``(aggregated_result, output_dir)`` tuple.
    """
    log = logger or _setup_logging()

    if output_dir is None:
        output_dir = _default_output_dir(debug_dir)

    settings = CitraSenseSettings.load()

    resolved_sensor_id = sensor_id or _resolve_sensor_id_from_bundle(debug_dir)
    target_sensor = None
    if settings.sensors:
        if resolved_sensor_id:
            target_sensor = settings.get_sensor_config(resolved_sensor_id)
            if target_sensor is None:
                ids = ", ".join(sc.id for sc in settings.sensors)
                raise ValueError(
                    f"Sensor id {resolved_sensor_id!r} not found in settings "
                    f"(available: {ids}). Pass --sensor-id explicitly."
                )
        elif len(settings.sensors) == 1:
            # Single-sensor site: unambiguous, use it.
            target_sensor = settings.sensors[0]
        else:
            ids = ", ".join(sc.id for sc in settings.sensors)
            raise ValueError(
                "Bundle does not carry a sensor_id and the site has multiple "
                f"sensors configured ({ids}); pass --sensor-id explicitly."
            )
        log.info("Reprocessing against sensor %r", target_sensor.id)

    if settings_overrides:
        from citrasense.settings.citrasense_settings import SensorConfig

        sensor_fields = SensorConfig.model_fields
        settings = copy.deepcopy(settings)
        # Re-resolve target sensor against the deep copy so mutations land on it.
        if target_sensor is not None:
            target_sensor = settings.get_sensor_config(target_sensor.id) or settings.sensors[0]
        for key, value in settings_overrides.items():
            if key in sensor_fields and target_sensor is not None:
                setattr(target_sensor, key, value)
            elif key in settings.model_fields:
                setattr(settings, key, value)
            else:
                log.warning("Unknown settings override ignored: %s", key)

    registry = PipelineRegistry(settings=settings, logger=log)

    context_settings: Any = target_sensor if target_sensor is not None else settings
    context = load_context_from_debug_dir(
        debug_dir=debug_dir,
        output_dir=output_dir,
        settings=context_settings,
        log=log,
        image_override=image_override,
    )

    start = time.time()
    result = registry.process_all(context)
    result.total_time = time.time() - start

    return result, output_dir


@click.command()
@click.argument("debug_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for reprocessing output.  Default: <debug_dir>_reprocessed_<timestamp>",
)
@click.option(
    "--image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override FITS image path instead of auto-discovering one in the debug directory.",
)
@click.option(
    "--sensor-id",
    "sensor_id",
    type=str,
    default=None,
    help="Sensor id whose SensorConfig should receive per-sensor overrides. "
    "Defaults to task.json's sensor_id, then the first configured sensor.",
)
def cli(debug_dir: Path, output_dir: Path | None, image: Path | None, sensor_id: str | None) -> None:
    """Replay a saved processing debug directory through the current pipeline."""
    log = _setup_logging()

    if output_dir is None:
        output_dir = _default_output_dir(debug_dir)

    click.echo(f"Debug directory : {debug_dir}")
    click.echo(f"Output directory: {output_dir}")
    if sensor_id:
        click.echo(f"Sensor id       : {sensor_id}")
    click.echo()

    click.echo("Running pipeline...")
    click.echo()
    result, output_dir = reprocess_bundle(
        debug_dir=debug_dir,
        output_dir=output_dir,
        image_override=image,
        logger=log,
        sensor_id=sensor_id,
    )

    report_path = output_dir / "report.html"
    click.echo()
    click.echo("=" * 60)
    click.echo("REPROCESSING SUMMARY")
    click.echo("=" * 60)
    for r in result.all_results:
        status = "OK" if r.should_upload and r.confidence > 0 else "FAIL"
        click.echo(f"  [{status:>4}] {r.processor_name:<25} {r.processing_time_seconds:.2f}s  {r.reason}")
    click.echo("-" * 60)
    click.echo(f"  Total time   : {result.total_time:.2f}s")
    click.echo(f"  Should upload: {result.should_upload}")
    if result.skip_reason:
        click.echo(f"  Skip reason  : {result.skip_reason}")
    click.echo(f"  Output       : {output_dir}")
    if report_path.exists():
        click.echo(f"  Report       : {report_path}")
    click.echo()


if __name__ == "__main__":
    cli()
