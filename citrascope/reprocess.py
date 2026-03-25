"""Replay a saved processing debug directory through the current pipeline.

This is a standalone developer tool for iterating on the citrascope image
processing pipeline.  It reconstructs a ProcessingContext from the JSON
artifacts and FITS image in a retained debug directory, then runs the full
processor chain against it.  No daemon, web server, or hardware is started.

Generating debug directories
-----------------------------
Enable **Keep Processing Output** in the citrascope web UI (Settings), or set
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

    python -m citrascope.reprocess /path/to/processing/<task_id>/

    # Override output location:
    python -m citrascope.reprocess /path/to/processing/<task_id>/ --output-dir /tmp/rerun

    # Use a specific FITS file instead of auto-discovery:
    python -m citrascope.reprocess /path/to/processing/<task_id>/ --image /path/to/raw.fits

Output is written to a new directory (default: ``<debug_dir>_reprocessed_<timestamp>``)
so the original debug capture is never modified.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from citrascope.processors.context_loader import load_context_from_debug_dir
from citrascope.processors.processor_registry import ProcessorRegistry
from citrascope.settings.citrascope_settings import CitraScopeSettings


def _setup_logging() -> logging.Logger:
    """Configure console logging for the reprocessing tool."""
    log = logging.getLogger("citrascope")
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
def cli(debug_dir: Path, output_dir: Path | None, image: Path | None) -> None:
    """Replay a saved processing debug directory through the current pipeline."""
    log = _setup_logging()

    if output_dir is None:
        output_dir = _default_output_dir(debug_dir)

    click.echo(f"Debug directory : {debug_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo()

    # Load current settings (for enabled_processors, etc.)
    settings = CitraScopeSettings.load()

    # Build the processor pipeline with current code
    registry = ProcessorRegistry(settings=settings, logger=log)

    # Reconstruct context from saved artifacts
    context = load_context_from_debug_dir(
        debug_dir=debug_dir,
        output_dir=output_dir,
        settings=settings,
        log=log,
        image_override=image,
    )

    # Run the pipeline
    click.echo("Running pipeline...")
    click.echo()
    start = time.time()
    result = registry.process_all(context)
    elapsed = time.time() - start

    # Print summary
    report_path = output_dir / "report.html"
    click.echo()
    click.echo("=" * 60)
    click.echo("REPROCESSING SUMMARY")
    click.echo("=" * 60)
    for r in result.all_results:
        status = "OK" if r.should_upload and r.confidence > 0 else "FAIL"
        click.echo(f"  [{status:>4}] {r.processor_name:<25} {r.processing_time_seconds:.2f}s  {r.reason}")
    click.echo("-" * 60)
    click.echo(f"  Total time   : {elapsed:.2f}s")
    click.echo(f"  Should upload: {result.should_upload}")
    if result.skip_reason:
        click.echo(f"  Skip reason  : {result.skip_reason}")
    click.echo(f"  Output       : {output_dir}")
    if report_path.exists():
        click.echo(f"  Report       : {report_path}")
    click.echo()


if __name__ == "__main__":
    cli()
