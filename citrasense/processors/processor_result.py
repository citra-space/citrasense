"""Data classes for image processor input and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from citrasense.tasks.task import Task

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ProcessorResult:
    """Result returned by processors."""

    should_upload: bool  # False = skip upload
    extracted_data: dict  # Metrics to attach to upload
    confidence: float  # 0.0-1.0 quality score
    reason: str  # Human-readable explanation
    processing_time_seconds: float  # For metrics
    processor_name: str  # Which processor returned this


@dataclass
class AggregatedResult:
    """Combined results from all processors."""

    should_upload: bool  # AND of all processor results
    extracted_data: dict  # Merged extracted data
    all_results: list[ProcessorResult]  # Individual results
    total_time: float  # Total processing time
    skip_reason: str | None  # Why upload was skipped (if any)


@dataclass
class ProcessingContext:
    """Rich context provided to image processors."""

    # Image data
    image_path: Path  # Original captured image
    working_image_path: Path  # Current working image (processors can update this)
    working_dir: Path  # Task-specific temp directory for intermediate files
    image_data: np.ndarray | None  # Pre-loaded for performance

    # Task context (None for manual captures)
    task: Task | None

    # Observatory context
    telescope_record: dict | None
    ground_station_record: dict | None

    # Settings
    settings: Any | None  # CitraSenseSettings instance

    # Services injected by ProcessingQueue (None when not available or in tests)
    location_service: Any | None = None  # LocationService instance
    elset_cache: Any | None = None  # ElsetCache instance
    apass_catalog: Any | None = None  # ApassCatalog instance for local photometry

    # Satellite data fetched from API during task execution (includes most_recent_elset with the TLE used for pointing)
    satellite_data: dict | None = None

    # Pointing telemetry from the iterative slew convergence loop (None for SEQUENCE_TO_CONTROLLER strategy)
    pointing_report: dict | None = None

    # Tracking mode used during imaging: "sidereal" or "rate". Set by the telescope task at queue time.
    tracking_mode: str | None = None

    # Source catalog extracted by the plate solver (SEP detections converted to sky coords).
    # Columns: ra, dec, mag, magerr, elongation (a/b axis ratio).  Populated by
    # PlateSolverProcessor after a successful solve.
    detected_sources: pd.DataFrame | None = field(default=None, repr=False)

    # Photometric zero point computed by PhotometryProcessor.  Downstream processors
    # (e.g. SatelliteMatcherProcessor) use this to convert instrumental magnitudes
    # to calibrated magnitudes: calibrated = instrumental + zero_point.
    zero_point: float | None = None

    # Logging
    logger: Any | None = None  # Logger instance for debugging
