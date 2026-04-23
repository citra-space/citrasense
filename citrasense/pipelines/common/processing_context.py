"""Processing context shared across all pipeline modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from citrasense.tasks.task import Task

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ProcessingContext:
    """Rich context provided to pipeline processors.

    Shared fields (image paths, task, settings, services) are used by every
    modality.  Optical-specific fields (``detected_sources``, ``zero_point``,
    etc.) remain here for backward compatibility; future modalities will use
    ``payload`` for their own data.
    """

    # -- Modality tag --------------------------------------------------------
    modality: str = "optical"

    # Modality-specific data bag.  Optical processors ignore this and use the
    # typed fields below; radar / RF processors will populate it with their own
    # detection-record dicts.
    payload: Any = None

    # -- Image data ----------------------------------------------------------
    image_path: Path = field(default_factory=Path)
    working_image_path: Path = field(default_factory=Path)
    working_dir: Path = field(default_factory=Path)
    image_data: np.ndarray | None = None

    # Task context (None for manual captures)
    task: Task | None = None

    # Observatory context
    telescope_record: dict | None = None
    ground_station_record: dict | None = None

    # Settings
    settings: Any | None = None  # CitraSenseSettings instance

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
