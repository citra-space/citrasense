"""Processing context shared across all pipeline modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from citrasense.tasks.task import Task


@dataclass
class ProcessingContext:
    """Base context shared by every pipeline modality.

    Holds image paths, the task, settings, injected services, and a logger.
    Modality-specific fields live on subclasses (e.g.
    ``OpticalProcessingContext``).
    """

    # -- Modality tag --------------------------------------------------------
    modality: str = "optical"

    # -- Sensor attribution --------------------------------------------------
    # ID of the sensor whose runtime is executing this pipeline. Used for log
    # prefixes and HTML report headers so artifacts from multi-sensor
    # deployments can be traced back to the right rig. Optional (defaults to
    # empty string for manual captures and older tests).
    sensor_id: str = ""

    # -- Image data ----------------------------------------------------------
    image_path: Path = field(default_factory=Path)
    working_image_path: Path = field(default_factory=Path)
    working_dir: Path = field(default_factory=Path)
    image_data: np.ndarray | None = None

    # Task context (None for manual captures)
    task: Task | None = None

    # Settings
    settings: Any | None = None  # CitraSenseSettings instance

    # Services injected by ProcessingQueue (None when not available or in tests)
    location_service: Any | None = None  # LocationService instance
    elset_cache: Any | None = None  # ElsetCache instance
    apass_catalog: Any | None = None  # ApassCatalog instance for local photometry

    # Logging
    logger: Any | None = None  # Logger instance for debugging
