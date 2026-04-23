"""Optical-modality processing context with telescope-specific fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from citrasense.pipelines.common.processing_context import ProcessingContext

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class OpticalProcessingContext(ProcessingContext):
    """ProcessingContext extended with optical/telescope-specific fields.

    Generic pipeline code (``PipelineRegistry.process_all``, artifact writers)
    accepts the base ``ProcessingContext``; optical processors narrow via
    ``assert isinstance(context, OpticalProcessingContext)``.
    """

    telescope_record: dict | None = None
    ground_station_record: dict | None = None

    satellite_data: dict | None = None
    pointing_report: dict | None = None
    tracking_mode: str | None = None

    detected_sources: pd.DataFrame | None = field(default=None, repr=False)
    zero_point: float | None = None
