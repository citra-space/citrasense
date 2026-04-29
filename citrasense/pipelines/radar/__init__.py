"""Passive-radar pipeline.

Radar processors don't fit the optical ``AbstractImageProcessor``
contract (no FITS, no image), so they live outside
:class:`~citrasense.pipelines.common.pipeline_registry.PipelineRegistry`.
The chain is assembled via :func:`build_radar_pipeline`.
"""

from __future__ import annotations

from citrasense.pipelines.radar.radar_artifact_writer import RadarArtifactWriter, radar_artifact_dir
from citrasense.pipelines.radar.radar_detection_filter import RadarDetectionFilter
from citrasense.pipelines.radar.radar_detection_formatter import RadarDetectionFormatter
from citrasense.pipelines.radar.radar_pipeline import RadarPipeline
from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext


def build_radar_pipeline() -> RadarPipeline:
    """Factory for the radar filter → formatter → writer chain."""
    return RadarPipeline()


__all__ = [
    "RadarArtifactWriter",
    "RadarDetectionFilter",
    "RadarDetectionFormatter",
    "RadarPipeline",
    "RadarProcessingContext",
    "build_radar_pipeline",
    "radar_artifact_dir",
]
