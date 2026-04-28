"""Ordered radar processor chain.

The optical pipeline uses :class:`PipelineRegistry` to iterate image
processors.  That machinery is image-shaped
(``AbstractImageProcessor.process(ProcessingContext)``); for radar we
don't have a shared image-like input, so this module provides a thin,
purpose-built chain that mirrors the spirit (filter → formatter →
artifact writer) without trying to force radar processors into the
optical registry.

The chain is intentionally short and data-oriented — each processor
mutates the :class:`RadarProcessingContext` in place and returns a
``bool`` indicating whether the next step should run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from citrasense.pipelines.radar.radar_artifact_writer import RadarArtifactWriter
from citrasense.pipelines.radar.radar_detection_filter import RadarDetectionFilter
from citrasense.pipelines.radar.radar_detection_formatter import RadarDetectionFormatter

if TYPE_CHECKING:
    from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext


class RadarPipeline:
    """Filter → format → write artifact. Artifact writer always runs."""

    def __init__(
        self,
        *,
        filter_: RadarDetectionFilter | None = None,
        formatter: RadarDetectionFormatter | None = None,
        writer: RadarArtifactWriter | None = None,
    ) -> None:
        self._filter = filter_ or RadarDetectionFilter()
        self._formatter = formatter or RadarDetectionFormatter()
        self._writer = writer or RadarArtifactWriter()

    def process(self, ctx: RadarProcessingContext) -> bool:
        """Run the chain. Returns ``True`` iff the observation is
        ready for upload (i.e., passed filter AND formatter).

        The artifact writer runs regardless — dropped observations are
        still persisted so operators can audit filter decisions.
        """
        passed_filter = self._filter.process(ctx)
        formatted = False
        if passed_filter:
            formatted = self._formatter.process(ctx)

        # Artifact writer is best-effort and never vetoes upload.
        try:
            self._writer.process(ctx)
        except Exception as exc:
            if ctx.logger:
                ctx.logger.warning("Radar artifact writer failed: %s", exc)

        return passed_filter and formatted
