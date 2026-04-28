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

Per-processor stats (``runs`` / ``failures`` / ``last_failure_reason``)
are tracked with the same shape the optical
:class:`~citrasense.pipelines.common.pipeline_registry.PipelineRegistry`
uses so the web UI can render radar and telescope processors through
the same ``status.pipeline_stats.processors`` aggregation.
"""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any

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

        self._stats_lock = threading.Lock()
        self._processor_stats: dict[str, dict[str, Any]] = {
            name: {"runs": 0, "failures": 0, "last_failure_reason": None}
            for name in (self._filter.name, self._formatter.name, self._writer.name)
        }

    @property
    def processors(self) -> list[Any]:
        """Mirror ``PipelineRegistry.processors`` for symmetry with the optical
        stats aggregator.  Order is always filter → formatter → writer."""
        return [self._filter, self._formatter, self._writer]

    def get_processor_stats(self) -> dict[str, dict[str, Any]]:
        """Snapshot of per-processor lifetime stats.

        Matches :meth:`PipelineRegistry.get_processor_stats` so the web
        status collector can pour both dicts into the same
        ``pipeline_stats.processors`` aggregate without caring about
        modality.
        """
        with self._stats_lock:
            return copy.deepcopy(self._processor_stats)

    def process(self, ctx: RadarProcessingContext) -> bool:
        """Run the chain. Returns ``True`` iff the observation is
        ready for upload (i.e., passed filter AND formatter).

        The artifact writer runs regardless — dropped observations are
        still persisted so operators can audit filter decisions.
        """
        passed_filter = self._filter.process(ctx)
        self._record_run(self._filter.name, failed=not passed_filter, reason=ctx.drop_reason)

        formatted = False
        if passed_filter:
            formatted = self._formatter.process(ctx)
            self._record_run(
                self._formatter.name,
                failed=not formatted,
                reason=ctx.drop_reason if not formatted else None,
            )

        try:
            self._writer.process(ctx)
            self._record_run(self._writer.name, failed=False, reason=None)
        except Exception as exc:
            self._record_run(self._writer.name, failed=True, reason=str(exc))
            if ctx.logger:
                ctx.logger.warning("Radar artifact writer failed: %s", exc)

        return passed_filter and formatted

    def _record_run(self, name: str, *, failed: bool, reason: str | None) -> None:
        with self._stats_lock:
            s = self._processor_stats.get(name)
            if s is None:
                return
            s["runs"] += 1
            if failed:
                s["failures"] += 1
                if reason:
                    s["last_failure_reason"] = reason
