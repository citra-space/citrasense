"""Modality-keyed pipeline registry for managing and executing processors."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from citrasense.pipelines.common.abstract_processor import AbstractImageProcessor
from citrasense.pipelines.common.artifact_writer import dump_processing_summary
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import AggregatedResult, ProcessorResult

if TYPE_CHECKING:
    pass

ContextHook = Callable[[ProcessingContext], None]
SummaryHook = Callable[[Path, AggregatedResult], None]


@dataclass
class PipelineDefinition:
    """Everything needed to instantiate a pipeline for a given modality."""

    factory: Callable[[], list[AbstractImageProcessor]]
    pre_hooks: list[ContextHook] = field(default_factory=list)
    post_hooks: list[SummaryHook] = field(default_factory=list)


def _build_optical_pipeline() -> PipelineDefinition:
    """Lazy-import and return the optical pipeline definition."""
    from citrasense.pipelines.optical.optical_artifacts import dump_optical_context_artifacts
    from citrasense.pipelines.optical.report_generator import generate_html_report

    def _build_processors() -> list[AbstractImageProcessor]:
        from citrasense.pipelines.optical.annotated_image_processor import AnnotatedImageProcessor
        from citrasense.pipelines.optical.calibration_processor import CalibrationProcessor
        from citrasense.pipelines.optical.photometry_processor import PhotometryProcessor
        from citrasense.pipelines.optical.plate_solver_processor import PlateSolverProcessor
        from citrasense.pipelines.optical.satellite_matcher_processor import SatelliteMatcherProcessor
        from citrasense.pipelines.optical.source_extractor_processor import SourceExtractorProcessor

        return [
            CalibrationProcessor(),
            PlateSolverProcessor(),
            SourceExtractorProcessor(),
            PhotometryProcessor(),
            SatelliteMatcherProcessor(),
            AnnotatedImageProcessor(),
        ]

    def _post_report(working_dir: Path, _aggregated: AggregatedResult) -> None:
        generate_html_report(working_dir)

    return PipelineDefinition(
        factory=_build_processors,
        pre_hooks=[dump_optical_context_artifacts],
        post_hooks=[_post_report],
    )


_PIPELINE_DEFS: dict[str, Callable[[], PipelineDefinition]] = {
    "optical": _build_optical_pipeline,
    "radar": lambda: PipelineDefinition(factory=lambda: []),
    "rf": lambda: PipelineDefinition(factory=lambda: []),
}


def get_pipeline(modality: str) -> list[AbstractImageProcessor]:
    """Return an ordered processor list for *modality*.

    Raises:
        ValueError: If *modality* is not registered.
    """
    return _get_definition(modality).factory()


def _get_definition(modality: str) -> PipelineDefinition:
    builder = _PIPELINE_DEFS.get(modality)
    if builder is None:
        available = ", ".join(f"'{m}'" for m in _PIPELINE_DEFS)
        raise ValueError(f"Unknown pipeline modality: '{modality}'. Valid options are: {available}")
    return builder()


def list_pipelines() -> list[str]:
    """Return all registered pipeline modality keys."""
    return list(_PIPELINE_DEFS.keys())


class PipelineRegistry:
    """Manages and executes processors for a given modality."""

    def __init__(self, settings, logger, modality: str = "optical"):
        """Initialize the pipeline registry.

        Args:
            settings: CitraSenseSettings instance
            logger: Logger instance for diagnostics
            modality: Pipeline modality key (default ``"optical"``)
        """
        self.settings = settings
        self.logger = logger.getChild(type(self).__name__)
        self.modality = modality

        defn = _get_definition(modality)
        self.processors: list[AbstractImageProcessor] = defn.factory()
        self._pre_hooks: list[ContextHook] = defn.pre_hooks
        self._post_hooks: list[SummaryHook] = defn.post_hooks

        self._stats_lock = threading.Lock()
        self._processor_stats: dict = {
            p.name: {"runs": 0, "failures": 0, "last_failure_reason": None} for p in self.processors
        }

        processor_names = [p.name for p in self.processors]
        self.logger.info(
            f"PipelineRegistry[{modality}] initialized with {len(self.processors)} processors: {processor_names}"
        )

    def get_processor_stats(self) -> dict:
        """Return a consistent snapshot of per-processor lifetime stats."""
        with self._stats_lock:
            return copy.deepcopy(self._processor_stats)

    def get_all_processors(self) -> list[dict]:
        """Get metadata for all processors (enabled and disabled).

        Returns:
            List of dicts with processor metadata
        """
        return [
            {
                "name": p.name,
                "friendly_name": p.friendly_name,
                "description": p.description,
                "enabled": self.settings.enabled_processors.get(p.name, True),
            }
            for p in self.processors
        ]

    def process_all(self, context: ProcessingContext) -> AggregatedResult:
        """Run all enabled processors on an image.

        Args:
            context: ProcessingContext with image and task data

        Returns:
            AggregatedResult with upload decision and combined data
        """
        start_time = time.time()

        if context.image_data is None:
            context.image_data = self._load_image(context.image_path)

        for hook in self._pre_hooks:
            hook(context)

        enabled_processors = [p for p in self.processors if self.settings.enabled_processors.get(p.name, True)]

        enabled_names = [p.name for p in enabled_processors]
        disabled_names = [p.name for p in self.processors if p not in enabled_processors]
        self.logger.info(f"Processing with {len(enabled_processors)} enabled processors: {enabled_names}")
        if disabled_names:
            self.logger.info(f"Skipping {len(disabled_names)} disabled processors: {disabled_names}")

        results = []
        for processor in enabled_processors:
            if context.task:
                context.task.set_status_msg(f"Running {processor.friendly_name}...")

            self.logger.info(f"Starting processor: {processor.name} ({processor.friendly_name})")
            proc_start = time.time()

            result = processor.process(context)
            results.append(result)

            proc_elapsed = time.time() - proc_start

            with self._stats_lock:
                s = self._processor_stats.get(processor.name)
                if s is not None:
                    s["runs"] += 1
                    if result.confidence == 0.0:
                        s["failures"] += 1
                        s["last_failure_reason"] = result.reason

            if result.confidence == 0.0 or not result.should_upload:
                self.logger.warning(
                    f"Processor {processor.name} FAILED in {proc_elapsed:.2f}s: "
                    f"confidence={result.confidence:.2f}, should_upload={result.should_upload}, "
                    f"reason='{result.reason}'"
                )
            else:
                self.logger.info(
                    f"Processor {processor.name} completed in {proc_elapsed:.2f}s: "
                    f"confidence={result.confidence:.2f}, should_upload={result.should_upload}, "
                    f"reason='{result.reason}', extracted_keys={list(result.extracted_data.keys())}"
                )

        total_time = time.time() - start_time
        aggregated = self._aggregate_results(results, total_time)
        self.logger.info(
            f"All processors completed in {total_time:.2f}s. "
            f"Total extracted keys: {len(aggregated.extracted_data)}, should_upload={aggregated.should_upload}"
        )

        dump_processing_summary(context.working_dir, aggregated)
        for hook in self._post_hooks:
            hook(context.working_dir, aggregated)

        return aggregated

    def _aggregate_results(self, results: list[ProcessorResult], total_time: float) -> AggregatedResult:
        """Combine processor results into upload decision."""
        should_upload = all(r.should_upload for r in results) if results else True

        combined_data = {}
        for result in results:
            for key, value in result.extracted_data.items():
                prefixed_key = f"{result.processor_name}.{key}"
                combined_data[prefixed_key] = value

        skip_reason = None
        for result in results:
            if not result.should_upload:
                skip_reason = f"{result.processor_name}: {result.reason}"
                break

        return AggregatedResult(
            should_upload=should_upload,
            extracted_data=combined_data,
            all_results=results,
            total_time=total_time,
            skip_reason=skip_reason,
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from FITS file."""
        data = fits.getdata(image_path)
        return np.asarray(data)


ProcessorRegistry = PipelineRegistry
