"""Registry for managing and executing image processors."""

import copy
import threading
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from citrascope.processors.abstract_processor import AbstractImageProcessor
from citrascope.processors.artifact_writer import dump_context_artifacts, dump_processing_summary
from citrascope.processors.builtin.annotated_image_processor import AnnotatedImageProcessor
from citrascope.processors.builtin.calibration_processor import CalibrationProcessor
from citrascope.processors.builtin.photometry_processor import PhotometryProcessor
from citrascope.processors.builtin.plate_solver_processor import PlateSolverProcessor
from citrascope.processors.builtin.satellite_matcher_processor import SatelliteMatcherProcessor
from citrascope.processors.processor_result import AggregatedResult, ProcessingContext, ProcessorResult
from citrascope.processors.report_generator import generate_html_report


class ProcessorRegistry:
    """Manages and executes image processors."""

    def __init__(self, settings, logger):
        """Initialize the processor registry.

        Args:
            settings: CitraScopeSettings instance
            logger: Logger instance for diagnostics
        """
        self.settings = settings
        self.logger = logger

        # Hardcode processor list (simple, explicit).
        # Source extraction is handled inline by PlateSolverProcessor (reuses SEP
        # detections from the plate solve pass) — no separate SExtractor step needed.
        self.processors: list[AbstractImageProcessor] = [
            CalibrationProcessor(),  # Step 0: apply bias/dark/flat calibration
            PlateSolverProcessor(),  # Step 1: plate solve + source extraction
            PhotometryProcessor(),  # Step 2: 2-5s (uses context.detected_sources)
            SatelliteMatcherProcessor(),  # Step 3: 1-2s (requires photometry)
            AnnotatedImageProcessor(),  # Step 4: <1s (renders visual overlay)
        ]

        # Per-processor lifetime stats — lock gives atomic multi-field snapshots in get_processor_stats().
        self._stats_lock = threading.Lock()
        self._processor_stats: dict = {
            p.name: {"runs": 0, "failures": 0, "last_failure_reason": None} for p in self.processors
        }

        # Log registered processors on startup
        processor_names = [p.name for p in self.processors]
        self.logger.info(f"ProcessorRegistry initialized with {len(self.processors)} processors: {processor_names}")

        # Pre-download and load the Tetra3 star database so first solve is immediate
        PlateSolverProcessor.warm_cache(logger=self.logger)

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
                "enabled": self.settings.enabled_processors.get(p.name, True),  # Default to enabled
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

        # Load image once for all processors if not already loaded
        if context.image_data is None:
            context.image_data = self._load_image(context.image_path)

        dump_context_artifacts(context)

        # Filter to only enabled processors
        enabled_processors = [
            p for p in self.processors if self.settings.enabled_processors.get(p.name, True)  # Default to enabled
        ]

        # Log enabled vs disabled processors
        enabled_names = [p.name for p in enabled_processors]
        disabled_names = [p.name for p in self.processors if p not in enabled_processors]
        self.logger.info(f"Processing with {len(enabled_processors)} enabled processors: {enabled_names}")
        if disabled_names:
            self.logger.info(f"Skipping {len(disabled_names)} disabled processors: {disabled_names}")

        results = []
        for processor in enabled_processors:
            # Update status message to show which processor is running
            if context.task:
                context.task.set_status_msg(f"Running {processor.friendly_name}...")

            self.logger.info(f"Starting processor: {processor.name} ({processor.friendly_name})")
            proc_start = time.time()

            # Let exceptions propagate - they will trigger retries in ProcessingQueue
            # After max retries, ProcessingQueue will fail-open and upload raw image
            result = processor.process(context)
            results.append(result)

            proc_elapsed = time.time() - proc_start

            # Update per-processor stats (key may be absent for processors injected after __init__)
            with self._stats_lock:
                s = self._processor_stats.get(processor.name)
                if s is not None:
                    s["runs"] += 1
                    if result.confidence == 0.0:
                        s["failures"] += 1
                        s["last_failure_reason"] = result.reason

            # Log failures as warnings, successes as info
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
        generate_html_report(context.working_dir)

        return aggregated

    def _aggregate_results(self, results: list[ProcessorResult], total_time: float) -> AggregatedResult:
        """Combine processor results into upload decision.

        Logic:
        - If ANY processor says don't upload → don't upload
        - Combine extracted_data with processor name prefixes

        Args:
            results: List of individual processor results
            total_time: Total processing time in seconds

        Returns:
            AggregatedResult with combined decision and data
        """
        should_upload = all(r.should_upload for r in results) if results else True

        # Combine extracted data with processor name prefixes to avoid collisions
        combined_data = {}
        for result in results:
            for key, value in result.extracted_data.items():
                prefixed_key = f"{result.processor_name}.{key}"
                combined_data[prefixed_key] = value

        # Find first rejection reason if any
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
        """Load image from FITS file.

        Args:
            image_path: Path to FITS file

        Returns:
            Numpy array with image data
        """
        data = fits.getdata(image_path)
        return np.asarray(data)
