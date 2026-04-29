"""Background processing queue for image processing.

Two item shapes flow through the queue, dispatched on ``item["kind"]``:

- ``"optical"`` (default, legacy): a telescope task's captured FITS
  image threaded through the optical :class:`PipelineRegistry`.
- ``"radar"``: a single :class:`RadarProcessingContext` carrying one
  ``pr_sensor`` observation event; the radar chain (filter → formatter
  → artifact writer) runs and either enqueues an upload or drops the
  observation with a logged reason.

Radar work does not retry — the chain is deterministic and idempotent
— so failures on that path are surfaced as ``_on_permanent_failure``
immediately by the base worker loop after the (small) max-retry
counter exhausts.  In practice the radar path either succeeds or the
observation is dropped by a filter, neither of which raises.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from citrasense.acquisition.base_work_queue import BaseWorkQueue
from citrasense.pipelines.optical.optical_processing_context import OpticalProcessingContext

if TYPE_CHECKING:
    from citrasense.pipelines.radar.radar_pipeline import RadarPipeline
    from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext


class ProcessingQueue(BaseWorkQueue):
    """
    Background worker queue for image processing and radar observation
    processing.
    """

    def __init__(self, num_workers: int = 1, settings=None, logger=None):
        """
        Initialize processing queue.

        Args:
            num_workers: Number of concurrent processing threads (default: 1)
            settings: Settings instance with retry configuration
            logger: Logger instance
        """
        super().__init__(num_workers, settings, logger)

    def submit(self, task_id: str, image_path: Path, context: dict, on_complete: Callable):
        """
        Submit image for processing.

        Args:
            task_id: Task identifier
            image_path: Path to captured image
            context: Processing context (task, settings, daemon, etc.)
            on_complete: Callback(task_id, result) when processing finishes
        """
        self.logger.info(f"Queuing task {task_id} for processing")
        self.work_queue.put(
            {
                "kind": "optical",
                "task_id": task_id,
                "image_path": image_path,
                "context": context,
                "on_complete": on_complete,
            }
        )

    def submit_radar_event(
        self,
        ctx: RadarProcessingContext,
        pipeline: RadarPipeline,
        on_complete: Callable[[RadarProcessingContext, bool], None],
    ) -> None:
        """Submit a single radar observation for processing.

        Args:
            ctx: Radar processing context carrying the raw observation
                event plus per-sensor configuration (antenna UUID,
                filter thresholds, artifact dir, ...).
            pipeline: The sensor's cached :class:`RadarPipeline`
                (filter → formatter → artifact writer).
            on_complete: Callback invoked with ``(ctx, upload_ready)``
                after processing finishes.  ``upload_ready=True``
                means the caller should submit ``ctx.upload_payload``
                to the upload queue.
        """
        task_id = f"radar:{ctx.sensor_id}:{id(ctx.event)}"
        self.work_queue.put(
            {
                "kind": "radar",
                "task_id": task_id,
                "ctx": ctx,
                "pipeline": pipeline,
                "on_complete": on_complete,
            }
        )

    def _get_working_dir(self, task_id: str, sensor_id: str = "") -> Path:
        """Return the task-specific working directory path.

        In multi-sensor deployments each task is namespaced by ``sensor_id``
        so two sensors can never collide on a shared ``processing/<task_id>``
        directory — even if the API task IDs ever stop being globally
        unique.  Legacy single-sensor tasks (no ``sensor_id`` in the
        runtime context) fall back to the flat layout for backwards
        compatibility with existing on-disk artifacts.
        """
        if self.settings and getattr(self.settings, "directories", None):
            base = self.settings.directories.processing_dir
        else:
            base = Path(tempfile.gettempdir()) / "citrasense" / "processing"
        if sensor_id:
            return base / sensor_id / task_id
        return base / task_id

    def _cleanup_working_dir(self, task_id: str, sensor_id: str = ""):
        """Remove the task-specific working directory, logging any failure."""
        try:
            working_dir = self._get_working_dir(task_id, sensor_id)
            if working_dir.exists():
                shutil.rmtree(working_dir)
                self.logger.debug(f"Cleaned up working directory: {working_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up working directory for {task_id}: {e}")

    def _execute_work(self, item):
        """Execute one queued work item — optical image or radar observation."""
        if item.get("kind") == "radar":
            return self._execute_radar(item)
        return self._execute_optical(item)

    def _execute_radar(self, item: dict[str, Any]):
        ctx: RadarProcessingContext = item["ctx"]
        pipeline: RadarPipeline = item["pipeline"]
        try:
            upload_ready = pipeline.process(ctx)
        except Exception as exc:
            self.logger.error("Radar pipeline failed for sensor %s: %s", ctx.sensor_id, exc, exc_info=True)
            return (False, None)
        return (True, {"upload_ready": upload_ready})

    def _execute_optical(self, item):
        """Execute image processing work."""
        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        timing_info = item["context"].get("timing_info")
        sensor_id = item["context"].get("sensor_id", "") or ""

        if timing_info:
            timing_info.stamp_now("processing_started_at")

        self.logger.info(f"Processing task {task_id}")

        try:
            # Create task-specific working directory
            working_dir = self._get_working_dir(task_id, sensor_id)
            working_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created working directory: {working_dir}")

            context = OpticalProcessingContext(
                sensor_id=sensor_id,
                image_path=item["image_path"],
                working_image_path=item["image_path"],
                working_dir=working_dir,
                image_data=None,
                task=task_obj,
                telescope_record=item["context"].get("telescope_record"),
                ground_station_record=item["context"].get("ground_station_record"),
                settings=item["context"].get("settings"),
                location_service=item["context"].get("location_service"),
                elset_cache=item["context"].get("elset_cache"),
                apass_catalog=item["context"].get("apass_catalog"),
                satellite_data=item["context"].get("satellite_data"),
                pointing_report=item["context"].get("pointing_report"),
                tracking_mode=item["context"].get("tracking_mode"),
                logger=self.logger,
            )
            processor_registry = item["context"].get("processor_registry")
            if processor_registry is None:
                raise ValueError(f"No processor_registry in context for task {task_id}")
            result = processor_registry.process_all(context)

            if timing_info:
                timing_info.stamp_now("processing_finished_at")

            # Promote final processed image back to the original file location
            # so the upload queue sends the calibrated + plate-solved version.
            # Save a copy of the raw original into the working dir first so
            # operators can compare raw vs. processed when keep_processing_output is on.
            if context.working_image_path != context.image_path:
                raw_backup = working_dir / f"original_{context.image_path.name}"
                shutil.copy2(context.image_path, raw_backup)
                shutil.copy2(context.working_image_path, context.image_path)
                self.logger.info(
                    "Promoted processed image to %s (raw backup in working dir)",
                    context.image_path.name,
                )

            # Success
            self.logger.info(f"Task {task_id} processed in {result.total_time:.2f}s")
            return (True, result)

        except Exception as e:
            self.logger.error(f"Processing failed for {task_id}: {e}", exc_info=True)
            return (False, None)

    def _on_success(self, item, result):
        """Handle successful processing completion."""
        if item.get("kind") == "radar":
            ctx: RadarProcessingContext = item["ctx"]
            upload_ready = bool((result or {}).get("upload_ready"))
            try:
                item["on_complete"](ctx, upload_ready)
            except Exception as exc:
                self.logger.error("Radar on_complete raised: %s", exc, exc_info=True)
            return

        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        sensor_id = item["context"].get("sensor_id", "") or ""
        on_complete = item["on_complete"]

        if task_obj:
            task_obj.set_status_msg("Processing complete")

        retention = getattr(self.settings, "processing_output_retention_hours", 0)
        if retention == 0:
            self._cleanup_working_dir(task_id, sensor_id)

        on_complete(task_id, result)

    def _on_permanent_failure(self, item):
        """Handle permanent processing failure (fail-open: upload raw image)."""
        if item.get("kind") == "radar":
            ctx: RadarProcessingContext = item["ctx"]
            ctx.drop_reason = ctx.drop_reason or "radar pipeline permanently failed"
            self.logger.error(
                "Radar observation for sensor %s permanently failed processing: %s",
                ctx.sensor_id,
                ctx.drop_reason,
            )
            try:
                item["on_complete"](ctx, False)
            except Exception as exc:
                self.logger.error("Radar on_complete raised on failure: %s", exc, exc_info=True)
            return

        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        sensor_id = item["context"].get("sensor_id", "") or ""
        on_complete = item["on_complete"]

        self.logger.error(f"Task {task_id} processing permanently failed, uploading raw image")

        if task_obj:
            task_obj.set_status_msg("Processing permanently failed (uploading raw image)")

        retention = getattr(self.settings, "processing_output_retention_hours", 0)
        if retention == 0:
            self._cleanup_working_dir(task_id, sensor_id)

        # Fail-open: notify with None result (will upload raw image)
        on_complete(task_id, None)

    def _get_task_from_item(self, item):
        """Get Task object from work item."""
        if item.get("kind") == "radar":
            return None
        return item["context"].get("task")
