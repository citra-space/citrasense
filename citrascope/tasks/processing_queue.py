"""Background processing queue for image processing."""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from citrascope.processors.processor_result import ProcessingContext
from citrascope.tasks.base_work_queue import BaseWorkQueue


class ProcessingQueue(BaseWorkQueue):
    """
    Background worker queue for image processing.
    Allows multiple processing tasks to run concurrently without blocking telescope.
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
            {"task_id": task_id, "image_path": image_path, "context": context, "on_complete": on_complete}
        )

    def _get_working_dir(self, task_id: str, settings) -> Path:
        """Return the task-specific working directory path."""
        if settings:
            return settings.directories.processing_dir / task_id
        return Path(tempfile.gettempdir()) / "citrascope" / "processing" / task_id

    def _cleanup_working_dir(self, task_id: str, settings):
        """Remove the task-specific working directory, logging any failure."""
        try:
            working_dir = self._get_working_dir(task_id, settings)
            if working_dir.exists():
                shutil.rmtree(working_dir)
                self.logger.debug(f"Cleaned up working directory: {working_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up working directory for {task_id}: {e}")

    def _execute_work(self, item):
        """Execute image processing work."""
        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        timing_info = item["context"].get("timing_info")

        if timing_info:
            timing_info.stamp_now("processing_started_at")

        self.logger.info(f"Processing task {task_id}")

        try:
            # Create task-specific working directory
            settings = item["context"].get("settings")
            working_dir = self._get_working_dir(task_id, settings)
            working_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created working directory: {working_dir}")

            context = ProcessingContext(
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
        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        on_complete = item["on_complete"]

        if task_obj:
            task_obj.set_status_msg("Processing complete")

        settings = item["context"].get("settings")
        if not settings or settings.processing_output_retention_hours == 0:
            self._cleanup_working_dir(task_id, settings)

        on_complete(task_id, result)

    def _on_permanent_failure(self, item):
        """Handle permanent processing failure (fail-open: upload raw image)."""
        task_id = item["task_id"]
        task_obj = item["context"].get("task")
        on_complete = item["on_complete"]

        self.logger.error(f"Task {task_id} processing permanently failed, uploading raw image")

        if task_obj:
            task_obj.set_status_msg("Processing permanently failed (uploading raw image)")

        settings = item["context"].get("settings")
        if not settings or settings.processing_output_retention_hours == 0:
            self._cleanup_working_dir(task_id, settings)

        # Fail-open: notify with None result (will upload raw image)
        on_complete(task_id, None)

    def _get_task_from_item(self, item):
        """Get Task object from work item."""
        return item["context"].get("task")
