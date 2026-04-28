"""Background upload queue for uploading images and radar observations.

The queue dispatches on ``item["kind"]``: ``"optical"`` (legacy FITS
or optical-observation path) and ``"radar"`` (one ``RadarObservationCreate``
payload per item, posted in a batch-of-one to
``POST /observations/radar``).

Completion (mark_task_complete, stage cleanup, stats) is handled by the
telescope task's _on_image_done callback after all images for a task finish.
Radar uploads run independently of any Task object.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from citrasense.acquisition.base_work_queue import BaseWorkQueue


class UploadQueue(BaseWorkQueue):
    """
    Background worker for uploading images.
    Uploads can be slow (network), so run in background.

    Note: mark_task_complete is NOT called here. The caller's on_complete
    callback aggregates per-image results and marks the task complete once
    all images for a task have finished (see _on_image_done).
    """

    def __init__(self, num_workers: int = 1, settings=None, logger=None):
        """
        Initialize upload queue.

        Args:
            num_workers: Number of concurrent upload threads (default: 1, network is bottleneck)
            settings: Settings instance with retry configuration
            logger: Logger instance
        """
        super().__init__(num_workers, settings, logger)

        # Session-scoped counters for upload path breakdown
        self.observation_uploads: int = 0
        self.image_uploads: int = 0
        self.satellites_identified: int = 0
        self.radar_observation_uploads: int = 0

    def get_stats(self) -> dict:
        """Return lifetime counters including upload path breakdown."""
        stats = super().get_stats()
        with self._stats_lock:
            stats["observation_uploads"] = self.observation_uploads
            stats["image_uploads"] = self.image_uploads
            stats["satellites_identified"] = self.satellites_identified
            stats["radar_observation_uploads"] = self.radar_observation_uploads
        return stats

    def submit(
        self,
        task_id: str,
        task,
        image_path: str,
        processing_result: dict | None,
        api_client,
        telescope_record: dict,
        settings,
        on_complete: Callable,
        sensor_location: dict | None = None,
    ):
        """
        Submit image for upload.

        Args:
            task_id: Task identifier
            task: Task object (for status updates)
            image_path: Path to image file
            processing_result: Result from processors (or None if skipped)
            api_client: API client instance
            telescope_record: Full telescope dict from the API (id, angularNoise, spectral bounds, …)
            settings: Settings instance (for keep_images flag)
            on_complete: Callback(task_id, success) when upload finishes
            sensor_location: Observer location dict with latitude/longitude/altitude keys
        """
        self.logger.info(f"Queuing task {task_id} for upload")
        self.work_queue.put(
            {
                "kind": "optical",
                "task_id": task_id,
                "task": task,
                "image_path": image_path,
                "processing_result": processing_result,
                "api_client": api_client,
                "telescope_record": telescope_record,
                "sensor_location": sensor_location,
                "settings": settings,
                "on_complete": on_complete,
            }
        )

    def submit_radar_observation(
        self,
        sensor_id: str,
        payload: dict[str, Any],
        api_client: Any,
        settings: Any,
        on_complete: Callable[[bool], None] | None = None,
    ) -> None:
        """Queue a single ``RadarObservationCreate`` dict for upload.

        The backend accepts a batch, so we always POST a list-of-one.
        Future follow-ups may introduce a tiny coalescing window (see
        issue #307's "Integration plan"), but v1 stays synchronous and
        straightforward.
        """
        task_id = f"radar:{sensor_id}:{id(payload)}"
        self.work_queue.put(
            {
                "kind": "radar",
                "task_id": task_id,
                "sensor_id": sensor_id,
                "payload": payload,
                "api_client": api_client,
                "settings": settings,
                "on_complete": on_complete,
            }
        )

    def _execute_work(self, item):
        """Execute one upload work item — optical or radar."""
        if item.get("kind") == "radar":
            return self._execute_radar(item)
        return self._execute_optical(item)

    def _execute_radar(self, item: dict[str, Any]):
        api_client = item["api_client"]
        payload = item["payload"]
        sensor_id = item["sensor_id"]
        try:
            ok = bool(api_client.upload_radar_observations([payload]))
        except Exception as exc:
            self.logger.error("Radar upload exception for sensor %s: %s", sensor_id, exc, exc_info=True)
            return (False, None)
        if not ok:
            self.logger.error("Radar upload failed for sensor %s", sensor_id)
            return (False, None)
        return (True, {"radar": True})

    def _execute_optical(self, item):
        """Execute optical upload work.

        If the processing pipeline produced satellite observations, upload them via
        POST /observations/optical and skip the FITS upload entirely. Otherwise fall
        back to uploading the FITS image as before.
        """
        task_id = item["task_id"]
        task_obj = item.get("task")

        self.logger.info(f"Uploading task {task_id}")

        # Decide which upload path to take
        sat_obs = []
        pr = item.get("processing_result")
        if pr and pr.extracted_data:
            sat_obs = pr.extracted_data.get("satellite_matcher.satellite_observations") or []

        telescope_record = item.get("telescope_record")
        telescope_id = telescope_record["id"] if telescope_record else None
        sensor_location = item.get("sensor_location")
        has_calibrated_mag = any(obs.get("mag") is not None for obs in sat_obs)
        can_upload_obs = bool(sat_obs and telescope_record and sensor_location and has_calibrated_mag)

        if sat_obs and not has_calibrated_mag:
            self.logger.warning(
                "Skipping optical observation upload for task %s: photometry failed — "
                "no calibrated magnitudes available. Falling back to FITS upload.",
                task_id,
            )

        self.logger.info(
            f"Upload path decision for task {task_id}: "
            f"sat_obs={len(sat_obs)}, telescope_record={'yes' if telescope_record else 'NO'}, "
            f"sensor_location={'yes' if sensor_location else 'NO'}, "
            f"calibrated_mag={'yes' if has_calibrated_mag else 'NO'} -> "
            f"{'observations' if can_upload_obs else 'FITS image'}"
        )

        if can_upload_obs:
            # Observations-only path: post structured data, no FITS needed
            if task_obj:
                task_obj.set_status_msg("Uploading observations...")
            self.logger.info(f"Uploading {len(sat_obs)} satellite observation(s) for task {task_id}")
            upload_ok = item["api_client"].upload_optical_observations(
                sat_obs, telescope_record, sensor_location, task_id=task_id
            )
        else:
            # Standard FITS upload path
            if sat_obs and not (telescope_record and sensor_location):
                self.logger.warning(
                    f"Have {len(sat_obs)} sat obs but missing telescope_record "
                    f"or sensor_location — falling back to FITS upload for task {task_id}"
                )
            if task_obj:
                task_obj.set_status_msg("Uploading image...")
            upload_ok = item["api_client"].upload_image(task_id, telescope_id, item["image_path"])

        if not upload_ok:
            self.logger.error(f"Upload failed for {task_id}")
            return (False, None)

        self.logger.info(f"Upload succeeded for task {task_id}")
        return (True, {"obs_path": can_upload_obs})

    def _on_success(self, item, result):
        """Handle successful upload completion."""
        if item.get("kind") == "radar":
            self.radar_observation_uploads += 1
            self.logger.info("Radar observation uploaded for sensor %s", item.get("sensor_id"))
            on_complete = item.get("on_complete")
            if on_complete:
                try:
                    on_complete(True)
                except Exception as exc:
                    self.logger.warning("Radar upload on_complete raised: %s", exc)
            return

        task_id = item["task_id"]
        task_obj = item.get("task")
        on_complete = item["on_complete"]
        obs_path = (result or {}).get("obs_path", False)

        if obs_path:
            self.observation_uploads += 1
            pr = item.get("processing_result")
            if pr and pr.extracted_data:
                sat_obs = pr.extracted_data.get("satellite_matcher.satellite_observations") or []
                self.satellites_identified += len(sat_obs)
            if task_obj:
                task_obj.set_status_msg("Observations uploaded")
        else:
            self.image_uploads += 1
            if task_obj:
                task_obj.set_status_msg("Upload complete")

        # Cleanup local FITS and working files if configured.
        # This runs on both the obs-only and FITS upload paths — when we took the obs path
        # the FITS was never sent to the server and should still be cleaned up locally.
        if not item["settings"].keep_images:
            if task_obj:
                task_obj.set_status_msg("Cleaning up...")
            self._cleanup_files(item["image_path"])

        on_complete(task_id, success=True)

    def _on_permanent_failure(self, item):
        """Handle permanent upload failure.

        Stage cleanup is deferred to the on_complete callback (_on_image_done)
        which waits until all images for the task have finished.
        """
        if item.get("kind") == "radar":
            self.logger.error("Radar observation upload permanently failed for sensor %s", item.get("sensor_id"))
            on_complete = item.get("on_complete")
            if on_complete:
                try:
                    on_complete(False)
                except Exception as exc:
                    self.logger.warning("Radar upload on_complete raised: %s", exc)
            return

        task_id = item["task_id"]
        task_obj = item.get("task")
        on_complete = item["on_complete"]

        self.logger.error(f"Task {task_id} upload permanently failed")

        if task_obj:
            task_obj.set_status_msg("Upload permanently failed")

        on_complete(task_id, success=False)

    def _get_task_from_item(self, item):
        """Get Task object from work item."""
        if item.get("kind") == "radar":
            return None
        return item.get("task")

    def _cleanup_files(self, filepath: str):
        """Clean up image files after successful upload."""
        try:
            path = Path(filepath)

            # Delete main file
            if path.exists():
                path.unlink()
                self.logger.debug(f"Deleted {filepath}")

            # Delete related files (.new, .cat, .wcs, etc.)
            related_suffixes = {".new", ".cat", ".wcs", ".solved", ".axy", ".corr", ".match", ".rdls"}
            for related in path.parent.glob(f"{path.stem}.*"):
                if related != path and related.suffix in related_suffixes:
                    related.unlink()
                    self.logger.debug(f"Deleted {related}")

        except Exception as e:
            self.logger.warning(f"Failed to cleanup files for {filepath}: {e}")
