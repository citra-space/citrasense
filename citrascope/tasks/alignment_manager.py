"""Mount alignment management: on-demand and startup plate-solve-and-sync."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citrascope.citra_scope_daemon import CitraScopeDaemon
    from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrascope.tasks.base_work_queue import BaseWorkQueue


class AlignmentManager:
    """Manages on-demand and startup plate-solve alignment.

    Takes a short exposure at the mount's current position, plate-solves it,
    and syncs the mount.  No slew is needed — the solve is blind.

    Follows the same request/check_and_execute pattern as AutofocusManager.
    """

    def __init__(
        self,
        logger: logging.Logger,
        hardware_adapter: AbstractAstroHardwareAdapter,
        daemon: CitraScopeDaemon,
        imaging_queue: BaseWorkQueue | None = None,
    ):
        self.logger = logger
        self.hardware_adapter = hardware_adapter
        self.daemon = daemon
        self.imaging_queue = imaging_queue
        self._requested = False
        self._running = False
        self._progress = ""
        self._lock = threading.Lock()

    def request(self) -> bool:
        """Request alignment to run at next safe point between tasks."""
        with self._lock:
            self._requested = True
            self.logger.info("Alignment requested — will run between tasks")
            return True

    def cancel(self) -> bool:
        """Cancel pending alignment request.

        Returns:
            True if alignment was cancelled, False if nothing to cancel.
        """
        with self._lock:
            was_requested = self._requested
            self._requested = False
            if was_requested:
                self.logger.info("Alignment request cancelled")
            return was_requested

    def is_requested(self) -> bool:
        with self._lock:
            return self._requested

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def progress(self) -> str:
        with self._lock:
            return self._progress

    def _set_progress(self, msg: str) -> None:
        with self._lock:
            self._progress = msg

    def check_and_execute(self) -> bool:
        """Check if alignment should run and execute if so.

        Call this between tasks in the runner loop.  Returns True if alignment ran.
        Waits for the imaging queue to drain before starting.
        """
        with self._lock:
            should_run = self._requested
            if should_run:
                self._requested = False

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Alignment deferred — waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        self._execute()
        return True

    def _get_exposure_seconds(self) -> float:
        if self.daemon.settings:
            return getattr(self.daemon.settings, "alignment_exposure_seconds", 2.0)
        return 2.0

    def _execute(self) -> None:
        """Execute alignment: take image → plate solve → sync mount."""
        with self._lock:
            self._running = True
            self._progress = "Starting alignment..."
        try:
            telescope_record: dict[str, Any] | None = getattr(self.hardware_adapter, "telescope_record", None)
            if not telescope_record:
                self.logger.error("Cannot align — no telescope_record available (configure telescope in Citra)")
                return

            camera: Any = getattr(self.hardware_adapter, "camera", None)
            mount: Any = getattr(self.hardware_adapter, "mount", None)
            if not camera or not mount:
                self.logger.error("Cannot align — camera and mount are both required")
                return

            from citrascope.processors.builtin.plate_solver_processor import PlateSolverProcessor

            safety_monitor = getattr(self.daemon, "safety_monitor", None)
            if safety_monitor and not safety_monitor.is_action_safe("capture"):
                self.logger.warning("Alignment aborted — safety monitor blocked capture")
                return

            exposure_s = self._get_exposure_seconds()
            self._set_progress(f"Exposing ({exposure_s:.0f}s)...")
            self.logger.info(f"Alignment: taking {exposure_s:.0f}s exposure...")

            try:
                image_path = self.hardware_adapter.take_image("alignment", exposure_s)
            except Exception as exc:
                self.logger.error(f"Alignment exposure failed: {exc}")
                return

            self._set_progress("Plate solving...")
            result = PlateSolverProcessor.solve(Path(image_path), telescope_record)

            if result is None:
                self.logger.error("Alignment failed — plate solve returned no solution")
                return

            solved_ra, solved_dec = result
            self._set_progress("Syncing mount...")
            mount.sync_to_radec(solved_ra, solved_dec)
            self.logger.info(f"Alignment successful: synced to RA={solved_ra:.4f}°, Dec={solved_dec:.4f}°")

        except Exception as e:
            self.logger.error(f"Alignment failed: {e!s}", exc_info=True)
        finally:
            with self._lock:
                self._running = False
                self._progress = ""
            if self.daemon.settings:
                self.daemon.settings.last_alignment_timestamp = int(time.time())
                self.daemon.settings.save()
