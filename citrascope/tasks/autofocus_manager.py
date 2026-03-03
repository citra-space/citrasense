"""Autofocus management: scheduling, target resolution, and execution."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from citrascope.constants import AUTOFOCUS_TARGET_PRESETS
from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter

if TYPE_CHECKING:
    from citrascope.citra_scope_daemon import CitraScopeDaemon
    from citrascope.tasks.base_work_queue import BaseWorkQueue


class AutofocusManager:
    """Manages autofocus requests, scheduling, and execution.

    Owns the autofocus request flag and lock, determines when scheduled
    autofocus should run, resolves the target star from settings, and
    executes the routine via the hardware adapter.
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
        self._cancel_event = threading.Event()

    def request(self) -> bool:
        """Request autofocus to run at next safe point between tasks."""
        with self._lock:
            self._requested = True
            self.logger.info("Autofocus requested - will run between tasks")
            return True

    def cancel(self) -> bool:
        """Cancel autofocus whether it is queued or actively running.

        Returns:
            True if something was cancelled, False if nothing to cancel.
        """
        with self._lock:
            was_requested = self._requested
            is_running = self._running
            self._requested = False

        if is_running:
            self._cancel_event.set()
            self.logger.info("Autofocus cancellation requested (run in progress)")
            return True
        if was_requested:
            self.logger.info("Autofocus request cancelled")
            return True
        return False

    def is_requested(self) -> bool:
        """Check if autofocus is currently requested/queued."""
        with self._lock:
            return self._requested

    def is_running(self) -> bool:
        """Check if autofocus is currently executing."""
        with self._lock:
            return self._running

    @property
    def progress(self) -> str:
        """Current autofocus progress description (empty if not running)."""
        with self._lock:
            return self._progress

    def _set_progress(self, msg: str) -> None:
        with self._lock:
            self._progress = msg

    def check_and_execute(self) -> bool:
        """Check if autofocus should run (manual or scheduled) and execute if so.

        Call this between tasks in the runner loop. Returns True if autofocus ran.
        Waits for the imaging queue to drain before starting so we don't slew
        mid-exposure.
        """
        with self._lock:
            should_run = self._requested
            if should_run:
                self._requested = False
            elif self._should_run_scheduled():
                should_run = True
                self._requested = False

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Autofocus deferred - waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        self._execute()
        return True

    def _should_run_scheduled(self) -> bool:
        """Check if scheduled autofocus should run based on settings."""
        if not self.daemon.settings:
            return False

        if not self.daemon.settings.scheduled_autofocus_enabled:
            return False

        if not self.hardware_adapter.supports_autofocus():
            return False

        interval_minutes = self.daemon.settings.autofocus_interval_minutes
        last_timestamp = self.daemon.settings.last_autofocus_timestamp

        if last_timestamp is None:
            return True

        elapsed_minutes = (int(time.time()) - last_timestamp) / 60
        return elapsed_minutes >= interval_minutes

    def _resolve_target(self) -> tuple[float | None, float | None]:
        """Resolve autofocus target RA/Dec from settings (preset or custom)."""
        settings = self.daemon.settings
        if not settings:
            return None, None

        preset_key = settings.autofocus_target_preset or "mirach"

        if preset_key == "custom":
            ra = settings.autofocus_target_custom_ra
            dec = settings.autofocus_target_custom_dec
            if ra is not None and dec is not None:
                self.logger.info(f"Autofocus target: custom (RA={ra:.4f}, Dec={dec:.4f})")
                return ra, dec
            self.logger.warning("Custom autofocus target missing RA/Dec, falling back to Mirach")
            preset_key = "mirach"

        preset = AUTOFOCUS_TARGET_PRESETS.get(preset_key)
        if not preset:
            self.logger.warning(f"Unknown autofocus preset '{preset_key}', falling back to Mirach")
            preset = AUTOFOCUS_TARGET_PRESETS["mirach"]

        self.logger.info(f"Autofocus target: {preset['name']} ({preset['designation']})")
        return preset["ra"], preset["dec"]

    def _execute(self) -> None:
        """Execute autofocus routine and update timestamp on both success and failure."""
        self._cancel_event.clear()
        with self._lock:
            self._running = True
            self._progress = "Starting..."
        try:
            target_ra, target_dec = self._resolve_target()
            self.logger.info("Starting autofocus routine...")
            self.hardware_adapter.do_autofocus(
                target_ra=target_ra,
                target_dec=target_dec,
                on_progress=self._set_progress,
                cancel_event=self._cancel_event,
            )

            if self.hardware_adapter.supports_filter_management():
                try:
                    filter_config = self.hardware_adapter.get_filter_config()
                    if filter_config and self.daemon.settings:
                        self.daemon.settings.adapter_settings["filters"] = filter_config
                        self.logger.info(f"Saved filter configuration with {len(filter_config)} filters")
                except Exception as e:
                    self.logger.warning(f"Failed to save filter configuration after autofocus: {e}")

            self.logger.info("Autofocus routine completed successfully")
        except Exception as e:
            self.logger.error(f"Autofocus failed: {e!s}", exc_info=True)
        finally:
            with self._lock:
                self._running = False
                self._progress = ""
            if self.daemon.settings:
                self.daemon.settings.last_autofocus_timestamp = int(time.time())
                self.daemon.settings.save()
