"""Mount homing management: operator-requested find-home with imaging queue gate."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrascope.hardware.abstract_astro_hardware_adapter import AbstractAstroHardwareAdapter
    from citrascope.tasks.base_work_queue import BaseWorkQueue

_HOME_POLL_INTERVAL_S = 1.0
_HOME_TIMEOUT_S = 120.0
_GRACE_POLLS = 5  # polls before we start checking for premature stops
_IDLE_THRESHOLD = 3  # consecutive idle polls to declare homing interrupted


class HomingManager:
    """Manages operator-requested mount homing with safe task gating.

    Follows the same request/check_and_execute pattern as AutofocusManager
    and AlignmentManager.  Homing will only begin once the imaging queue
    has drained, preventing hardware collisions.

    The mount firmware is the authority on calibration state — we do not
    track "has been homed" in software.  The task runner should check
    ``is_running()`` / ``is_requested()`` to defer tasks while the mount
    is physically moving during a homing routine.
    """

    def __init__(
        self,
        logger: logging.Logger,
        hardware_adapter: AbstractAstroHardwareAdapter,
        imaging_queue: BaseWorkQueue | None = None,
    ):
        self.logger = logger
        self.hardware_adapter = hardware_adapter
        self.imaging_queue = imaging_queue
        self._requested = False
        self._running = False
        self._progress = ""
        self._lock = threading.Lock()

    def request(self) -> bool:
        with self._lock:
            if self._running:
                self.logger.info("Homing already in progress")
                return False
            self._requested = True
            self.logger.info("Mount homing requested — will run when imaging queue is idle")
            return True

    def cancel(self) -> bool:
        with self._lock:
            was_requested = self._requested
            self._requested = False
            if was_requested:
                self.logger.info("Mount homing request cancelled")
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

    def check_and_execute(self) -> bool:
        """Check if homing should run and execute if so.

        Call this between tasks in the runner loop.  Returns True if homing ran.
        Waits for the imaging queue to drain before starting.
        """
        with self._lock:
            should_run = self._requested
            if should_run:
                self._requested = False

        if not should_run:
            return False

        if self.imaging_queue and not self.imaging_queue.is_idle():
            self.logger.info("Homing deferred — waiting for imaging queue to drain")
            with self._lock:
                self._requested = True
            return False

        self._execute()
        return True

    def _execute(self) -> None:
        """Send find-home and poll until the mount reports at-home.

        After a grace period for motion to start, we also watch for the
        mount stopping without reaching home (e.g. safety abort during a
        cable-wrap emergency).  This prevents the UI from showing a
        homing spinner for the full timeout when the slew was interrupted.
        """
        with self._lock:
            self._running = True
            self._progress = "Homing..."
        try:
            self.logger.info("Mount homing: initiating find-home...")
            success = self.hardware_adapter.home_mount()
            if not success:
                self.logger.error("Mount does not support homing or homing failed to initiate")
                return

            deadline = time.monotonic() + _HOME_TIMEOUT_S
            poll_count = 0
            idle_count = 0
            while time.monotonic() < deadline:
                try:
                    at_home = self.hardware_adapter.is_telescope_connected() and self.hardware_adapter.is_mount_homed()
                except Exception:
                    at_home = False
                if at_home:
                    self.logger.info("Mount homing complete — encoder position established")
                    return

                poll_count += 1
                if poll_count > _GRACE_POLLS:
                    try:
                        still_moving = self.hardware_adapter.telescope_is_moving()
                    except Exception:
                        still_moving = True
                    if not still_moving:
                        idle_count += 1
                        if idle_count >= _IDLE_THRESHOLD:
                            self.logger.warning(
                                "Mount stopped without reaching home (poll %d) — homing interrupted",
                                poll_count,
                            )
                            return
                    else:
                        idle_count = 0

                time.sleep(_HOME_POLL_INTERVAL_S)

            self.logger.error("Mount homing timed out after %.0fs", _HOME_TIMEOUT_S)
        except NotImplementedError:
            self.logger.error("Mount adapter does not support homing")
        except Exception as e:
            self.logger.error("Mount homing failed: %s", e, exc_info=True)
        finally:
            with self._lock:
                self._running = False
                self._progress = ""
