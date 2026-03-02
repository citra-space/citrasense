"""Disk space safety check — prevents imaging when storage is critically low."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from citrascope.safety.safety_monitor import SafetyAction, SafetyCheck

WARN_BYTES = 1_000_000_000  # 1 GB
CRITICAL_BYTES = 200_000_000  # 200 MB


class DiskSpaceCheck(SafetyCheck):
    """Monitors free disk space on the images volume."""

    def __init__(self, logger: logging.Logger, images_dir: Path) -> None:
        self._logger = logger
        self._images_dir = images_dir
        self._free_bytes: int | None = None

    @property
    def name(self) -> str:
        return "disk_space"

    def check(self) -> SafetyAction:
        try:
            usage = shutil.disk_usage(self._images_dir)
            self._free_bytes = usage.free
        except Exception:
            self._logger.warning("Could not read disk usage — treating as WARN", exc_info=True)
            self._free_bytes = None
            return SafetyAction.WARN

        if self._free_bytes < CRITICAL_BYTES:
            self._logger.error(
                "Disk space critical: %d MB free (threshold %d MB)",
                self._free_bytes // 1_000_000,
                CRITICAL_BYTES // 1_000_000,
            )
            return SafetyAction.QUEUE_STOP
        if self._free_bytes < WARN_BYTES:
            self._logger.warning("Disk space low: %d MB free", self._free_bytes // 1_000_000)
            return SafetyAction.WARN
        return SafetyAction.SAFE

    def check_proposed_action(self, action_type: str, **kwargs) -> bool:
        if action_type == "capture":
            if self._free_bytes is None:
                return False
            return self._free_bytes >= CRITICAL_BYTES
        return True

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "free_mb": self._free_bytes // 1_000_000 if self._free_bytes is not None else None,
            "warn_mb": WARN_BYTES // 1_000_000,
            "critical_mb": CRITICAL_BYTES // 1_000_000,
        }
