"""Periodic cleanup of processing output directories and analysis previews."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger("citrascope")

_PREVIEW_RETENTION_DAYS = 30


def cleanup_processing_output(processing_dir: Path, retention_hours: int) -> int:
    """Delete processing subdirectories older than *retention_hours*.

    Returns the number of directories removed.  Skips if *retention_hours*
    is ``0`` (immediate cleanup handled by ProcessingQueue) or ``-1`` (keep
    forever).
    """
    if retention_hours <= 0:
        return 0
    if not processing_dir.is_dir():
        return 0

    cutoff = time.time() - (retention_hours * 3600)
    removed = 0
    for child in processing_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
                removed += 1
        except Exception as e:
            logger.warning("Failed to remove expired processing dir %s: %s", child.name, e)
    if removed:
        logger.info(
            "Retention cleanup: removed %d expired processing director%s",
            removed,
            "y" if removed == 1 else "ies",
        )
    return removed


def cleanup_previews(previews_dir: Path, retention_days: int = _PREVIEW_RETENTION_DAYS) -> int:
    """Delete preview images older than *retention_days*.

    Returns the number of files removed.
    """
    if retention_days <= 0 or not previews_dir.is_dir():
        return 0

    cutoff = time.time() - (retention_days * 86400)
    removed = 0
    for child in previews_dir.iterdir():
        if not child.is_file():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                child.unlink()
                removed += 1
        except Exception as e:
            logger.warning("Failed to remove expired preview %s: %s", child.name, e)
    if removed:
        logger.info("Preview cleanup: removed %d expired file%s", removed, "s" if removed != 1 else "")
    return removed
