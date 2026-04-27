"""Periodic cleanup of processing output directories and analysis previews."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger("citrasense.Retention")

_PREVIEW_RETENTION_DAYS = 30


def resolve_task_dir(processing_dir: Path, task_id: str) -> Path:
    """Resolve a task's working dir under either layout.

    Returns ``processing_dir/task_id`` (legacy flat layout) when that dir
    exists, otherwise scans one level deeper for
    ``processing_dir/<sensor_id>/task_id``.  Falls back to the flat path
    when nothing matches so callers can still ``is_dir()``-check a stable
    path.
    """
    flat = processing_dir / task_id
    if flat.is_dir():
        return flat
    if processing_dir.is_dir():
        for child in processing_dir.iterdir():
            if child.is_dir():
                nested = child / task_id
                if nested.is_dir():
                    return nested
    return flat


def cleanup_processing_output(processing_dir: Path, retention_hours: int) -> int:
    """Delete processing subdirectories older than *retention_hours*.

    Supports both the legacy flat layout (``processing/<task_id>/``) and the
    multi-sensor layout introduced alongside sensor-scoped runtimes
    (``processing/<sensor_id>/<task_id>/``).  A subdirectory is treated as
    a sensor dir if it contains nested dirs with processing artifacts — the
    cleanup then walks one level deeper before removing stale task dirs,
    and finally prunes empty sensor dirs.

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

    def _is_task_dir(d: Path) -> bool:
        # Task dirs contain pipeline artifacts.  Any file is enough to treat
        # the dir as a leaf rather than a sensor namespace.
        try:
            return any(p.is_file() for p in d.iterdir())
        except OSError:
            return False

    for child in processing_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            if _is_task_dir(child):
                # Legacy flat layout (no sensor_id) — treat as task dir.
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child)
                    removed += 1
                continue
            # Sensor-namespaced layout: iterate task subdirs.
            for task_dir in child.iterdir():
                if not task_dir.is_dir():
                    continue
                try:
                    if task_dir.stat().st_mtime < cutoff:
                        shutil.rmtree(task_dir)
                        removed += 1
                except Exception as e:
                    logger.warning("Failed to remove expired processing dir %s/%s: %s", child.name, task_dir.name, e)
            # Prune empty sensor dir so it doesn't linger forever.
            try:
                if not any(child.iterdir()):
                    child.rmdir()
            except OSError:
                pass
        except Exception as e:
            logger.warning("Failed to process retention for %s: %s", child.name, e)
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
