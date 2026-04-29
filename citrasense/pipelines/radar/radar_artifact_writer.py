"""Persist radar observation artifacts to disk.

Every event the radar pipeline processes produces a single JSON file
under ``<analysis_dir>/radar/<sensor_id>/<timestamp>.json`` containing:

- ``observation`` — the raw ``pr_sensor`` enriched observation dict
  (what came off ``radar.sensor.{id}.observations``).
- ``drop_reason`` — if the filter or formatter dropped the
  observation, the short reason string; ``null`` when the observation
  progressed to upload.
- ``upload_payload`` — the camelCase payload the formatter built, or
  ``null`` when dropped.

This mirrors the optical pipeline's approach of keeping the raw input
alongside the processor output so the web Analysis tab and operator
debugging can reason about what was dropped and why.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from citrasense.pipelines.common.artifact_writer import dump_json

if TYPE_CHECKING:
    from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext

_UNSAFE_CHARS = re.compile(r"[^0-9A-Za-z._-]")


class RadarArtifactWriter:
    """Best-effort disk artifact for a single radar observation."""

    name = "radar_artifact_writer"

    def process(self, ctx: RadarProcessingContext) -> bool:
        artifact_dir = ctx.artifact_dir
        if artifact_dir is None:
            return True  # no-op when deployed without a configured dir

        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            if ctx.logger:
                ctx.logger.warning("Failed to mkdir %s: %s", artifact_dir, exc)
            return True  # best-effort only

        filename = _filename_for(ctx)
        data = {
            "sensor_id": ctx.sensor_id,
            "received_at": (ctx.received_at or datetime.now(timezone.utc)).isoformat(),
            "observation": ctx.event.payload,
            "drop_reason": ctx.drop_reason,
            "upload_payload": ctx.upload_payload,
        }
        dump_json(artifact_dir, filename, data, logger=ctx.logger)
        return True


def _filename_for(ctx: RadarProcessingContext) -> str:
    ts = ctx.event.timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    obs_id = ctx.event.payload.get("observation_id") or "obs"
    safe_obs = _UNSAFE_CHARS.sub("_", str(obs_id))[:40]
    return f"{ts}Z_{safe_obs}.json"


def radar_artifact_dir(base: Path, sensor_id: str) -> Path:
    """Compute the canonical artifact directory for a radar sensor.

    ``base`` should be the site-wide analysis / processing root (see
    :class:`~citrasense.settings.directory_manager.DirectoryManager`).
    """
    return base / "radar" / sensor_id
