"""Gate radar observations before upload.

Two gates live here: an SNR floor and an optional "only tasked
satellites" filter.  Both are configurable per-sensor via
:class:`~citrasense.settings.citrasense_settings.SensorConfig.adapter_settings`.

The filter *mutates* :class:`RadarProcessingContext` in place: a
rejected observation sets ``ctx.drop_reason`` and returns ``False``.
The upload path short-circuits on a non-``None`` ``drop_reason``, but
the artifact writer still persists the raw payload so operators can
review what was dropped and why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext


class RadarDetectionFilter:
    """Apply SNR and tasked-sats gates to a radar observation."""

    name = "radar_detection_filter"

    def process(self, ctx: RadarProcessingContext) -> bool:
        """Return ``True`` iff the observation should progress to upload."""
        payload = ctx.event.payload

        snr_db = _get_snr_db(payload)
        if snr_db is None:
            ctx.drop_reason = "missing SNR in observation payload"
            _log(ctx, "Dropping observation: %s", ctx.drop_reason)
            return False

        if snr_db < ctx.detection_min_snr_db:
            ctx.drop_reason = f"SNR {snr_db:.1f} dB < threshold {ctx.detection_min_snr_db:.1f} dB"
            _log(ctx, "Dropping observation: %s", ctx.drop_reason)
            return False

        if ctx.forward_only_tasked_satellites:
            citra_uuid = (payload.get("target") or {}).get("citra_uuid")
            if not citra_uuid:
                ctx.drop_reason = "forward_only_tasked_satellites=True but target.citra_uuid missing"
                _log(ctx, "Dropping observation: %s", ctx.drop_reason)
                return False
            if not _is_tasked(ctx, citra_uuid):
                ctx.drop_reason = f"satellite {citra_uuid} not tasked at this site"
                _log(ctx, "Dropping observation: %s", ctx.drop_reason)
                return False

        return True


def _get_snr_db(payload: dict) -> float | None:
    """Fish SNR out of ``pr_sensor``'s enriched ``Observation`` schema.

    ``Observation.quality.snr_db`` is the canonical location; we fall
    back to a flat ``snr_db`` for robustness against schema drift.
    """
    quality = payload.get("quality")
    if isinstance(quality, dict):
        val = quality.get("snr_db")
        if isinstance(val, (int, float)):
            return float(val)
    flat = payload.get("snr_db")
    if isinstance(flat, (int, float)):
        return float(flat)
    return None


def _is_tasked(ctx: RadarProcessingContext, citra_uuid: str) -> bool:
    """Lookup currently-tasked satellites via the dispatcher snapshot.

    ``task_index`` is expected to expose ``get_tasks_snapshot()`` (the
    :class:`~citrasense.tasks.task_dispatcher.TaskDispatcher` method
    referenced from the radar issue).  When ``task_index`` is ``None``
    we fail open — the gate reduces to a no-op, matching v1 intent.
    """
    ti = ctx.task_index
    if ti is None:
        return True
    try:
        tasks = ti.get_tasks_snapshot()
    except Exception:
        return True
    for task in tasks:
        if getattr(task, "satelliteId", None) == citra_uuid:
            return True
    return False


def _log(ctx: RadarProcessingContext, msg: str, *args) -> None:
    if ctx.logger:
        ctx.logger.info(msg, *args)
