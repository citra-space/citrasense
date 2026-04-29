"""Unit tests for :class:`RadarDetectionFilter`.

Covers the two gates — SNR floor and tasked-satellites — plus
graceful degradation when ``task_index`` is missing (per the
``_is_tasked`` docstring).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from citrasense.pipelines.radar.radar_detection_filter import RadarDetectionFilter
from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext
from citrasense.sensors.radar.events import RadarObservationEvent


def _make_ctx(
    payload: dict,
    *,
    detection_min_snr_db: float = 0.0,
    forward_only_tasked: bool = False,
    task_index=None,
) -> RadarProcessingContext:
    event = RadarObservationEvent(
        sensor_id="radar-test",
        modality="radar",
        timestamp=datetime(2025, 11, 11, 18, 38, 11, tzinfo=timezone.utc),
        payload=payload,
    )
    return RadarProcessingContext(
        sensor_id="radar-test",
        event=event,
        detection_min_snr_db=detection_min_snr_db,
        forward_only_tasked_satellites=forward_only_tasked,
        task_index=task_index,
    )


class TestSnrGate:
    def test_observation_without_snr_is_dropped(self):
        ctx = _make_ctx({"target": {"citra_uuid": "abc"}})
        ok = RadarDetectionFilter().process(ctx)
        assert ok is False
        assert ctx.drop_reason is not None
        assert "SNR" in ctx.drop_reason

    def test_snr_below_floor_is_dropped(self):
        ctx = _make_ctx(
            {"quality": {"snr_db": 5.0}, "target": {"citra_uuid": "abc"}},
            detection_min_snr_db=10.0,
        )
        ok = RadarDetectionFilter().process(ctx)
        assert ok is False
        assert ctx.drop_reason is not None
        assert "5.0" in ctx.drop_reason
        assert "10.0" in ctx.drop_reason

    def test_snr_above_floor_passes(self):
        ctx = _make_ctx(
            {"quality": {"snr_db": 12.5}, "target": {"citra_uuid": "abc"}},
            detection_min_snr_db=10.0,
        )
        ok = RadarDetectionFilter().process(ctx)
        assert ok is True
        assert ctx.drop_reason is None

    def test_flat_snr_db_is_honored(self):
        # Fallback path when quality.snr_db is absent but a flat
        # ``snr_db`` is present on the payload.
        ctx = _make_ctx(
            {"snr_db": 15.0, "target": {"citra_uuid": "abc"}},
            detection_min_snr_db=10.0,
        )
        ok = RadarDetectionFilter().process(ctx)
        assert ok is True


class TestTaskedSatellitesGate:
    def _payload(self, citra_uuid: str | None) -> dict:
        payload: dict = {"quality": {"snr_db": 20.0}}
        if citra_uuid is not None:
            payload["target"] = {"citra_uuid": citra_uuid}
        return payload

    def test_no_task_index_allows_everything(self):
        ctx = _make_ctx(
            self._payload("sat-123"),
            forward_only_tasked=True,
            task_index=None,
        )
        assert RadarDetectionFilter().process(ctx) is True

    def test_tasked_sat_passes(self):
        task_index = SimpleNamespace(get_tasks_snapshot=lambda: [SimpleNamespace(satelliteId="sat-123")])
        ctx = _make_ctx(
            self._payload("sat-123"),
            forward_only_tasked=True,
            task_index=task_index,
        )
        assert RadarDetectionFilter().process(ctx) is True

    def test_untasked_sat_dropped(self):
        task_index = SimpleNamespace(get_tasks_snapshot=lambda: [SimpleNamespace(satelliteId="sat-999")])
        ctx = _make_ctx(
            self._payload("sat-123"),
            forward_only_tasked=True,
            task_index=task_index,
        )
        assert RadarDetectionFilter().process(ctx) is False
        assert ctx.drop_reason is not None
        assert "sat-123" in ctx.drop_reason

    def test_missing_citra_uuid_with_gate_enabled_is_dropped(self):
        ctx = _make_ctx(
            self._payload(None),
            forward_only_tasked=True,
            task_index=SimpleNamespace(get_tasks_snapshot=lambda: []),
        )
        assert RadarDetectionFilter().process(ctx) is False

    def test_gate_fails_open_on_snapshot_error(self):
        # If the dispatcher's snapshot method raises, we must NOT
        # silently drop observations — a transient heap-lock contention
        # shouldn't blackhole the pipeline.
        def _raise() -> list:
            raise RuntimeError("transient")

        task_index = SimpleNamespace(get_tasks_snapshot=_raise)
        ctx = _make_ctx(
            self._payload("sat-123"),
            forward_only_tasked=True,
            task_index=task_index,
        )
        assert RadarDetectionFilter().process(ctx) is True


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"quality": {"snr_db": 3.0}}, False),
        ({"quality": {"snr_db": 12.0}}, True),
    ],
)
def test_snr_parametrized(payload, expected):
    payload["target"] = {"citra_uuid": "abc"}
    ctx = _make_ctx(payload, detection_min_snr_db=10.0)
    assert RadarDetectionFilter().process(ctx) is expected
