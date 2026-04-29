"""Unit tests for :class:`RadarPipeline`.

Focuses on the per-processor stats instrumentation that feeds the
``status.pipeline_stats.processors`` aggregation the web UI renders.
The filter/formatter/writer steps have their own dedicated test
modules; here we only care that the pipeline correctly counts runs,
failures, and records the last drop reason.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from citrasense.pipelines.radar.radar_pipeline import RadarPipeline
from citrasense.pipelines.radar.radar_processing_context import RadarProcessingContext
from citrasense.sensors.radar.events import RadarObservationEvent


def _make_ctx(payload: dict, tmp_path: Path) -> RadarProcessingContext:
    event = RadarObservationEvent(
        sensor_id="radar-test",
        modality="radar",
        timestamp=datetime(2025, 11, 11, 18, 38, 11, tzinfo=timezone.utc),
        payload=payload,
    )
    return RadarProcessingContext(
        sensor_id="radar-test",
        event=event,
        antenna_id="antenna-uuid",
        detection_min_snr_db=0.0,
        forward_only_tasked_satellites=False,
        artifact_dir=tmp_path,
    )


@pytest.fixture
def pipeline() -> RadarPipeline:
    return RadarPipeline()


def test_initial_stats_are_zero(pipeline):
    stats = pipeline.get_processor_stats()
    assert set(stats.keys()) == {
        "radar_detection_filter",
        "radar_detection_formatter",
        "radar_artifact_writer",
    }
    for s in stats.values():
        assert s == {"runs": 0, "failures": 0, "last_failure_reason": None}


def test_stats_shape_matches_pipeline_registry(pipeline):
    # The status collector aggregates radar + telescope processors into
    # one dict; both sources must expose the same per-entry shape.
    stats = pipeline.get_processor_stats()
    for entry in stats.values():
        assert set(entry.keys()) == {"runs", "failures", "last_failure_reason"}


def test_dropped_observation_increments_filter_failure(pipeline, tmp_path):
    # Observation has no SNR → filter drops it → formatter never runs.
    ctx = _make_ctx({"target": {"citra_uuid": "abc"}}, tmp_path)
    ok = pipeline.process(ctx)
    assert ok is False
    stats = pipeline.get_processor_stats()
    assert stats["radar_detection_filter"]["runs"] == 1
    assert stats["radar_detection_filter"]["failures"] == 1
    assert "SNR" in (stats["radar_detection_filter"]["last_failure_reason"] or "")
    # Formatter should not have run since the filter rejected upstream.
    assert stats["radar_detection_formatter"]["runs"] == 0
    # Artifact writer always runs — even on dropped observations — so
    # operators can audit filter decisions.
    assert stats["radar_artifact_writer"]["runs"] == 1
    assert stats["radar_artifact_writer"]["failures"] == 0


def test_filter_pass_formatter_runs(pipeline, tmp_path):
    # Observation with SNR + target UUID + receiver lat/lon passes
    # filter → formatter runs.  SNR lives on ``quality.snr_db`` per the
    # canonical ``pr_sensor`` ``Observation`` schema.
    ctx = _make_ctx(
        {
            "quality": {"snr_db": 12.0},
            "detection": {"range_km": 1200.0, "range_rate_km_s": 3.0},
            "target": {"citra_uuid": "sat-uuid-1", "name": "ISS"},
            "receiver": {"lat_deg": 38.9, "lon_deg": -104.8, "alt_m": 1940},
            "timestamp": "2025-11-11T18:38:11Z",
        },
        tmp_path,
    )
    pipeline.process(ctx)
    stats = pipeline.get_processor_stats()
    assert stats["radar_detection_filter"]["runs"] == 1
    assert stats["radar_detection_filter"]["failures"] == 0
    assert stats["radar_detection_formatter"]["runs"] == 1
    assert stats["radar_artifact_writer"]["runs"] == 1


def test_repeated_drops_accumulate_counts(pipeline, tmp_path):
    # Five consecutive drops → filter should show 5 runs / 5 failures.
    for _ in range(5):
        ctx = _make_ctx({"target": {"citra_uuid": "abc"}}, tmp_path)
        pipeline.process(ctx)
    stats = pipeline.get_processor_stats()
    assert stats["radar_detection_filter"]["runs"] == 5
    assert stats["radar_detection_filter"]["failures"] == 5
    assert stats["radar_artifact_writer"]["runs"] == 5


def test_stats_snapshot_is_a_copy(pipeline, tmp_path):
    # get_processor_stats must return a deep copy so mutation by the
    # caller can't corrupt the running counters.
    ctx = _make_ctx({"target": {"citra_uuid": "abc"}}, tmp_path)
    pipeline.process(ctx)
    snap = pipeline.get_processor_stats()
    snap["radar_detection_filter"]["runs"] = 9999
    again = pipeline.get_processor_stats()
    assert again["radar_detection_filter"]["runs"] == 1
