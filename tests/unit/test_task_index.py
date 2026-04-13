"""Unit tests for the analysis TaskIndex (SQLite)."""

from unittest.mock import MagicMock

import pytest

from citrascope.analysis.task_index import (
    TaskIndex,
    _angular_distance_deg,
    _bool_int,
    _iso_diff_seconds,
)
from citrascope.processors.processor_result import AggregatedResult, ProcessorResult

# ── Helpers ────────────────────────────────────────────────────────────


def _make_task(**overrides):
    """Return a mock Task with sensible defaults."""
    t = MagicMock()
    t.id = overrides.get("id", "task-001")
    t.satelliteId = overrides.get("satelliteId", "25544")
    t.satelliteName = overrides.get("satelliteName", "ISS")
    t.taskStart = overrides.get("taskStart", "2026-04-13T02:00:00+00:00")
    t.taskStop = overrides.get("taskStop", "2026-04-13T02:05:00+00:00")
    t.assigned_filter_name = overrides.get("assigned_filter_name", "V")
    return t


def _make_result(**overrides):
    """Build a minimal AggregatedResult."""
    return AggregatedResult(
        should_upload=overrides.get("should_upload", True),
        extracted_data=overrides.get(
            "extracted_data",
            {
                "plate_solver.plate_solved": True,
                "plate_solver.ra_center": 180.0,
                "plate_solver.dec_center": 45.0,
                "plate_solver.pixel_scale": 1.2,
                "plate_solver.field_width_deg": 0.5,
                "plate_solver.field_height_deg": 0.4,
                "source_extractor.num_sources": 120,
                "photometry.zero_point": 22.5,
                "photometry.num_calibration_stars": 15,
                "photometry.filter": "V",
                "satellite_matcher.num_satellites_detected": 2,
                "satellite_matcher.satellite_observations": [
                    {"norad_id": "25544", "apparent_magnitude": 3.5},
                    {"norad_id": "99999", "apparent_magnitude": 8.1},
                ],
            },
        ),
        all_results=overrides.get(
            "all_results",
            [
                ProcessorResult(True, {}, 1.0, "ok", 0.1, "calibration"),
                ProcessorResult(True, {}, 1.0, "ok", 2.3, "plate_solver"),
                ProcessorResult(True, {}, 1.0, "ok", 0.5, "source_extractor"),
                ProcessorResult(True, {}, 1.0, "ok", 0.8, "photometry"),
                ProcessorResult(True, {}, 1.0, "ok", 1.2, "satellite_matcher"),
                ProcessorResult(True, {}, 1.0, "ok", 0.3, "annotated_image"),
            ],
        ),
        total_time=overrides.get("total_time", 5.2),
        skip_reason=overrides.get("skip_reason", None),
    )


def _make_timing(**overrides):
    t = MagicMock()
    t.slew_started_at = overrides.get("slew_started_at", "2026-04-13T02:00:10+00:00")
    t.imaging_started_at = overrides.get("imaging_started_at", "2026-04-13T02:00:30+00:00")
    t.imaging_finished_at = overrides.get("imaging_finished_at", "2026-04-13T02:01:00+00:00")
    t.processing_queued_at = overrides.get("processing_queued_at", "2026-04-13T02:01:01+00:00")
    t.processing_started_at = overrides.get("processing_started_at", "2026-04-13T02:01:02+00:00")
    t.processing_finished_at = overrides.get("processing_finished_at", "2026-04-13T02:01:07+00:00")
    return t


def _make_pointing(**overrides):
    return {
        "convergence_threshold_deg": 0.1,
        "attempts": 2,
        "converged": True,
        "final_angular_distance_deg": 0.05,
        "final_telescope_ra_deg": 180.01,
        "final_telescope_dec_deg": 45.01,
        "iterations": [
            {"actual_slew_time_s": 3.0, "observed_slew_rate_deg_per_s": 5.0},
            {"actual_slew_time_s": 1.5, "observed_slew_rate_deg_per_s": 5.2},
        ],
    }


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.fixture
def index(tmp_path):
    db = tmp_path / "test.db"
    return TaskIndex(db)


class TestRecordAndGet:
    def test_record_and_get_roundtrip(self, index):
        task = _make_task()
        result = _make_result()
        timing = _make_timing()
        pointing = _make_pointing()

        index.record_task(task=task, result=result, pointing_report=pointing, timing_info=timing)

        row = index.get_task("task-001")
        assert row is not None
        assert row["task_id"] == "task-001"
        assert row["target_name"] == "ISS"
        assert row["plate_solved"] == 1
        assert row["converged"] == 1
        assert row["convergence_attempts"] == 2
        assert row["source_count"] == 120
        assert row["zero_point"] == 22.5
        assert row["calibration_star_count"] == 15
        assert row["target_matched"] == 1
        assert row["incidental_matches"] == 1
        assert row["total_satellites_detected"] == 2
        assert row["should_upload"] == 1
        assert row["upload_success"] is None
        assert row["total_processing_time_s"] == pytest.approx(5.2)
        assert row["plate_solve_time_s"] == pytest.approx(2.3)
        assert row["calibration_time_s"] == pytest.approx(0.1)
        assert row["filter_name"] == "V"

    def test_get_missing_returns_none(self, index):
        assert index.get_task("nonexistent") is None

    def test_record_with_none_result(self, index):
        task = _make_task(id="task-none")
        index.record_task(task=task, result=None, pointing_report=None, timing_info=None)
        row = index.get_task("task-none")
        assert row is not None
        assert row["plate_solved"] is None
        assert row["total_processing_time_s"] is None

    def test_replace_on_duplicate(self, index):
        task = _make_task()
        r1 = _make_result(extracted_data={"photometry.zero_point": 20.0})
        index.record_task(task=task, result=r1, pointing_report=None, timing_info=None)
        r2 = _make_result(extracted_data={"photometry.zero_point": 22.0})
        index.record_task(task=task, result=r2, pointing_report=None, timing_info=None)
        row = index.get_task("task-001")
        assert row["zero_point"] == 22.0


class TestUploadResult:
    def test_update_upload_result(self, index):
        task = _make_task()
        index.record_task(task=task, result=_make_result(), pointing_report=None, timing_info=None)
        assert index.get_task("task-001")["upload_success"] is None

        index.update_upload_result("task-001", True)
        assert index.get_task("task-001")["upload_success"] == 1

    def test_update_upload_failure(self, index):
        task = _make_task()
        index.record_task(task=task, result=_make_result(), pointing_report=None, timing_info=None)
        index.update_upload_result("task-001", False)
        assert index.get_task("task-001")["upload_success"] == 0


class TestQueryTasks:
    def _insert_n(self, index, n):
        for i in range(n):
            t = _make_task(id=f"task-{i:03d}", satelliteName=f"SAT-{i}")
            index.record_task(task=t, result=_make_result(), pointing_report=None, timing_info=None)

    def test_pagination(self, index):
        self._insert_n(index, 10)
        page1 = index.query_tasks(limit=3, offset=0)
        assert len(page1["tasks"]) == 3
        assert page1["total"] == 10

        page2 = index.query_tasks(limit=3, offset=3)
        assert len(page2["tasks"]) == 3
        ids_1 = {t["task_id"] for t in page1["tasks"]}
        ids_2 = {t["task_id"] for t in page2["tasks"]}
        assert ids_1.isdisjoint(ids_2)

    def test_filter_by_target_name(self, index):
        self._insert_n(index, 5)
        result = index.query_tasks(target_name="SAT-3")
        assert result["total"] == 1
        assert result["tasks"][0]["target_name"] == "SAT-3"

    def test_sort_order(self, index):
        self._insert_n(index, 3)
        asc = index.query_tasks(sort="completed_at", order="asc")
        desc = index.query_tasks(sort="completed_at", order="desc")
        assert asc["tasks"][0]["task_id"] != desc["tasks"][0]["task_id"]

    def test_invalid_sort_falls_back(self, index):
        self._insert_n(index, 2)
        result = index.query_tasks(sort="DROP TABLE; --")
        assert result["total"] == 2  # no SQL injection


class TestGetStats:
    def test_empty_stats(self, index):
        stats = index.get_stats(hours=24)
        assert stats["task_count"] == 0
        assert stats["plate_solve_rate"] is None

    def test_stats_with_data(self, index):
        for i in range(5):
            t = _make_task(id=f"t-{i}")
            index.record_task(
                task=t, result=_make_result(), pointing_report=_make_pointing(), timing_info=_make_timing()
            )
        stats = index.get_stats(hours=24)
        assert stats["task_count"] == 5
        assert stats["plate_solve_rate"] == 100.0
        assert stats["per_processor_timing"]["plate_solve_s"] == pytest.approx(2.3)
        assert stats["per_processor_timing"]["calibration_s"] == pytest.approx(0.1)
        assert stats["avg_total_processing_s"] == pytest.approx(5.2)


class TestHelpers:
    def test_iso_diff_seconds(self):
        assert _iso_diff_seconds("2026-04-13T00:00:00+00:00", "2026-04-13T00:01:00+00:00") == pytest.approx(60.0)
        assert _iso_diff_seconds(None, "2026-04-13T00:00:00+00:00") is None
        assert _iso_diff_seconds("bad", "2026-04-13T00:00:00+00:00") is None

    def test_bool_int(self):
        assert _bool_int(True) == 1
        assert _bool_int(False) == 0
        assert _bool_int(None) is None

    def test_angular_distance(self):
        d = _angular_distance_deg(0, 0, 0, 1)
        assert d == pytest.approx(1.0, abs=0.001)
        d2 = _angular_distance_deg(0, 0, 0, 0)
        assert d2 == pytest.approx(0.0)
