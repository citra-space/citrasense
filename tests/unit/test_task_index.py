"""Unit tests for the analysis TaskIndex (SQLite)."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from citrasense.analysis.task_index import (
    _MIGRATIONS,
    _SCHEMA,
    _SCHEMA_VERSION,
    TaskIndex,
    _angular_distance_deg,
    _bool_int,
    _iso_diff_seconds,
)
from citrasense.processors.processor_result import AggregatedResult, ProcessorResult

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

    def test_filter_by_filter_name(self, index):
        t1 = _make_task(id="t-r", assigned_filter_name="r")
        t2 = _make_task(id="t-v", assigned_filter_name="V")
        r1 = _make_result(extracted_data={"photometry.filter": "r"})
        r2 = _make_result(extracted_data={"photometry.filter": "V"})
        index.record_task(task=t1, result=r1, pointing_report=None, timing_info=None)
        index.record_task(task=t2, result=r2, pointing_report=None, timing_info=None)
        result = index.query_tasks(filter_name="r")
        assert result["total"] == 1
        assert result["tasks"][0]["task_id"] == "t-r"

    def test_match_detail_matched(self, index):
        t = _make_task(id="t-matched", satelliteId="25544")
        obs = [{"norad_id": "25544", "apparent_magnitude": 3.5}]
        r = _make_result(
            extracted_data={
                "satellite_matcher.num_satellites_detected": 1,
                "satellite_matcher.satellite_observations": obs,
            }
        )
        index.record_task(task=t, result=r, pointing_report=None, timing_info=None)
        result = index.query_tasks(match_detail="matched")
        assert result["total"] == 1

    def test_match_detail_not_matched(self, index):
        t = _make_task(id="t-miss", satelliteId="25544")
        r = _make_result(
            extracted_data={
                "satellite_matcher.num_satellites_detected": 0,
                "satellite_matcher.satellite_observations": [],
            }
        )
        index.record_task(task=t, result=r, pointing_report=None, timing_info=None)
        result = index.query_tasks(match_detail="not_matched")
        assert result["total"] == 1

    def test_match_detail_has_extras(self, index):
        t = _make_task(id="t-extra", satelliteId="25544")
        obs = [
            {"norad_id": "25544", "apparent_magnitude": 3.5},
            {"norad_id": "99999", "apparent_magnitude": 8.0},
        ]
        r = _make_result(
            extracted_data={
                "satellite_matcher.num_satellites_detected": 2,
                "satellite_matcher.satellite_observations": obs,
            }
        )
        index.record_task(task=t, result=r, pointing_report=None, timing_info=None)
        result = index.query_tasks(match_detail="has_extras")
        assert result["total"] == 1

    def test_match_detail_no_detections(self, index):
        t = _make_task(id="t-empty", satelliteId="25544")
        r = _make_result(
            extracted_data={
                "satellite_matcher.num_satellites_detected": 0,
                "satellite_matcher.satellite_observations": [],
            }
        )
        index.record_task(task=t, result=r, pointing_report=None, timing_info=None)
        result = index.query_tasks(match_detail="no_detections")
        assert result["total"] == 1

    def test_upload_status_uploaded(self, index):
        t = _make_task(id="t-up")
        index.record_task(task=t, result=_make_result(), pointing_report=None, timing_info=None)
        index.update_upload_result("t-up", True)
        assert index.query_tasks(upload_status="uploaded")["total"] == 1
        assert index.query_tasks(upload_status="failed")["total"] == 0

    def test_upload_status_failed(self, index):
        t = _make_task(id="t-fail")
        index.record_task(task=t, result=_make_result(), pointing_report=None, timing_info=None)
        index.update_upload_result("t-fail", False)
        assert index.query_tasks(upload_status="failed")["total"] == 1
        assert index.query_tasks(upload_status="uploaded")["total"] == 0

    def test_upload_status_skipped(self, index):
        t = _make_task(id="t-skip")
        r = _make_result(should_upload=False, skip_reason="No satellite match")
        index.record_task(task=t, result=r, pointing_report=None, timing_info=None)
        assert index.query_tasks(upload_status="skipped")["total"] == 1


class TestDistinctFilterNames:
    def test_returns_sorted_non_null(self, index):
        for name, filt in [("t-v", "V"), ("t-r", "r"), ("t-b", "B"), ("t-none", None)]:
            t = _make_task(id=name, assigned_filter_name=filt)
            ed = {"photometry.filter": filt} if filt else {}
            r = _make_result(extracted_data=ed)
            index.record_task(task=t, result=r, pointing_report=None, timing_info=None)
        names = index.get_distinct_filter_names()
        assert names == ["B", "V", "r"]

    def test_empty_db_returns_empty_list(self, index):
        assert index.get_distinct_filter_names() == []


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


class TestSchemaMigration:
    """Tests for the _migrate() schema migration system."""

    @staticmethod
    def _create_v0_db(db_path):
        """Create a DB with only the v0 baseline schema and user_version=0."""
        conn = sqlite3.connect(str(db_path))
        conn.executescript(_SCHEMA)
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
        conn.close()

    def test_v0_migrates_to_current(self, tmp_path):
        """A v0 DB gets all migrations applied up to _SCHEMA_VERSION."""
        db = tmp_path / "v0.db"
        self._create_v0_db(db)

        idx = TaskIndex(db)
        version = idx._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == _SCHEMA_VERSION

        task = _make_task()
        pointing = _make_pointing()
        pointing["slew_ahead"] = {
            "exposure_seconds": 2.5,
            "num_exposures": 3,
            "adaptive_exposure_active": True,
        }
        idx.record_task(task=task, result=_make_result(), pointing_report=pointing, timing_info=_make_timing())
        row = idx.get_task("task-001")
        assert row["exposure_seconds"] == pytest.approx(2.5)
        assert row["num_exposures"] == 3
        assert row["adaptive_exposure_active"] == 1
        idx.close()

    def test_already_current_is_noop(self, tmp_path):
        """Opening a DB that is already at _SCHEMA_VERSION runs no migrations."""
        db = tmp_path / "current.db"
        idx1 = TaskIndex(db)
        idx1.record_task(task=_make_task(), result=_make_result(), pointing_report=None, timing_info=None)
        idx1.close()

        idx2 = TaskIndex(db)
        version = idx2._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == _SCHEMA_VERSION
        assert idx2.get_task("task-001") is not None
        idx2.close()

    def test_partial_migration(self, tmp_path):
        """A DB at v1 only runs migrations from v1 onward."""
        db = tmp_path / "partial.db"
        self._create_v0_db(db)

        conn = sqlite3.connect(str(db))
        _desc, stmts = _MIGRATIONS[0]
        for sql in stmts:
            conn.execute(sql)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        idx = TaskIndex(db)
        version = idx._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == _SCHEMA_VERSION

        task = _make_task()
        pointing = _make_pointing()
        pointing["slew_ahead"] = {"adaptive_exposure_active": False}
        idx.record_task(task=task, result=_make_result(), pointing_report=pointing, timing_info=_make_timing())
        row = idx.get_task("task-001")
        assert row["adaptive_exposure_active"] == 0
        idx.close()
