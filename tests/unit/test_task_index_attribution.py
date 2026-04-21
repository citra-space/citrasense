"""Unit tests for lateness-attribution helpers in ``citrasense.analysis.task_index``.

These exercise the pure-Python enrichment functions in isolation — the SQL CTE
that supplies ``prev_imaging_finished_at`` / ``prev_task_id`` /
``prev_target_name`` is exercised separately in ``test_task_index.py`` via the
real ``TaskIndex`` round-trips.  Keeping these tests pure-Python means the
attribution math is locked down without needing to spin up a SQLite DB on
every run.
"""

from __future__ import annotations

import json

from citrasense.analysis.task_index import (
    CROSS_SESSION_GAP_S,
    SLIP_ORIGIN_THRESHOLD_S,
    _enrich_with_attribution,
    _enrich_with_pointing_diag,
)


def _row(**overrides) -> dict:
    """Build a minimal row dict with the columns the helpers consult."""
    base = {
        "task_id": "task-002",
        "target_name": "ISS",
        "window_start": "2026-04-13T02:00:00+00:00",
        "window_start_delay_s": 0.0,
        "prev_imaging_finished_at": None,
        "prev_task_id": None,
        "prev_target_name": None,
        "total_slew_time_s": None,
        "pointing_report_json": None,
    }
    base.update(overrides)
    return base


# ── _enrich_with_attribution ──────────────────────────────────────────


class TestAttributionNoLateness:
    def test_imaged_on_time_marks_nothing(self):
        row = _row(window_start_delay_s=0.0)
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        assert out["self_delay_s"] == 0.0
        assert out["is_slip_origin"] is False

    def test_imaged_early_clamps_to_zero(self):
        # window_start_delay_s < 0 means imaging started before the window —
        # we treat it as on time, not as negative lateness.
        row = _row(window_start_delay_s=-3.5)
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        assert out["self_delay_s"] == 0.0
        assert out["is_slip_origin"] is False

    def test_missing_window_start_short_circuits(self):
        row = _row(window_start=None, window_start_delay_s=42.0)
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        # We still report the raw lateness even when we can't attribute it.
        assert out["self_delay_s"] == 42.0
        assert out["is_slip_origin"] is True


class TestAttributionWithPreviousTask:
    def test_no_previous_task_all_self(self):
        row = _row(window_start_delay_s=12.0, prev_imaging_finished_at=None)
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        assert out["self_delay_s"] == 12.0

    def test_previous_finished_before_window_opens(self):
        # prev_finish at 01:59:30, window opens at 02:00:00 → gap=30s, no bleed.
        row = _row(
            window_start_delay_s=20.0,
            prev_imaging_finished_at="2026-04-13T01:59:30+00:00",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        assert out["self_delay_s"] == 20.0
        # Recent prior task that finished in time — keep the linkage so the
        # operator can still navigate to it; we just don't blame it.
        assert out["prev_task_id"] is None  # caller didn't set one in this test

    def test_previous_overran_more_than_delay_all_inherited(self):
        # prev_finish at 02:00:50, window opened at 02:00:00, delay=30s.
        # Bleed=50s ≥ delay=30s ⇒ inherited=30s, self=0.
        row = _row(
            window_start_delay_s=30.0,
            prev_imaging_finished_at="2026-04-13T02:00:50+00:00",
            prev_task_id="task-001",
            prev_target_name="DIRECTV 15",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 30.0
        assert out["self_delay_s"] == 0.0
        assert out["is_slip_origin"] is False
        # Linkage preserved — operator can jump to the upstream culprit.
        assert out["prev_task_id"] == "task-001"
        assert out["prev_target_name"] == "DIRECTV 15"

    def test_previous_partially_overran_split(self):
        # prev_finish at 02:00:18, delay=45s.  Bleed=18s, self=27s.
        row = _row(
            window_start_delay_s=45.0,
            prev_imaging_finished_at="2026-04-13T02:00:18+00:00",
            prev_task_id="task-001",
            prev_target_name="DIRECTV 15",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 18.0
        assert out["self_delay_s"] == 27.0
        # 27s self-added crosses the 15s threshold ⇒ still an origin.
        assert out["is_slip_origin"] is True


class TestAttributionCrossSessionGap:
    def test_cross_session_gap_drops_linkage(self):
        # Previous task finished an hour ago (way > CROSS_SESSION_GAP_S=600s).
        # We should NOT attribute lateness across observing sessions, AND
        # we should drop the prev-task linkage so the UI doesn't show a
        # stale "inherited from <hours-old task>" hint.
        row = _row(
            window_start_delay_s=10.0,
            prev_imaging_finished_at="2026-04-13T01:00:00+00:00",
            prev_task_id="task-from-yesterday",
            prev_target_name="OLD TARGET",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        assert out["self_delay_s"] == 10.0
        assert out["prev_task_id"] is None
        assert out["prev_target_name"] is None

    def test_just_inside_cross_session_threshold_keeps_linkage(self):
        # Gap exactly CROSS_SESSION_GAP_S - 1s ⇒ still "same session", no
        # bleed (positive gap) but linkage preserved for navigability.
        prev_finish_offset = -(CROSS_SESSION_GAP_S - 1)
        from datetime import datetime, timedelta, timezone

        ws = datetime(2026, 4, 13, 2, 0, 0, tzinfo=timezone.utc)
        prev_finish = (ws + timedelta(seconds=prev_finish_offset)).isoformat()

        row = _row(
            window_start_delay_s=5.0,
            prev_imaging_finished_at=prev_finish,
            prev_task_id="task-001",
            prev_target_name="PRIOR",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 0.0
        assert out["self_delay_s"] == 5.0
        # Within the same session — keep the linkage.
        assert out["prev_task_id"] == "task-001"


class TestSlipOriginThreshold:
    def test_just_below_threshold_not_origin(self):
        # self_delay_s = SLIP_ORIGIN_THRESHOLD_S - 0.1 ⇒ NOT an origin.
        row = _row(window_start_delay_s=SLIP_ORIGIN_THRESHOLD_S - 0.1)
        out = _enrich_with_attribution(row)

        assert out["self_delay_s"] < SLIP_ORIGIN_THRESHOLD_S
        assert out["is_slip_origin"] is False

    def test_at_threshold_is_origin(self):
        # Exactly the threshold counts (>=, not >).
        row = _row(window_start_delay_s=SLIP_ORIGIN_THRESHOLD_S)
        out = _enrich_with_attribution(row)

        assert out["self_delay_s"] == SLIP_ORIGIN_THRESHOLD_S
        assert out["is_slip_origin"] is True

    def test_inherited_dominant_not_origin(self):
        # 30s late but 28s of it is inherited ⇒ self=2s, NOT an origin.
        row = _row(
            window_start_delay_s=30.0,
            prev_imaging_finished_at="2026-04-13T02:00:28+00:00",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 28.0
        assert out["self_delay_s"] == 2.0
        assert out["is_slip_origin"] is False

    def test_self_dominant_is_origin(self):
        # 30s late but only 2s inherited ⇒ self=28s, IS an origin.
        row = _row(
            window_start_delay_s=30.0,
            prev_imaging_finished_at="2026-04-13T02:00:02+00:00",
        )
        out = _enrich_with_attribution(row)

        assert out["inherited_delay_s"] == 2.0
        assert out["self_delay_s"] == 28.0
        assert out["is_slip_origin"] is True


# ── _enrich_with_pointing_diag ────────────────────────────────────────


class TestPointingDiag:
    def test_missing_json_yields_none(self):
        row = _row(pointing_report_json=None, total_slew_time_s=10.0)
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] is None
        assert out["slew_overrun_s"] is None

    def test_empty_string_json_yields_none(self):
        row = _row(pointing_report_json="", total_slew_time_s=10.0)
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] is None
        assert out["slew_overrun_s"] is None

    def test_malformed_json_yields_none(self):
        row = _row(pointing_report_json="{not valid json", total_slew_time_s=10.0)
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] is None
        assert out["slew_overrun_s"] is None

    def test_well_formed_report_sums_estimates(self):
        report = {
            "iterations": [
                {"estimated_slew_time_s": 20.0, "actual_slew_time_s": 25.0},
                {"estimated_slew_time_s": 5.0, "actual_slew_time_s": 8.0},
            ]
        }
        row = _row(
            pointing_report_json=json.dumps(report),
            total_slew_time_s=33.0,
        )
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] == 25.0
        # 33s actual vs 25s estimated ⇒ +8s overrun.
        assert out["slew_overrun_s"] == 8.0

    def test_missing_per_iteration_estimate_skipped(self):
        # Iterations without estimated_slew_time_s contribute 0 to the sum,
        # but at least one must have an estimate or we report None.
        report = {
            "iterations": [
                {"actual_slew_time_s": 10.0},  # no estimate ⇒ skip
                {"estimated_slew_time_s": 7.0, "actual_slew_time_s": 8.0},
            ]
        }
        row = _row(
            pointing_report_json=json.dumps(report),
            total_slew_time_s=18.0,
        )
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] == 7.0
        assert out["slew_overrun_s"] == 11.0

    def test_no_estimates_at_all_yields_none(self):
        # Report exists but no iteration has an estimate ⇒ we have no signal,
        # so report None rather than implying a 0-second estimate.
        report = {
            "iterations": [
                {"actual_slew_time_s": 10.0},
                {"actual_slew_time_s": 5.0},
            ]
        }
        row = _row(
            pointing_report_json=json.dumps(report),
            total_slew_time_s=15.0,
        )
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] is None
        assert out["slew_overrun_s"] is None

    def test_garbage_estimate_value_skipped(self):
        # A non-numeric estimated_slew_time_s shouldn't blow up the sum.
        report = {
            "iterations": [
                {"estimated_slew_time_s": "not a number"},
                {"estimated_slew_time_s": 4.0},
            ]
        }
        row = _row(
            pointing_report_json=json.dumps(report),
            total_slew_time_s=10.0,
        )
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] == 4.0
        assert out["slew_overrun_s"] == 6.0

    def test_no_iterations_key_yields_none(self):
        row = _row(
            pointing_report_json=json.dumps({"attempts": 0}),
            total_slew_time_s=None,
        )
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] is None
        assert out["slew_overrun_s"] is None

    def test_actual_slew_time_missing_overrun_none(self):
        # We have an estimate but no actual ⇒ can't compute overrun.
        report = {"iterations": [{"estimated_slew_time_s": 12.0}]}
        row = _row(pointing_report_json=json.dumps(report), total_slew_time_s=None)
        out = _enrich_with_pointing_diag(row)

        assert out["slew_estimate_total_s"] == 12.0
        assert out["slew_overrun_s"] is None
