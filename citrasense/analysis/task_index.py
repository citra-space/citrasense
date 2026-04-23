"""SQLite-backed index of completed task results for the Analysis dashboard.

Single writer (processing thread for INSERT, upload thread for UPDATE),
multiple readers (web handlers).  Uses WAL mode for concurrent read access.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from citrasense.pipelines.common.processor_result import AggregatedResult
from citrasense.tasks.views.telescope_task_view import TelescopeTaskView

logger = logging.getLogger("citrasense.TaskIndex")

# Columns that are safe to sort/filter on (prevents SQL injection via user input).
_SORTABLE_COLUMNS = frozenset(
    {
        "completed_at",
        "target_name",
        "filter_name",
        "pointing_error_deg",
        "zero_point",
        "total_processing_time_s",
        "convergence_attempts",
        "total_slew_time_s",
        "source_count",
    }
)

# ── Lateness attribution (Tier 1) ──────────────────────────────────────
#
# A task is flagged as a "slip origin" when ``self_delay_s`` — the lateness
# it added itself, after subtracting any portion inherited from the prior
# task running long into its window — crosses this threshold.  Tuned for
# operational relevance: tens of seconds of self-added delay matters, sub-15s
# jitter is noise.  Adjust here, not at call sites.
SLIP_ORIGIN_THRESHOLD_S = 15.0

# When the gap between the previous task's imaging-finish and this task's
# window-start exceeds this, we treat the previous task as belonging to a
# different observing session and do NOT attribute lateness across the gap
# (we also drop the prev-task linkage so the UI doesn't surface a stale
# "inherited from <hours-old task>" hint).
CROSS_SESSION_GAP_S = 600.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS completed_tasks (
    -- Identity & metadata
    task_id                     TEXT PRIMARY KEY,
    target_name                 TEXT,
    completed_at                TEXT NOT NULL,
    filter_name                 TEXT,
    calibration_applied         INTEGER,

    -- Pointing / convergence
    requested_ra                REAL,
    requested_dec               REAL,
    solved_ra                   REAL,
    solved_dec                  REAL,
    pointing_error_deg          REAL,
    convergence_attempts        INTEGER,
    converged                   INTEGER,
    convergence_threshold_deg   REAL,
    total_slew_time_s           REAL,
    final_target_distance_deg   REAL,
    observed_slew_rate_deg_per_s REAL,
    pointing_report_json        TEXT,

    -- Task-window timing
    window_start                TEXT,
    window_stop                 TEXT,
    slew_started_at             TEXT,
    imaging_started_at          TEXT,
    imaging_finished_at         TEXT,
    window_start_delay_s        REAL,
    window_time_remaining_s     REAL,
    missed_window               INTEGER,

    -- Plate solve
    plate_solved                INTEGER,
    source_count                INTEGER,
    pixel_scale                 REAL,
    field_width_deg             REAL,
    field_height_deg            REAL,

    -- Photometry
    zero_point                  REAL,
    calibration_star_count      INTEGER,

    -- Satellite matching
    target_satellite_id         TEXT,
    target_satellite_name       TEXT,
    target_matched              INTEGER,
    target_satellite_mag        REAL,
    incidental_matches          INTEGER,
    total_satellites_detected   INTEGER,
    satellite_observations_json TEXT,

    -- Pipeline decision & upload
    should_upload               INTEGER,
    skip_reason                 TEXT,
    upload_type                 TEXT,
    upload_success              INTEGER,

    -- Processing timing
    processing_queued_at        TEXT,
    processing_started_at       TEXT,
    processing_finished_at      TEXT,
    processing_queue_wait_s     REAL,
    total_processing_time_s     REAL,
    calibration_time_s          REAL,
    plate_solve_time_s          REAL,
    source_extractor_time_s     REAL,
    photometry_time_s           REAL,
    matcher_time_s              REAL,
    annotated_image_time_s      REAL,

    -- Artifacts
    has_annotated_image         INTEGER,
    annotated_image_path        TEXT,

    -- Overflow
    extracted_data_json         TEXT
);
"""


# Ordered migration scripts.  Index 0 → v1, index 1 → v2, etc.
# Each entry is a (description, [sql_statements]) tuple.
# To add a migration: append a new tuple and you're done —
# _SCHEMA_VERSION derives from len() automatically.
_MIGRATIONS: list[tuple[str, list[str]]] = [
    # v1: imaging parameter columns for adaptive exposure
    (
        "add imaging parameter columns (exposure_seconds, num_exposures)",
        [
            "ALTER TABLE completed_tasks ADD COLUMN exposure_seconds REAL",
            "ALTER TABLE completed_tasks ADD COLUMN num_exposures INTEGER",
        ],
    ),
    # v2: track whether adaptive exposure was active
    (
        "add adaptive_exposure_active flag",
        [
            "ALTER TABLE completed_tasks ADD COLUMN adaptive_exposure_active INTEGER",
        ],
    ),
    # v3: index supporting LAG()-over-imaging_started_at for lateness attribution.
    # Without this, CTE/window-function ordering on a column with no index turns
    # into a full sort on every query_tasks() / get_task() call.
    (
        "add idx_imaging_started_at for lateness-attribution LAG() ordering",
        [
            "CREATE INDEX IF NOT EXISTS idx_imaging_started_at " "ON completed_tasks(imaging_started_at)",
        ],
    ),
]

_SCHEMA_VERSION = len(_MIGRATIONS)


class TaskIndex:
    """Persistent index of completed-task pipeline metrics in a local SQLite DB."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Run incremental schema migrations tracked by ``PRAGMA user_version``.

        ``_SCHEMA`` is the v0 baseline (the original CREATE TABLE).
        All subsequent changes live in ``_MIGRATIONS`` — the single source
        of truth for schema evolution.  On startup, any pending migrations
        are applied in order and ``user_version`` is stamped.
        """
        version: int = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= _SCHEMA_VERSION:
            logger.debug("Analysis DB schema is current (v%d)", version)
            return

        logger.info("Analysis DB schema v%d → v%d: running migrations", version, _SCHEMA_VERSION)

        for i in range(version, _SCHEMA_VERSION):
            desc, statements = _MIGRATIONS[i]
            for sql in statements:
                self._conn.execute(sql)
            logger.info("  v%d: %s", i + 1, desc)

        self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    # ── Writes ─────────────────────────────────────────────────────────

    def record_task(
        self,
        *,
        task: Any,
        result: AggregatedResult | None,
        pointing_report: dict | None,
        timing_info: Any | None,
    ) -> None:
        """INSERT a completed-task row from the processing thread."""
        now = datetime.now(timezone.utc).isoformat()
        ed = result.extracted_data if result else {}
        tv = TelescopeTaskView(task) if task and getattr(task, "sensor_type", None) == "telescope" else None

        # Per-processor timing from individual results
        proc_times: dict[str, float] = {}
        if result:
            for pr in result.all_results:
                proc_times[pr.processor_name] = pr.processing_time_seconds

        # Pointing / convergence
        pr_dict = pointing_report or {}
        iterations = pr_dict.get("iterations", [])
        slew_ahead = pr_dict.get("slew_ahead", {})
        total_slew = sum(it.get("actual_slew_time_s", 0) for it in iterations)

        # Solved position from plate solver
        solved_ra = _float(ed.get("plate_solver.ra_center"))
        solved_dec = _float(ed.get("plate_solver.dec_center"))

        # Intended target position — what the system was actually aiming at,
        # before pointing model correction.  Fall back through: last iteration's
        # computed lead target → pointing model's target → task RA/Dec.
        pmc = pr_dict.get("pointing_model_correction") or {}
        last_iter = iterations[-1] if iterations else {}
        requested_ra = (
            _float(last_iter.get("target_lead_ra_deg"))
            or _float(pmc.get("target_ra_deg"))
            or _float(getattr(task, "ra", None) if task else None)
        )
        requested_dec = (
            _float(last_iter.get("target_lead_dec_deg"))
            or _float(pmc.get("target_dec_deg"))
            or _float(getattr(task, "dec", None) if task else None)
        )

        pointing_error = None
        if solved_ra is not None and requested_ra is not None and solved_dec is not None and requested_dec is not None:
            pointing_error = _angular_distance_deg(requested_ra, requested_dec, solved_ra, solved_dec)

        # Satellite matching
        observations = ed.get("satellite_matcher.satellite_observations") or []
        target_sat_id = tv.satellite_id if tv else None
        target_matched = False
        target_mag: float | None = None
        incidental = 0
        for obs in observations:
            norad = str(obs.get("norad_id", ""))
            if norad and target_sat_id and norad == str(target_sat_id):
                target_matched = True
                target_mag = _float(obs.get("apparent_magnitude"))
            else:
                incidental += 1

        # Calibration
        cal_applied_list = ed.get("calibration.calibration_applied")
        calibration_applied = bool(cal_applied_list) if cal_applied_list else False

        # Task window timing
        ti = timing_info
        window_start = task.taskStart if task else None
        window_stop = task.taskStop if task else None
        slew_started_at = getattr(ti, "slew_started_at", None) if ti else None
        imaging_started_at = getattr(ti, "imaging_started_at", None) if ti else None
        imaging_finished_at = getattr(ti, "imaging_finished_at", None) if ti else None
        processing_queued_at = getattr(ti, "processing_queued_at", None) if ti else None
        processing_started_at = getattr(ti, "processing_started_at", None) if ti else None
        processing_finished_at = getattr(ti, "processing_finished_at", None) if ti else None

        window_start_delay_s = _iso_diff_seconds(window_start, imaging_started_at)
        window_time_remaining_s = _iso_diff_seconds(imaging_finished_at, window_stop)
        missed_window = (
            bool(window_time_remaining_s is not None and window_time_remaining_s < 0) if window_stop else None
        )
        queue_wait_s = _iso_diff_seconds(processing_queued_at, processing_started_at)

        # Upload type heuristic
        upload_type: str | None = None
        if result and result.should_upload:
            upload_type = "observations" if observations else "fits_image"
        elif result and not result.should_upload:
            upload_type = "skipped"

        # Annotated image
        annotated_path = ed.get("annotated_image.image_path")
        has_annotated = bool(annotated_path and Path(str(annotated_path)).exists())

        row = {
            "task_id": task.id if task else "unknown",
            "target_name": tv.satellite_name if tv else None,
            "completed_at": now,
            "filter_name": _str(ed.get("photometry.filter")) or (tv.assigned_filter_name if tv else None),
            "calibration_applied": _bool_int(calibration_applied),
            "requested_ra": requested_ra,
            "requested_dec": requested_dec,
            "solved_ra": solved_ra,
            "solved_dec": solved_dec,
            "pointing_error_deg": pointing_error,
            "convergence_attempts": pr_dict.get("attempts"),
            "converged": _bool_int(pr_dict.get("converged")),
            "convergence_threshold_deg": _float(pr_dict.get("convergence_threshold_deg")),
            "total_slew_time_s": total_slew if total_slew else None,
            "final_target_distance_deg": _float(pr_dict.get("final_angular_distance_deg")),
            "observed_slew_rate_deg_per_s": (
                _float(iterations[-1].get("observed_slew_rate_deg_per_s")) if iterations else None
            ),
            "pointing_report_json": json.dumps(pr_dict) if pr_dict else None,
            "exposure_seconds": _float(slew_ahead.get("exposure_seconds")),
            "num_exposures": _int(slew_ahead.get("num_exposures")),
            "adaptive_exposure_active": _bool_int(slew_ahead.get("adaptive_exposure_active")),
            "window_start": window_start,
            "window_stop": window_stop,
            "slew_started_at": slew_started_at,
            "imaging_started_at": imaging_started_at,
            "imaging_finished_at": imaging_finished_at,
            "window_start_delay_s": window_start_delay_s,
            "window_time_remaining_s": window_time_remaining_s,
            "missed_window": _bool_int(missed_window),
            "plate_solved": _bool_int(ed.get("plate_solver.plate_solved")),
            "source_count": _int(ed.get("source_extractor.num_sources")),
            "pixel_scale": _float(ed.get("plate_solver.pixel_scale")),
            "field_width_deg": _float(ed.get("plate_solver.field_width_deg")),
            "field_height_deg": _float(ed.get("plate_solver.field_height_deg")),
            "zero_point": _float(ed.get("photometry.zero_point")),
            "calibration_star_count": _int(ed.get("photometry.num_calibration_stars")),
            "target_satellite_id": target_sat_id,
            "target_satellite_name": tv.satellite_name if tv else None,
            "target_matched": _bool_int(target_matched),
            "target_satellite_mag": target_mag,
            "incidental_matches": incidental,
            "total_satellites_detected": _int(ed.get("satellite_matcher.num_satellites_detected")),
            "satellite_observations_json": json.dumps(observations) if observations else None,
            "should_upload": _bool_int(result.should_upload) if result else None,
            "skip_reason": result.skip_reason if result else None,
            "upload_type": upload_type,
            "upload_success": None,
            "processing_queued_at": processing_queued_at,
            "processing_started_at": processing_started_at,
            "processing_finished_at": processing_finished_at,
            "processing_queue_wait_s": queue_wait_s,
            "total_processing_time_s": result.total_time if result else None,
            "calibration_time_s": proc_times.get("calibration"),
            "plate_solve_time_s": proc_times.get("plate_solver"),
            "source_extractor_time_s": proc_times.get("source_extractor"),
            "photometry_time_s": proc_times.get("photometry"),
            "matcher_time_s": proc_times.get("satellite_matcher"),
            "annotated_image_time_s": proc_times.get("annotated_image"),
            "has_annotated_image": _bool_int(has_annotated),
            "annotated_image_path": str(annotated_path) if annotated_path else None,
            "extracted_data_json": json.dumps(ed) if ed else None,
        }

        columns = ", ".join(row.keys())
        placeholders = ", ".join(":" + k for k in row.keys())

        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO completed_tasks ({columns}) VALUES ({placeholders})",
                row,
            )
            self._conn.commit()

    def update_upload_result(self, task_id: str, success: bool) -> None:
        """UPDATE upload_success after upload completes (called from upload thread)."""
        with self._lock:
            self._conn.execute(
                "UPDATE completed_tasks SET upload_success = ? WHERE task_id = ?",
                (_bool_int(success), task_id),
            )
            self._conn.commit()

    # ── Reads ──────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> dict | None:
        """Return a single task row as a dict, or None.

        The row is augmented with lateness-attribution fields
        (``inherited_delay_s`` / ``self_delay_s`` / ``is_slip_origin``,
        plus ``prev_task_id`` / ``prev_target_name`` for UI linking) and
        slew estimate diagnostics from ``pointing_report_json``
        (``slew_estimate_total_s`` / ``slew_overrun_s``).  All derived,
        no schema columns required.
        """
        with self._lock:
            cur = self._conn.execute(_TASK_QUERY_WITH_LAG + " WHERE task_id = ?", (task_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = _row_to_dict(cur.description, row)
        _enrich_with_attribution(d)
        _enrich_with_pointing_diag(d)
        return d

    def query_tasks(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        sort: str = "completed_at",
        order: str = "desc",
        target_name: str | None = None,
        plate_solved: bool | None = None,
        target_matched: bool | None = None,
        missed_window: bool | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        filter_name: str | None = None,
        match_detail: str | None = None,
        upload_status: str | None = None,
    ) -> dict:
        """Paginated, filterable task list.

        Returns ``{"tasks": [...], "total": int}``.

        Extra filters (beyond the originals):

        ``filter_name``
            Exact match on the optical filter used (e.g. ``"r"``).

        ``match_detail``
            Richer satellite-match filter:
            ``"matched"``          — target satellite was detected
            ``"not_matched"``      — target satellite was NOT detected (but one was assigned)
            ``"has_extras"``       — incidental (non-target) detections > 0
            ``"no_detections"``    — total_satellites_detected = 0

        ``upload_status``
            ``"uploaded"``  — upload_success = 1
            ``"failed"``    — upload_success = 0
            ``"skipped"``   — should_upload = 0
        """
        if sort not in _SORTABLE_COLUMNS:
            sort = "completed_at"
        if order.lower() not in ("asc", "desc"):
            order = "desc"

        where_clauses: list[str] = []
        params: list[Any] = []

        if target_name:
            where_clauses.append("target_name LIKE ?")
            params.append(f"%{target_name}%")
        if plate_solved is not None:
            where_clauses.append("plate_solved = ?")
            params.append(_bool_int(plate_solved))
        if target_matched is not None:
            where_clauses.append("target_matched = ?")
            params.append(_bool_int(target_matched))
        if missed_window is not None:
            where_clauses.append("missed_window = ?")
            params.append(_bool_int(missed_window))
        if date_from:
            where_clauses.append("completed_at >= ?")
            params.append(date_from)
        if date_to:
            where_clauses.append("completed_at <= ?")
            params.append(date_to)
        if filter_name:
            where_clauses.append("filter_name = ?")
            params.append(filter_name)
        if match_detail:
            if match_detail == "matched":
                where_clauses.append("target_matched = 1")
            elif match_detail == "not_matched":
                where_clauses.append("target_matched = 0")
                where_clauses.append("target_satellite_id IS NOT NULL")
            elif match_detail == "has_extras":
                where_clauses.append("incidental_matches > 0")
            elif match_detail == "no_detections":
                where_clauses.append("COALESCE(total_satellites_detected, 0) = 0")
        if upload_status:
            if upload_status == "uploaded":
                where_clauses.append("upload_success = 1")
            elif upload_status == "failed":
                where_clauses.append("upload_success = 0")
            elif upload_status == "skipped":
                where_clauses.append("should_upload = 0")

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        with self._lock:
            count_row = self._conn.execute(f"SELECT COUNT(*) FROM completed_tasks{where_sql}", params).fetchone()
            total = count_row[0] if count_row else 0

            # sort/order are validated above — safe to interpolate.
            # The CTE adds prev_imaging_finished_at / prev_task_id /
            # prev_target_name via LAG(); _SORTABLE_COLUMNS only references
            # base columns that pass through SELECT *, so the existing sort
            # contract is unchanged.
            rows_cur = self._conn.execute(
                _TASK_QUERY_WITH_LAG + f"{where_sql} ORDER BY {sort} {order} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            )
            tasks = [_row_to_dict(rows_cur.description, row) for row in rows_cur.fetchall()]

        for t in tasks:
            _enrich_with_attribution(t)
            _enrich_with_pointing_diag(t)
        return {"tasks": tasks, "total": total}

    def get_distinct_filter_names(self) -> list[str]:
        """Return all distinct non-NULL filter names, sorted."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT filter_name FROM completed_tasks WHERE filter_name IS NOT NULL ORDER BY filter_name"
            ).fetchall()
        return [row[0] for row in rows]

    def get_stats(self, hours: int = 24) -> dict:
        """Aggregate statistics over the most recent *hours*."""
        cutoff = datetime.now(timezone.utc)
        from datetime import timedelta

        cutoff_iso = (cutoff - timedelta(hours=hours)).isoformat()

        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                COUNT(*)                                              AS task_count,
                SUM(CASE WHEN plate_solved = 1 THEN 1 ELSE 0 END)    AS plate_solved_count,
                SUM(CASE WHEN converged = 1 THEN 1 ELSE 0 END)       AS converged_count,
                AVG(convergence_attempts)                             AS avg_convergence_attempts,
                AVG(total_slew_time_s)                                AS avg_slew_time_s,
                AVG(pointing_error_deg)                               AS avg_pointing_error_deg,
                SUM(CASE WHEN target_matched = 1 THEN 1 ELSE 0 END)  AS target_matched_count,
                SUM(COALESCE(incidental_matches, 0))                  AS total_incidental,
                SUM(CASE WHEN upload_success = 1 THEN 1 ELSE 0 END)  AS upload_success_count,
                SUM(CASE WHEN upload_success IS NOT NULL THEN 1 ELSE 0 END)
                                                                      AS upload_attempted_count,
                AVG(zero_point)                                       AS avg_zero_point,
                -- stddev via Var = E[X²] - E[X]² (NULL-safe)
                CASE WHEN COUNT(zero_point) > 1
                    THEN SQRT(AVG(zero_point * zero_point) - AVG(zero_point) * AVG(zero_point))
                    ELSE NULL END                                     AS stddev_zero_point,
                SUM(CASE WHEN missed_window = 1 THEN 1 ELSE 0 END)   AS missed_window_count,
                SUM(CASE WHEN missed_window = 0 THEN 1 ELSE 0 END)   AS on_time_count,
                AVG(window_start_delay_s)                             AS avg_window_start_delay_s,
                -- Per-processor timing
                AVG(processing_queue_wait_s)                          AS avg_queue_wait_s,
                AVG(total_processing_time_s)                          AS avg_total_processing_s,
                AVG(calibration_time_s)                               AS avg_calibration_s,
                AVG(plate_solve_time_s)                               AS avg_plate_solve_s,
                AVG(source_extractor_time_s)                          AS avg_source_extractor_s,
                AVG(photometry_time_s)                                AS avg_photometry_s,
                AVG(matcher_time_s)                                   AS avg_matcher_s,
                AVG(annotated_image_time_s)                           AS avg_annotated_image_s,
                -- Target satellite
                SUM(CASE WHEN target_satellite_id IS NOT NULL THEN 1 ELSE 0 END)
                                                                      AS satellite_task_count
                FROM completed_tasks
                WHERE completed_at >= ?
                """,
                (cutoff_iso,),
            ).fetchone()

        if not row or row["task_count"] == 0:
            return empty_stats()

        tc = row["task_count"] or 1
        sat_tc = row["satellite_task_count"] or 0
        missed = row["missed_window_count"] or 0
        on_time = row["on_time_count"] or 0

        elapsed_hours = hours or 1
        return {
            "task_count": tc,
            "tasks_per_hour": round(tc / elapsed_hours, 2),
            "plate_solve_rate": _pct(row["plate_solved_count"], tc),
            "convergence_rate": _pct(row["converged_count"], tc),
            "avg_convergence_attempts": _rnd(row["avg_convergence_attempts"]),
            "avg_slew_time_s": _rnd(row["avg_slew_time_s"]),
            "avg_pointing_error_deg": _rnd(row["avg_pointing_error_deg"], 4),
            "target_match_rate": _pct(row["target_matched_count"], sat_tc) if sat_tc else None,
            "total_incidental_detections": row["total_incidental"] or 0,
            "upload_success_rate": (
                _pct(row["upload_success_count"], row["upload_attempted_count"])
                if row["upload_attempted_count"]
                else None
            ),
            "avg_zero_point": _rnd(row["avg_zero_point"]),
            "stddev_zero_point": _rnd(row["stddev_zero_point"]),
            "window_compliance_rate": _pct(on_time, missed + on_time) if (missed + on_time) > 0 else None,
            "missed_window_count": missed,
            "avg_window_start_delay_s": _rnd(row["avg_window_start_delay_s"]),
            "avg_queue_wait_s": _rnd(row["avg_queue_wait_s"]),
            "avg_total_processing_s": _rnd(row["avg_total_processing_s"]),
            "per_processor_timing": {
                "calibration_s": _rnd(row["avg_calibration_s"]),
                "plate_solve_s": _rnd(row["avg_plate_solve_s"]),
                "source_extractor_s": _rnd(row["avg_source_extractor_s"]),
                "photometry_s": _rnd(row["avg_photometry_s"]),
                "satellite_matcher_s": _rnd(row["avg_matcher_s"]),
                "annotated_image_s": _rnd(row["avg_annotated_image_s"]),
            },
            "hours": hours,
        }


# ── Helpers ────────────────────────────────────────────────────────────


def empty_stats() -> dict:
    return {
        "task_count": 0,
        "tasks_per_hour": 0,
        "plate_solve_rate": None,
        "convergence_rate": None,
        "avg_convergence_attempts": None,
        "avg_slew_time_s": None,
        "avg_pointing_error_deg": None,
        "target_match_rate": None,
        "total_incidental_detections": 0,
        "upload_success_rate": None,
        "avg_zero_point": None,
        "stddev_zero_point": None,
        "window_compliance_rate": None,
        "missed_window_count": 0,
        "avg_window_start_delay_s": None,
        "avg_queue_wait_s": None,
        "avg_total_processing_s": None,
        "per_processor_timing": {
            "calibration_s": None,
            "plate_solve_s": None,
            "source_extractor_s": None,
            "photometry_s": None,
            "satellite_matcher_s": None,
            "annotated_image_s": None,
        },
        "hours": 24,
    }


# Shared CTE that joins each completed-task row with the previous task's
# imaging-finish timestamp (and identity), used by both ``get_task`` and
# ``query_tasks`` so the lateness-attribution math has the same input
# regardless of which path produced the row.  Ordering is by
# ``imaging_started_at`` (NOT ``completed_at``) because completion order
# is shuffled by parallel processing — only imaging order represents the
# actual on-sky sequence the operator cares about.  Rows with NULL
# ``imaging_started_at`` get arbitrary LAG values; the enrichment helper
# guards on null timestamps so this is harmless.
_TASK_QUERY_WITH_LAG = """
    WITH ordered AS (
        SELECT
            *,
            LAG(imaging_finished_at) OVER (ORDER BY imaging_started_at) AS prev_imaging_finished_at,
            LAG(task_id)             OVER (ORDER BY imaging_started_at) AS prev_task_id,
            LAG(target_name)         OVER (ORDER BY imaging_started_at) AS prev_target_name
        FROM completed_tasks
    )
    SELECT * FROM ordered
"""


def _row_to_dict(description: Any, row: Any) -> dict:
    """Convert a sqlite3.Row (or plain tuple with cursor description) to a dict."""
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return {col[0]: val for col, val in zip(description, row, strict=False)}


def _enrich_with_attribution(row: dict) -> dict:
    """Attribute lateness to inherited (from prior task) vs self (added here).

    Mutates and returns ``row``.  Adds three derived fields:

      ``inherited_delay_s``  Seconds of this task's window already burned
                             when the window opened, because the prior task
                             ran long.  Always ``>= 0`` and ``<= window_start_delay_s``.
      ``self_delay_s``       Remainder of ``window_start_delay_s`` not
                             explained by the prior task.  Always ``>= 0``.
      ``is_slip_origin``     True when ``self_delay_s >= SLIP_ORIGIN_THRESHOLD_S``.
                             These are the rows worth investigating first
                             when an operator asks "why are we late?".

    Inputs (set by the caller / the CTE):
      - ``window_start_delay_s`` (column on the row)
      - ``window_start`` (column on the row)
      - ``prev_imaging_finished_at`` (LAG output from ``_TASK_QUERY_WITH_LAG``)

    Cross-session guard: when the previous task finished more than
    ``CROSS_SESSION_GAP_S`` before this task's window opened, we drop the
    prev-task linkage entirely so the UI doesn't surface an irrelevant
    "inherited from <hours-old task>" hint.
    """
    delay = row.get("window_start_delay_s")
    window_start = row.get("window_start")
    prev_finish = row.get("prev_imaging_finished_at")

    # Early start, missing start delay, or missing window — nothing to attribute.
    if delay is None or delay <= 0 or not window_start:
        row["inherited_delay_s"] = 0.0
        row["self_delay_s"] = max(0.0, delay or 0.0)
        row["is_slip_origin"] = row["self_delay_s"] >= SLIP_ORIGIN_THRESHOLD_S
        return row

    inherited = 0.0
    if prev_finish:
        # Positive ``bleed`` means the prior task ran into our window.
        # Negative means the prior task ended in time and there was a gap.
        bleed = _iso_diff_seconds(window_start, prev_finish) or 0.0
        if bleed >= 0:
            inherited = min(delay, bleed)
        elif -bleed > CROSS_SESSION_GAP_S:
            # Different observing session — drop the linkage so the UI
            # doesn't show a misleading "inherited from <stale task>" hint.
            row["prev_task_id"] = None
            row["prev_target_name"] = None

    row["inherited_delay_s"] = round(inherited, 3)
    row["self_delay_s"] = round(max(0.0, delay - inherited), 3)
    row["is_slip_origin"] = row["self_delay_s"] >= SLIP_ORIGIN_THRESHOLD_S
    return row


def _enrich_with_pointing_diag(row: dict) -> dict:
    """Surface slew estimate-vs-actual signals from ``pointing_report_json``.

    Tier-2 diagnosis: when a task is flagged as a slip origin, the operator's
    next question is "why was the slew slow?".  The pointing report records
    ``estimated_slew_time_s`` and ``actual_slew_time_s`` per convergence
    iteration; we surface the totals so the UI can show "slew took N seconds
    longer than estimated" without re-parsing the JSON blob client-side.

    Mutates and returns ``row``.  Adds:
      ``slew_estimate_total_s``  Sum of per-iteration ``estimated_slew_time_s``.
                                 ``None`` when the JSON is missing/malformed
                                 or no iteration recorded an estimate.
      ``slew_overrun_s``         ``total_slew_time_s - slew_estimate_total_s``.
                                 Positive ⇒ slew took longer than estimated.

    The convergence-attempts count is already a top-level column, so we
    deliberately don't duplicate it here.
    """
    raw = row.get("pointing_report_json")
    if not raw:
        row["slew_estimate_total_s"] = None
        row["slew_overrun_s"] = None
        return row

    try:
        report = json.loads(raw)
    except (TypeError, ValueError):
        row["slew_estimate_total_s"] = None
        row["slew_overrun_s"] = None
        return row

    estimate_total = 0.0
    have_any_estimate = False
    for it in report.get("iterations") or []:
        est = it.get("estimated_slew_time_s")
        if est is None:
            continue
        try:
            estimate_total += float(est)
            have_any_estimate = True
        except (TypeError, ValueError):
            continue

    if not have_any_estimate:
        row["slew_estimate_total_s"] = None
        row["slew_overrun_s"] = None
        return row

    row["slew_estimate_total_s"] = round(estimate_total, 3)
    actual = row.get("total_slew_time_s")
    row["slew_overrun_s"] = round(actual - estimate_total, 3) if actual is not None else None
    return row


def _float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _str(v: Any) -> str | None:
    return str(v) if v is not None else None


def _bool_int(v: Any) -> int | None:
    if v is None:
        return None
    return 1 if v else 0


def _pct(numerator: int | None, denominator: int | None) -> float | None:
    if not numerator or not denominator:
        return 0.0
    return round(numerator / denominator * 100, 1)


def _rnd(v: float | None, digits: int = 3) -> float | None:
    if v is None:
        return None
    return round(v, digits)


def _iso_diff_seconds(earlier_iso: str | None, later_iso: str | None) -> float | None:
    """Return seconds between two ISO timestamps, or None if either is missing."""
    if not earlier_iso or not later_iso:
        return None
    try:
        t1 = datetime.fromisoformat(earlier_iso)
        t2 = datetime.fromisoformat(later_iso)
        return (t2 - t1).total_seconds()
    except (ValueError, TypeError):
        return None


def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Spherical law of cosines angular distance between two (RA, Dec) points in degrees."""
    import math

    ra1_r, ra2_r = math.radians(ra1), math.radians(ra2)
    dec1_r, dec2_r = math.radians(dec1), math.radians(dec2)
    cos_d = math.sin(dec1_r) * math.sin(dec2_r) + math.cos(dec1_r) * math.cos(dec2_r) * math.cos(ra1_r - ra2_r)
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))
