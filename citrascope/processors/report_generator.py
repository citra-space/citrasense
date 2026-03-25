"""Generate a self-contained HTML report from processing pipeline artifacts.

Reads JSON artifacts and the annotated PNG from a processing working directory
and produces a single ``report.html`` that can be opened in any browser.  The
annotated image is base64-embedded so the file is fully portable.

Works on both normal ``processing/<task_id>/`` debug directories (when
``keep_processing_output`` is enabled) and reprocessed output directories,
since both use the same artifact layout.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("citrascope.report_generator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _deg_to_hms(deg: float) -> str:
    """Convert RA in degrees to HH:MM:SS.ss string."""
    h = deg / 15.0
    hh = int(h)
    m = (h - hh) * 60
    mm = int(m)
    ss = (m - mm) * 60
    return f"{hh:02d}h {mm:02d}m {ss:05.2f}s"


def _deg_to_dms(deg: float) -> str:
    """Convert Dec in degrees to +DD:MM:SS.s string."""
    sign = "+" if deg >= 0 else "-"
    deg = abs(deg)
    dd = int(deg)
    m = (deg - dd) * 60
    mm = int(m)
    ss = (m - mm) * 60
    return f"{sign}{dd:02d}° {mm:02d}' {ss:04.1f}\""


def _fmt(val: Any, precision: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def _embed_image(path: Path) -> str | None:
    """Read a PNG and return a base64 data URI, or None if missing."""
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Section builders — each returns an HTML fragment string
# ---------------------------------------------------------------------------


def _section_header(task: dict | None, fits_header: dict | None) -> str:
    if not task:
        return ""
    sat = task.get("satelliteName", "Unknown")
    task_type = task.get("type", "")
    task_id = task.get("id", "")
    gs = task.get("groundStationName", "")
    date_obs = (fits_header or {}).get("DATE-OBS", "")
    return f"""
    <div class="header">
        <h1>{_esc(sat)}</h1>
        <div class="subtitle">{_esc(task_type)} &middot; {_esc(date_obs)} &middot; {_esc(gs)}</div>
        <div class="task-id">Task {_esc(task_id)}</div>
    </div>"""


def _section_image(working_dir: Path) -> str:
    for name in ("annotated.png", "latest_preview.png"):
        uri = _embed_image(working_dir / name)
        if uri:
            return f"""
    <div class="section">
        <h2>Annotated Image</h2>
        <img src="{uri}" class="annotated-img" alt="Annotated image"/>
    </div>"""
    return ""


def _section_pipeline_summary(summary: dict | None) -> str:
    if not summary:
        return ""
    rows = ""
    for p in summary.get("processors", []):
        ok = p.get("should_upload", False) and p.get("confidence", 0) > 0
        badge = '<span class="badge ok">OK</span>' if ok else '<span class="badge fail">FAIL</span>'
        rows += f"""
            <tr>
                <td>{badge}</td>
                <td>{_esc(p.get('processor_name', ''))}</td>
                <td>{p.get('processing_time_seconds', 0):.2f}s</td>
                <td>{_esc(p.get('reason', ''))}</td>
            </tr>"""
    total = summary.get("total_time", 0)
    upload = summary.get("should_upload", False)
    skip = summary.get("skip_reason")
    footer = f"Total: {total:.2f}s &middot; Upload: {'Yes' if upload else 'No'}"
    if skip:
        footer += f" &middot; Skip: {_esc(skip)}"
    return f"""
    <div class="section">
        <h2>Pipeline Summary</h2>
        <table>
            <thead><tr><th></th><th>Processor</th><th>Time</th><th>Result</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div class="footer-note">{footer}</div>
    </div>"""


def _section_pointing(pointing: dict | None) -> str:
    if not pointing:
        return ""
    converged = pointing.get("converged", False)
    attempts = pointing.get("attempts", 0)
    max_attempts = pointing.get("max_attempts", "?")
    final_dist = pointing.get("final_angular_distance_deg")
    final_ra = pointing.get("final_telescope_ra_deg")
    final_dec = pointing.get("final_telescope_dec_deg")
    threshold = pointing.get("convergence_threshold_deg")
    slew_rate = pointing.get("configured_slew_rate_deg_per_s")

    status = "Converged" if converged else "Did not converge"
    status_class = "ok" if converged else "fail"

    rows = ""
    for it in pointing.get("iterations", []):
        rows += f"""
            <tr>
                <td>{it.get('attempt', '?')}</td>
                <td>{_fmt(it.get('target_lead_ra_deg'))}, {_fmt(it.get('target_lead_dec_deg'))}</td>
                <td>{_fmt(it.get('post_slew_ra_deg'))}, {_fmt(it.get('post_slew_dec_deg'))}</td>
                <td>{_fmt(it.get('angular_distance_to_satellite_deg'), 5)}&deg;</td>
                <td>{_fmt(it.get('observed_slew_rate_deg_per_s'), 3)}&deg;/s</td>
            </tr>"""

    iter_table = ""
    if rows:
        iter_table = f"""
        <table>
            <thead><tr><th>#</th><th>Target Lead RA, Dec</th><th>Actual RA, Dec</th>
            <th>Angular Dist</th><th>Slew Rate</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    return f"""
    <div class="section">
        <h2>Pointing Analysis</h2>
        <div class="kv-grid">
            <span class="label">Status</span><span class="badge {status_class}">{status}</span>
            <span class="label">Attempts</span><span>{attempts} / {max_attempts}</span>
            <span class="label">Final Offset</span><span>{_fmt(final_dist, 5)}&deg;</span>
            <span class="label">Threshold</span><span>{_fmt(threshold, 5)}&deg;</span>
            <span class="label">Final Mount Position</span>
            <span>{_fmt(final_ra)}, {_fmt(final_dec)}</span>
            <span class="label">Configured Slew Rate</span><span>{_fmt(slew_rate, 3)}&deg;/s</span>
        </div>
        {iter_table}
    </div>"""


def _section_plate_solve(plate: dict | None, sat_debug: dict | None) -> str:
    if not plate:
        return ""
    ed = plate.get("extracted_data", {})
    solved = ed.get("plate_solved", False)
    if not solved:
        return f"""
    <div class="section">
        <h2>Plate Solve</h2>
        <div class="badge fail">FAILED</div>
        <p>{_esc(plate.get('reason', ''))}</p>
    </div>"""

    ra = ed.get("ra_center")
    dec = ed.get("dec_center")
    ra_str = _deg_to_hms(ra) if ra is not None else "N/A"
    dec_str = _deg_to_dms(dec) if dec is not None else "N/A"

    # Pointing TLE vs field center offset
    tle_info = ""
    if sat_debug and isinstance(sat_debug.get("target_satellite"), dict):
        ts = sat_debug["target_satellite"]
        tle_match = ts.get("tle_match")
        match_text = "Match" if tle_match else ("Stale" if tle_match is False else "Unknown")
        match_class = "ok" if tle_match else "fail"
        tle_info = f"""
            <span class="label">Pointing TLE vs Cache</span>
            <span class="badge {match_class}">{match_text}</span>
            <span class="label">Pointing Elset Epoch</span>
            <span>{_esc(str(ts.get('pointing_elset_epoch', 'N/A')))}</span>"""

    return f"""
    <div class="section">
        <h2>Plate Solve</h2>
        <div class="kv-grid">
            <span class="label">Field Center (RA)</span>
            <span>{ra_str} ({_fmt(ra)}&deg;)</span>
            <span class="label">Field Center (Dec)</span>
            <span>{dec_str} ({_fmt(dec)}&deg;)</span>
            <span class="label">Pixel Scale</span>
            <span>{_fmt(ed.get('pixel_scale'), 3)} arcsec/px</span>
            <span class="label">Field Size</span>
            <span>{_fmt(ed.get('field_width_deg'), 3)}&deg; &times; {_fmt(ed.get('field_height_deg'), 3)}&deg;</span>
            <span class="label">Sources Extracted</span>
            <span>{ed.get('num_sources', 'N/A')}</span>{tle_info}
        </div>
    </div>"""


def _section_source_classification(sat_debug: dict | None) -> str:
    if not sat_debug:
        return ""
    sc = sat_debug.get("source_classification")
    if not sc:
        return ""
    return f"""
    <div class="section">
        <h2>Source Classification</h2>
        <div class="kv-grid">
            <span class="label">Total Sources</span><span>{sc.get('total_sources', 'N/A')}</span>
            <span class="label">Satellite Candidates</span><span>{sc.get('satellite_candidate_count', 'N/A')}</span>
            <span class="label">Star-like</span><span>{sc.get('star_like_count', 'N/A')}</span>
            <span class="label">Tracking Mode</span><span>{_esc(str(sat_debug.get('tracking_mode', 'N/A')))}</span>
            <span class="label">Elongation Filter</span>
            <span>{'Applied' if sat_debug.get('elongation_filter_applied') else 'Not applied'}
            (threshold: {_fmt(sat_debug.get('elongation_threshold'), 2)})</span>
            <span class="label">Elongation (min / median / max)</span>
            <span>{_fmt(sc.get('elongation_min'), 2)} / {_fmt(sc.get('elongation_median'), 2)} \
/ {_fmt(sc.get('elongation_max'), 2)}</span>
        </div>
    </div>"""


def _section_photometry(phot: dict | None) -> str:
    if not phot:
        return ""
    ed = phot.get("extracted_data", {})
    return f"""
    <div class="section">
        <h2>Photometry</h2>
        <div class="kv-grid">
            <span class="label">Zero Point</span><span>{_fmt(ed.get('zero_point'), 3)}</span>
            <span class="label">Calibration Stars</span><span>{ed.get('num_calibration_stars', 'N/A')}</span>
            <span class="label">Filter Band</span><span>{_esc(str(ed.get('filter', 'N/A')))}</span>
        </div>
    </div>"""


def _section_predictions_in_field(sat_debug: dict | None) -> str:
    if not sat_debug:
        return ""
    preds = sat_debug.get("predictions_in_field", [])
    if not preds:
        return ""
    count = sat_debug.get("predictions_in_field_count", len(preds))
    total = sat_debug.get("elset_count", "?")
    rows = ""
    for p in preds:
        rows += f"""
            <tr>
                <td>{_esc(p.get('name', ''))}</td>
                <td>{_fmt(p.get('predicted_ra_deg'))}</td>
                <td>{_fmt(p.get('predicted_dec_deg'))}</td>
                <td>{_fmt(p.get('distance_from_center_deg'), 3)}&deg;</td>
                <td>{_fmt(p.get('phase_angle'), 1)}&deg;</td>
            </tr>"""
    return f"""
    <div class="section">
        <h2>Satellite Predictions In Field</h2>
        <div class="footer-note">{count} predicted in field out of {total} TLEs searched</div>
        <table>
            <thead><tr><th>Satellite</th><th>Predicted RA</th><th>Predicted Dec</th>
            <th>Dist from Center</th><th>Phase Angle</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""


def _section_match_results(sat_debug: dict | None) -> str:
    if not sat_debug:
        return ""
    observations = sat_debug.get("satellite_observations", [])
    reverse = sat_debug.get("reverse_match", [])

    # Build matched satellite table
    matched_rows = ""
    for obs in observations:
        matched_rows += f"""
            <tr>
                <td>{_esc(obs.get('name', ''))}</td>
                <td>{_fmt(obs.get('ra'))}</td>
                <td>{_fmt(obs.get('dec'))}</td>
                <td>{_fmt(obs.get('mag'), 2)}</td>
                <td>{_fmt(obs.get('mag_instrumental'), 2)}</td>
                <td>{_fmt(obs.get('phase_angle'), 1)}&deg;</td>
                <td>{_fmt(obs.get('elongation'), 2)}</td>
            </tr>"""

    matched_table = ""
    if matched_rows:
        matched_table = f"""
        <h3>Matched Satellites ({len(observations)})</h3>
        <table>
            <thead><tr><th>Satellite</th><th>Obs RA</th><th>Obs Dec</th>
            <th>Cal Mag</th><th>Inst Mag</th><th>Phase</th><th>Elongation</th></tr></thead>
            <tbody>{matched_rows}</tbody>
        </table>"""

    # Build unmatched predictions table from reverse_match
    unmatched_rows = ""
    unmatched_count = 0
    for rm in reverse:
        if rm.get("within_match_radius"):
            continue
        unmatched_count += 1
        unmatched_rows += f"""
            <tr>
                <td>{_esc(rm.get('name', ''))}</td>
                <td>{_fmt(rm.get('predicted_ra'))}, {_fmt(rm.get('predicted_dec'))}</td>
                <td>{_fmt(rm.get('nearest_source_distance_arcmin'), 2)}'</td>
                <td>{_fmt(rm.get('nearest_source_mag'), 2)}</td>
                <td>Beyond match radius</td>
            </tr>"""

    unmatched_table = ""
    if unmatched_rows:
        unmatched_table = f"""
        <h3>Unmatched Predictions ({unmatched_count})</h3>
        <table>
            <thead><tr><th>Satellite</th><th>Predicted RA, Dec</th>
            <th>Nearest Source</th><th>Source Mag</th><th>Reason</th></tr></thead>
            <tbody>{unmatched_rows}</tbody>
        </table>"""

    if not matched_table and not unmatched_table:
        return ""

    match_radius = sat_debug.get("match_radius_arcmin", "?")
    return f"""
    <div class="section">
        <h2>Satellite Matching Results</h2>
        <div class="footer-note">Match radius: {match_radius} arcmin</div>
        {matched_table}
        {unmatched_table}
    </div>"""


def _section_target_tle(sat_debug: dict | None) -> str:
    if not sat_debug:
        return ""
    ts = sat_debug.get("target_satellite")
    if not ts or not isinstance(ts, dict):
        return ""
    tle_match = ts.get("tle_match")
    match_text = "Identical" if tle_match else ("Different" if tle_match is False else "Unknown")
    match_class = "ok" if tle_match else "fail"

    pointing_tle = ts.get("pointing_tle") or ["N/A", "N/A"]
    cache_tle = ts.get("cache_tle") or ["N/A", "N/A"]

    return f"""
    <div class="section">
        <h2>Target Satellite TLE Comparison</h2>
        <div class="kv-grid">
            <span class="label">Satellite ID</span><span>{_esc(str(ts.get('satellite_id', '')))}</span>
            <span class="label">TLE Match</span><span class="badge {match_class}">{match_text}</span>
            <span class="label">Pointing Elset Epoch</span>
            <span>{_esc(str(ts.get('pointing_elset_epoch', 'N/A')))}</span>
        </div>
        <h3>Pointing TLE (used to aim telescope)</h3>
        <pre>{_esc(str(pointing_tle[0]))}\n{_esc(str(pointing_tle[1]))}</pre>
        <h3>Cache TLE (used for matching)</h3>
        <pre>{_esc(str(cache_tle[0]))}\n{_esc(str(cache_tle[1]))}</pre>
    </div>"""


def _section_observation_details(fits_header: dict | None, sat_debug: dict | None) -> str:
    if not fits_header:
        return ""
    tracking = (sat_debug or {}).get("tracking_mode", "N/A")
    return f"""
    <div class="section">
        <h2>Observation Details</h2>
        <div class="kv-grid">
            <span class="label">Date/Time (UTC)</span>
            <span>{_esc(str(fits_header.get('DATE-OBS', 'N/A')))}</span>
            <span class="label">Exposure</span><span>{fits_header.get('EXPTIME', 'N/A')}s</span>
            <span class="label">Filter</span><span>{_esc(str(fits_header.get('FILTER', 'N/A')))}</span>
            <span class="label">Binning</span>
            <span>{fits_header.get('XBINNING', '?')} &times; {fits_header.get('YBINNING', '?')}</span>
            <span class="label">Image Size</span>
            <span>{fits_header.get('NAXIS1', '?')} &times; {fits_header.get('NAXIS2', '?')} px</span>
            <span class="label">Camera</span><span>{_esc(str(fits_header.get('INSTRUME', 'N/A')))}</span>
            <span class="label">Gain</span><span>{fits_header.get('GAIN', 'N/A')}</span>
            <span class="label">Telescope</span><span>{_esc(str(fits_header.get('TELESCOP', 'N/A')))}</span>
            <span class="label">Tracking Mode</span><span>{_esc(str(tracking))}</span>
            <span class="label">Site Location</span>
            <span>{fits_header.get('SITELAT', 'N/A')}, {fits_header.get('SITELONG', 'N/A')} \
(alt {fits_header.get('SITEALT', 'N/A')}m)</span>
        </div>
    </div>"""


def _section_task_metadata(task: dict | None) -> str:
    if not task:
        return ""
    return f"""
    <div class="section">
        <h2>Task Metadata</h2>
        <div class="kv-grid">
            <span class="label">Task ID</span><span class="mono">{_esc(task.get('id', ''))}</span>
            <span class="label">Type</span><span>{_esc(task.get('type', ''))}</span>
            <span class="label">Status</span><span>{_esc(task.get('status', ''))}</span>
            <span class="label">Target</span>
            <span>{_esc(task.get('satelliteName', ''))} ({_esc(task.get('satelliteId', ''))})</span>
            <span class="label">Time Window</span>
            <span>{_esc(task.get('taskStart', ''))} &rarr; {_esc(task.get('taskStop', ''))}</span>
            <span class="label">User</span><span>{_esc(task.get('username', ''))}</span>
            <span class="label">Telescope</span><span>{_esc(task.get('telescopeName', ''))}</span>
            <span class="label">Ground Station</span><span>{_esc(task.get('groundStationName', ''))}</span>
            <span class="label">Filter</span>
            <span>{_esc(str(task.get('assigned_filter_name', 'N/A')))}</span>
        </div>
    </div>"""


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #5b9bd5;
    --ok: #4caf50;
    --fail: #f44336;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    max-width: 1100px; margin: 0 auto; padding: 24px;
    line-height: 1.5;
}
.header { text-align: center; margin-bottom: 32px; }
.header h1 { font-size: 2em; margin-bottom: 4px; }
.subtitle { color: var(--text-dim); font-size: 1.1em; }
.task-id { color: var(--text-dim); font-size: 0.85em; font-family: monospace; margin-top: 4px; }
.section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin-bottom: 20px;
}
.section h2 { font-size: 1.2em; margin-bottom: 12px; color: var(--accent); }
.section h3 { font-size: 1em; margin: 16px 0 8px 0; color: var(--text-dim); }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td {
    text-align: left; padding: 6px 10px;
    border-bottom: 1px solid var(--border); font-size: 0.9em;
}
th { color: var(--text-dim); font-weight: 600; font-size: 0.8em; text-transform: uppercase; }
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.8em; font-weight: 600;
}
.badge.ok { background: rgba(76,175,80,0.15); color: var(--ok); }
.badge.fail { background: rgba(244,67,54,0.15); color: var(--fail); }
.kv-grid {
    display: grid; grid-template-columns: 200px 1fr;
    gap: 6px 16px; align-items: baseline;
}
.kv-grid .label { color: var(--text-dim); font-size: 0.9em; }
.footer-note { color: var(--text-dim); font-size: 0.85em; margin-top: 8px; }
.annotated-img { max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto; }
pre {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 10px; font-size: 0.8em; overflow-x: auto; white-space: pre-wrap;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.9em; }
.generated { text-align: center; color: var(--text-dim); font-size: 0.75em; margin-top: 32px; }
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_html_report(working_dir: Path) -> Path | None:
    """Generate an HTML report from processing artifacts in *working_dir*.

    Best-effort: logs warnings on failure, never raises.  Returns the path
    to ``report.html`` on success, or ``None`` on failure.
    """
    try:
        return _build_report(working_dir)
    except Exception as exc:
        logger.warning("Failed to generate HTML report: %s", exc)
        return None


def _build_report(working_dir: Path) -> Path:
    task = _load_json(working_dir / "task.json")
    fits_header = _load_json(working_dir / "fits_header.json")
    summary = _load_json(working_dir / "processing_summary.json")
    plate = _load_json(working_dir / "plate_solver_result.json")
    phot = _load_json(working_dir / "photometry_result.json")
    sat_debug = _load_json(working_dir / "satellite_matcher_debug.json")
    pointing = _load_json(working_dir / "pointing_report.json")

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    body = "".join(
        [
            _section_header(task, fits_header),
            _section_image(working_dir),
            _section_pipeline_summary(summary),
            _section_pointing(pointing),
            _section_plate_solve(plate, sat_debug),
            _section_source_classification(sat_debug),
            _section_photometry(phot),
            _section_predictions_in_field(sat_debug),
            _section_match_results(sat_debug),
            _section_target_tle(sat_debug),
            _section_observation_details(fits_header, sat_debug),
            _section_task_metadata(task),
        ]
    )

    title = "Processing Report"
    if task:
        title = f"{task.get('satelliteName', 'Unknown')} — Processing Report"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{_esc(title)}</title>
    <style>{_CSS}</style>
</head>
<body>
{body}
    <div class="generated">Generated by citrascope &middot; {now}</div>
</body>
</html>
"""
    out_path = working_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", out_path)
    return out_path
