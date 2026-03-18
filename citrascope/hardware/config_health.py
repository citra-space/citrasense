"""Telescope configuration health assessment.

Compares the server-side telescope record against values observed from
hardware (camera sensor info), astrometry (plate solving), and mount
dynamics (tracking).  Returns structured results so the UI can render
them without doing any maths.

Each check carries a ``group`` field ("optics" or "telescope") so the
UI knows which card to render it on without hardcoded name matching.

Usage from the status loop::

    from citrascope.hardware.config_health import assess_config_health

    health = assess_config_health(
        telescope_record=daemon.telescope_record,
        camera_info=camera_info_dict,
        observed_pixel_scale=adapter.observed_pixel_scale_arcsec,
        observed_fov_w=adapter.observed_fov_w_deg,
        observed_fov_h=adapter.observed_fov_h_deg,
        observed_slew_rate=adapter.observed_slew_rate_deg_per_s,
    )
    status.config_health = health.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, field

MISMATCH_THRESHOLD_PCT = 10.0


@dataclass
class HardwareConfigCheck:
    """One configured-vs-observed comparison."""

    name: str
    label: str
    group: str = "optics"
    configured: float | None = None
    configured_fmt: str = ""
    observed: float | None = None
    observed_fmt: str = ""
    unit: str = ""
    source: str = ""
    pct_diff: float | None = None
    status: str = "pending"


@dataclass
class ConfigHealth:
    """Full telescope configuration health assessment."""

    checks: list[HardwareConfigCheck] = field(default_factory=list)
    has_warnings: bool = False

    def to_dict(self) -> dict:
        out: dict = {"has_warnings": self.has_warnings, "checks": []}
        for c in self.checks:
            out["checks"].append(
                {
                    "name": c.name,
                    "label": c.label,
                    "group": c.group,
                    "configured": c.configured,
                    "configured_fmt": c.configured_fmt,
                    "observed": c.observed,
                    "observed_fmt": c.observed_fmt,
                    "unit": c.unit,
                    "source": c.source,
                    "pct_diff": c.pct_diff,
                    "status": c.status,
                }
            )
        return out


def _pct(cfg: float, obs: float) -> float:
    if cfg == 0:
        return 0.0
    return abs(obs - cfg) / cfg * 100.0


def _status(pct: float, threshold: float = MISMATCH_THRESHOLD_PCT) -> str:
    return "warning" if pct > threshold else "ok"


def _fmt(val: float | None, decimals: int = 2, suffix: str = "") -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}{suffix}"


def assess_config_health(
    telescope_record: dict | None = None,
    camera_info: dict | None = None,
    observed_pixel_scale: float | None = None,
    observed_fov_w: float | None = None,
    observed_fov_h: float | None = None,
    observed_slew_rate: float | None = None,
) -> ConfigHealth:
    """Build a complete telescope configuration health assessment.

    All comparison logic lives here — the UI just renders the result.
    """
    health = ConfigHealth()

    if not telescope_record:
        return health

    ps_um = _safe_float(telescope_record.get("pixelSize"))
    fl_mm = _safe_float(telescope_record.get("focalLength"))
    f_ratio = _safe_float(telescope_record.get("focalRatio"))
    cfg_h_px = _safe_int(telescope_record.get("horizontalPixelCount"))
    cfg_v_px = _safe_int(telescope_record.get("verticalPixelCount"))
    cfg_slew = _safe_float(telescope_record.get("maxSlewRate"))

    cfg_pixel_scale = ps_um / fl_mm * 206.265 if ps_um and fl_mm else None

    # ── Focal length & ratio (config only, no hardware source) ──
    if fl_mm:
        ratio_str = f"  f/{f_ratio}" if f_ratio else ""
        health.checks.append(
            HardwareConfigCheck(
                name="focal_length",
                label="Focal Length",
                group="optics",
                configured=fl_mm,
                configured_fmt=f"{fl_mm:.0f} mm{ratio_str}",
                unit="mm",
            )
        )

    # ── Pixel size (config vs camera hardware) ──
    if ps_um:
        chk = HardwareConfigCheck(
            name="pixel_size",
            label="Pixel Size",
            group="optics",
            configured=ps_um,
            configured_fmt=f"{ps_um} \u00b5m",
            unit="\u00b5m",
        )
        hw_ps = _safe_float(camera_info.get("pixel_size_um")) if camera_info else None
        if hw_ps:
            pct = _pct(ps_um, hw_ps)
            chk.observed = hw_ps
            chk.observed_fmt = f"{hw_ps} \u00b5m"
            chk.source = "camera"
            chk.pct_diff = round(pct, 1)
            chk.status = _status(pct)
        health.checks.append(chk)

    # ── Pixel scale (config vs plate solve) ──
    if cfg_pixel_scale:
        chk = HardwareConfigCheck(
            name="pixel_scale",
            label="Pixel Scale",
            group="optics",
            configured=round(cfg_pixel_scale, 2),
            configured_fmt=f'{cfg_pixel_scale:.2f}"/px',
            unit='"/px',
        )
        if observed_pixel_scale is not None:
            pct = _pct(cfg_pixel_scale, observed_pixel_scale)
            chk.observed = round(observed_pixel_scale, 2)
            chk.observed_fmt = f'{observed_pixel_scale:.2f}"/px'
            chk.source = "plate_solve"
            chk.pct_diff = round(pct, 1)
            chk.status = _status(pct)
        health.checks.append(chk)

    # ── Sensor resolution (config vs camera hardware) ──
    if cfg_h_px and cfg_v_px:
        chk = HardwareConfigCheck(
            name="sensor_resolution",
            label="Sensor",
            group="optics",
            configured=cfg_h_px,
            configured_fmt=f"{cfg_h_px}\u00d7{cfg_v_px} px",
            unit="px",
        )
        hw_w = _safe_int(camera_info.get("width")) if camera_info else None
        hw_h = _safe_int(camera_info.get("height")) if camera_info else None
        if hw_w and hw_h:
            chk.observed = hw_w
            chk.observed_fmt = f"{hw_w}\u00d7{hw_h} px"
            chk.source = "camera"
            matches = hw_w == cfg_h_px and hw_h == cfg_v_px
            chk.pct_diff = 0.0 if matches else max(_pct(cfg_h_px, hw_w), _pct(cfg_v_px, hw_h))
            chk.status = "ok" if matches else "warning"
        # Append sensor physical size as extra context
        if ps_um:
            sw = ps_um * cfg_h_px / 1000.0
            sh = ps_um * cfg_v_px / 1000.0
            chk.configured_fmt += f"  {sw:.1f}\u00d7{sh:.1f} mm"
        health.checks.append(chk)

    # ── FOV (config vs plate solve) ──
    if cfg_pixel_scale and cfg_h_px and cfg_v_px:
        cfg_fov_w = cfg_pixel_scale * cfg_h_px / 3600.0
        cfg_fov_h = cfg_pixel_scale * cfg_v_px / 3600.0
        chk = HardwareConfigCheck(
            name="fov",
            label="FOV",
            group="optics",
            configured=round(cfg_fov_w, 2),
            configured_fmt=f"{cfg_fov_w:.2f}\u00b0 \u00d7 {cfg_fov_h:.2f}\u00b0",
            unit="\u00b0",
        )
        if observed_fov_w is not None and observed_fov_h is not None:
            pct = max(_pct(cfg_fov_w, observed_fov_w), _pct(cfg_fov_h, observed_fov_h))
            chk.observed = round(observed_fov_w, 2)
            chk.observed_fmt = f"{observed_fov_w:.2f}\u00b0 \u00d7 {observed_fov_h:.2f}\u00b0"
            chk.source = "plate_solve"
            chk.pct_diff = round(pct, 1)
            chk.status = _status(pct)
        health.checks.append(chk)

    # ── Slew rate (config vs observed tracking) ──
    if cfg_slew:
        chk = HardwareConfigCheck(
            name="slew_rate",
            label="Slew Rate",
            group="telescope",
            configured=cfg_slew,
            configured_fmt=f"{cfg_slew} \u00b0/s",
            unit="\u00b0/s",
        )
        if observed_slew_rate is not None:
            pct = _pct(cfg_slew, observed_slew_rate)
            chk.observed = round(observed_slew_rate, 1)
            chk.observed_fmt = f"{observed_slew_rate:.1f} \u00b0/s"
            chk.source = "tracking"
            chk.pct_diff = round(pct, 1)
            chk.status = _status(pct)
        health.checks.append(chk)

    health.has_warnings = any(c.status == "warning" for c in health.checks)
    return health


def _safe_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: object) -> int | None:
    if val is None:
        return None
    try:
        i = int(val)  # type: ignore[arg-type]
        return i if i > 0 else None
    except (TypeError, ValueError):
        return None
