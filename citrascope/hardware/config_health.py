"""Telescope configuration health assessment.

Compares the server-side telescope record against values observed from
hardware (camera sensor info), astrometry (plate solving), and mount
dynamics (tracking).  Returns structured results so the UI can render
them without doing any maths.

Each check carries a ``group`` field ("optics" or "telescope") so the
UI knows which card to render it on without hardcoded name matching.
Each check also carries a ``short_label`` for the compact one-line
optics strip and a ``label`` for the long-form warning banner.

The telescope record stores **unbinned** sensor dimensions and pixel
size; ``camera_info`` also reports unbinned sensor specs (Moravian
``GIP_CHIP_*``, NINA/ASCOM ``XSize/YSize/PixelSizeX``).  Binning is
adapter-level imaging intent — when the camera saves a frame it is
already binned, which means the plate-solver's observed pixel scale
is per **binned** pixel.  ``assess_config_health`` therefore scales
the configured pixel scale by binning before comparing, so a 2x2
bin does not trip a 100% mismatch warning.

Usage from the status loop::

    from citrascope.hardware.config_health import assess_config_health

    bx, by = adapter.get_current_binning()
    health = assess_config_health(
        telescope_record=daemon.telescope_record,
        camera_info=camera_info_dict,
        binning=(bx, by),
        observed_pixel_scale=adapter.observed_pixel_scale_arcsec,
        observed_fov_w=adapter.observed_fov_w_deg,
        observed_fov_h=adapter.observed_fov_h_deg,
        observed_slew_rate=adapter.observed_slew_rate_deg_per_s,
        slew_rate_samples=adapter.slew_rate_tracker.count,
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
    short_label: str = ""
    group: str = "optics"
    configured: float | None = None
    configured_fmt: str = ""
    observed: float | None = None
    observed_fmt: str = ""
    unit: str = ""
    source: str = ""
    source_samples: int | None = None
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
                    "short_label": c.short_label,
                    "group": c.group,
                    "configured": c.configured,
                    "configured_fmt": c.configured_fmt,
                    "observed": c.observed,
                    "observed_fmt": c.observed_fmt,
                    "unit": c.unit,
                    "source": c.source,
                    "source_samples": c.source_samples,
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
    binning: tuple[int, int] = (1, 1),
    observed_pixel_scale: float | None = None,
    observed_fov_w: float | None = None,
    observed_fov_h: float | None = None,
    observed_slew_rate: float | None = None,
    slew_rate_samples: int | None = None,
) -> ConfigHealth:
    """Build a complete telescope configuration health assessment.

    All comparison logic lives here — the UI just renders the result.

    Args:
        telescope_record: Citra API telescope record with unbinned sensor specs.
        camera_info: ``AbstractAstroHardwareAdapter.get_camera_info()`` output;
            ``width``/``height``/``pixel_size_um`` are unbinned sensor values.
        binning: ``(bx, by)`` from ``adapter.get_current_binning()``.  Used to
            scale the configured pixel scale so the plate-solver observed value
            (per binned pixel) compares apples-to-apples.
        observed_pixel_scale: arcsec per saved (binned) pixel from plate solve.
        observed_fov_w / observed_fov_h: angular FOV from plate solve.  FOV is
            invariant to binning so no scaling is required.
        observed_slew_rate: rolling-mean mount slew rate (deg/s).
        slew_rate_samples: number of samples in the rolling mean; rendered as
            ``(n=N)`` next to the slew rate readout.
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

    bx, by = binning if binning else (1, 1)
    bx = max(1, int(bx))
    by = max(1, int(by))
    bin_suffix = ""
    if bx != 1 or by != 1:
        bin_suffix = f" @ {bx}\u00d7{by}" if bx != by else f" @ {bx}\u00d7{bx}"

    cfg_pixel_scale = ps_um / fl_mm * 206.265 if ps_um and fl_mm else None
    # Pixel scale per *saved* pixel after binning — what plate solver returns.
    cfg_pixel_scale_binned = cfg_pixel_scale * bx if cfg_pixel_scale else None

    # ── Focal length & ratio (config only, no hardware source) ──
    if fl_mm:
        ratio_str = f"  f/{f_ratio}" if f_ratio else ""
        health.checks.append(
            HardwareConfigCheck(
                name="focal_length",
                label="Focal Length",
                short_label="FL",
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
            short_label="PX",
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
    #
    # Plate solver operates on the saved (binned) frame, so its pixel scale
    # is arcsec per binned pixel.  Scale the configured value by ``bx`` so
    # 2x2 binning doesn't false-flag a 100% mismatch.  The ``@ NxN`` suffix
    # tells the operator that binning is in play.
    if cfg_pixel_scale_binned:
        chk = HardwareConfigCheck(
            name="pixel_scale",
            label="Pixel Scale",
            short_label="SCALE",
            group="optics",
            configured=round(cfg_pixel_scale_binned, 2),
            configured_fmt=f'{cfg_pixel_scale_binned:.2f}"/px{bin_suffix}',
            unit='"/px',
        )
        if observed_pixel_scale is not None:
            pct = _pct(cfg_pixel_scale_binned, observed_pixel_scale)
            chk.observed = round(observed_pixel_scale, 2)
            chk.observed_fmt = f'{observed_pixel_scale:.2f}"/px'
            chk.source = "plate_solve"
            chk.pct_diff = round(pct, 1)
            chk.status = _status(pct)
        health.checks.append(chk)

    # ── Sensor resolution (config vs camera hardware) ──
    #
    # Hardware comparison is always unbinned-vs-unbinned — both sides of
    # the pct_diff come from the raw chip dims.  The *display*, however,
    # shows the **effective binned output size** with an ``@ NxN`` suffix
    # when binning is active (mirroring pixel_scale's convention).  That
    # matches what an operator actually sees coming off the camera and
    # what they'll use for FOV / framing decisions.
    if cfg_h_px and cfg_v_px:
        # Use the sanitized bx/by/bin_suffix from the top of the function —
        # no re-unpacking of the raw ``binning`` parameter here (it may be
        # None), and no risk of drifting from the pixel_scale block's conventions.
        eff_h_cfg = cfg_h_px // bx
        eff_v_cfg = cfg_v_px // by
        chk = HardwareConfigCheck(
            name="sensor_resolution",
            label="Sensor",
            short_label="SENSOR",
            group="optics",
            configured=cfg_h_px,
            configured_fmt=f"{eff_h_cfg}\u00d7{eff_v_cfg} px{bin_suffix}",
            unit="px",
        )
        hw_w = _safe_int(camera_info.get("width")) if camera_info else None
        hw_h = _safe_int(camera_info.get("height")) if camera_info else None
        if hw_w and hw_h:
            eff_hw_w = hw_w // bx
            eff_hw_h = hw_h // by
            chk.observed = hw_w
            chk.observed_fmt = f"{eff_hw_w}\u00d7{eff_hw_h} px{bin_suffix}"
            chk.source = "camera"
            matches = hw_w == cfg_h_px and hw_h == cfg_v_px
            chk.pct_diff = 0.0 if matches else max(_pct(cfg_h_px, hw_w), _pct(cfg_v_px, hw_h))
            chk.status = "ok" if matches else "warning"
        health.checks.append(chk)

    # ── FOV (config vs plate solve) ──
    #
    # FOV is invariant to binning: ``naxis_binned * scale_binned`` equals
    # ``naxis_unbinned * scale_unbinned``.  No binning scaling required.
    if cfg_pixel_scale and cfg_h_px and cfg_v_px:
        cfg_fov_w = cfg_pixel_scale * cfg_h_px / 3600.0
        cfg_fov_h = cfg_pixel_scale * cfg_v_px / 3600.0
        chk = HardwareConfigCheck(
            name="fov",
            label="FOV",
            short_label="FOV",
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
            short_label="SLEW",
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
            chk.source_samples = slew_rate_samples
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
