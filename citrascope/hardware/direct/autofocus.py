"""V-curve autofocus for the DirectHardwareAdapter.

Sweeps the focuser through a range of positions, measures HFR (Half-Flux
Radius) at each step via SEP, fits a hyperbola with outlier rejection,
optionally refines with a fine sweep, and moves to the optimum.

The defocus HFR curve is physically a hyperbola — defocus blur grows
linearly with distance from focus, so HFR(x) = sqrt(a*(x-c)^2 + b^2).
A parabolic fit is kept as fallback when the hyperbolic fit fails to
converge.

SEP (Source Extractor for Python) handles background subtraction, source
detection, and half-flux radius computation — it is already installed as a
dependency of pixelemon.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import sep

if TYPE_CHECKING:
    import threading

    from citrascope.hardware.devices.camera import AbstractCamera
    from citrascope.hardware.devices.focuser import AbstractFocuser

MIN_STARS_FOR_HFR = 5
MAX_STARS_FOR_HFR = 20
MAX_ELONGATION = 3.0
SETTLE_DELAY = 0.5
HFR_RMAX = 50.0


# ---------------------------------------------------------------------------
# Focus metric
# ---------------------------------------------------------------------------


def _ensure_2d(image: np.ndarray) -> np.ndarray:
    """Convert a multi-channel image to 2-D grayscale if needed."""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] == 3:
            weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
            return np.tensordot(image.astype(np.float64), weights, axes=([2], [0]))
        return np.mean(image, axis=2)
    return image


def _crop_center(image: np.ndarray, ratio: float) -> np.ndarray:
    """Return the central *ratio* fraction of *image*."""
    if ratio >= 1.0:
        return image
    h, w = image.shape
    ch, cw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return image[y0 : y0 + ch, x0 : x0 + cw]


def compute_hfr(image: np.ndarray, crop_ratio: float = 0.5) -> float | None:
    """Compute median Half-Flux Radius of detected stars using SEP.

    Returns None if fewer than MIN_STARS_FOR_HFR usable sources are found,
    signalling the caller to skip this position.
    """
    img = _crop_center(_ensure_2d(image), crop_ratio).astype(np.float64, copy=True)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    bkg = sep.Background(img)
    data_sub: np.ndarray = img - bkg

    objects = sep.extract(data_sub, thresh=3.0, err=bkg.globalrms, minarea=5)
    if len(objects) < MIN_STARS_FOR_HFR:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        elong = objects["a"] / np.maximum(objects["b"], 1e-6)
    star_mask = elong < MAX_ELONGATION
    objects = objects[star_mask]
    if len(objects) < MIN_STARS_FOR_HFR:
        return None

    rmax = np.full(len(objects), HFR_RMAX)
    hfr, flags = sep.flux_radius(data_sub, objects["x"], objects["y"], rmax, frac=0.5)
    good = (flags == 0) & np.isfinite(hfr) & (hfr > 0)
    hfr_good = hfr[good]
    if len(hfr_good) < MIN_STARS_FOR_HFR:
        return None

    flux_order = np.argsort(objects["flux"][good])[::-1]
    top = hfr_good[flux_order[:MAX_STARS_FOR_HFR]]
    return float(np.median(top))


# ---------------------------------------------------------------------------
# V-curve autofocus algorithm
# ---------------------------------------------------------------------------


def _wait_for_focuser(focuser: AbstractFocuser, timeout: float = 60.0) -> None:
    """Block until the focuser stops moving (or timeout)."""
    deadline = time.monotonic() + timeout
    while focuser.is_moving():
        if time.monotonic() > deadline:
            focuser.abort_move()
            raise RuntimeError("Focuser movement timeout during autofocus")
        time.sleep(0.15)


def _robust_polyfit(
    pos_arr: np.ndarray,
    val_arr: np.ndarray,
    sigma_clip: float = 3.0,
    max_iters: int = 3,
) -> tuple[float, float, float]:
    """Fit a quadratic with trimmed-then-sigma-clipped outlier rejection.

    High-leverage outliers (at the edges of the x-range) distort ordinary
    least-squares enough that residual-based rejection misses them. To
    counter this, the first pass trims the worst-fitting fraction of
    points, producing a clean initial fit. Subsequent passes apply
    MAD-based sigma clipping against that better fit.

    Returns (a, b, c) coefficients of a*x^2 + b*x + c.

    Raises ValueError if the fit fails or the parabola opens downward.
    """
    n = len(pos_arr)
    if n < 3:
        raise ValueError("Need at least 3 points for quadratic fit")

    # First pass: fit all, then trim the worst 25% (at least 1 point) to get
    # a cleaner initial fit that isn't pulled by high-leverage outliers.
    coeffs = np.polyfit(pos_arr, val_arr, 2)
    residuals = np.abs(val_arr - np.polyval(coeffs, pos_arr))
    n_keep = max(3, int(n * 0.75))
    keep_idx = np.argsort(residuals)[:n_keep]
    mask = np.zeros(n, dtype=bool)
    mask[keep_idx] = True

    # Subsequent passes: refit on kept points, then MAD-based sigma clip.
    for _ in range(max_iters):
        p = pos_arr[mask]
        v = val_arr[mask]
        if len(p) < 3:
            raise ValueError("Too few points remaining after outlier rejection")

        coeffs = np.polyfit(p, v, 2)

        residuals = val_arr - np.polyval(coeffs, pos_arr)
        med_res = float(np.median(np.abs(residuals[mask])))
        mad_sigma = 1.4826 * med_res
        if mad_sigma < 1e-10:
            break

        new_mask = np.abs(residuals) < sigma_clip * mad_sigma
        if np.sum(new_mask) < 3:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    a, b, c = np.polyfit(pos_arr[mask], val_arr[mask], 2)
    if a <= 0:
        raise ValueError("Parabola opens downward — no valid minimum")
    return float(a), float(b), float(c)


def _hyperbolic_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """HFR(x) = sqrt(a * (x - c)^2 + b^2).

    Physical model of defocus: blur radius grows linearly with focuser
    distance from optimal, producing a V-shaped (hyperbolic) HFR curve.
    """
    return np.sqrt(a * (x - c) ** 2 + b**2)


def _hyperbolic_fit(
    pos_arr: np.ndarray,
    val_arr: np.ndarray,
    sigma_clip: float = 3.0,
    max_iters: int = 3,
) -> tuple[float, float, float]:
    """Fit a hyperbolic V-curve with MAD-based outlier rejection.

    Model: HFR(x) = sqrt(a * (x - c)^2 + b^2)
      - c: optimal focuser position (the vertex)
      - b: minimum achievable HFR (best focus)
      - a: slope parameter (defocus rate)

    Returns (c, b, a) — the optimal position, minimum HFR, and slope.

    Raises ValueError if the fit fails. Caller should fall back to
    _robust_polyfit (parabolic) in that case.
    """
    from scipy.optimize import curve_fit

    n = len(pos_arr)
    if n < 3:
        raise ValueError("Need at least 3 points for hyperbolic fit")

    best_idx = int(np.argmin(val_arr))
    c0 = float(pos_arr[best_idx])
    b0 = max(float(val_arr[best_idx]), 0.1)
    pos_range = float(pos_arr.max() - pos_arr.min()) or 1.0
    hfr_range = float(val_arr.max() - val_arr.min()) or 1.0
    a0 = max((hfr_range / (pos_range / 2)) ** 2, 1e-12)

    margin = pos_range * 0.5
    bounds_lo = [1e-15, 0.01, float(pos_arr.min()) - margin]
    bounds_hi = [np.inf, float(val_arr.max()) * 2, float(pos_arr.max()) + margin]

    mask = np.ones(n, dtype=bool)

    for iteration in range(max_iters + 1):
        p = pos_arr[mask]
        v = val_arr[mask]
        if len(p) < 3:
            raise ValueError("Too few points remaining after outlier rejection")

        try:
            popt, _ = curve_fit(
                _hyperbolic_model,
                p,
                v,
                p0=[a0, b0, c0],
                bounds=(bounds_lo, bounds_hi),
                maxfev=5000,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Hyperbolic curve_fit failed: {exc}") from exc

        a0, b0, c0 = popt

        if iteration == max_iters:
            break

        residuals = val_arr - _hyperbolic_model(pos_arr, *popt)
        med_res = float(np.median(np.abs(residuals[mask])))
        mad_sigma = 1.4826 * med_res
        if mad_sigma < 1e-10:
            break

        new_mask = np.abs(residuals) < sigma_clip * mad_sigma
        if np.sum(new_mask) < 3:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    a_fit, b_fit, c_fit = a0, b0, c0
    if a_fit <= 0:
        raise ValueError("Hyperbolic fit: slope parameter a <= 0")
    return float(c_fit), float(b_fit), float(a_fit)


def _is_monotonic(pos_arr: np.ndarray, val_arr: np.ndarray, tolerance: float = 0.8) -> bool:
    """Check if HFR values are mostly monotonic (no clear V shape).

    Returns True if more than *tolerance* fraction of the position-sorted
    differences share a single sign — i.e. the data is a slope, not a V.
    A true V-curve has roughly equal positive and negative diffs.
    """
    order = np.argsort(pos_arr)
    diffs = np.diff(val_arr[order])
    if len(diffs) < 2:
        return False
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    dominant = max(n_pos, n_neg)
    return dominant / len(diffs) >= tolerance


def _fit_measurements(
    pos_arr: np.ndarray,
    val_arr: np.ndarray,
    log: logging.Logger,
) -> float | None:
    """Fit V-curve data and return the optimal position, or None on failure.

    Tries hyperbolic fit first (correct physics), falls back to parabolic.
    """
    # Hyperbolic fit (preferred — matches the physics of defocus)
    try:
        c, b_min, a_slope = _hyperbolic_fit(pos_arr, val_arr)
        log.info(f"Hyperbolic fit: optimum at {c:.0f} (min HFR={b_min:.3f}, slope={a_slope:.2e})")
        return c
    except (ValueError, RuntimeError) as exc:
        log.debug(f"Hyperbolic fit failed ({exc}), trying parabolic fallback")

    # Parabolic fallback
    try:
        a, b, _c = _robust_polyfit(pos_arr, val_arr)
        vertex = -b / (2 * a)
        log.info(f"Parabolic fit: optimum at {vertex:.0f} (a={a:.2e}, b={b:.2e})")
        return vertex
    except (ValueError, np.linalg.LinAlgError) as exc:
        log.warning(f"Parabolic fit also failed: {exc}")
        return None


def _sweep_positions(
    camera: AbstractCamera,
    focuser: AbstractFocuser,
    positions: list[int],
    *,
    exposure_time: float,
    crop_ratio: float,
    step_size: int,
    label: str,
    log: logging.Logger,
    report: Callable[[str], None],
    cancel_event: threading.Event | None,
    on_point: Callable[[int, float], None] | None,
) -> list[tuple[int, float]]:
    """Sweep through a list of focuser positions and measure HFR at each.

    Handles backlash compensation, settling, cancellation, and progress
    reporting.  Returns the list of (position, hfr) measurements.
    """
    overshoot = max(0, positions[0] - step_size)
    report(f"{label}: backlash overshoot to {overshoot}")
    if focuser.move_absolute(overshoot):
        _wait_for_focuser(focuser)
    else:
        log.warning(f"Failed to move to overshoot {overshoot}, proceeding anyway")

    total = len(positions)
    measurements: list[tuple[int, float]] = []

    for idx, pos in enumerate(positions, 1):
        if cancel_event and cancel_event.is_set():
            log.info("Autofocus cancelled by user")
            raise RuntimeError("Autofocus cancelled")

        report(f"{label} {idx}/{total}: moving to {pos}")

        if not focuser.move_absolute(pos):
            log.warning(f"Failed to move focuser to {pos}, skipping")
            continue
        _wait_for_focuser(focuser)
        time.sleep(SETTLE_DELAY)

        report(f"{label} {idx}/{total}: exposing {exposure_time:.1f}s")

        try:
            raw = camera.capture_array(
                duration=exposure_time,
                binning=camera.get_default_binning(),
            )
            image_data = _ensure_2d(raw).astype(np.float64)
        except Exception as e:
            log.warning(f"Exposure failed at position {pos}: {e}")
            continue

        hfr = compute_hfr(image_data, crop_ratio)
        if hfr is None:
            log.warning(f"HFR detection failed at position {pos} (too few stars), skipping")
            continue

        measurements.append((pos, hfr))
        if on_point:
            on_point(pos, hfr)
        report(f"{label} {idx}/{total}: pos={pos} HFR={hfr:.3f}")
        log.info(f"Autofocus point: position={pos}, HFR={hfr:.3f}")

    return measurements


def run_autofocus(
    camera: AbstractCamera,
    focuser: AbstractFocuser,
    *,
    step_size: int = 500,
    num_steps: int = 5,
    fine_steps: int = 3,
    exposure_time: float = 3.0,
    crop_ratio: float = 0.5,
    on_progress: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
    cancel_event: threading.Event | None = None,
    on_point: Callable[[int, float], None] | None = None,
) -> int:
    """Run two-pass V-curve autofocus and return the best focuser position.

    Pass 1 (coarse): sweeps (2 * num_steps + 1) positions at step_size,
    fits a hyperbolic curve to estimate the optimum.

    Pass 2 (fine): sweeps (2 * fine_steps + 1) positions at step_size // 4
    centered on the coarse estimate, then fits the combined measurements
    for sub-step-size precision.  Set fine_steps=0 to disable.

    Args:
        camera: Connected camera device.
        focuser: Connected focuser device.
        step_size: Focuser steps between coarse samples.
        num_steps: Number of coarse samples on each side of centre.
        fine_steps: Number of fine samples per side (0 disables refinement).
        exposure_time: Seconds per sample exposure.
        crop_ratio: Fraction of image centre to analyse (0 < x <= 1).
        on_progress: Optional callback for progress strings.
        logger: Optional logger.
        cancel_event: If set, abort the sweep at the next step boundary.
        on_point: Optional callback(position, hfr) fired after each sample.

    Returns:
        Optimal focuser position (integer steps).

    Raises:
        RuntimeError: If autofocus cannot determine a valid position
            (including cancellation).
    """
    log = logger or logging.getLogger(__name__)
    report = on_progress or (lambda _msg: None)

    center = focuser.get_position()
    if center is None:
        raise RuntimeError("Cannot read current focuser position")
    max_pos = focuser.get_max_position() or 200_000

    # --- Coarse sweep ---
    positions = [center + (i - num_steps) * step_size for i in range(2 * num_steps + 1)]
    positions = [max(0, min(p, max_pos)) for p in positions]
    positions = list(dict.fromkeys(positions))

    log.info(f"Coarse sweep: {len(positions)} positions from {positions[0]} to {positions[-1]} (step {step_size})")

    measurements = _sweep_positions(
        camera,
        focuser,
        positions,
        exposure_time=exposure_time,
        crop_ratio=crop_ratio,
        step_size=step_size,
        label="Coarse",
        log=log,
        report=report,
        cancel_event=cancel_event,
        on_point=on_point,
    )

    if len(measurements) < 3:
        raise RuntimeError(
            f"Too few valid HFR measurements ({len(measurements)}/{len(positions)}) for curve fit. "
            "Check that the field has enough stars and the exposure time is sufficient."
        )

    pos_arr = np.array([m[0] for m in measurements], dtype=np.float64)
    val_arr = np.array([m[1] for m in measurements], dtype=np.float64)

    # --- Monotonic slope guard ---
    if _is_monotonic(pos_arr, val_arr):
        best_idx = int(np.argmin(val_arr))
        best_pos = measurements[best_idx][0]
        best_hfr = measurements[best_idx][1]
        msg = (
            "Coarse sweep appears one-sided (no V-curve detected). "
            "Focus may be far from optimal — consider adjusting manually and re-running."
        )
        log.warning(msg)
        report(msg)
        report(f"Moving to best measured position: {best_pos}")
        if not focuser.move_absolute(best_pos):
            raise RuntimeError(f"Failed to move focuser to position {best_pos}")
        _wait_for_focuser(focuser)
        report(f"Autofocus complete (one-sided): position {best_pos}, HFR {best_hfr:.1f}")
        return best_pos

    coarse_vertex = _fit_measurements(pos_arr, val_arr, log)

    # --- Fine sweep (refinement pass) ---
    if coarse_vertex is not None and fine_steps > 0:
        margin = step_size
        if positions[0] - margin <= coarse_vertex <= positions[-1] + margin:
            fine_step = max(step_size // 4, 1)
            fine_center = round(coarse_vertex)
            fine_positions = [fine_center + (i - fine_steps) * fine_step for i in range(2 * fine_steps + 1)]
            fine_positions = [max(0, min(p, max_pos)) for p in fine_positions]
            fine_positions = list(dict.fromkeys(fine_positions))

            log.info(f"Fine sweep: {len(fine_positions)} positions around {fine_center} (step {fine_step})")

            fine_measurements = _sweep_positions(
                camera,
                focuser,
                fine_positions,
                exposure_time=exposure_time,
                crop_ratio=crop_ratio,
                step_size=fine_step,
                label="Fine",
                log=log,
                report=report,
                cancel_event=cancel_event,
                on_point=on_point,
            )

            if fine_measurements:
                all_measurements = measurements + fine_measurements
                all_pos = np.array([m[0] for m in all_measurements], dtype=np.float64)
                all_val = np.array([m[1] for m in all_measurements], dtype=np.float64)

                refined_vertex = _fit_measurements(all_pos, all_val, log)
                if refined_vertex is not None:
                    coarse_vertex = refined_vertex
                    log.info(f"Refined optimum: {coarse_vertex:.0f}")
        else:
            log.warning(f"Coarse vertex {coarse_vertex:.0f} outside sampled range, skipping refinement")

    # --- Determine final position ---
    best_pos: int
    if coarse_vertex is not None:
        best_pos = max(0, min(round(coarse_vertex), max_pos))
    else:
        best_idx = int(np.argmin(val_arr))
        best_pos = measurements[best_idx][0]
        log.info(f"Fallback: best measured position {best_pos} (HFR={measurements[best_idx][1]:.3f})")

    # Move to best position
    report(f"Moving to optimal position: {best_pos}")
    if not focuser.move_absolute(best_pos):
        raise RuntimeError(f"Failed to move focuser to optimal position {best_pos}")
    _wait_for_focuser(focuser)

    # Verification exposure
    try:
        time.sleep(SETTLE_DELAY)
        verify_raw = camera.capture_array(
            duration=exposure_time,
            binning=camera.get_default_binning(),
        )
        verify_data = _ensure_2d(verify_raw).astype(np.float64)
        verify_hfr = compute_hfr(verify_data, crop_ratio)
        if verify_hfr is not None:
            log.info(f"Verification: position={best_pos}, HFR={verify_hfr:.3f}")
    except Exception as e:
        log.debug(f"Verification exposure failed (non-critical): {e}")

    report(f"Autofocus complete: position {best_pos}")
    return best_pos
