"""V-curve autofocus for the DirectHardwareAdapter.

Sweeps the focuser through a range of positions, measures HFR (Half-Flux
Radius) at each step via SEP, fits a parabola with outlier rejection, and
moves to the optimum.

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


def run_autofocus(
    camera: AbstractCamera,
    focuser: AbstractFocuser,
    *,
    step_size: int = 500,
    num_steps: int = 5,
    exposure_time: float = 3.0,
    crop_ratio: float = 0.5,
    on_progress: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """Run V-curve autofocus and return the best focuser position.

    Sweeps the focuser through (2 * num_steps + 1) positions centred on the
    current position, measures HFR at each via SEP, fits a parabola with
    outlier rejection, and moves to the computed optimum.

    Args:
        camera: Connected camera device.
        focuser: Connected focuser device.
        step_size: Focuser steps between samples.
        num_steps: Number of samples on each side of centre.
        exposure_time: Seconds per sample exposure.
        crop_ratio: Fraction of image centre to analyse (0 < x <= 1).
        on_progress: Optional callback for progress strings.
        logger: Optional logger.
        cancel_event: If set, abort the sweep at the next step boundary.

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

    positions = [center + (i - num_steps) * step_size for i in range(2 * num_steps + 1)]
    positions = [max(0, min(p, max_pos)) for p in positions]
    positions = list(dict.fromkeys(positions))

    total = len(positions)
    log.info(f"Autofocus sweep: {total} positions from {positions[0]} to {positions[-1]} (step {step_size})")

    # --- Backlash-aware initial positioning ---
    # Overshoot past the first position, then approach from below so all
    # subsequent measurement moves are in the same (upward) direction.
    overshoot = max(0, positions[0] - step_size)
    report(f"Moving to overshoot position {overshoot} for backlash compensation")
    if focuser.move_absolute(overshoot):
        _wait_for_focuser(focuser)
    else:
        log.warning(f"Failed to move to overshoot position {overshoot}, proceeding anyway")

    measurements: list[tuple[int, float]] = []

    for idx, pos in enumerate(positions, 1):
        if cancel_event and cancel_event.is_set():
            log.info("Autofocus cancelled by user")
            raise RuntimeError("Autofocus cancelled")

        report(f"Step {idx}/{total}: moving to {pos}")

        if not focuser.move_absolute(pos):
            log.warning(f"Failed to move focuser to {pos}, skipping")
            continue
        _wait_for_focuser(focuser)

        time.sleep(SETTLE_DELAY)

        report(f"Step {idx}/{total}: exposing {exposure_time:.1f}s")

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
        report(f"Step {idx}/{total}: pos={pos} HFR={hfr:.3f}")
        log.info(f"Autofocus point: position={pos}, HFR={hfr:.3f}")

    if len(measurements) < 3:
        raise RuntimeError(
            f"Too few valid HFR measurements ({len(measurements)}/{total}) for curve fit. "
            "Check that the field has enough stars and the exposure time is sufficient."
        )

    # --- Curve fitting with outlier rejection ---
    pos_arr = np.array([m[0] for m in measurements], dtype=np.float64)
    val_arr = np.array([m[1] for m in measurements], dtype=np.float64)

    best_pos: int
    try:
        a, b, _c = _robust_polyfit(pos_arr, val_arr)
        vertex = -b / (2 * a)

        margin = step_size
        if vertex < positions[0] - margin or vertex > positions[-1] + margin:
            raise ValueError(f"Vertex {vertex:.0f} outside sampled range [{positions[0]}, {positions[-1]}]")

        best_pos = round(vertex)
        best_pos = max(0, min(best_pos, max_pos))
        log.info(f"Parabolic fit: optimum at position {best_pos} (coefficients a={a:.2e}, b={b:.2e})")

    except (ValueError, np.linalg.LinAlgError) as e:
        log.warning(f"Curve fit failed ({e}), using best measured position")
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
