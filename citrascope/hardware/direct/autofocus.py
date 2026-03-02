"""V-curve autofocus for the DirectHardwareAdapter.

Sweeps the focuser through a range of positions, measures a focus metric
at each step, fits a parabola, and moves to the optimum.

Two metrics are supported with automatic fallback:
  - HFR (Half-Flux Radius) for star fields — lower is better
  - Laplacian variance for daytime / featureless scenes — higher is better

Only numpy and scipy are required (both already project dependencies).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from citrascope.hardware.devices.camera import AbstractCamera
    from citrascope.hardware.devices.focuser import AbstractFocuser

MIN_STARS_FOR_HFR = 5
MAX_STARS_FOR_HFR = 20
DEFAULT_SIGMA_THRESHOLD = 5.0
MIN_SOURCE_PIXELS = 5
MAX_ELONGATION = 3.0


# ---------------------------------------------------------------------------
# Focus metrics
# ---------------------------------------------------------------------------


def _crop_center(image: np.ndarray, ratio: float) -> np.ndarray:
    """Return the central *ratio* fraction of *image*."""
    if ratio >= 1.0:
        return image
    h, w = image.shape
    ch, cw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return image[y0 : y0 + ch, x0 : x0 + cw]


def _sigma_clipped_stats(data: np.ndarray, sigma: float = 3.0, iters: int = 3) -> tuple[float, float]:
    """Return (median, std) after iterative sigma-clipping."""
    d = data.ravel().astype(np.float64)
    for _ in range(iters):
        med = float(np.median(d))
        std = float(np.std(d))
        if std == 0:
            break
        mask = np.abs(d - med) < sigma * std
        d = d[mask]
    return float(np.median(d)), float(np.std(d))


def compute_hfr(image: np.ndarray, crop_ratio: float = 0.5) -> float | None:
    """Compute median Half-Flux Radius of detected stars.

    Returns None if fewer than MIN_STARS_FOR_HFR sources are found,
    signalling the caller to fall back to a sharpness metric.
    """
    img = _crop_center(image, crop_ratio).astype(np.float64)

    med, std = _sigma_clipped_stats(img)
    if std == 0:
        return None

    threshold = med + DEFAULT_SIGMA_THRESHOLD * std
    binary = img > threshold
    label_result: tuple[np.ndarray, int] = ndimage.label(binary)  # type: ignore[assignment]
    labeled = label_result[0]
    n_objects = label_result[1]

    if n_objects < MIN_STARS_FOR_HFR:
        return None

    # Measure each source
    sources: list[tuple[float, float]] = []  # (total_flux, hfr)
    for label_id in range(1, n_objects + 1):
        slices = ndimage.find_objects(labeled == label_id)
        if not slices:
            continue
        sy, sx = slices[0]

        cutout_mask = labeled[sy, sx] == label_id
        cutout_data = img[sy, sx]

        n_pixels = int(np.sum(cutout_mask))
        if n_pixels < MIN_SOURCE_PIXELS:
            continue

        # Check elongation via bounding box aspect ratio
        h, w = cutout_mask.shape
        aspect = max(h, w) / max(min(h, w), 1)
        if aspect > MAX_ELONGATION:
            continue

        # Flux-weighted centroid within the cutout
        ys, xs = np.mgrid[0 : cutout_data.shape[0], 0 : cutout_data.shape[1]]
        flux = cutout_data * cutout_mask
        total = float(np.sum(flux))
        if total <= 0:
            continue

        cy = float(np.sum(ys * flux)) / total
        cx = float(np.sum(xs * flux)) / total

        # Radial distances from centroid
        r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        r_flat = r[cutout_mask]
        flux_flat = flux[cutout_mask]

        # Sort by radius and find the radius enclosing 50 % of flux
        order = np.argsort(r_flat)
        cumflux = np.cumsum(flux_flat[order])
        half_idx = np.searchsorted(cumflux, total * 0.5)
        if half_idx >= len(r_flat):
            half_idx = len(r_flat) - 1

        hfr = float(r_flat[order][half_idx])
        if hfr > 0:
            sources.append((total, hfr))

    if len(sources) < MIN_STARS_FOR_HFR:
        return None

    # Use median HFR of the brightest sources
    sources.sort(key=lambda s: s[0], reverse=True)
    top = sources[:MAX_STARS_FOR_HFR]
    return float(np.median([s[1] for s in top]))


def compute_sharpness(image: np.ndarray, crop_ratio: float = 0.5) -> float:
    """Laplacian-variance sharpness metric (higher = sharper)."""
    img = _crop_center(image, crop_ratio).astype(np.float64)
    lap = ndimage.laplace(img)
    return float(np.var(lap))


def compute_focus_metric(image: np.ndarray, crop_ratio: float = 0.5) -> tuple[float, bool]:
    """Compute the best available focus metric.

    Returns:
        (value, minimize) where *minimize* is True for HFR (lower=better)
        and False for sharpness (higher=better).
    """
    hfr = compute_hfr(image, crop_ratio)
    if hfr is not None:
        return hfr, True
    return compute_sharpness(image, crop_ratio), False


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


def _load_fits_data(path: Path) -> np.ndarray:
    """Read the primary HDU of a FITS file as a 2-D numpy array."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdu = hdul[0]
        assert isinstance(hdu, fits.PrimaryHDU)
        data = hdu.data
        assert data is not None
        return np.array(data, dtype=np.float64)


def run_autofocus(
    camera: AbstractCamera,
    focuser: AbstractFocuser,
    images_dir: Path,
    *,
    step_size: int = 500,
    num_steps: int = 5,
    exposure_time: float = 3.0,
    crop_ratio: float = 0.5,
    on_progress: Callable[[str], None] | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """Run V-curve autofocus and return the best focuser position.

    Sweeps the focuser through (2 * num_steps + 1) positions centred on the
    current position, measures a focus metric at each, fits a parabola, and
    moves to the computed optimum.

    Args:
        camera: Connected camera device.
        focuser: Connected focuser device.
        images_dir: Directory for temporary autofocus exposures.
        step_size: Focuser steps between samples.
        num_steps: Number of samples on each side of centre.
        exposure_time: Seconds per sample exposure.
        crop_ratio: Fraction of image centre to analyse (0 < x <= 1).
        on_progress: Optional callback for progress strings.
        logger: Optional logger.

    Returns:
        Optimal focuser position (integer steps).

    Raises:
        RuntimeError: If autofocus cannot determine a valid position.
    """
    log = logger or logging.getLogger(__name__)
    report = on_progress or (lambda _msg: None)

    # Determine sweep centre and range
    center = focuser.get_position()
    if center is None:
        raise RuntimeError("Cannot read current focuser position")
    max_pos = focuser.get_max_position() or 200_000

    positions = [center + (i - num_steps) * step_size for i in range(2 * num_steps + 1)]
    positions = [max(0, min(p, max_pos)) for p in positions]
    # Deduplicate (clamp may have collapsed extremes)
    positions = list(dict.fromkeys(positions))

    total = len(positions)
    log.info(f"Autofocus sweep: {total} positions from {positions[0]} to {positions[-1]} (step {step_size})")

    measurements: list[tuple[int, float]] = []
    metric_mode: bool | None = None  # True = minimize (HFR), False = maximize (sharpness)
    mode_name = "unknown"

    for idx, pos in enumerate(positions, 1):
        report(f"Step {idx}/{total}: moving to {pos}")

        if not focuser.move_absolute(pos):
            log.warning(f"Failed to move focuser to {pos}, skipping")
            continue
        _wait_for_focuser(focuser)

        # Take exposure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = images_dir / f"autofocus_{timestamp}_{idx:02d}.fits"
        report(f"Step {idx}/{total}: exposing {exposure_time:.1f}s")

        try:
            image_path = camera.take_exposure(
                duration=exposure_time,
                binning=camera.get_default_binning(),
                save_path=save_path,
            )
            image_data = _load_fits_data(Path(image_path))
        except Exception as e:
            log.warning(f"Exposure failed at position {pos}: {e}")
            continue

        # Compute metric; lock mode on first successful measurement
        if metric_mode is None:
            value, minimize = compute_focus_metric(image_data, crop_ratio)
            metric_mode = minimize
            mode_name = "HFR" if minimize else "sharpness"
            log.info(f"Autofocus: using {mode_name} metric")
        else:
            if metric_mode:
                hfr = compute_hfr(image_data, crop_ratio)
                value = hfr if hfr is not None else math.inf
            else:
                value = compute_sharpness(image_data, crop_ratio)

        measurements.append((pos, value))
        report(f"Step {idx}/{total}: pos={pos} {mode_name}={value:.3f}")
        log.info(f"Autofocus point: position={pos}, {mode_name}={value:.3f}")

        # Clean up temporary file
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError:
            pass

    if len(measurements) < 3:
        raise RuntimeError(f"Too few valid measurements ({len(measurements)}) for curve fit")

    # --- Curve fitting ---
    pos_arr = np.array([m[0] for m in measurements], dtype=np.float64)
    val_arr = np.array([m[1] for m in measurements], dtype=np.float64)

    # For sharpness (maximize), negate so we can always look for a minimum
    fit_vals = val_arr if metric_mode else -val_arr

    best_pos: int
    try:
        coeffs = np.polyfit(pos_arr, fit_vals, 2)
        a, b, _c = coeffs

        # Parabola must open upward (a > 0) for a valid minimum
        if a <= 0:
            raise ValueError("Parabola opens downward — no valid minimum")

        vertex = -b / (2 * a)

        # Vertex must be within (or near) the sampled range
        margin = step_size
        if vertex < positions[0] - margin or vertex > positions[-1] + margin:
            raise ValueError(f"Vertex {vertex:.0f} outside sampled range [{positions[0]}, {positions[-1]}]")

        best_pos = round(vertex)
        best_pos = max(0, min(best_pos, max_pos))
        log.info(f"Parabolic fit: optimum at position {best_pos} (coefficients a={a:.2e}, b={b:.2e})")

    except (ValueError, np.linalg.LinAlgError) as e:
        log.warning(f"Curve fit failed ({e}), using best measured position")
        if metric_mode:
            best_idx = int(np.argmin(val_arr))
        else:
            best_idx = int(np.argmax(val_arr))
        best_pos = measurements[best_idx][0]
        log.info(f"Fallback: best measured position {best_pos} ({mode_name}={measurements[best_idx][1]:.3f})")

    # Move to best position
    report(f"Moving to optimal position: {best_pos}")
    if not focuser.move_absolute(best_pos):
        raise RuntimeError(f"Failed to move focuser to optimal position {best_pos}")
    _wait_for_focuser(focuser)

    # Verification exposure
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        verify_path = images_dir / f"autofocus_verify_{timestamp}.fits"
        img_path = camera.take_exposure(
            duration=exposure_time,
            binning=camera.get_default_binning(),
            save_path=verify_path,
        )
        verify_data = _load_fits_data(Path(img_path))
        if metric_mode:
            verify_val = compute_hfr(verify_data, crop_ratio)
            if verify_val is not None:
                log.info(f"Verification: position={best_pos}, HFR={verify_val:.3f}")
        else:
            verify_val_s = compute_sharpness(verify_data, crop_ratio)
            log.info(f"Verification: position={best_pos}, sharpness={verify_val_s:.3f}")
        try:
            Path(img_path).unlink(missing_ok=True)
        except OSError:
            pass
    except Exception as e:
        log.debug(f"Verification exposure failed (non-critical): {e}")

    report(f"Autofocus complete: position {best_pos}")
    return best_pos
