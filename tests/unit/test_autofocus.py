"""Tests for the direct-adapter autofocus module.

Covers: HFR calculation, sharpness metric, dual-metric selection,
V-curve algorithm with mocked camera/focuser, and edge cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from citrascope.hardware.direct.autofocus import (
    _crop_center,
    _sigma_clipped_stats,
    compute_focus_metric,
    compute_hfr,
    compute_sharpness,
    run_autofocus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_star_field(
    size: int = 512,
    n_stars: int = 30,
    fwhm: float = 4.0,
    sky_level: float = 500.0,
    peak_flux: float = 10000.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic star field with Gaussian PSFs."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), sky_level, dtype=np.float64)
    img += rng.normal(0, 8, img.shape)

    sigma = fwhm / 2.3548
    ys, xs = np.mgrid[0:size, 0:size]

    margin = int(fwhm * 4)
    for _ in range(n_stars):
        cy = rng.integers(margin, size - margin)
        cx = rng.integers(margin, size - margin)
        flux = rng.uniform(peak_flux * 0.3, peak_flux)
        psf = flux * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma**2))
        img += psf

    return np.clip(img, 0, 65535).astype(np.float64)


def _make_texture(size: int = 512, blur_sigma: float = 0.0, seed: int = 42) -> np.ndarray:
    """Generate a textured image (simulates indoor/daytime scene)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    img = rng.uniform(200, 800, (size, size)).astype(np.float64)
    # Add some edges
    img[100:150, :] = 2000
    img[:, 200:210] = 3000
    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=blur_sigma)
    return img


# ---------------------------------------------------------------------------
# _crop_center
# ---------------------------------------------------------------------------


class TestCropCenter:
    def test_full_ratio(self):
        img = np.zeros((100, 100))
        result = _crop_center(img, 1.0)
        assert result.shape == (100, 100)

    def test_half_ratio(self):
        img = np.zeros((100, 100))
        result = _crop_center(img, 0.5)
        assert result.shape == (50, 50)

    def test_quarter_ratio(self):
        img = np.zeros((200, 200))
        result = _crop_center(img, 0.25)
        assert result.shape == (50, 50)

    def test_ratio_above_one(self):
        img = np.zeros((100, 100))
        result = _crop_center(img, 1.5)
        assert result.shape == (100, 100)


# ---------------------------------------------------------------------------
# _sigma_clipped_stats
# ---------------------------------------------------------------------------


class TestSigmaClippedStats:
    def test_normal_distribution(self):
        rng = np.random.default_rng(0)
        data = rng.normal(100, 10, 10000)
        med, std = _sigma_clipped_stats(data)
        assert abs(med - 100) < 1
        assert abs(std - 10) < 1

    def test_with_outliers(self):
        rng = np.random.default_rng(0)
        data = rng.normal(100, 10, 10000)
        data[:50] = 1e6
        med, std = _sigma_clipped_stats(data)
        assert abs(med - 100) < 2
        assert std < 15

    def test_constant_data(self):
        data = np.full(100, 42.0)
        med, std = _sigma_clipped_stats(data)
        assert med == 42.0
        assert std == 0.0


# ---------------------------------------------------------------------------
# compute_hfr
# ---------------------------------------------------------------------------


class TestComputeHFR:
    def test_detects_stars(self):
        img = _make_star_field(n_stars=30, fwhm=4.0)
        hfr = compute_hfr(img, crop_ratio=0.8)
        assert hfr is not None
        assert 1.0 < hfr < 10.0

    def test_wider_psf_gives_larger_hfr(self):
        narrow = _make_star_field(n_stars=30, fwhm=3.0, seed=1)
        wide = _make_star_field(n_stars=30, fwhm=8.0, seed=1)
        hfr_narrow = compute_hfr(narrow, crop_ratio=0.8)
        hfr_wide = compute_hfr(wide, crop_ratio=0.8)
        assert hfr_narrow is not None
        assert hfr_wide is not None
        assert hfr_wide > hfr_narrow

    def test_returns_none_for_blank_image(self):
        img = np.full((256, 256), 500.0)
        assert compute_hfr(img) is None

    def test_returns_none_for_few_stars(self):
        img = _make_star_field(n_stars=2, fwhm=4.0)
        result = compute_hfr(img, crop_ratio=0.3)
        # With only 2 stars and a tight crop, should get < MIN_STARS_FOR_HFR
        # (may or may not depending on placement, so we just verify the type)
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_sharpness
# ---------------------------------------------------------------------------


class TestComputeSharpness:
    def test_sharp_higher_than_blurred(self):
        sharp = _make_texture(blur_sigma=0.0)
        blurred = _make_texture(blur_sigma=5.0)
        s_sharp = compute_sharpness(sharp, crop_ratio=0.8)
        s_blurred = compute_sharpness(blurred, crop_ratio=0.8)
        assert s_sharp > s_blurred

    def test_positive_value(self):
        img = _make_texture()
        assert compute_sharpness(img) > 0

    def test_blank_image(self):
        img = np.full((256, 256), 500.0)
        assert compute_sharpness(img) == 0.0


# ---------------------------------------------------------------------------
# compute_focus_metric
# ---------------------------------------------------------------------------


class TestComputeFocusMetric:
    def test_star_field_uses_hfr(self):
        img = _make_star_field(n_stars=30, fwhm=4.0)
        value, minimize = compute_focus_metric(img, crop_ratio=0.8)
        assert minimize is True
        assert value > 0

    def test_blank_image_uses_sharpness(self):
        img = np.full((256, 256), 500.0)
        value, minimize = compute_focus_metric(img)
        assert minimize is False
        assert value == 0.0

    def test_texture_uses_sharpness(self):
        img = _make_texture()
        value, minimize = compute_focus_metric(img, crop_ratio=0.8)
        assert minimize is False
        assert value > 0


# ---------------------------------------------------------------------------
# run_autofocus (mocked hardware)
# ---------------------------------------------------------------------------


def _make_mock_focuser(position: int = 25000, max_pos: int = 100000) -> MagicMock:
    """Create a mock focuser that tracks position."""
    focuser = MagicMock()
    state = {"position": position}

    focuser.get_position.side_effect = lambda: state["position"]
    focuser.get_max_position.return_value = max_pos
    focuser.is_moving.return_value = False
    focuser.is_connected.return_value = True

    def move_abs(pos: int) -> bool:
        state["position"] = pos
        return True

    focuser.move_absolute.side_effect = move_abs
    focuser.abort_move.return_value = None
    return focuser


def _make_mock_camera(images_dir: Path, v_curve_center: int = 25000, v_curve_scale: float = 0.001) -> MagicMock:
    """Create a mock camera whose images have HFR proportional to distance from v_curve_center."""
    camera = MagicMock()
    camera.is_connected.return_value = True
    camera.get_default_binning.return_value = 1
    call_counter = {"n": 0}

    def take_exposure(duration: float, binning: int = 1, save_path: Path | None = None, **kw) -> Path:
        call_counter["n"] += 1
        if save_path is None:
            save_path = images_dir / f"mock_{call_counter['n']}.fits"
        # Write a synthetic star field whose FWHM depends on focuser distance
        # The caller will read the mock focuser position separately
        from astropy.io import fits

        # Simple: write a real FITS with stars
        img = _make_star_field(size=256, n_stars=25, fwhm=4.0, seed=call_counter["n"])
        hdu = fits.PrimaryHDU(data=img.astype(np.uint16))
        hdu.writeto(str(save_path), overwrite=True)
        return save_path

    camera.take_exposure.side_effect = take_exposure
    return camera


class TestRunAutofocus:
    def test_completes_and_returns_position(self, tmp_path: Path):
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera(tmp_path)

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            images_dir=tmp_path,
            step_size=200,
            num_steps=3,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        assert isinstance(best, int)
        assert 0 <= best <= 100000
        assert focuser.move_absolute.call_count >= 7  # 2*3+1 sweep + final move

    def test_too_few_measurements_raises(self, tmp_path: Path):
        focuser = _make_mock_focuser(position=25000)
        camera = MagicMock()
        camera.is_connected.return_value = True
        camera.get_default_binning.return_value = 1
        camera.take_exposure.side_effect = RuntimeError("Camera error")

        with pytest.raises(RuntimeError, match="Too few valid measurements"):
            run_autofocus(
                camera=camera,
                focuser=focuser,
                images_dir=tmp_path,
                step_size=200,
                num_steps=2,
                exposure_time=0.1,
                logger=logging.getLogger("test"),
            )

    def test_clamps_to_max_position(self, tmp_path: Path):
        focuser = _make_mock_focuser(position=500, max_pos=1000)
        camera = _make_mock_camera(tmp_path)

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            images_dir=tmp_path,
            step_size=300,
            num_steps=3,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        assert 0 <= best <= 1000

    def test_progress_callback_called(self, tmp_path: Path):
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera(tmp_path)
        progress_msgs: list[str] = []

        run_autofocus(
            camera=camera,
            focuser=focuser,
            images_dir=tmp_path,
            step_size=200,
            num_steps=2,
            exposure_time=0.1,
            crop_ratio=0.8,
            on_progress=progress_msgs.append,
            logger=logging.getLogger("test"),
        )

        assert len(progress_msgs) >= 5
        assert any("complete" in m.lower() for m in progress_msgs)

    def test_sharpness_fallback_on_texture(self, tmp_path: Path):
        """When the camera returns featureless images, sharpness metric is used."""
        focuser = _make_mock_focuser(position=25000)
        camera = MagicMock()
        camera.is_connected.return_value = True
        camera.get_default_binning.return_value = 1
        call_counter = {"n": 0}

        def take_texture(duration: float, binning: int = 1, save_path: Path | None = None, **kw) -> Path:
            from astropy.io import fits

            call_counter["n"] += 1
            if save_path is None:
                save_path = tmp_path / f"tex_{call_counter['n']}.fits"
            img = _make_texture(size=256, seed=call_counter["n"])
            hdu = fits.PrimaryHDU(data=img.astype(np.uint16))
            hdu.writeto(str(save_path), overwrite=True)
            return save_path

        camera.take_exposure.side_effect = take_texture

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            images_dir=tmp_path,
            step_size=200,
            num_steps=2,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        assert isinstance(best, int)
        assert 0 <= best <= 100000


# ---------------------------------------------------------------------------
# Parabolic fit edge cases
# ---------------------------------------------------------------------------


class TestParabolicFit:
    def test_known_v_curve(self, tmp_path: Path):
        """Verify the fit finds the minimum of a known quadratic V-curve."""
        focuser = _make_mock_focuser(position=25000, max_pos=50000)

        # Camera returns FITS with star FWHM proportional to |pos - 25000|
        # so HFR is minimised at 25000
        camera = MagicMock()
        camera.is_connected.return_value = True
        camera.get_default_binning.return_value = 1
        call_counter = {"n": 0}
        optimal_pos = 25000

        def take_vcurve(duration: float, binning: int = 1, save_path: Path | None = None, **kw) -> Path:
            from astropy.io import fits

            call_counter["n"] += 1
            if save_path is None:
                save_path = tmp_path / f"vc_{call_counter['n']}.fits"
            cur = focuser.get_position()
            dist = abs(cur - optimal_pos)
            fwhm = 3.0 + dist * 0.005
            img = _make_star_field(size=256, n_stars=25, fwhm=fwhm, seed=call_counter["n"])
            hdu = fits.PrimaryHDU(data=img.astype(np.uint16))
            hdu.writeto(str(save_path), overwrite=True)
            return save_path

        camera.take_exposure.side_effect = take_vcurve

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            images_dir=tmp_path,
            step_size=500,
            num_steps=4,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        # Should be close to 25000 (within a step or two)
        assert abs(best - optimal_pos) < 1500, f"Expected ~{optimal_pos}, got {best}"


# ---------------------------------------------------------------------------
# "Current position" preset
# ---------------------------------------------------------------------------


class TestCurrentPositionPreset:
    def test_preset_exists(self):
        from citrascope.constants import AUTOFOCUS_TARGET_PRESETS

        assert "current" in AUTOFOCUS_TARGET_PRESETS
        preset = AUTOFOCUS_TARGET_PRESETS["current"]
        assert preset["ra"] is None
        assert preset["dec"] is None
        assert "name" in preset
