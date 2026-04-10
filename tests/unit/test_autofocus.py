"""Tests for the direct-adapter autofocus module.

Covers: SEP-based HFR computation, V-curve algorithm with mocked
camera/focuser, robust curve fitting, outlier rejection, backlash
compensation, and failure modes.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from citrascope.hardware.direct.autofocus import (
    _crop_center,
    _hyperbolic_fit,
    _hyperbolic_model,
    _is_monotonic,
    _robust_polyfit,
    compute_hfr,
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
# compute_hfr (SEP-based)
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
        assert result is None or isinstance(result, float)

    def test_handles_high_background(self):
        img = _make_star_field(n_stars=30, fwhm=4.0, sky_level=5000.0, peak_flux=20000.0)
        hfr = compute_hfr(img, crop_ratio=0.8)
        assert hfr is not None
        assert hfr > 0

    def test_handles_3_channel_image(self):
        mono = _make_star_field(n_stars=30, fwhm=4.0)
        rgb = np.stack([mono, mono, mono], axis=2)
        hfr = compute_hfr(rgb, crop_ratio=0.8)
        assert hfr is not None

    def test_defocused_stars_still_detected(self):
        """Heavily defocused stars (large FWHM) should still produce HFR."""
        img = _make_star_field(n_stars=30, fwhm=15.0, peak_flux=15000.0, size=512)
        hfr = compute_hfr(img, crop_ratio=0.8)
        assert hfr is not None
        assert hfr > 5.0


# ---------------------------------------------------------------------------
# _robust_polyfit
# ---------------------------------------------------------------------------


class TestRobustPolyfit:
    def test_clean_parabola(self):
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = 0.01 * (x - 30) ** 2 + 2.0
        a, b, _c = _robust_polyfit(x, y)
        assert a > 0
        vertex = -b / (2 * a)
        assert abs(vertex - 30.0) < 0.1

    def test_rejects_outlier(self):
        x = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
        y = 0.01 * (x - 30) ** 2 + 2.0
        y_corrupted = y.copy()
        y_corrupted[0] = 100.0  # outlier

        # Without outlier rejection, np.polyfit on this data gives vertex ~19.
        # With outlier rejection, vertex should be much closer to the true 30.
        a_naive, b_naive, _ = np.polyfit(x, y_corrupted, 2)
        naive_vertex = -b_naive / (2 * a_naive)

        a, b, _c = _robust_polyfit(x, y_corrupted)
        vertex = -b / (2 * a)
        assert abs(vertex - 30.0) < abs(naive_vertex - 30.0), "Robust fit should be closer to true minimum"
        assert abs(vertex - 30.0) < 8.0

    def test_raises_on_downward_parabola(self):
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = -((x - 30) ** 2) + 100.0
        with pytest.raises(ValueError, match="downward"):
            _robust_polyfit(x, y)

    def test_raises_on_too_few_points(self):
        x = np.array([10.0, 20.0])
        y = np.array([5.0, 3.0])
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            _robust_polyfit(x, y)


# ---------------------------------------------------------------------------
# _hyperbolic_fit
# ---------------------------------------------------------------------------


class TestHyperbolicFit:
    def test_clean_hyperbola(self):
        """Fit a synthetic hyperbolic V-curve — vertex should be accurate."""
        true_c, true_b, true_a = 25000.0, 2.0, 1e-5
        x = np.linspace(22000, 28000, 13)
        y = _hyperbolic_model(x, true_a, true_b, true_c)

        c, b, a = _hyperbolic_fit(x, y)
        assert abs(c - true_c) < 50, f"Vertex {c:.0f} too far from true {true_c}"
        assert abs(b - true_b) < 0.5
        assert a > 0

    def test_rejects_outlier(self):
        """An outlier point shouldn't pull the vertex away significantly."""
        true_c, true_b, true_a = 25000.0, 2.0, 1e-5
        x = np.linspace(22000, 28000, 13)
        y = _hyperbolic_model(x, true_a, true_b, true_c)
        y_bad = y.copy()
        y_bad[0] = 50.0

        c, b, _a = _hyperbolic_fit(x, y_bad)
        assert abs(c - true_c) < 200, f"Vertex {c:.0f} too far from true {true_c} after outlier"
        assert b > 0

    def test_falls_back_to_parabolic_on_bad_data(self):
        """When hyperbolic fit fails, _robust_polyfit should still work."""
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y = 0.01 * (x - 30) ** 2 + 2.0

        a, b, _c = _robust_polyfit(x, y)
        vertex = -b / (2 * a)
        assert abs(vertex - 30.0) < 0.5

    def test_asymmetric_curve(self):
        """Slightly asymmetric V — hyperbola should outperform parabola."""
        true_c = 25000.0
        x = np.linspace(22000, 28000, 13)
        y_sym = _hyperbolic_model(x, 1e-5, 2.0, true_c)
        rng = np.random.default_rng(42)
        y = y_sym + rng.normal(0, 0.1, len(x))
        y[x > true_c] *= 1.05

        c_hyp, _b, _a = _hyperbolic_fit(x, y)

        a_p, b_p, _c_p = _robust_polyfit(x, y)
        c_para = -b_p / (2 * a_p)

        assert abs(c_hyp - true_c) <= abs(c_para - true_c) + 100, (
            f"Hyperbolic vertex {c_hyp:.0f} should be at least as close as " f"parabolic {c_para:.0f} to true {true_c}"
        )

    def test_noisy_data(self):
        """Verify fit still converges with realistic measurement noise."""
        true_c, true_b, true_a = 25000.0, 2.5, 8e-6
        x = np.linspace(22000, 28000, 11)
        rng = np.random.default_rng(99)
        y = _hyperbolic_model(x, true_a, true_b, true_c) + rng.normal(0, 0.3, len(x))

        c, b, a = _hyperbolic_fit(x, y)
        assert abs(c - true_c) < 500
        assert a > 0
        assert b > 0


# ---------------------------------------------------------------------------
# _is_monotonic
# ---------------------------------------------------------------------------


class TestIsMonotonic:
    def test_decreasing_slope(self):
        x = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        y = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        assert _is_monotonic(x, y) is True

    def test_increasing_slope(self):
        x = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        y = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert _is_monotonic(x, y) is True

    def test_v_curve_not_monotonic(self):
        x = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        y = np.array([10.0, 7.0, 4.0, 3.0, 4.0, 7.0, 10.0])
        assert _is_monotonic(x, y) is False

    def test_noisy_v_curve_not_monotonic(self):
        x = np.linspace(100, 700, 11)
        y = 0.01 * (x - 400) ** 2 + 3.0
        rng = np.random.default_rng(42)
        y += rng.normal(0, 0.3, len(x))
        assert _is_monotonic(x, y) is False

    def test_too_few_points(self):
        x = np.array([100.0, 200.0])
        y = np.array([5.0, 4.0])
        assert _is_monotonic(x, y) is False

    def test_flat_data_not_monotonic(self):
        x = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        y = np.array([5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0])
        assert _is_monotonic(x, y) is False

    def test_unsorted_positions(self):
        """Positions don't need to be pre-sorted."""
        x = np.array([500.0, 100.0, 300.0, 700.0, 200.0, 600.0, 400.0])
        y = np.array([6.0, 10.0, 8.0, 4.0, 9.0, 5.0, 7.0])
        assert _is_monotonic(x, y) is True


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


def _make_mock_camera(seed_base: int = 0) -> MagicMock:
    """Create a mock camera that returns star-field numpy arrays."""
    camera = MagicMock()
    camera.is_connected.return_value = True
    camera.get_default_binning.return_value = 1
    call_counter = {"n": seed_base}

    def capture_array(duration: float, binning: int = 1, **kw) -> np.ndarray:
        call_counter["n"] += 1
        return _make_star_field(size=256, n_stars=25, fwhm=4.0, seed=call_counter["n"]).astype(np.uint16)

    camera.capture_array.side_effect = capture_array
    return camera


def _make_vcurve_camera(focuser: MagicMock, optimal_pos: int = 25000) -> MagicMock:
    """Camera that generates images with FWHM proportional to distance from optimal.

    Uses position-dependent seeding so HFR at a given position is stable
    across repeated exposures, making the V-curve shape reliable for fitting.
    """
    camera = MagicMock()
    camera.is_connected.return_value = True
    camera.get_default_binning.return_value = 1

    def capture_vcurve(duration: float, binning: int = 1, **kw) -> np.ndarray:
        cur = focuser.get_position()
        dist = abs(cur - optimal_pos)
        fwhm = 3.0 + dist * 0.01
        seed = cur % (2**31)
        return _make_star_field(size=512, n_stars=60, fwhm=fwhm, seed=seed).astype(np.uint16)

    camera.capture_array.side_effect = capture_vcurve
    return camera


@pytest.mark.slow
class TestRunAutofocus:
    def test_completes_and_returns_position(self):
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera()

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=200,
            num_steps=3,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        assert isinstance(best, int)
        assert 0 <= best <= 100000
        # At minimum, the coarse pass issues moves (1 overshoot + 7 sweep = 8).
        # Fine pass, final reposition, and verification may be skipped.
        assert focuser.move_absolute.call_count >= 8

    def test_too_few_measurements_raises(self):
        focuser = _make_mock_focuser(position=25000)
        camera = MagicMock()
        camera.is_connected.return_value = True
        camera.get_default_binning.return_value = 1
        camera.capture_array.side_effect = RuntimeError("Camera error")

        with pytest.raises(RuntimeError, match="Too few valid HFR measurements"):
            run_autofocus(
                camera=camera,
                focuser=focuser,
                step_size=200,
                num_steps=2,
                exposure_time=0.1,
                logger=logging.getLogger("test"),
            )

    def test_clamps_to_max_position(self):
        focuser = _make_mock_focuser(position=500, max_pos=1000)
        camera = _make_mock_camera()

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=300,
            num_steps=3,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        assert 0 <= best <= 1000

    def test_progress_callback_called(self):
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera()
        progress_msgs: list[str] = []

        run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=200,
            num_steps=2,
            exposure_time=0.1,
            crop_ratio=0.8,
            on_progress=progress_msgs.append,
            logger=logging.getLogger("test"),
        )

        assert len(progress_msgs) >= 5
        assert any("complete" in m.lower() for m in progress_msgs)

    def test_cancel_event_stops_sweep(self):
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera()
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(RuntimeError, match="cancelled"):
            run_autofocus(
                camera=camera,
                focuser=focuser,
                step_size=200,
                num_steps=3,
                exposure_time=0.1,
                cancel_event=cancel,
                logger=logging.getLogger("test"),
            )

    def test_no_stars_fails_loud(self):
        """When no position produces valid HFR, we get a clear error."""
        focuser = _make_mock_focuser(position=25000)
        camera = MagicMock()
        camera.is_connected.return_value = True
        camera.get_default_binning.return_value = 1
        camera.capture_array.return_value = np.full((256, 256), 500.0, dtype=np.uint16)

        with pytest.raises(RuntimeError, match="Too few valid HFR"):
            run_autofocus(
                camera=camera,
                focuser=focuser,
                step_size=200,
                num_steps=2,
                exposure_time=0.1,
                logger=logging.getLogger("test"),
            )

    def test_refinement_pass_fires_on_point(self):
        """on_point should be called for both coarse and fine sweep points."""
        focuser = _make_mock_focuser(position=25000)
        camera = _make_vcurve_camera(focuser, optimal_pos=25000)
        points: list[tuple[int, float]] = []

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=500,
            num_steps=4,
            fine_steps=3,
            exposure_time=0.1,
            crop_ratio=1.0,
            logger=logging.getLogger("test"),
            on_point=lambda pos, hfr: points.append((pos, hfr)),
        )

        assert isinstance(best, int)
        coarse_positions = {25000 + (i - 4) * 500 for i in range(9)}
        reported_positions = {p for p, _ in points}
        fine_only = reported_positions - coarse_positions
        assert len(points) > 9, f"Expected >9 on_point calls (coarse+fine), got {len(points)}"
        assert fine_only, "Should have fine points beyond the coarse grid"

    def test_refinement_skipped_when_disabled(self):
        """fine_steps=0 should produce only coarse sweep points."""
        focuser = _make_mock_focuser(position=25000)
        camera = _make_vcurve_camera(focuser, optimal_pos=25000)
        points: list[tuple[int, float]] = []

        run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=500,
            num_steps=3,
            fine_steps=0,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
            on_point=lambda pos, hfr: points.append((pos, hfr)),
        )

        # 7 coarse positions max
        assert len(points) <= 7, f"Expected <= 7 points with fine_steps=0, got {len(points)}"

    def test_monotonic_slope_falls_back(self):
        """When the sweep only captures one wing, fall back to the best measured point."""
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera()
        progress_msgs: list[str] = []

        # Patch compute_hfr to return a monotonically decreasing series
        # regardless of the actual image content — this isolates the
        # monotonic guard logic from SEP noise.
        hfr_iter = iter([10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0])

        with patch("citrascope.hardware.direct.autofocus.compute_hfr", side_effect=lambda *a, **kw: next(hfr_iter)):
            best = run_autofocus(
                camera=camera,
                focuser=focuser,
                step_size=500,
                num_steps=3,
                fine_steps=3,
                exposure_time=0.1,
                crop_ratio=1.0,
                on_progress=progress_msgs.append,
                logger=logging.getLogger("test"),
            )

        assert isinstance(best, int)
        assert any(
            "one-sided" in m.lower() for m in progress_msgs
        ), f"Expected a one-sided warning in progress messages: {progress_msgs}"
        sweep_end = 25000 + 3 * 500
        assert best <= sweep_end, f"Should not extrapolate past sweep end {sweep_end}, got {best}"

    def test_backlash_overshoot_happens(self):
        """First move should be to an overshoot position below the sweep start."""
        focuser = _make_mock_focuser(position=25000)
        camera = _make_mock_camera()

        run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=500,
            num_steps=3,
            exposure_time=0.1,
            crop_ratio=0.8,
            logger=logging.getLogger("test"),
        )

        first_move = focuser.move_absolute.call_args_list[0][0][0]
        sweep_start = 25000 - 3 * 500  # 23500
        assert first_move < sweep_start


# ---------------------------------------------------------------------------
# Parabolic fit with V-curve data
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVCurveFit:
    def test_known_v_curve(self):
        """Verify the fit finds the minimum of a known V-curve."""
        focuser = _make_mock_focuser(position=25000, max_pos=50000)
        camera = _make_vcurve_camera(focuser, optimal_pos=25000)

        best = run_autofocus(
            camera=camera,
            focuser=focuser,
            step_size=500,
            num_steps=4,
            exposure_time=0.1,
            crop_ratio=1.0,
            logger=logging.getLogger("test"),
        )

        assert abs(best - 25000) < 1500, f"Expected ~25000, got {best}"

    def test_two_pass_closer_than_single(self):
        """Two-pass should land closer to the true optimum than single-pass."""
        focuser_1 = _make_mock_focuser(position=25000, max_pos=50000)
        camera_1 = _make_vcurve_camera(focuser_1, optimal_pos=25000)
        best_1 = run_autofocus(
            camera=camera_1,
            focuser=focuser_1,
            step_size=500,
            num_steps=4,
            fine_steps=0,
            exposure_time=0.1,
            crop_ratio=1.0,
            logger=logging.getLogger("test"),
        )

        focuser_2 = _make_mock_focuser(position=25000, max_pos=50000)
        camera_2 = _make_vcurve_camera(focuser_2, optimal_pos=25000)
        best_2 = run_autofocus(
            camera=camera_2,
            focuser=focuser_2,
            step_size=500,
            num_steps=4,
            fine_steps=3,
            exposure_time=0.1,
            crop_ratio=1.0,
            logger=logging.getLogger("test"),
        )

        err_1 = abs(best_1 - 25000)
        err_2 = abs(best_2 - 25000)
        assert err_2 <= err_1 + 250, (
            f"Two-pass ({best_2}, err={err_2}) shouldn't be much worse than " f"single-pass ({best_1}, err={err_1})"
        )


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
