"""Tests for AlignmentManager calibration rollback and safety gating."""

from __future__ import annotations

import logging
import math
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from citrasense.hardware.devices.mount.altaz_pointing_model import AltAzPointingModel, altaz_to_radec
from citrasense.sensors.telescope.managers.alignment_manager import AlignmentManager


@pytest.fixture(autouse=True)
def _patch_gast_degrees():
    """Patch ``gast_degrees`` as imported into ``altaz_pointing_model``.

    See the matching fixture in ``tests/unit/test_altaz_pointing_model.py``
    for the full note on why we target the consumer's namespace rather than
    ``citrasense.astro.sidereal``.
    """
    with patch(
        "citrasense.hardware.devices.mount.altaz_pointing_model.gast_degrees",
        return_value=90.0,
    ):
        yield


def _synthetic_points(
    AN: float = 0.3, AW: float = 0.2, IE: float = 0.1, n: int = 10
) -> list[tuple[float, float, float, float, float, float]]:
    """Minimal synthetic calibration data."""
    rng = np.random.RandomState(42)
    lat, lon = 40.0, -74.0
    points = []
    for _ in range(n):
        az_enc = rng.uniform(30.0, 330.0)
        alt_enc = rng.uniform(25.0, 75.0)
        az_rad = math.radians(az_enc)
        alt_rad = math.radians(alt_enc)
        d_az = AN * math.sin(az_rad) * math.tan(alt_rad) - AW * math.cos(az_rad) * math.tan(alt_rad)
        d_alt = IE - AN * math.cos(az_rad) - AW * math.sin(az_rad)
        mount_ra, mount_dec = altaz_to_radec(az_enc, alt_enc, lat, lon)
        solved_ra, solved_dec = altaz_to_radec(az_enc + d_az, alt_enc + d_alt, lat, lon)
        points.append((mount_ra, mount_dec, solved_ra, solved_dec, lat, lon))
    return points


def _trained_model() -> AltAzPointingModel:
    model = AltAzPointingModel()
    for args in _synthetic_points(n=10):
        model.add_point(*args)
    assert model.is_active
    return model


def _make_manager(
    safety_monitor=None,
    pointing_model: AltAzPointingModel | None = None,
) -> AlignmentManager:
    adapter = MagicMock()
    adapter.telescope_record = {"id": "test"}
    adapter.is_camera_connected.return_value = True
    adapter.angular_distance.return_value = 0.01

    mock_mount = MagicMock()
    mock_mount.get_radec.return_value = (180.0, 45.0)
    mock_mount.get_azimuth.return_value = 180.0
    mock_mount.slew_to_radec.return_value = True
    mock_mount.is_slewing.return_value = False
    mock_mount.get_limits.return_value = (15, 89)
    adapter.mount = mock_mount

    settings = MagicMock()
    settings.alignment_exposure_seconds = 1.0

    location_service = MagicMock()
    location_service.get_current_location.return_value = {"latitude": 40.0, "longitude": -74.0}

    sensor_config = MagicMock()
    sensor_config.alignment_exposure_seconds = 1.0
    mgr = AlignmentManager(
        logger=logging.getLogger("test"),
        hardware_adapter=adapter,
        settings=settings,
        sensor_id="scope-1",
        sensor_config=sensor_config,
        safety_monitor=safety_monitor,
        location_service=location_service,
    )
    if pointing_model is not None:
        mgr.set_pointing_model(pointing_model)
    return mgr


# ------------------------------------------------------------------
# Issue 1: Calibration rollback on failure
# ------------------------------------------------------------------


class TestCalibrationRollback:
    def test_model_restored_when_insufficient_points(self):
        """If calibration collects < 3 points, the previous model is restored."""
        model = _trained_model()
        original_terms = model.status()["terms"].copy()
        mgr = _make_manager(pointing_model=model)

        with patch.object(mgr, "_plate_solve_at_current_position", return_value=None):
            with patch.object(mgr, "_unwind_before_calibration"):
                mgr._execute_calibration()

        assert model.is_active, "Previous model should have been restored"
        for term in ("AN", "AW", "IE"):
            assert model.status()["terms"][term] == original_terms[term]

    def test_model_restored_when_cancelled_after_reset(self):
        """If cancelled after the model was reset, the previous model is restored."""
        model = _trained_model()
        mgr = _make_manager(pointing_model=model)

        mgr._calibration_cancel.set()

        with patch.object(mgr, "_plate_solve_at_current_position", return_value=(180.0, 45.0)):
            with patch.object(mgr, "_unwind_before_calibration"):
                mgr._execute_calibration()

        assert model.is_active, "Previous model should have been restored after cancel"

    def test_no_restore_when_calibration_succeeds(self):
        """A successful calibration keeps the new model."""
        model = _trained_model()
        old_rms = model.rms_deg
        mgr = _make_manager(pointing_model=model)

        call_count = 0

        def _solve(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (180.0, 45.0)
            return (180.0 + call_count * 0.5, 45.0 + call_count * 0.3)

        with patch.object(mgr, "_plate_solve_at_current_position", side_effect=_solve):
            with patch.object(mgr, "_unwind_before_calibration"):
                with patch.object(mgr, "_unwind_if_needed"):
                    with patch.object(mgr, "_verify_calibration"):
                        mgr._execute_calibration()

        assert model.is_active
        assert model.rms_deg != old_rms, "New calibration should have different RMS"

    def test_no_crash_when_no_previous_model(self):
        """A failed calibration with no previous model should not crash."""
        model = AltAzPointingModel()
        assert not model.is_active
        mgr = _make_manager(pointing_model=model)

        with patch.object(mgr, "_plate_solve_at_current_position", return_value=None):
            with patch.object(mgr, "_unwind_before_calibration"):
                mgr._execute_calibration()

        assert not model.is_active


# ------------------------------------------------------------------
# Issue 2: Slew safety during calibration
# ------------------------------------------------------------------


class TestCalibrationSlewSafety:
    def test_calibration_aborts_when_slew_blocked_at_start(self):
        """If the safety monitor blocks slews at the start, calibration aborts."""
        safety = MagicMock()
        safety.is_action_safe.side_effect = lambda action, **kw: action != "slew"

        model = AltAzPointingModel()
        mgr = _make_manager(safety_monitor=safety, pointing_model=model)

        with patch.object(mgr, "_plate_solve_at_current_position") as mock_solve:
            with patch.object(mgr, "_unwind_before_calibration"):
                mgr._execute_calibration()

        mock_solve.assert_not_called()

    def test_calibration_aborts_when_capture_blocked_at_start(self):
        """If the safety monitor blocks captures at the start, calibration aborts."""
        safety = MagicMock()
        safety.is_action_safe.side_effect = lambda action, **kw: action != "capture"

        model = AltAzPointingModel()
        mgr = _make_manager(safety_monitor=safety, pointing_model=model)

        with patch.object(mgr, "_plate_solve_at_current_position") as mock_solve:
            with patch.object(mgr, "_unwind_before_calibration"):
                mgr._execute_calibration()

        mock_solve.assert_not_called()

    def test_per_point_slew_check_skips_blocked_points(self):
        """If slew is blocked mid-calibration, that point is skipped but calibration continues."""
        # Allow the initial safety checks to pass, then block the first grid walk slew,
        # then allow subsequent ones.
        slew_check_count = [0]

        def _safety_check(action, **kw):
            if action == "slew":
                slew_check_count[0] += 1
                # Call 1 = initial check at top of _do_calibration → allow
                # Call 2 = first grid walk point → block
                # Call 3+ = subsequent grid walk points → allow
                return slew_check_count[0] != 2
            return True

        safety = MagicMock()
        safety.is_action_safe.side_effect = _safety_check
        safety.get_check.return_value = None

        model = AltAzPointingModel()
        mgr = _make_manager(safety_monitor=safety, pointing_model=model)

        solve_count = [0]

        def _solve(*_args, **_kwargs):
            solve_count[0] += 1
            if solve_count[0] == 1:
                return (180.0, 45.0)
            return (180.0 + solve_count[0], 45.0)

        mount = cast(MagicMock, mgr.hardware_adapter.mount)

        with patch.object(mgr, "_plate_solve_at_current_position", side_effect=_solve):
            with patch.object(mgr, "_unwind_before_calibration"):
                with patch.object(mgr, "_unwind_if_needed"):
                    with patch.object(mgr, "_verify_calibration"):
                        mgr._execute_calibration()

        total_slew_calls = mount.slew_to_radec.call_count
        total_slew_checks = slew_check_count[0]

        # Initial check at top = 1, then one per grid point.
        # The first grid point is blocked → no slew_to_radec for it.
        assert total_slew_checks > 2, "Safety should be checked at start + per grid point"
        # Grid has 15 points; first is blocked, so at most 14 slew_to_radec calls.
        assert total_slew_calls < 15, f"Blocked point should have been skipped (got {total_slew_calls} slews)"

    def test_safety_monitor_called_with_slew_action(self):
        """The safety monitor receives 'slew' action type during calibration."""
        safety = MagicMock()
        safety.is_action_safe.return_value = True
        safety.get_check.return_value = None

        model = AltAzPointingModel()
        mgr = _make_manager(safety_monitor=safety, pointing_model=model)

        solve_count = [0]

        def _solve(*_args, **_kwargs):
            solve_count[0] += 1
            if solve_count[0] == 1:
                return (180.0, 45.0)
            return (180.0 + solve_count[0], 45.0)

        with patch.object(mgr, "_plate_solve_at_current_position", side_effect=_solve):
            with patch.object(mgr, "_unwind_before_calibration"):
                with patch.object(mgr, "_unwind_if_needed"):
                    with patch.object(mgr, "_verify_calibration"):
                        mgr._execute_calibration()

        action_types = [call[0][0] for call in safety.is_action_safe.call_args_list]
        assert "slew" in action_types, "Safety monitor should have been checked for 'slew'"
        assert "capture" in action_types, "Safety monitor should have been checked for 'capture'"
