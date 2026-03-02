"""Tests for CableWrapCheck — shortest-arc math, accumulation, limits, persistence."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from citrascope.hardware.devices.mount.mount_state_cache import MountSnapshot
from citrascope.safety.cable_wrap_check import (
    HARD_LIMIT_DEG,
    SOFT_LIMIT_DEG,
    CableWrapCheck,
    _shortest_arc,
)
from citrascope.safety.safety_monitor import SafetyAction

# ------------------------------------------------------------------
# Shortest-arc math
# ------------------------------------------------------------------


class TestShortestArc:
    def test_zero_delta(self):
        assert _shortest_arc(100.0, 100.0) == 0.0

    def test_small_cw(self):
        assert _shortest_arc(10.0, 20.0) == pytest.approx(10.0)

    def test_small_ccw(self):
        assert _shortest_arc(20.0, 10.0) == pytest.approx(-10.0)

    def test_wrap_cw(self):
        assert _shortest_arc(350.0, 10.0) == pytest.approx(20.0)

    def test_wrap_ccw(self):
        assert _shortest_arc(10.0, 350.0) == pytest.approx(-20.0)

    def test_half_circle(self):
        result = _shortest_arc(0.0, 180.0)
        assert result == pytest.approx(180.0) or result == pytest.approx(-180.0)

    def test_large_cw_through_zero(self):
        assert _shortest_arc(300.0, 60.0) == pytest.approx(120.0)


# ------------------------------------------------------------------
# CableWrapCheck
# ------------------------------------------------------------------


class _CachedStateSequence:
    """Returns sequential az_deg values from mount.cached_state attribute access."""

    def __init__(self, azimuths: list[float | None]):
        self._iter = iter(azimuths)

    @property
    def az_deg(self) -> float | None:
        return next(self._iter, None)


def _make_mount(mode: str = "altaz", azimuths: list[float | None] | None = None):
    mount = MagicMock()
    mount.get_mount_mode.return_value = mode
    if azimuths is not None:
        mount.get_azimuth.side_effect = list(azimuths[1:]) if len(azimuths) > 1 else []
        mount.cached_state = _CachedStateSequence(azimuths)
    else:
        mount.get_azimuth.return_value = None
        mount.cached_state = MountSnapshot()
    mount.start_move.return_value = True
    mount.stop_move.return_value = True
    mount.stop_tracking.return_value = True
    mount.get_radec.side_effect = Exception("not wired")
    return mount


class TestCableWrapCheckBasics:
    def test_equatorial_mode_always_safe(self):
        mount = _make_mount(mode="equatorial", azimuths=[100.0])
        check = CableWrapCheck(MagicMock(), mount)
        assert check.check() == SafetyAction.SAFE

    def test_no_azimuth_warns_in_altaz(self):
        """Lost azimuth in alt-az mode is WARN (fail-closed)."""
        mount = _make_mount(mode="altaz")
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        assert check.check() == SafetyAction.WARN

    def test_initial_reading_is_safe(self):
        mount = _make_mount(azimuths=[10.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        assert check.check() == SafetyAction.SAFE
        assert check._cumulative_deg == 0.0

    def test_accumulation_basic(self):
        mount = _make_mount(azimuths=[10.0, 30.0, 50.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._observe_once()
        check._observe_once()
        assert check._cumulative_deg == pytest.approx(40.0)

    def test_check_is_pure_read(self):
        """Calling check() multiple times doesn't change internal state."""
        mount = _make_mount(azimuths=[10.0, 30.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._observe_once()
        cumulative_before = check._cumulative_deg
        check.check()
        check.check()
        check.check()
        assert check._cumulative_deg == cumulative_before


class TestCableWrapCheckLimits:
    def test_soft_limit_triggers_queue_stop(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = SOFT_LIMIT_DEG
        assert check.check() == SafetyAction.QUEUE_STOP

    def test_hard_limit_triggers_emergency(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = HARD_LIMIT_DEG
        assert check.check() == SafetyAction.EMERGENCY

    def test_below_soft_limit_is_safe(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = SOFT_LIMIT_DEG - 1
        assert check.check() == SafetyAction.SAFE

    def test_negative_cumulative_soft_limit(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = -SOFT_LIMIT_DEG
        assert check.check() == SafetyAction.QUEUE_STOP


class TestCableWrapCheckReset:
    def test_reset_clears_state(self):
        mount = _make_mount(azimuths=[0.0, 100.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._observe_once()
        assert check._cumulative_deg != 0.0
        check.reset()
        assert check._cumulative_deg == 0.0
        assert check._last_az is None


class TestCableWrapCheckPersistence:
    def test_save_and_load(self, tmp_path: Path):
        state_file = tmp_path / "wrap.json"
        mount = _make_mount(azimuths=[0.0, 100.0])

        check = CableWrapCheck(MagicMock(), mount, state_file=state_file)
        check._observe_once()
        check._last_save_time = 0.0
        check._observe_once()
        assert state_file.exists()

        data = json.loads(state_file.read_text())
        assert data["cumulative_deg"] == pytest.approx(100.0)

        mount2 = _make_mount(azimuths=[100.0])
        check2 = CableWrapCheck(MagicMock(), mount2, state_file=state_file)
        assert check2._cumulative_deg == pytest.approx(100.0)

    def test_missing_state_file_warns(self, tmp_path: Path):
        state_file = tmp_path / "nonexistent.json"
        logger = MagicMock()
        mount = _make_mount(azimuths=[0.0])
        CableWrapCheck(logger, mount, state_file=state_file)
        logger.warning.assert_called_once()


class TestCableWrapCheckProposedAction:
    def test_slew_allowed_with_plenty_of_headroom(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        assert check.check_proposed_action("slew") is True

    def test_slew_blocked_at_soft_limit(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = SOFT_LIMIT_DEG
        assert check.check_proposed_action("slew") is False

    def test_slew_allowed_just_below_soft_limit(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = SOFT_LIMIT_DEG - 1
        assert check.check_proposed_action("slew") is True

    def test_slew_blocked_during_unwind(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._unwinding = True
        assert check.check_proposed_action("slew") is False

    def test_non_slew_allowed_during_unwind(self):
        """Non-slew actions are still blocked during unwind (safety blanket)."""
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._unwinding = True
        assert check.check_proposed_action("capture") is False

    def test_capture_allowed_below_soft_limit(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        assert check.check_proposed_action("capture") is True

    def test_home_blocked_at_soft_limit(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = SOFT_LIMIT_DEG
        assert check.check_proposed_action("home") is False

    def test_home_allowed_below_soft_limit(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        assert check.check_proposed_action("home") is True


class TestCableWrapCheckUnwindBehavior:
    def test_check_returns_queue_stop_during_unwind(self):
        """During unwind, check() returns QUEUE_STOP (not EMERGENCY) so the
        watchdog doesn't fire abort_slew and fight the unwind."""
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._unwinding = True
        assert check.check() == SafetyAction.QUEUE_STOP

    def test_execute_action_guards_double_entry(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._unwinding = True
        check.execute_action()
        mount.stop_tracking.assert_not_called()

    def test_observe_once_yields_during_unwind(self):
        """Observer thread should not accumulate while unwind is active."""
        mount = _make_mount(azimuths=[10.0, 30.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._unwinding = True
        check._observe_once()
        assert check._cumulative_deg == 0.0


class TestCableWrapStallDetection:
    """Stall detection during unwind must handle the 0/360 azimuth boundary."""

    def test_stall_detected_near_zero_boundary(self):
        """Tiny steps crossing the 0/360 boundary must still be detected
        as a stall using wrapped deltas, not raw span."""
        azimuths = [
            0.0,  # cached_state for _observe_once()
            0.0,  # _do_unwind: start_az
            359.96,  # poll 1
            359.97,  # poll 2
            359.98,  # poll 3
            359.99,  # poll 4
            0.00,  # poll 5
            0.01,  # poll 6 — stall (max step 0.01° < 0.1°)
            0.01,  # finally: end_az
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 280.0

        check.execute_action()
        mount.stop_move.assert_called_once()

    def test_real_motion_not_flagged_as_stall(self):
        """Steps well above the stall threshold converge without stall."""
        azimuths = [
            10.0,  # cached_state for _observe_once()
            10.0,  # _do_unwind: start_az
            10.0,  # poll 1
            7.0,  # poll 2
            4.0,  # poll 3 — cumulative 8-6=2 < convergence(5)
            4.0,  # finally: end_az
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 8.0

        check.execute_action()
        mount.stop_move.assert_called_once()
        logger = check._logger
        for call in logger.error.call_args_list:
            assert "stall" not in str(call).lower()


class TestCableWrapCheckUnwindReset:
    """Unwind should only reset cumulative on convergence, not on failure."""

    def test_convergence_resets_cumulative(self):
        """When the unwind converges, cumulative is reset to 0."""
        azimuths = [
            10.0,  # baseline for _observe_once()
            10.0,  # _do_unwind: start_az
            10.0,  # poll 1
            7.0,  # poll 2
            4.0,  # poll 3 — cumulative drops below _CONVERGENCE_DEG
            4.0,  # finally: end_az
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 8.0

        check.execute_action()
        assert check._cumulative_deg == 0.0

    def test_stall_preserves_cumulative(self):
        """When unwind stalls, cumulative is NOT reset."""
        azimuths = [
            0.0,  # cached_state for _observe_once()
            0.0,  # _do_unwind: start_az
            0.0,  # poll 1
            0.0,  # poll 2
            0.0,  # poll 3
            0.0,  # poll 4
            0.0,  # poll 5
            0.0,  # poll 6 — stall detected (0° max step)
            0.0,  # finally: end_az
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 250.0

        check.execute_action()
        assert check._cumulative_deg == pytest.approx(250.0)

    def test_lost_azimuth_preserves_cumulative(self):
        """When azimuth reading is lost during unwind, cumulative is preserved."""
        azimuths = [
            10.0,  # baseline for _observe_once()
            10.0,  # _do_unwind: start_az
            None,  # poll 1 — lost azimuth
            None,  # finally: end_az
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 250.0

        check.execute_action()
        assert check._cumulative_deg == pytest.approx(250.0)


class TestCableWrapCheckStatus:
    def test_get_status(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        status = check.get_status()
        assert status["name"] == "cable_wrap"
        assert "cumulative_deg" in status
        assert status["soft_limit"] == SOFT_LIMIT_DEG
        assert status["hard_limit"] == HARD_LIMIT_DEG
        assert status["intervention_required"] is False
        assert status["consecutive_failures"] == 0

    def test_status_shows_intervention_required(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._intervention_required = True
        check._consecutive_unwind_failures = 3
        status = check.get_status()
        assert status["intervention_required"] is True
        assert status["consecutive_failures"] == 3


class TestCableWrapRetryCap:
    """Unwind retry cap latches to intervention-required after repeated failures."""

    def _make_stalling_mount(self):
        """Mount that always stalls (zero motion) with enough readings."""
        azimuths = [0.0] * 10
        return _make_mount(azimuths=azimuths)

    def test_single_failure_increments_counter(self):
        mount = _make_mount(azimuths=[0.0] * 10)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 280.0
        check.execute_action()
        assert check._consecutive_unwind_failures == 1
        assert check._intervention_required is False

    def test_three_failures_latch_intervention(self):
        for i in range(3):
            mount = _make_mount(azimuths=[0.0] * 10)
            if i == 0:
                check = CableWrapCheck(MagicMock(), mount)
                check._observe_once()
            else:
                check._mount = mount
                check._last_az = 0.0
            check._cumulative_deg = 280.0
            check.execute_action()
        assert check._consecutive_unwind_failures == 3
        assert check._intervention_required is True

    def test_intervention_blocks_further_unwinds(self):
        mount = _make_mount(azimuths=[0.0] * 10)
        check = CableWrapCheck(MagicMock(), mount)
        check._intervention_required = True
        check._cumulative_deg = 280.0
        check.execute_action()
        mount.stop_tracking.assert_not_called()

    def test_reset_clears_intervention(self):
        mount = _make_mount(azimuths=[0.0])
        check = CableWrapCheck(MagicMock(), mount)
        check._intervention_required = True
        check._consecutive_unwind_failures = 3
        check.reset()
        assert check._intervention_required is False
        assert check._consecutive_unwind_failures == 0

    def test_convergence_resets_failure_counter(self):
        mount = _make_mount(azimuths=[0.0] * 10)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 280.0
        check.execute_action()
        assert check._consecutive_unwind_failures == 1

        azimuths_converge = [
            10.0,  # cached_state for _observe_once
            10.0,  # start_az
            10.0,  # poll 1
            7.0,  # poll 2
            4.0,  # poll 3 — convergence (cumulative 8-6=2 < 5)
            4.0,  # end_az
        ]
        check._mount = _make_mount(azimuths=azimuths_converge)
        check._observe_once()
        check._cumulative_deg = 8.0
        check.execute_action()
        assert check._consecutive_unwind_failures == 0


class TestCableWrapMultiSegmentUnwind:
    """Multi-segment unwind: firmware caps motion at ~191°, so we stop/restart."""

    def test_firmware_limit_triggers_restart_and_converges(self):
        """Segment 1 travels 40° then stalls (firmware limit).
        Segment 2 finishes the job and converges."""
        azimuths = [
            200.0,  # cached_state for _observe_once
            200.0,  # _do_unwind: start_az
            # Segment 1 polls (7):
            180.0,  # poll 1 — moving
            160.0,  # poll 2 — moving
            160.0,  # poll 3 — stalled
            160.0,  # poll 4
            160.0,  # poll 5
            160.0,  # poll 6 — window still has big step from 180→160
            160.0,  # poll 7 — window is [160]*6, max_step=0 → stall
            # Segment 2 polls (2):
            155.0,  # poll 1 — moving (cumulative: 10→5)
            150.0,  # poll 2 — converged (cumulative: 5→0)
            # _do_unwind: end_az
            150.0,
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 50.0

        check.execute_action()

        assert check._cumulative_deg == 0.0
        assert check._consecutive_unwind_failures == 0
        assert mount.start_move.call_count == 2
        assert mount.stop_move.call_count == 2

    def test_real_stall_after_tiny_travel_aborts(self):
        """If a segment barely moves (<10°) before stalling, it's a real
        obstruction — don't restart, count it as a failure."""
        azimuths = [
            0.0,  # cached_state for _observe_once
            0.0,  # _do_unwind: start_az
            # Segment 1: barely moves then stalls
            1.0,  # poll 1
            1.0,  # poll 2
            1.0,  # poll 3
            1.0,  # poll 4
            1.0,  # poll 5
            1.0,  # poll 6 — window = 6 readings, max_step ≈ 0
            # _do_unwind: end_az
            1.0,
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 280.0

        check.execute_action()

        assert check._consecutive_unwind_failures == 1
        assert check._cumulative_deg > 0
        assert mount.start_move.call_count == 1

    def test_max_restarts_respected(self):
        """Even if every segment looks like a firmware limit, we stop after
        _MAX_SEGMENT_RESTARTS + 1 segments."""
        from citrascope.safety.cable_wrap_check import _MAX_SEGMENT_RESTARTS

        # 7 polls per segment (same pattern as segment 1 above)
        segment_polls = [80.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
        num_segments = _MAX_SEGMENT_RESTARTS + 2  # more than we'd ever need
        all_polls = segment_polls * num_segments
        azimuths = [
            100.0,  # cached_state for _observe_once
            100.0,  # _do_unwind: start_az
            *all_polls,
            60.0,  # _do_unwind: end_az
        ]
        mount = _make_mount(azimuths=azimuths)
        check = CableWrapCheck(MagicMock(), mount)
        check._observe_once()
        check._cumulative_deg = 500.0

        check.execute_action()

        assert mount.start_move.call_count == _MAX_SEGMENT_RESTARTS + 1


class TestCableWrapObserverLifecycle:
    def test_start_stop(self):
        mount = _make_mount(azimuths=[10.0] * 100)
        check = CableWrapCheck(MagicMock(), mount)
        check.start()
        assert check._observe_thread is not None
        assert check._observe_thread.is_alive()
        check.stop()
        assert check._observe_thread is None

    def test_start_is_idempotent(self):
        mount = _make_mount(azimuths=[10.0] * 100)
        check = CableWrapCheck(MagicMock(), mount)
        check.start()
        thread = check._observe_thread
        check.start()
        assert check._observe_thread is thread
        check.stop()
