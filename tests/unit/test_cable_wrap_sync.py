"""Tests for cable wrap sync notification — issue #118.

Verifies that alignment sync (`:CM#`) does not produce phantom azimuth
accumulation in CableWrapCheck, and that the mount's state cache is
updated atomically alongside the listener notification.
"""

import logging
import time
from unittest.mock import MagicMock

import pytest

from citrasense.hardware.devices.mount.mount_state_cache import MountSnapshot, MountStateCache
from citrasense.safety.cable_wrap_check import CableWrapCheck

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _FixedCachedState:
    """Returns fixed fields from mount.cached_state."""

    def __init__(self, az: float | None, is_at_home: bool = False):
        self.az_deg = az
        self.is_at_home = is_at_home


def _make_mount(mode: str = "altaz", az: float | None = 100.0) -> MagicMock:
    mount = MagicMock()
    mount.get_mount_mode.return_value = mode
    mount.cached_state = _FixedCachedState(az)
    mount.get_azimuth.return_value = az
    return mount


def _make_cable_check(mount: MagicMock, cumulative: float = 0.0) -> CableWrapCheck:
    cc = CableWrapCheck(logging.getLogger("test"), mount)
    cc._cumulative_deg = cumulative
    return cc


# ------------------------------------------------------------------
# MountStateCache.update_azimuth
# ------------------------------------------------------------------


class TestUpdateAzimuth:
    def test_updates_az_preserves_other_fields(self):
        mount = MagicMock()
        cache = MountStateCache(mount, poll_interval=60)
        cache._snapshot = MountSnapshot(
            timestamp=1.0,
            ra_deg=45.0,
            dec_deg=30.0,
            az_deg=100.0,
            alt_deg=50.0,
            is_tracking=True,
            is_slewing=False,
            is_at_home=False,
            is_parked=False,
            mount_mode="altaz",
        )

        cache.update_azimuth(120.0)

        snap = cache.snapshot
        assert snap.az_deg == 120.0
        assert snap.ra_deg == 45.0
        assert snap.dec_deg == 30.0
        assert snap.alt_deg == 50.0
        assert snap.is_tracking is True
        assert snap.mount_mode == "altaz"
        assert snap.timestamp > 1.0

    def test_updates_timestamp(self):
        mount = MagicMock()
        cache = MountStateCache(mount, poll_interval=60)
        before = time.monotonic()
        cache.update_azimuth(99.0)
        after = time.monotonic()
        assert before <= cache.snapshot.timestamp <= after


# ------------------------------------------------------------------
# CableWrapCheck.notify_sync
# ------------------------------------------------------------------


class TestNotifySync:
    def test_rebaselines_last_az(self):
        mount = _make_mount(az=100.0)
        cc = _make_cable_check(mount, cumulative=50.0)
        cc._last_az = 100.0

        cc.notify_sync(120.0)

        assert cc._last_az == 120.0
        assert cc._cumulative_deg == 50.0

    def test_preserves_cumulative(self):
        mount = _make_mount(az=100.0)
        cc = _make_cable_check(mount, cumulative=-35.0)
        cc._last_az = 100.0

        cc.notify_sync(80.0)

        assert cc._last_az == 80.0
        assert cc._cumulative_deg == -35.0

    def test_none_az_does_not_change_last_az(self):
        mount = _make_mount(az=100.0)
        cc = _make_cable_check(mount)
        cc._last_az = 100.0

        cc.notify_sync(None)

        assert cc._last_az == 100.0


# ------------------------------------------------------------------
# No phantom after sync + cache update
# ------------------------------------------------------------------


class TestNoPhantomAfterSync:
    def test_clean_transition_no_stale_cache(self):
        """Cache already has post-sync value — delta is near zero."""
        mount = _make_mount(az=120.0)
        cc = _make_cable_check(mount, cumulative=50.0)
        cc._last_az = 100.0

        cc.notify_sync(120.0)
        cc._observe_once()

        assert cc._cumulative_deg == pytest.approx(50.0, abs=0.01)

    def test_stale_cache_self_corrects(self):
        """Cache still has pre-sync value, then refreshes — net zero phantom."""
        mount = _make_mount(az=100.0)
        cc = _make_cable_check(mount, cumulative=50.0)
        cc._last_az = 100.0

        cc.notify_sync(120.0)

        # First observe: stale cache (100), _last_az=120 → delta=-20
        cc._observe_once()
        after_stale = cc._cumulative_deg

        # Cache refreshes to post-sync value
        mount.cached_state = _FixedCachedState(120.0)
        cc._observe_once()

        assert cc._cumulative_deg == pytest.approx(50.0, abs=0.01)
        assert after_stale == pytest.approx(30.0, abs=0.01)

    def test_normal_motion_still_accumulates(self):
        """Real physical motion after sync is tracked normally."""
        mount = _make_mount(az=120.0)
        cc = _make_cable_check(mount, cumulative=50.0)
        cc._last_az = 100.0

        cc.notify_sync(120.0)
        cc._observe_once()

        mount.cached_state = _FixedCachedState(125.0)
        cc._observe_once()

        assert cc._cumulative_deg == pytest.approx(55.0, abs=0.01)


# ------------------------------------------------------------------
# Sync fires listeners
# ------------------------------------------------------------------


class TestSyncFiresListeners:
    def test_dummy_mount_fires_listeners(self):
        from citrasense.hardware.dummy_adapter import _DummyMount

        mount = _DummyMount(logging.getLogger("test"))
        received: list[float | None] = []
        mount.register_sync_listener(lambda az: received.append(az))

        mount.sync_to_radec(180.0, 45.0)

        assert len(received) == 1
        assert isinstance(received[0], float)

    def test_multiple_listeners(self):
        from citrasense.hardware.dummy_adapter import _DummyMount

        mount = _DummyMount(logging.getLogger("test"))
        calls_a: list[float | None] = []
        calls_b: list[float | None] = []
        mount.register_sync_listener(lambda az: calls_a.append(az))
        mount.register_sync_listener(lambda az: calls_b.append(az))

        mount.sync_to_radec(90.0, 30.0)

        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_listener_exception_does_not_break_sync(self):
        from citrasense.hardware.dummy_adapter import _DummyMount

        mount = _DummyMount(logging.getLogger("test"))

        def bad_listener(az: float | None) -> None:
            raise RuntimeError("boom")

        good_calls: list[float | None] = []
        mount.register_sync_listener(bad_listener)
        mount.register_sync_listener(lambda az: good_calls.append(az))

        result = mount.sync_to_radec(180.0, 45.0)

        assert result is True
        assert len(good_calls) == 1


# ------------------------------------------------------------------
# Homing transition absorbs phantom
# ------------------------------------------------------------------


class TestHomingTransition:
    def test_homing_absorbs_phantom(self):
        """When is_at_home transitions True, az snap is not accumulated."""
        mount = _make_mount(az=138.0)
        cc = _make_cable_check(mount, cumulative=25.0)
        cc._last_az = 138.0
        cc._was_at_home = False

        # Simulate one normal observe so _was_at_home is set
        cc._observe_once()
        assert cc._cumulative_deg == pytest.approx(25.0, abs=0.01)

        # Mount finishes homing: az snaps to 0, is_at_home becomes True
        mount.cached_state = _FixedCachedState(0.0, is_at_home=True)
        cc._observe_once()

        assert cc._last_az == 0.0
        assert cc._cumulative_deg == pytest.approx(25.0, abs=0.01)

    def test_normal_motion_after_homing_accumulates(self):
        """Real motion after homing is tracked normally."""
        mount = _make_mount(az=138.0)
        cc = _make_cable_check(mount, cumulative=25.0)
        cc._last_az = 138.0
        cc._was_at_home = False

        # Homing completes
        mount.cached_state = _FixedCachedState(0.0, is_at_home=True)
        cc._observe_once()
        assert cc._cumulative_deg == pytest.approx(25.0, abs=0.01)

        # Mount slews away from home
        mount.cached_state = _FixedCachedState(10.0, is_at_home=False)
        cc._observe_once()
        assert cc._cumulative_deg == pytest.approx(35.0, abs=0.01)

    def test_repeated_at_home_does_not_rebaseline(self):
        """Once the transition is absorbed, subsequent at-home polls accumulate normally."""
        mount = _make_mount(az=138.0)
        cc = _make_cable_check(mount, cumulative=25.0)
        cc._last_az = 138.0
        cc._was_at_home = False

        # Homing completes
        mount.cached_state = _FixedCachedState(0.0, is_at_home=True)
        cc._observe_once()
        assert cc._was_at_home is True

        # Still at home, small az drift (e.g. tracking started)
        mount.cached_state = _FixedCachedState(0.5, is_at_home=True)
        cc._observe_once()

        assert cc._cumulative_deg == pytest.approx(25.5, abs=0.01)

    def test_already_at_home_on_startup(self):
        """If the mount is already homed when cable wrap starts, no phantom on first read."""
        mount = _make_mount(az=0.0)
        mount.cached_state = _FixedCachedState(0.0, is_at_home=True)
        cc = _make_cable_check(mount, cumulative=0.0)

        cc._observe_once()

        assert cc._last_az == 0.0
        assert cc._cumulative_deg == pytest.approx(0.0, abs=0.01)
        assert cc._was_at_home is True
