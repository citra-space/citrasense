"""Tests for MountStateCache — polling, snapshots, lifecycle, and staleness."""

import time
from unittest.mock import MagicMock

import pytest

from citrascope.hardware.devices.mount.mount_state_cache import (
    MountSnapshot,
    MountStateCache,
)


def _make_mount(**overrides):
    """Build a mock AbstractMount with sensible defaults."""
    mount = MagicMock()
    mount.get_radec.return_value = overrides.get("radec", (180.0, 45.0))
    mount.get_azimuth.return_value = overrides.get("azimuth", 120.5)
    mount.get_altitude.return_value = overrides.get("altitude", 55.0)
    mount.is_tracking.return_value = overrides.get("tracking", True)
    mount.is_slewing.return_value = overrides.get("slewing", False)
    mount.is_home.return_value = overrides.get("home", False)
    mount.is_parked.return_value = overrides.get("parked", False)
    mount.get_mount_mode.return_value = overrides.get("mode", "altaz")
    mount.get_mount_info.return_value = overrides.get("info", {"model": "TestMount", "supports_sync": True})
    mount.get_limits.return_value = overrides.get("limits", (0, 90))
    return mount


# ------------------------------------------------------------------
# Snapshot dataclass
# ------------------------------------------------------------------


class TestMountSnapshot:
    def test_defaults(self):
        snap = MountSnapshot()
        assert snap.timestamp == 0.0
        assert snap.ra_deg is None
        assert snap.az_deg is None
        assert snap.mount_mode == "unknown"

    def test_frozen(self):
        snap = MountSnapshot(timestamp=1.0, ra_deg=10.0)
        with pytest.raises(AttributeError):
            snap.ra_deg = 20.0  # type: ignore[misc]


# ------------------------------------------------------------------
# Cache lifecycle
# ------------------------------------------------------------------


class TestMountStateCacheLifecycle:
    def test_start_stop(self):
        mount = _make_mount()
        cache = MountStateCache(mount, poll_interval=0.05)
        cache.start()
        assert cache._thread is not None
        assert cache._thread.is_alive()
        cache.stop()
        assert cache._thread is None

    def test_start_is_idempotent(self):
        mount = _make_mount()
        cache = MountStateCache(mount, poll_interval=0.05)
        cache.start()
        thread = cache._thread
        cache.start()
        assert cache._thread is thread
        cache.stop()

    def test_stop_without_start_is_safe(self):
        mount = _make_mount()
        cache = MountStateCache(mount, poll_interval=0.05)
        cache.stop()


# ------------------------------------------------------------------
# Polling behaviour
# ------------------------------------------------------------------


class TestMountStateCachePolling:
    def test_poll_once_populates_snapshot(self):
        mount = _make_mount(radec=(100.0, -30.0), azimuth=200.0, altitude=40.0)
        cache = MountStateCache(mount)
        cache._poll_once()

        snap = cache.snapshot
        assert snap.ra_deg == pytest.approx(100.0)
        assert snap.dec_deg == pytest.approx(-30.0)
        assert snap.az_deg == pytest.approx(200.0)
        assert snap.alt_deg == pytest.approx(40.0)
        assert snap.is_tracking is True
        assert snap.is_slewing is False
        assert snap.mount_mode == "altaz"
        assert snap.timestamp > 0

    def test_snapshot_updates_over_time(self):
        mount = _make_mount()
        cache = MountStateCache(mount, poll_interval=0.02)
        cache.start()
        time.sleep(0.15)
        cache.stop()

        snap = cache.snapshot
        assert snap.timestamp > 0
        assert mount.get_radec.call_count >= 2
        assert mount.get_azimuth.call_count >= 2

    def test_poll_survives_mount_errors(self):
        mount = _make_mount()
        mount.get_radec.side_effect = RuntimeError("serial timeout")
        mount.get_azimuth.side_effect = RuntimeError("serial timeout")

        cache = MountStateCache(mount)
        cache._poll_once()

        snap = cache.snapshot
        assert snap.ra_deg is None
        assert snap.az_deg is None
        assert snap.timestamp > 0

    def test_partial_failure_preserves_good_values(self):
        mount = _make_mount(azimuth=99.9, altitude=33.3)
        mount.get_radec.side_effect = RuntimeError("boom")

        cache = MountStateCache(mount)
        cache._poll_once()

        snap = cache.snapshot
        assert snap.ra_deg is None
        assert snap.dec_deg is None
        assert snap.az_deg == pytest.approx(99.9)
        assert snap.alt_deg == pytest.approx(33.3)


# ------------------------------------------------------------------
# Static data
# ------------------------------------------------------------------


class TestMountStateCacheStaticData:
    def test_refresh_static(self):
        mount = _make_mount(info={"model": "AM5", "supports_sync": True}, limits=(5, 85))
        cache = MountStateCache(mount)
        cache.refresh_static()

        assert cache.mount_info == {"model": "AM5", "supports_sync": True}
        assert cache.limits == (5, 85)

    def test_refresh_limits_only(self):
        mount = _make_mount(limits=(10, 80))
        cache = MountStateCache(mount)
        cache.refresh_static()

        mount.get_limits.return_value = (-5, 90)
        cache.refresh_limits()
        assert cache.limits == (-5, 90)
        assert mount.get_mount_info.call_count == 1

    def test_refresh_static_survives_errors(self):
        mount = _make_mount()
        mount.get_mount_info.side_effect = RuntimeError("oops")
        mount.get_limits.side_effect = RuntimeError("oops")

        cache = MountStateCache(mount)
        cache.refresh_static()
        assert cache.mount_info == {}
        assert cache.limits == (None, None)


# ------------------------------------------------------------------
# Staleness
# ------------------------------------------------------------------


class TestMountStateCacheStaleness:
    def test_fresh_snapshot_has_recent_timestamp(self):
        mount = _make_mount()
        cache = MountStateCache(mount)
        before = time.monotonic()
        cache._poll_once()
        after = time.monotonic()

        snap = cache.snapshot
        assert before <= snap.timestamp <= after

    def test_initial_snapshot_has_zero_timestamp(self):
        mount = _make_mount()
        cache = MountStateCache(mount)
        assert cache.snapshot.timestamp == 0.0
