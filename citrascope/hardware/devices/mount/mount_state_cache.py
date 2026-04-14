"""Cached mount state — single polling thread, fast reads for consumers.

All periodic serial reads are consolidated into one 2 Hz thread that updates
an immutable snapshot behind a minimal lock.  The web UI, CableWrapCheck, and
any other status reader get mount state from the snapshot (zero serial I/O,
trivial contention) while operational commands (slew, sync, abort, home)
still go direct.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citrascope.hardware.devices.mount.abstract_mount import AbstractMount

logger = logging.getLogger("citrascope.MountStateCache")

_DEFAULT_POLL_INTERVAL_S = 0.5  # 2 Hz


@dataclass(frozen=True)
class MountSnapshot:
    """Point-in-time capture of all commonly-read mount state.

    Immutable so it can be swapped atomically under a lock and read
    without synchronisation by any number of consumers.
    """

    timestamp: float = 0.0
    ra_deg: float | None = None
    dec_deg: float | None = None
    az_deg: float | None = None
    alt_deg: float | None = None
    is_tracking: bool = False
    is_slewing: bool = False
    is_at_home: bool = False
    is_parked: bool = False
    mount_mode: str = "unknown"


@dataclass
class _StaticMountData:
    """Infrequently-changing data refreshed explicitly, not polled."""

    mount_info: dict = field(default_factory=dict)
    limits: tuple[int | None, int | None] = (None, None)


class MountStateCache:
    """Single-thread poller that keeps an up-to-date ``MountSnapshot``.

    Usage::

        cache = MountStateCache(mount)
        cache.refresh_static()
        cache.start()

        # Any thread, non-blocking:
        snap = cache.snapshot
        print(snap.az_deg, snap.is_slewing)

        cache.stop()
    """

    def __init__(self, mount: AbstractMount, poll_interval: float = _DEFAULT_POLL_INTERVAL_S) -> None:
        self._mount = mount
        self._poll_interval = poll_interval

        self._snapshot = MountSnapshot()
        self._snap_lock = threading.Lock()

        self._static = _StaticMountData()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> MountSnapshot:
        with self._snap_lock:
            return self._snapshot

    @property
    def mount_info(self) -> dict:
        return self._static.mount_info

    @property
    def limits(self) -> tuple[int | None, int | None]:
        return self._static.limits

    # ------------------------------------------------------------------
    # Explicit refresh for static/rare data
    # ------------------------------------------------------------------

    def refresh_static(self) -> None:
        """Read mount_info and limits from hardware.  Call once at connect
        and again after the user changes limits."""
        try:
            self._static.mount_info = self._mount.get_mount_info()
        except Exception:
            logger.warning("Failed to read mount_info for cache", exc_info=True)
        try:
            self._static.limits = self._mount.get_limits()
        except Exception:
            logger.warning("Failed to read limits for cache", exc_info=True)

    def refresh_limits(self) -> None:
        """Re-read just the altitude limits (after user changes them)."""
        try:
            self._static.limits = self._mount.get_limits()
        except Exception:
            logger.warning("Failed to refresh limits for cache", exc_info=True)

    def update_azimuth(self, az_deg: float) -> None:
        """Immediately update the cached azimuth (e.g. after alignment sync).

        Swaps in a new snapshot with only ``az_deg`` and ``timestamp``
        changed, so consumers see the post-sync value without waiting
        for the next polling cycle.
        """
        with self._snap_lock:
            old = self._snapshot
            self._snapshot = MountSnapshot(
                timestamp=time.monotonic(),
                ra_deg=old.ra_deg,
                dec_deg=old.dec_deg,
                az_deg=az_deg,
                alt_deg=old.alt_deg,
                is_tracking=old.is_tracking,
                is_slewing=old.is_slewing,
                is_at_home=old.is_at_home,
                is_parked=old.is_parked,
                mount_mode=old.mount_mode,
            )

    # ------------------------------------------------------------------
    # Polling thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="mount-state-cache")
        self._thread.start()
        logger.info("Mount state cache started (%.1f Hz)", 1.0 / self._poll_interval)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self._poll_interval + 2)
        self._thread = None
        logger.info("Mount state cache stopped")

    # ------------------------------------------------------------------
    # Internal poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.error("Mount state poll failed", exc_info=True)
            self._stop.wait(self._poll_interval)

    def _poll_once(self) -> None:
        """Read position + status from the mount and swap the snapshot."""
        mount = self._mount

        ra: float | None = None
        dec: float | None = None
        az: float | None = None
        alt: float | None = None
        is_tracking = False
        is_slewing = False
        is_at_home = False
        is_parked = False
        mode = "unknown"

        try:
            ra, dec = mount.get_radec()
        except Exception:
            logger.debug("Cache poll: get_radec failed", exc_info=True)

        try:
            az = mount.get_azimuth()
        except Exception:
            logger.debug("Cache poll: get_azimuth failed", exc_info=True)

        try:
            alt = mount.get_altitude()
        except Exception:
            logger.debug("Cache poll: get_altitude failed", exc_info=True)

        try:
            is_tracking = mount.is_tracking()
        except Exception:
            logger.debug("Cache poll: is_tracking failed", exc_info=True)

        try:
            is_slewing = mount.is_slewing()
        except Exception:
            logger.debug("Cache poll: is_slewing failed", exc_info=True)

        try:
            is_at_home = mount.is_home()
        except Exception:
            logger.debug("Cache poll: is_home failed", exc_info=True)

        try:
            is_parked = mount.is_parked()
        except Exception:
            logger.debug("Cache poll: is_parked failed", exc_info=True)

        try:
            mode = mount.get_mount_mode()
        except Exception:
            logger.debug("Cache poll: get_mount_mode failed", exc_info=True)

        snap = MountSnapshot(
            timestamp=time.monotonic(),
            ra_deg=ra,
            dec_deg=dec,
            az_deg=az,
            alt_deg=alt,
            is_tracking=is_tracking,
            is_slewing=is_slewing,
            is_at_home=is_at_home,
            is_parked=is_parked,
            mount_mode=mode,
        )

        with self._snap_lock:
            self._snapshot = snap
