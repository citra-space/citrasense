"""Tests for CalibrationManager's FlatCaptureBackend wiring and scheduled auto-capture."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from citrasense.calibration.calibration_library import CalibrationLibrary
from citrasense.sensors.telescope.managers.calibration_manager import CalibrationManager


class _FakeFlatBackend:
    """In-memory FlatCaptureBackend that writes stub FITS files and records args."""

    def __init__(self, median_value: int = 30000, shape=(8, 8)) -> None:
        self._median = median_value
        self._shape = shape
        self.calls: list[dict] = []
        self.cancelled = False

    @property
    def supported_frame_types(self) -> set[str]:
        return {"flat"}

    def cancel(self) -> None:
        self.cancelled = True

    def capture_flat_frames(self, **kwargs) -> list[Path]:
        self.calls.append(dict(kwargs))
        library: CalibrationLibrary = kwargs["library"]
        count: int = kwargs["count"]
        paths: list[Path] = []
        for i in range(count):
            data = np.full(self._shape, self._median, dtype=np.uint16)
            p = library.tmp_dir / f"fake_flat_{i}.fits"
            fits.PrimaryHDU(data).writeto(p, overwrite=True)
            paths.append(p)
        return paths


@pytest.fixture
def library(tmp_path) -> CalibrationLibrary:
    return CalibrationLibrary(root=tmp_path / "calibration")


@pytest.fixture
def adapter_stub() -> MagicMock:
    a = MagicMock()
    a.camera = None  # NINA-style: no direct camera
    a.filter_map = {1: {"name": "R", "enabled": True}}
    a.supports_filter_management = MagicMock(return_value=True)
    a.get_filter_config = MagicMock(return_value={1: {"name": "R", "enabled": True}})
    a.get_calibration_profile_summary = MagicMock(
        return_value={
            "camera_id": "SN-TEST",
            "model": "TestCam",
            "current_gain": 100,
            "current_binning": 1,
            "current_temperature": -10.0,
            "target_temperature": -10.0,
            "read_mode": "default",
            "has_mechanical_shutter": False,
            "has_cooling": True,
        }
    )
    return a


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.calibration_manager")


def _make_manager(adapter_stub, library, logger, flat_backend=None, location_provider=None, settings=None):
    return CalibrationManager(
        logger=logger,
        hardware_adapter=adapter_stub,
        library=library,
        imaging_queue=None,
        flat_backend=flat_backend,
        settings=settings,
        sensor_id="test-sensor",
        location_provider=location_provider,
    )


def test_supports_frame_type_flat_true_with_backend(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=backend)
    assert mgr.supports_frame_type("flat") is True


def test_supports_frame_type_bias_false_without_camera(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=backend)
    assert mgr.supports_frame_type("bias") is False
    assert mgr.supports_frame_type("dark") is False


def test_request_rejects_unsupported_frame_type(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=backend)
    assert mgr.request({"frame_type": "bias", "count": 5}) is False


def test_request_suite_rejects_unsupported_types(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=backend)
    jobs = [
        {"frame_type": "flat", "count": 5, "filter_position": 1, "filter_name": "R"},
        {"frame_type": "bias", "count": 5},
    ]
    assert mgr.request_suite(jobs) is False


def test_execute_flat_via_backend_saves_master(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=backend)

    params = {
        "frame_type": "flat",
        "count": 3,
        "gain": 100,
        "binning": 1,
        "filter_position": 1,
        "filter_name": "R",
    }
    mgr._execute_flat_via_backend(params)

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["count"] == 3
    assert call["filter_slot"].position == 1
    assert call["filter_slot"].name == "R"

    # Master flat should have been saved to the library.
    masters = list((library.masters_dir).glob("master_flat_*.fits"))
    assert len(masters) == 1


def test_scheduled_no_op_without_backend(adapter_stub, library, logger):
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=None)
    assert mgr._maybe_auto_capture_flats() is False


def test_scheduled_no_op_without_settings(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    mgr = _make_manager(adapter_stub, library, logger, flat_backend=backend)
    assert mgr._maybe_auto_capture_flats() is False


def test_scheduled_dedup_within_same_window(adapter_stub, library, logger):
    """Once _last_served_window_start matches the current window, no re-fire."""
    backend = _FakeFlatBackend()

    # Minimal settings stub with a sensor config that has auto-capture enabled.
    sensor_cfg = MagicMock()
    sensor_cfg.auto_capture_flats_enabled = True
    sensor_cfg.last_flats_capture_iso = None
    sensor_cfg.flat_frame_count = 5
    settings = MagicMock()
    settings.get_sensor_config = MagicMock(return_value=sensor_cfg)
    settings.update_and_save = MagicMock()

    def _loc() -> dict:
        return {"latitude": 35.0, "longitude": -106.0}

    mgr = _make_manager(
        adapter_stub,
        library,
        logger,
        flat_backend=backend,
        settings=settings,
        location_provider=_loc,
    )

    # Simulate a just-served window
    mgr._last_served_window_start = "2026-04-29T18:00:00"

    # Patch compute_twilight to return the same window start.
    fake_window = MagicMock()
    fake_window.start = "2026-04-29T18:00:00"
    fake_window.end = "2026-04-29T18:20:00"
    twilight = MagicMock()
    twilight.in_flat_window = True
    twilight.flat_window = fake_window

    import citrasense.sensors.telescope.managers.calibration_manager as cm

    original = cm.compute_twilight
    cm.compute_twilight = MagicMock(return_value=twilight)  # type: ignore[assignment]
    try:
        fired = mgr._maybe_auto_capture_flats()
    finally:
        cm.compute_twilight = original  # type: ignore[assignment]

    assert fired is False


def test_scheduled_fires_on_new_window(adapter_stub, library, logger):
    """When in a flat window and no record of serving it, auto-capture fires."""
    backend = _FakeFlatBackend()

    sensor_cfg = MagicMock()
    sensor_cfg.auto_capture_flats_enabled = True
    sensor_cfg.last_flats_capture_iso = "2026-04-28T18:00:00"  # a previous day
    sensor_cfg.flat_frame_count = 5
    settings = MagicMock()
    settings.get_sensor_config = MagicMock(return_value=sensor_cfg)
    settings.update_and_save = MagicMock()

    def _loc() -> dict:
        return {"latitude": 35.0, "longitude": -106.0}

    mgr = _make_manager(
        adapter_stub,
        library,
        logger,
        flat_backend=backend,
        settings=settings,
        location_provider=_loc,
    )

    fake_window = MagicMock()
    fake_window.start = "2026-04-29T18:00:00"
    fake_window.end = "2026-04-29T18:20:00"
    twilight = MagicMock()
    twilight.in_flat_window = True
    twilight.flat_window = fake_window

    import citrasense.sensors.telescope.managers.calibration_manager as cm

    original = cm.compute_twilight
    cm.compute_twilight = MagicMock(return_value=twilight)  # type: ignore[assignment]
    try:
        fired = mgr._maybe_auto_capture_flats()
    finally:
        cm.compute_twilight = original  # type: ignore[assignment]

    assert fired is True
    # Should have queued a suite (one job per enabled filter).
    with mgr._lock:
        assert len(mgr._job_queue) == 1
        assert mgr._requested is True
    assert mgr._last_served_window_start == "2026-04-29T18:00:00"


def test_scheduled_skipped_when_imaging_queue_busy(adapter_stub, library, logger):
    backend = _FakeFlatBackend()

    sensor_cfg = MagicMock()
    sensor_cfg.auto_capture_flats_enabled = True
    sensor_cfg.last_flats_capture_iso = None
    sensor_cfg.flat_frame_count = 5
    settings = MagicMock()
    settings.get_sensor_config = MagicMock(return_value=sensor_cfg)

    queue = MagicMock()
    queue.is_idle = MagicMock(return_value=False)

    mgr = CalibrationManager(
        logger=logger,
        hardware_adapter=adapter_stub,
        library=library,
        imaging_queue=queue,
        flat_backend=backend,
        settings=settings,
        sensor_id="test-sensor",
        location_provider=lambda: {"latitude": 35.0, "longitude": -106.0},
    )

    fired = mgr._maybe_auto_capture_flats()
    assert fired is False


def test_mark_flats_capture_complete_persists(adapter_stub, library, logger):
    backend = _FakeFlatBackend()
    settings = MagicMock()
    settings.update_and_save = MagicMock()

    mgr = _make_manager(
        adapter_stub,
        library,
        logger,
        flat_backend=backend,
        settings=settings,
    )

    mgr.mark_flats_capture_complete("2026-04-29T18:30:00")
    settings.update_and_save.assert_called_once()
    call_arg = settings.update_and_save.call_args[0][0]
    assert call_arg == {"sensors": [{"id": "test-sensor", "last_flats_capture_iso": "2026-04-29T18:30:00"}]}
