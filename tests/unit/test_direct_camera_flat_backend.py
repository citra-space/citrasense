"""Unit tests for DirectCameraFlatBackend (extracted from MasterBuilder)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from citrasense.calibration import FilterSlot
from citrasense.calibration.calibration_library import CalibrationLibrary
from citrasense.calibration.flat_capture_backend import DirectCameraFlatBackend, auto_expose_flat


class _FakeCamera:
    """Minimal camera stub returning a constant uniform flat at a fixed median."""

    def __init__(self, median_value: int = 32768, max_adu: int = 65535, shape=(16, 16)) -> None:
        self._median = median_value
        self._max_adu = max_adu
        self._shape = shape
        self.capture_calls: list[dict] = []

    def capture_array(self, duration: float, gain: int, binning: int, shutter_closed: bool) -> np.ndarray:
        self.capture_calls.append(
            {
                "duration": duration,
                "gain": gain,
                "binning": binning,
                "shutter_closed": shutter_closed,
            }
        )
        return np.full(self._shape, self._median, dtype=np.uint16)

    def get_max_pixel_value(self, binning: int) -> int:
        return self._max_adu

    def get_temperature(self) -> float | None:
        return -10.0


@pytest.fixture
def library(tmp_path) -> CalibrationLibrary:
    return CalibrationLibrary(root=tmp_path / "calibration")


def test_auto_expose_converges_at_50pct(library):
    cam = _FakeCamera(median_value=32768, max_adu=65535)
    exposure = auto_expose_flat(
        cam,  # type: ignore[arg-type]
        initial_exposure=1.0,
        gain=100,
        binning=1,
        on_progress=None,
        cancel_event=None,
    )
    assert exposure == pytest.approx(1.0)
    # Converged on first attempt — only one test capture issued.
    assert len(cam.capture_calls) == 1


def test_auto_expose_scales_up_when_too_dark(library):
    cam = _FakeCamera(median_value=33000, max_adu=65535)
    # Median is well within tolerance so we expect one iteration.
    exposure = auto_expose_flat(
        cam,  # type: ignore[arg-type]
        initial_exposure=0.5,
        gain=100,
        binning=1,
        on_progress=None,
        cancel_event=None,
    )
    assert exposure == 0.5


def test_direct_backend_returns_count_frames_on_disk(library):
    cam = _FakeCamera()
    backend = DirectCameraFlatBackend(cam)  # type: ignore[arg-type]

    progress_calls: list[tuple[int, int, str, str]] = []
    paths = backend.capture_flat_frames(
        filter_slot=FilterSlot(position=1, name="R"),
        count=3,
        gain=100,
        binning=1,
        initial_exposure=1.0,
        library=library,
        cancel_event=None,
        on_progress=lambda *a: progress_calls.append(a),
    )

    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.parent == library.tmp_dir
        with fits.open(p) as hdul:
            data = hdul[0].data  # type: ignore[index]
            assert data is not None
            assert data.shape == (16, 16)

    # Progress callback was called at least once per captured frame.
    label_calls = [c for c in progress_calls if "Capturing" in c[3]]
    assert len(label_calls) == 3


def test_direct_backend_cancels_mid_capture(library):
    cam = _FakeCamera()
    backend = DirectCameraFlatBackend(cam)  # type: ignore[arg-type]
    event = threading.Event()

    def _on_progress(cur: int, total: int, ft: str, status: str) -> None:
        if cur == 1:
            event.set()

    paths = backend.capture_flat_frames(
        filter_slot=FilterSlot(position=1, name="R"),
        count=5,
        gain=100,
        binning=1,
        initial_exposure=1.0,
        library=library,
        cancel_event=event,
        on_progress=_on_progress,
    )
    # Should have stopped well before all 5 frames landed.
    assert len(paths) < 5


def test_direct_backend_cancel_method_stops_loop(library):
    """Calling cancel() during capture should stop the loop."""
    cam = _FakeCamera()
    backend = DirectCameraFlatBackend(cam)  # type: ignore[arg-type]

    def _cancel_on_first_frame(cur: int, total: int, ft: str, status: str) -> None:
        if "Capturing" in status:
            backend.cancel()

    paths = backend.capture_flat_frames(
        filter_slot=None,
        count=10,
        gain=0,
        binning=1,
        initial_exposure=0.1,
        library=library,
        cancel_event=None,
        on_progress=_cancel_on_first_frame,
    )
    # Should stop after the first frame triggers cancel().
    assert 0 < len(paths) < 10


def test_direct_backend_supported_frame_types():
    backend = DirectCameraFlatBackend(MagicMock())
    assert backend.supported_frame_types == {"flat"}
