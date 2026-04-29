"""Unit tests for NinaTrainedFlatBackend."""

from __future__ import annotations

import base64
import io
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

from citrasense.calibration import FilterSlot
from citrasense.calibration.calibration_library import CalibrationLibrary
from citrasense.calibration.nina_trained_flat_backend import NinaTrainedFlatBackend


def _fits_payload() -> str:
    """Return a base64-encoded FITS file with a tiny uniform array."""
    hdu = fits.PrimaryHDU(np.full((8, 8), 30000, dtype=np.uint16))
    buf = io.BytesIO()
    hdu.writeto(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def library(tmp_path) -> CalibrationLibrary:
    return CalibrationLibrary(root=tmp_path / "calibration")


@pytest.fixture
def adapter_stub() -> MagicMock:
    a = MagicMock()
    a.nina_api_path = "http://nina:1888/v2/api"
    a.COMMAND_TIMEOUT = 30
    a.run_trained_flat = MagicMock()
    a.poll_flat_status = MagicMock(return_value={"State": "Finished", "TotalImageCount": 2})
    a.stop_flats = MagicMock(return_value=True)
    a.list_recent_flat_images = MagicMock(
        return_value=[
            {"Filename": "flat_0.fits", "Date": "2026-04-29T18:00:00", "_index": 10},
            {"Filename": "flat_1.fits", "Date": "2026-04-29T18:00:01", "_index": 11},
        ]
    )
    a.event_listener = MagicMock()
    a.event_listener.on_image_save = None
    return a


def test_happy_path_downloads_frames(library, adapter_stub):
    backend = NinaTrainedFlatBackend(adapter_stub, event_listener=adapter_stub.event_listener)

    def _start_flats_emits_saves(**_kwargs):
        # When NINA starts, emit two IMAGE-SAVE events to release the wait.
        cb = adapter_stub.event_listener.on_image_save
        cb({"ImageType": "FLAT", "Filename": "flat_0.fits"})
        cb({"ImageType": "FLAT", "Filename": "flat_1.fits"})

    adapter_stub.run_trained_flat.side_effect = _start_flats_emits_saves

    with patch("citrasense.calibration.nina_trained_flat_backend.requests.get") as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"Success": True, "Response": _fits_payload()}
        mock_get.return_value = response

        paths = backend.capture_flat_frames(
            filter_slot=FilterSlot(position=3, name="R"),
            count=2,
            gain=100,
            binning=1,
            initial_exposure=1.0,
            library=library,
            cancel_event=None,
            on_progress=None,
        )

    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.parent == library.tmp_dir

    adapter_stub.run_trained_flat.assert_called_once_with(filter_id=3, count=2, gain=100, binning=1)
    adapter_stub.list_recent_flat_images.assert_called_once()


def test_cancel_invokes_stop_flats(library, adapter_stub):
    backend = NinaTrainedFlatBackend(adapter_stub, event_listener=adapter_stub.event_listener)

    # Cancel before run_trained_flat to exit the loop immediately after start.
    cancel = threading.Event()

    def _start_then_cancel(**_kwargs):
        cancel.set()

    adapter_stub.run_trained_flat.side_effect = _start_then_cancel
    adapter_stub.list_recent_flat_images.return_value = []

    paths = backend.capture_flat_frames(
        filter_slot=FilterSlot(position=0, name="L"),
        count=10,
        gain=100,
        binning=1,
        initial_exposure=1.0,
        library=library,
        cancel_event=cancel,
        on_progress=None,
    )
    assert paths == []
    adapter_stub.stop_flats.assert_called()


def test_cancel_method_stops_running_capture(library, adapter_stub):
    backend = NinaTrainedFlatBackend(adapter_stub, event_listener=adapter_stub.event_listener)

    def _start_then_cancel(**_kwargs):
        backend.cancel()

    adapter_stub.run_trained_flat.side_effect = _start_then_cancel
    adapter_stub.list_recent_flat_images.return_value = []

    paths = backend.capture_flat_frames(
        filter_slot=FilterSlot(position=0, name="L"),
        count=5,
        gain=100,
        binning=1,
        initial_exposure=1.0,
        library=library,
        cancel_event=None,
        on_progress=None,
    )
    assert paths == []
    adapter_stub.stop_flats.assert_called()


def test_requires_filter_slot(library, adapter_stub):
    backend = NinaTrainedFlatBackend(adapter_stub, event_listener=adapter_stub.event_listener)
    with pytest.raises(ValueError, match="filter slot"):
        backend.capture_flat_frames(
            filter_slot=None,
            count=1,
            gain=0,
            binning=1,
            initial_exposure=0.1,
            library=library,
            cancel_event=None,
            on_progress=None,
        )


def test_raises_when_no_event_listener(library):
    adapter = MagicMock()
    adapter.event_listener = None
    backend = NinaTrainedFlatBackend(adapter, event_listener=None)
    with pytest.raises(RuntimeError, match="event listener"):
        backend.capture_flat_frames(
            filter_slot=FilterSlot(position=0, name="L"),
            count=1,
            gain=0,
            binning=1,
            initial_exposure=0.1,
            library=library,
            cancel_event=None,
            on_progress=None,
        )


def test_supported_frame_types(adapter_stub):
    backend = NinaTrainedFlatBackend(adapter_stub, event_listener=adapter_stub.event_listener)
    assert backend.supported_frame_types == {"flat"}


def test_status_polling_fallback_when_ws_silent(library, adapter_stub, monkeypatch):
    """If no IMAGE-SAVE events come in, /flats/status Finished should release the wait."""
    backend = NinaTrainedFlatBackend(adapter_stub, event_listener=adapter_stub.event_listener)
    # Shorten the internal wait interval so the test completes quickly.
    monkeypatch.setattr(backend, "WS_WAIT_INTERVAL", 0.01)
    monkeypatch.setattr(backend, "STATUS_POLL_INTERVAL", 0.0)

    adapter_stub.run_trained_flat.return_value = None  # no WS emission
    adapter_stub.poll_flat_status.return_value = {
        "State": "Finished",
        "TotalImageCount": 2,
    }

    with patch("citrasense.calibration.nina_trained_flat_backend.requests.get") as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"Success": True, "Response": _fits_payload()}
        mock_get.return_value = response

        paths = backend.capture_flat_frames(
            filter_slot=FilterSlot(position=0, name="L"),
            count=2,
            gain=0,
            binning=1,
            initial_exposure=1.0,
            library=library,
            cancel_event=None,
            on_progress=None,
        )

    assert len(paths) == 2
    adapter_stub.poll_flat_status.assert_called()
