"""Color-channel ordering tests for :class:`UsbCamera`.

OpenCV's :meth:`VideoCapture.read` returns frames in BGR order, but every
other camera adapter and downstream consumer in CitraSense (Pillow,
matplotlib, the allsky preview bus, FITS pipelines) speaks RGB.  Without
an explicit BGR→RGB swap inside :meth:`UsbCamera.capture_array`, the
JPEG encoder in :func:`array_to_jpeg_data_url` renders skies and skin
tones with red and blue exchanged — the bug operators saw on the allsky
USB live preview.

These tests stub out :class:`cv2.VideoCapture` with a hand-crafted frame
in BGR order so we can prove the public ``capture_array`` contract: the
returned 3-channel array MUST be RGB.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from citrasense.hardware.devices.camera.usb_camera import UsbCamera  # noqa: E402


def _make_connected_camera_with_bgr_frame(bgr_frame: np.ndarray) -> UsbCamera:
    """Build a UsbCamera whose ``VideoCapture.read`` returns *bgr_frame*.

    Bypasses the real ``connect()`` so the test stays unit-level: we
    inject the cv2 module and a stub VideoCapture directly.
    """
    cam = UsbCamera(logger=logging.getLogger("test.usb_camera"))
    cam._cv2_module = cv2
    fake_capture = MagicMock()
    fake_capture.isOpened.return_value = True
    fake_capture.read.return_value = (True, bgr_frame)
    cam._camera = fake_capture
    cam._connected = True
    return cam


def test_capture_array_returns_rgb_not_bgr() -> None:
    """A pure-blue BGR frame must come back as RGB with blue in channel 2."""
    h, w = 4, 4
    bgr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    bgr_frame[..., 0] = 255  # cv2 BGR → blue is channel 0

    cam = _make_connected_camera_with_bgr_frame(bgr_frame)
    rgb_frame = cam.capture_array(duration=0.0)

    assert rgb_frame.shape == (h, w, 3)
    assert rgb_frame.dtype == np.uint8
    # After the BGR→RGB swap, blue must live in channel 2.
    assert int(rgb_frame[0, 0, 0]) == 0, "channel 0 (R) should be 0"
    assert int(rgb_frame[0, 0, 1]) == 0, "channel 1 (G) should be 0"
    assert int(rgb_frame[0, 0, 2]) == 255, "channel 2 (B) should be 255"


def test_capture_array_preserves_red_and_green() -> None:
    """Distinct R and G channels survive the swap with their values intact."""
    h, w = 2, 2
    bgr_frame = np.zeros((h, w, 3), dtype=np.uint8)
    bgr_frame[..., 0] = 10  # B
    bgr_frame[..., 1] = 128  # G (untouched by swap)
    bgr_frame[..., 2] = 200  # R

    cam = _make_connected_camera_with_bgr_frame(bgr_frame)
    rgb = cam.capture_array(duration=0.0)

    assert int(rgb[0, 0, 0]) == 200, "RGB channel 0 should be the original R"
    assert int(rgb[0, 0, 1]) == 128, "G is untouched"
    assert int(rgb[0, 0, 2]) == 10, "RGB channel 2 should be the original B"


def test_capture_array_passes_through_grayscale_frames() -> None:
    """A 2-D frame from a mono USB camera shouldn't be touched by the swap.

    cvtColor only runs when the frame has three channels; a grayscale
    capture must round-trip pixel-identically.
    """
    gray_frame = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    cam = _make_connected_camera_with_bgr_frame(gray_frame)
    out = cam.capture_array(duration=0.0)

    assert out.shape == (2, 2)
    assert np.array_equal(out, gray_frame)
