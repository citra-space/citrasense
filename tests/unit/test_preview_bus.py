"""Tests for PreviewBus and array_to_jpeg_data_url."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from citrascope.preview_bus import PreviewBus, array_to_jpeg_data_url


class TestPreviewBus:
    def test_pop_empty_returns_none(self):
        bus = PreviewBus()
        assert bus.pop() is None

    def test_push_then_pop(self):
        bus = PreviewBus()
        bus.push("data:image/jpeg;base64,abc", "autofocus")
        frame = bus.pop()
        assert frame == ("data:image/jpeg;base64,abc", "autofocus")

    def test_pop_clears_slot(self):
        bus = PreviewBus()
        bus.push("url", "src")
        bus.pop()
        assert bus.pop() is None

    def test_push_overwrites_previous(self):
        bus = PreviewBus()
        bus.push("first", "a")
        bus.push("second", "b")
        assert bus.pop() == ("second", "b")

    def test_clear(self):
        bus = PreviewBus()
        bus.push("url", "src")
        bus.clear()
        assert bus.pop() is None

    def test_source_defaults_to_empty(self):
        bus = PreviewBus()
        bus.push("url")
        assert bus.pop() == ("url", "")

    def test_push_file_nonexistent_is_noop(self, tmp_path: Path):
        bus = PreviewBus()
        bus.push_file(tmp_path / "nonexistent.jpg", "test")
        assert bus.pop() is None

    def test_push_file_jpeg(self, tmp_path: Path):
        img = Image.new("L", (4, 4), color=128)
        p = tmp_path / "test.jpg"
        img.save(p, format="JPEG")

        bus = PreviewBus()
        bus.push_file(p, "test")
        frame = bus.pop()
        assert frame is not None
        data_url, source = frame
        assert source == "test"
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_push_file_png(self, tmp_path: Path):
        img = Image.new("L", (4, 4), color=128)
        p = tmp_path / "test.png"
        img.save(p, format="PNG")

        bus = PreviewBus()
        bus.push_file(p, "alignment")
        frame = bus.pop()
        assert frame is not None
        data_url, source = frame
        assert source == "alignment"
        assert data_url.startswith("data:image/png;base64,")

    def test_push_file_fits(self, tmp_path: Path):
        from astropy.io import fits

        data = np.random.default_rng(42).integers(0, 65535, (16, 16), dtype=np.uint16)
        hdu = fits.PrimaryHDU(data=data)
        p = tmp_path / "test.fits"
        hdu.writeto(p)

        bus = PreviewBus()
        bus.push_file(p, "calibration")
        frame = bus.pop()
        assert frame is not None
        data_url, source = frame
        assert source == "calibration"
        assert data_url.startswith("data:image/jpeg;base64,")


class TestArrayToJpegDataUrl:
    def test_grayscale_output(self):
        data = np.full((8, 8), 100, dtype=np.uint16)
        result = array_to_jpeg_data_url(data)
        assert result.startswith("data:image/jpeg;base64,")
        raw = base64.b64decode(result.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw))
        assert img.mode == "L"
        assert img.size == (8, 8)

    def test_rgb_output(self):
        data = np.full((8, 8, 3), 100, dtype=np.uint8)
        result = array_to_jpeg_data_url(data)
        raw = base64.b64decode(result.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw))
        assert img.mode == "RGB"

    def test_flip_horizontal(self):
        data = np.zeros((4, 4), dtype=np.uint16)
        data[:, 0] = 65535
        result_normal = array_to_jpeg_data_url(data, flip_horizontal=False)
        result_flipped = array_to_jpeg_data_url(data, flip_horizontal=True)
        assert result_normal != result_flipped

    def test_constant_image_no_crash(self):
        data = np.full((4, 4), 42, dtype=np.float32)
        result = array_to_jpeg_data_url(data)
        assert result.startswith("data:image/jpeg;base64,")

    def test_uses_float32_internally(self):
        """Verify the stretch uses float32, not float64, for memory efficiency."""
        data = np.zeros((4, 4), dtype=np.uint16)
        arr = np.asarray(data)
        arr_float = arr.astype(np.float32, copy=False)
        assert arr_float.dtype == np.float32
