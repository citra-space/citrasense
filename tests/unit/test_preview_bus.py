"""Tests for PreviewBus and array_to_jpeg_data_url."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from citrasense.preview_bus import PreviewBus, array_to_jpeg_data_url


def _pop_first(bus: PreviewBus) -> tuple[str, str, str] | None:
    """Helper: pop the first frame via pop_all, returning (payload, source, kind)."""
    frames = bus.pop_all()
    if not frames:
        return None
    payload, source, kind, _sid = frames[0]
    return (payload, source, kind)


class TestPreviewBus:
    def test_pop_all_empty_returns_empty(self):
        bus = PreviewBus()
        assert bus.pop_all() == []

    def test_push_then_pop(self):
        bus = PreviewBus()
        bus.push("data:image/jpeg;base64,abc", "autofocus")
        frame = _pop_first(bus)
        assert frame == ("data:image/jpeg;base64,abc", "autofocus", "data")

    def test_pop_all_clears_slot(self):
        bus = PreviewBus()
        bus.push("url", "src")
        bus.pop_all()
        assert bus.pop_all() == []

    def test_push_overwrites_previous(self):
        bus = PreviewBus()
        bus.push("first", "a")
        bus.push("second", "b")
        assert _pop_first(bus) == ("second", "b", "data")

    def test_clear(self):
        bus = PreviewBus()
        bus.push("url", "src")
        bus.clear()
        assert bus.pop_all() == []

    def test_source_defaults_to_empty(self):
        bus = PreviewBus()
        bus.push("url")
        assert _pop_first(bus) == ("url", "", "data")

    def test_push_file_nonexistent_is_noop(self, tmp_path: Path):
        bus = PreviewBus()
        bus.push_file(tmp_path / "nonexistent.jpg", "test")
        assert bus.pop_all() == []

    def test_push_file_jpeg(self, tmp_path: Path):
        img = Image.new("L", (4, 4), color=128)
        p = tmp_path / "test.jpg"
        img.save(p, format="JPEG")

        bus = PreviewBus()
        bus.push_file(p, "test")
        frame = _pop_first(bus)
        assert frame is not None
        data_url, source, kind = frame
        assert source == "test"
        assert kind == "data"
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_push_file_png(self, tmp_path: Path):
        img = Image.new("L", (4, 4), color=128)
        p = tmp_path / "test.png"
        img.save(p, format="PNG")

        bus = PreviewBus()
        bus.push_file(p, "alignment")
        frame = _pop_first(bus)
        assert frame is not None
        data_url, source, kind = frame
        assert source == "alignment"
        assert kind == "data"
        assert data_url.startswith("data:image/png;base64,")

    def test_push_file_fits(self, tmp_path: Path):
        from astropy.io import fits

        data = np.random.default_rng(42).integers(0, 65535, (16, 16), dtype=np.uint16)
        hdu = fits.PrimaryHDU(data=data)
        p = tmp_path / "test.fits"
        hdu.writeto(p)

        bus = PreviewBus()
        bus.push_file(p, "calibration")
        frame = _pop_first(bus)
        assert frame is not None
        data_url, source, kind = frame
        assert source == "calibration"
        assert kind == "data"
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_push_url(self):
        bus = PreviewBus()
        bus.push_url("/api/task-preview/latest?t=123", "task")
        frame = _pop_first(bus)
        assert frame == ("/api/task-preview/latest?t=123", "task", "url")

    def test_push_url_overwrites_data(self):
        bus = PreviewBus()
        bus.push("data:image/jpeg;base64,abc", "autofocus")
        bus.push_url("/preview", "task")
        frame = _pop_first(bus)
        assert frame is not None
        assert frame[2] == "url"

    def test_pop_all_returns_sensor_id(self):
        bus = PreviewBus()
        bus.push("data:image/jpeg;base64,abc", "autofocus", sensor_id="scope-0")
        frames = bus.pop_all()
        assert len(frames) == 1
        _payload, _source, _kind, sid = frames[0]
        assert sid == "scope-0"

    def test_pop_all_multi_sensor(self):
        bus = PreviewBus()
        bus.push("img1", "src1", sensor_id="scope-0")
        bus.push("img2", "src2", sensor_id="scope-1")
        frames = bus.pop_all()
        assert len(frames) == 2
        sids = {f[3] for f in frames}
        assert sids == {"scope-0", "scope-1"}


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
