"""Thread-safe per-sensor preview image bus for the web UI.

Any backend component can call ``push(data_url, source, sensor_id)`` and the
image will be broadcast to all connected WebSocket clients within ~1 second.
Each sensor_id maintains its own latest-frame slot so frames from different
sensors never overwrite each other.

Also contains :func:`array_to_jpeg_data_url`, a pure utility that
converts numpy pixel arrays into browser-renderable JPEG data URLs.
"""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image


def array_to_jpeg_data_url(
    data: np.ndarray,
    *,
    quality: int = 85,
    percentile_lo: float = 2.0,
    percentile_hi: float = 98.0,
    flip_horizontal: bool = False,
) -> str:
    """Auto-stretch a 2-D pixel array and encode as a JPEG data URL.

    Args:
        data: 2-D (grayscale) or 3-D (H x W x C) numpy array of any dtype.
        quality: JPEG quality (1-100).
        percentile_lo: Low clip percentile for stretch.
        percentile_hi: High clip percentile for stretch.
        flip_horizontal: Mirror left/right (corrects for diagonal mirrors
            in the optical path).

    Returns:
        ``"data:image/jpeg;base64,..."`` string ready for ``<img src=...>``.
    """
    arr = np.asarray(data)
    if flip_horizontal:
        arr = np.fliplr(arr)

    arr_float = arr.astype(np.float32, copy=False)
    if arr_float.ndim == 3:
        gray = np.mean(arr_float, axis=2, dtype=np.float32)
        lo = np.percentile(gray, percentile_lo)
        hi = np.percentile(gray, percentile_hi)
    else:
        lo = np.percentile(arr_float, percentile_lo)
        hi = np.percentile(arr_float, percentile_hi)

    if hi <= lo:
        hi = lo + 1.0

    stretched = np.clip((arr_float - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    if stretched.ndim == 2:
        img = Image.fromarray(stretched, mode="L")
    else:
        img = Image.fromarray(stretched, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Frame tuple: (payload, source, kind, sensor_id)
_Frame = tuple[str, str, str, str]


class PreviewBus:
    """Per-sensor preview image bus.

    Thread-safe: producers call :meth:`push` from any thread; the web
    server's broadcast loop calls :meth:`pop_all` from the asyncio event
    loop thread.

    Each ``sensor_id`` maintains its own latest-frame slot so frames from
    different sensors never overwrite each other.

    Two frame types are supported:

    * **data URL** (``push`` / ``push_file``): the frame payload *is* the
      image, base64-encoded.  Used for live camera previews where latency
      matters and frames are small.
    * **URL notification** (``push_url``): the frame payload is a plain
      HTTP URL.  The client fetches the image over a normal ``<img>``
      request, which can be cached and doesn't block the WebSocket.
      Used for annotated task images that are already on disk.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, _Frame] = {}

    def push(self, data_url: str, source: str = "", sensor_id: str = "") -> None:
        """Store a data-URL preview frame for *sensor_id*."""
        with self._lock:
            self._frames[sensor_id] = (data_url, source, "data", sensor_id)

    def push_url(self, url: str, source: str = "", sensor_id: str = "") -> None:
        """Store a URL notification frame for *sensor_id*."""
        with self._lock:
            self._frames[sensor_id] = (url, source, "url", sensor_id)

    def pop_all(self) -> list[_Frame]:
        """Return and clear all pending frames (one per sensor).

        Returns:
            List of ``(payload, source, kind, sensor_id)`` tuples.
        """
        with self._lock:
            frames = list(self._frames.values())
            self._frames.clear()
            return frames

    def pop(self) -> tuple[str, str, str] | None:
        """Legacy single-frame pop -- returns the first pending frame.

        Prefer :meth:`pop_all` for multi-sensor awareness.
        """
        with self._lock:
            if not self._frames:
                return None
            key = next(iter(self._frames))
            payload, source, kind, _sid = self._frames.pop(key)
            return (payload, source, kind)

    def clear(self) -> None:
        """Discard all pending frames."""
        with self._lock:
            self._frames.clear()

    def push_file(self, path: str | Path, source: str = "", sensor_id: str = "") -> None:
        """Read an image file from disk and push it as a data URL.

        Supports JPEG, PNG, and FITS formats. FITS files are auto-stretched
        via :func:`array_to_jpeg_data_url`.
        """
        p = Path(path)
        if not p.exists():
            return
        suffix = p.suffix.lower()
        if suffix in (".fits", ".fit"):
            from astropy.io import fits

            with fits.open(p) as hdul:
                primary = hdul[0]
                assert isinstance(primary, fits.PrimaryHDU)
                data = primary.data
            if data is not None:
                self.push(array_to_jpeg_data_url(data), source, sensor_id)
        else:
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            self.push(f"data:{mime};base64,{b64}", source, sensor_id)
