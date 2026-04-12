"""Thread-safe single-slot bus for pushing preview images to the web UI.

Any backend component can call ``push(data_url, source)`` and the image
will be broadcast to all connected WebSocket clients within ~1 second.
Only the latest frame is kept — a new push overwrites any unpopped frame.

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
    arr = np.asarray(data, dtype=np.float64)
    if flip_horizontal:
        arr = np.fliplr(arr)

    if arr.ndim == 3:
        gray = np.mean(arr, axis=2)
        lo = np.percentile(gray, percentile_lo)
        hi = np.percentile(gray, percentile_hi)
    else:
        lo = np.percentile(arr, percentile_lo)
        hi = np.percentile(arr, percentile_hi)

    if hi <= lo:
        hi = lo + 1.0

    stretched = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    if stretched.ndim == 2:
        img = Image.fromarray(stretched, mode="L")
    else:
        img = Image.fromarray(stretched, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class PreviewBus:
    """Single-slot preview image bus.

    Thread-safe: producers call :meth:`push` from any thread; the web
    server's broadcast loop calls :meth:`pop` from the asyncio event loop
    thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: tuple[str, str] | None = None

    def push(self, data_url: str, source: str = "") -> None:
        """Store a preview frame, overwriting any previous unpopped frame."""
        with self._lock:
            self._frame = (data_url, source)

    def pop(self) -> tuple[str, str] | None:
        """Return and clear the current frame, or None if empty."""
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def clear(self) -> None:
        """Discard the current frame without returning it."""
        with self._lock:
            self._frame = None

    def push_file(self, path: str | Path, source: str = "") -> None:
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
                self.push(array_to_jpeg_data_url(data), source)
        else:
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            self.push(f"data:{mime};base64,{b64}", source)
