"""In-memory image preview conversion for the web UI.

Converts raw camera pixel arrays into browser-renderable JPEG data URLs
with automatic stretch. No disk I/O.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def array_to_jpeg_data_url(
    data: np.ndarray,
    *,
    quality: int = 85,
    percentile_lo: float = 2.0,
    percentile_hi: float = 98.0,
) -> str:
    """Auto-stretch a 2-D pixel array and encode as a JPEG data URL.

    Args:
        data: 2-D (grayscale) or 3-D (H x W x C) numpy array of any dtype.
        quality: JPEG quality (1-100).
        percentile_lo: Low clip percentile for stretch.
        percentile_hi: High clip percentile for stretch.

    Returns:
        ``"data:image/jpeg;base64,..."`` string ready for ``<img src=...>``.
    """
    arr = np.asarray(data, dtype=np.float64)

    # For multi-channel images, stretch on luminance but keep all channels
    if arr.ndim == 3:
        # Convert to grayscale for stretch reference, apply per-channel
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
