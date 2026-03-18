"""Calibration frame management — master frame library and builder."""

from __future__ import annotations

from typing import NamedTuple


class FilterSlot(NamedTuple):
    """A filter wheel position with its human-readable name."""

    position: int
    name: str
