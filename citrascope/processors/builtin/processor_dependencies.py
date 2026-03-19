"""Dependency checking and shared utilities for MSI science processors."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def normalize_fits_timestamp(timestamp: str) -> str:
    """Truncate FITS DATE-OBS fractional seconds to 6 digits (microseconds).

    NINA on Windows writes DATE-OBS with 7 fractional digits using Windows
    FILETIME (100ns) resolution, e.g. "2025-11-12T01:38:11.1054519".
    Python 3.10's datetime.fromisoformat() only accepts up to 6 fractional
    digits; 3.11+ relaxed this restriction. Truncate here so any downstream
    fromisoformat() call is safe on all supported Python versions.
    """
    if timestamp and "." in timestamp:
        dot = timestamp.index(".")
        return timestamp[: dot + 7]  # dot + 6 digits = microseconds
    return timestamp


def check_pixelemon() -> bool:
    """Check if Pixelemon (Tetra3) plate solving is available.

    Returns:
        True if pixelemon can be imported and provides Telescope, TelescopeImage, TetraSolver
    """
    try:
        from pixelemon import Telescope, TelescopeImage, TetraSolver  # noqa: F401

        return True
    except Exception:
        return False


def check_all_dependencies(settings) -> dict:
    """Check for all required external tools and data files.

    Args:
        settings: CitraScopeSettings instance

    Returns:
        Dictionary with dependency check results
    """
    return {
        "pixelemon": check_pixelemon(),
        "sextractor": check_sextractor(),
    }


def check_sextractor() -> bool:
    """Check if SExtractor is installed.

    Returns:
        True if source-extractor or sex command is available
    """
    return shutil.which("source-extractor") is not None or shutil.which("sex") is not None


def read_source_catalog(catalog_path: Path) -> pd.DataFrame:
    """Read an output.cat source catalog, auto-detecting the format.

    Supports both the new 5-column layout written by PlateSolverProcessor
    (mag, magerr, ra, dec, elongation) and the legacy 11-column SExtractor
    layout (columns 4,5,8,9,10 → mag, magerr, ra, dec, elongation).
    """
    with open(catalog_path) as f:
        for line in f:
            if not line.startswith("#"):
                ncols = len(line.split())
                break
        else:
            ncols = 5

    if ncols <= 5:
        return pd.read_csv(
            catalog_path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=["mag", "magerr", "ra", "dec", "elongation"],
        )
    return pd.read_csv(
        catalog_path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[4, 5, 8, 9, 10],
        names=["mag", "magerr", "ra", "dec", "elongation"],
    )
