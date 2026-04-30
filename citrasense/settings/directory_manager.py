"""Centralized directory resolution for CitraSense data, images, processing, logs, and cache."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import platformdirs

from citrasense.constants import APP_AUTHOR, APP_NAME


class DirectoryManager:
    """Owns all directory paths — resolves custom overrides vs platformdirs defaults.

    Created once during ``CitraSenseSettings.load()`` and stored on the
    settings instance.  Every other layer (daemon, web, queues) reads
    paths through this object rather than computing them independently.
    """

    def __init__(self, custom_data_dir: str = "", custom_log_dir: str = "", custom_cache_dir: str = "") -> None:
        self._data_dir = Path(custom_data_dir) if custom_data_dir else self.default_data_dir()
        self._log_dir = Path(custom_log_dir) if custom_log_dir else self.default_log_dir()
        self._cache_dir = Path(custom_cache_dir) if custom_cache_dir else self.default_cache_dir()

    # ── Path properties ───────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        """Base data directory (parent of images/ and processing/)."""
        return self._data_dir

    @property
    def images_dir(self) -> Path:
        return self._data_dir / "images"

    @property
    def processing_dir(self) -> Path:
        return self._data_dir / "processing"

    @property
    def analysis_dir(self) -> Path:
        return self._data_dir / "analysis"

    @property
    def analysis_previews_dir(self) -> Path:
        return self._data_dir / "analysis" / "previews"

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def cache_dir(self) -> Path:
        """Base cache directory (for temp files that can be regenerated)."""
        return self._cache_dir

    # ── Derived paths (used by subsystems) ────────────────────────────

    @property
    def elset_cache_path(self) -> Path:
        """TLE/elset cache JSON file."""
        return self._data_dir / "processing" / "elset_cache.json"

    @property
    def catalogs_dir(self) -> Path:
        """Star catalogs (APASS, etc.)."""
        return self._data_dir / "catalogs"

    @property
    def calibration_dir(self) -> Path:
        """Calibration master frames library."""
        return self._data_dir / "calibration"

    @property
    def kstars_cache_dir(self) -> Path:
        """KStars adapter temp sequence/job files."""
        return self._cache_dir / "kstars"

    # ── Derived paths (log) ───────────────────────────────────────────

    def current_log_path(self) -> Path:
        """Today's dated log file path."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._log_dir / f"citrasense-{today}.log"

    # ── Directory creation ────────────────────────────────────────────

    def ensure_data_directories(self) -> None:
        """Create images directory if it doesn't exist."""
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def ensure_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ── Platform defaults (for UI placeholders) ───────────────────────

    @staticmethod
    def default_data_dir() -> Path:
        return Path(platformdirs.user_data_dir(APP_NAME, appauthor=APP_AUTHOR))

    @staticmethod
    def default_log_dir() -> Path:
        return Path(platformdirs.user_log_dir(APP_NAME, appauthor=APP_AUTHOR))

    @staticmethod
    def default_cache_dir() -> Path:
        return Path(platformdirs.user_cache_dir(APP_NAME, appauthor=APP_AUTHOR))
