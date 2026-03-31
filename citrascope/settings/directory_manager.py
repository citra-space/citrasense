"""Centralized directory resolution for CitraScope data, images, processing, and logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import platformdirs

from citrascope.settings.citrascope_settings import APP_AUTHOR, APP_NAME


class DirectoryManager:
    """Owns all directory paths — resolves custom overrides vs platformdirs defaults.

    Created once during ``CitraScopeSettings.load()`` and stored on the
    settings instance.  Every other layer (daemon, web, queues) reads
    paths through this object rather than computing them independently.
    """

    def __init__(self, custom_data_dir: str = "", custom_log_dir: str = "") -> None:
        self._data_dir = Path(custom_data_dir) if custom_data_dir else self.default_data_dir()
        self._log_dir = Path(custom_log_dir) if custom_log_dir else self.default_log_dir()

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
    def log_dir(self) -> Path:
        return self._log_dir

    # ── Derived paths ─────────────────────────────────────────────────

    def current_log_path(self) -> Path:
        """Today's dated log file path."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._log_dir / f"citrascope-{today}.log"

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
