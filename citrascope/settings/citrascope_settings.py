"""CitraScope settings as a Pydantic BaseModel.

Each persisted setting is a single field declaration (type + default).
Serialization via ``model_dump()`` and loading via ``model_validate()``
eliminate the need to maintain separate field lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import platformdirs
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

# Application constants for platformdirs
# Defined before imports to avoid circular dependency
APP_NAME = "citrascope"
APP_AUTHOR = "citra-space"

from citrascope.constants import DEFAULT_API_PORT, DEFAULT_WEB_PORT, PROD_API_HOST
from citrascope.logging import CITRASCOPE_LOGGER
from citrascope.settings.settings_file_manager import SettingsFileManager


class CitraScopeSettings(BaseModel):
    """Settings for CitraScope loaded from JSON configuration file.

    Each field below is defined once — type, default, and optional validator.
    ``model_dump()`` auto-generates the serialization dict; ``model_validate()``
    handles loading from a config dict with defaults for missing keys.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # ── Persisted settings ────────────────────────────────────────────
    # API
    host: str = PROD_API_HOST
    port: int = DEFAULT_API_PORT
    use_ssl: bool = True
    personal_access_token: str = ""
    telescope_id: str = ""
    use_dummy_api: bool = False

    # Hardware
    hardware_adapter: str = ""

    # Runtime / UI-configurable
    log_level: str = "INFO"
    keep_images: bool = False
    keep_processing_output: bool = False

    # Processors
    processors_enabled: bool = True
    enabled_processors: dict[str, bool] = Field(default_factory=dict)

    # Task retry
    max_task_retries: int = 3
    initial_retry_delay_seconds: int = 30
    max_retry_delay_seconds: int = 300

    # Logging
    file_logging_enabled: bool = True
    log_retention_days: int = 30

    # Autofocus
    scheduled_autofocus_enabled: bool = False
    autofocus_interval_minutes: int = 60
    last_autofocus_timestamp: int | None = None
    autofocus_target_preset: str = "mirach"
    autofocus_target_custom_ra: float | None = None
    autofocus_target_custom_dec: float | None = None

    # Alignment
    alignment_exposure_seconds: float = 2.0
    align_on_startup: bool = False
    last_alignment_timestamp: int | None = None

    # Time synchronization
    time_check_interval_minutes: int = 5
    time_offset_pause_ms: float = 500.0

    # GPS
    gps_location_updates_enabled: bool = True
    gps_update_interval_minutes: int = 5

    # Task processing
    task_processing_paused: bool = False

    # Observation mode: "auto", "tracking", or "static"
    observation_mode: str = "auto"

    # MSI / elset cache
    elset_refresh_interval_hours: float = 6

    # ── Non-persisted public attrs (excluded from model_dump) ─────────
    web_port: int = Field(default=DEFAULT_WEB_PORT, exclude=True)
    adapter_settings: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # ── Private infrastructure ────────────────────────────────────────
    _config_manager: SettingsFileManager = PrivateAttr()
    _images_dir: Path = PrivateAttr()
    _all_adapter_settings: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    # ── Validators (warn-and-fallback, never raise) ───────────────────

    @field_validator("autofocus_target_custom_ra", mode="before")
    @classmethod
    def _validate_custom_ra(cls, v: Any) -> float | None:
        if v is not None:
            try:
                v = float(v)
            except (ValueError, TypeError):
                CITRASCOPE_LOGGER.warning("Invalid autofocus_target_custom_ra (%s). Clearing.", v)
                return None
            if not (0 <= v <= 360):
                CITRASCOPE_LOGGER.warning("Invalid autofocus_target_custom_ra (%s). Clearing.", v)
                return None
        return v

    @field_validator("autofocus_target_custom_dec", mode="before")
    @classmethod
    def _validate_custom_dec(cls, v: Any) -> float | None:
        if v is not None:
            try:
                v = float(v)
            except (ValueError, TypeError):
                CITRASCOPE_LOGGER.warning("Invalid autofocus_target_custom_dec (%s). Clearing.", v)
                return None
            if not (-90 <= v <= 90):
                CITRASCOPE_LOGGER.warning("Invalid autofocus_target_custom_dec (%s). Clearing.", v)
                return None
        return v

    @field_validator("autofocus_interval_minutes", mode="before")
    @classmethod
    def _validate_autofocus_interval(cls, v: Any) -> int:
        try:
            v = int(v)
        except (ValueError, TypeError):
            CITRASCOPE_LOGGER.warning("Invalid autofocus_interval_minutes (%s). Setting to default 60 minutes.", v)
            return 60
        if v < 1 or v > 1439:
            CITRASCOPE_LOGGER.warning("Invalid autofocus_interval_minutes (%s). Setting to default 60 minutes.", v)
            return 60
        return v

    @field_validator("observation_mode", mode="before")
    @classmethod
    def _validate_observation_mode(cls, v: Any) -> str:
        if v not in ("auto", "tracking", "static"):
            CITRASCOPE_LOGGER.warning("Invalid observation_mode (%r). Falling back to 'auto'.", v)
            return "auto"
        return v

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, web_port: int = DEFAULT_WEB_PORT) -> CitraScopeSettings:
        """Load settings from the JSON config file on disk.

        Args:
            web_port: Port for web interface (CLI bootstrap option only).
        """
        mgr = SettingsFileManager()
        config = mgr.load_config()

        all_adapter_settings: dict[str, dict[str, Any]] = config.pop("adapter_settings", {})
        config["web_port"] = web_port

        instance = cls.model_validate(config)
        instance._config_manager = mgr
        instance._images_dir = Path(platformdirs.user_data_dir(APP_NAME, appauthor=APP_AUTHOR)) / "images"
        instance._all_adapter_settings = all_adapter_settings
        instance.adapter_settings = all_adapter_settings.get(instance.hardware_adapter, {})
        return instance

    # ── Public helpers ────────────────────────────────────────────────

    @property
    def config_manager(self) -> SettingsFileManager:
        """Access the underlying file manager (for path queries, etc.)."""
        return self._config_manager

    def get_images_dir(self) -> Path:
        """Get the path to the images directory."""
        return self._images_dir

    def ensure_images_directory(self) -> None:
        """Create images directory if it doesn't exist."""
        if not self._images_dir.exists():
            self._images_dir.mkdir(parents=True)

    def is_configured(self) -> bool:
        """Check if minimum required configuration is present."""
        return bool(self.personal_access_token and self.telescope_id and self.hardware_adapter)

    # ── Serialization & persistence ───────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for serialization.

        Returns:
            Dictionary of all persisted settings (excludes runtime-only
            ``web_port`` and ``adapter_settings`` which are handled separately).
        """
        d = self.model_dump()
        d["adapter_settings"] = self._all_adapter_settings
        return d

    def save(self) -> None:
        """Save current settings to JSON config file."""
        if self.hardware_adapter:
            self._all_adapter_settings[self.hardware_adapter] = self.adapter_settings

        self._config_manager.save_config(self.to_dict())
        CITRASCOPE_LOGGER.info("Configuration saved to %s", self._config_manager.get_config_path())

    def update_and_save(self, config: dict[str, Any]) -> None:
        """Update settings from dict and save, preserving other adapters' settings.

        Merges the incoming *config* on top of the current ``to_dict()`` so
        that fields not present in the incoming payload are preserved (e.g.
        backend-only settings that the web UI never sends).

        Args:
            config: Configuration dict with flat adapter_settings for current adapter.
        """
        config.pop("web_port", None)

        adapter = config.get("hardware_adapter", self.hardware_adapter)
        if adapter:
            self._all_adapter_settings[adapter] = config.get("adapter_settings", {})
        config["adapter_settings"] = self._all_adapter_settings

        merged = self.to_dict()
        merged.update(config)

        allowed_keys = {k for k, v in type(self).model_fields.items() if not v.exclude} | {"adapter_settings"}
        merged = {k: v for k, v in merged.items() if k in allowed_keys}

        self._config_manager.save_config(merged)
