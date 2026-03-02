"""CitraScope settings class using JSON-based configuration."""

from pathlib import Path
from typing import Any

import platformdirs

# Application constants for platformdirs
# Defined before imports to avoid circular dependency
APP_NAME = "citrascope"
APP_AUTHOR = "citra-space"

from citrascope.constants import DEFAULT_API_PORT, DEFAULT_WEB_PORT, PROD_API_HOST
from citrascope.logging import CITRASCOPE_LOGGER
from citrascope.settings.settings_file_manager import SettingsFileManager


class CitraScopeSettings:
    """Settings for CitraScope loaded from JSON configuration file."""

    def __init__(self, web_port: int = DEFAULT_WEB_PORT):
        """Initialize settings from JSON config file.

        Args:
            web_port: Port for web interface (default: 24872) - bootstrap option only
        """
        self.config_manager = SettingsFileManager()

        # Load configuration from file
        config = self.config_manager.load_config()

        # Application data directories
        self._images_dir = Path(platformdirs.user_data_dir(APP_NAME, appauthor=APP_AUTHOR)) / "images"

        # API Settings (all loaded from config file)
        self.host: str = config.get("host", PROD_API_HOST)
        self.port: int = config.get("port", DEFAULT_API_PORT)
        self.use_ssl: bool = config.get("use_ssl", True)
        self.personal_access_token: str = config.get("personal_access_token", "")
        self.telescope_id: str = config.get("telescope_id", "")
        self.use_dummy_api: bool = config.get("use_dummy_api", False)

        # Hardware adapter selection
        self.hardware_adapter: str = config.get("hardware_adapter", "")

        # Hardware adapter-specific settings stored as nested dict per adapter
        # Format: {"adapter_name": {"setting_key": value, ...}, ...}
        self._all_adapter_settings: dict[str, dict[str, Any]] = config.get("adapter_settings", {})

        # Current adapter's settings slice
        self.adapter_settings: dict[str, Any] = self._all_adapter_settings.get(self.hardware_adapter, {})

        # Runtime settings (all loaded from config file, configurable via web UI)
        self.log_level: str = config.get("log_level", "INFO")
        self.keep_images: bool = config.get("keep_images", False)

        # Processor configuration
        self.processors_enabled: bool = config.get("processors_enabled", True)
        self.enabled_processors: dict[str, bool] = config.get("enabled_processors", {})

        # Web port: CLI-only, never loaded from or saved to config file
        self.web_port: int = web_port

        # Task retry configuration
        self.max_task_retries: int = config.get("max_task_retries", 3)
        self.initial_retry_delay_seconds: int = config.get("initial_retry_delay_seconds", 30)
        self.max_retry_delay_seconds: int = config.get("max_retry_delay_seconds", 300)

        # Log file configuration
        self.file_logging_enabled: bool = config.get("file_logging_enabled", True)
        self.log_retention_days: int = config.get("log_retention_days", 30)

        # Autofocus configuration (top-level/global settings)
        self.scheduled_autofocus_enabled: bool = config.get("scheduled_autofocus_enabled", False)
        self.autofocus_interval_minutes: int = config.get("autofocus_interval_minutes", 60)
        self.last_autofocus_timestamp: int | None = config.get("last_autofocus_timestamp")
        self.autofocus_target_preset: str = config.get("autofocus_target_preset", "mirach")
        self.autofocus_target_custom_ra: float | None = config.get("autofocus_target_custom_ra")
        self.autofocus_target_custom_dec: float | None = config.get("autofocus_target_custom_dec")

        # Validate custom RA/Dec ranges
        if self.autofocus_target_custom_ra is not None:
            if not (0 <= self.autofocus_target_custom_ra <= 360):
                CITRASCOPE_LOGGER.warning(
                    f"Invalid autofocus_target_custom_ra ({self.autofocus_target_custom_ra}). Clearing."
                )
                self.autofocus_target_custom_ra = None
        if self.autofocus_target_custom_dec is not None:
            if not (-90 <= self.autofocus_target_custom_dec <= 90):
                CITRASCOPE_LOGGER.warning(
                    f"Invalid autofocus_target_custom_dec ({self.autofocus_target_custom_dec}). Clearing."
                )
                self.autofocus_target_custom_dec = None

        # Validate autofocus interval
        if (
            not isinstance(self.autofocus_interval_minutes, int)
            or self.autofocus_interval_minutes < 1
            or self.autofocus_interval_minutes > 1439
        ):
            CITRASCOPE_LOGGER.warning(
                f"Invalid autofocus_interval_minutes ({self.autofocus_interval_minutes}). "
                "Setting to default 60 minutes."
            )
            self.autofocus_interval_minutes = 60

        # Alignment configuration
        self.alignment_exposure_seconds: float = config.get("alignment_exposure_seconds", 2.0)
        self.align_on_startup: bool = config.get("align_on_startup", False)
        self.last_alignment_timestamp: int | None = config.get("last_alignment_timestamp")

        # Time synchronization monitoring configuration (always enabled)
        self.time_check_interval_minutes: int = config.get("time_check_interval_minutes", 5)
        self.time_offset_pause_ms: float = config.get("time_offset_pause_ms", 500.0)

        # GPS location updates configuration
        self.gps_location_updates_enabled: bool = config.get("gps_location_updates_enabled", True)
        self.gps_update_interval_minutes: int = config.get("gps_update_interval_minutes", 5)

        # Task processing state (persisted so pause survives restarts)
        self.task_processing_paused: bool = config.get("task_processing_paused", False)

        # MSI Processor configuration
        # Note: Individual processor enable/disable is handled via enabled_processors dict (already exists)
        self.elset_refresh_interval_hours: float = config.get("elset_refresh_interval_hours", 6)

    def get_images_dir(self) -> Path:
        """Get the path to the images directory.

        Returns:
            Path object pointing to the images directory.
        """
        return self._images_dir

    def ensure_images_directory(self) -> None:
        """Create images directory if it doesn't exist."""
        if not self._images_dir.exists():
            self._images_dir.mkdir(parents=True)

    def is_configured(self) -> bool:
        """Check if minimum required configuration is present.

        Returns:
            True if personal_access_token, telescope_id, and hardware_adapter are set.
        """
        return bool(self.personal_access_token and self.telescope_id and self.hardware_adapter)

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for serialization.

        Returns:
            Dictionary of all settings (excluding runtime-only settings like web_port).
        """
        return {
            "host": self.host,
            "port": self.port,
            "use_ssl": self.use_ssl,
            "personal_access_token": self.personal_access_token,
            "telescope_id": self.telescope_id,
            "use_dummy_api": self.use_dummy_api,
            "hardware_adapter": self.hardware_adapter,
            "adapter_settings": self._all_adapter_settings,
            "log_level": self.log_level,
            "keep_images": self.keep_images,
            "max_task_retries": self.max_task_retries,
            "initial_retry_delay_seconds": self.initial_retry_delay_seconds,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "file_logging_enabled": self.file_logging_enabled,
            "log_retention_days": self.log_retention_days,
            "scheduled_autofocus_enabled": self.scheduled_autofocus_enabled,
            "autofocus_interval_minutes": self.autofocus_interval_minutes,
            "last_autofocus_timestamp": self.last_autofocus_timestamp,
            "autofocus_target_preset": self.autofocus_target_preset,
            "autofocus_target_custom_ra": self.autofocus_target_custom_ra,
            "autofocus_target_custom_dec": self.autofocus_target_custom_dec,
            "alignment_exposure_seconds": self.alignment_exposure_seconds,
            "align_on_startup": self.align_on_startup,
            "last_alignment_timestamp": self.last_alignment_timestamp,
            "time_check_interval_minutes": self.time_check_interval_minutes,
            "time_offset_pause_ms": self.time_offset_pause_ms,
            "gps_location_updates_enabled": self.gps_location_updates_enabled,
            "gps_update_interval_minutes": self.gps_update_interval_minutes,
            "processors_enabled": self.processors_enabled,
            "enabled_processors": self.enabled_processors,
            "task_processing_paused": self.task_processing_paused,
            "elset_refresh_interval_hours": self.elset_refresh_interval_hours,
        }

    def save(self) -> None:
        """Save current settings to JSON config file."""
        # Update nested dict with current adapter's settings before saving
        if self.hardware_adapter:
            self._all_adapter_settings[self.hardware_adapter] = self.adapter_settings

        self.config_manager.save_config(self.to_dict())
        CITRASCOPE_LOGGER.info(f"Configuration saved to {self.config_manager.get_config_path()}")

    def update_and_save(self, config: dict[str, Any]) -> None:
        """Update settings from dict and save, preserving other adapters' settings.

        Args:
            config: Configuration dict with flat adapter_settings for current adapter.
        """
        # Remove runtime-only settings that should never be persisted
        config.pop("web_port", None)

        # Nest incoming adapter_settings under hardware_adapter key
        adapter = config.get("hardware_adapter", self.hardware_adapter)
        if adapter:
            self._all_adapter_settings[adapter] = config.get("adapter_settings", {})
        config["adapter_settings"] = self._all_adapter_settings

        self.config_manager.save_config(config)
