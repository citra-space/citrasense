"""CitraSense settings as a Pydantic BaseModel.

Each persisted setting is a single field declaration (type + default).
Serialization via ``model_dump()`` and loading via ``model_validate()``
eliminate the need to maintain separate field lists.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

# Current on-disk schema version. Incremented whenever the config file layout
# changes in an observer-visible way. Older files are migrated transparently
# in ``CitraSenseSettings.load``.
#
# v1: legacy scalar fields (``hardware_adapter`` / ``telescope_id``) plus a
#     top-level ``adapter_settings`` dict keyed by adapter name.
# v2: introduces ``sensors: list[SensorConfig]`` as the forward shape (phase 1
#     of citra-space/citrasense#306). Scalar fields kept as compatibility aliases.
# v3: ``sensors[]`` is the sole source of truth. Legacy scalars removed from
#     the model; ``load()`` migrates v1/v2 files automatically.
# v4: operational settings (task_processing_paused, observing_session_*,
#     self_tasking_*) moved from global to per-sensor.
# v8: ``hfr_baseline`` promoted from ``adapter_settings["hfr_baseline"]``
#     to a typed ``SensorConfig.hfr_baseline`` field. The post-autofocus
#     baseline is a per-sensor measurement, not an adapter setting.
CONFIG_VERSION = 8
# Legacy scalar-only shape — synthesized ``sensors`` entries are marked with
# this id so they round-trip cleanly on resave.
# Migration-only: default sensor id assigned when upgrading legacy single-sensor configs.
DEFAULT_TELESCOPE_SENSOR_ID = "telescope-0"

from citrasense.constants import DEFAULT_API_PORT, DEFAULT_WEB_PORT, PROD_API_HOST
from citrasense.logging import CITRASENSE_LOGGER
from citrasense.settings.directory_manager import DirectoryManager
from citrasense.settings.settings_file_manager import SettingsFileManager

_VALID_SEXTRACTOR_FILTERS = frozenset({"default", "gauss_1.5_3x3", "gauss_2.5_5x5", "tophat_3.0_3x3", "tophat_5.0_5x5"})


def _hoist_to_sensors(config: dict[str, Any], fields: tuple[str, ...]) -> None:
    """Move top-level legacy fields onto each sensor via ``setdefault``.

    Used by the v3→v4, v4→v5, v5→v6, and v6→v7 migration steps, which all
    share the same shape: pop a tuple of keys off the top-level config and
    copy them into every sensor entry (synthesizing a default telescope
    entry if the list is empty).
    """
    top = {k: config.pop(k) for k in fields if k in config}
    if not top:
        return
    sensors = config.get("sensors") or []
    if not sensors:
        sensors = [{"id": DEFAULT_TELESCOPE_SENSOR_ID, "type": "telescope"}]
        config["sensors"] = sensors
    for sd in sensors:
        for k, v in top.items():
            sd.setdefault(k, v)


class SensorConfig(BaseModel):
    """Per-sensor configuration block.

    Each sensor entry carries its own adapter type, connection settings,
    scheduling flags, and operational state.  The config list supports
    multiple simultaneous telescope, radar, or RF sensors.

    Fields:
        id: Unique local sensor id (e.g. ``"telescope-0"``). Used by the
            daemon to look up the live sensor via the sensor manager.
        type: Registry key resolved through
            :mod:`citrasense.sensors.sensor_registry`.
        adapter: For telescope sensors, the short adapter key (``"nina"``,
            ``"kstars"``, ``"indi"``, ``"direct"``, ``"dummy"``) resolved
            through :mod:`citrasense.hardware.adapter_registry`. Empty for
            sensor types that don't use the adapter registry.
        adapter_settings: Adapter-specific settings blob. Merged (not
            replaced) by :meth:`CitraSenseSettings.update_and_save` so
            partial web-form saves don't wipe keys like ``"filters"``.
        citra_sensor_id: Backend-side id that identifies this sensor to the
            Citra API (what ``CitraSenseSettings.telescope_id`` holds today
            for the single-telescope case).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    adapter: str = ""
    adapter_settings: dict[str, Any] = Field(default_factory=dict)
    citra_sensor_id: str = ""

    # Per-sensor operational settings (moved from global in config_version 4).
    task_processing_paused: bool = False

    observing_session_enabled: bool = False
    observing_session_sun_altitude_threshold: float = -12.0
    observing_session_do_pointing_calibration: bool = False
    observing_session_do_autofocus: bool = True
    observing_session_do_park: bool = True

    self_tasking_enabled: bool = False
    self_tasking_satellite_group_ids: list[str] = Field(default_factory=list)
    self_tasking_include_orbit_regimes: list[str] = Field(default_factory=list)
    self_tasking_exclude_object_types: list[str] = Field(default_factory=list)
    self_tasking_collection_type: str = "Track"

    # Per-sensor observation settings (moved from global in config_version 5).
    observation_mode: str = "auto"
    exposure_seconds: float = 2.0
    num_exposures: int = 3
    adaptive_exposure: bool = False
    adaptive_exposure_max_trail_pixels: float = 3.0
    adaptive_exposure_min_seconds: float = 0.1
    adaptive_exposure_max_seconds: float = 30.0

    # Per-sensor processing tuning + toggles (moved from global in config_version 5).
    processors_enabled: bool = True
    enabled_processors: dict[str, bool] = Field(default_factory=dict)
    skip_upload: bool = False
    plate_solve_timeout: int = 60
    astrometry_index_path: str = ""
    sextractor_detect_thresh: float = 5.0
    sextractor_detect_minarea: int = 3
    sextractor_filter_name: str = "default"

    # Per-sensor autofocus settings (moved from global in config_version 5).
    scheduled_autofocus_enabled: bool = False
    autofocus_schedule_mode: str = "interval"
    autofocus_interval_minutes: int = 60
    autofocus_after_sunset_offset_minutes: int = 60
    autofocus_target_preset: str = "mirach"
    autofocus_target_custom_ra: float | None = None
    autofocus_target_custom_dec: float | None = None
    autofocus_on_hfr_increase_enabled: bool = False
    autofocus_hfr_increase_percent: int = 30
    autofocus_hfr_sample_window: int = 5

    # Post-autofocus HFR baseline — the minimum HFR observed at the end of
    # the most recent autofocus run. Promoted from ``adapter_settings`` in
    # config_version 8 because it's a per-sensor measurement, not an
    # adapter-specific setting.
    hfr_baseline: float | None = None

    # Per-sensor timestamps (moved from global in config_version 6).
    last_autofocus_timestamp: int | None = None
    last_alignment_timestamp: int | None = None

    # Per-sensor hardware-dependent capture settings (moved from global in
    # config_version 7). Each rig has its own camera/optics, so defaults
    # like alignment exposure and calibration frame counts can't be shared.
    alignment_exposure_seconds: float = 2.0
    calibration_frame_count: int = 30
    flat_frame_count: int = 15

    @field_validator("observing_session_sun_altitude_threshold", mode="before")
    @classmethod
    def _validate_sun_altitude_threshold(cls, v: Any) -> float:
        valid = (-6.0, -12.0, -18.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return -12.0
        if v not in valid:
            return -12.0
        return v

    @field_validator("observation_mode", mode="before")
    @classmethod
    def _validate_sensor_observation_mode(cls, v: Any) -> str:
        if v == "static":
            return "sidereal"
        if v not in ("auto", "tracking", "sidereal"):
            return "auto"
        return v

    @field_validator("exposure_seconds", mode="before")
    @classmethod
    def _validate_sensor_exposure_seconds(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 2.0
        return max(0.01, min(300.0, v))

    @field_validator("num_exposures", mode="before")
    @classmethod
    def _validate_sensor_num_exposures(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 3
        return max(1, min(50, v))

    @field_validator("plate_solve_timeout", mode="before")
    @classmethod
    def _validate_sensor_plate_solve_timeout(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 60
        return max(10, min(300, v))

    @field_validator("sextractor_detect_thresh", mode="before")
    @classmethod
    def _validate_sensor_sextractor_thresh(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 5.0
        return max(1.0, min(20.0, v))

    @field_validator("sextractor_detect_minarea", mode="before")
    @classmethod
    def _validate_sensor_sextractor_minarea(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 3
        return max(1, min(50, v))

    @field_validator("sextractor_filter_name", mode="before")
    @classmethod
    def _validate_sensor_sextractor_filter(cls, v: Any) -> str:
        v = str(v) if v else "default"
        if v not in _VALID_SEXTRACTOR_FILTERS:
            return "default"
        return v

    @field_validator("autofocus_schedule_mode", mode="before")
    @classmethod
    def _validate_sensor_af_schedule_mode(cls, v: Any) -> str:
        if v not in ("interval", "after_sunset"):
            return "interval"
        return v

    @field_validator("autofocus_interval_minutes", mode="before")
    @classmethod
    def _validate_sensor_af_interval(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 60
        if v < 1 or v > 1440:
            return 60
        return v

    @field_validator("autofocus_hfr_increase_percent", mode="before")
    @classmethod
    def _validate_sensor_hfr_percent(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 30
        return max(10, min(200, v))

    @field_validator("autofocus_hfr_sample_window", mode="before")
    @classmethod
    def _validate_sensor_hfr_window(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 5
        return max(3, min(20, v))

    @field_validator("autofocus_target_custom_ra", mode="before")
    @classmethod
    def _validate_custom_ra(cls, v: Any) -> float | None:
        if v is not None:
            try:
                v = float(v)
            except (ValueError, TypeError):
                return None
            if not (0 <= v <= 360):
                return None
        return v

    @field_validator("autofocus_target_custom_dec", mode="before")
    @classmethod
    def _validate_custom_dec(cls, v: Any) -> float | None:
        if v is not None:
            try:
                v = float(v)
            except (ValueError, TypeError):
                return None
            if not (-90 <= v <= 90):
                return None
        return v

    @field_validator("alignment_exposure_seconds", mode="before")
    @classmethod
    def _validate_sensor_alignment_exposure(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 2.0
        return max(0.1, min(60.0, v))

    @field_validator("calibration_frame_count", mode="before")
    @classmethod
    def _validate_sensor_calibration_frame_count(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            CITRASENSE_LOGGER.warning("Invalid calibration_frame_count (%r). Falling back to 30.", v)
            return 30
        if v < 5 or v > 100:
            clamped = max(5, min(100, v))
            CITRASENSE_LOGGER.warning("calibration_frame_count %d out of range [5, 100]. Clamped to %d.", v, clamped)
            return clamped
        return v

    @field_validator("flat_frame_count", mode="before")
    @classmethod
    def _validate_sensor_flat_frame_count(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            CITRASENSE_LOGGER.warning("Invalid flat_frame_count (%r). Falling back to 15.", v)
            return 15
        if v < 5 or v > 50:
            clamped = max(5, min(50, v))
            CITRASENSE_LOGGER.warning("flat_frame_count %d out of range [5, 50]. Clamped to %d.", v, clamped)
            return clamped
        return v

    @model_validator(mode="after")
    def _validate_sensor_adaptive_range(self) -> SensorConfig:
        if self.adaptive_exposure_min_seconds > self.adaptive_exposure_max_seconds:
            self.adaptive_exposure_min_seconds = self.adaptive_exposure_max_seconds
        return self


class CitraSenseSettings(BaseModel):
    """Settings for CitraSense loaded from JSON configuration file.

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
    use_dummy_api: bool = False

    # Sensors — single source of truth for all sensor configuration.
    # Each entry describes one physical sensor (telescope, radar, RF, …).
    sensors: list[SensorConfig] = Field(default_factory=list)

    # Persisted config schema version. Missing/older values trigger the
    # one-shot legacy migration in :meth:`CitraSenseSettings.load`.
    config_version: int = CONFIG_VERSION

    # Runtime / UI-configurable
    log_level: str = "INFO"
    keep_images: bool = False

    # Global pipeline infrastructure (stays global in v5)
    processing_output_retention_hours: int = 0
    use_local_apass_catalog: bool = False
    max_task_retries: int = 3
    initial_retry_delay_seconds: int = 30
    max_retry_delay_seconds: int = 300

    # Logging
    file_logging_enabled: bool = True
    log_retention_days: int = 30

    # Custom directory overrides (empty = use platformdirs defaults)
    custom_data_dir: str = ""
    custom_log_dir: str = ""

    # Time synchronization
    time_check_interval_minutes: int = 5
    time_offset_pause_ms: float = 500.0

    # Hardware safety
    hardware_safety_check_enabled: bool = False

    # GPS
    gps_monitoring_enabled: bool = True
    gps_location_updates_enabled: bool = True
    gps_update_interval_minutes: int = 5

    # MSI / elset cache
    elset_refresh_interval_hours: float = 6

    # ── Non-persisted public attrs (excluded from model_dump) ─────────
    web_port: int = Field(default=DEFAULT_WEB_PORT, exclude=True)

    # ── Private infrastructure ────────────────────────────────────────
    _config_manager: SettingsFileManager = PrivateAttr()
    _dir_manager: DirectoryManager = PrivateAttr()
    # Serializes concurrent ``save()`` / ``update_and_save()`` calls so
    # per-sensor managers (autofocus, alignment, …) finishing near
    # simultaneously don't race on the config file. Module-level import so
    # the lock is shared across all instances in the process.
    _save_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    # ── Validators (only for fields that remain on CitraSenseSettings) ─

    @field_validator("processing_output_retention_hours", mode="before")
    @classmethod
    def _validate_processing_output_retention(cls, v: Any) -> int:
        if v is True:
            return -1
        if v is False:
            return 0
        try:
            v = int(v)
        except (TypeError, ValueError):
            CITRASENSE_LOGGER.warning("Invalid processing_output_retention_hours (%r). Falling back to 0.", v)
            return 0
        if v < -1:
            CITRASENSE_LOGGER.warning("processing_output_retention_hours (%d) below -1. Clamped to -1.", v)
            return -1
        return v

    @field_validator("custom_data_dir", "custom_log_dir", mode="before")
    @classmethod
    def _validate_custom_dir(cls, v: Any) -> str:
        if not v:
            return ""
        v = str(v)
        p = Path(v).expanduser().resolve()
        if not p.is_absolute():
            CITRASENSE_LOGGER.warning("custom dir path %r is not absolute. Ignoring.", v)
            return ""
        return str(p)

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, web_port: int = DEFAULT_WEB_PORT) -> CitraSenseSettings:
        """Load settings from the JSON config file on disk.

        Handles transparent migration from v1/v2 config files:

        * **v1** (scalars only): ``hardware_adapter``, ``telescope_id``, and a
          top-level ``adapter_settings`` dict keyed by adapter name are
          converted into a single ``sensors[0]`` entry.
        * **v2** (dual-write): ``sensors[]`` exists alongside scalars. The
          scalars are stripped; any ``adapter_settings`` archive entries are
          injected into the matching sensor if its own blob is empty.

        Args:
            web_port: Port for web interface (CLI bootstrap option only).
        """
        mgr = SettingsFileManager()
        config = mgr.load_config()

        # Pop the legacy nested adapter_settings archive (keyed by adapter name).
        all_adapter_settings: dict[str, dict[str, Any]] = config.pop("adapter_settings", {})
        config["web_port"] = web_port

        # Migrate legacy keep_processing_output bool → processing_output_retention_hours
        if "keep_processing_output" in config and "processing_output_retention_hours" not in config:
            old_val = config.pop("keep_processing_output")
            config["processing_output_retention_hours"] = -1 if old_val else 0
        else:
            config.pop("keep_processing_output", None)

        # ── v1/v2 → v3 migration ──────────────────────────────────────
        # TODO(multi-sensor cleanup — drop on or after CitraSense v1.0):
        # The v1/v2 → v3/v4/v5/v6 migration ladder below is only relevant
        # for operators upgrading from the pre-sensors-list era. Once the
        # 1.0 release notes warn users to upgrade through v0.x first, this
        # block and the ``_OPS_FIELDS`` / ``_V5_FIELDS`` tuples can be
        # removed — ``load()`` will then only need to handle v6→v7 (and
        # later) migrations.
        hw_adapter = config.pop("hardware_adapter", "")
        telescope_id = config.pop("telescope_id", "")

        sensors_raw: list[dict[str, Any]] | None = config.get("sensors")

        if not sensors_raw and hw_adapter:
            # v1: no sensors list — synthesize from legacy scalars.
            config["sensors"] = [
                {
                    "id": DEFAULT_TELESCOPE_SENSOR_ID,
                    "type": "telescope",
                    "adapter": hw_adapter,
                    "adapter_settings": all_adapter_settings.get(hw_adapter, {}),
                    "citra_sensor_id": telescope_id,
                }
            ]
        elif sensors_raw:
            # v2: sensors exist.  Inject adapter_settings from the archive
            # if the sensor's own blob is empty.
            for sd in sensors_raw:
                adapter = sd.get("adapter", "")
                if not sd.get("adapter_settings") and adapter and adapter in all_adapter_settings:
                    sd["adapter_settings"] = all_adapter_settings[adapter]

        # ── v3 → v4: move operational settings into each sensor ───────
        _hoist_to_sensors(
            config,
            (
                "task_processing_paused",
                "observing_session_enabled",
                "observing_session_sun_altitude_threshold",
                "observing_session_do_pointing_calibration",
                "observing_session_do_autofocus",
                "observing_session_do_park",
                "self_tasking_enabled",
                "self_tasking_satellite_group_ids",
                "self_tasking_include_orbit_regimes",
                "self_tasking_exclude_object_types",
                "self_tasking_collection_type",
            ),
        )

        # ── v4 → v5: move observation, processing tuning, and autofocus
        #    settings into each sensor ──────────────────────────────────
        _hoist_to_sensors(
            config,
            (
                # Observation
                "observation_mode",
                "exposure_seconds",
                "num_exposures",
                "adaptive_exposure",
                "adaptive_exposure_max_trail_pixels",
                "adaptive_exposure_min_seconds",
                "adaptive_exposure_max_seconds",
                # Processing tuning + toggles
                "processors_enabled",
                "enabled_processors",
                "skip_upload",
                "plate_solve_timeout",
                "astrometry_index_path",
                "sextractor_detect_thresh",
                "sextractor_detect_minarea",
                "sextractor_filter_name",
                # Autofocus
                "scheduled_autofocus_enabled",
                "autofocus_schedule_mode",
                "autofocus_interval_minutes",
                "autofocus_after_sunset_offset_minutes",
                "autofocus_target_preset",
                "autofocus_target_custom_ra",
                "autofocus_target_custom_dec",
                "autofocus_on_hfr_increase_enabled",
                "autofocus_hfr_increase_percent",
                "autofocus_hfr_sample_window",
            ),
        )

        # ── v5 → v6: move timestamps into each sensor ─────────────────
        _hoist_to_sensors(config, ("last_autofocus_timestamp", "last_alignment_timestamp"))

        # ── v6 → v7: move hardware-dependent capture settings into each
        # sensor. These are camera/optics-specific, so every rig needs its
        # own defaults.
        _hoist_to_sensors(
            config,
            (
                "alignment_exposure_seconds",
                "calibration_frame_count",
                "flat_frame_count",
            ),
        )

        # ── v7 → v8: promote ``hfr_baseline`` out of each sensor's
        # ``adapter_settings`` into a typed top-level field.  The baseline
        # is a per-sensor post-autofocus measurement, not an
        # adapter-specific setting, so it belongs next to the other
        # autofocus timestamps on ``SensorConfig``.
        for sd in config.get("sensors") or []:
            aset = sd.get("adapter_settings") or {}
            if "hfr_baseline" in aset:
                sd.setdefault("hfr_baseline", aset.pop("hfr_baseline"))

        instance = cls.model_validate(config)
        instance._config_manager = mgr
        instance._dir_manager = DirectoryManager(instance.custom_data_dir, instance.custom_log_dir)
        return instance

    # ── Public helpers ────────────────────────────────────────────────

    @property
    def config_manager(self) -> SettingsFileManager:
        """Access the underlying file manager (for path queries, etc.)."""
        return self._config_manager

    @property
    def directories(self) -> DirectoryManager:
        """Centralized directory paths (data, images, processing, logs)."""
        return self._dir_manager

    def is_configured(self) -> bool:
        """Check if minimum required configuration is present.

        For multi-sensor deployments this now evaluates *every* sensor —
        any sensor missing an adapter or a Citra-side sensor id counts as
        "not configured". That matches the setup-wizard's mental model:
        the UI should surface configuration gaps for every rig, not only
        the first.
        """
        if not self.personal_access_token or not self.sensors:
            return False
        return all(bool(sc.citra_sensor_id and sc.adapter) for sc in self.sensors)

    def get_sensor_config(self, sensor_id: str) -> SensorConfig | None:
        """Look up a sensor config by its local id."""
        for s in self.sensors:
            if s.id == sensor_id:
                return s
        return None

    # ── Serialization & persistence ───────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for serialization.

        Returns:
            Dictionary of all persisted settings (excludes runtime-only
            ``web_port``).  ``sensors[]`` is the canonical sensor config.
        """
        d = self.model_dump()
        d["config_version"] = CONFIG_VERSION
        return d

    def save(self) -> None:
        """Save current settings to JSON config file.

        Serialized via ``_save_lock`` so per-sensor managers finishing
        near-simultaneously (autofocus/alignment writing timestamps, etc.)
        don't interleave writes to the JSON file.
        """
        with self._save_lock:
            self._config_manager.save_config(self.to_dict())
        CITRASENSE_LOGGER.info("Configuration saved to %s", self._config_manager.get_config_path())

    def update_and_save(self, config: dict[str, Any]) -> None:
        """Update settings from dict and save.

        Merges the incoming *config* on top of the current ``to_dict()`` so
        that fields not present in the incoming payload are preserved (e.g.
        backend-only settings that the web UI never sends).

        Per-sensor ``adapter_settings`` are shallow-merged so partial saves
        from the web form don't wipe keys like ``"filters"`` that the form
        never sends.

        Args:
            config: Configuration dict.  May contain ``sensors[]`` with
                per-sensor config including ``adapter_settings``.
        """
        config.pop("web_port", None)
        # Strip legacy scalars that old clients may still send.
        config.pop("hardware_adapter", None)
        config.pop("telescope_id", None)

        # Smart-merge per-sensor adapter_settings.
        incoming_sensors: list[dict[str, Any]] | None = config.get("sensors")
        if incoming_sensors and self.sensors:
            existing_by_id = {s.id: s for s in self.sensors}
            for sd in incoming_sensors:
                existing = existing_by_id.get(sd.get("id", ""))
                if existing:
                    merged_as = {**existing.adapter_settings, **sd.get("adapter_settings", {})}
                    sd["adapter_settings"] = merged_as

        # Top-level ``adapter_settings`` blobs from pre-multi-sensor clients
        # are no longer honoured.  The shim that funnelled them into
        # ``sensors[0]`` was wrong for multi-rig deployments and has been
        # removed — the in-tree web UI has shipped per-sensor
        # ``sensors[].adapter_settings`` for several releases.  We still
        # pop the key so it doesn't leak into the persisted config, but
        # we drop it on the floor with a warning so stale clients are
        # loudly told to update.
        if "adapter_settings" in config:
            import logging

            config.pop("adapter_settings", None)
            logging.getLogger("citrasense.Settings").warning(
                "Ignoring deprecated top-level 'adapter_settings' in config update; "
                "update the client to send sensors[].adapter_settings instead.",
            )

        with self._save_lock:
            merged = self.to_dict()
            merged.update(config)

            allowed_keys = {k for k, v in type(self).model_fields.items() if not v.exclude}
            merged = {k: v for k, v in merged.items() if k in allowed_keys}
            merged["config_version"] = CONFIG_VERSION

            self._config_manager.save_config(merged)
