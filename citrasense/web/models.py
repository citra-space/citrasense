"""Pydantic models used by the web layer."""

from typing import Any

from pydantic import BaseModel


class SystemStatus(BaseModel):
    """Current system status."""

    telescope_connected: bool = False
    camera_connected: bool = False
    supports_direct_camera_control: bool = False
    current_task: str | None = None
    tasks_pending: int = 0
    processing_active: bool = True
    automated_scheduling: bool = False
    hardware_adapter: str = "unknown"
    telescope_ra: float | None = None
    telescope_dec: float | None = None
    telescope_az: float | None = None
    telescope_alt: float | None = None
    ground_station_id: str | None = None
    ground_station_name: str | None = None
    ground_station_url: str | None = None
    ground_station_latitude: float | None = None
    ground_station_longitude: float | None = None
    ground_station_altitude: float | None = None
    location_source: str | None = None  # "gps" or "ground_station"
    location_latitude: float | None = None
    location_longitude: float | None = None
    location_altitude: float | None = None
    autofocus_requested: bool = False
    autofocus_running: bool = False
    autofocus_progress: str = ""
    autofocus_points: list[dict[str, int | float | str]] = []
    autofocus_filter_results: list[dict[str, str | int | float]] = []
    autofocus_last_result: str = ""
    autofocus_target_name: str = ""
    last_autofocus_timestamp: int | None = None
    next_autofocus_minutes: int | None = None
    hfr_history: list[dict[str, int | float | str]] = []
    last_hfr_median: float | None = None
    hfr_baseline: float | None = None
    hfr_increase_percent: int = 30
    hfr_refocus_enabled: bool = False
    hfr_sample_window: int = 5
    time_health: dict[str, Any] | None = None
    gpsd_fix: dict[str, Any] | None = None
    adapter_gps: dict[str, Any] | None = None
    last_update: str = ""
    # Entries: {device_type, device_name, missing_packages, install_cmd} — all str
    missing_dependencies: list[dict[str, str]] = []
    active_processors: list[str] = []  # Names of enabled image processors
    tasks_by_stage: dict[str, list[dict]] | None = None  # Tasks in each pipeline stage
    pipeline_stats: dict[str, Any] | None = None  # Lifetime counters for queues, processors, and tasks
    supports_alignment: bool = False
    supports_autofocus: bool = False
    supports_hardware_safety_monitor: bool = False
    supports_manual_sync: bool = False
    mount_at_home: bool = False
    mount_homing: bool = False
    mount_horizon_limit: int | None = None
    mount_overhead_limit: int | None = None
    # Telescope record's `minElevation` (degrees), distinct from the mount's
    # hardware altitude limit -- this is the user-defined minimum at which
    # the scope is willing to observe (trees, neighbor's lights, etc).
    telescope_min_elevation: float | None = None
    alignment_requested: bool = False
    alignment_running: bool = False
    alignment_progress: str = ""
    last_alignment_timestamp: int | None = None
    pointing_model: dict[str, Any] | None = None
    pointing_calibration_running: bool = False
    pointing_calibration_progress: dict[str, int] | None = None
    fov_short_deg: float | None = None
    camera_temperature: float | None = None
    current_filter_position: int | None = None
    current_filter_name: str | None = None
    focuser_connected: bool = False
    focuser_position: int | None = None
    focuser_max_position: int | None = None
    focuser_temperature: float | None = None
    focuser_moving: bool = False
    mount_tracking: bool = False
    mount_slewing: bool = False
    supports_direct_mount_control: bool = False
    safety_status: dict[str, Any] | None = None
    elset_health: dict[str, Any] | None = None
    latest_task_image_url: str | None = None
    calibration_status: dict[str, Any] | None = None
    config_health: dict[str, Any] | None = None
    status_collection_ms: float | None = None
    status_collection_breakdown: dict[str, float] | None = None
    # System busy guard — true when automated hardware operations are in progress
    system_busy: bool = False
    system_busy_reason: str = ""
    # Observing session / self-tasking
    observing_session_enabled: bool = False
    self_tasking_enabled: bool = False
    observing_session_state: str = "daytime"
    session_activity: str | None = None
    observing_session_threshold: float = -12.0
    sun_altitude: float | None = None
    dark_window_start: str | None = None
    dark_window_end: str | None = None
    last_batch_request: float | None = None
    last_batch_created: int | None = None
    next_request_seconds: float | None = None


class HardwareConfig(BaseModel):
    """Hardware configuration settings."""

    adapter: str
    indi_server_url: str | None = None
    indi_server_port: int | None = None
    indi_telescope_name: str | None = None
    indi_camera_name: str | None = None
    nina_url_prefix: str | None = None
