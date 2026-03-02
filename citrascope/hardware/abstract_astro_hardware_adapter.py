from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from citrascope.safety.safety_monitor import SafetyMonitor


class SettingSchemaEntry(TypedDict, total=False):
    name: str
    friendly_name: str  # Human-readable display name for UI
    type: str  # e.g., 'float', 'int', 'str', 'bool'
    default: Any | None
    description: str
    required: bool  # Whether this field is required
    placeholder: str  # Placeholder text for UI inputs
    min: float  # Minimum value for numeric types
    max: float  # Maximum value for numeric types
    pattern: str  # Regex pattern for string validation
    options: list[str]  # List of valid options for select/dropdown inputs
    group: str  # Group name for organizing settings in UI (e.g., 'Camera', 'Mount', 'Advanced')
    visible_when: dict[str, str]  # Conditional visibility: {"field": "other_field_name", "value": "expected_value"}


class FilterConfig(TypedDict):
    """Type definition for filter configuration.

    Attributes:
        name: Human-readable filter name (e.g., 'Luminance', 'Red', 'Ha')
        focus_position: Focuser position for this filter in steps
        enabled: Whether this filter is enabled for observations (default: True)
    """

    name: str
    focus_position: int
    enabled: bool


class ObservationStrategy(Enum):
    MANUAL = 1
    SEQUENCE_TO_CONTROLLER = 2


class AbstractAstroHardwareAdapter(ABC):
    logger: logging.Logger  # Logger instance, must be provided by subclasses
    images_dir: Path  # Path to images directory, must be provided during initialization

    _slew_min_distance_deg: float = 2.0
    scope_slew_rate_degrees_per_second: float = 0.0
    telescope_record: dict | None = None
    DEFAULT_FOCUS_POSITION: int = 0  # Default focus position, can be overridden by subclasses

    def __init__(self, images_dir: Path, **kwargs):
        """Initialize the adapter with images directory and optional filter configuration.

        Args:
            images_dir: Path to the images directory
            **kwargs: Additional configuration including 'filters' dict
        """
        self.images_dir = images_dir
        self.filter_map = {}
        self._safety_monitor: SafetyMonitor | None = None

        # Load filter configuration from settings if available
        saved_filters = kwargs.get("filters", {})
        for filter_id, filter_data in saved_filters.items():
            try:
                # Default enabled to True for backward compatibility
                if "enabled" not in filter_data:
                    filter_data["enabled"] = True
                self.filter_map[int(filter_id)] = filter_data
            except (ValueError, TypeError):
                pass  # Skip invalid filter IDs

    @classmethod
    @abstractmethod
    def get_settings_schema(cls, **kwargs) -> list[SettingSchemaEntry]:
        """
        Return a schema describing configurable settings for this hardware adapter.

        Each setting is described as a SettingSchemaEntry TypedDict with keys:
            - name (str): The setting's name
            - type (str): The expected Python type (e.g., 'float', 'int', 'str', 'bool')
            - default (optional): The default value
            - description (str): Human-readable description of the setting

        Returns:
            list[SettingSchemaEntry]: List of setting schema entries.
        """
        pass

    def set_safety_monitor(self, monitor: SafetyMonitor) -> None:
        """Wire a SafetyMonitor for pre-action safety gates."""
        self._safety_monitor = monitor

    def point_telescope(self, ra: float, dec: float):
        """Point the telescope to the specified RA/Dec coordinates."""
        if self._safety_monitor and not self._safety_monitor.is_action_safe("slew", ra=ra, dec=dec):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Slew blocked by safety monitor")
        self._do_point_telescope(ra, dec)

    @abstractmethod
    def _do_point_telescope(self, ra: float, dec: float):
        """Hardware-specific implementation to point the telescope."""
        pass

    def angular_distance(
        self, ra1_degrees: float, dec1_degrees: float, ra2_degrees: float, dec2_degrees: float
    ) -> float:  # TODO: move this out of the hardware adapter... this isn't hardware stuff
        """Compute angular distance between two (RA deg, Dec deg) points in degrees."""

        # Convert to radians
        ra1_rad = math.radians(ra1_degrees)
        ra2_rad = math.radians(ra2_degrees)
        dec1_rad = math.radians(dec1_degrees)
        dec2_rad = math.radians(dec2_degrees)
        # Spherical law of cosines
        cos_angle = math.sin(dec1_rad) * math.sin(dec2_rad) + math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(
            ra1_rad - ra2_rad
        )
        # Clamp for safety
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    """
    Abstract base class for controlling astrophotography hardware.

    This adapter provides a common interface for interacting with telescopes, cameras,
    filter wheels, focus dials, and other astrophotography devices.
    """

    @abstractmethod
    def get_observation_strategy(self) -> ObservationStrategy:
        """Get the current observation strategy from the hardware."""
        pass

    @abstractmethod
    def perform_observation_sequence(self, task, satellite_data) -> str:
        """For hardware driven by sequences, perform the observation sequence and return image path."""
        pass

    def set_location_service(self, location_service) -> None:  # noqa: B027
        """Provide a location service for site-coordinate synchronisation.

        Called by the daemon after creating the adapter but before ``connect()``.
        Adapters that need site location (e.g. for mount initialisation) should
        store the reference; the default implementation is a no-op.
        """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the hardware server."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the hardware server."""
        pass

    @abstractmethod
    def is_telescope_connected(self) -> bool:
        """Check if telescope is connected and responsive."""
        pass

    @abstractmethod
    def is_camera_connected(self) -> bool:
        """Check if camera is connected and responsive."""
        pass

    @abstractmethod
    def list_devices(self) -> list[str]:
        """List all connected devices."""
        pass

    @abstractmethod
    def select_telescope(self, device_name: str) -> bool:
        """Select a specific camera by name."""
        pass

    @abstractmethod
    def get_telescope_direction(self) -> tuple[float, float]:
        """Read the current telescope direction (RA degrees, DEC degrees)."""
        pass

    def abort_slew(self) -> None:  # noqa: B027
        """Abort any in-progress slew. No-op if not supported or not slewing."""

    @abstractmethod
    def telescope_is_moving(self) -> bool:
        """Check if the telescope is currently moving."""
        pass

    @abstractmethod
    def select_camera(self, device_name: str) -> bool:
        """Select a specific camera by name."""
        pass

    @abstractmethod
    def take_image(self, task_id: str, exposure_duration_seconds=1.0) -> str:
        """Capture an image with the currently selected camera. Returns the file path of the saved image."""
        pass

    @abstractmethod
    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        """Set the tracking rate for the telescope in RA and Dec (arcseconds per second)."""
        pass

    @abstractmethod
    def get_tracking_rate(self) -> tuple[float, float]:
        """Get the current tracking rate for the telescope in RA and Dec (arcseconds per second)."""
        pass

    def perform_alignment(self, target_ra: float, target_dec: float) -> bool:
        """Perform plate-solve alignment to correct the mount's pointing model.

        Optional capability — override in adapters that manage alignment directly
        (e.g. direct hardware control).  Adapters that delegate to external
        software (NINA, KStars) can inherit this default which returns ``True``.

        Args:
            target_ra: Target Right Ascension in degrees.
            target_dec: Target Declination in degrees.

        Returns:
            True if alignment succeeded or is not applicable.
        """
        return True

    def update_from_plate_solve(  # noqa: B027
        self,
        solved_ra_deg: float,
        solved_dec_deg: float,
        expected_ra_deg: float | None = None,
        expected_dec_deg: float | None = None,
    ) -> None:
        """Update mount model from a plate solve result (pipeline → adapter).

        Called by the daemon/task manager when processing completes with plate solve
        data, so the adapter can apply pointing corrections (e.g. alignment offsets).
        Default implementation does nothing. Override in adapters that maintain
        a mount model.

        Args:
            solved_ra_deg: Solved right ascension in degrees.
            solved_dec_deg: Solved declination in degrees.
            expected_ra_deg: Optional target RA (e.g. from task/mount); if set with
                expected_dec_deg, adapter can compute offset = solved - expected.
            expected_dec_deg: Optional target Dec in degrees.
        """
        pass

    def home_if_needed(self) -> bool:
        """Home the mount if required, blocking until complete.

        Called after connect() and after the SafetyMonitor is online, so
        safety checks (e.g. CableWrapCheck) are actively monitoring during
        any physical motion.  Default is a no-op for adapters that don't
        need an explicit homing step.

        Returns:
            True if homing succeeded or was unnecessary.
        """
        return True

    def home_mount(self) -> bool:
        """Initiate the mount's homing routine.

        Override in adapters that support mount homing. Default raises
        NotImplementedError.

        Returns:
            True if homing was initiated successfully.
        """
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        raise NotImplementedError(f"{self.__class__.__name__} does not support mount homing")

    def is_mount_homed(self) -> bool:
        """Check whether the mount has been homed.

        Returns:
            True if the mount is at its home/calibrated position.
        """
        return False

    def get_mount_limits(self) -> tuple[int | None, int | None]:
        """Get the mount's altitude limits.

        Returns:
            (horizon_limit, overhead_limit) in integer degrees.
        """
        return None, None

    def set_mount_horizon_limit(self, degrees: int) -> bool:
        """Set the minimum altitude the mount will slew to.

        Returns:
            True if accepted, False if unsupported.
        """
        return False

    def set_mount_overhead_limit(self, degrees: int) -> bool:
        """Set the maximum altitude the mount will slew to.

        Returns:
            True if accepted, False if unsupported.
        """
        return False

    def do_autofocus(
        self,
        target_ra: float | None = None,
        target_dec: float | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Perform autofocus routine for all filters.

        This is an optional method for adapters that support filter management.
        Default implementation raises NotImplementedError. Override in subclasses
        that support autofocus.

        Args:
            target_ra: RA of the slew target in degrees (J2000), or None for adapter default
            target_dec: Dec of the slew target in degrees (J2000), or None for adapter default
            on_progress: Optional callback(str) to report progress updates

        Raises:
            NotImplementedError: If the adapter doesn't support autofocus
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support autofocus")

    def supports_autofocus(self) -> bool:
        """Indicates whether this adapter supports autofocus functionality.

        Returns:
            bool: True if the adapter can perform autofocus, False otherwise.
        """
        return False

    def supports_filter_management(self) -> bool:
        """Indicates whether this adapter supports filter/focus management.

        Returns:
            bool: True if the adapter manages filters and focus positions, False otherwise.
        """
        return False

    def expose_camera(self, exposure_time: float, **kwargs) -> str:
        """Capture a single exposure with direct camera control.

        Override in adapters that support manual camera operation (direct hardware).
        Default raises NotImplementedError.

        Args:
            exposure_time: Exposure duration in seconds
            **kwargs: Adapter-specific options (gain, offset, count, etc.)

        Returns:
            str: File path of the saved image
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support direct camera control")

    def get_missing_dependencies(self) -> list[dict[str, str]]:
        """Check for missing optional dependencies required by this adapter.

        Returns:
            list: List of dicts with keys device_type, device_name, missing_packages, install_cmd.
                  Empty list if all dependencies are met.
        """
        return []

    def supports_direct_camera_control(self) -> bool:
        """Indicates whether this adapter supports direct camera control.

        Direct camera control allows manual test captures and camera operations
        from the UI. Remote scheduler adapters (NINA, KStars) typically do not
        support this, as camera control is managed by the external software.
        Override to return True in adapters that implement expose_camera().

        Returns:
            bool: True if the adapter supports expose_camera() and manual captures
        """
        return False

    def is_hyperspectral(self) -> bool:
        """Indicates whether this adapter uses a hyperspectral camera.

        Hyperspectral cameras capture multiple spectral bands simultaneously
        (e.g., snapshot mosaic sensors) and do not require discrete filter changes.

        Returns:
            bool: True if using hyperspectral imaging, False otherwise (default)
        """
        return False

    def select_filters_for_task(self, task, allow_no_filter: bool = False) -> dict | None:
        """Select which filters to use for a task based on assignment.

        This method handles the common logic for filter selection:
        - If task specifies assigned_filter_name, find and validate that filter
        - If no filter specified, look for Clear/Luminance filter (case-insensitive)
        - Fall back to first enabled filter, or None if allow_no_filter=True

        Args:
            task: Task object with optional assigned_filter_name field
            allow_no_filter: If True, return None when no filters available (for KStars '--')

        Returns:
            dict: Dictionary mapping filter IDs to filter info {id: {name, focus_position, enabled}}
                  Returns None only if allow_no_filter=True and no suitable filter found

        Raises:
            RuntimeError: If assigned filter not found, disabled, or no filters available when required
        """
        # Task specifies a specific filter - find it
        if task and task.assigned_filter_name:
            target_filter_id = None
            target_filter_info = None
            for fid, fdata in self.filter_map.items():
                # Case-insensitive comparison
                if fdata["name"].lower() == task.assigned_filter_name.lower():
                    if not fdata.get("enabled", True):
                        raise RuntimeError(
                            f"Requested filter '{task.assigned_filter_name}' is disabled for task {task.id}"
                        )
                    target_filter_id = fid
                    target_filter_info = fdata
                    break

            if target_filter_id is None:
                raise RuntimeError(
                    f"Requested filter '{task.assigned_filter_name}' not found in filter map for task {task.id}"
                )

            task_id_str = task.id if task else "unknown"
            self.logger.info(f"Using filter '{task.assigned_filter_name}' for task {task_id_str}")
            return {target_filter_id: target_filter_info}

        # No filter specified - look for Clear or Luminance (case-insensitive)
        clear_filter_names = ["clear", "luminance", "lum", "l"]
        for fid, fdata in self.filter_map.items():
            if fdata.get("enabled", True) and fdata["name"].lower() in clear_filter_names:
                task_id_str = task.id if task else "unknown"
                self.logger.info(f"Using default filter '{fdata['name']}' for task {task_id_str}")
                return {fid: fdata}

        # No clear filter found - try first enabled filter
        enabled_filters = {fid: fdata for fid, fdata in self.filter_map.items() if fdata.get("enabled", True)}
        if enabled_filters:
            first_filter_id = next(iter(enabled_filters))
            task_id_str = task.id if task else "unknown"
            self.logger.info(
                f"Using first available filter '{enabled_filters[first_filter_id]['name']}' for task {task_id_str}"
            )
            return {first_filter_id: enabled_filters[first_filter_id]}

        # No enabled filters available
        if allow_no_filter:
            return None
        raise RuntimeError("No enabled filters available for observation sequence")

    def get_filter_config(self) -> dict[str, FilterConfig]:
        """Get the current filter configuration including focus positions.

        Returns:
            dict: Dictionary mapping filter IDs (as strings) to FilterConfig.
                  Each FilterConfig contains:
                  - name (str): Filter name
                  - focus_position (int): Focuser position for this filter
                  - enabled (bool): Whether filter is enabled for observations

        Example:
            {
                "1": {"name": "Luminance", "focus_position": 9000, "enabled": True},
                "2": {"name": "Red", "focus_position": 9050, "enabled": False}
            }
        """
        return {
            str(filter_id): {
                "name": filter_data["name"],
                "focus_position": filter_data["focus_position"],
                "enabled": filter_data.get("enabled", True),
            }
            for filter_id, filter_data in self.filter_map.items()
        }

    def update_filter_focus(self, filter_id: str, focus_position: int) -> bool:
        """Update the focus position for a specific filter.

        Args:
            filter_id: Filter ID as string
            focus_position: New focus position in steps

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            filter_id_int = int(filter_id)
            if filter_id_int in self.filter_map:
                self.filter_map[filter_id_int]["focus_position"] = focus_position
                return True
            return False
        except (ValueError, KeyError):
            return False

    def update_filter_enabled(self, filter_id: str, enabled: bool) -> bool:
        """Update the enabled state for a specific filter.

        Args:
            filter_id: Filter ID as string
            enabled: New enabled state

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            filter_id_int = int(filter_id)
            if filter_id_int in self.filter_map:
                self.filter_map[filter_id_int]["enabled"] = enabled
                return True
            return False
        except (ValueError, KeyError):
            return False
