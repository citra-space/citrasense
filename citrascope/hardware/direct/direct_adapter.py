"""Direct hardware adapter using composable device adapters."""

import logging
import time
from pathlib import Path
from typing import Any, cast

from citrascope.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
    SettingSchemaEntry,
)
from citrascope.hardware.devices.camera import AbstractCamera
from citrascope.hardware.devices.device_registry import (
    check_dependencies,
    get_camera_class,
    get_device_schema,
    get_filter_wheel_class,
    get_focuser_class,
    get_mount_class,
    list_devices,
)
from citrascope.hardware.devices.filter_wheel import AbstractFilterWheel
from citrascope.hardware.devices.focuser import AbstractFocuser
from citrascope.hardware.devices.mount import AbstractMount
from citrascope.hardware.devices.mount.mount_state_cache import MountStateCache


class DirectHardwareAdapter(AbstractAstroHardwareAdapter):
    """Hardware adapter that directly controls individual device components.

    This adapter composes individual device adapters (camera, mount, filter wheel, focuser)
    to provide complete telescope system control. It's designed for direct device control
    via USB, serial, or network protocols rather than through orchestration software.

    Device types are selected via settings, and device-specific configuration is passed
    through to each device adapter.
    """

    def __init__(self, logger: logging.Logger, images_dir: Path, **kwargs):
        """Initialize the direct hardware adapter.

        Args:
            logger: Logger instance
            images_dir: Directory for saving images
            **kwargs: Configuration including:
                - camera_type: Type of camera device (e.g., "ximea", "zwo")
                - mount_type: Type of mount device (e.g., "celestron", "skywatcher")
                - filter_wheel_type: Optional filter wheel type
                - focuser_type: Optional focuser type
                - camera_*: Camera-specific settings
                - mount_*: Mount-specific settings
                - filter_wheel_*: Filter wheel-specific settings
                - focuser_*: Focuser-specific settings
        """
        super().__init__(images_dir, **kwargs)
        self.logger = logger
        self.location_service: Any | None = None
        self._mount_cache: MountStateCache | None = None

        # Track dependency issues for reporting
        self._dependency_issues: list[dict[str, str]] = []

        # Extract device types from settings
        camera_type = kwargs.get("camera_type")
        mount_type = kwargs.get("mount_type")
        filter_wheel_type = kwargs.get("filter_wheel_type")
        focuser_type = kwargs.get("focuser_type")

        if not camera_type:
            raise ValueError("camera_type is required in settings")

        # Extract device-specific settings
        camera_settings = {k[7:]: v for k, v in kwargs.items() if k.startswith("camera_")}
        mount_settings = {k[6:]: v for k, v in kwargs.items() if k.startswith("mount_")}
        filter_wheel_settings = {k[13:]: v for k, v in kwargs.items() if k.startswith("filter_wheel_")}
        focuser_settings = {k[8:]: v for k, v in kwargs.items() if k.startswith("focuser_")}

        # Check and instantiate camera (required)
        self.logger.info(f"Checking camera dependencies: {camera_type}")
        camera_class = get_camera_class(camera_type)
        camera_deps = check_dependencies(camera_class)

        self.camera: AbstractCamera | None = None
        if not camera_deps["available"]:
            self.logger.warning(
                f"Camera '{camera_type}' missing dependencies: {', '.join(camera_deps['missing'])}. "
                f"Install with: {camera_deps['install_cmd']}"
            )
            self._dependency_issues.append(
                {
                    "device_type": "Camera",
                    "device_name": camera_class.get_friendly_name(),
                    "missing_packages": ", ".join(camera_deps["missing"]),
                    "install_cmd": camera_deps["install_cmd"],
                }
            )
        else:
            self.logger.info(f"Instantiating camera: {camera_type}")
            self.camera = camera_class(logger=self.logger, **camera_settings)

        # Check and instantiate mount (optional)
        self.mount: AbstractMount | None = None
        if mount_type:
            self.logger.info(f"Checking mount dependencies: {mount_type}")
            mount_class = get_mount_class(mount_type)
            mount_deps = check_dependencies(mount_class)

            if not mount_deps["available"]:
                self.logger.warning(
                    f"Mount '{mount_type}' missing dependencies: {', '.join(mount_deps['missing'])}. "
                    f"Install with: {mount_deps['install_cmd']}"
                )
                self._dependency_issues.append(
                    {
                        "device_type": "Mount",
                        "device_name": mount_class.get_friendly_name(),
                        "missing_packages": ", ".join(mount_deps["missing"]),
                        "install_cmd": mount_deps["install_cmd"],
                    }
                )
            else:
                self.logger.info(f"Instantiating mount: {mount_type}")
                self.mount = mount_class(logger=self.logger, **mount_settings)

        # Check and instantiate filter wheel (optional)
        self.filter_wheel: AbstractFilterWheel | None = None
        if filter_wheel_type:
            self.logger.info(f"Checking filter wheel dependencies: {filter_wheel_type}")
            filter_wheel_class = get_filter_wheel_class(filter_wheel_type)
            fw_deps = check_dependencies(filter_wheel_class)

            if not fw_deps["available"]:
                self.logger.warning(
                    f"Filter wheel '{filter_wheel_type}' missing dependencies: {', '.join(fw_deps['missing'])}. "
                    f"Install with: {fw_deps['install_cmd']}"
                )
                self._dependency_issues.append(
                    {
                        "device_type": "Filter Wheel",
                        "device_name": filter_wheel_class.get_friendly_name(),
                        "missing_packages": ", ".join(fw_deps["missing"]),
                        "install_cmd": fw_deps["install_cmd"],
                    }
                )
            else:
                self.logger.info(f"Instantiating filter wheel: {filter_wheel_type}")
                self.filter_wheel = filter_wheel_class(logger=self.logger, **filter_wheel_settings)

        # Check and instantiate focuser (optional)
        self.focuser: AbstractFocuser | None = None
        if focuser_type:
            self.logger.info(f"Checking focuser dependencies: {focuser_type}")
            focuser_class = get_focuser_class(focuser_type)
            focuser_deps = check_dependencies(focuser_class)

            if not focuser_deps["available"]:
                self.logger.warning(
                    f"Focuser '{focuser_type}' missing dependencies: {', '.join(focuser_deps['missing'])}. "
                    f"Install with: {focuser_deps['install_cmd']}"
                )
                self._dependency_issues.append(
                    {
                        "device_type": "Focuser",
                        "device_name": focuser_class.get_friendly_name(),
                        "missing_packages": ", ".join(focuser_deps["missing"]),
                        "install_cmd": focuser_deps["install_cmd"],
                    }
                )
            else:
                self.logger.info(f"Instantiating focuser: {focuser_type}")
                self.focuser = focuser_class(logger=self.logger, **focuser_settings)

        # State tracking
        self._current_filter_position: int | None = None
        self._current_focus_position: int | None = None

        self.logger.info("DirectHardwareAdapter initialized with:")
        self.logger.info(f"  Camera: {camera_type}")
        if mount_type:
            self.logger.info(f"  Mount: {mount_type}")
        else:
            self.logger.info("  Mount: None (static camera)")
        if filter_wheel_type:
            self.logger.info(f"  Filter Wheel: {filter_wheel_type}")
        if focuser_type:
            self.logger.info(f"  Focuser: {focuser_type}")

    @classmethod
    def get_settings_schema(cls, **kwargs) -> list[SettingSchemaEntry]:
        """Return schema for direct hardware adapter settings.

        This includes device type selection and adapter-level settings.
        If device types are provided in kwargs, will dynamically include
        device-specific settings with appropriate prefixes.

        Args:
            **kwargs: Can include camera_type, mount_type, etc. to get dynamic schemas

        Returns:
            List of setting schema entries
        """
        # Get available devices for dropdown options
        camera_devices = list_devices("camera")
        mount_devices = list_devices("mount")
        filter_wheel_devices = list_devices("filter_wheel")
        focuser_devices = list_devices("focuser")

        # Build options as list of dicts with value (key) and display (friendly_name)
        # Format: [{"value": "rpi_hq", "label": "Raspberry Pi HQ Camera"}, ...]
        camera_options = [{"value": k, "label": v["friendly_name"]} for k, v in camera_devices.items()]
        mount_options = [{"value": k, "label": v["friendly_name"]} for k, v in mount_devices.items()]
        filter_wheel_options = [{"value": k, "label": v["friendly_name"]} for k, v in filter_wheel_devices.items()]
        focuser_options = [{"value": k, "label": v["friendly_name"]} for k, v in focuser_devices.items()]

        schema: list[Any] = [
            # Camera is always required
            {
                "name": "camera_type",
                "friendly_name": "Camera Type",
                "type": "str",
                "default": camera_options[0]["value"] if camera_options else "",
                "description": "Type of camera device to use",
                "required": True,
                "options": camera_options,
                "group": "Camera",
            },
        ]

        # Only include optional device fields if devices are available
        if mount_options:
            schema.append(
                {
                    "name": "mount_type",
                    "friendly_name": "Mount Type",
                    "type": "str",
                    "default": "",
                    "description": "Type of mount device (leave empty for static camera setups)",
                    "required": False,
                    "options": mount_options,
                    "group": "Mount",
                }
            )

        all_fw_options: list[dict[str, str]] = [
            {"value": "", "label": "None (or use camera's integrated wheel)"},
            *filter_wheel_options,
        ]
        schema.append(
            {
                "name": "filter_wheel_type",
                "friendly_name": "Filter Wheel Type",
                "type": "str",
                "default": "",
                "description": "Leave empty to auto-detect an integrated filter wheel from the camera",
                "required": False,
                "options": all_fw_options,
                "group": "Filter Wheel",
            }
        )

        if focuser_options:
            schema.append(
                {
                    "name": "focuser_type",
                    "friendly_name": "Focuser Type",
                    "type": "str",
                    "default": "",
                    "description": "Type of focuser device (leave empty if none)",
                    "required": False,
                    "options": focuser_options,
                    "group": "Focuser",
                }
            )

        def _prefix_schema(entries: list, prefix: str) -> None:
            for entry in entries:
                prefixed_entry = dict(entry)
                prefixed_entry["name"] = f"{prefix}{entry['name']}"
                if "visible_when" in entry:
                    vw = dict(entry["visible_when"])
                    vw["field"] = f"{prefix}{vw['field']}"
                    prefixed_entry["visible_when"] = vw
                schema.append(prefixed_entry)

        # Dynamically add device-specific settings if device types are provided
        camera_type = kwargs.get("camera_type")
        if camera_type and camera_type in camera_devices:
            _prefix_schema(get_device_schema("camera", camera_type), "camera_")

        mount_type = kwargs.get("mount_type")
        if mount_type and mount_type in mount_devices:
            _prefix_schema(get_device_schema("mount", mount_type), "mount_")

        filter_wheel_type = kwargs.get("filter_wheel_type")
        if filter_wheel_type and filter_wheel_type in filter_wheel_devices:
            _prefix_schema(get_device_schema("filter_wheel", filter_wheel_type), "filter_wheel_")

        focuser_type = kwargs.get("focuser_type")
        if focuser_type and focuser_type in focuser_devices:
            _prefix_schema(get_device_schema("focuser", focuser_type), "focuser_")
        return cast(list[SettingSchemaEntry], schema)

    def set_location_service(self, location_service) -> None:
        self.location_service = location_service

    def supports_filter_management(self) -> bool:
        return self.filter_wheel is not None

    def update_filter_name(self, filter_id: str, name: str) -> bool:
        """Rename a filter position (only meaningful for direct-controlled wheels)."""
        try:
            fid = int(filter_id)
            if fid in self.filter_map:
                self.filter_map[fid]["name"] = name
                if self.filter_wheel:
                    names = self.filter_wheel.get_filter_names()
                    if fid < len(names):
                        names[fid] = name
                        self.filter_wheel.set_filter_names(names)
                return True
            return False
        except (ValueError, KeyError):
            return False

    def get_observation_strategy(self) -> ObservationStrategy:
        """Get the observation strategy for direct control.

        Returns:
            ObservationStrategy.MANUAL - direct control handles exposures manually
        """
        return ObservationStrategy.MANUAL

    def perform_observation_sequence(self, task, satellite_data) -> str:
        """Not implemented for manual observation strategy.

        Direct hardware adapter uses manual control - exposures are taken
        via explicit calls to expose_camera() rather than sequences.

        Raises:
            NotImplementedError: This adapter uses manual observation
        """
        raise NotImplementedError(
            "DirectHardwareAdapter uses MANUAL observation strategy. "
            "Use expose_camera() to take individual exposures."
        )

    def connect(self) -> bool:
        """Connect to all hardware devices.

        Returns:
            True if all required devices connected successfully
        """
        self.logger.info("Connecting to direct hardware devices...")

        success = True

        # Connect mount (if present)
        if self.mount:
            if not self.mount.connect():
                self.logger.error("Failed to connect to mount")
                success = False
            else:
                self._sync_mount_site_and_time()
                cache = MountStateCache(self.mount)
                cache.refresh_static()
                cache.start()
                self.mount._state_cache = cache  # type: ignore[attr-defined]
                self._mount_cache = cache
        else:
            self.logger.info("No mount configured (static camera mode)")

        # Connect camera
        if not self.camera:
            self.logger.error("Camera not initialized (missing dependencies)")
            return False
        if not self.camera.connect():
            self.logger.error("Failed to connect to camera")
            success = False

        # Connect optional devices
        if self.filter_wheel and not self.filter_wheel.connect():
            self.logger.warning("Failed to connect to filter wheel (optional)")

        # Auto-detect integrated filter wheel if no standalone one configured
        if not self.filter_wheel and self.camera:
            integrated_fw = self.camera.get_integrated_filter_wheel()
            if integrated_fw:
                integrated_fw.connect()
                self.filter_wheel = integrated_fw
                self.logger.info("Using camera's integrated filter wheel")

        # Populate filter_map from hardware filter wheel
        if self.filter_wheel:
            self._populate_filter_map_from_hardware()

        if self.focuser and not self.focuser.connect():
            self.logger.warning("Failed to connect to focuser (optional)")

        if success:
            self.logger.info("All required devices connected successfully")

        return success

    def _populate_filter_map_from_hardware(self) -> None:
        """Sync filter_map with hardware filter wheel, preserving user config."""
        assert self.filter_wheel is not None
        hw_names = self.filter_wheel.get_filter_names()
        hw_count = self.filter_wheel.get_filter_count()

        for i in range(hw_count):
            if i not in self.filter_map:
                name = hw_names[i] if i < len(hw_names) else f"Filter {i + 1}"
                self.filter_map[i] = {
                    "name": name,
                    "focus_position": 0,
                    "enabled": True,
                }
            else:
                existing = self.filter_map[i]
                existing_name = existing.get("name", "")
                if i < len(hw_names) and (existing_name.startswith("Filter ") or existing_name == "Undefined"):
                    existing["name"] = hw_names[i]

        self.logger.info(f"Filter map: {self.filter_map}")

    def _sync_mount_site_and_time(self) -> None:
        """Push site location, time, and operational config to the mount.

        Called after a successful handshake.  All steps are best-effort:
        logs warnings on failure but never blocks connect().

        For satellite observation we keep the mount in **alt-az mode** —
        no sidereal tracking needed, no meridian flip constraints, and
        no polar alignment required.  Meridian flip config is only
        relevant in equatorial mode so we skip it in alt-az.
        """
        assert self.mount is not None

        # Log mount mode — alt-az is expected for satellite work
        mode = self.mount.get_mount_mode()
        self.logger.info("Mount operating mode: %s", mode)

        # Sync site location from LocationService
        if self.location_service:
            location = self.location_service.get_current_location()
            if location:
                ok = self.mount.set_site_location(location["latitude"], location["longitude"], location["altitude"])
                if not ok:
                    self.logger.warning("Mount rejected site location")
            else:
                self.logger.warning("No location available — mount site not set")
        else:
            self.logger.warning("No LocationService — mount site not set")

        # Sync system UTC clock to mount
        if not self.mount.sync_datetime():
            self.logger.warning("Mount time sync not supported or failed")

        # Unpark if the mount is parked — GoTo is rejected while parked
        if self.mount.is_parked():
            self.mount.unpark()
            self.logger.info("Mount was parked — unparked for operation")

        # Configure altitude limits for full-sky access.
        # Set values BEFORE enabling so we don't accidentally enforce
        # restrictive defaults.  On firmware 1.1.2 the :GL/:SL limit
        # commands collide with Get/Set Local Time and silently fail.
        try:
            upper_ok = self.mount.set_overhead_limit(90)
            lower_ok = self.mount.set_horizon_limit(0)
            if upper_ok and lower_ok:
                self.mount.set_altitude_limits_enabled(True)
            else:
                self.logger.info("Altitude limit commands not supported on this firmware — using mount defaults")
        except Exception:
            self.logger.debug("Altitude limit configuration not supported", exc_info=True)

        # Meridian flip is only relevant in equatorial mode.
        # In alt-az (the default for satellite observation) there is no
        # meridian concept, so we skip this entirely.
        if mode == "equatorial":
            try:
                if self.mount.get_meridian_auto_flip() is not True:
                    self.mount.set_meridian_auto_flip(True)
            except Exception:
                self.logger.debug("Meridian auto-flip not supported", exc_info=True)

    def disconnect(self):
        """Disconnect from all hardware devices."""
        self.logger.info("Disconnecting from direct hardware devices...")

        if self._mount_cache:
            self._mount_cache.stop()
            self._mount_cache = None

        if self.camera:
            self.camera.disconnect()

        if self.mount:
            self.mount.disconnect()

        if self.filter_wheel:
            self.filter_wheel.disconnect()

        if self.focuser:
            self.focuser.disconnect()

        self.logger.info("All devices disconnected")

    def is_telescope_connected(self) -> bool:
        """Check if telescope mount is connected.

        Returns:
            True if mount is connected and responsive, or True if no mount (static camera)
        """
        if not self.mount:
            return True  # No mount required for static camera
        return self.mount.is_connected()

    def is_camera_connected(self) -> bool:
        """Check if camera is connected.

        Returns:
            True if camera is connected and responsive
        """
        if not self.camera:
            return False
        return self.camera.is_connected()

    def _do_point_telescope(self, ra: float, dec: float):
        """Point the telescope to specified RA/Dec coordinates.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
        """
        if not self.mount:
            self.logger.warning("No mount configured - cannot point telescope (static camera mode)")
            return

        self.logger.info(f"Slewing telescope to RA={ra:.4f}°, Dec={dec:.4f}°")

        if not self.mount.slew_to_radec(ra, dec):
            raise RuntimeError(f"Failed to initiate slew to RA={ra}, Dec={dec}")

        # Wait for slew to complete
        timeout = 300  # 5 minute timeout
        start_time = time.time()

        while self.mount.is_slewing():
            if time.time() - start_time > timeout:
                self.mount.abort_slew()
                raise RuntimeError("Slew timeout exceeded")
            time.sleep(0.5)

        self.logger.info("Slew complete")

        # Ensure tracking is enabled
        if not self.mount.is_tracking():
            self.logger.info("Starting sidereal tracking")
            self.mount.start_tracking("sidereal")

    def home_mount(self) -> bool:
        if not self.mount or not self.mount.is_connected():
            self.logger.warning("No mount connected — cannot home")
            return False
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        return self.mount.find_home()

    def home_if_needed(self) -> bool:
        """Home the mount if not already homed, blocking until complete.

        In alt-az mode the mount can't convert RA/Dec to Alt/Az without a
        calibrated azimuth reference, so GoTo will fail until homing completes.
        The AM5 uses absolute encoders; :hC# triggers a physical slew to the
        home index — typically takes only a few seconds.

        This is called after connect() and after the SafetyMonitor is online,
        so the CableWrapCheck is actively observing during the homing slew.
        """
        if not self.mount or not self.mount.is_connected():
            self.logger.info("No mount connected — skipping home")
            return True
        if self.mount.is_home():
            self.logger.info("Mount already homed")
            return True

        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")

        self.logger.info("Mount not homed — initiating find-home (required for GoTo)")
        self.mount.find_home()

        _TIMEOUT_S = 60
        _GRACE_POLLS = 5
        _IDLE_THRESHOLD = 3
        deadline = time.monotonic() + _TIMEOUT_S
        poll_count = 0
        idle_count = 0
        while time.monotonic() < deadline:
            time.sleep(1)
            try:
                if self.mount.is_home():
                    self.logger.info("Mount homed successfully")
                    return True
            except Exception:
                self.logger.debug("is_home() check failed during homing poll", exc_info=True)

            poll_count += 1
            if poll_count > _GRACE_POLLS:
                try:
                    still_moving = self.mount.is_slewing()
                except Exception:
                    still_moving = True
                if not still_moving:
                    idle_count += 1
                    if idle_count >= _IDLE_THRESHOLD:
                        self.logger.warning(
                            "Mount stopped without reaching home (poll %d) — homing interrupted",
                            poll_count,
                        )
                        return False
                else:
                    idle_count = 0

        self.logger.warning("Homing did not complete within %d s — GoTo may fail", _TIMEOUT_S)
        return False

    def is_mount_homed(self) -> bool:
        if not self.mount or not self.mount.is_connected():
            return False
        return self.mount.is_home()

    def get_mount_limits(self) -> tuple[int | None, int | None]:
        if not self.mount or not self.mount.is_connected():
            return None, None
        return self.mount.get_limits()

    def set_mount_horizon_limit(self, degrees: int) -> bool:
        if not self.mount or not self.mount.is_connected():
            return False
        ok = self.mount.set_horizon_limit(degrees)
        if ok and self._mount_cache:
            self._mount_cache.refresh_limits()
        return ok

    def set_mount_overhead_limit(self, degrees: int) -> bool:
        if not self.mount or not self.mount.is_connected():
            return False
        ok = self.mount.set_overhead_limit(degrees)
        if ok and self._mount_cache:
            self._mount_cache.refresh_limits()
        return ok

    def get_scope_radec(self) -> tuple[float, float]:
        """Get current telescope RA/Dec position.

        Returns:
            Tuple of (RA in degrees, Dec in degrees), or (0.0, 0.0) if no mount
        """
        if not self.mount:
            # self.logger.warning("No mount configured - returning default RA/Dec")
            return (0.0, 0.0)
        return self.mount.get_radec()

    def _get_camera_file_extension(self) -> str:
        """Get the preferred file extension from the camera.

        Delegates to the camera's get_preferred_file_extension() method,
        which allows each camera type to define its own file format logic.

        Returns:
            File extension string (e.g., 'fits', 'png', 'jpg')
        """
        if not self.camera:
            return "fits"

        # Let the camera decide its preferred file extension
        return self.camera.get_preferred_file_extension()

    def supports_direct_camera_control(self) -> bool:
        return True

    def expose_camera(
        self,
        exposure_time: float,
        gain: int | None = None,
        offset: int | None = None,
        count: int = 1,
    ) -> str:
        """Take camera exposure(s).

        Args:
            exposure_time: Exposure duration in seconds
            gain: Camera gain setting
            offset: Camera offset setting
            count: Number of exposures to take

        Returns:
            Path to the last saved image
        """
        if not self.camera:
            raise RuntimeError("Camera not initialized (missing dependencies)")

        self.logger.info(f"Taking {count} exposure(s): {exposure_time}s, " f"gain={gain}, offset={offset}")

        last_image_path = ""

        for i in range(count):
            if count > 1:
                self.logger.info(f"Exposure {i+1}/{count}")

            # Generate save path with camera's preferred file extension
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_ext = self._get_camera_file_extension()
            save_path = self.images_dir / f"direct_capture_{timestamp}_{i:03d}.{output_ext}"

            # Take exposure
            image_path = self.camera.take_exposure(
                duration=exposure_time,
                gain=gain,
                offset=offset,
                binning=1,
                save_path=save_path,
            )

            last_image_path = str(image_path)

        return last_image_path

    def set_filter(self, filter_position: int) -> bool:
        """Change to specified filter.

        Args:
            filter_position: Filter position (0-indexed)

        Returns:
            True if filter change successful
        """
        if not self.filter_wheel:
            self.logger.warning("No filter wheel available")
            return False

        self.logger.info(f"Changing to filter position {filter_position}")

        if not self.filter_wheel.set_filter_position(filter_position):
            self.logger.error(f"Failed to set filter position {filter_position}")
            return False

        # Wait for filter wheel to finish moving
        timeout = 30
        start_time = time.time()

        while self.filter_wheel.is_moving():
            if time.time() - start_time > timeout:
                self.logger.error("Filter wheel movement timeout")
                return False
            time.sleep(0.1)

        self._current_filter_position = filter_position

        # Adjust focus if configured
        if self.focuser and filter_position in self.filter_map:
            focus_position = self.filter_map[filter_position].get("focus_position")
            if focus_position is not None:
                self.logger.info(f"Adjusting focus to {focus_position} for filter {filter_position}")
                self.set_focus(focus_position)

        self.logger.info(f"Filter change complete: position {filter_position}")
        return True

    def get_filter_position(self) -> int | None:
        """Get current filter position.

        Returns:
            Current filter position (0-indexed), or None if unavailable
        """
        if not self.filter_wheel:
            return None
        return self.filter_wheel.get_filter_position()

    def set_focus(self, position: int) -> bool:
        """Move focuser to absolute position.

        Args:
            position: Target focus position in steps

        Returns:
            True if focus move successful
        """
        if not self.focuser:
            self.logger.warning("No focuser available")
            return False

        self.logger.info(f"Moving focuser to position {position}")

        if not self.focuser.move_absolute(position):
            self.logger.error(f"Failed to move focuser to {position}")
            return False

        # Wait for focuser to finish moving
        timeout = 60
        start_time = time.time()

        while self.focuser.is_moving():
            if time.time() - start_time > timeout:
                self.logger.error("Focuser movement timeout")
                return False
            time.sleep(0.1)

        self._current_focus_position = position
        self.logger.info(f"Focus move complete: position {position}")
        return True

    def get_focus_position(self) -> int | None:
        """Get current focuser position.

        Returns:
            Current focus position in steps, or None if unavailable
        """
        if not self.focuser:
            return None
        return self.focuser.get_position()

    def get_sensor_temperature(self) -> float | None:
        """Get camera sensor temperature.

        Returns:
            Temperature in Celsius, or None if unavailable
        """
        if not self.camera:
            return None
        return self.camera.get_temperature()

    def is_hyperspectral(self) -> bool:
        """Indicates whether this adapter uses a hyperspectral camera.

        Returns:
            bool: True if camera is hyperspectral, False otherwise
        """
        if self.camera:
            return self.camera.is_hyperspectral()
        return False

    def get_missing_dependencies(self) -> list[dict[str, str]]:
        """Check for missing dependencies on all configured devices.

        Returns:
            List of dicts with keys: device_type, device_name, missing_packages, install_cmd
        """
        return self._dependency_issues

    def abort_current_operation(self):
        """Abort any ongoing operations."""
        self.logger.warning("Aborting all operations")

        # Abort camera exposure if running
        if self.camera:
            self.camera.abort_exposure()

        # Stop mount slew if running
        if self.mount and self.mount.is_slewing():
            self.mount.abort_slew()

        # Stop focuser if moving
        if self.focuser and self.focuser.is_moving():
            self.focuser.abort_move()

    # Required abstract method implementations

    def list_devices(self) -> list[str]:
        """List all connected devices.

        Returns:
            List of device names/descriptions
        """
        devices = []

        if self.camera:
            devices.append(f"Camera: {self.camera.get_friendly_name()}")
        else:
            devices.append("Camera: Not initialized (missing dependencies)")

        if self.mount:
            devices.append(f"Mount: {self.mount.get_friendly_name()}")
        else:
            devices.append("Mount: None (static camera mode)")

        if self.filter_wheel:
            devices.append(f"Filter Wheel: {self.filter_wheel.get_friendly_name()}")

        if self.focuser:
            devices.append(f"Focuser: {self.focuser.get_friendly_name()}")

        return devices

    def select_telescope(self, device_name: str) -> bool:
        """Select telescope device (not applicable for direct control).

        Direct hardware adapter has mount pre-configured at initialization.

        Args:
            device_name: Ignored

        Returns:
            True if mount is configured and connected
        """
        if not self.mount:
            self.logger.warning("No mount configured")
            return False
        return self.mount.is_connected()

    def get_telescope_direction(self) -> tuple[float, float]:
        """Get current telescope RA/Dec position.

        Returns:
            Tuple of (RA in degrees, Dec in degrees)
        """
        return self.get_scope_radec()

    def telescope_is_moving(self) -> bool:
        """Check if telescope is currently moving.

        Returns:
            True if mount is slewing, False otherwise
        """
        if not self.mount:
            return False
        return self.mount.is_slewing()

    def select_camera(self, device_name: str) -> bool:
        """Select camera device (not applicable for direct control).

        Direct hardware adapter has camera pre-configured at initialization.

        Args:
            device_name: Ignored

        Returns:
            True if camera is connected
        """
        if not self.camera:
            return False
        return self.camera.is_connected()

    def take_image(self, task_id: str, exposure_duration_seconds: float = 1.0) -> str:
        """Capture an image with the camera.

        Args:
            task_id: Task ID for organizing images
            exposure_duration_seconds: Exposure time in seconds

        Returns:
            Path to the saved image
        """
        if not self.camera:
            raise RuntimeError("Camera not initialized (missing dependencies)")

        # Generate save path with task ID
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Use camera's preferred file extension
        output_ext = self._get_camera_file_extension()
        save_path = self.images_dir / f"task_{task_id}_{timestamp}.{output_ext}"

        return str(
            self.camera.take_exposure(
                duration=exposure_duration_seconds,
                binning=self.camera.get_default_binning(),
                save_path=save_path,
            )
        )

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        """Set custom tracking rate for telescope.

        Args:
            ra_rate: RA tracking rate in arcseconds/second
            dec_rate: Dec tracking rate in arcseconds/second
        """
        if not self.mount:
            self.logger.warning("No mount configured - cannot set tracking rate")
            return

        self.logger.info(f'Setting custom tracking rate: RA={ra_rate}"/s, Dec={dec_rate}"/s')
        if not self.mount.set_custom_tracking_rates(ra_rate, dec_rate):
            self.logger.warning("Mount does not support custom tracking rates")

    def get_tracking_rate(self) -> tuple[float, float]:
        """Get current telescope tracking rate.

        Returns:
            Tuple of (RA rate in arcsec/s, Dec rate in arcsec/s), or (0.0, 0.0) if no mount
        """
        if not self.mount:
            return (0.0, 0.0)
        if hasattr(self.mount, "get_tracking_rate"):
            return self.mount.get_tracking_rate()  # type: ignore
        return (0.0, 0.0)

    def update_from_plate_solve(
        self,
        solved_ra_deg: float,
        solved_dec_deg: float,
        expected_ra_deg: float | None = None,
        expected_dec_deg: float | None = None,
    ) -> None:
        # Intentional no-op: async plate solves from the processing queue
        # arrive after the mount has potentially moved on to the next task.
        # Syncing stale coordinates corrupts the pointing model.
        # Mount syncs happen only through the explicit AlignmentManager path.
        if expected_ra_deg is not None and expected_dec_deg is not None:
            error = self.angular_distance(solved_ra_deg, solved_dec_deg, expected_ra_deg, expected_dec_deg)
            self.logger.info(
                "Plate solve result: pointing error %.1f arcmin (not syncing — use alignment instead)", error * 60
            )

    def perform_alignment(self, target_ra: float, target_dec: float) -> bool:
        """Plate-solve at the current position and sync the mount.

        Takes a short exposure, plate-solves it, and syncs the mount to the
        solved coordinates.  Retries with increasing exposure if the solve fails.

        Args:
            target_ra: Expected RA in degrees (for logging/error reporting).
            target_dec: Expected Dec in degrees (for logging/error reporting).

        Returns:
            True if plate solve + sync succeeded.
        """
        if not self.mount:
            self.logger.warning("No mount configured — cannot perform alignment")
            return False
        if not self.camera:
            self.logger.warning("No camera configured — cannot perform alignment")
            return False
        if not self.telescope_record:
            self.logger.warning("No telescope_record available — cannot plate-solve for alignment")
            return False

        from citrascope.processors.builtin.plate_solver_processor import PlateSolverProcessor

        exposure_attempts = [2.0, 4.0, 8.0]
        for exposure_s in exposure_attempts:
            self.logger.info(f"Alignment: taking {exposure_s:.0f}s exposure for plate solve...")
            try:
                image_path = self.take_image("alignment", exposure_s)
            except Exception as exc:
                self.logger.error(f"Alignment exposure failed: {exc}")
                continue

            result = PlateSolverProcessor.solve(
                Path(image_path),
                self.telescope_record,
            )
            if result is not None:
                solved_ra, solved_dec = result
                self.mount.sync_to_radec(solved_ra, solved_dec)
                error = self.angular_distance(solved_ra, solved_dec, target_ra, target_dec)
                self.logger.info(
                    f"Alignment successful: solved RA={solved_ra:.4f}°, Dec={solved_dec:.4f}° "
                    f"(error from target: {error * 60:.1f} arcmin)"
                )
                return True

            self.logger.warning(f"Plate solve failed with {exposure_s:.0f}s exposure, retrying...")

        self.logger.error("Alignment failed: plate solve did not converge after all attempts")
        return False
