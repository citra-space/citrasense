"""Direct hardware adapter using composable device adapters."""

import logging
import threading
import time
from collections.abc import Callable
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
from citrascope.hardware.devices.mount.altaz_pointing_model import AltAzPointingModel
from citrascope.hardware.devices.mount.mount_state_cache import MountStateCache
from citrascope.hardware.filter_sync import is_trash_filter_name
from citrascope.location.gps_fix import GPSFix


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
        self._mount_cache: MountStateCache | None = None
        self._preview_lock = threading.Lock()
        self._pointing_model_state_file: Path | None = None

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

        self._camera: AbstractCamera | None = None
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
            self._camera = camera_class(logger=self.logger, **camera_settings)

        # Check and instantiate mount (optional)
        self._mount: AbstractMount | None = None
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
                self._mount = mount_class(logger=self.logger, **mount_settings)

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
        self._focuser: AbstractFocuser | None = None
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
                self._focuser = focuser_class(logger=self.logger, **focuser_settings)

        # State tracking
        self._current_filter_position: int | None = None
        self._current_focus_position: int | None = None

        # Autofocus tuning (adapter-level settings)
        self._af_step_size: int = int(kwargs.get("autofocus_step_size", 500))
        self._af_num_steps: int = int(kwargs.get("autofocus_num_steps", 5))
        self._af_exposure: float = float(kwargs.get("autofocus_exposure", 3.0))
        self._af_crop: float = float(kwargs.get("autofocus_crop", 0.5))

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

        # Autofocus tuning — shown alongside focuser settings
        if focuser_type:
            schema.extend(
                [
                    {
                        "name": "autofocus_step_size",
                        "friendly_name": "AF Step Size",
                        "type": "int",
                        "default": 500,
                        "description": "Focuser steps between each autofocus sample",
                        "required": False,
                        "min": 10,
                        "max": 5000,
                        "group": "Focuser",
                    },
                    {
                        "name": "autofocus_num_steps",
                        "friendly_name": "AF Steps Per Side",
                        "type": "int",
                        "default": 5,
                        "description": "Number of samples on each side of centre (total = 2N+1)",
                        "required": False,
                        "min": 2,
                        "max": 15,
                        "group": "Focuser",
                    },
                    {
                        "name": "autofocus_exposure",
                        "friendly_name": "AF Exposure Time",
                        "type": "float",
                        "default": 3.0,
                        "description": "Exposure duration in seconds per autofocus sample",
                        "required": False,
                        "min": 0.01,
                        "max": 30.0,
                        "step": 0.01,
                        "group": "Focuser",
                    },
                    {
                        "name": "autofocus_crop",
                        "friendly_name": "AF Crop Ratio",
                        "type": "float",
                        "default": 0.5,
                        "description": "Fraction of image centre to analyse (0.1–1.0)",
                        "required": False,
                        "min": 0.1,
                        "max": 1.0,
                        "group": "Focuser",
                    },
                ]
            )

        return cast(list[SettingSchemaEntry], schema)

    @property
    def mount(self) -> AbstractMount | None:
        return self._mount

    @property
    def focuser(self) -> AbstractFocuser | None:
        return self._focuser

    @property
    def camera(self) -> AbstractCamera | None:
        return self._camera

    def get_gps_location(self) -> GPSFix | None:
        if self._camera is not None:
            return self._camera.get_gps_location()
        return None

    def supports_filter_management(self) -> bool:
        return self.filter_wheel is not None

    def supports_filter_rename(self) -> bool:
        return self.filter_wheel is not None

    def update_filter_name(self, filter_id: str, name: str) -> bool:
        """Rename a filter position (only meaningful for direct-controlled wheels)."""
        try:
            fid = int(filter_id)
            if fid not in self.filter_map:
                return False
            # Update hardware first; roll back if it rejects the change
            if self.filter_wheel:
                names = self.filter_wheel.get_filter_names()
                if fid < len(names):
                    names[fid] = name
                    if not self.filter_wheel.set_filter_names(names):
                        return False
            self.filter_map[fid]["name"] = name
            return True
        except (ValueError, KeyError):
            return False

    def get_observation_strategy(self) -> ObservationStrategy:
        """Get the observation strategy for direct control.

        Returns:
            ObservationStrategy.MANUAL - direct control handles exposures manually
        """
        return ObservationStrategy.MANUAL

    def perform_observation_sequence(self, task, satellite_data) -> list[str]:
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
        if self._mount:
            if not self._mount.connect():
                self.logger.error("Failed to connect to mount")
                success = False
            else:
                self._sync_mount_site_and_time()
                cache = MountStateCache(self._mount)
                cache.refresh_static()
                cache.start()
                self._mount._state_cache = cache  # type: ignore[attr-defined]
                self._mount_cache = cache
                self._init_pointing_model()
        else:
            self.logger.info("No mount configured (static camera mode)")

        # Connect camera
        if not self._camera:
            self.logger.error("Camera not initialized (missing dependencies)")
            return False
        if not self._camera.connect():
            self.logger.error("Failed to connect to camera")
            success = False

        # Connect optional devices
        if self.filter_wheel and not self.filter_wheel.connect():
            self.logger.warning("Failed to connect to filter wheel (optional)")
            self.filter_wheel = None

        # Auto-detect integrated filter wheel if no standalone one configured
        if not self.filter_wheel and self._camera:
            integrated_fw = self._camera.get_integrated_filter_wheel()
            if integrated_fw:
                if integrated_fw.connect():
                    self.filter_wheel = integrated_fw
                    self.logger.info("Using camera's integrated filter wheel")
                else:
                    self.logger.warning("Failed to connect to camera's integrated filter wheel (optional)")

        # Populate filter_map from hardware filter wheel
        if self.filter_wheel:
            self._populate_filter_map_from_hardware()

        if self._focuser and not self._focuser.connect():
            self.logger.warning("Failed to connect to focuser (optional)")

        if success:
            self.logger.info("All required devices connected successfully")
            self._warm_schema_cache()

        return success

    def _warm_schema_cache(self) -> None:
        """Pre-run device hardware probes in a background thread.

        Populates the probe cache so the first web UI schema request
        is an instant cache hit instead of blocking on subprocess probes.
        """

        def _probe():
            for device in (self._camera, self._mount, self.filter_wheel, self._focuser):
                if device is not None:
                    try:
                        device.get_settings_schema()
                    except Exception:
                        pass
            self.logger.debug("Hardware probe cache warmed")

        threading.Thread(target=_probe, daemon=True, name="probe-cache-warmup").start()

    def _populate_filter_map_from_hardware(self) -> None:
        """Sync filter_map with hardware filter wheel, preserving user config.

        For each hardware position:
        - New position with a real hardware name: use it.
        - New position with a trash name (Undefined, blank, etc.): use "Filter {N}".
        - Existing position with a real saved name: always keep it.
        - Existing position with a trash saved name + real hardware name: use hardware.
        """
        assert self.filter_wheel is not None
        hw_names = self.filter_wheel.get_filter_names()
        hw_count = self.filter_wheel.get_filter_count()

        for i in range(hw_count):
            raw_hw_name = hw_names[i] if i < len(hw_names) else ""
            hw_name = raw_hw_name.strip() if raw_hw_name else ""

            if i not in self.filter_map:
                name = hw_name if not is_trash_filter_name(hw_name) else f"Filter {i + 1}"
                self.filter_map[i] = {
                    "name": name,
                    "focus_position": None,
                    "enabled": True,
                }
            else:
                existing = self.filter_map[i]
                existing_name = existing.get("name", "")
                if is_trash_filter_name(existing_name) and not is_trash_filter_name(hw_name):
                    existing["name"] = hw_name

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
        assert self._mount is not None

        # Log mount mode — alt-az is expected for satellite work
        mode = self._mount.get_mount_mode()
        self.logger.info("Mount operating mode: %s", mode)

        # Sync site location from LocationService
        if self.location_service:
            location = self.location_service.get_current_location()
            if location:
                ok = self._mount.set_site_location(location["latitude"], location["longitude"], location["altitude"])
                if not ok:
                    self.logger.warning("Mount rejected site location")
            else:
                self.logger.warning("No location available — mount site not set")
        else:
            self.logger.warning("No LocationService — mount site not set")

        # Sync system UTC clock to mount
        if not self._mount.sync_datetime():
            self.logger.warning("Mount time sync not supported or failed")

        # Unpark if the mount is parked — GoTo is rejected while parked
        if self._mount.is_parked():
            self._mount.unpark()
            self.logger.info("Mount was parked — unparked for operation")

        # Altitude limits: software-enforced via AltitudeLimitCheck in the
        # safety monitor (firmware limits broken on AM5 fw 1.1.2 where
        # :GL/:SL commands collide with Get/Set Local Time).
        # Log current state for diagnostics but don't try to configure.
        try:
            limits_on = self._mount.get_altitude_limits_enabled()
            lower, upper = self._mount.get_limits()
            self.logger.info("Firmware altitude limits: enabled=%s lower=%s° upper=%s°", limits_on, lower, upper)
        except Exception:
            self.logger.debug("Could not read firmware altitude limits", exc_info=True)

        # Meridian flip is only relevant in equatorial mode.
        # In alt-az (the default for satellite observation) there is no
        # meridian concept, so we skip this entirely.
        if mode == "equatorial":
            try:
                if self._mount.get_meridian_auto_flip() is not True:
                    self._mount.set_meridian_auto_flip(True)
            except Exception:
                self.logger.debug("Meridian auto-flip not supported", exc_info=True)

    def _init_pointing_model(self) -> None:
        """Initialize the pointing model, loading persisted state if available."""
        if self._pointing_model_state_file:
            self._pointing_model = AltAzPointingModel(state_file=self._pointing_model_state_file)
        else:
            import platformdirs

            data_dir = Path(platformdirs.user_data_dir("citrascope", appauthor="citrascope"))
            state_file = data_dir / "pointing_model_state.json"
            self._pointing_model = AltAzPointingModel(state_file=state_file)
            self._pointing_model_state_file = state_file

        if self._pointing_model.is_active:
            status = self._pointing_model.status()
            self.logger.info(
                "Pointing model loaded: %s (%d pts, tilt=%.3f° toward %s, accuracy=%.1f')",
                status["state"],
                status["point_count"],
                status["tilt_deg"],
                status["tilt_direction_label"],
                status["pointing_accuracy_arcmin"],
            )

    def disconnect(self):
        """Disconnect from all hardware devices."""
        self.logger.info("Disconnecting from direct hardware devices...")

        if self._mount_cache:
            self._mount_cache.stop()
            self._mount_cache = None

        if self._camera:
            self._camera.disconnect()

        if self._mount:
            self._mount.disconnect()

        if self.filter_wheel:
            self.filter_wheel.disconnect()

        if self._focuser:
            self._focuser.disconnect()

        self.logger.info("All devices disconnected")

    def is_telescope_connected(self) -> bool:
        """Check if telescope mount is connected.

        Returns:
            True if mount is connected and responsive, or True if no mount (static camera)
        """
        if not self._mount:
            return True  # No mount required for static camera
        return self._mount.is_connected()

    def is_camera_connected(self) -> bool:
        """Check if camera is connected.

        Returns:
            True if camera is connected and responsive
        """
        if not self._camera:
            return False
        return self._camera.is_connected()

    def _do_point_telescope(self, ra: float, dec: float):
        """Hardware-specific slew implementation.

        Coordinates arriving here are already corrected by the base class's
        pointing model logic in ``point_telescope()``.
        """
        if not self._mount:
            self.logger.warning("No mount configured - cannot point telescope (static camera mode)")
            return

        self.logger.info(f"Slewing telescope to RA={ra:.4f}°, Dec={dec:.4f}°")

        if not self._mount.slew_to_radec(ra, dec):
            raise RuntimeError(f"Failed to initiate slew to RA={ra}, Dec={dec}")

        # Wait for slew to complete
        timeout = 300  # 5 minute timeout
        start_time = time.time()

        while self._mount.is_slewing():
            if time.time() - start_time > timeout:
                self._mount.abort_slew()
                raise RuntimeError("Slew timeout exceeded")
            time.sleep(0.5)

        self.logger.info("Slew complete")

        # Ensure tracking is enabled
        if not self._mount.is_tracking():
            self.logger.info("Starting sidereal tracking")
            self._mount.start_tracking("sidereal")

    def home_mount(self) -> bool:
        if not self._mount or not self._mount.is_connected():
            self.logger.warning("No mount connected — cannot home")
            return False
        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")
        return self._mount.find_home()

    def home_if_needed(self) -> bool:
        """Home the mount if not already homed, blocking until complete.

        In alt-az mode the mount can't convert RA/Dec to Alt/Az without a
        calibrated azimuth reference, so GoTo will fail until homing completes.
        The AM5 uses absolute encoders; :hC# triggers a physical slew to the
        home index — typically takes only a few seconds.

        This is called after connect() and after the SafetyMonitor is online,
        so the CableWrapCheck is actively observing during the homing slew.
        """
        if not self._mount or not self._mount.is_connected():
            self.logger.info("No mount connected — skipping home")
            return True
        if self._mount.is_home():
            self.logger.info("Mount already homed")
            return True

        if self._safety_monitor and not self._safety_monitor.is_action_safe("home"):
            from citrascope.safety.safety_monitor import SafetyError

            raise SafetyError("Homing blocked by safety monitor")

        pre_az = self._mount.get_azimuth()
        self.logger.info("Mount not homed — initiating find-home (pre-home az=%.1f°)", pre_az or 0.0)
        self._mount.find_home()

        _TIMEOUT_S = 60
        _GRACE_POLLS = 5
        _IDLE_THRESHOLD = 3
        deadline = time.monotonic() + _TIMEOUT_S
        poll_count = 0
        idle_count = 0
        while time.monotonic() < deadline:
            time.sleep(1)

            try:
                if self._mount.is_home():
                    post_az = self._mount.get_azimuth()
                    self.logger.info(
                        "Mount homed successfully — az %.1f° → %.1f° (delta=%.1f°)",
                        pre_az or 0.0,
                        post_az or 0.0,
                        abs((post_az or 0.0) - (pre_az or 0.0)),
                    )
                    return True
            except Exception:
                self.logger.debug("is_home() check failed during homing poll", exc_info=True)

            poll_count += 1
            if poll_count > _GRACE_POLLS:
                try:
                    still_moving = self._mount.is_slewing()
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
        if not self._mount or not self._mount.is_connected():
            return False
        return self._mount.is_home()

    def get_mount_limits(self) -> tuple[int | None, int | None]:
        if not self._mount or not self._mount.is_connected():
            return None, None
        return self._mount.get_limits()

    def set_mount_horizon_limit(self, degrees: int) -> bool:
        if not self._mount or not self._mount.is_connected():
            return False
        ok = self._mount.set_horizon_limit(degrees)
        if ok and self._mount_cache:
            self._mount_cache.refresh_limits()
        return ok

    def set_mount_overhead_limit(self, degrees: int) -> bool:
        if not self._mount or not self._mount.is_connected():
            return False
        ok = self._mount.set_overhead_limit(degrees)
        if ok and self._mount_cache:
            self._mount_cache.refresh_limits()
        return ok

    def get_scope_radec(self) -> tuple[float, float]:
        """Get current telescope RA/Dec position.

        Returns:
            Tuple of (RA in degrees, Dec in degrees), or (0.0, 0.0) if no mount
        """
        if not self._mount:
            # self.logger.warning("No mount configured - returning default RA/Dec")
            return (0.0, 0.0)
        return self._mount.get_radec()

    def _get_camera_file_extension(self) -> str:
        """Get the preferred file extension from the camera.

        Delegates to the camera's get_preferred_file_extension() method,
        which allows each camera type to define its own file format logic.

        Returns:
            File extension string (e.g., 'fits', 'png', 'jpg')
        """
        if not self._camera:
            return "fits"

        # Let the camera decide its preferred file extension
        return self._camera.get_preferred_file_extension()

    def supports_direct_camera_control(self) -> bool:
        return True

    def capture_preview(self, exposure_time: float, flip_horizontal: bool = False) -> str:
        if not self._camera:
            raise RuntimeError("Camera not initialized")
        if not self._preview_lock.acquire(blocking=False):
            raise RuntimeError("Preview capture already in progress")

        try:
            from citrascope.web.preview import array_to_jpeg_data_url

            data = self._camera.capture_array(
                duration=exposure_time,
                binning=self._camera.get_default_binning(),
            )
            return array_to_jpeg_data_url(data, flip_horizontal=flip_horizontal)
        finally:
            self._preview_lock.release()

    def expose_camera(
        self,
        exposure_time: float,
        gain: int | None = None,
        offset: int | None = None,
        count: int = 1,
        shutter_closed: bool = False,
    ) -> str:
        """Take camera exposure(s).

        Args:
            exposure_time: Exposure duration in seconds
            gain: Camera gain setting
            offset: Camera offset setting
            count: Number of exposures to take
            shutter_closed: If True, request dark frame (shutter stays closed).

        Returns:
            Path to the last saved image
        """
        if not self._camera:
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
            image_path = self._camera.take_exposure(
                duration=exposure_time,
                gain=gain,
                offset=offset,
                binning=self._camera.get_default_binning(),
                save_path=save_path,
                shutter_closed=shutter_closed,
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
        if self._focuser and filter_position in self.filter_map:
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
        if not self._focuser:
            self.logger.warning("No focuser available")
            return False

        self.logger.info(f"Moving focuser to position {position}")

        if not self._focuser.move_absolute(position):
            self.logger.error(f"Failed to move focuser to {position}")
            return False

        # Wait for focuser to finish moving
        timeout = 60
        start_time = time.time()

        while self._focuser.is_moving():
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
        if not self._focuser:
            return None
        return self._focuser.get_position()

    def supports_autofocus(self) -> bool:
        """Autofocus is available when both a camera and focuser are connected."""
        return (
            self._camera is not None
            and self._camera.is_connected()
            and self._focuser is not None
            and self._focuser.is_connected()
        )

    def do_autofocus(
        self,
        target_ra: float | None = None,
        target_dec: float | None = None,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Run V-curve autofocus for each enabled filter.

        Slews to target (if mount available and coordinates given),
        then sweeps the focuser at each filter to find optimal focus.
        """
        from citrascope.hardware.direct.autofocus import run_autofocus

        assert self._camera is not None
        assert self._focuser is not None

        report = on_progress or (lambda _msg: None)

        # Slew to target star if we have a mount and coordinates
        if target_ra is not None and target_dec is not None and self._mount:
            report("Slewing to autofocus target...")
            self._do_point_telescope(target_ra, target_dec)

        # Determine which filters to autofocus
        enabled_filters = {fid: fdata for fid, fdata in self.filter_map.items() if fdata.get("enabled", True)}

        # Retrieve autofocus tuning from adapter settings
        af_step = int(self._af_step_size)
        af_steps = int(self._af_num_steps)
        af_exp = float(self._af_exposure)
        af_crop = float(self._af_crop)

        if not enabled_filters or not self.filter_wheel:
            report("Running autofocus (no filter wheel)...")
            best = run_autofocus(
                camera=self._camera,
                focuser=self._focuser,
                step_size=af_step,
                num_steps=af_steps,
                exposure_time=af_exp,
                crop_ratio=af_crop,
                on_progress=on_progress,
                logger=self.logger,
                cancel_event=cancel_event,
            )
            self.logger.info(f"Autofocus result: position {best}")
            return

        total_filters = len(enabled_filters)
        for idx, (fid, fdata) in enumerate(enabled_filters.items(), 1):
            if cancel_event and cancel_event.is_set():
                report("Autofocus cancelled")
                self.logger.info("Autofocus cancelled between filters")
                return

            fname = fdata.get("name", f"Filter {fid}")
            report(f"Filter {idx}/{total_filters}: {fname}")
            self.logger.info(f"Autofocusing filter '{fname}' (position {fid})")

            if not self.set_filter(fid):
                self.logger.error(f"Failed to switch to filter {fid}, skipping")
                continue

            best = run_autofocus(
                camera=self._camera,
                focuser=self._focuser,
                step_size=af_step,
                num_steps=af_steps,
                exposure_time=af_exp,
                crop_ratio=af_crop,
                on_progress=on_progress,
                logger=self.logger,
                cancel_event=cancel_event,
            )

            self.filter_map[fid]["focus_position"] = best
            self.logger.info(f"Filter '{fname}' focus position: {best}")

        report("Autofocus complete for all filters")

    def get_sensor_temperature(self) -> float | None:
        """Get camera sensor temperature.

        Returns:
            Temperature in Celsius, or None if unavailable
        """
        if not self._camera:
            return None
        return self._camera.get_temperature()

    def is_hyperspectral(self) -> bool:
        """Indicates whether this adapter uses a hyperspectral camera.

        Returns:
            bool: True if camera is hyperspectral, False otherwise
        """
        if self._camera:
            return self._camera.is_hyperspectral()
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
        if self._camera:
            self._camera.abort_exposure()

        # Stop mount slew if running
        if self._mount and self._mount.is_slewing():
            self._mount.abort_slew()

        # Stop focuser if moving
        if self._focuser and self._focuser.is_moving():
            self._focuser.abort_move()

    # Required abstract method implementations

    def list_devices(self) -> list[str]:
        """List all connected devices.

        Returns:
            List of device names/descriptions
        """
        devices = []

        if self._camera:
            devices.append(f"Camera: {self._camera.get_friendly_name()}")
        else:
            devices.append("Camera: Not initialized (missing dependencies)")

        if self._mount:
            devices.append(f"Mount: {self._mount.get_friendly_name()}")
        else:
            devices.append("Mount: None (static camera mode)")

        if self.filter_wheel:
            devices.append(f"Filter Wheel: {self.filter_wheel.get_friendly_name()}")

        if self._focuser:
            devices.append(f"Focuser: {self._focuser.get_friendly_name()}")

        return devices

    def select_telescope(self, device_name: str) -> bool:
        """Select telescope device (not applicable for direct control).

        Direct hardware adapter has mount pre-configured at initialization.

        Args:
            device_name: Ignored

        Returns:
            True if mount is configured and connected
        """
        if not self._mount:
            self.logger.warning("No mount configured")
            return False
        return self._mount.is_connected()

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
        if not self._mount:
            return False
        return self._mount.is_slewing()

    def select_camera(self, device_name: str) -> bool:
        """Select camera device (not applicable for direct control).

        Direct hardware adapter has camera pre-configured at initialization.

        Args:
            device_name: Ignored

        Returns:
            True if camera is connected
        """
        if not self._camera:
            return False
        return self._camera.is_connected()

    def take_image(self, task_id: str, exposure_duration_seconds: float = 1.0) -> str:
        """Capture an image with the camera.

        Args:
            task_id: Task ID for organizing images
            exposure_duration_seconds: Exposure time in seconds

        Returns:
            Path to the saved image
        """
        if not self._camera:
            raise RuntimeError("Camera not initialized (missing dependencies)")

        # Generate save path with task ID
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Use camera's preferred file extension
        output_ext = self._get_camera_file_extension()
        save_path = self.images_dir / f"task_{task_id}_{timestamp}.{output_ext}"

        return str(
            self._camera.take_exposure(
                duration=exposure_duration_seconds,
                binning=self._camera.get_default_binning(),
                save_path=save_path,
            )
        )

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float) -> bool:
        """Set custom tracking rate for telescope.

        Args:
            ra_rate: RA tracking rate in arcseconds/second
            dec_rate: Dec tracking rate in arcseconds/second

        Returns:
            True if rates were accepted by the mount, False otherwise.
        """
        if not self._mount:
            self.logger.warning("No mount configured - cannot set tracking rate")
            return False

        self.logger.info(f'Setting custom tracking rate: RA={ra_rate}"/s, Dec={dec_rate}"/s')
        if not self._mount.set_custom_tracking_rates(ra_rate, dec_rate):
            self.logger.warning("Mount does not support custom tracking rates")
            return False
        return True

    def reset_tracking_rates(self) -> None:
        """Zero out custom tracking rate offsets, returning to base sidereal tracking."""
        if not self._mount:
            return
        self._mount.reset_tracking_rates()

    @property
    def supports_custom_tracking(self) -> bool:
        """Check if the connected mount supports custom tracking rates."""
        if not self._mount:
            return False
        try:
            info = self._mount.get_mount_info()
        except Exception as exc:
            self.logger.warning("Failed to query mount info for custom tracking support: %s", exc)
            return False
        return info.get("supports_custom_tracking", False)

    def get_tracking_rate(self) -> tuple[float, float]:
        """Get current telescope tracking rate.

        Returns:
            Tuple of (RA rate in arcsec/s, Dec rate in arcsec/s), or (0.0, 0.0) if no mount
        """
        if not self._mount:
            return (0.0, 0.0)
        if hasattr(self._mount, "get_tracking_rate"):
            return self._mount.get_tracking_rate()  # type: ignore
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
        """Plate-solve at the current position and conditionally sync the mount.

        When the pointing model is trained, skips the sync (model handles
        corrections) and instead runs a health check comparing the observed
        residual against the model's prediction.  When the model is not
        trained, syncs as before.

        Args:
            target_ra: Expected RA in degrees (for logging/error reporting).
            target_dec: Expected Dec in degrees (for logging/error reporting).

        Returns:
            True if plate solve succeeded.
        """
        if not self._mount:
            self.logger.warning("No mount configured — cannot perform alignment")
            return False
        if not self._camera:
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
                error = self.angular_distance(solved_ra, solved_dec, target_ra, target_dec)

                if self._pointing_model and self._pointing_model.is_trained:
                    residual_arcmin = error * 60.0
                    self._pointing_model.record_verification_residual(residual_arcmin)
                    self.logger.info(
                        "Alignment verified (model active, no sync): solved RA=%.4f° Dec=%.4f° " "(residual: %.1f')",
                        solved_ra,
                        solved_dec,
                        residual_arcmin,
                    )
                else:
                    self._mount.sync_to_radec(solved_ra, solved_dec)
                    self.logger.info(
                        "Alignment successful: solved RA=%.4f° Dec=%.4f° " "(error from target: %.1f', synced)",
                        solved_ra,
                        solved_dec,
                        error * 60,
                    )
                return True

            self.logger.warning(f"Plate solve failed with {exposure_s:.0f}s exposure, retrying...")

        self.logger.error("Alignment failed: plate solve did not converge after all attempts")
        return False
