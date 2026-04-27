import logging
import time
from pathlib import Path

import PyIndi  # type: ignore[reportMissingImports]

from citrasense.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
    SettingSchemaEntry,
)


# The IndiClient class which inherits from the module PyIndi.BaseClient class
# Note that all INDI constants are accessible from the module as PyIndi.CONSTANTNAME
class IndiAdapter(PyIndi.BaseClient, AbstractAstroHardwareAdapter):
    # Minimum angular distance (degrees) to consider a move significant for slew rate measurement
    _slew_min_distance_deg: float = 2.0

    our_scope: PyIndi.BaseDevice
    our_camera: PyIndi.BaseDevice

    def __init__(self, logger: logging.Logger, images_dir: Path, **kwargs):
        PyIndi.BaseClient.__init__(self)
        AbstractAstroHardwareAdapter.__init__(self, images_dir=images_dir)
        self.logger: logging.Logger = logger
        if self.logger:
            self.logger.debug("creating an instance of IndiClient")
        self.host = kwargs.get("host", "localhost")
        self.port = int(kwargs.get("port", 7624))
        self.telescope_name = kwargs.get("telescope_name", "")
        self.camera_name = kwargs.get("camera_name", "")
        # Per-instance mutable state. These used to live at class scope,
        # which silently aliased two IndiAdapter instances into the same
        # slots for any code path that accidentally used ``IndiAdapter._x``
        # instead of ``self._x``.  Multi-sensor deployments construct one
        # IndiAdapter per telescope sensor — keep the state local.
        self._current_task_id: str = ""
        self._last_saved_filename: str = ""
        self._alignment_offset_ra: float = 0.0
        self._alignment_offset_dec: float = 0.0

    @classmethod
    def get_settings_schema(cls, **kwargs) -> list[SettingSchemaEntry]:
        """
        Return a schema describing configurable settings for the INDI adapter.
        """
        return [
            {
                "name": "host",
                "friendly_name": "INDI Server Host",
                "type": "str",
                "default": "localhost",
                "description": "INDI server hostname or IP address",
                "required": True,
                "placeholder": "localhost or 192.168.1.100",
                "group": "Connection",
            },
            {
                "name": "port",
                "friendly_name": "INDI Server Port",
                "type": "int",
                "default": 7624,
                "description": "INDI server port",
                "required": True,
                "placeholder": "7624",
                "min": 1,
                "max": 65535,
                "group": "Connection",
            },
            {
                "name": "telescope_name",
                "friendly_name": "Telescope Device Name",
                "type": "str",
                "default": "",
                "description": "Name of the telescope device (leave empty to auto-detect)",
                "required": False,
                "placeholder": "Telescope Simulator",
                "group": "Devices",
            },
            {
                "name": "camera_name",
                "friendly_name": "Camera Device Name",
                "type": "str",
                "default": "",
                "description": "Name of the camera device (leave empty to auto-detect)",
                "required": False,
                "placeholder": "CCD Simulator",
                "group": "Devices",
            },
        ]

    def newDevice(self, d):
        """Emmited when a new device is created from INDI server."""
        self.logger.info(f"new device {d.getDeviceName()}")
        # TODO: if it's the scope we want, set our_scope

    def removeDevice(self, d):
        """Emmited when a device is deleted from INDI server."""
        self.logger.info(f"remove device {d.getDeviceName()}")
        # TODO: if it's our_scope, set our_scope to None, and react accordingly

    def newProperty(self, p):
        """Emmited when a new property is created for an INDI driver."""
        self.logger.debug(f"new property {p.getName()} as {p.getTypeAsString()} for device {p.getDeviceName()}")

    def updateProperty(self, p):
        """Emmited when a new property value arrives from INDI server."""
        self.logger.debug(f"update property {p.getName()} as {p.getTypeAsString()} for device {p.getDeviceName()}")
        try:

            if (
                hasattr(self, "our_scope")
                and self.our_scope is not None
                and p.getDeviceName() == self.our_scope.getDeviceName()
            ):
                value = None
                changed_type = p.getTypeAsString()
                if changed_type == "INDI_TEXT":
                    value = self.our_scope.getText(p.getName())[0].value
                if changed_type == "INDI_NUMBER":
                    value = self.our_scope.getNumber(p.getName())[0].value
                self.logger.debug(
                    f"Scope '{self.our_scope.getDeviceName()}' property {p.getName()} updated value: {value}"
                )

            if p.getType() == PyIndi.INDI_BLOB:
                blobProperty = self.our_camera.getBLOB(p.getName())
                format = blobProperty[0].getFormat()
                bloblen = blobProperty[0].getBlobLen()
                size = blobProperty[0].getSize()
                self.logger.debug(f"Received BLOB of format {format}, size {size}, length {bloblen}")

                # if there's a task underway, save the image to a file
                if self._current_task_id != "":
                    self._last_saved_filename = str(self.images_dir / f"citra_task_{self._current_task_id}_image.fits")
                    for b in blobProperty:
                        with open(self._last_saved_filename, "wb") as f:
                            f.write(b.getblobdata())
                            self.logger.info(f"Saved {self._last_saved_filename}")
                    self._current_task_id = ""
        except Exception as e:
            self.logger.error(f"Error processing updated property {p.getName()}: {e}")

    def removeProperty(self, p):
        """Emmited when a property is deleted for an INDI driver."""
        self.logger.debug(f"remove property {p.getName()} as {p.getTypeAsString()} for device {p.getDeviceName()}")

    def newMessage(self, d, m):
        """Emmited when a new message arrives from INDI server."""
        msg = d.messageQueue(m)
        if "error" in msg.lower():
            self.logger.error(f"new Message {msg}")
        else:
            self.logger.debug(f"new Message {msg}")

    def serverConnected(self):
        """Emmited when the server is connected."""
        self.logger.info(f"INDI Server connected ({self.getHost()}:{self.getPort()})")

    def serverDisconnected(self, code):
        """Emmited when the server gets disconnected."""
        self.logger.info(f"INDI Server disconnected (exit code = {code},{self.getHost()}:{self.getPort()})")

    def newBLOB(self, bp):
        """INDI BLOB arrival callback.

        Actual image persistence happens in :meth:`updateProperty` where we
        have access to the current task id and can write into the sensor's
        ``images_dir``. Writing ``image.fits`` into the process CWD here
        would clobber between multiple INDI sensors sharing one daemon.
        """
        try:
            count = sum(1 for _ in bp)
        except Exception:
            count = 0
        self.logger.debug("newBLOB received %d payload(s); handled in updateProperty", count)

    # ========================= AstroHardwareAdapter Methods =========================

    def connect(self) -> bool:
        self.setServer(self.host, self.port)
        connected = self.connectServer()

        if not connected:
            self.logger.error("Failed to connect to INDI server")
            return False

        time.sleep(1)  # Give server time to enumerate devices

        # Auto-select telescope if name provided, or auto-detect first telescope
        if self.telescope_name:
            device_list = self.list_devices()
            if self.telescope_name not in device_list:
                self.logger.error(f"Could not find configured telescope ({self.telescope_name}) on INDI server.")
                return False

            if not self.select_telescope(self.telescope_name):
                self.logger.error(f"Failed to select telescope: {self.telescope_name}")
                return False
            self.logger.info(f"Found and connected to telescope: {self.telescope_name}")
        else:
            # Auto-detect: try to find and select first telescope device
            telescope_found = self._auto_detect_telescope()
            if not telescope_found:
                self.logger.warning(
                    "No telescope_name configured and auto-detection failed. "
                    "Please configure telescope_name in adapter_settings."
                )
                # Don't return False here - allow operation to continue for cameras-only setups

        # Auto-select camera if name provided, or auto-detect first camera
        if self.camera_name:
            device_list = self.list_devices()
            if self.camera_name not in device_list:
                self.logger.error(f"Could not find configured camera ({self.camera_name}) on INDI server.")
                return False

            if not self.select_camera(self.camera_name):
                self.logger.error(f"Failed to select camera: {self.camera_name}")
                return False
            self.logger.info(f"Found and connected to camera: {self.camera_name}")
        else:
            # Auto-detect: try to find and select first camera device
            camera_found = self._auto_detect_camera()
            if not camera_found:
                self.logger.warning(
                    "No camera_name configured and auto-detection failed. "
                    "Please configure camera_name in adapter_settings."
                )
                # Don't return False here - allow operation to continue

        return True

    def list_devices(self):
        names = []
        for device in self.getDevices():
            names.append(device.getDeviceName())
        return names

    def _auto_detect_telescope(self) -> bool:
        """Auto-detect and select the first available telescope device.

        Returns:
            True if a telescope was found and selected, False otherwise.
        """
        devices = self.getDevices()
        for device in devices:
            device_name = device.getDeviceName()
            # Check if device has EQUATORIAL_EOD_COORD property (indicates it's a telescope)
            # We need to wait a bit for properties to be available
            for _ in range(5):
                telescope_radec = device.getNumber("EQUATORIAL_EOD_COORD")
                if telescope_radec:
                    self.logger.info(f"Auto-detected telescope device: {device_name}")
                    if self.select_telescope(device_name):
                        return True
                    break
                time.sleep(0.5)

        self.logger.debug("No telescope device auto-detected")
        return False

    def _auto_detect_camera(self) -> bool:
        """Auto-detect and select the first available camera device.

        Returns:
            True if a camera was found and selected, False otherwise.
        """
        devices = self.getDevices()
        for device in devices:
            device_name = device.getDeviceName()
            # Check if device has CCD_EXPOSURE property (indicates it's a camera)
            # We need to wait a bit for properties to be available
            for _ in range(5):
                ccd_exposure = device.getNumber("CCD_EXPOSURE")
                if ccd_exposure:
                    self.logger.info(f"Auto-detected camera device: {device_name}")
                    if self.select_camera(device_name):
                        return True
                    break
                time.sleep(0.5)

        self.logger.debug("No camera device auto-detected")
        return False

    def select_telescope(self, device_name: str) -> bool:
        """Select a specific telescope by name and wait for it to be ready.

        Args:
            device_name: Name of the telescope device to select

        Returns:
            True if telescope was found and is ready, False otherwise
        """
        devices = self.getDevices()
        for device in devices:
            if device.getDeviceName() == device_name:
                self.our_scope = device
                self.logger.debug(f"Found telescope device: {device_name}")

                # Connect the telescope device if not already connected
                connect_prop = device.getSwitch("CONNECTION")
                if connect_prop:
                    if not device.isConnected():
                        self.logger.info(f"Connecting telescope device: {device_name}")
                        connect_prop[0].setState(PyIndi.ISS_ON)  # CONNECT
                        connect_prop[1].setState(PyIndi.ISS_OFF)  # DISCONNECT
                        self.sendNewSwitch(connect_prop)
                        time.sleep(1)  # Give device time to connect
                    else:
                        self.logger.debug(f"Telescope device {device_name} already connected")
                else:
                    self.logger.warning(f"Telescope device {device_name} has no CONNECTION property")

                # Wait for EQUATORIAL_EOD_COORD property to be available
                for attempt in range(10):
                    telescope_radec = device.getNumber("EQUATORIAL_EOD_COORD")
                    if telescope_radec and len(telescope_radec) >= 2:
                        self.logger.info(f"Telescope {device_name} is ready (EQUATORIAL_EOD_COORD available)")
                        return True
                    self.logger.debug(f"Waiting for EQUATORIAL_EOD_COORD property... ({attempt + 1}/10)")
                    time.sleep(0.5)

                self.logger.error(
                    f"Telescope {device_name} selected but EQUATORIAL_EOD_COORD property not available. "
                    f"Device may not be fully initialized."
                )
                return False

        self.logger.error(f"Telescope device '{device_name}' not found in INDI device list")
        return False

    def disconnect(self):
        self.disconnectServer()

    def is_telescope_connected(self) -> bool:
        """Check if telescope is connected and responsive."""
        if not self.isServerConnected():
            return False
        if not hasattr(self, "our_scope") or self.our_scope is None:
            return False
        # Check if we can read the telescope position (indicates it's working)
        try:
            telescope_radec = self.our_scope.getNumber("EQUATORIAL_EOD_COORD")
            return telescope_radec is not None and len(telescope_radec) >= 2
        except Exception:
            return False

    def is_camera_connected(self) -> bool:
        """Check if camera is connected and responsive."""
        if not self.isServerConnected():
            return False
        if not hasattr(self, "our_camera") or self.our_camera is None:
            return False
        # Check if camera is connected via INDI
        try:
            if not self.our_camera.isConnected():
                return False
            # Check if we can read the exposure property (indicates it's working)
            ccd_exposure = self.our_camera.getNumber("CCD_EXPOSURE")
            return ccd_exposure is not None
        except Exception:
            return False

    def _do_point_telescope(self, ra: float, dec: float):
        """Hardware-specific implementation to point the telescope to the specified RA/Dec coordinates."""
        # Check if connected to INDI server first
        if not self.isServerConnected():
            self.logger.debug("Not connected to INDI server")
            return

        # Check if telescope is selected
        if not hasattr(self, "our_scope") or self.our_scope is None:
            self.logger.error(
                "No telescope selected. Please configure 'telescope_name' in adapter_settings, "
                "or ensure your INDI server has a telescope device with EQUATORIAL_EOD_COORD property."
            )
            return

        # Get the property
        telescope_radec = self.our_scope.getNumber("EQUATORIAL_EOD_COORD")

        # Check if property exists and is valid
        if not telescope_radec:
            self.logger.error("EQUATORIAL_EOD_COORD property not found on telescope")
            return

        # Check if property is ready (not busy)
        if telescope_radec.getState() == PyIndi.IPS_BUSY:
            self.logger.warning("Telescope is currently busy, waiting for it to be ready...")
            # Could add a wait loop here if needed

        # Check if property has the expected number of elements
        if len(telescope_radec) < 2:
            self.logger.error(f"EQUATORIAL_EOD_COORD has {len(telescope_radec)} elements, expected 2")
            return

        ra_hours = float(ra) / 15.0
        new_dec = float(dec)
        telescope_radec[0].setValue(ra_hours)  # RA converted from degrees to hours for INDI
        telescope_radec[1].setValue(new_dec)  # DEC in degrees

        try:
            self.sendNewNumber(telescope_radec)
            self.logger.info(f"Sent telescope coordinates: RA={ra_hours:.4f}h ({ra:.4f}°), DEC={new_dec}°")
        except Exception as e:
            self.logger.error(f"Error sending new RA/DEC to telescope: {e}")
            return

    def get_telescope_direction(self) -> tuple[float, float]:
        """Read the current telescope direction (RA degrees, DEC degrees)."""
        # Check if connected to INDI server first
        if not self.isServerConnected():
            self.logger.debug("Not connected to INDI server")
            return (0.0, 0.0)

        # Check if telescope is selected
        if not hasattr(self, "our_scope") or self.our_scope is None:
            self.logger.error(
                "No telescope selected. Please configure 'telescope_name' in adapter_settings, "
                "or ensure your INDI server has a telescope device with EQUATORIAL_EOD_COORD property."
            )
            return (0.0, 0.0)

        telescope_radec = self.our_scope.getNumber("EQUATORIAL_EOD_COORD")

        if not telescope_radec:
            self.logger.error("EQUATORIAL_EOD_COORD property not found on telescope")
            self.logger.error("Could not read telescope coordinates")
            return (0.0, 0.0)

        if len(telescope_radec) < 2:
            self.logger.error(f"EQUATORIAL_EOD_COORD has {len(telescope_radec)} elements, expected 2")
            self.logger.error("Could not read telescope coordinates")
            return (0.0, 0.0)

        self.logger.debug(
            f"Telescope currently pointed to RA: {telescope_radec[0].value * 15.0} degrees, "
            f"DEC: {telescope_radec[1].value} degrees"
        )
        return telescope_radec[0].value * 15.0, telescope_radec[1].value

    def telescope_is_moving(self) -> bool:
        """Check if the telescope is currently moving."""
        # Check if connected to INDI server first
        if not self.isServerConnected():
            return False

        if not hasattr(self, "our_scope") or self.our_scope is None:
            return False

        telescope_radec = self.our_scope.getNumber("EQUATORIAL_EOD_COORD")

        if not telescope_radec:
            return False

        return telescope_radec.getState() == PyIndi.IPS_BUSY

    def select_camera(self, device_name: str) -> bool:
        """Select a specific camera by name."""
        devices = self.getDevices()
        for device in devices:
            if device.getDeviceName() == device_name:
                self.our_camera = device
                self.setBLOBMode(PyIndi.B_ALSO, device_name, None)

                # Connect the camera device if not already connected
                connect_prop = device.getSwitch("CONNECTION")
                if connect_prop:
                    if not device.isConnected():
                        self.logger.info(f"Connecting camera device: {device_name}")
                        self._set_switch(device, "CONNECTION", "CONNECT")
                        time.sleep(2)  # Give device time to connect
                    else:
                        self.logger.debug(f"Camera device {device_name} already connected")

                # Configure camera parameters
                self._configure_camera_params()

                return True
        return False

    def _configure_camera_params(self):
        """Initialize camera parameters to match EKOS behavior."""
        if not hasattr(self, "our_camera") or self.our_camera is None:
            return

        self._set_switch(self.our_camera, "CONNECTION", "CONNECT")
        self._set_switch(self.our_camera, "CCD_FRAME_TYPE", "FRAME_LIGHT")
        self._set_switch(self.our_camera, "UPLOAD_MODE", "UPLOAD_CLIENT")
        self._set_switch(
            self.our_camera, "CCD_TRANSFER_FORMAT", "FORMAT_FITS"
        )  # No ISwitch '' in CCD Simulator.CCD_TRANSFER_FORMAT..?
        self._set_switch(self.our_camera, "CCD_COMPRESSION", "INDI_DISABLED")

        self._set_numbers(
            self.our_camera,
            "CCD_BINNING",
            {
                "HOR_BIN": 1,
                "VER_BIN": 1,
            },
        )

        # Get CCD_INFO to find max dimensions
        ccd_info = self.our_camera.getNumber("CCD_INFO")
        max_x = None
        max_y = None
        if ccd_info:
            for item in ccd_info:
                if item.getName() == "CCD_MAX_X":
                    max_x = item.value
                elif item.getName() == "CCD_MAX_Y":
                    max_y = item.value

        if max_x and max_y:
            self._set_numbers(
                self.our_camera,
                "CCD_FRAME",
                {
                    "X": 0,
                    "Y": 0,
                    "WIDTH": max_x,
                    "HEIGHT": max_y,
                },
            )

    def _set_switch(self, device, property_name: str, switch_name: str) -> bool:
        svp = device.getSwitch(property_name)
        if svp is None:
            return False

        available_names = [item.getName() for item in svp]
        if switch_name not in available_names:
            self.logger.warning(f"INDI Switch '{property_name}' only supports {available_names}, not '{switch_name}'")
            return False

        # Turn everything OFF
        for item in svp:
            item.setState(PyIndi.ISS_OFF)

        # Turn the desired one ON
        matched = False
        for item in svp:
            item_name = item.getName()
            if item_name == switch_name:
                item.setState(PyIndi.ISS_ON)
                matched = True
                break

        if not matched:
            return False

        self.sendNewSwitch(svp)
        return True

    def _set_numbers(self, device, property_name: str, values: dict) -> bool:
        """
        values: { "ELEMENT_NAME": value, ... }
        """
        nvp = device.getNumber(property_name)
        if nvp is None:
            return False

        # Map element names → items
        items = {item.getName(): item for item in nvp}

        # Ensure all requested elements exist
        for name in values.keys():
            if name not in items:
                return False  # or raise

        # Set values
        for name, val in values.items():
            items[name].setValue(float(val))

        self.sendNewNumber(nvp)
        return True

    def take_image(self, task_id: str, exposure_duration_seconds=1.0):
        """Capture an image with the currently selected camera."""

        # Check if camera is selected
        if not hasattr(self, "our_camera") or self.our_camera is None:
            self.logger.error("No camera selected. Call select_camera() first.")
            return None

        # Get the CCD_EXPOSURE property
        ccd_exposure = self.our_camera.getNumber("CCD_EXPOSURE")

        # Check if property exists and is valid
        if not ccd_exposure:
            self.logger.error("CCD_EXPOSURE property not found on camera")
            return None

        # Check if property has at least one element
        if len(ccd_exposure) < 1:
            self.logger.error(f"CCD_EXPOSURE has {len(ccd_exposure)} elements, expected at least 1")
            return None

        self.logger.info(f"Taking {exposure_duration_seconds} second exposure...")
        self._current_task_id = task_id

        try:
            ccd_exposure[0].setValue(exposure_duration_seconds)
            self.sendNewNumber(ccd_exposure)
        except Exception as e:
            self.logger.error(f"Error sending exposure command to camera: {e}")
            self._current_task_id = ""
            return None

        while self.is_camera_busy() and self._current_task_id != "":
            self.logger.debug("Waiting for camera to finish exposure...")
            time.sleep(0.2)

        filename = self._last_saved_filename
        self._last_saved_filename = ""
        return filename

    def is_camera_busy(self) -> bool:
        """Check if the camera is currently busy taking an image."""
        # Check if camera is selected
        if not hasattr(self, "our_camera") or self.our_camera is None:
            return False

        ccd_exposure = self.our_camera.getNumber("CCD_EXPOSURE")

        # Check if property exists
        if not ccd_exposure:
            return False

        return ccd_exposure.getState() == PyIndi.IPS_BUSY

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        """Set the tracking rate for the telescope in RA and Dec (arcseconds per second)."""
        self.logger.info(f"Setting tracking rate: RA {ra_rate} arcseconds/s, Dec {dec_rate} arcseconds/s")
        try:

            track_state_prop = self.our_scope.getSwitch("TELESCOPE_TRACK_STATE")
            track_state_prop[0].setState(PyIndi.ISS_OFF)
            self.sendNewSwitch(track_state_prop)

            track_mode_prop = self.our_scope.getSwitch("TELESCOPE_TRACK_MODE")
            track_mode_prop[0].setState(PyIndi.ISS_OFF)  # TRACK_SIDEREAL
            track_mode_prop[1].setState(PyIndi.ISS_OFF)  # TRACK_SOLAR
            track_mode_prop[2].setState(PyIndi.ISS_OFF)  # TRACK_LUNAR
            track_mode_prop[3].setState(PyIndi.ISS_ON)  # TRACK_CUSTOM
            self.sendNewSwitch(track_mode_prop)

            indi_tracking_rate = self.our_scope.getNumber("TELESCOPE_TRACK_RATE")
            self.logger.info(
                f"Current INDI tracking rates: 0: {indi_tracking_rate[0].value} 1: {indi_tracking_rate[1].value}"
            )
            indi_tracking_rate[0].setValue(ra_rate)
            indi_tracking_rate[1].setValue(dec_rate)
            self.sendNewNumber(indi_tracking_rate)

            track_state_prop[0].setState(PyIndi.ISS_ON)  # Turn tracking ON
            self.sendNewSwitch(track_state_prop)
            return True

        except Exception as e:
            self.logger.error(f"Error setting tracking rates: {e}")
            return False

    def get_tracking_rate(self) -> tuple[float, float]:
        """Get the current tracking rate for the telescope in RA and Dec (arcseconds per second)."""
        ra_rate = self.our_scope.getNumber("TELESCOPE_TRACK_RATE_RA")[0].value
        dec_rate = self.our_scope.getNumber("TELESCOPE_TRACK_RATE_DEC")[0].value
        return ra_rate, dec_rate

    def perform_alignment(self, target_ra: float, target_dec: float) -> bool:
        """Alignment is handled by INDI/KStars, not by CitraSense.

        Pointing corrections come via ``update_from_plate_solve()`` after each
        observation image is plate-solved in the processing pipeline.
        """
        return True

    def update_from_plate_solve(
        self,
        solved_ra_deg: float,
        solved_dec_deg: float,
        expected_ra_deg: float | None = None,
        expected_dec_deg: float | None = None,
        target_ra_deg: float | None = None,
        target_dec_deg: float | None = None,
    ) -> None:
        """Update alignment offsets from pipeline plate solve result."""
        if expected_ra_deg is not None and expected_dec_deg is not None:
            self._alignment_offset_ra = solved_ra_deg - expected_ra_deg
            self._alignment_offset_dec = solved_dec_deg - expected_dec_deg
            self.logger.info(
                f"Alignment offsets updated from plate solve: RA offset={self._alignment_offset_ra:.4f} deg, "
                f"Dec offset={self._alignment_offset_dec:.4f} deg"
            )

    def get_observation_strategy(self) -> ObservationStrategy:
        return ObservationStrategy.MANUAL

    def perform_observation_sequence(self, task, satellite_data) -> list[str]:
        raise NotImplementedError
