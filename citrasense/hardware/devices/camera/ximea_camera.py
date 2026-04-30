"""Ximea hyperspectral imaging camera adapter."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from citrasense.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrasense.hardware.devices.camera import AbstractCamera


class XimeaHyperspectralCamera(AbstractCamera):
    """Adapter for Ximea hyperspectral imaging cameras.

    Supports Ximea MQ series cameras with snapshot mosaic hyperspectral sensors.
    Requires ximea-api package (xiAPI Python wrapper).

    Configuration:
        serial_number (str): Camera serial number for multi-camera setups
        default_gain (float): Default gain in dB
        default_exposure_ms (float): Default exposure time in milliseconds
        spectral_bands (int): Number of spectral bands (e.g., 16 for SM4X4, 25 for SM5X5)
        data_mode (str): Data structure - 'raw' (2D mosaic) or 'datacube' (3D separated bands)
        vertical_flip (bool): Flip image vertically
        horizontal_flip (bool): Flip image horizontally

    Note: Always outputs FITS format files regardless of data_mode.
    """

    @classmethod
    def get_friendly_name(cls) -> str:
        """Return human-readable name for this camera device.

        Returns:
            Friendly display name
        """
        return "Ximea Hyperspectral Camera (MQ Series)"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        """Return required Python packages.

        Returns:
            Dict with packages and install extra
        """
        return {
            "packages": ["ximea"],
            "install_extra": "ximea",
        }

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        """Return schema for Ximea camera settings.

        Returns:
            List of setting schema entries (without 'camera_' prefix)
        """
        schema = [
            {
                "name": "serial_number",
                "friendly_name": "Camera Serial Number",
                "type": "str",
                "default": "",
                "description": "Camera serial number (for multi-camera setups)",
                "required": False,
                "placeholder": "Leave empty to auto-detect",
                "group": "Camera",
            },
            {
                "name": "default_gain",
                "friendly_name": "Default Gain (dB)",
                "type": "float",
                "default": 0.0,
                "description": "Default camera gain setting in dB",
                "required": False,
                "min": 0.0,
                "max": 24.0,
                "group": "Camera",
            },
            {
                "name": "default_exposure_ms",
                "friendly_name": "Default Exposure (ms)",
                "type": "float",
                "default": 100.0,
                "description": "Default exposure time in milliseconds (e.g., 300 = 0.3 seconds)",
                "required": False,
                "min": 0.001,
                "max": 1000.0,  # Hardware limitation: ~1 second max for MQ series
                "group": "Camera",
            },
            {
                "name": "spectral_bands",
                "friendly_name": "Spectral Bands",
                "type": "int",
                "default": 25,
                "description": "Number of spectral bands (e.g., 25 for MQ022HG-IM-SM5X5)",
                "required": False,
                "min": 1,
                "max": 500,
                "group": "Camera",
            },
            {
                "name": "data_mode",
                "friendly_name": "Data Mode",
                "type": "str",
                "default": "raw",
                "description": "Data structure: raw (2D mosaic) or datacube (3D separated bands)",
                "required": False,
                "options": ["raw", "datacube"],
                "group": "Camera",
            },
            {
                "name": "vertical_flip",
                "friendly_name": "Vertical Flip",
                "type": "bool",
                "default": False,
                "description": "Flip image vertically (upside down)",
                "required": False,
                "group": "Camera",
            },
            {
                "name": "horizontal_flip",
                "friendly_name": "Horizontal Flip",
                "type": "bool",
                "default": False,
                "description": "Flip image horizontally (mirror)",
                "required": False,
                "group": "Camera",
            },
            {
                "name": "wavelength_calibration",
                "friendly_name": "Wavelength Calibration (nm)",
                "type": "str",
                "default": "",
                "description": (
                    "Comma-separated wavelengths in nanometers for each spectral band " "(e.g., 470,520,570,620)"
                ),
                "required": False,
                "placeholder": "Leave empty if not calibrated",
                "group": "Camera",
            },
        ]
        return cast(list[SettingSchemaEntry], schema)

    def __init__(self, logger: logging.Logger, **kwargs):
        """Initialize the Ximea camera.

        Args:
            logger: Logger instance for this device
            **kwargs: Configuration including serial_number, default_gain, etc.
        """
        super().__init__(logger, **kwargs)

        self.serial_number: str | None = kwargs.get("serial_number")
        self.default_gain: float = kwargs.get("default_gain", 0.0)
        self.default_exposure_ms: float = kwargs.get("default_exposure_ms", 100.0)
        self.spectral_bands: int = kwargs.get("spectral_bands", 25)
        self.data_mode: str = kwargs.get("data_mode", "raw")  # raw mosaic or datacube
        self.vertical_flip: bool = kwargs.get("vertical_flip", False)
        self.horizontal_flip: bool = kwargs.get("horizontal_flip", False)

        # Parse wavelength calibration (comma-separated string to list of floats)
        wavelength_str = kwargs.get("wavelength_calibration", "")
        self.wavelength_calibration: list[float] = []
        if wavelength_str:
            try:
                self.wavelength_calibration = [float(w.strip()) for w in wavelength_str.split(",")]
                if len(self.wavelength_calibration) != self.spectral_bands:
                    self.logger.warning(
                        f"Wavelength calibration has {len(self.wavelength_calibration)} values "
                        f"but spectral_bands is {self.spectral_bands}. Calibration will not be used."
                    )
                    self.wavelength_calibration = []
            except ValueError as e:
                self.logger.warning(f"Invalid wavelength_calibration format: {e}. Expected comma-separated numbers.")
                self.wavelength_calibration = []

        # Camera handle (will be initialized on connect)
        self._camera = None
        self._is_connected = False
        self._last_exposure_start: datetime | None = None

        # Camera info cache
        self._camera_info = {}

    @property
    def _cam(self) -> Any:
        """Typed-as-Any accessor for SDK calls that lack stubs."""
        return self._camera

    def connect(self) -> bool:
        """Connect to the Ximea camera.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import ximea API (lazy import to avoid hard dependency)
            try:
                from ximea import xiapi  # type: ignore[reportMissingImports]
            except ImportError:
                self.logger.error(
                    "XIMEA Python bindings not found. Installation instructions:\n"
                    "1. Download and mount XIMEA macOS Software Package from:\n"
                    "   https://www.ximea.com/support/wiki/apis/XIMEA_macOS_Software_Package\n"
                    "2. Run the installer on the mounted volume\n"
                    "3. Copy Python bindings to your venv:\n"
                    "   cp -r /Volumes/XIMEA/Examples/xiPython/v3/ximea $VIRTUAL_ENV/lib/python*/site-packages/\n"
                    "4. Verify installation: python -c 'import ximea; print(ximea.__version__)'\n"
                    "Note: The XIMEA bindings are not available via pip and must be installed manually."
                )
                return False

            self.logger.info("Connecting to Ximea hyperspectral camera...")
            self.logger.debug("Ximea xiapi module imported successfully")

            # Create camera instance
            self.logger.debug("Creating Ximea camera instance...")
            self._camera = xiapi.Camera()
            self.logger.debug("Camera instance created")

            # Open camera (by serial number if specified)
            if self.serial_number:
                self.logger.info(f"Opening camera with serial number: {self.serial_number}")
                try:
                    self._camera.open_device_by_SN(self.serial_number)
                    self.logger.debug(f"Camera with SN {self.serial_number} opened successfully")
                except Exception as e:
                    self.logger.error(f"Failed to open camera by serial number {self.serial_number}: {e}")
                    raise
            else:
                self.logger.info("Opening first available Ximea camera (no serial number specified)")
                try:
                    self._camera.open_device()
                    self.logger.debug("First available camera opened successfully")
                except Exception as e:
                    self.logger.error(f"Failed to open first available camera: {e}")
                    self.logger.info("Make sure camera is connected and no other application is using it")
                    raise

            # Configure camera
            self.logger.debug("Configuring camera...")
            self._configure_camera()
            self.logger.debug("Camera configuration complete")

            # Cache camera info
            self.logger.debug("Reading camera info...")
            self._camera_info = self._read_camera_info()
            self.logger.debug(f"Camera info: {self._camera_info}")

            self._is_connected = True
            self.logger.info(
                f"Connected to Ximea camera: {self._camera_info.get('model', 'Unknown')} "
                f"(SN: {self._camera_info.get('serial_number', 'Unknown')})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Ximea camera: {e}", exc_info=True)
            self._is_connected = False
            if self._camera is not None:
                try:
                    self._camera.close_device()
                except Exception as close_error:
                    self.logger.debug(f"Error closing camera after failed connection: {close_error}")
                self._camera = None
            return False

    def disconnect(self):
        """Disconnect from the Ximea camera."""
        if self._camera is not None:
            try:
                self.logger.info("Disconnecting from Ximea camera...")
                self._camera.close_device()
                self._is_connected = False
                self.logger.info("Ximea camera disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting from Ximea camera: {e}")
            finally:
                self._camera = None

    def is_connected(self) -> bool:
        """Check if camera is connected and responsive.

        Returns:
            True if connected, False otherwise
        """
        return self._is_connected and self._camera is not None

    def capture_array(
        self,
        duration: float,
        gain: int | float | None = None,
        offset: int | None = None,
        binning: int = 1,
        shutter_closed: bool = False,
    ) -> np.ndarray:
        if not self.is_connected():
            raise RuntimeError("Camera not connected")

        try:
            from ximea import xiapi  # type: ignore[reportMissingImports]
        except ImportError as e:
            raise RuntimeError("ximea-api package not installed") from e

        assert self._camera is not None

        self.logger.debug(
            f"Starting hyperspectral exposure: {duration}s, "
            f"gain={gain if gain is not None else self.default_gain}dB, "
            f"binning={binning}x{binning}"
        )

        exposure_us = duration * 1000000.0
        self._camera.set_exposure(int(exposure_us))
        self._last_exposure_us = float(self._cam.get_exposure())

        gain_to_use = gain if gain is not None else self.default_gain
        try:
            min_gain = float(self._cam.get_gain_minimum())
            max_gain = float(self._cam.get_gain_maximum())
            if gain_to_use < min_gain or gain_to_use > max_gain:
                self.logger.warning(
                    f"Requested gain {gain_to_use} dB outside valid range " f"[{min_gain}, {max_gain}] dB. Clamping."
                )
                gain_to_use = max(min_gain, min(gain_to_use, max_gain))
            self._camera.set_gain(float(gain_to_use))
            self._last_gain_db = float(self._cam.get_gain())
        except Exception as e:
            self.logger.warning(f"Could not set gain: {e}. Continuing with current setting.")

        if binning > 1:
            self._camera.set_downsampling(str(binning))

        img = xiapi.Image()
        self._camera.start_acquisition()

        try:
            timeout_ms = int((exposure_us / 1000.0) + 5000)
            self._last_exposure_start = datetime.now(timezone.utc)
            self._camera.get_image(img, timeout=timeout_ms)
            data: np.ndarray = img.get_image_data_numpy()
            return data
        finally:
            self._camera.stop_acquisition()
            if binning > 1:
                self._camera.set_downsampling("1")

    def take_exposure(
        self,
        duration: float,
        gain: int | float | None = None,
        offset: int | None = None,
        binning: int = 1,
        save_path: Path | None = None,
        shutter_closed: bool = False,
    ) -> Path:
        data = self.capture_array(duration, gain, offset, binning, shutter_closed=shutter_closed)

        if save_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"ximea_hyperspectral_{timestamp}.tiff")

        self._save_hyperspectral_data(data, save_path)
        self.logger.info(f"Hyperspectral image saved to: {save_path}")
        return save_path

    def abort_exposure(self):
        """Abort the current exposure if one is in progress."""
        if self.is_connected() and self._camera is not None:
            try:
                self._camera.stop_acquisition()
                self.logger.info("Ximea exposure aborted")
            except Exception as e:
                self.logger.error(f"Error aborting exposure: {e}")

    def get_temperature(self) -> float | None:
        """Get the current camera sensor temperature.

        Returns:
            Temperature in degrees Celsius, or None if not available
        """
        if not self.is_connected():
            return None

        assert self._camera is not None

        try:
            return float(self._cam.get_temp())
        except Exception as e:
            self.logger.warning(f"Could not read camera temperature: {e}")
            return None

    def set_temperature(self, temperature: float) -> bool:
        """Set the target camera sensor temperature.

        Note: Most Ximea cameras do not support active cooling.

        Args:
            temperature: Target temperature in degrees Celsius

        Returns:
            False (Ximea cameras typically don't support temperature control)
        """
        self.logger.warning("Ximea cameras do not support temperature control")
        return False

    def start_cooling(self) -> bool:
        """Enable camera cooling system.

        Returns:
            False (Ximea cameras typically don't have active cooling)
        """
        self.logger.warning("Ximea cameras do not have active cooling")
        return False

    def stop_cooling(self) -> bool:
        """Disable camera cooling system.

        Returns:
            False (Ximea cameras typically don't have active cooling)
        """
        return False

    def is_hyperspectral(self) -> bool:
        """Indicates whether this camera captures hyperspectral data.

        Returns:
            bool: True (Ximea MQ cameras are hyperspectral)
        """
        return True

    def get_preferred_file_extension(self) -> str:
        """Get the preferred file extension for saved images.

        Ximea hyperspectral cameras always output FITS format.

        Returns:
            'fits' - FITS format with hyperspectral metadata
        """
        return "fits"

    def get_camera_info(self) -> dict:
        """Get camera capabilities and information.

        Returns:
            Dictionary containing camera specs
        """
        return self._camera_info.copy()

    # Helper methods

    def _configure_camera(self):
        """Configure camera with default settings."""
        if self._camera is None:
            return

        try:
            # Set default exposure (xiAPI expects microseconds)
            default_exposure_us = self.default_exposure_ms * 1000
            self._camera.set_exposure(int(default_exposure_us))

            # Set default gain
            try:
                min_gain = float(self._cam.get_gain_minimum())
                max_gain = float(self._cam.get_gain_maximum())
                gain_to_set = max(min_gain, min(self.default_gain, max_gain))
                self._camera.set_gain(gain_to_set)
                self.logger.debug(f"Gain set to {gain_to_set} dB (range: {min_gain}-{max_gain} dB)")
            except Exception as e:
                self.logger.warning(f"Could not set default gain: {e}")

            # Set image format
            # For hyperspectral, typically use RAW16 or RAW8
            self._camera.set_imgdataformat("XI_RAW16")

            # Set image orientation
            if self.vertical_flip:
                self.logger.debug("Enabling vertical flip")
                self._camera.enable_vertical_flip()
            else:
                self._camera.disable_vertical_flip()

            if self.horizontal_flip:
                self.logger.debug("Enabling horizontal flip")
                self._camera.enable_horizontal_flip()
            else:
                self._camera.disable_horizontal_flip()

            self.logger.info("Ximea camera configured with default settings")

        except Exception as e:
            self.logger.warning(f"Error configuring camera settings: {e}")

    def _read_camera_info(self) -> dict:
        """Read camera information and capabilities."""
        info = {}

        if self._camera is None:
            return info

        try:
            device_name = self._cam.get_device_name()
            info["model"] = device_name.decode() if isinstance(device_name, bytes) else str(device_name)
            device_sn = self._cam.get_device_sn()
            info["serial_number"] = device_sn.decode() if isinstance(device_sn, bytes) else str(device_sn)
            info["width"] = self._camera.get_width()
            info["height"] = self._camera.get_height()
            info["pixel_size_um"] = 3.45  # MQ series typically 3.45µm
            info["bit_depth"] = 12  # MQ series typically 12-bit
            info["spectral_bands"] = self.spectral_bands
            info["type"] = "hyperspectral"

        except Exception as e:
            self.logger.warning(f"Error reading camera info: {e}")

        return info

    def _save_hyperspectral_data(self, data: np.ndarray, save_path: Path) -> None:
        """Save hyperspectral image data from a numpy array.

        Args:
            data: 2-D numpy array of raw pixel data
            save_path: Path to save the image
        """
        if self.data_mode == "datacube":
            # Create 3D datacube by demosaicing the spectral mosaic
            datacube = self._demosaic_to_datacube(data)

            # Save as multi-extension FITS (one extension per spectral band)
            try:
                # Create primary HDU with basic metadata
                from astropy.io import fits

                primary_hdu = fits.PrimaryHDU()
                # Hyperspectral metadata
                primary_hdu.header["HIERARCH SPECTRAL_TYPE"] = "hyperspectral"
                primary_hdu.header["HIERARCH SPECTRAL_BANDS"] = self.spectral_bands
                primary_hdu.header["HIERARCH DATA_MODE"] = "datacube"
                primary_hdu.header["HIERARCH SENSOR_TYPE"] = "snapshot_mosaic"

                # Capture metadata
                if hasattr(self, "_last_exposure_us"):
                    primary_hdu.header["EXPTIME"] = self._last_exposure_us / 1000000.0  # seconds
                if hasattr(self, "_last_gain_db"):
                    primary_hdu.header["GAIN"] = self._last_gain_db
                date_obs = self._last_exposure_start or datetime.now(timezone.utc)
                primary_hdu.header["DATE-OBS"] = date_obs.isoformat()

                # Camera metadata
                if self._camera_info:
                    if "serial_number" in self._camera_info:
                        primary_hdu.header["CAMSER"] = self._camera_info["serial_number"]
                    if "model" in self._camera_info:
                        primary_hdu.header["INSTRUME"] = self._camera_info["model"]

                # Wavelength calibration metadata in primary header (if available)
                if self.wavelength_calibration:
                    primary_hdu.header["HIERARCH WAVELENGTH_UNIT"] = "nm"
                    primary_hdu.header["HIERARCH WAVELENGTH_COUNT"] = len(self.wavelength_calibration)

                # Create image HDU for each spectral band
                hdu_list = [primary_hdu]
                for i in range(datacube.shape[2]):
                    band_hdu = fits.ImageHDU(datacube[:, :, i], name=f"BAND_{i:03d}")
                    band_hdu.header["BANDNUM"] = i

                    # Add wavelength information if calibrated
                    if self.wavelength_calibration and i < len(self.wavelength_calibration):
                        band_hdu.header["WAVELENG"] = self.wavelength_calibration[i]
                        band_hdu.header["WAVEUNIT"] = "nm"

                    hdu_list.append(band_hdu)  # type: ignore[reportArgumentType]

                hdul = fits.HDUList(hdu_list)
                hdul.writeto(save_path, overwrite=True)
                self.logger.debug(f"Saved hyperspectral datacube as multi-extension FITS: {save_path}")
            except ImportError:
                # Fallback: save as 3D numpy array
                np.save(save_path.with_suffix(".npy"), datacube)
                self.logger.warning("astropy not available, saved datacube as .npy")

        else:  # "raw" or default
            # Save raw mosaic as-is
            self._save_raw_image(data, save_path)

    def _save_raw_image(self, data, save_path: Path):
        """Save raw image data based on file extension.

        Args:
            data: Numpy array of image data
            save_path: Path to save the image
        """
        import numpy as np

        suffix = save_path.suffix.lower()

        if suffix == ".fits":
            try:
                from astropy.io import fits

                hdu = fits.PrimaryHDU(data)
                # Hyperspectral metadata
                hdu.header["HIERARCH SPECTRAL_TYPE"] = "hyperspectral"
                hdu.header["HIERARCH SPECTRAL_BANDS"] = self.spectral_bands
                hdu.header["HIERARCH DATA_MODE"] = "raw"
                hdu.header["HIERARCH SENSOR_TYPE"] = "snapshot_mosaic"

                # Capture metadata
                if hasattr(self, "_last_exposure_us"):
                    hdu.header["EXPTIME"] = self._last_exposure_us / 1000000.0  # seconds
                if hasattr(self, "_last_gain_db"):
                    hdu.header["GAIN"] = self._last_gain_db
                date_obs = self._last_exposure_start or datetime.now(timezone.utc)
                hdu.header["DATE-OBS"] = date_obs.isoformat()

                # Camera metadata
                if self._camera_info:
                    if "serial_number" in self._camera_info:
                        hdu.header["CAMSER"] = self._camera_info["serial_number"]
                    if "model" in self._camera_info:
                        hdu.header["INSTRUME"] = self._camera_info["model"]

                # Wavelength calibration metadata (if available)
                if self.wavelength_calibration:
                    hdu.header["HIERARCH WAVELENGTH_UNIT"] = "nm"
                    hdu.header["HIERARCH WAVELENGTH_COUNT"] = len(self.wavelength_calibration)
                    # Store wavelengths as comma-separated values (FITS has 80 char limit per line)
                    wavelength_str = ",".join(str(w) for w in self.wavelength_calibration)
                    # Split into multiple HIERARCH keywords if needed
                    if len(wavelength_str) <= 68:  # 80 - len('HIERARCH WAVELENGTHS = ')
                        hdu.header["HIERARCH WAVELENGTHS"] = wavelength_str
                    else:
                        # Split into chunks
                        chunk_size = 68
                        for i, chunk_start in enumerate(range(0, len(wavelength_str), chunk_size)):
                            chunk = wavelength_str[chunk_start : chunk_start + chunk_size]
                            hdu.header[f"HIERARCH WAVELENGTHS_{i}"] = chunk

                hdu.writeto(save_path, overwrite=True)
                self.logger.debug(f"Saved hyperspectral image as FITS: {save_path}")
            except ImportError:
                np.save(save_path.with_suffix(".npy"), data)
                self.logger.warning("astropy not available, saved as .npy instead of FITS")
        elif suffix in [".tif", ".tiff"]:
            try:
                from PIL import Image

                pil_img = Image.fromarray(data)
                pil_img.save(save_path)
                self.logger.debug(f"Saved hyperspectral image as TIFF: {save_path}")
            except ImportError:
                np.save(save_path.with_suffix(".npy"), data)
                self.logger.warning("PIL not available, saved as .npy instead of TIFF")
        else:
            # Default: save as numpy array
            np.save(save_path.with_suffix(".npy"), data)
            self.logger.debug(f"Saved hyperspectral image as numpy array: {save_path.with_suffix('.npy')}")

    def _demosaic_to_datacube(self, mosaic_data):
        """Demosaic snapshot mosaic into spectral datacube.

        Extracts individual spectral bands from a snapshot mosaic pattern.
        The mosaic must be a perfect square (N×N pattern) where N² = spectral_bands.

        Args:
            mosaic_data: 2D numpy array with spectral mosaic pattern

        Returns:
            3D numpy array (height, width, bands) with separated spectral bands

        Raises:
            ValueError: If spectral_bands is not a perfect square
        """
        import math

        import numpy as np

        # Calculate pattern size from spectral_bands using square root
        # This works for any N×N pattern (4×4=16, 5×5=25, 6×6=36, 7×7=49, 8×8=64, etc.)
        pattern_size = int(math.sqrt(self.spectral_bands))

        # Validate that spectral_bands is a perfect square
        if pattern_size * pattern_size != self.spectral_bands:
            raise ValueError(
                f"spectral_bands ({self.spectral_bands}) is not a perfect square. "
                f"Snapshot mosaic sensors must have N×N pattern (e.g., 4×4=16, 5×5=25, 6×6=36). "
                f"Nearest valid values: {(pattern_size)**2} ({pattern_size}×{pattern_size}) "
                f"or {(pattern_size+1)**2} ({pattern_size+1}×{pattern_size+1})."
            )
        height, width = mosaic_data.shape[:2]
        out_height = height // pattern_size
        out_width = width // pattern_size
        num_bands = pattern_size * pattern_size

        # Create output datacube
        datacube = np.zeros((out_height, out_width, num_bands), dtype=mosaic_data.dtype)

        # Extract each spectral band from the mosaic
        for band_idx in range(num_bands):
            row_offset = band_idx // pattern_size
            col_offset = band_idx % pattern_size

            # Extract this band's pixels from the mosaic
            band_data = mosaic_data[row_offset::pattern_size, col_offset::pattern_size]

            # Handle size mismatch (if image dimensions aren't perfect multiples)
            datacube[:, :, band_idx] = band_data[:out_height, :out_width]

        self.logger.debug(f"Demosaiced {height}x{width} mosaic into {out_height}x{out_width}x{num_bands} datacube")

        return datacube
