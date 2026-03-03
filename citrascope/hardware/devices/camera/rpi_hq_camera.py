"""Raspberry Pi High Quality Camera adapter."""

import logging
import time
from pathlib import Path
from typing import cast

import numpy as np

from citrascope.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry
from citrascope.hardware.devices.camera import AbstractCamera


class RaspberryPiHQCamera(AbstractCamera):
    """Adapter for Raspberry Pi High Quality Camera (12.3MP IMX477 sensor).

    Uses the picamera2 library for camera control. Supports long exposures
    suitable for astrophotography.

    Configuration:
        default_gain (float): Default analog gain (default: 1.0)
        default_exposure_ms (float): Default exposure time in milliseconds
        output_format (str): Output format - 'fits', 'png', 'jpg', 'raw'
    """

    @classmethod
    def get_friendly_name(cls) -> str:
        """Return human-readable name for this camera device.

        Returns:
            Friendly display name
        """
        return "Raspberry Pi HQ Camera"

    @classmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        """Return required Python packages.

        Returns:
            Dict with packages and install extra
        """
        return {
            "packages": ["picamera2"],
            "install_extra": "rpi",
        }

    @classmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        """Return schema for Raspberry Pi HQ Camera settings.

        Returns:
            List of setting schema entries
        """
        schema = [
            {
                "name": "default_gain",
                "friendly_name": "Default Gain",
                "type": "float",
                "default": 1.0,
                "description": "Default analog gain (1.0-16.0)",
                "required": False,
                "min": 1.0,
                "max": 16.0,
                "group": "Camera",
            },
            {
                "name": "default_exposure_ms",
                "friendly_name": "Default Exposure (ms)",
                "type": "float",
                "default": 1000.0,
                "description": "Default exposure time in milliseconds",
                "required": False,
                "min": 0.1,
                "max": 600000.0,
                "group": "Camera",
            },
            {
                "name": "output_format",
                "friendly_name": "Output Format",
                "type": "str",
                "default": "fits",
                "description": "Image output format",
                "required": False,
                "options": ["fits", "png", "jpg", "raw"],
                "group": "Camera",
            },
        ]
        return cast(list[SettingSchemaEntry], schema)

    def __init__(self, logger: logging.Logger, **kwargs):
        """Initialize the Raspberry Pi HQ Camera.

        Args:
            logger: Logger instance
            **kwargs: Configuration parameters matching the schema
        """
        super().__init__(logger, **kwargs)

        # Hardware specs (fixed by IMX477 sensor)
        self.sensor_width = 4056
        self.sensor_height = 3040

        # Camera settings
        self.default_gain = kwargs.get("default_gain", 1.0)
        self.default_exposure_ms = kwargs.get("default_exposure_ms", 1000.0)
        self.output_format = kwargs.get("output_format", "fits")

        # Camera instance (lazy loaded)
        self._camera = None
        self._connected = False

        # Lazy import picamera2 to avoid hard dependency
        self._picamera2_module = None

    def connect(self) -> bool:
        """Connect to the Raspberry Pi HQ Camera.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Lazy import
            if self._picamera2_module is None:
                import picamera2  # type: ignore

                self._picamera2_module = picamera2

            self.logger.info("Connecting to Raspberry Pi HQ Camera...")

            # Initialize camera
            self._camera = self._picamera2_module.Picamera2()

            # Configure camera for full sensor resolution
            config = self._camera.create_still_configuration(
                main={
                    "size": (self.sensor_width, self.sensor_height),
                    "format": "RGB888",  # We'll convert to desired format later
                },
                buffer_count=2,
            )
            self._camera.configure(config)

            self._camera.start()
            self._connected = True

            self.logger.info(f"Raspberry Pi HQ Camera connected: {self.sensor_width}x{self.sensor_height}")
            return True

        except ImportError:
            self.logger.error("picamera2 library not found. Install with: pip install picamera2")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Raspberry Pi HQ Camera: {e}")
            return False

    def disconnect(self):
        """Disconnect from the Raspberry Pi HQ Camera."""
        if self._camera:
            try:
                self._camera.stop()
                self._camera.close()
                self.logger.info("Raspberry Pi HQ Camera disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting camera: {e}")

        self._camera = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check if camera is connected and responsive.

        Returns:
            True if connected, False otherwise
        """
        return self._connected and self._camera is not None

    def capture_array(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
    ) -> np.ndarray:
        if not self.is_connected():
            raise RuntimeError("Camera not connected")
        if self._camera is None:
            raise RuntimeError("Camera instance is None")

        try:
            actual_gain = gain if gain is not None else self.default_gain
            exposure_us = int(duration * 1_000_000)

            self.logger.info(f"Taking {duration}s exposure, gain={actual_gain}")
            self._camera.set_controls({"ExposureTime": exposure_us, "AnalogueGain": float(actual_gain)})

            request = self._camera.capture_request()
            image_data = request.make_array("main")
            request.release()

            if binning > 1:
                image_data = self._apply_binning(image_data, binning)

            return np.asarray(image_data)

        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            raise RuntimeError(f"Image capture failed: {e}") from e

    def take_exposure(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        save_path: Path | None = None,
    ) -> Path:
        image_data = self.capture_array(duration, gain, offset, binning)

        if save_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"/tmp/rpi_hq_{timestamp}.{self.output_format}")

        if self.output_format == "fits":
            self._save_as_fits(image_data, save_path)
        elif self.output_format == "png":
            self._save_as_png(image_data, save_path)
        elif self.output_format == "jpg":
            self._save_as_jpg(image_data, save_path)
        elif self.output_format == "raw":
            self._save_as_raw(image_data, save_path)

        self.logger.info(f"Image saved to {save_path}")
        return save_path

    def abort_exposure(self):
        """Abort current exposure.

        Note: Picamera2 doesn't support aborting exposures, this is a no-op.
        """
        self.logger.warning("Raspberry Pi camera does not support aborting exposures")

    def get_temperature(self) -> float | None:
        """Get camera sensor temperature.

        Returns:
            Temperature in Celsius, or None if unavailable

        Note: RPi HQ camera does not expose temperature readings via picamera2
        """
        return None

    def _apply_binning(self, image_data, binning: int):
        """Apply pixel binning to reduce resolution and increase sensitivity.

        Args:
            image_data: Image array (H, W, C) or (H, W)
            binning: Binning factor (2 or 4)

        Returns:
            Binned image array
        """

        if binning == 1:
            return image_data

        # Crop to be evenly divisible by binning factor
        h, w = image_data.shape[:2]
        h_crop = (h // binning) * binning
        w_crop = (w // binning) * binning
        cropped = image_data[:h_crop, :w_crop]

        # Reshape and average
        if len(cropped.shape) == 3:  # RGB
            h, w, c = cropped.shape
            binned = cropped.reshape(h // binning, binning, w // binning, binning, c).mean(axis=(1, 3))
        else:  # Grayscale
            h, w = cropped.shape
            binned = cropped.reshape(h // binning, binning, w // binning, binning).mean(axis=(1, 3))

        return binned.astype(image_data.dtype)

    def _save_as_fits(self, image_data, save_path: Path):
        """Save image as FITS format."""
        try:
            import numpy as np
            from astropy.io import fits

            # Convert RGB to grayscale for astronomy (luminance)
            if len(image_data.shape) == 3:
                gray = np.mean(image_data, axis=2).astype(np.uint16)
            else:
                gray = image_data.astype(np.uint16)

            hdu = fits.PrimaryHDU(gray)
            hdu.header["INSTRUME"] = "Raspberry Pi HQ Camera"
            hdu.header["CAMERA"] = "IMX477"
            hdu.writeto(save_path, overwrite=True)

        except ImportError:
            self.logger.error("astropy not installed. Install with: pip install astropy")
            raise

    def _save_as_png(self, image_data, save_path: Path):
        """Save image as PNG format."""
        try:
            from PIL import Image

            img = Image.fromarray(image_data)
            img.save(save_path)

        except ImportError:
            self.logger.error("Pillow not installed. Install with: pip install Pillow")
            raise

    def _save_as_jpg(self, image_data, save_path: Path):
        """Save image as JPEG format."""
        try:
            from PIL import Image

            img = Image.fromarray(image_data)
            img.save(save_path, quality=95)

        except ImportError:
            self.logger.error("Pillow not installed. Install with: pip install Pillow")
            raise

    def _save_as_raw(self, image_data, save_path: Path):
        """Save raw image data as numpy array."""
        try:
            import numpy as np

            np.save(save_path.with_suffix(".npy"), image_data)

        except ImportError:
            self.logger.error("numpy not installed. Install with: pip install numpy")
            raise
