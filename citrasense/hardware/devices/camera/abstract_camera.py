"""Abstract camera device interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from citrasense.hardware.devices.abstract_hardware_device import AbstractHardwareDevice
from citrasense.location.gps_fix import GPSFix

if TYPE_CHECKING:
    from citrasense.hardware.devices.filter_wheel import AbstractFilterWheel


@dataclass
class CalibrationProfile:
    """Reports a camera's calibration capabilities and current state.

    Used by MasterBuilder to know what to capture, by CalibrationLibrary
    to tag masters, and by the web UI to show the right controls.
    """

    calibration_applicable: bool
    camera_id: str
    model: str
    has_mechanical_shutter: bool
    has_cooling: bool
    current_gain: int | None
    current_binning: int
    current_temperature: float | None
    target_temperature: float | None = None
    bit_depth: int = 16
    read_mode: str = ""
    gain_range: tuple[int, int] | None = None
    supported_binning: list[int] = field(default_factory=lambda: [1])


class AbstractCamera(AbstractHardwareDevice):
    """Abstract base class for camera devices.

    Provides a common interface for controlling imaging cameras including
    CCDs, CMOS sensors, and hyperspectral cameras.
    """

    @abstractmethod
    def capture_array(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        shutter_closed: bool = False,
    ) -> np.ndarray:
        """Acquire raw pixel data as a 2-D numpy array. No file I/O.

        This is the fundamental acquisition primitive. ``take_exposure``
        calls this internally and then saves the result to disk.

        Args:
            duration: Exposure duration in seconds
            gain: Camera gain setting (device-specific units)
            offset: Camera offset/black level setting
            binning: Pixel binning factor (1=no binning, 2=2x2, etc.)
            shutter_closed: If True, request a dark frame (firmware closes shutter
                on cameras with mechanical shutters; ignored on others).

        Returns:
            2-D numpy array of pixel data (uint8 or uint16).
        """
        ...

    @abstractmethod
    def take_exposure(
        self,
        duration: float,
        gain: int | None = None,
        offset: int | None = None,
        binning: int = 1,
        save_path: Path | None = None,
        shutter_closed: bool = False,
    ) -> Path:
        """Capture an image exposure and save to disk.

        Args:
            duration: Exposure duration in seconds
            gain: Camera gain setting (device-specific units)
            offset: Camera offset/black level setting
            binning: Pixel binning factor (1=no binning, 2=2x2, etc.)
            save_path: Optional path to save the image (if None, use default)
            shutter_closed: If True, request a dark frame (firmware closes shutter
                on cameras with mechanical shutters; ignored on others).

        Returns:
            Path to the saved image file
        """
        ...

    @abstractmethod
    def abort_exposure(self):
        """Abort the current exposure if one is in progress."""
        pass

    @abstractmethod
    def get_temperature(self) -> float | None:
        """Get the current camera sensor temperature.

        Returns:
            Temperature in degrees Celsius, or None if not available
        """
        pass

    @abstractmethod
    def set_temperature(self, temperature: float) -> bool:
        """Set the target camera sensor temperature.

        Args:
            temperature: Target temperature in degrees Celsius

        Returns:
            True if temperature setpoint accepted, False otherwise
        """
        pass

    @abstractmethod
    def start_cooling(self) -> bool:
        """Enable camera cooling system.

        Returns:
            True if cooling started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_cooling(self) -> bool:
        """Disable camera cooling system.

        Returns:
            True if cooling stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_camera_info(self) -> dict:
        """Get camera capabilities and information.

        Returns:
            Dictionary containing camera specs (resolution, pixel size, bit depth, etc.)
        """
        pass

    def get_default_binning(self) -> int:
        """Return the camera's default binning setting (e.g., 1 for 1x1, 2 for 2x2)."""
        return 1

    def is_hyperspectral(self) -> bool:
        """Indicates whether this camera captures hyperspectral data.

        Hyperspectral cameras capture multiple spectral bands simultaneously
        (e.g., snapshot mosaic sensors like Ximea MQ series).

        Returns:
            bool: True if hyperspectral camera, False otherwise (default)
        """
        return False

    def get_integrated_filter_wheel(self) -> AbstractFilterWheel | None:
        """Return an integrated filter wheel if the camera has one built in.

        Override in subclasses where the filter wheel is physically part of
        the camera and shares its control handle (e.g. Moravian Cx/Gx).
        DirectHardwareAdapter calls this after camera connect to auto-detect
        integrated filter wheels without requiring separate user configuration.
        """
        return None

    def get_calibration_profile(self) -> CalibrationProfile:
        """Report calibration capabilities and current camera state.

        Override in subclasses that benefit from CCD-style calibration.
        The default returns calibration_applicable=False, which hides all
        calibration UI and silently skips the CalibrationProcessor.
        """
        return CalibrationProfile(
            calibration_applicable=False,
            camera_id=self.__class__.__name__,
            model=self.__class__.__name__,
            has_mechanical_shutter=False,
            has_cooling=False,
            current_gain=None,
            current_binning=1,
            current_temperature=None,
        )

    def get_max_pixel_value(self, binning: int = 1) -> int:
        """Return the maximum possible pixel value for the current camera settings.

        Accounts for read mode, gain, and binning (e.g. pixel-adding binning
        on Moravian C2 cameras multiplies the per-pixel max by binning^2).
        Used by calibration auto-expose to know the true saturation point.

        Override in subclasses that can query the hardware directly.
        """
        bit_depth = getattr(self, "_camera_info", {}).get("bit_depth", 16)
        return (2**bit_depth) - 1

    def get_gps_location(self) -> GPSFix | None:
        """Return GPS fix from an integrated receiver, or None if unavailable.

        Override in cameras with built-in GPS modules (e.g. Moravian Cx).
        """
        return None

    def get_preferred_file_extension(self) -> str:
        """Get the preferred file extension for saved images.

        This method allows each camera to define what file format it wants
        to use, without the hardware adapter needing to know camera internals.

        Returns:
            File extension string without the dot (e.g., 'fits', 'png', 'jpg')
        """
        # Default implementation: use output_format if available, otherwise FITS
        return getattr(self, "output_format", "fits")
