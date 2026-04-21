"""Camera device adapters."""

from citrasense.hardware.devices.camera.abstract_camera import AbstractCamera, CalibrationProfile
from citrasense.hardware.devices.camera.rpi_hq_camera import RaspberryPiHQCamera
from citrasense.hardware.devices.camera.usb_camera import UsbCamera
from citrasense.hardware.devices.camera.ximea_camera import XimeaHyperspectralCamera

__all__ = [
    "AbstractCamera",
    "CalibrationProfile",
    "RaspberryPiHQCamera",
    "UsbCamera",
    "XimeaHyperspectralCamera",
]
