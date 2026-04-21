"""Device adapter registry.

This module provides a centralized registry for all device adapters.
Similar to the hardware adapter registry, but for individual device types.
"""

import importlib
from typing import Any

from citrasense.hardware.devices.camera import AbstractCamera
from citrasense.hardware.devices.filter_wheel import AbstractFilterWheel
from citrasense.hardware.devices.focuser import AbstractFocuser
from citrasense.hardware.devices.mount import AbstractMount

# Registry of available camera devices
CAMERA_DEVICES: dict[str, dict[str, str]] = {
    "ximea": {
        "module": "citrasense.hardware.devices.camera.ximea_camera",
        "class_name": "XimeaHyperspectralCamera",
        "description": "Ximea Hyperspectral Camera (MQ series)",
    },
    "rpi_hq": {
        "module": "citrasense.hardware.devices.camera.rpi_hq_camera",
        "class_name": "RaspberryPiHQCamera",
        "description": "Raspberry Pi High Quality Camera (IMX477)",
    },
    "usb_camera": {
        "module": "citrasense.hardware.devices.camera.usb_camera",
        "class_name": "UsbCamera",
        "description": "USB Camera via OpenCV (guide cameras, planetary cameras, etc.)",
    },
    "moravian": {
        "module": "citrasense.hardware.devices.camera.moravian_camera",
        "class_name": "MoravianCamera",
        "description": "Moravian Instruments Camera (Gx/Cx series)",
    },
    # Future cameras:
    # "zwo": {...},
    # "ascom": {...},
    # "qhy": {...},
}

# Registry of available mount devices
MOUNT_DEVICES: dict[str, dict[str, str]] = {
    "zwo_am": {
        "module": "citrasense.hardware.devices.mount.zwo_am_mount",
        "class_name": "ZwoAmMount",
        "description": "ZWO AM3/AM5/AM7 Mount (USB serial or WiFi TCP)",
    },
    # Future mounts:
    # "celestron": {...},
    # "skywatcher": {...},
    # "ascom": {...},
}

# Registry of available filter wheel devices
FILTER_WHEEL_DEVICES: dict[str, dict[str, str]] = {
    "moravian": {
        "module": "citrasense.hardware.devices.filter_wheel.moravian_filter_wheel",
        "class_name": "MoravianFilterWheel",
        "description": "Moravian Instruments Filter Wheel (External)",
    },
    # Future filter wheels:
    # "zwo": {...},
    # "ascom": {...},
}

# Registry of available focuser devices
FOCUSER_DEVICES: dict[str, dict[str, str]] = {
    "zwo_eaf": {
        "module": "citrasense.hardware.devices.focuser.zwo_eaf",
        "class_name": "ZwoEafFocuser",
        "description": "ZWO Electronic Automatic Focuser (EAF)",
    },
}


def get_camera_class(camera_type: str) -> type[AbstractCamera]:
    """Get the camera class for the given camera type.

    Args:
        camera_type: The type of camera (e.g., "ximea", "zwo")

    Returns:
        The camera adapter class

    Raises:
        ValueError: If the camera type is not registered
        ImportError: If the camera module cannot be imported
    """
    if camera_type not in CAMERA_DEVICES:
        available = ", ".join(f"'{name}'" for name in CAMERA_DEVICES.keys())
        raise ValueError(f"Unknown camera type: '{camera_type}'. Valid options are: {available}")

    device_info = CAMERA_DEVICES[camera_type]
    module = importlib.import_module(device_info["module"])
    device_class = getattr(module, device_info["class_name"])

    return device_class


def get_mount_class(mount_type: str) -> type[AbstractMount]:
    """Get the mount class for the given mount type.

    Args:
        mount_type: The type of mount

    Returns:
        The mount adapter class

    Raises:
        ValueError: If the mount type is not registered
        ImportError: If the mount module cannot be imported
    """
    if mount_type not in MOUNT_DEVICES:
        available = ", ".join(f"'{name}'" for name in MOUNT_DEVICES.keys())
        raise ValueError(f"Unknown mount type: '{mount_type}'. Valid options are: {available}")

    device_info = MOUNT_DEVICES[mount_type]
    module = importlib.import_module(device_info["module"])
    device_class = getattr(module, device_info["class_name"])

    return device_class


def get_filter_wheel_class(filter_wheel_type: str) -> type[AbstractFilterWheel]:
    """Get the filter wheel class for the given filter wheel type.

    Args:
        filter_wheel_type: The type of filter wheel

    Returns:
        The filter wheel adapter class

    Raises:
        ValueError: If the filter wheel type is not registered
        ImportError: If the filter wheel module cannot be imported
    """
    if filter_wheel_type not in FILTER_WHEEL_DEVICES:
        available = ", ".join(f"'{name}'" for name in FILTER_WHEEL_DEVICES.keys())
        raise ValueError(f"Unknown filter wheel type: '{filter_wheel_type}'. Valid options are: {available}")

    device_info = FILTER_WHEEL_DEVICES[filter_wheel_type]
    module = importlib.import_module(device_info["module"])
    device_class = getattr(module, device_info["class_name"])

    return device_class


def get_focuser_class(focuser_type: str) -> type[AbstractFocuser]:
    """Get the focuser class for the given focuser type.

    Args:
        focuser_type: The type of focuser

    Returns:
        The focuser adapter class

    Raises:
        ValueError: If the focuser type is not registered
        ImportError: If the focuser module cannot be imported
    """
    if focuser_type not in FOCUSER_DEVICES:
        available = ", ".join(f"'{name}'" for name in FOCUSER_DEVICES.keys())
        raise ValueError(f"Unknown focuser type: '{focuser_type}'. Valid options are: {available}")

    device_info = FOCUSER_DEVICES[focuser_type]
    module = importlib.import_module(device_info["module"])
    device_class = getattr(module, device_info["class_name"])

    return device_class


def list_devices(device_type: str) -> dict[str, dict[str, str]]:
    """Get a dictionary of all registered devices of a specific type.

    Args:
        device_type: Type of device ("camera", "mount", "filter_wheel", "focuser")

    Returns:
        Dict mapping device names to their info including friendly_name
    """
    registries = {
        "camera": CAMERA_DEVICES,
        "mount": MOUNT_DEVICES,
        "filter_wheel": FILTER_WHEEL_DEVICES,
        "focuser": FOCUSER_DEVICES,
    }

    registry = registries.get(device_type, {})
    result = {}

    for name, info in registry.items():
        # Try to get friendly name from device class
        try:
            if device_type == "camera":
                device_class = get_camera_class(name)
            elif device_type == "mount":
                device_class = get_mount_class(name)
            elif device_type == "filter_wheel":
                device_class = get_filter_wheel_class(name)
            elif device_type == "focuser":
                device_class = get_focuser_class(name)
            else:
                continue

            friendly_name = device_class.get_friendly_name()
        except Exception:
            # Fallback to description if friendly_name not available
            friendly_name = info["description"]

        result[name] = {
            "friendly_name": friendly_name,
            "description": info["description"],
            "module": info["module"],
            "class_name": info["class_name"],
        }

    return result


def get_device_schema(device_type: str, device_name: str) -> list:
    """Get the configuration schema for a specific device.

    Args:
        device_type: Type of device ("camera", "mount", "filter_wheel", "focuser")
        device_name: The name of the device (e.g., "ximea", "celestron")

    Returns:
        The device's settings schema

    Raises:
        ValueError: If the device type or name is not registered
        ImportError: If the device module cannot be imported
    """
    if device_type == "camera":
        device_class = get_camera_class(device_name)
    elif device_type == "mount":
        device_class = get_mount_class(device_name)
    elif device_type == "filter_wheel":
        device_class = get_filter_wheel_class(device_name)
    elif device_type == "focuser":
        device_class = get_focuser_class(device_name)
    else:
        raise ValueError(f"Unknown device type: '{device_type}'")

    return device_class.get_settings_schema()


def check_dependencies(device_class: type[Any]) -> dict[str, Any]:
    """Check if dependencies for a device are available.

    Args:
        device_class: Device class to check

    Returns:
        Dict with keys:
            - available (bool): True if all dependencies installed
            - missing (list[str]): List of missing package names
            - install_cmd (str): Command to install missing packages
    """
    import time

    start_time = time.time()

    deps = device_class.get_dependencies()
    packages = deps.get("packages", [])
    install_extra = deps.get("install_extra", "")

    missing = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)

    available = len(missing) == 0
    if install_extra:
        install_cmd = f"uv tool install citrasense --with citrasense[{install_extra}]"
    else:
        install_cmd = f"uv pip install {' '.join(missing)}"

    elapsed = time.time() - start_time
    if elapsed > 0.05:  # Log if takes more than 50ms
        from citrasense.logging import CITRASENSE_LOGGER

        CITRASENSE_LOGGER.info(f"Dependency check for {device_class.__name__} took {elapsed:.3f}s")

    return {
        "available": available,
        "missing": missing,
        "install_cmd": install_cmd,
    }
