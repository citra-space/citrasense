"""Tests for device dependency management system."""

import pytest

from citrasense.hardware.devices.device_registry import (
    CAMERA_DEVICES,
    FILTER_WHEEL_DEVICES,
    FOCUSER_DEVICES,
    MOUNT_DEVICES,
    check_dependencies,
    get_camera_class,
    get_filter_wheel_class,
    get_focuser_class,
    get_mount_class,
)


class TestDeviceRegistryStructure:
    """Test suite for device registry data structure consistency."""

    def test_camera_registry_structure(self):
        """Verify all camera registry entries have required keys."""
        required_keys = {"module", "class_name", "description"}

        for camera_name, camera_info in CAMERA_DEVICES.items():
            # Check all required keys are present
            assert required_keys.issubset(camera_info.keys()), (
                f"Camera '{camera_name}' missing required keys. "
                f"Expected: {required_keys}, Got: {set(camera_info.keys())}"
            )

            # Check values are non-empty strings
            assert isinstance(camera_info["module"], str)
            assert camera_info["module"]
            assert isinstance(camera_info["class_name"], str)
            assert camera_info["class_name"]
            assert isinstance(camera_info["description"], str)
            assert camera_info["description"]

    def test_mount_registry_structure(self):
        """Verify all mount registry entries have required keys."""
        if len(MOUNT_DEVICES) == 0:
            pytest.skip("No mounts registered yet")

        required_keys = {"module", "class_name", "description"}

        for mount_name, mount_info in MOUNT_DEVICES.items():
            assert required_keys.issubset(mount_info.keys()), f"Mount '{mount_name}' missing required keys"
            assert isinstance(mount_info["module"], str)
            assert mount_info["module"]
            assert isinstance(mount_info["class_name"], str)
            assert mount_info["class_name"]
            assert isinstance(mount_info["description"], str)
            assert mount_info["description"]

    def test_filter_wheel_registry_structure(self):
        """Verify all filter wheel registry entries have required keys."""
        if len(FILTER_WHEEL_DEVICES) == 0:
            pytest.skip("No filter wheels registered yet")

        required_keys = {"module", "class_name", "description"}

        for fw_name, fw_info in FILTER_WHEEL_DEVICES.items():
            assert required_keys.issubset(fw_info.keys()), f"Filter wheel '{fw_name}' missing required keys"
            assert isinstance(fw_info["module"], str)
            assert fw_info["module"]
            assert isinstance(fw_info["class_name"], str)
            assert fw_info["class_name"]
            assert isinstance(fw_info["description"], str)
            assert fw_info["description"]

    def test_focuser_registry_structure(self):
        """Verify all focuser registry entries have required keys."""
        if len(FOCUSER_DEVICES) == 0:
            pytest.skip("No focusers registered yet")

        required_keys = {"module", "class_name", "description"}

        for focuser_name, focuser_info in FOCUSER_DEVICES.items():
            assert required_keys.issubset(focuser_info.keys()), f"Focuser '{focuser_name}' missing required keys"
            assert isinstance(focuser_info["module"], str)
            assert focuser_info["module"]
            assert isinstance(focuser_info["class_name"], str)
            assert focuser_info["class_name"]
            assert isinstance(focuser_info["description"], str)
            assert focuser_info["description"]


class TestDeviceDependencies:
    """Test suite for device dependency management."""

    def test_all_cameras_have_dependencies(self):
        """Verify all registered cameras implement get_dependencies()."""
        for camera_name in CAMERA_DEVICES.keys():
            camera_class = get_camera_class(camera_name)

            # Should have get_dependencies method
            assert hasattr(camera_class, "get_dependencies")
            assert callable(camera_class.get_dependencies)

            # Should return proper structure
            deps = camera_class.get_dependencies()
            assert isinstance(deps, dict)
            assert "packages" in deps
            assert "install_extra" in deps
            assert isinstance(deps["packages"], list)
            assert isinstance(deps["install_extra"], str)

            # Packages should be a list of strings (may be empty for native-library-only drivers)
            for pkg in deps["packages"]:
                assert isinstance(pkg, str)
                assert len(pkg) > 0

    def test_check_dependencies_structure(self):
        """Verify check_dependencies returns proper structure."""
        for camera_name in CAMERA_DEVICES.keys():
            camera_class = get_camera_class(camera_name)
            result = check_dependencies(camera_class)

            # Should return dict with required keys
            assert isinstance(result, dict)
            assert "available" in result
            assert "missing" in result
            assert "install_cmd" in result

            # Types should be correct
            assert isinstance(result["available"], bool)
            assert isinstance(result["missing"], list)
            assert isinstance(result["install_cmd"], str)

            # If not available, should have missing packages
            if not result["available"]:
                assert len(result["missing"]) > 0
                assert "citrasense" in result["install_cmd"]

    def test_camera_friendly_names(self):
        """Verify all cameras have friendly names."""
        for camera_name in CAMERA_DEVICES.keys():
            camera_class = get_camera_class(camera_name)

            # Should have get_friendly_name method
            assert hasattr(camera_class, "get_friendly_name")
            assert callable(camera_class.get_friendly_name)

            # Should return non-empty string
            friendly_name = camera_class.get_friendly_name()
            assert isinstance(friendly_name, str)
            assert len(friendly_name) > 0

    def test_dependency_check_is_consistent(self):
        """Verify check_dependencies gives consistent results."""
        for camera_name in CAMERA_DEVICES.keys():
            camera_class = get_camera_class(camera_name)

            # Run check twice
            result1 = check_dependencies(camera_class)
            result2 = check_dependencies(camera_class)

            # Results should be identical (no caching side effects)
            assert result1["available"] == result2["available"]
            assert result1["missing"] == result2["missing"]
            assert result1["install_cmd"] == result2["install_cmd"]

    def test_known_camera_dependencies(self):
        """Verify specific known camera dependencies are correct."""
        # USB camera
        usb_camera_class = get_camera_class("usb_camera")
        usb_camera_deps = usb_camera_class.get_dependencies()
        assert "cv2" in usb_camera_deps["packages"]
        assert usb_camera_deps["install_extra"] == "usb-camera"

        # Raspberry Pi HQ camera
        rpi_class = get_camera_class("rpi_hq")
        rpi_deps = rpi_class.get_dependencies()
        assert "picamera2" in rpi_deps["packages"]
        assert rpi_deps["install_extra"] == "rpi"

        # Ximea camera
        ximea_class = get_camera_class("ximea")
        ximea_deps = ximea_class.get_dependencies()
        assert "ximea" in ximea_deps["packages"]
        assert ximea_deps["install_extra"] == "ximea"

    @pytest.mark.parametrize(
        ("device_registry", "get_class_func"),
        [
            (MOUNT_DEVICES, get_mount_class),
            (FILTER_WHEEL_DEVICES, get_filter_wheel_class),
            (FOCUSER_DEVICES, get_focuser_class),
        ],
    )
    def test_future_devices_will_have_dependencies(self, device_registry, get_class_func):
        """Verify any future devices will also have dependency support.

        This test will start passing once devices are added to these registries.
        For now it just verifies the test infrastructure is ready.
        """
        if len(device_registry) == 0:
            # No devices registered yet - that's expected
            pytest.skip("No devices registered in this category yet")

        for device_name in device_registry.keys():
            device_class = get_class_func(device_name)

            # Should have get_dependencies method
            assert hasattr(device_class, "get_dependencies")
            assert callable(device_class.get_dependencies)

            # Should return proper structure
            deps = device_class.get_dependencies()
            assert isinstance(deps, dict)
            assert "packages" in deps
            assert "install_extra" in deps


class TestDependencyCheckOutput:
    """Test dependency check output formatting."""

    def test_install_command_format(self):
        """Verify install commands are properly formatted for cameras with deps."""
        for camera_name in CAMERA_DEVICES.keys():
            camera_class = get_camera_class(camera_name)
            deps = camera_class.get_dependencies()

            if not deps["packages"] and not deps["install_extra"]:
                continue

            result = check_dependencies(camera_class)
            cmd = result["install_cmd"]
            assert cmd.startswith("uv ")
            assert "citrasense" in cmd

    def test_missing_packages_list(self):
        """Verify missing packages list is accurate."""
        for camera_name in CAMERA_DEVICES.keys():
            camera_class = get_camera_class(camera_name)
            deps = camera_class.get_dependencies()
            result = check_dependencies(camera_class)

            # All missing packages should be in the declared packages list
            for missing_pkg in result["missing"]:
                assert missing_pkg in deps["packages"]

            # If available, missing list should be empty
            if result["available"]:
                assert len(result["missing"]) == 0
