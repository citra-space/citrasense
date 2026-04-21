import pytest

from citrasense.hardware.adapter_registry import REGISTERED_ADAPTERS, get_adapter_schema

from .utils import DummyLogger


class DummyDevice:
    def getDeviceName(self):
        return "TestScope"


class DummyProperty:
    def getName(self):
        return "TestProp"

    def getTypeAsString(self):
        return "INDI_TEXT"

    def getDeviceName(self):
        return "TestScope"


# Synthetic adapter for testing
class TestHardwareAdapter:
    def __init__(self, logger, host, port):
        self.logger = logger
        self.host = host
        self.port = port

    def newDevice(self, device):
        self.logger.infos.append(f"new device: {device.getDeviceName()}")

    def newProperty(self, prop):
        self.logger.debugs.append(
            f"new property: {prop.getName()} ({prop.getTypeAsString()}) on {prop.getDeviceName()}"
        )


@pytest.mark.usefixtures("monkeypatch")
def test_new_device_logs(monkeypatch):
    logger = DummyLogger()
    client = TestHardwareAdapter(logger, "", 1234)
    device = DummyDevice()
    client.newDevice(device)
    assert any("new device" in msg for msg in logger.infos)


@pytest.mark.usefixtures("monkeypatch")
def test_new_property_logs(monkeypatch):
    logger = DummyLogger()
    client = TestHardwareAdapter(logger, "", 1234)
    prop = DummyProperty()
    client.newProperty(prop)
    assert any("new property" in msg for msg in logger.debugs)


def test_all_adapters_schema_extraction():
    """Test that all registered adapters can have their schemas extracted without instantiation."""
    for adapter_name in REGISTERED_ADAPTERS.keys():
        try:
            schema = get_adapter_schema(adapter_name)
        except ImportError as e:
            # Skip adapters with missing optional dependencies (e.g., PyQt for kstars)
            pytest.skip(f"Skipping {adapter_name}: missing dependency ({e})")

        # Validate the schema is a list
        assert isinstance(schema, list), f"{adapter_name} schema should be a list"

        # Validate each entry has required fields
        for entry in schema:
            assert "name" in entry, f"{adapter_name} schema entry missing 'name'"
            assert "type" in entry, f"{adapter_name} schema entry missing 'type'"
            assert "friendly_name" in entry, f"{adapter_name} schema entry missing 'friendly_name'"
            assert isinstance(entry["name"], str), f"{adapter_name} schema 'name' should be a string"
            assert isinstance(entry["type"], str), f"{adapter_name} schema 'type' should be a string"
