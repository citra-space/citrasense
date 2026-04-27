"""Tests for filter configuration persistence across saves and adapter switches (#137)."""

from unittest.mock import MagicMock, patch

import pytest

from citrasense.hardware.filter_sync import (
    TRASH_FILTER_NAMES,
    is_trash_filter_name,
    sync_filters_to_backend,
)

# ---------------------------------------------------------------------------
# is_trash_filter_name
# ---------------------------------------------------------------------------


class TestIsTrashFilterName:
    @pytest.mark.parametrize("name", ["Undefined", "undefined", "UNDEFINED", "Unknown", "unknown", "N/A", "none", ""])
    def test_trash_names_detected(self, name: str):
        assert is_trash_filter_name(name) is True

    def test_blank_and_whitespace(self):
        assert is_trash_filter_name("") is True
        assert is_trash_filter_name("   ") is True
        assert is_trash_filter_name("  undefined  ") is True

    @pytest.mark.parametrize("name", ["Luminance", "Red", "Ha", "OIII", "SII", "Clear", "Filter 1", "My Custom"])
    def test_real_names_pass(self, name: str):
        assert is_trash_filter_name(name) is False

    def test_frozenset_is_all_lowercase(self):
        for entry in TRASH_FILTER_NAMES:
            assert entry == entry.lower(), f"TRASH_FILTER_NAMES entry '{entry}' must be lowercase"


# ---------------------------------------------------------------------------
# sync_filters_to_backend — trash name gating
# ---------------------------------------------------------------------------


class TestSyncTrashNameGating:
    def test_all_trash_names_skips_sync(self):
        logger = MagicMock()
        api = MagicMock()
        config = {
            "0": {"name": "Undefined", "enabled": True},
            "1": {"name": "Undefined", "enabled": True},
        }
        result = sync_filters_to_backend(api, "t1", config, logger)
        assert result is False
        api.expand_filters.assert_not_called()
        logger.warning.assert_called()

    def test_mix_of_trash_and_real_syncs_only_real(self):
        logger = MagicMock()
        api = MagicMock()
        api.expand_filters.return_value = {
            "filters": [
                {"name": "Luminance", "central_wavelength_nm": 550.0, "bandwidth_nm": 300.0, "is_known": True},
            ]
        }
        api.update_telescope_spectral_config.return_value = {"status": "ok"}
        config = {
            "0": {"name": "Luminance", "enabled": True},
            "1": {"name": "Undefined", "enabled": True},
            "2": {"name": "unknown", "enabled": True},
        }
        result = sync_filters_to_backend(api, "t1", config, logger)
        assert result is True
        api.expand_filters.assert_called_once_with(["Luminance"])


# ---------------------------------------------------------------------------
# update_and_save — filter preservation
# ---------------------------------------------------------------------------


class TestUpdateAndSaveFilterPreservation:
    """Verify update_and_save merges adapter settings instead of replacing."""

    def _make_settings(self, config_on_disk: dict):
        with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
            instance = MockSFM.return_value
            instance.load_config.return_value = config_on_disk
            from citrasense.settings.citrasense_settings import CitraSenseSettings

            s = CitraSenseSettings.load()
        return s, instance

    def test_filters_preserved_when_not_in_payload(self):
        """Saving config without filters in adapter_settings must not wipe them."""
        s, mock_sfm = self._make_settings(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {
                    "dummy": {
                        "simulate_slow_operations": False,
                        "filters": {
                            "0": {"name": "Luminance", "focus_position": 9000, "enabled": True},
                            "1": {"name": "Red", "focus_position": 9050, "enabled": True},
                        },
                    }
                },
            }
        )

        sensor_id = s.sensors[0].id
        s.update_and_save(
            {
                "sensors": [
                    {
                        "id": sensor_id,
                        "adapter": "dummy",
                        "adapter_settings": {"simulate_slow_operations": True},
                    }
                ],
            }
        )

        saved = mock_sfm.save_config.call_args[0][0]
        sensor_as = saved["sensors"][0]["adapter_settings"]
        assert "filters" in sensor_as
        assert sensor_as["filters"]["0"]["name"] == "Luminance"
        assert sensor_as["filters"]["1"]["name"] == "Red"
        assert sensor_as["simulate_slow_operations"] is True

    def test_filters_updated_when_explicitly_sent(self):
        """If the payload includes filters, they should override."""
        s, mock_sfm = self._make_settings(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {
                    "dummy": {
                        "filters": {
                            "0": {"name": "Luminance", "focus_position": 9000, "enabled": True},
                        },
                    }
                },
            }
        )

        new_filters = {"0": {"name": "Ha", "focus_position": 8500, "enabled": True}}
        sensor_id = s.sensors[0].id
        s.update_and_save(
            {
                "sensors": [
                    {
                        "id": sensor_id,
                        "adapter": "dummy",
                        "adapter_settings": {"filters": new_filters},
                    }
                ],
            }
        )

        saved = mock_sfm.save_config.call_args[0][0]
        assert saved["sensors"][0]["adapter_settings"]["filters"]["0"]["name"] == "Ha"

    def test_sensors_merge_preserves_filters(self):
        """Updating a sensor via sensors[] must not wipe keys the UI didn't send."""
        s, mock_sfm = self._make_settings(
            {
                "hardware_adapter": "NinaAdvancedHttpAdapter",
                "adapter_settings": {
                    "NinaAdvancedHttpAdapter": {
                        "url_prefix": "http://localhost:1888",
                        "filters": {
                            "0": {"name": "Luminance", "focus_position": 9000, "enabled": True},
                        },
                    },
                },
            }
        )

        s.update_and_save(
            {
                "sensors": [
                    {
                        "id": "telescope-0",
                        "type": "telescope",
                        "adapter": "NinaAdvancedHttpAdapter",
                        "adapter_settings": {"url_prefix": "http://localhost:9999"},
                        "citra_sensor_id": "",
                    }
                ],
            }
        )

        saved = mock_sfm.save_config.call_args[0][0]
        sensor_as = saved["sensors"][0]["adapter_settings"]
        assert sensor_as["url_prefix"] == "http://localhost:9999"
        assert sensor_as["filters"]["0"]["name"] == "Luminance"

    def test_in_memory_adapter_settings_flushed_on_save(self):
        """In-memory changes to adapter_settings are captured on save."""
        s, mock_sfm = self._make_settings(
            {
                "hardware_adapter": "NinaAdvancedHttpAdapter",
                "adapter_settings": {
                    "NinaAdvancedHttpAdapter": {
                        "filters": {
                            "0": {"name": "Luminance", "focus_position": 9000, "enabled": True},
                        },
                    },
                },
            }
        )

        s.sensors[0].adapter_settings["filters"]["0"]["focus_position"] = 9500

        s.update_and_save(
            {
                "sensors": [
                    {
                        "id": "telescope-0",
                        "type": "telescope",
                        "adapter": "NinaAdvancedHttpAdapter",
                        "adapter_settings": {},
                        "citra_sensor_id": "",
                    }
                ],
            }
        )

        saved = mock_sfm.save_config.call_args[0][0]
        sensor_as = saved["sensors"][0]["adapter_settings"]
        assert sensor_as["filters"]["0"]["focus_position"] == 9500

    def test_new_adapter_settings_applied(self):
        """Switching adapter applies new settings; merge preserves old keys harmlessly."""
        s, mock_sfm = self._make_settings(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {
                    "dummy": {"simulate_slow_operations": False},
                },
            }
        )

        s.update_and_save(
            {
                "sensors": [
                    {
                        "id": "telescope-0",
                        "type": "telescope",
                        "adapter": "DirectHardwareAdapter",
                        "adapter_settings": {"camera_type": "moravian"},
                        "citra_sensor_id": "",
                    }
                ],
            }
        )

        saved = mock_sfm.save_config.call_args[0][0]
        sensor_as = saved["sensors"][0]["adapter_settings"]
        assert sensor_as["camera_type"] == "moravian"
        assert saved["sensors"][0]["adapter"] == "DirectHardwareAdapter"


# ---------------------------------------------------------------------------
# _populate_filter_map_from_hardware — trash name handling
# ---------------------------------------------------------------------------


class TestPopulateFilterMapTrashNames:
    """Verify DirectHardwareAdapter._populate_filter_map_from_hardware respects saved names."""

    def _make_adapter_with_filter_map(self, filter_map: dict, hw_names: list[str]):
        """Create a minimal DirectHardwareAdapter-like object for testing."""

        adapter = MagicMock()
        adapter.filter_map = dict(filter_map)
        adapter.logger = MagicMock()

        fw = MagicMock()
        fw.get_filter_names.return_value = hw_names
        fw.get_filter_count.return_value = len(hw_names)
        adapter.filter_wheel = fw
        return adapter

    def test_new_positions_with_trash_hw_names_get_placeholder(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        adapter = self._make_adapter_with_filter_map({}, ["Undefined", "Undefined", "Undefined"])
        DirectHardwareAdapter._populate_filter_map_from_hardware(adapter)

        assert adapter.filter_map[0]["name"] == "Filter 1"
        assert adapter.filter_map[1]["name"] == "Filter 2"
        assert adapter.filter_map[2]["name"] == "Filter 3"

    def test_new_positions_with_real_hw_names_use_them(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        adapter = self._make_adapter_with_filter_map({}, ["Luminance", "Red", "Green"])
        DirectHardwareAdapter._populate_filter_map_from_hardware(adapter)

        assert adapter.filter_map[0]["name"] == "Luminance"
        assert adapter.filter_map[1]["name"] == "Red"
        assert adapter.filter_map[2]["name"] == "Green"

    def test_saved_real_names_never_overwritten_by_trash(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        saved = {
            0: {"name": "Luminance", "focus_position": 9000, "enabled": True},
            1: {"name": "Ha", "focus_position": 9100, "enabled": True},
        }
        adapter = self._make_adapter_with_filter_map(saved, ["Undefined", "Undefined"])
        DirectHardwareAdapter._populate_filter_map_from_hardware(adapter)

        assert adapter.filter_map[0]["name"] == "Luminance"
        assert adapter.filter_map[0]["focus_position"] == 9000
        assert adapter.filter_map[1]["name"] == "Ha"

    def test_saved_trash_names_replaced_by_real_hw_names(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        saved = {
            0: {"name": "Undefined", "focus_position": None, "enabled": True},
            1: {"name": "Filter 2", "focus_position": None, "enabled": True},
        }
        adapter = self._make_adapter_with_filter_map(saved, ["Luminance", "Red"])

        DirectHardwareAdapter._populate_filter_map_from_hardware(adapter)

        assert adapter.filter_map[0]["name"] == "Luminance"
        # "Filter 2" is not in TRASH_FILTER_NAMES, so it's kept as user-assigned
        assert adapter.filter_map[1]["name"] == "Filter 2"

    def test_both_trash_keeps_existing(self):
        from citrasense.hardware.direct.direct_adapter import DirectHardwareAdapter

        saved = {0: {"name": "Undefined", "focus_position": 5000, "enabled": True}}
        adapter = self._make_adapter_with_filter_map(saved, ["Undefined"])
        DirectHardwareAdapter._populate_filter_map_from_hardware(adapter)

        assert adapter.filter_map[0]["name"] == "Undefined"
        assert adapter.filter_map[0]["focus_position"] == 5000
