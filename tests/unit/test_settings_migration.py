"""Tests for the settings migration from legacy scalar fields to sensors list.

Exercises the one-shot auto-migration introduced in phase 1 (#306) and
verifies the "merge, don't replace" invariant across round-trips.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from citrasense.settings.citrasense_settings import (
    CONFIG_VERSION,
    DEFAULT_TELESCOPE_SENSOR_ID,
    CitraSenseSettings,
    SensorConfig,
)


def _make_settings(config_on_disk: dict[str, Any]) -> tuple[CitraSenseSettings, Any]:
    """Load settings from a mock config dict, returning the instance and the mock file manager."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = config_on_disk
        s = CitraSenseSettings.load()
    return s, instance


class TestLegacyAutoMigration:
    def test_legacy_config_produces_sensors_list(self):
        s, _ = _make_settings(
            {
                "hardware_adapter": "nina",
                "telescope_id": "tel-1",
                "adapter_settings": {
                    "nina": {"host": "192.168.1.100", "port": 1888, "filters": {"0": {"name": "Luminance"}}}
                },
            }
        )
        assert len(s.sensors) == 1
        head = s.sensors[0]
        assert head.id == DEFAULT_TELESCOPE_SENSOR_ID
        assert head.type == "telescope"
        assert head.adapter == "nina"
        assert head.citra_sensor_id == "tel-1"
        assert head.adapter_settings["host"] == "192.168.1.100"
        assert "filters" in head.adapter_settings

    def test_scalars_still_readable(self):
        s, _ = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "tel-2",
            }
        )
        assert s.sensors[0].adapter == "dummy"
        assert s.sensors[0].citra_sensor_id == "tel-2"

    def test_empty_config_has_no_sensors(self):
        s, _ = _make_settings({})
        assert s.sensors == []

    def test_forward_shape_round_trips(self):
        s, mock_sfm = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "tel-3",
                "adapter_settings": {"dummy": {"simulate_slow_operations": True}},
            }
        )
        s.save()
        saved: dict = mock_sfm.save_config.call_args[0][0]
        assert "sensors" in saved
        assert isinstance(saved["sensors"], list)
        assert len(saved["sensors"]) == 1
        assert saved["sensors"][0]["id"] == DEFAULT_TELESCOPE_SENSOR_ID
        assert saved["config_version"] == CONFIG_VERSION


class TestMigrationIdempotency:
    def test_second_load_with_sensors_key_is_unchanged(self):
        s1, _ = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "tel-4",
                "adapter_settings": {"dummy": {}},
            }
        )
        assert len(s1.sensors) == 1

        # Simulate round-tripping the first save's output through a second load
        s2, _ = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "tel-4",
                "adapter_settings": {"dummy": {}},
                "sensors": [s1.sensors[0].model_dump()],
                "config_version": CONFIG_VERSION,
            }
        )
        assert len(s2.sensors) == 1
        assert s2.sensors[0].id == DEFAULT_TELESCOPE_SENSOR_ID


class TestFilterPreservationAcrossMigration:
    def test_filters_survive_update_and_save(self):
        disk = {
            "hardware_adapter": "nina",
            "telescope_id": "tel-5",
            "adapter_settings": {
                "nina": {
                    "host": "192.168.1.100",
                    "filters": {
                        "0": {"name": "Luminance", "focus_position": 9000, "enabled": True},
                        "1": {"name": "Ha", "focus_position": 8500, "enabled": True},
                    },
                }
            },
        }
        s, mock_sfm = _make_settings(disk)

        # Send the update in the modern per-sensor shape (the legacy
        # top-level ``adapter_settings`` blob is now dropped — see
        # test_settings_update_and_save_ignores_legacy_adapter_settings).
        sensor_id = s.sensors[0].id
        s.update_and_save(
            {
                "hardware_adapter": "nina",
                "sensors": [
                    {
                        "id": sensor_id,
                        "adapter": "nina",
                        "adapter_settings": {"host": "10.0.0.1"},
                    }
                ],
            }
        )

        saved: dict = mock_sfm.save_config.call_args[0][0]
        nina_settings = saved["sensors"][0]["adapter_settings"]
        assert "filters" in nina_settings
        assert nina_settings["filters"]["0"]["name"] == "Luminance"
        assert nina_settings["host"] == "10.0.0.1"

    def test_sensors_adapter_settings_updated_on_save(self):
        s, mock_sfm = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "tel-6",
                "adapter_settings": {"dummy": {"key": "original"}},
            }
        )
        s.sensors[0].adapter_settings["key"] = "modified"
        s.save()

        saved: dict = mock_sfm.save_config.call_args[0][0]
        assert saved["sensors"][0]["adapter_settings"]["key"] == "modified"


class TestAllAdapterMigration:
    """Verify that a legacy config for each registered adapter type auto-migrates cleanly."""

    @pytest.mark.parametrize("adapter_name", ["nina", "kstars", "indi", "direct", "dummy"])
    def test_legacy_adapter_migrates(self, adapter_name: str):
        s, _ = _make_settings(
            {
                "hardware_adapter": adapter_name,
                "telescope_id": f"tel-{adapter_name}",
                "adapter_settings": {adapter_name: {"some_key": "some_value"}},
            }
        )
        assert len(s.sensors) == 1
        head = s.sensors[0]
        assert head.id == DEFAULT_TELESCOPE_SENSOR_ID
        assert head.type == "telescope"
        assert head.adapter == adapter_name
        assert head.citra_sensor_id == f"tel-{adapter_name}"
        assert head.adapter_settings.get("some_key") == "some_value"

    @pytest.mark.parametrize("adapter_name", ["nina", "kstars", "indi", "direct", "dummy"])
    def test_legacy_adapter_save_round_trips(self, adapter_name: str):
        s, mock_sfm = _make_settings(
            {
                "hardware_adapter": adapter_name,
                "telescope_id": f"tel-{adapter_name}",
                "adapter_settings": {adapter_name: {"key": "val"}},
            }
        )
        s.save()
        saved: dict = mock_sfm.save_config.call_args[0][0]
        assert saved["config_version"] == CONFIG_VERSION
        assert len(saved["sensors"]) == 1
        assert saved["sensors"][0]["adapter"] == adapter_name
        assert saved["sensors"][0]["adapter_settings"]["key"] == "val"


class TestUpdateAndSaveSyncsCitraSensorId:
    """``update_and_save`` must propagate sensor ID changes via sensors[]."""

    def test_citra_sensor_id_change_updates_sensor(self):
        s, mock_sfm = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "old-id",
                "adapter_settings": {"dummy": {}},
            }
        )
        assert s.sensors[0].citra_sensor_id == "old-id"

        s.update_and_save(
            {
                "sensors": [
                    {
                        "id": DEFAULT_TELESCOPE_SENSOR_ID,
                        "type": "telescope",
                        "adapter": "dummy",
                        "adapter_settings": {},
                        "citra_sensor_id": "new-id",
                    }
                ],
            }
        )

        saved: dict = mock_sfm.save_config.call_args[0][0]
        assert saved["sensors"][0]["citra_sensor_id"] == "new-id"

    def test_citra_sensor_id_unchanged_preserves_sensor(self):
        s, mock_sfm = _make_settings(
            {
                "hardware_adapter": "dummy",
                "telescope_id": "keep-me",
                "adapter_settings": {"dummy": {}},
            }
        )
        s.update_and_save({"personal_access_token": "tok"})

        saved: dict = mock_sfm.save_config.call_args[0][0]
        assert saved["sensors"][0]["citra_sensor_id"] == "keep-me"


class TestLoadGuardsForwardShapeAdapterSettings:
    """Fix 2: ``load()`` must not clobber a forward-shape sensor's populated ``adapter_settings``."""

    def test_forward_shape_adapter_settings_preserved(self):
        s, _ = _make_settings(
            {
                "hardware_adapter": "nina",
                "telescope_id": "tel-fw",
                "adapter_settings": {},
                "sensors": [
                    {
                        "id": DEFAULT_TELESCOPE_SENSOR_ID,
                        "type": "telescope",
                        "adapter": "nina",
                        "citra_sensor_id": "tel-fw",
                        "adapter_settings": {"host": "10.0.0.5", "port": 1888},
                    }
                ],
                "config_version": CONFIG_VERSION,
            }
        )
        head = s.sensors[0]
        assert head.adapter_settings["host"] == "10.0.0.5"
        assert head.adapter_settings["port"] == 1888

    def test_synthesized_entry_still_patched(self):
        s, _ = _make_settings(
            {
                "hardware_adapter": "nina",
                "telescope_id": "tel-synth",
                "adapter_settings": {"nina": {"host": "192.168.1.1"}},
            }
        )
        head = s.sensors[0]
        assert head.adapter_settings["host"] == "192.168.1.1"


class TestSensorConfigModel:
    def test_basic_construction(self):
        cfg = SensorConfig(
            id="telescope-0",
            type="telescope",
            adapter="nina",
            adapter_settings={"host": "1.2.3.4"},
            citra_sensor_id="tel-abc",
        )
        assert cfg.id == "telescope-0"
        assert cfg.type == "telescope"
        assert cfg.adapter_settings["host"] == "1.2.3.4"

    def test_defaults(self):
        cfg = SensorConfig(id="s1", type="test")
        assert cfg.adapter == ""
        assert cfg.adapter_settings == {}
        assert cfg.citra_sensor_id == ""
        assert cfg.hfr_baseline is None


class TestHfrBaselinePromotion:
    """v7 → v8 migration: ``hfr_baseline`` leaves ``adapter_settings``."""

    def test_baseline_moves_out_of_adapter_settings(self):
        s, _ = _make_settings(
            {
                "config_version": 7,
                "sensors": [
                    {
                        "id": "telescope-0",
                        "type": "telescope",
                        "adapter": "nina",
                        "adapter_settings": {
                            "host": "1.2.3.4",
                            "hfr_baseline": 1.87,
                            "filters": {"0": {"name": "Luminance"}},
                        },
                    }
                ],
            }
        )
        head = s.sensors[0]
        assert head.hfr_baseline == pytest.approx(1.87)
        assert "hfr_baseline" not in head.adapter_settings
        # Other adapter_settings keys must survive the migration.
        assert head.adapter_settings["host"] == "1.2.3.4"
        assert head.adapter_settings["filters"] == {"0": {"name": "Luminance"}}

    def test_missing_baseline_leaves_field_none(self):
        s, _ = _make_settings(
            {
                "config_version": 7,
                "sensors": [
                    {
                        "id": "telescope-0",
                        "type": "telescope",
                        "adapter": "nina",
                        "adapter_settings": {"host": "1.2.3.4"},
                    }
                ],
            }
        )
        assert s.sensors[0].hfr_baseline is None
        assert "hfr_baseline" not in s.sensors[0].adapter_settings

    def test_explicit_top_level_value_wins(self):
        # If a sensor already carries a top-level ``hfr_baseline`` and the
        # adapter_settings blob also has one (e.g. a partial manual edit),
        # the already-promoted top-level value takes precedence — the
        # migration uses ``setdefault`` which is a no-op when the key
        # already exists.
        s, _ = _make_settings(
            {
                "config_version": 7,
                "sensors": [
                    {
                        "id": "telescope-0",
                        "type": "telescope",
                        "adapter": "nina",
                        "hfr_baseline": 2.5,
                        "adapter_settings": {"hfr_baseline": 9.9},
                    }
                ],
            }
        )
        assert s.sensors[0].hfr_baseline == pytest.approx(2.5)
        assert "hfr_baseline" not in s.sensors[0].adapter_settings
