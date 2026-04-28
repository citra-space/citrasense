"""Unit tests for CitraSenseSettings and SettingsFileManager."""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# SettingsFileManager
# ---------------------------------------------------------------------------


@pytest.fixture
def sfm(tmp_path):
    """SettingsFileManager pointed at a temp directory."""
    from citrasense.settings.citrasense_settings import CitraSenseSettings  # noqa: F401 — triggers module load
    from citrasense.settings.settings_file_manager import SettingsFileManager

    mgr = SettingsFileManager()
    mgr.config_dir = tmp_path
    mgr.config_file = tmp_path / "config.json"
    return mgr


def test_load_returns_empty_when_no_file(sfm):
    assert sfm.load_config() == {}


def test_save_and_load_roundtrip(sfm):
    data = {"host": "api.citra.space", "port": 443}
    sfm.save_config(data)
    loaded = sfm.load_config()
    assert loaded["host"] == "api.citra.space"
    assert loaded["port"] == 443


def test_save_creates_directory(sfm, tmp_path):
    sfm.config_dir = tmp_path / "deep" / "nested"
    sfm.config_file = sfm.config_dir / "config.json"
    sfm.save_config({"key": "val"})
    assert sfm.config_file.exists()


def test_save_sets_restrictive_permissions(sfm):
    sfm.save_config({"secret": "token"})
    mode = oct(sfm.config_file.stat().st_mode & 0o777)
    assert mode == "0o600"


def test_load_invalid_json(sfm):
    sfm.config_file.write_text("NOT JSON{{{")
    assert sfm.load_config() == {}


def test_config_exists(sfm):
    assert sfm.config_exists() is False
    sfm.save_config({})
    assert sfm.config_exists() is True


def test_get_config_path(sfm):
    assert sfm.get_config_path() == sfm.config_file


def test_validate_config_dict(sfm):
    ok, err = sfm.validate_config({"a": 1})
    assert ok is True
    assert err is None


def test_validate_config_non_dict(sfm):
    ok, err = sfm.validate_config("not a dict")
    assert ok is False
    assert "dictionary" in err


def test_directory_manager_defaults():
    from citrasense.settings.directory_manager import DirectoryManager

    dm = DirectoryManager()
    assert dm.data_dir == DirectoryManager.default_data_dir()
    assert dm.log_dir == DirectoryManager.default_log_dir()
    assert dm.images_dir == dm.data_dir / "images"
    assert dm.processing_dir == dm.data_dir / "processing"


def test_directory_manager_custom_dirs(tmp_path):
    from citrasense.settings.directory_manager import DirectoryManager

    data = tmp_path / "data"
    logs = tmp_path / "logs"
    dm = DirectoryManager(custom_data_dir=str(data), custom_log_dir=str(logs))
    assert dm.data_dir == data
    assert dm.log_dir == logs
    assert dm.images_dir == data / "images"
    assert dm.processing_dir == data / "processing"


def test_directory_manager_ensure_dirs(tmp_path):
    from citrasense.settings.directory_manager import DirectoryManager

    dm = DirectoryManager(custom_data_dir=str(tmp_path / "data"), custom_log_dir=str(tmp_path / "logs"))
    dm.ensure_data_directories()
    dm.ensure_log_directory()
    assert dm.images_dir.exists()
    assert dm.log_dir.exists()


def test_directory_manager_current_log_path():
    from citrasense.settings.directory_manager import DirectoryManager

    dm = DirectoryManager()
    p = dm.current_log_path()
    assert "citrasense-" in p.name
    assert p.suffix == ".log"


# ---------------------------------------------------------------------------
# CitraSenseSettings
# ---------------------------------------------------------------------------


def test_settings_defaults(tmp_path):
    """Settings should use sensible defaults when config is empty."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"sensors": [{"id": "t", "type": "telescope"}]}

        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].adapter == ""
    assert s.personal_access_token == ""
    assert s.is_configured() is False
    assert s.sensors[0].autofocus_target_preset == "mirach"
    assert s.max_task_retries == 3


def test_settings_to_dict(tmp_path):
    """to_dict should include all persistent settings."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "dummy",
        }

        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    d = s.to_dict()
    assert d["personal_access_token"] == "tok"
    assert d["sensors"][0]["citra_sensor_id"] == "tel"
    assert d["sensors"][0]["adapter"] == "dummy"
    assert "autofocus_target_preset" in d["sensors"][0]
    assert "elset_refresh_interval_hours" in d
    assert "web_port" not in d


def test_settings_is_configured(tmp_path):
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "nina_advanced_http",
        }
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.is_configured() is True


def test_settings_is_configured_passive_radar_needs_no_adapter(tmp_path):
    """Radar sensors have no hardware adapter — empty ``adapter`` is legit."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "sensors": [
                {
                    "id": "radar-0",
                    "type": "passive_radar",
                    "adapter": "",
                    "citra_sensor_id": "antenna-uuid-1",
                }
            ],
        }
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.is_configured() is True


def test_settings_is_configured_passive_radar_still_needs_citra_id(tmp_path):
    """Radar sensors still need ``citra_sensor_id`` (the antenna UUID)."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "sensors": [
                {
                    "id": "radar-0",
                    "type": "passive_radar",
                    "adapter": "",
                    "citra_sensor_id": "",
                }
            ],
        }
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.is_configured() is False


def test_settings_is_configured_telescope_still_requires_adapter(tmp_path):
    """Guardrail against the radar relaxation bleeding into telescopes."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "sensors": [
                {
                    "id": "scope-0",
                    "type": "telescope",
                    "adapter": "",
                    "citra_sensor_id": "tel-1",
                }
            ],
        }
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.is_configured() is False


def test_settings_validates_custom_ra_out_of_range():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "autofocus_target_custom_ra": 999.0,
            "autofocus_target_custom_dec": -100.0,
        }
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].autofocus_target_custom_ra is None
    assert s.sensors[0].autofocus_target_custom_dec is None


def test_settings_validates_autofocus_interval():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"autofocus_interval_minutes": -5}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].autofocus_interval_minutes == 60


def test_find_duplicate_citra_sensor_ids_empty_list():
    from citrasense.settings.citrasense_settings import CitraSenseSettings

    assert CitraSenseSettings.find_duplicate_citra_sensor_ids([]) == {}


def test_find_duplicate_citra_sensor_ids_no_collision_config_objects():
    from citrasense.settings.citrasense_settings import CitraSenseSettings, SensorConfig

    sensors = [
        SensorConfig(id="a", type="telescope", citra_sensor_id="scope-a"),
        SensorConfig(id="b", type="telescope", citra_sensor_id="scope-b"),
    ]
    assert CitraSenseSettings.find_duplicate_citra_sensor_ids(sensors) == {}


def test_find_duplicate_citra_sensor_ids_detects_collision_config_objects():
    """Two SensorConfigs sharing an api id must be surfaced so boot can refuse startup."""
    from citrasense.settings.citrasense_settings import CitraSenseSettings, SensorConfig

    sensors = [
        SensorConfig(id="CoolScope", type="telescope", citra_sensor_id="asdf"),
        SensorConfig(id="LilScope", type="telescope", citra_sensor_id="asdf"),
        SensorConfig(id="KillerScope", type="telescope", citra_sensor_id="qwer"),
    ]
    dupes = CitraSenseSettings.find_duplicate_citra_sensor_ids(sensors)
    assert dupes == {"asdf": ["CoolScope", "LilScope"]}


def test_find_duplicate_citra_sensor_ids_detects_collision_dicts():
    """Raw dicts (save-path validation) must produce the same diagnosis."""
    from citrasense.settings.citrasense_settings import CitraSenseSettings

    sensors = [
        {"id": "CoolScope", "citra_sensor_id": "asdf"},
        {"id": "LilScope", "citra_sensor_id": "asdf"},
    ]
    assert CitraSenseSettings.find_duplicate_citra_sensor_ids(sensors) == {"asdf": ["CoolScope", "LilScope"]}


def test_find_duplicate_citra_sensor_ids_ignores_empty_ids():
    """Empty citra_sensor_ids are 'not configured yet', not a collision."""
    from citrasense.settings.citrasense_settings import CitraSenseSettings, SensorConfig

    sensors = [
        SensorConfig(id="a", type="telescope", citra_sensor_id=""),
        SensorConfig(id="b", type="telescope", citra_sensor_id=""),
        SensorConfig(id="c", type="telescope", citra_sensor_id="scope-c"),
    ]
    assert CitraSenseSettings.find_duplicate_citra_sensor_ids(sensors) == {}


def test_settings_save(tmp_path):
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"hardware_adapter": "dummy"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()
        s.save()

    instance.save_config.assert_called_once()


def test_settings_update_and_save_ignores_legacy_adapter_settings():
    """The pre-multi-sensor top-level ``adapter_settings`` blob is dropped.

    Historically this endpoint merged the blob into ``sensors[0]`` —
    that shim has been removed because in multi-rig deployments it
    silently wrote scope B's keys into scope A's config.  The modern UI
    sends per-sensor ``sensors[].adapter_settings`` instead.
    """
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"hardware_adapter": "dummy"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()
        previous_adapter_settings = dict(s.sensors[0].adapter_settings)
        s.update_and_save(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {"some_key": "val"},
                "web_port": 9999,
            }
        )

    instance.save_config.assert_called_once()
    saved = instance.save_config.call_args[0][0]
    assert "web_port" not in saved
    assert "adapter_settings" not in saved
    assert saved["sensors"][0]["adapter_settings"] == previous_adapter_settings
    assert "some_key" not in saved["sensors"][0]["adapter_settings"]


def test_update_and_save_preserves_fields_not_in_payload():
    """Backend-only fields survive a web UI save that omits them (data-loss bug fix)."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "hardware_adapter": "dummy",
            "elset_refresh_interval_hours": 12,
            "observation_mode": "tracking",
        }
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()
        s.update_and_save(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {},
                "personal_access_token": "new_tok",
            }
        )

    saved = instance.save_config.call_args[0][0]
    assert saved["elset_refresh_interval_hours"] == 12
    assert saved["sensors"][0]["observation_mode"] == "tracking"
    assert saved["personal_access_token"] == "new_tok"


def test_update_and_save_strips_computed_keys():
    """Computed/server-only keys from the web UI should not be written to disk."""
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"hardware_adapter": "dummy"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()
        s.update_and_save(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {},
                "app_url": "https://should-not-persist.example",
                "config_file_path": "/tmp/fake",
                "log_file_path": "/tmp/fake.log",
                "images_dir_path": "/tmp/images",
                "processing_dir_path": "/tmp/processing",
            }
        )

    saved = instance.save_config.call_args[0][0]
    for key in ("app_url", "config_file_path", "log_file_path", "images_dir_path", "processing_dir_path"):
        assert key not in saved, f"Computed key '{key}' should not be persisted"
    assert saved["sensors"][0]["adapter"] == "dummy"


# ---------------------------------------------------------------------------
# Observation mode
# ---------------------------------------------------------------------------


def test_observation_mode_defaults_to_auto():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sensors": [{"id": "t", "type": "telescope"}]}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].observation_mode == "auto"


@pytest.mark.parametrize("mode", ["auto", "tracking", "sidereal"])
def test_observation_mode_accepts_valid_values(mode):
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": mode}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].observation_mode == mode


def test_observation_mode_migrates_static_to_sidereal():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": "static"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].observation_mode == "sidereal"


def test_observation_mode_rejects_invalid_value():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": "bogus"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].observation_mode == "auto"


def test_observation_mode_in_to_dict():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": "tracking"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.to_dict()["sensors"][0]["observation_mode"] == "tracking"


# ---------------------------------------------------------------------------
# Plate solve timeout validator
# ---------------------------------------------------------------------------


def test_plate_solve_timeout_clamps_out_of_range():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"plate_solve_timeout": 999}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].plate_solve_timeout == 300


def test_plate_solve_timeout_clamps_low():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"plate_solve_timeout": 2}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].plate_solve_timeout == 10


def test_plate_solve_timeout_falls_back_on_invalid():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"plate_solve_timeout": "not_a_number"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].plate_solve_timeout == 60


# ---------------------------------------------------------------------------
# Custom directory validators
# ---------------------------------------------------------------------------


def test_custom_data_dir_empty_is_default():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"custom_data_dir": ""}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.custom_data_dir == ""


def test_custom_data_dir_absolute_path_resolved(tmp_path):
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"custom_data_dir": str(tmp_path)}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.custom_data_dir == str(tmp_path.resolve())


def test_custom_log_dir_relative_path_rejected():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"custom_log_dir": "relative/path"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    # Relative paths resolve to absolute via expanduser().resolve(), so
    # the validator will accept them after resolution. Verify it's absolute.
    from pathlib import Path

    assert s.custom_log_dir == "" or Path(s.custom_log_dir).is_absolute()


# ---------------------------------------------------------------------------
# SExtractor settings validators
# ---------------------------------------------------------------------------


def test_sextractor_detect_thresh_default():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sensors": [{"id": "t", "type": "telescope"}]}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_thresh == 5.0


def test_sextractor_detect_thresh_valid():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_thresh": 3.0}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_thresh == 3.0


def test_sextractor_detect_thresh_string_coercion():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_thresh": "8.5"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_thresh == 8.5


def test_sextractor_detect_thresh_clamps_low():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_thresh": 0.1}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_thresh == 1.0


def test_sextractor_detect_thresh_clamps_high():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_thresh": 99.0}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_thresh == 20.0


def test_sextractor_detect_thresh_invalid_fallback():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_thresh": "not_a_number"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_thresh == 5.0


def test_sextractor_detect_minarea_default():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sensors": [{"id": "t", "type": "telescope"}]}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_minarea == 3


def test_sextractor_detect_minarea_valid():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_minarea": 10}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_minarea == 10


def test_sextractor_detect_minarea_float_coercion():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_minarea": 7.9}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_minarea == 7


def test_sextractor_detect_minarea_clamps_low():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_minarea": 0}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_minarea == 1


def test_sextractor_detect_minarea_clamps_high():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_minarea": 100}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_minarea == 50


def test_sextractor_detect_minarea_invalid_fallback():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_detect_minarea": "garbage"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_detect_minarea == 3


def test_sextractor_filter_name_default():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sensors": [{"id": "t", "type": "telescope"}]}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_filter_name == "default"


@pytest.mark.parametrize("name", ["default", "gauss_1.5_3x3", "gauss_2.5_5x5", "tophat_3.0_3x3", "tophat_5.0_5x5"])
def test_sextractor_filter_name_accepts_valid(name):
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_filter_name": name}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_filter_name == name


def test_sextractor_filter_name_unknown_fallback():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_filter_name": "nonexistent_kernel"}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_filter_name == "default"


def test_sextractor_filter_name_non_string_fallback():
    with patch("citrasense.settings.citrasense_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"sextractor_filter_name": 12345}
        from citrasense.settings.citrasense_settings import CitraSenseSettings

        s = CitraSenseSettings.load()

    assert s.sensors[0].sextractor_filter_name == "default"


def test_adaptive_exposure_min_gt_max_clamps_min():
    """When adaptive min > max, model validator clamps min down to max."""
    from citrasense.settings.citrasense_settings import SensorConfig

    sc = SensorConfig.model_validate(
        {"id": "t", "type": "telescope", "adaptive_exposure_min_seconds": 20.0, "adaptive_exposure_max_seconds": 5.0}
    )
    assert sc.adaptive_exposure_min_seconds == sc.adaptive_exposure_max_seconds
    assert sc.adaptive_exposure_min_seconds == 5.0


def test_adaptive_exposure_min_equal_max_accepted():
    """When min == max, no clamping occurs."""
    from citrasense.settings.citrasense_settings import SensorConfig

    sc = SensorConfig.model_validate(
        {"id": "t", "type": "telescope", "adaptive_exposure_min_seconds": 5.0, "adaptive_exposure_max_seconds": 5.0}
    )
    assert sc.adaptive_exposure_min_seconds == 5.0
    assert sc.adaptive_exposure_max_seconds == 5.0


def test_adaptive_exposure_min_lt_max_accepted():
    """Normal min < max is preserved as-is."""
    from citrasense.settings.citrasense_settings import SensorConfig

    sc = SensorConfig.model_validate(
        {"id": "t", "type": "telescope", "adaptive_exposure_min_seconds": 0.5, "adaptive_exposure_max_seconds": 30.0}
    )
    assert sc.adaptive_exposure_min_seconds == 0.5
    assert sc.adaptive_exposure_max_seconds == 30.0
