"""Unit tests for CitraScopeSettings and SettingsFileManager."""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# SettingsFileManager
# ---------------------------------------------------------------------------


@pytest.fixture
def sfm(tmp_path):
    """SettingsFileManager pointed at a temp directory."""
    from citrascope.settings.citrascope_settings import CitraScopeSettings  # noqa: F401 — triggers module load
    from citrascope.settings.settings_file_manager import SettingsFileManager

    mgr = SettingsFileManager()
    mgr.config_dir = tmp_path
    mgr.config_file = tmp_path / "config.json"
    mgr.log_dir = tmp_path / "logs"
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


def test_ensure_log_directory(sfm):
    sfm.ensure_log_directory()
    assert sfm.log_dir.exists()


def test_get_log_dir(sfm):
    assert sfm.get_log_dir() == sfm.log_dir


def test_get_current_log_path(sfm):
    p = sfm.get_current_log_path()
    assert "citrascope-" in p.name
    assert p.suffix == ".log"


# ---------------------------------------------------------------------------
# CitraScopeSettings
# ---------------------------------------------------------------------------


def test_settings_defaults(tmp_path):
    """Settings should use sensible defaults when config is empty."""
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {}

        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.hardware_adapter == ""
    assert s.personal_access_token == ""
    assert s.is_configured() is False
    assert s.autofocus_target_preset == "mirach"
    assert s.max_task_retries == 3


def test_settings_to_dict(tmp_path):
    """to_dict should include all persistent settings."""
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "dummy",
        }

        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    d = s.to_dict()
    assert d["personal_access_token"] == "tok"
    assert d["telescope_id"] == "tel"
    assert d["hardware_adapter"] == "dummy"
    assert "autofocus_target_preset" in d
    assert "elset_refresh_interval_hours" in d
    assert "web_port" not in d


def test_settings_is_configured(tmp_path):
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "nina_advanced_http",
        }
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.is_configured() is True


def test_settings_validates_custom_ra_out_of_range():
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "autofocus_target_custom_ra": 999.0,
            "autofocus_target_custom_dec": -100.0,
        }
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.autofocus_target_custom_ra is None
    assert s.autofocus_target_custom_dec is None


def test_settings_validates_autofocus_interval():
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"autofocus_interval_minutes": -5}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.autofocus_interval_minutes == 60


def test_settings_save(tmp_path):
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"hardware_adapter": "dummy"}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()
        s.save()

    instance.save_config.assert_called_once()


def test_settings_update_and_save():
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {"hardware_adapter": "dummy"}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()
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
    assert saved["adapter_settings"]["dummy"]["some_key"] == "val"


def test_update_and_save_preserves_fields_not_in_payload():
    """Backend-only fields survive a web UI save that omits them (data-loss bug fix)."""
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        instance = MockSFM.return_value
        instance.load_config.return_value = {
            "hardware_adapter": "dummy",
            "elset_refresh_interval_hours": 12,
            "observation_mode": "tracking",
        }
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()
        s.update_and_save(
            {
                "hardware_adapter": "dummy",
                "adapter_settings": {},
                "personal_access_token": "new_tok",
            }
        )

    saved = instance.save_config.call_args[0][0]
    assert saved["elset_refresh_interval_hours"] == 12
    assert saved["observation_mode"] == "tracking"
    assert saved["personal_access_token"] == "new_tok"


# ---------------------------------------------------------------------------
# Observation mode
# ---------------------------------------------------------------------------


def test_observation_mode_defaults_to_auto():
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.observation_mode == "auto"


@pytest.mark.parametrize("mode", ["auto", "tracking", "static"])
def test_observation_mode_accepts_valid_values(mode):
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": mode}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.observation_mode == mode


def test_observation_mode_rejects_invalid_value():
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": "bogus"}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.observation_mode == "auto"


def test_observation_mode_in_to_dict():
    with patch("citrascope.settings.citrascope_settings.SettingsFileManager") as MockSFM:
        MockSFM.return_value.load_config.return_value = {"observation_mode": "tracking"}
        from citrascope.settings.citrascope_settings import CitraScopeSettings

        s = CitraScopeSettings.load()

    assert s.to_dict()["observation_mode"] == "tracking"
