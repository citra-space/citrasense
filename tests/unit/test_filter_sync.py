"""Tests for filter synchronization utilities."""

from unittest.mock import MagicMock

from citrasense.hardware.filter_sync import (
    build_spectral_config_from_expanded,
    extract_enabled_filter_names,
    sync_filters_to_backend,
)


class TestExtractEnabledFilterNames:
    def test_basic_extraction(self):
        config = {
            "1": {"name": "Red", "enabled": True},
            "2": {"name": "Green", "enabled": False},
            "3": {"name": "Blue", "enabled": True},
        }
        result = extract_enabled_filter_names(config)
        assert result == ["Red", "Blue"]

    def test_empty_config(self):
        assert extract_enabled_filter_names({}) == []

    def test_none_enabled(self):
        config = {
            "1": {"name": "Red", "enabled": False},
            "2": {"name": "Green", "enabled": False},
        }
        assert extract_enabled_filter_names(config) == []

    def test_missing_enabled_key(self):
        config = {"1": {"name": "Red"}}
        assert extract_enabled_filter_names(config) == []


class TestBuildSpectralConfigFromExpanded:
    def test_all_known(self):
        expanded = [
            {"name": "Red", "central_wavelength_nm": 635.0, "bandwidth_nm": 120.0, "is_known": True},
            {"name": "Green", "central_wavelength_nm": 530.0, "bandwidth_nm": 100.0, "is_known": True},
        ]
        config, unknown = build_spectral_config_from_expanded(expanded)
        assert config["type"] == "discrete"
        assert len(config["filters"]) == 2
        assert unknown == []

    def test_unknown_filters_reported(self):
        expanded = [
            {"name": "CustomBand", "central_wavelength_nm": 550.0, "bandwidth_nm": 100.0, "is_known": False},
        ]
        _, unknown = build_spectral_config_from_expanded(expanded)
        assert unknown == ["CustomBand"]

    def test_mixed_known_unknown(self):
        expanded = [
            {"name": "Red", "central_wavelength_nm": 635.0, "bandwidth_nm": 120.0, "is_known": True},
            {"name": "MyFilter", "central_wavelength_nm": 700.0, "bandwidth_nm": 50.0, "is_known": False},
        ]
        config, unknown = build_spectral_config_from_expanded(expanded)
        assert len(config["filters"]) == 2
        assert unknown == ["MyFilter"]

    def test_empty_list(self):
        config, unknown = build_spectral_config_from_expanded([])
        assert config == {"type": "discrete", "filters": []}
        assert unknown == []

    def test_missing_is_known_defaults_true(self):
        expanded = [{"name": "Ha", "central_wavelength_nm": 656.3, "bandwidth_nm": 7.0}]
        _, unknown = build_spectral_config_from_expanded(expanded)
        assert unknown == []


class TestSyncFiltersToBackend:
    def test_empty_config_returns_false(self):
        logger = MagicMock()
        assert sync_filters_to_backend(MagicMock(), "t1", {}, logger) is False

    def test_none_config_returns_false(self):
        logger = MagicMock()
        assert sync_filters_to_backend(MagicMock(), "t1", None, logger) is False

    def test_no_enabled_filters_returns_false(self):
        logger = MagicMock()
        config = {"1": {"name": "Red", "enabled": False}}
        assert sync_filters_to_backend(MagicMock(), "t1", config, logger) is False

    def test_successful_sync(self):
        logger = MagicMock()
        api = MagicMock()
        api.expand_filters.return_value = {
            "filters": [
                {"name": "Red", "central_wavelength_nm": 635.0, "bandwidth_nm": 120.0, "is_known": True},
            ]
        }
        api.update_telescope_spectral_config.return_value = {"status": "ok"}
        config = {"1": {"name": "Red", "enabled": True}}
        assert sync_filters_to_backend(api, "t1", config, logger) is True
        api.expand_filters.assert_called_once_with(["Red"])
        api.update_telescope_spectral_config.assert_called_once()

    def test_expand_fails_returns_false(self):
        logger = MagicMock()
        api = MagicMock()
        api.expand_filters.return_value = None
        config = {"1": {"name": "Red", "enabled": True}}
        assert sync_filters_to_backend(api, "t1", config, logger) is False

    def test_expand_missing_filters_key_returns_false(self):
        logger = MagicMock()
        api = MagicMock()
        api.expand_filters.return_value = {"error": "bad"}
        config = {"1": {"name": "Red", "enabled": True}}
        assert sync_filters_to_backend(api, "t1", config, logger) is False

    def test_update_fails_returns_false(self):
        logger = MagicMock()
        api = MagicMock()
        api.expand_filters.return_value = {
            "filters": [
                {"name": "Red", "central_wavelength_nm": 635.0, "bandwidth_nm": 120.0, "is_known": True},
            ]
        }
        api.update_telescope_spectral_config.return_value = None
        config = {"1": {"name": "Red", "enabled": True}}
        assert sync_filters_to_backend(api, "t1", config, logger) is False

    def test_unknown_filters_logged(self):
        logger = MagicMock()
        api = MagicMock()
        api.expand_filters.return_value = {
            "filters": [
                {"name": "MyBand", "central_wavelength_nm": 550.0, "bandwidth_nm": 100.0, "is_known": False},
            ]
        }
        api.update_telescope_spectral_config.return_value = {"status": "ok"}
        config = {"1": {"name": "MyBand", "enabled": True}}
        sync_filters_to_backend(api, "t1", config, logger)
        logger.warning.assert_called()
