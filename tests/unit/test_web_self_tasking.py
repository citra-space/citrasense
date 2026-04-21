"""Tests for the observing-session and self-tasking web endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from citrasense.web.app import CitraSenseWebApp


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.observing_session_enabled = False
    s.self_tasking_enabled = False
    s.host = "api.citra.space"
    s.port = 443
    s.use_ssl = True
    s.personal_access_token = "tok"
    s.telescope_id = "tel-1"
    s.use_dummy_api = False
    s.hardware_adapter = "nina"
    s._all_adapter_settings = {}
    s.adapter_settings = {}
    s.log_level = "INFO"
    s.keep_images = False
    s.file_logging_enabled = True
    s.log_retention_days = 30
    s.max_task_retries = 3
    s.initial_retry_delay_seconds = 30
    s.max_retry_delay_seconds = 300
    s.scheduled_autofocus_enabled = False
    s.autofocus_interval_minutes = 60
    s.last_autofocus_timestamp = None
    s.autofocus_target_preset = "mirach"
    s.autofocus_target_custom_ra = None
    s.autofocus_target_custom_dec = None
    s.time_check_interval_minutes = 5
    s.time_offset_pause_ms = 500.0
    s.gps_location_updates_enabled = True
    s.gps_update_interval_minutes = 5
    s.processors_enabled = True
    s.enabled_processors = {}
    s.config_manager = MagicMock()
    s.config_manager.get_config_path.return_value = "/tmp/config.json"
    s.directories = MagicMock()
    s.directories.images_dir = MagicMock(__str__=lambda self: "/tmp/images", exists=lambda: False)
    s.directories.processing_dir = MagicMock(__str__=lambda self: "/tmp/processing")
    s.directories.current_log_path.return_value = "/tmp/citrasense.log"
    s.is_configured.return_value = True
    s.keep_processing_output = False
    s.alignment_exposure_seconds = 2.0
    s.last_alignment_timestamp = None
    s.task_processing_paused = False
    s.observation_mode = "auto"
    s.elset_refresh_interval_hours = 6
    s.to_dict.return_value = {}
    return s


@pytest.fixture
def mock_daemon(mock_settings):
    d = MagicMock()
    d.settings = mock_settings
    d.hardware_adapter = MagicMock()
    d.hardware_adapter.is_telescope_connected.return_value = True
    d.hardware_adapter.is_camera_connected.return_value = True
    d.hardware_adapter.get_telescope_direction.return_value = (180.0, 45.0)
    d.hardware_adapter.supports_direct_camera_control.return_value = False
    d.hardware_adapter.supports_filter_management.return_value = False
    d.hardware_adapter.supports_autofocus.return_value = False
    d.hardware_adapter.get_missing_dependencies.return_value = []
    d.hardware_adapter.mount = None
    d.task_manager = MagicMock()
    d.task_manager.current_task_id = None
    d.task_manager.autofocus_manager = MagicMock()
    d.task_manager.autofocus_manager.is_requested.return_value = False
    d.task_manager.autofocus_manager.is_running.return_value = False
    d.task_manager.autofocus_manager.progress = ""
    d.task_manager.autofocus_manager.get_next_autofocus_minutes.return_value = None
    d.task_manager.alignment_manager = MagicMock()
    d.task_manager.alignment_manager.is_requested.return_value = False
    d.task_manager.alignment_manager.is_running.return_value = False
    d.task_manager.alignment_manager.progress = ""
    d.task_manager.is_processing_active.return_value = True
    d.task_manager.automated_scheduling = False
    d.task_manager.observing_session_manager = None
    d.task_manager.self_tasking_manager = None
    d.task_manager.get_tasks_by_stage.return_value = {}
    d.task_manager.get_tasks_snapshot.return_value = []
    d.task_manager.pending_task_count = 0
    d.task_manager.imaging_queue = MagicMock()
    d.task_manager.imaging_queue.get_stats.return_value = {"attempts": 0, "successes": 0, "permanent_failures": 0}
    d.task_manager.processing_queue = MagicMock()
    d.task_manager.processing_queue.get_stats.return_value = {"attempts": 0, "successes": 0, "permanent_failures": 0}
    d.task_manager.upload_queue = MagicMock()
    d.task_manager.upload_queue.get_stats.return_value = {"attempts": 0, "successes": 0, "permanent_failures": 0}
    d.task_manager.get_task_stats.return_value = {}
    d.processor_registry = MagicMock()
    d.processor_registry.processors = []
    d.processor_registry.get_all_processors.return_value = []
    d.processor_registry.get_processor_stats.return_value = {}
    d.time_monitor = None
    d.location_service = None
    d.ground_station = None
    d.api_client = MagicMock()
    d.telescope_record = {"id": "tel-1", "name": "Test Scope"}
    return d


@pytest.fixture
def web_app(mock_daemon):
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=mock_daemon)
    return app


@pytest.fixture
def client(web_app):
    return TestClient(web_app.app)


def test_toggle_observing_session_enable(client, mock_daemon):
    resp = client.patch("/api/observing-session", json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert mock_daemon.settings.observing_session_enabled is True
    mock_daemon.settings.save.assert_called()


def test_toggle_observing_session_disable(client, mock_daemon):
    mock_daemon.settings.observing_session_enabled = True
    resp = client.patch("/api/observing-session", json={"enabled": False})
    assert resp.status_code == 200
    assert mock_daemon.settings.observing_session_enabled is False


def test_toggle_observing_session_missing_field(client):
    resp = client.patch("/api/observing-session", json={})
    assert resp.status_code == 400


def test_toggle_observing_session_no_daemon(web_app):
    web_app.daemon = None
    client = TestClient(web_app.app)
    resp = client.patch("/api/observing-session", json={"enabled": True})
    assert resp.status_code == 503


def test_toggle_self_tasking_enable(client, mock_daemon):
    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    mock_daemon.settings.save.assert_called()
    assert mock_daemon.settings.self_tasking_enabled is True


def test_toggle_self_tasking_enable_cascades_observing_session(client, mock_daemon):
    """Enabling self-tasking should auto-enable observing session."""
    mock_daemon.settings.observing_session_enabled = False

    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 200
    assert mock_daemon.settings.observing_session_enabled is True


def test_toggle_self_tasking_enable_cascades_scheduling(client, mock_daemon):
    """Enabling self-tasking should auto-enable scheduling on the server."""
    mock_daemon.task_manager.automated_scheduling = False
    mock_daemon.api_client.update_telescope_automated_scheduling.return_value = True

    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 200

    mock_daemon.api_client.update_telescope_automated_scheduling.assert_called_once_with("tel-1", True)
    assert mock_daemon.task_manager.automated_scheduling is True


def test_toggle_self_tasking_enable_cascades_processing(client, mock_daemon):
    """Enabling self-tasking should auto-resume processing if paused."""
    mock_daemon.task_manager.is_processing_active.return_value = False

    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 200

    mock_daemon.task_manager.resume.assert_called_once()
    assert mock_daemon.settings.task_processing_paused is False


def test_toggle_self_tasking_enable_skips_cascade_when_already_active(client, mock_daemon):
    """No cascade calls when scheduling, processing, and session are already active."""
    mock_daemon.settings.observing_session_enabled = True
    mock_daemon.task_manager.automated_scheduling = True
    mock_daemon.task_manager.is_processing_active.return_value = True

    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 200

    mock_daemon.task_manager.resume.assert_not_called()
    mock_daemon.api_client.update_telescope_automated_scheduling.assert_not_called()


def test_toggle_self_tasking_enable_scheduling_failure_non_blocking(client, mock_daemon):
    """If the scheduling API call fails, self-tasking still enables."""
    mock_daemon.task_manager.automated_scheduling = False
    mock_daemon.api_client.update_telescope_automated_scheduling.side_effect = Exception("network error")

    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 200
    assert mock_daemon.settings.self_tasking_enabled is True


def test_toggle_self_tasking_disable_no_cascade(client, mock_daemon):
    """Disabling self-tasking should NOT touch scheduling, processing, or session."""
    mock_daemon.settings.self_tasking_enabled = True
    mock_daemon.settings.observing_session_enabled = True
    mock_daemon.task_manager.automated_scheduling = True
    mock_daemon.task_manager.is_processing_active.return_value = True

    resp = client.patch("/api/self-tasking", json={"enabled": False})
    assert resp.status_code == 200
    assert mock_daemon.settings.self_tasking_enabled is False
    assert mock_daemon.settings.observing_session_enabled is True

    mock_daemon.task_manager.resume.assert_not_called()
    mock_daemon.api_client.update_telescope_automated_scheduling.assert_not_called()


def test_toggle_self_tasking_missing_field(client):
    resp = client.patch("/api/self-tasking", json={})
    assert resp.status_code == 400


def test_toggle_self_tasking_no_daemon(web_app):
    web_app.daemon = None
    client = TestClient(web_app.app)
    resp = client.patch("/api/self-tasking", json={"enabled": True})
    assert resp.status_code == 503


def test_status_includes_self_tasking_fields(web_app, mock_daemon):
    """Verify the SystemStatus model has observing session and self-tasking fields."""
    from citrasense.web.app import SystemStatus

    status = SystemStatus()
    assert hasattr(status, "observing_session_enabled")
    assert hasattr(status, "self_tasking_enabled")
    assert hasattr(status, "observing_session_state")
    assert hasattr(status, "sun_altitude")
    assert hasattr(status, "dark_window_start")
    assert hasattr(status, "dark_window_end")
    assert hasattr(status, "last_batch_request")
    assert hasattr(status, "last_batch_created")
    assert status.observing_session_state == "daytime"
