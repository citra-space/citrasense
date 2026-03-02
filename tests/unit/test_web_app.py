"""Unit tests for the FastAPI web application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from citrascope.constants import AUTOFOCUS_TARGET_PRESETS
from citrascope.web.app import CitraScopeWebApp, ConnectionManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.host = "api.citra.space"
    s.port = 443
    s.use_ssl = True
    s.personal_access_token = "tok_abc"
    s.telescope_id = "tel-1"
    s.use_dummy_api = False
    s.hardware_adapter = "nina"
    s._all_adapter_settings = {"nina": {"url_prefix": "http://nina:1888"}}
    s.adapter_settings = {"url_prefix": "http://nina:1888"}
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
    s.config_manager.get_current_log_path.return_value = "/tmp/citrascope.log"
    s.get_images_dir.return_value = MagicMock(
        __str__=lambda self: "/tmp/images",
        exists=lambda: False,
        parent=MagicMock(__truediv__=lambda self, x: MagicMock(__str__=lambda self: f"/tmp/{x}")),
    )
    s.is_configured.return_value = True
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
    d.hardware_adapter.supports_filter_management.return_value = True
    d.hardware_adapter.supports_autofocus.return_value = True
    d.hardware_adapter.get_filter_config.return_value = {0: {"name": "L", "enabled": True}}
    d.hardware_adapter.get_missing_dependencies.return_value = []
    d.hardware_adapter.mount.cached_state = None
    d.task_manager = MagicMock()
    d.task_manager.current_task_id = None
    d.task_manager.task_heap = []
    d.task_manager.heap_lock = MagicMock()
    d.task_manager.heap_lock.__enter__ = MagicMock(return_value=None)
    d.task_manager.heap_lock.__exit__ = MagicMock(return_value=False)
    d.task_manager._stage_lock = MagicMock()
    d.task_manager._stage_lock.__enter__ = MagicMock(return_value=None)
    d.task_manager._stage_lock.__exit__ = MagicMock(return_value=False)
    d.task_manager.imaging_tasks = {}
    d.task_manager.processing_tasks = {}
    d.task_manager.uploading_tasks = {}
    d.task_manager.autofocus_manager = MagicMock()
    d.task_manager.autofocus_manager.is_requested.return_value = False
    d.task_manager.autofocus_manager.is_running.return_value = False
    d.task_manager.autofocus_manager.progress = ""
    d.task_manager.is_processing_active.return_value = True
    d.task_manager._automated_scheduling = False
    d.task_manager.get_tasks_by_stage.return_value = {}
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
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=mock_daemon)
    return app


@pytest.fixture
def client(web_app):
    return TestClient(web_app.app)


# ---------------------------------------------------------------------------
# Version / Adapters
# ---------------------------------------------------------------------------


def test_get_version(client):
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_get_hardware_adapters(client):
    resp = client.get("/api/hardware-adapters")
    assert resp.status_code == 200
    data = resp.json()
    assert "adapters" in data
    assert "descriptions" in data


def test_get_adapter_schema(client):
    resp = client.get("/api/hardware-adapters/nina/schema")
    assert resp.status_code == 200
    assert "schema" in resp.json()


def test_get_adapter_schema_unknown(client):
    resp = client.get("/api/hardware-adapters/does_not_exist/schema")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_get_config(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["hardware_adapter"] == "nina"
    assert data["telescope_id"] == "tel-1"
    assert "autofocus_target_preset" in data


def test_get_config_no_daemon(mock_daemon):
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.get("/api/config")
    assert resp.status_code == 503


def test_get_config_status(client):
    resp = client.get("/api/config/status")
    assert resp.status_code == 200
    assert resp.json()["configured"] is True


def test_get_config_status_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.get("/api/config/status")
    assert resp.status_code == 200
    assert resp.json()["configured"] is False


def test_post_config(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (True, None)
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok_new",
            "telescope_id": "tel-2",
            "hardware_adapter": "nina",
            "adapter_settings": {"nina_api_path": "http://new:1888/v2/api"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_post_config_missing_field(client):
    resp = client.post("/api/config", json={"personal_access_token": "tok"})
    assert resp.status_code == 400


def test_post_config_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.post(
        "/api/config",
        json={"personal_access_token": "x", "telescope_id": "y", "hardware_adapter": "z"},
    )
    assert resp.status_code == 503


def test_post_config_reload_fails(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (False, "adapter not found")
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "nina",
            "adapter_settings": {"nina_api_path": "http://localhost:1888/v2/api"},
        },
    )
    assert resp.status_code == 500
    assert "reload failed" in resp.json()["message"]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_get_status(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["telescope_connected"] is True
    assert data["camera_connected"] is True


def test_get_status_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.get("/api/status")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def test_get_tasks_empty(client):
    resp = client.get("/api/tasks")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_active_tasks_empty(client):
    resp = client.get("/api/tasks/active")
    assert resp.status_code == 200
    assert resp.json() == []


def test_pause_tasks(client, mock_daemon):
    resp = client.post("/api/tasks/pause")
    assert resp.status_code == 200
    mock_daemon.task_manager.pause.assert_called_once()


def test_resume_tasks(client, mock_daemon):
    resp = client.post("/api/tasks/resume")
    assert resp.status_code == 200
    mock_daemon.task_manager.resume.assert_called_once()


def test_pause_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/tasks/pause").status_code == 503


# ---------------------------------------------------------------------------
# Emergency stop
# ---------------------------------------------------------------------------


def test_emergency_stop(client, mock_daemon):
    mock_daemon.task_manager.clear_pending_tasks.return_value = 2
    resp = client.post("/api/emergency-stop")
    assert resp.status_code == 202
    data = resp.json()
    assert data["success"] is True
    assert "2" in data["message"]
    mock_daemon.safety_monitor.activate_operator_stop.assert_called_once()
    mock_daemon.task_manager.pause.assert_called_once()
    mock_daemon.task_manager.clear_pending_tasks.assert_called_once()
    assert mock_daemon.settings.task_processing_paused is True
    mock_daemon.settings.save.assert_called_once()


def test_emergency_stop_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/emergency-stop").status_code == 503


def test_emergency_stop_no_task_manager(mock_daemon):
    """Mount halt and operator stop still fire even without a task manager."""
    mock_daemon.task_manager = None
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=mock_daemon)
    c = TestClient(app.app)
    resp = c.post("/api/emergency-stop")
    assert resp.status_code == 202
    assert mock_daemon.settings.task_processing_paused is True
    mock_daemon.safety_monitor.activate_operator_stop.assert_called_once()


def test_clear_operator_stop(client, mock_daemon):
    resp = client.post("/api/safety/operator-stop/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    mock_daemon.safety_monitor.clear_operator_stop.assert_called_once()
    mock_daemon.task_manager.resume.assert_called_once()
    assert mock_daemon.settings.task_processing_paused is False
    mock_daemon.settings.save.assert_called_once()


def test_clear_operator_stop_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/safety/operator-stop/clear").status_code == 503


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def test_get_logs_no_handler(client):
    resp = client.get("/api/logs")
    assert resp.status_code == 200
    assert resp.json() == {"logs": []}


def test_get_logs_with_handler(mock_daemon):
    handler = MagicMock()
    handler.get_recent_logs.return_value = [{"level": "INFO", "message": "hello"}]
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=mock_daemon, web_log_handler=handler)
    c = TestClient(app.app)
    resp = c.get("/api/logs?limit=50")
    assert resp.status_code == 200
    assert len(resp.json()["logs"]) == 1


# ---------------------------------------------------------------------------
# Autofocus
# ---------------------------------------------------------------------------


def test_get_autofocus_presets(client):
    resp = client.get("/api/adapter/autofocus/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "presets" in data
    assert len(data["presets"]) == len(AUTOFOCUS_TARGET_PRESETS)


def test_trigger_autofocus(client, mock_daemon):
    mock_daemon.trigger_autofocus.return_value = (True, None)
    resp = client.post("/api/adapter/autofocus")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_trigger_autofocus_no_daemon():
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/adapter/autofocus").status_code == 503


def test_cancel_autofocus(client, mock_daemon):
    mock_daemon.cancel_autofocus.return_value = True
    resp = client.post("/api/adapter/autofocus/cancel")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_get_filters(client):
    resp = client.get("/api/adapter/filters")
    assert resp.status_code == 200
    assert "filters" in resp.json()


def test_get_filters_no_adapter():
    d = MagicMock()
    d.settings = MagicMock()
    d.settings.get_images_dir.return_value = MagicMock(exists=lambda: False)
    d.hardware_adapter = None
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=d)
    c = TestClient(app.app)
    assert c.get("/api/adapter/filters").status_code == 503


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def test_get_processors(client):
    resp = client.get("/api/processors")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------


def test_camera_capture_not_supported(client, mock_daemon):
    mock_daemon.hardware_adapter.supports_direct_camera_control.return_value = False
    resp = client.post("/api/camera/capture", json={"duration": 1.0})
    assert resp.status_code == 400


def test_camera_capture_bad_duration(client, mock_daemon):
    mock_daemon.hardware_adapter.supports_direct_camera_control.return_value = True
    resp = client.post("/api/camera/capture", json={"duration": -1})
    assert resp.status_code == 400


def test_camera_capture_too_long(client, mock_daemon):
    mock_daemon.hardware_adapter.supports_direct_camera_control.return_value = True
    resp = client.post("/api/camera/capture", json={"duration": 999})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Automated scheduling
# ---------------------------------------------------------------------------


def test_toggle_automated_scheduling(client, mock_daemon):
    mock_daemon.api_client._request.return_value = True
    resp = client.patch("/api/telescope/automated-scheduling", json={"enabled": True})
    assert resp.status_code == 200


def test_toggle_automated_scheduling_missing_field(client):
    resp = client.patch("/api/telescope/automated-scheduling", json={})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


def test_connection_manager_disconnect_unknown():
    cm = ConnectionManager()
    cm.disconnect(MagicMock())
    assert len(cm.active_connections) == 0


# ---------------------------------------------------------------------------
# Filter batch updates
# ---------------------------------------------------------------------------


def _batch_client(filter_map=None):
    """Create a test client with filter map configured."""
    d = MagicMock()
    d.settings = MagicMock()
    d.settings.get_images_dir.return_value = MagicMock(exists=lambda: False)
    d.hardware_adapter = MagicMock()
    d.hardware_adapter.filter_map = filter_map or {
        0: {"name": "Lum", "focus_position": 9000, "enabled": True},
        1: {"name": "Red", "focus_position": 9050, "enabled": True},
    }
    d.hardware_adapter.update_filter_focus.return_value = True
    d.hardware_adapter.update_filter_enabled.return_value = True
    d.hardware_adapter.supports_filter_management.return_value = True
    with patch("citrascope.web.app.StaticFiles"):
        app = CitraScopeWebApp(daemon=d)
    return TestClient(app.app), d


def test_filter_batch_update_focus():
    c, d = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "0", "focus_position": 5000}])
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 1
    d.hardware_adapter.update_filter_focus.assert_called_once_with("0", 5000)


def test_filter_batch_update_enabled():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "1", "enabled": False}])
    assert resp.status_code == 200


def test_filter_batch_missing_filter_id():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"focus_position": 5000}])
    assert resp.status_code == 400


def test_filter_batch_invalid_filter_id():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "abc"}])
    assert resp.status_code == 400


def test_filter_batch_not_found_filter():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "999"}])
    assert resp.status_code == 404


def test_filter_batch_bad_focus_position():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "0", "focus_position": -1}])
    assert resp.status_code == 400


def test_filter_batch_too_high_focus():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "0", "focus_position": 99999}])
    assert resp.status_code == 400


def test_filter_batch_non_int_focus():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "0", "focus_position": "oops"}])
    assert resp.status_code == 400


def test_filter_batch_non_bool_enabled():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[{"filter_id": "0", "enabled": "yes"}])
    assert resp.status_code == 400


def test_filter_batch_disable_all_blocked():
    c, _ = _batch_client()
    resp = c.post(
        "/api/adapter/filters/batch",
        json=[
            {"filter_id": "0", "enabled": False},
            {"filter_id": "1", "enabled": False},
        ],
    )
    assert resp.status_code == 400
    assert "Cannot disable all" in resp.json()["error"]


def test_filter_batch_empty_array():
    c, _ = _batch_client()
    resp = c.post("/api/adapter/filters/batch", json=[])
    assert resp.status_code == 400


def test_filter_sync():
    c, d = _batch_client()
    resp = c.post("/api/adapter/filters/sync")
    assert resp.status_code == 200
    d._sync_filters_to_backend.assert_called_once()


# ---------------------------------------------------------------------------
# Status with rich daemon state
# ---------------------------------------------------------------------------


def test_status_with_time_monitor(client, mock_daemon):
    mock_daemon.time_monitor = MagicMock()
    health = MagicMock()
    health.to_dict.return_value = {"offset_ms": 5.0, "status": "ok"}
    mock_daemon.time_monitor.get_current_health.return_value = health
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["time_health"]["status"] == "ok"


def test_status_with_ground_station(client, mock_daemon):
    mock_daemon.ground_station = {"id": "gs-1", "name": "Desert Station"}
    resp = client.get("/api/status")
    data = resp.json()
    assert data["ground_station_name"] == "Desert Station"


def test_status_telescope_disconnected(client, mock_daemon):
    mock_daemon.hardware_adapter.is_telescope_connected.side_effect = Exception("no connection")
    resp = client.get("/api/status")
    assert resp.json()["telescope_connected"] is False


def test_status_with_processors(client, mock_daemon):
    p = MagicMock()
    p.name = "plate_solver"
    mock_daemon.processor_registry.processors = [p]
    resp = client.get("/api/status")
    assert "plate_solver" in resp.json()["active_processors"]


def test_status_scheduled_autofocus(client, mock_daemon):
    mock_daemon.settings.scheduled_autofocus_enabled = True
    mock_daemon.settings.last_autofocus_timestamp = int(__import__("time").time()) - 1800
    resp = client.get("/api/status")
    data = resp.json()
    assert data["next_autofocus_minutes"] is not None


def test_status_scheduled_autofocus_never_run(client, mock_daemon):
    mock_daemon.settings.scheduled_autofocus_enabled = True
    mock_daemon.settings.last_autofocus_timestamp = None
    resp = client.get("/api/status")
    assert resp.json()["next_autofocus_minutes"] == 0


# ---------------------------------------------------------------------------
# POST config with int/float adapter settings validation
# ---------------------------------------------------------------------------


def test_post_config_validates_int_field(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (True, None)
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "nina",
            "adapter_settings": {"nina_api_path": "http://x:1888/v2/api", "binning_x": "not_int"},
        },
    )
    assert resp.status_code == 400
    assert "integer" in resp.json()["error"]


def test_post_config_validates_int_range(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (True, None)
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok",
            "telescope_id": "tel",
            "hardware_adapter": "nina",
            "adapter_settings": {"nina_api_path": "http://x:1888/v2/api", "binning_x": 99},
        },
    )
    assert resp.status_code == 400
