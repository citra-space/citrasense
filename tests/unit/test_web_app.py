"""Unit tests for the FastAPI web application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from citrasense.constants import AUTOFOCUS_TARGET_PRESETS
from citrasense.web.app import CitraSenseWebApp, ConnectionManager

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
    s.directories = MagicMock()
    s.directories.images_dir = MagicMock(
        __str__=lambda self: "/tmp/images",
        exists=lambda: False,
    )
    s.directories.processing_dir = MagicMock(__str__=lambda self: "/tmp/processing")
    s.directories.current_log_path.return_value = "/tmp/citrasense.log"
    s.is_configured.return_value = True
    s.keep_processing_output = False
    s.alignment_exposure_seconds = 2.0
    s.last_alignment_timestamp = None
    s.observation_mode = "auto"
    s.elset_refresh_interval_hours = 6

    sc = MagicMock()
    sc.id = "scope-0"
    sc.task_processing_paused = False
    sc.observing_session_enabled = False
    sc.self_tasking_enabled = False
    s.sensors = [sc]
    s.get_sensor_config = lambda sid: next((c for c in s.sensors if c.id == sid), None)

    s.to_dict.return_value = {
        "host": s.host,
        "port": s.port,
        "use_ssl": s.use_ssl,
        "personal_access_token": s.personal_access_token,
        "telescope_id": s.telescope_id,
        "use_dummy_api": s.use_dummy_api,
        "hardware_adapter": s.hardware_adapter,
        "adapter_settings": s._all_adapter_settings,
        "log_level": s.log_level,
        "keep_images": s.keep_images,
        "keep_processing_output": s.keep_processing_output,
        "processors_enabled": s.processors_enabled,
        "enabled_processors": s.enabled_processors,
        "max_task_retries": s.max_task_retries,
        "initial_retry_delay_seconds": s.initial_retry_delay_seconds,
        "max_retry_delay_seconds": s.max_retry_delay_seconds,
        "file_logging_enabled": s.file_logging_enabled,
        "log_retention_days": s.log_retention_days,
        "scheduled_autofocus_enabled": s.scheduled_autofocus_enabled,
        "autofocus_interval_minutes": s.autofocus_interval_minutes,
        "last_autofocus_timestamp": s.last_autofocus_timestamp,
        "autofocus_target_preset": s.autofocus_target_preset,
        "autofocus_target_custom_ra": s.autofocus_target_custom_ra,
        "autofocus_target_custom_dec": s.autofocus_target_custom_dec,
        "alignment_exposure_seconds": s.alignment_exposure_seconds,
        "last_alignment_timestamp": s.last_alignment_timestamp,
        "time_check_interval_minutes": s.time_check_interval_minutes,
        "time_offset_pause_ms": s.time_offset_pause_ms,
        "gps_location_updates_enabled": s.gps_location_updates_enabled,
        "gps_update_interval_minutes": s.gps_update_interval_minutes,
        "task_processing_paused": False,
        "observation_mode": s.observation_mode,
        "elset_refresh_interval_hours": s.elset_refresh_interval_hours,
    }
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
    d.task_dispatcher = MagicMock()
    d.task_dispatcher.current_task_id = None
    d.task_dispatcher.autofocus_manager = MagicMock()
    d.task_dispatcher.autofocus_manager.is_requested.return_value = False
    d.task_dispatcher.autofocus_manager.is_running.return_value = False
    d.task_dispatcher.autofocus_manager.progress = ""
    d.task_dispatcher.autofocus_manager.get_next_autofocus_minutes.return_value = None
    d.task_dispatcher.alignment_manager = MagicMock()
    d.task_dispatcher.alignment_manager.is_requested.return_value = False
    d.task_dispatcher.alignment_manager.is_running.return_value = False
    d.task_dispatcher.alignment_manager.progress = ""
    d.task_dispatcher.is_processing_active.return_value = True
    d.task_dispatcher.automated_scheduling = False
    d.task_dispatcher.get_tasks_by_stage.return_value = {}
    d.task_dispatcher.get_tasks_snapshot.return_value = []
    d.task_dispatcher.pending_task_count = 0
    d.task_dispatcher.imaging_queue = MagicMock()
    d.task_dispatcher.imaging_queue.get_stats.return_value = {"attempts": 0, "successes": 0, "permanent_failures": 0}
    d.task_dispatcher.processing_queue = MagicMock()
    d.task_dispatcher.processing_queue.get_stats.return_value = {"attempts": 0, "successes": 0, "permanent_failures": 0}
    d.task_dispatcher.upload_queue = MagicMock()
    d.task_dispatcher.upload_queue.get_stats.return_value = {"attempts": 0, "successes": 0, "permanent_failures": 0}
    d.task_dispatcher.get_task_stats.return_value = {}
    d.processor_registry = MagicMock()
    d.processor_registry.processors = []
    d.processor_registry.get_all_processors.return_value = []
    d.processor_registry.get_processor_stats.return_value = {}
    d.time_monitor = None
    d.location_service = None
    d.ground_station = None
    d.api_client = MagicMock()
    d.telescope_record = {"id": "tel-1", "name": "Test Scope"}

    mock_sensor = MagicMock()
    mock_sensor.sensor_id = "scope-0"
    mock_sensor.sensor_type = "telescope"
    mock_sensor.is_connected.return_value = True
    mock_sensor.adapter = d.hardware_adapter
    mock_sensor.name = "Test Scope"
    mock_sensor.adapter_key = "nina"
    mock_sensor.citra_record = {"id": "tel-1", "automated_scheduling": False}

    mock_sm = MagicMock()
    mock_sm.get.return_value = mock_sensor
    mock_sm.get_sensor.return_value = mock_sensor
    mock_sm.first_of_type.return_value = mock_sensor
    mock_sm.__iter__ = lambda self: iter([mock_sensor])
    mock_sm.iter_by_type.return_value = iter([mock_sensor])
    d.sensor_manager = mock_sm

    mock_runtime = MagicMock()
    mock_runtime.sensor_id = "scope-0"
    mock_runtime.sensor = mock_sensor
    mock_runtime.homing_manager = d.task_dispatcher.homing_manager
    mock_runtime.alignment_manager = d.task_dispatcher.alignment_manager
    mock_runtime.autofocus_manager = d.task_dispatcher.autofocus_manager
    mock_runtime.acquisition_queue = d.task_dispatcher.imaging_queue
    mock_runtime.processing_queue = d.task_dispatcher.processing_queue
    mock_runtime.upload_queue = d.task_dispatcher.upload_queue
    mock_runtime.calibration_manager = None
    mock_runtime.observing_session_manager = None
    mock_runtime.self_tasking_manager = None
    d.task_dispatcher.get_runtime.return_value = mock_runtime
    d.task_dispatcher._telescope_runtimes.return_value = [mock_runtime]
    d.task_dispatcher._runtimes = {"scope-0": mock_runtime}
    return d


@pytest.fixture
def web_app(mock_daemon):
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=mock_daemon)
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
    data = resp.json()
    assert "version" in data
    assert "install_type" in data
    assert data["install_type"] in ("pypi", "editable", "source")
    assert "git_hash" in data
    assert "git_dirty" in data


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


def test_scan_hardware_returns_schema(client):
    """POST /api/hardware/scan clears probe caches and returns fresh schema."""
    from citrasense.hardware.devices.abstract_hardware_device import AbstractHardwareDevice

    AbstractHardwareDevice._hardware_probe_cache["stale:key"] = (["old"], 0)

    resp = client.post(
        "/api/hardware/scan",
        json={"adapter_name": "nina", "current_settings": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "schema" in data
    assert isinstance(data["schema"], list)
    assert "stale:key" not in AbstractHardwareDevice._hardware_probe_cache


def test_scan_hardware_missing_adapter_name(client):
    resp = client.post("/api/hardware/scan", json={})
    assert resp.status_code == 400


def test_scan_hardware_unknown_adapter(client):
    resp = client.post(
        "/api/hardware/scan",
        json={"adapter_name": "does_not_exist", "current_settings": {}},
    )
    assert resp.status_code == 404


def test_scan_hardware_invalid_current_settings(client):
    resp = client.post(
        "/api/hardware/scan",
        json={"adapter_name": "nina", "current_settings": "not_a_dict"},
    )
    assert resp.status_code == 400
    assert "current_settings" in resp.json()["error"]


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
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.get("/api/config")
    assert resp.status_code == 503


def test_get_config_status(client):
    resp = client.get("/api/config/status")
    assert resp.status_code == 200
    assert resp.json()["configured"] is True


def test_get_config_status_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.get("/api/config/status")
    assert resp.status_code == 200
    assert resp.json()["configured"] is False


def _sensor_config(adapter="nina", citra_sensor_id="tel-1", adapter_settings=None):
    return {
        "id": "telescope-0",
        "type": "telescope",
        "adapter": adapter,
        "citra_sensor_id": citra_sensor_id,
        "adapter_settings": adapter_settings or {},
    }


def test_post_config(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (True, None)
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok_new",
            "sensors": [
                _sensor_config(citra_sensor_id="tel-2", adapter_settings={"nina_api_path": "http://new:1888/v2/api"})
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_post_config_missing_field(client):
    resp = client.post("/api/config", json={"personal_access_token": "tok"})
    assert resp.status_code == 400


def test_post_config_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    resp = c.post(
        "/api/config",
        json={
            "personal_access_token": "x",
            "sensors": [_sensor_config()],
        },
    )
    assert resp.status_code == 503


def test_post_config_reload_fails(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (False, "adapter not found")
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok",
            "sensors": [_sensor_config(adapter_settings={"nina_api_path": "http://localhost:1888/v2/api"})],
        },
    )
    assert resp.status_code == 500
    assert "reload failed" in resp.json()["message"]


# ---------------------------------------------------------------------------
# Hardware reconnect
# ---------------------------------------------------------------------------


def test_reconnect_hardware_success(client, mock_daemon):
    mock_daemon.retry_connection.return_value = (True, None)
    resp = client.post("/api/hardware/reconnect")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    mock_daemon.retry_connection.assert_called_once()


def test_reconnect_hardware_failure(client, mock_daemon):
    mock_daemon.retry_connection.return_value = (False, "NINA not reachable")
    resp = client.post("/api/hardware/reconnect")
    assert resp.status_code == 500
    assert "NINA not reachable" in resp.json()["error"]


def test_reconnect_hardware_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/hardware/reconnect").status_code == 503


def test_reconnect_hardware_not_configured(client, mock_daemon):
    mock_daemon.settings.is_configured.return_value = False
    resp = client.post("/api/hardware/reconnect")
    assert resp.status_code == 400
    assert "incomplete" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_get_status(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    sensor = data["sensors"]["scope-0"]
    assert sensor["telescope_connected"] is True
    assert sensor["camera_connected"] is True


def test_get_status_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
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
    mock_daemon.task_dispatcher.pause.assert_called_once()


def test_resume_tasks(client, mock_daemon):
    resp = client.post("/api/tasks/resume")
    assert resp.status_code == 200
    mock_daemon.task_dispatcher.resume.assert_called_once()


def test_pause_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/tasks/pause").status_code == 503


def test_cancel_task_success(client, mock_daemon):
    mock_daemon.api_client.cancel_task.return_value = True
    resp = client.post("/api/tasks/abc-123/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["task_id"] == "abc-123"
    mock_daemon.api_client.cancel_task.assert_called_once_with("abc-123")
    mock_daemon.task_dispatcher.drop_scheduled_task.assert_called_once_with("abc-123")


def test_cancel_task_server_refused(client, mock_daemon):
    mock_daemon.api_client.cancel_task.return_value = False
    resp = client.post("/api/tasks/abc-123/cancel")
    assert resp.status_code == 409
    mock_daemon.task_dispatcher.drop_scheduled_task.assert_not_called()


def test_cancel_task_active_refused(client, mock_daemon):
    mock_daemon.task_dispatcher.current_task_id = "abc-123"
    resp = client.post("/api/tasks/abc-123/cancel")
    assert resp.status_code == 409
    mock_daemon.api_client.cancel_task.assert_not_called()
    mock_daemon.task_dispatcher.drop_scheduled_task.assert_not_called()


def test_cancel_task_no_api_client(mock_daemon):
    mock_daemon.api_client = None
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=mock_daemon)
    c = TestClient(app.app)
    assert c.post("/api/tasks/abc/cancel").status_code == 503


# ---------------------------------------------------------------------------
# Emergency stop
# ---------------------------------------------------------------------------


def test_emergency_stop(client, mock_daemon):
    mock_daemon.task_dispatcher.clear_pending_tasks.return_value = 2
    resp = client.post("/api/emergency-stop")
    assert resp.status_code == 202
    data = resp.json()
    assert data["success"] is True
    assert "2" in data["message"]
    mock_daemon.safety_monitor.activate_operator_stop.assert_called_once()
    mock_daemon.task_dispatcher.pause.assert_called_once()
    mock_daemon.task_dispatcher.clear_pending_tasks.assert_called_once()
    assert mock_daemon.settings.sensors[0].task_processing_paused is True
    mock_daemon.settings.save.assert_called_once()


def test_emergency_stop_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/emergency-stop").status_code == 503


def test_emergency_stop_no_task_dispatcher(mock_daemon):
    """Mount halt and operator stop still fire even without a task manager."""
    mock_daemon.task_dispatcher = None
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=mock_daemon)
    c = TestClient(app.app)
    resp = c.post("/api/emergency-stop")
    assert resp.status_code == 202
    assert mock_daemon.settings.sensors[0].task_processing_paused is True
    mock_daemon.safety_monitor.activate_operator_stop.assert_called_once()


def test_clear_operator_stop(client, mock_daemon):
    resp = client.post("/api/safety/operator-stop/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    mock_daemon.safety_monitor.clear_operator_stop.assert_called_once()
    mock_daemon.task_dispatcher.resume.assert_not_called()
    mock_daemon.settings.save.assert_not_called()


def test_clear_operator_stop_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
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
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=mock_daemon, web_log_handler=handler)
    c = TestClient(app.app)
    resp = c.get("/api/logs?limit=50")
    assert resp.status_code == 200
    assert len(resp.json()["logs"]) == 1


# ---------------------------------------------------------------------------
# Autofocus
# ---------------------------------------------------------------------------


def test_get_autofocus_presets(client):
    resp = client.get("/api/sensors/scope-0/autofocus/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "presets" in data
    assert len(data["presets"]) == len(AUTOFOCUS_TARGET_PRESETS)


def test_trigger_autofocus(client, mock_daemon):
    mock_daemon.trigger_autofocus.return_value = (True, None)
    resp = client.post("/api/sensors/scope-0/autofocus")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_trigger_autofocus_no_daemon():
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=None)
    c = TestClient(app.app)
    assert c.post("/api/sensors/scope-0/autofocus").status_code == 503


def test_cancel_autofocus(client, mock_daemon):
    mock_daemon.cancel_autofocus.return_value = True
    resp = client.post("/api/sensors/scope-0/autofocus/cancel")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_get_filters(client):
    resp = client.get("/api/sensors/scope-0/filters")
    assert resp.status_code == 200
    assert "filters" in resp.json()


def test_get_filters_no_sensor():
    d = MagicMock()
    d.settings = MagicMock()
    d.settings.directories.images_dir = MagicMock(exists=lambda: False)
    d.sensor_manager = None
    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=d)
    c = TestClient(app.app)
    assert c.get("/api/sensors/scope-0/filters").status_code == 503


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
    resp = client.post("/api/sensors/scope-0/camera/capture", json={"duration": 1.0})
    assert resp.status_code == 400


def test_camera_capture_bad_duration(client, mock_daemon):
    mock_daemon.hardware_adapter.supports_direct_camera_control.return_value = True
    resp = client.post("/api/sensors/scope-0/camera/capture", json={"duration": -1})
    assert resp.status_code == 400


def test_camera_capture_too_long(client, mock_daemon):
    mock_daemon.hardware_adapter.supports_direct_camera_control.return_value = True
    resp = client.post("/api/sensors/scope-0/camera/capture", json={"duration": 999})
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
    d.settings.directories.images_dir = MagicMock(exists=lambda: False)

    adapter = MagicMock()
    adapter.filter_map = filter_map or {
        0: {"name": "Lum", "focus_position": 9000, "enabled": True},
        1: {"name": "Red", "focus_position": 9050, "enabled": True},
    }
    adapter.update_filter_focus.return_value = True
    adapter.update_filter_enabled.return_value = True
    adapter.supports_filter_management.return_value = True

    sensor = MagicMock()
    sensor.sensor_id = "scope-0"
    sensor.adapter = adapter

    sm = MagicMock()
    sm.get_sensor.return_value = sensor

    runtime = MagicMock()
    d.sensor_manager = sm
    d.task_dispatcher = MagicMock()
    d.task_dispatcher.get_runtime.return_value = runtime

    with patch("citrasense.web.app.StaticFiles"):
        app = CitraSenseWebApp(daemon=d)
    return TestClient(app.app), adapter, d


def test_filter_batch_update_focus():
    c, adapter, _daemon = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "0", "focus_position": 5000}])
    assert resp.status_code == 200
    assert resp.json()["updated_count"] == 1
    adapter.update_filter_focus.assert_called_once_with("0", 5000)


def test_filter_batch_update_enabled():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "1", "enabled": False}])
    assert resp.status_code == 200


def test_filter_batch_missing_filter_id():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"focus_position": 5000}])
    assert resp.status_code == 400


def test_filter_batch_invalid_filter_id():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "abc"}])
    assert resp.status_code == 400


def test_filter_batch_not_found_filter():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "999"}])
    assert resp.status_code == 404


def test_filter_batch_bad_focus_position():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "0", "focus_position": -1}])
    assert resp.status_code == 400


def test_filter_batch_too_high_focus():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "0", "focus_position": 99999}])
    assert resp.status_code == 400


def test_filter_batch_non_int_focus():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "0", "focus_position": "oops"}])
    assert resp.status_code == 400


def test_filter_batch_non_bool_enabled():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[{"filter_id": "0", "enabled": "yes"}])
    assert resp.status_code == 400


def test_filter_batch_disable_all_blocked():
    c, *_ = _batch_client()
    resp = c.post(
        "/api/sensors/scope-0/filters/batch",
        json=[
            {"filter_id": "0", "enabled": False},
            {"filter_id": "1", "enabled": False},
        ],
    )
    assert resp.status_code == 400
    assert "Cannot disable all" in resp.json()["error"]


def test_filter_batch_empty_array():
    c, *_ = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/batch", json=[])
    assert resp.status_code == 400


def test_filter_sync():
    c, _adapter, daemon = _batch_client()
    resp = c.post("/api/sensors/scope-0/filters/sync")
    assert resp.status_code == 200
    daemon.sync_filters_to_backend.assert_called_once()


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
    sensor = resp.json()["sensors"]["scope-0"]
    assert sensor["telescope_connected"] is False


def test_status_with_processors(client, mock_daemon):
    p = MagicMock()
    p.name = "plate_solver"
    mock_daemon.processor_registry.processors = [p]
    resp = client.get("/api/status")
    assert "plate_solver" in resp.json()["active_processors"]


def test_status_scheduled_autofocus(client, mock_daemon):
    mock_daemon.settings.scheduled_autofocus_enabled = True
    mock_daemon.settings.last_autofocus_timestamp = int(__import__("time").time()) - 1800
    mock_daemon.task_dispatcher.autofocus_manager.get_next_autofocus_minutes.return_value = 30
    resp = client.get("/api/status")
    sensor = resp.json()["sensors"]["scope-0"]
    assert sensor["next_autofocus_minutes"] is not None


def test_status_scheduled_autofocus_never_run(client, mock_daemon):
    mock_daemon.settings.scheduled_autofocus_enabled = True
    mock_daemon.settings.last_autofocus_timestamp = None
    mock_daemon.task_dispatcher.autofocus_manager.get_next_autofocus_minutes.return_value = 0
    resp = client.get("/api/status")
    sensor = resp.json()["sensors"]["scope-0"]
    assert sensor["next_autofocus_minutes"] == 0


# ---------------------------------------------------------------------------
# POST config with int/float adapter settings validation
# ---------------------------------------------------------------------------


def test_post_config_validates_int_field(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (True, None)
    as_ = {"nina_api_path": "http://x:1888/v2/api", "binning_x": "not_int"}
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok",
            "sensors": [_sensor_config(adapter_settings=as_)],
        },
    )
    assert resp.status_code == 400
    assert "integer" in resp.json()["error"]


def test_post_config_validates_int_range(client, mock_daemon):
    mock_daemon.reload_configuration.return_value = (True, None)
    as_ = {"nina_api_path": "http://x:1888/v2/api", "binning_x": 99}
    resp = client.post(
        "/api/config",
        json={
            "personal_access_token": "tok",
            "sensors": [_sensor_config(adapter_settings=as_)],
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Config sensor CRUD
# ---------------------------------------------------------------------------


def test_list_sensor_configs(client, mock_daemon):
    mock_daemon.settings.sensors[0].model_dump = MagicMock(return_value={"id": "scope-0", "type": "telescope"})
    resp = client.get("/api/config/sensors")
    assert resp.status_code == 200
    assert len(resp.json()["sensors"]) == 1


def test_add_sensor_config(client, mock_daemon):
    resp = client.post(
        "/api/config/sensors",
        json={"id": "scope-1", "type": "telescope", "adapter": "nina"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["sensor"]["id"] == "scope-1"
    assert len(mock_daemon.settings.sensors) == 2
    mock_daemon.settings.save.assert_called()


def test_add_sensor_config_duplicate(client, mock_daemon):
    resp = client.post(
        "/api/config/sensors",
        json={"id": "scope-0", "type": "telescope", "adapter": "nina"},
    )
    assert resp.status_code == 409


def test_add_sensor_config_missing_id(client):
    resp = client.post("/api/config/sensors", json={"type": "telescope"})
    assert resp.status_code == 400


def test_add_sensor_config_invalid_id(client):
    resp = client.post(
        "/api/config/sensors",
        json={"id": "bad id!", "type": "telescope", "adapter": "nina"},
    )
    assert resp.status_code == 400


def test_remove_sensor_config(client, mock_daemon):
    from citrasense.settings.citrasense_settings import SensorConfig

    mock_daemon.settings.sensors.append(SensorConfig(id="scope-1", type="telescope", adapter="nina"))
    resp = client.delete("/api/config/sensors/scope-1")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(mock_daemon.settings.sensors) == 1
    mock_daemon.settings.save.assert_called()


def test_remove_sensor_config_not_found(client):
    resp = client.delete("/api/config/sensors/nonexistent")
    assert resp.status_code == 404


def test_remove_last_sensor_blocked(client, mock_daemon):
    assert len(mock_daemon.settings.sensors) == 1
    resp = client.delete("/api/config/sensors/scope-0")
    assert resp.status_code == 400
    assert "last sensor" in resp.json()["error"].lower()
