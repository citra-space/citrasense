"""Integration-style test: daemon init with 2+ telescope SensorConfig entries.

Verifies that _initialize_telescope correctly:
- Iterates all telescope sensors (not just the first)
- Creates a SensorRuntime per telescope
- Registers each runtime with the TaskDispatcher
- Restores task metadata only once (site-level)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from citrasense.citrasense_daemon import CitraSenseDaemon


def _make_telescope_sensor(sensor_id: str):
    s = MagicMock()
    s.sensor_id = sensor_id
    s.sensor_type = "telescope"
    s.adapter = MagicMock()
    s.adapter.connect.return_value = True
    s.adapter.is_mount_homed.return_value = True
    s.adapter.supports_direct_camera_control.return_value = False
    s.adapter.get_filter_config.return_value = {}
    s.citra_record = None
    return s


def _make_daemon(sensors):
    daemon = CitraSenseDaemon.__new__(CitraSenseDaemon)

    daemon.settings = MagicMock()
    configs = []
    for s in sensors:
        sc = MagicMock()
        sc.id = s.sensor_id
        sc.citra_sensor_id = f"api-{s.sensor_id}"
        configs.append(sc)
    daemon.settings.sensors = configs
    daemon.settings.get_sensor_config = lambda sid: next((c for c in configs if c.id == sid), None)

    daemon.api_client = MagicMock()
    daemon.api_client.does_api_server_accept_key.return_value = True

    def _get_telescope(api_id):
        return {
            "id": api_id,
            "groundStationId": "gs-1",
            "maxSlewRate": 3.0,
            "name": f"Scope {api_id}",
        }

    daemon.api_client.get_telescope.side_effect = _get_telescope
    daemon.api_client.get_ground_station.return_value = {"id": "gs-1", "name": "TestGS"}

    sm = MagicMock()
    sm.iter_by_type.return_value = iter(sensors)
    sm.__iter__ = lambda self: iter(sensors)
    daemon.sensor_manager = sm

    daemon.location_service = MagicMock()
    daemon.time_monitor = MagicMock()
    daemon.web_server = MagicMock()
    daemon.elset_cache = MagicMock()
    daemon.apass_catalog = MagicMock()
    daemon.processor_registry = MagicMock()
    daemon.task_dispatcher = None
    daemon.safety_monitor = MagicMock()
    daemon.ground_station = None
    daemon.latest_annotated_image_path = None
    daemon.preview_bus = MagicMock()
    daemon.task_index = MagicMock()
    daemon.sensor_bus = MagicMock()
    daemon._retention_timer = None
    daemon._stop_requested = False

    return daemon


class TestMultiSensorDaemonInit:
    def test_two_telescopes_each_get_runtime(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        daemon = _make_daemon([scope_a, scope_b])

        with (
            patch("citrasense.citrasense_daemon.SensorRuntime") as MockRT,
            patch("citrasense.citrasense_daemon.TaskDispatcher") as MockTD,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_td = MagicMock()
            mock_td.task_dict = {}
            mock_td.imaging_tasks = {}
            mock_td.processing_tasks = {}
            mock_td.uploading_tasks = {}
            MockTD.return_value = mock_td

            rt_a, rt_b = MagicMock(), MagicMock()
            MockRT.side_effect = [rt_a, rt_b]

            success, error = daemon._initialize_telescope()

        assert success is True
        assert error is None
        assert MockRT.call_count == 2
        assert mock_td.register_runtime.call_count == 2
        mock_td.register_runtime.assert_any_call(rt_a)
        mock_td.register_runtime.assert_any_call(rt_b)

    def test_both_telescopes_get_citra_record(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        daemon = _make_daemon([scope_a, scope_b])

        with (
            patch("citrasense.citrasense_daemon.SensorRuntime"),
            patch("citrasense.citrasense_daemon.TaskDispatcher") as MockTD,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_td = MagicMock()
            mock_td.task_dict = {}
            mock_td.imaging_tasks = {}
            mock_td.processing_tasks = {}
            mock_td.uploading_tasks = {}
            MockTD.return_value = mock_td

            daemon._initialize_telescope()

        assert scope_a.citra_record is not None
        assert scope_b.citra_record is not None
        assert scope_a.citra_record["id"] == "api-scope-a"
        assert scope_b.citra_record["id"] == "api-scope-b"

    def test_second_telescope_failure_aborts(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        scope_b.adapter.connect.return_value = False
        daemon = _make_daemon([scope_a, scope_b])

        with (
            patch("citrasense.citrasense_daemon.SensorRuntime"),
            patch("citrasense.citrasense_daemon.TaskDispatcher") as MockTD,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_td = MagicMock()
            mock_td.task_dict = {}
            mock_td.imaging_tasks = {}
            mock_td.processing_tasks = {}
            mock_td.uploading_tasks = {}
            MockTD.return_value = mock_td

            success, error = daemon._initialize_telescope()

        assert success is False
        assert "Failed to connect" in error

    def test_task_dict_restored_once_for_site(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        daemon = _make_daemon([scope_a, scope_b])

        with (
            patch("citrasense.citrasense_daemon.SensorRuntime"),
            patch("citrasense.citrasense_daemon.TaskDispatcher") as MockTD,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_td = MagicMock()
            mock_td.task_dict = {}
            mock_td.imaging_tasks = {}
            mock_td.processing_tasks = {}
            mock_td.uploading_tasks = {}
            MockTD.return_value = mock_td

            old_tasks = {"t1": MagicMock(), "t2": MagicMock()}
            daemon._initialize_telescope(old_task_dict=old_tasks)

        assert "t1" in mock_td.task_dict
        assert "t2" in mock_td.task_dict

    def test_ground_station_set_from_first_telescope_only(self):
        scope_a = _make_telescope_sensor("scope-a")
        scope_b = _make_telescope_sensor("scope-b")
        daemon = _make_daemon([scope_a, scope_b])

        with (
            patch("citrasense.citrasense_daemon.SensorRuntime"),
            patch("citrasense.citrasense_daemon.TaskDispatcher") as MockTD,
            patch.object(daemon, "_initialize_safety_monitor"),
            patch.object(daemon, "save_filter_config"),
            patch.object(daemon, "sync_filters_to_backend"),
        ):
            mock_td = MagicMock()
            mock_td.task_dict = {}
            mock_td.imaging_tasks = {}
            mock_td.processing_tasks = {}
            mock_td.uploading_tasks = {}
            MockTD.return_value = mock_td

            daemon._initialize_telescope()

        assert daemon.ground_station is not None
        assert daemon.ground_station["id"] == "gs-1"
        daemon.location_service.set_ground_station.assert_called_once()
