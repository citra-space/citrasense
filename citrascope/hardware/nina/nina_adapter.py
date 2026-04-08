import base64
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import requests

from citrascope.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
    SettingSchemaEntry,
)
from citrascope.hardware.nina.nina_event_listener import NinaEventListener, derive_ws_url


class NinaAdvancedHttpAdapter(AbstractAstroHardwareAdapter):
    """HTTP adapter for controlling astronomical equipment through N.I.N.A.
    (Nighttime Imaging 'N' Astronomy) Advanced API.
    https://bump.sh/christian-photo/doc/advanced-api/"""

    DEFAULT_FOCUS_POSITION = 9000

    # API endpoint paths
    CAM_URL = "/equipment/camera/"
    FILTERWHEEL_URL = "/equipment/filterwheel/"
    FOCUSER_URL = "/equipment/focuser/"
    MOUNT_URL = "/equipment/mount/"
    SAFETYMON_URL = "/equipment/safetymonitor/"
    SEQUENCE_URL = "/sequence/"

    # HTTP request timeouts (seconds) — tiered by expected response latency
    HEALTH_CHECK_TIMEOUT = 2
    CONNECT_TIMEOUT = 5
    INFO_QUERY_TIMEOUT = 10
    COMMAND_TIMEOUT = 30

    # Hardware operation timeouts (seconds) — waiting for physical movement to complete
    HARDWARE_MOVE_TIMEOUT = 60
    MOUNT_PARK_TIMEOUT = 120  # park/unpark can require a full-sky slew
    AUTOFOCUS_TIMEOUT = 60 * 15  # 15 minutes
    SEQUENCE_TIMEOUT_MINUTES = 60

    # Polling intervals (seconds)
    SLEW_POLL_INTERVAL = 2
    FOCUSER_POLL_INTERVAL = 5

    def __init__(self, logger: logging.Logger, images_dir: Path, **kwargs):
        super().__init__(images_dir=images_dir, **kwargs)
        self.logger: logging.Logger = logger
        self.nina_api_path = kwargs.get("nina_api_path", "http://nina:1888/v2/api")

        self.binning_x = kwargs.get("binning_x", 1)
        self.binning_y = kwargs.get("binning_y", 1)
        self._event_listener: NinaEventListener | None = None

    @classmethod
    def get_settings_schema(cls, **kwargs) -> list[SettingSchemaEntry]:
        """
        Return a schema describing configurable settings for the NINA Advanced HTTP adapter.
        """
        return [
            {
                "name": "nina_api_path",
                "friendly_name": "N.I.N.A. API URL",
                "type": "str",
                "default": "http://nina:1888/v2/api",
                "description": "Base URL for the NINA Advanced HTTP API",
                "required": True,
                "placeholder": "http://localhost:1888/v2/api",
                "pattern": r"^https?://.*",
                "group": "Connection",
            },
            {
                "name": "binning_x",
                "friendly_name": "Binning X",
                "type": "int",
                "default": 1,
                "description": "Horizontal pixel binning for observations (1=no binning, 2=2x2, etc.)",
                "required": False,
                "placeholder": "1",
                "min": 1,
                "max": 4,
                "group": "Imaging",
            },
            {
                "name": "binning_y",
                "friendly_name": "Binning Y",
                "type": "int",
                "default": 1,
                "description": "Vertical pixel binning for observations (1=no binning, 2=2x2, etc.)",
                "required": False,
                "placeholder": "1",
                "min": 1,
                "max": 4,
                "group": "Imaging",
            },
        ]

    def do_autofocus(
        self,
        target_ra: float | None = None,
        target_dec: float | None = None,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """Perform autofocus routine for all enabled filters.

        Slews telescope to a bright reference star and runs autofocus
        for each enabled filter in the filter map, updating focus positions.

        Note: cancel_event is accepted for interface compatibility but not
        checked — NINA runs autofocus internally and there is no REST API
        endpoint to cancel a NINA autofocus in progress.

        Args:
            target_ra: RA of the slew target in degrees (J2000), or None to
                focus at the current telescope position (no slew).
            target_dec: Dec of the slew target in degrees (J2000), or None to
                focus at the current telescope position (no slew).
            on_progress: Optional callback(str) to report progress updates
            cancel_event: Unused (NINA manages its own autofocus lifecycle)

        Raises:
            RuntimeError: If no filters discovered or no enabled filters
            ValueError: If only one of target_ra/target_dec is None
        """

        def report(msg):
            if on_progress:
                on_progress(msg)

        if (target_ra is None) != (target_dec is None):
            raise ValueError(
                f"target_ra and target_dec must both be set or both be None, " f"got ra={target_ra}, dec={target_dec}"
            )

        if not self.filter_map:
            raise RuntimeError("No filters discovered. Cannot perform autofocus.")

        enabled_filters = {fid: fdata for fid, fdata in self.filter_map.items() if fdata.get("enabled", True)}
        if not enabled_filters:
            raise RuntimeError("No enabled filters. Cannot perform autofocus.")

        total = len(enabled_filters)
        self.logger.info(f"Performing autofocus routine on {total} enabled filter(s) ...")

        if target_ra is not None and target_dec is not None:
            report("Slewing to target...")
            self.logger.info(f"Slewing to autofocus target (RA={target_ra:.4f}, Dec={target_dec:.4f}) ...")
            try:
                response = requests.get(
                    f"{self.nina_api_path}{self.MOUNT_URL}slew?ra={target_ra}&dec={target_dec}",
                    timeout=self.COMMAND_TIMEOUT,
                )
                response.raise_for_status()
                mount_status = response.json()
                if not mount_status.get("Success"):
                    raise RuntimeError(f"Mount slew rejected by NINA: {mount_status.get('Error')}")
                self.logger.info(f"Mount {mount_status['Response']}")
            except requests.Timeout as e:
                raise RuntimeError("Mount slew request timed out") from e
            except requests.RequestException as e:
                raise RuntimeError(f"Mount slew failed: {e}") from e

            time.sleep(self.SLEW_POLL_INTERVAL)
            while self.telescope_is_moving():
                self.logger.info("Waiting for mount to finish slewing...")
                time.sleep(self.SLEW_POLL_INTERVAL)
        else:
            self.logger.info("Autofocus at current position (no slew)")

        for idx, (id, filter) in enumerate(enabled_filters.items(), 1):
            name = filter["name"]
            report(f"Filter {idx}/{total}: {name} — focusing...")
            self.logger.info(f"Focusing Filter ID: {id}, Name: {name}")
            existing_focus = filter.get("focus_position", self.DEFAULT_FOCUS_POSITION)

            def af_point_progress(position: int, hfr: float, _idx=idx, _total=total, _name=name):
                report(f"Filter {_idx}/{_total}: {_name} — pos {position}, HFR {hfr:.2f}")

            focus_value = self._auto_focus_one_filter(id, name, existing_focus, on_af_point=af_point_progress)
            self.filter_map[id]["focus_position"] = focus_value
            report(f"Filter {idx}/{total}: {name} — done (position {focus_value})")

    def _auto_focus_one_filter(
        self,
        filter_id: int,
        filter_name: str,
        existing_focus_position: int,
        on_af_point: Callable[[int, float], None] | None = None,
    ) -> int:
        assert self._event_listener is not None

        current_filter = self._get_current_filter_id()
        if current_filter == filter_id:
            self.logger.info(f"Already on filter {filter_id} ({filter_name}), skipping change")
        else:
            self._event_listener.filter_changed.clear()
            resp = requests.get(
                self.nina_api_path + self.FILTERWHEEL_URL + "change-filter?filterId=" + str(filter_id),
                timeout=self.COMMAND_TIMEOUT,
            ).json()
            if not resp.get("Success"):
                raise RuntimeError(f"Filter change to {filter_id} rejected by NINA: {resp.get('Error')}")
            if not self._event_listener.filter_changed.wait(timeout=self.HARDWARE_MOVE_TIMEOUT):
                raise RuntimeError(
                    f"Filterwheel failed to change to filter {filter_id} within {self.HARDWARE_MOVE_TIMEOUT}s"
                )
            self.logger.info(f"Filter changed to ID {filter_id}")

        self.logger.info("Moving focus to autofocus starting position ...")
        starting_focus_position = (
            existing_focus_position if existing_focus_position is not None else self.DEFAULT_FOCUS_POSITION
        )
        move_resp = requests.get(
            self.nina_api_path + self.FOCUSER_URL + "move?position=" + str(starting_focus_position),
            timeout=self.COMMAND_TIMEOUT,
        ).json()
        if not move_resp.get("Success"):
            raise RuntimeError(f"Focuser move rejected by NINA: {move_resp.get('Error')}")
        deadline = time.time() + self.HARDWARE_MOVE_TIMEOUT
        while True:
            focuser_status = requests.get(
                self.nina_api_path + self.FOCUSER_URL + "info", timeout=self.INFO_QUERY_TIMEOUT
            ).json()
            if not focuser_status.get("Success"):
                raise RuntimeError(f"Focuser info query failed: {focuser_status.get('Error')}")
            if int(focuser_status["Response"]["Position"]) == starting_focus_position:
                break
            if time.time() > deadline:
                raise RuntimeError(
                    f"Focuser failed to reach position {starting_focus_position} within {self.HARDWARE_MOVE_TIMEOUT}s"
                )
            self.logger.info("Waiting for focuser to reach starting position ...")
            time.sleep(self.FOCUSER_POLL_INTERVAL)

        self.logger.info("Starting autofocus ...")

        self._event_listener.autofocus_finished.clear()
        self._event_listener.autofocus_error.clear()
        prev_af_callback = self._event_listener.on_af_point
        self._event_listener.on_af_point = on_af_point

        af_resp = requests.get(
            self.nina_api_path + self.FOCUSER_URL + "auto-focus", timeout=self.COMMAND_TIMEOUT
        ).json()
        if not af_resp.get("Success"):
            raise RuntimeError(f"Autofocus trigger rejected by NINA: {af_resp.get('Error')}")
        self.logger.info(f"Focuser {af_resp['Response']}")

        try:
            deadline = time.time() + self.AUTOFOCUS_TIMEOUT
            while not self._event_listener.autofocus_finished.is_set():
                if self._event_listener.autofocus_error.is_set():
                    self.logger.warning(f"Autofocus error reported via WebSocket for filter {filter_name}")
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.logger.warning(f"Autofocus timed out after {self.AUTOFOCUS_TIMEOUT}s for filter {filter_name}")
                    break
                self._event_listener.autofocus_finished.wait(timeout=min(remaining, 1.0))
        finally:
            self._event_listener.on_af_point = prev_af_callback

        if self._event_listener.autofocus_finished.is_set():
            last_af = requests.get(
                self.nina_api_path + self.FOCUSER_URL + "last-af", timeout=self.INFO_QUERY_TIMEOUT
            ).json()
            if last_af.get("Success"):
                resp = last_af["Response"]
                position = resp["CalculatedFocusPoint"]["Position"]
                hfr = resp["CalculatedFocusPoint"]["Value"]
                self.logger.info(f"Autofocus complete for filter {filter_name}: Position {position}, HFR {hfr}")
                return position
            self.logger.warning(f"last-af fetch after AUTOFOCUS-FINISHED failed: {last_af.get('Error')}")

        self.logger.warning(f"Preserving existing focus position {existing_focus_position} for {filter_name}")
        return existing_focus_position

    def _find_task_images(self, task_id: str, expected_count: int) -> list[int]:
        """Query NINA /image-history and return indices of images matching task_id."""
        resp = requests.get(f"{self.nina_api_path}/image-history?all=true").json()
        if not resp.get("Success"):
            self.logger.error(f"Failed to get image history: {resp.get('Error')}")
            raise RuntimeError("Failed to get images list from NINA")
        all_images = resp["Response"]
        search_window = expected_count + 10
        matches = []
        for i in range(max(0, len(all_images) - search_window), len(all_images)):
            if task_id in all_images[i].get("Filename", ""):
                matches.append(i)
        return matches

    def _do_point_telescope(self, ra: float, dec: float):
        self.logger.info(f"Slewing to RA: {ra}, Dec: {dec}")
        try:
            response = requests.get(
                f"{self.nina_api_path}{self.MOUNT_URL}slew?ra={ra}&dec={dec}", timeout=self.COMMAND_TIMEOUT
            )
            response.raise_for_status()
            slew_response = response.json()

            if slew_response.get("Success"):
                self.logger.info(f"Mount slew initiated: {slew_response['Response']}")
                return True
            else:
                self.logger.error(f"Failed to slew mount: {slew_response.get('Error')}")
                return False
        except requests.Timeout:
            self.logger.error("Mount slew request timed out")
            return False
        except requests.RequestException as e:
            self.logger.error(f"Mount slew request failed: {e}")
            return False

    def connect(self) -> bool:
        try:
            # start connection to all equipments
            self.logger.info("Connecting camera ...")
            cam_status = requests.get(
                self.nina_api_path + self.CAM_URL + "connect", timeout=self.CONNECT_TIMEOUT
            ).json()
            if not cam_status["Success"]:
                self.logger.error(f"Failed to connect camera: {cam_status.get('Error')}")
                return False
            self.logger.info("Camera Connected!")

            self.logger.info("Starting camera cooling ...")
            cool_status = requests.get(self.nina_api_path + self.CAM_URL + "cool", timeout=self.CONNECT_TIMEOUT).json()
            if not cool_status["Success"]:
                self.logger.warning(f"Failed to start camera cooling: {cool_status.get('Error')}")
            else:
                self.logger.info("Cooler started!")

            self.logger.info("Connecting filterwheel ...")
            filterwheel_status = requests.get(
                self.nina_api_path + self.FILTERWHEEL_URL + "connect", timeout=self.CONNECT_TIMEOUT
            ).json()
            if not filterwheel_status["Success"]:
                self.logger.warning(f"Failed to connect filterwheel: {filterwheel_status.get('Error')}")
            else:
                self.logger.info("Filterwheel Connected!")

            self.logger.info("Connecting focuser ...")
            focuser_status = requests.get(
                self.nina_api_path + self.FOCUSER_URL + "connect", timeout=self.CONNECT_TIMEOUT
            ).json()
            if not focuser_status["Success"]:
                self.logger.warning(f"Failed to connect focuser: {focuser_status.get('Error')}")
            else:
                self.logger.info("Focuser Connected!")

            self.logger.info("Connecting mount ...")
            mount_status = requests.get(
                self.nina_api_path + self.MOUNT_URL + "connect", timeout=self.CONNECT_TIMEOUT
            ).json()
            if not mount_status["Success"]:
                self.logger.error(f"Failed to connect mount: {mount_status.get('Error')}")
                return False
            self.logger.info("Mount Connected!")

            self.logger.info("Unparking mount ...")
            mount_status = requests.get(
                self.nina_api_path + self.MOUNT_URL + "unpark", timeout=self.CONNECT_TIMEOUT
            ).json()
            if not mount_status["Success"]:
                self.logger.error(f"Failed to unpark mount: {mount_status.get('Error')}")
                return False
            self.logger.info("Mount Unparked!")

            # Discover available filters (focus positions loaded from saved settings)
            self.discover_filters()

            # Start WebSocket event listener for reactive hardware monitoring
            ws_url = derive_ws_url(self.nina_api_path)
            self._event_listener = NinaEventListener(ws_url, self.logger)
            self._event_listener.start()

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to NINA Advanced API: {e}")
            return False

    def _get_current_filter_id(self) -> int | None:
        """Query NINA for the currently selected filter wheel position.

        Returns the filter Id (0-indexed int) or None if the query fails.
        """
        try:
            resp = requests.get(
                self.nina_api_path + self.FILTERWHEEL_URL + "info", timeout=self.INFO_QUERY_TIMEOUT
            ).json()
            if not resp.get("Success"):
                return None
            selected = resp.get("Response", {}).get("SelectedFilter")
            if selected is not None and "Id" in selected:
                return int(selected["Id"])
            return None
        except Exception:
            return None

    def discover_filters(self):
        self.logger.info("Discovering filters ...")
        filterwheel_info = requests.get(
            self.nina_api_path + self.FILTERWHEEL_URL + "info", timeout=self.CONNECT_TIMEOUT
        ).json()
        if not filterwheel_info.get("Success"):
            self.logger.error(f"Failed to get filterwheel info: {filterwheel_info.get('Error')}")
            raise RuntimeError("Failed to get filterwheel info")

        filters = filterwheel_info["Response"]["AvailableFilters"]
        for filter in filters:
            filter_id = filter["Id"]
            filter_name = filter["Name"]
            # Use existing focus position and enabled state if filter already in map
            if filter_id in self.filter_map:
                focus_position = self.filter_map[filter_id].get("focus_position", self.DEFAULT_FOCUS_POSITION)
                enabled = self.filter_map[filter_id].get("enabled", True)
                self.logger.info(
                    f"Discovered filter: {filter_name} with ID: {filter_id}, "
                    f"using saved focus position: {focus_position}, enabled: {enabled}"
                )
            else:
                focus_position = self.DEFAULT_FOCUS_POSITION
                enabled = True  # Default new filters to enabled
                self.logger.info(
                    f"Discovered new filter: {filter_name} with ID: {filter_id}, "
                    f"using default focus position: {focus_position}"
                )

            self.filter_map[filter_id] = {"name": filter_name, "focus_position": focus_position, "enabled": enabled}

    def disconnect(self):
        if self._event_listener:
            try:
                self._event_listener.stop()
            finally:
                self._event_listener = None

    def get_filter_position(self) -> int | None:
        """Get the current filter wheel position from NINA."""
        return self._get_current_filter_id()

    def supports_autofocus(self) -> bool:
        """Indicates that NINA adapter supports autofocus."""
        return True

    def supports_filter_management(self) -> bool:
        """Indicates that NINA adapter supports filter/focus management."""
        return True

    @property
    def supports_hardware_safety_monitor(self) -> bool:
        return True

    def query_hardware_safety(self) -> bool | None:
        """Query NINA's safety monitor device for environmental safety status."""
        try:
            resp = requests.get(
                f"{self.nina_api_path}{self.SAFETYMON_URL}info",
                timeout=self.HEALTH_CHECK_TIMEOUT,
            ).json()
            if not resp.get("Success"):
                return None
            response = resp.get("Response", {})
            if not response.get("Connected"):
                return None
            is_safe = response.get("IsSafe")
            if isinstance(is_safe, bool):
                return is_safe
            return None
        except Exception:
            return None

    def is_telescope_connected(self) -> bool:
        """Check if telescope is connected and responsive."""
        try:
            mount_info = requests.get(
                f"{self.nina_api_path}{self.MOUNT_URL}info", timeout=self.HEALTH_CHECK_TIMEOUT
            ).json()
            return mount_info.get("Success", False) and mount_info.get("Response", {}).get("Connected", False)
        except Exception:
            return False

    def is_camera_connected(self) -> bool:
        """Check if camera is connected and responsive."""
        try:
            cam_info = requests.get(f"{self.nina_api_path}{self.CAM_URL}info", timeout=self.HEALTH_CHECK_TIMEOUT).json()
            return cam_info.get("Success", False) and cam_info.get("Response", {}).get("Connected", False)
        except Exception:
            return False

    def list_devices(self) -> list[str]:
        return []

    def select_telescope(self, device_name: str) -> bool:
        return True

    def get_telescope_direction(self) -> tuple[float, float]:
        mount_info = requests.get(self.nina_api_path + self.MOUNT_URL + "info", timeout=self.INFO_QUERY_TIMEOUT).json()
        if mount_info.get("Success"):
            ra_degrees = mount_info["Response"]["Coordinates"]["RADegrees"]
            dec_degrees = mount_info["Response"]["Coordinates"]["Dec"]
            return (ra_degrees, dec_degrees)
        else:
            self.logger.error(f"Failed to get telescope direction: {mount_info.get('Error')}")
            raise RuntimeError(f"Failed to get mount info: {mount_info.get('Error')}")

    def telescope_is_moving(self) -> bool:
        mount_info = requests.get(self.nina_api_path + self.MOUNT_URL + "info", timeout=self.INFO_QUERY_TIMEOUT).json()
        if mount_info.get("Success"):
            return mount_info["Response"]["Slewing"]
        else:
            self.logger.error(f"Failed to get telescope status: {mount_info.get('Error')}")
            return False

    def park_mount(self) -> bool:
        try:
            resp = requests.get(f"{self.nina_api_path}{self.MOUNT_URL}park", timeout=self.MOUNT_PARK_TIMEOUT).json()
            if resp.get("Success"):
                self.logger.info("Mount parked via NINA")
                return True
            self.logger.error(f"Failed to park mount: {resp.get('Error')}")
            return False
        except Exception as e:
            self.logger.error(f"Error parking mount: {e}")
            return False

    def unpark_mount(self) -> bool:
        try:
            resp = requests.get(f"{self.nina_api_path}{self.MOUNT_URL}unpark", timeout=self.MOUNT_PARK_TIMEOUT).json()
            if resp.get("Success"):
                self.logger.info("Mount unparked via NINA")
                return True
            self.logger.error(f"Failed to unpark mount: {resp.get('Error')}")
            return False
        except Exception as e:
            self.logger.error(f"Error unparking mount: {e}")
            return False

    def supports_park(self) -> bool:
        return True

    def get_camera_info(self) -> dict | None:
        """Query NINA camera info endpoint for sensor specs."""
        try:
            resp = requests.get(f"{self.nina_api_path}{self.CAM_URL}info", timeout=self.HEALTH_CHECK_TIMEOUT).json()
            if not resp.get("Success"):
                return None
            r = resp.get("Response", {})
            if not r.get("Connected"):
                return None
            info: dict = {}
            if r.get("XSize"):
                info["width"] = int(r["XSize"])
            if r.get("YSize"):
                info["height"] = int(r["YSize"])
            if r.get("PixelSizeX"):
                info["pixel_size_um"] = float(r["PixelSizeX"])
            if r.get("Name"):
                info["model"] = r["Name"]
            return info if info else None
        except Exception:
            return None

    def select_camera(self, device_name: str) -> bool:
        return True

    def take_image(self, task_id: str, exposure_duration_seconds=1) -> str:
        raise NotImplementedError

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        pass  # TODO: make real

    def get_tracking_rate(self) -> tuple[float, float]:
        return (0, 0)  # TODO: make real

    @property
    def sequence_provides_tracking(self) -> bool:
        return True

    def _get_sequence_template(self) -> str:
        """Load the sequence template as a string for placeholder replacement."""
        template_path = os.path.join(os.path.dirname(__file__), "survey_template.json")
        with open(template_path) as f:
            return f.read()

    def get_observation_strategy(self) -> ObservationStrategy:
        return ObservationStrategy.SEQUENCE_TO_CONTROLLER

    def _find_by_id(self, data, target_id):
        """Recursively search for an item with a specific $id in the JSON structure.

        Args:
            data: The JSON data structure to search (dict, list, or primitive)
            target_id: The $id value to search for (as string)

        Returns:
            The item with the matching $id, or None if not found
        """
        if isinstance(data, dict):
            if data.get("$id") == target_id:
                return data
            for value in data.values():
                result = self._find_by_id(value, target_id)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_by_id(item, target_id)
                if result is not None:
                    return result
        return None

    def _get_max_id(self, data):
        """Recursively find the maximum $id value in the JSON structure.

        Args:
            data: The JSON data structure to search (dict, list, or primitive)

        Returns:
            The maximum numeric $id value found, or 0 if none found
        """
        max_id = 0
        if isinstance(data, dict):
            if "$id" in data:
                try:
                    max_id = max(max_id, int(data["$id"]))
                except (ValueError, TypeError):
                    pass
            for value in data.values():
                max_id = max(max_id, self._get_max_id(value))
        elif isinstance(data, list):
            for item in data:
                max_id = max(max_id, self._get_max_id(item))
        return max_id

    def _update_all_ids(self, data, id_counter):
        """Recursively update all $id values in a data structure.

        Args:
            data: The JSON data structure to update (dict, list, or primitive)
            id_counter: A list with a single integer element [current_id] that gets incremented

        Returns:
            None (modifies data in place)
        """
        if isinstance(data, dict):
            if "$id" in data:
                data["$id"] = str(id_counter[0])
                id_counter[0] += 1
            for value in data.values():
                self._update_all_ids(value, id_counter)
        elif isinstance(data, list):
            for item in data:
                self._update_all_ids(item, id_counter)

    def perform_observation_sequence(self, task, satellite_data) -> list[str]:
        """Create and execute a NINA sequence for the given satellite.

        Args:
            task: Task object containing id and filter assignment
            satellite_data: Satellite data including TLE information

        Returns:
            list[str]: Paths to the captured images
        """
        assert self._event_listener is not None
        elset = satellite_data["most_recent_elset"]

        # Load template as JSON and replace binning placeholders
        template_str = self._get_sequence_template()
        template_str = template_str.replace("{binning_x}", str(self.binning_x))
        template_str = template_str.replace("{binning_y}", str(self.binning_y))
        sequence_json = json.loads(template_str)

        nina_sequence_name = f"Citra Target: {satellite_data['name']}, Task: {task.id}"

        # Replace basic placeholders (use \r\n for Windows NINA compatibility)
        tle_data = f"{elset['tle'][0]}\r\n{elset['tle'][1]}"
        sequence_json["Name"] = nina_sequence_name

        # Navigate to the TLE container (ID 20 in the template)
        target_container = self._find_by_id(sequence_json, "20")
        if not target_container:
            raise RuntimeError("Could not find TLE container (ID 20) in sequence template")

        target_container["TLEData"] = tle_data
        target_container["Name"] = satellite_data["name"]
        target_container["Target"]["TargetName"] = satellite_data["name"]

        # Find the TLE control item and update it
        tle_items = target_container["Items"]["$values"]
        for item in tle_items:
            if item.get("$type") == "DaleGhent.NINA.PlaneWaveTools.TLE.TLEControl, PlaneWave Tools":
                item["Line1"] = elset["tle"][0]
                item["Line2"] = elset["tle"][1]
                break

        # Find the template triplet (filter/focus/expose) - should be items 1, 2, 3
        # (item 0 is TLE control)
        template_triplet = tle_items[1:4]  # SwitchFilter, MoveFocuserAbsolute, TakeExposure

        # Clear the items list and rebuild with TLE control + triplets for each filter
        new_items = [tle_items[0]]  # Keep TLE control as first item

        # Generate triplet for each discovered filter
        # Find the maximum ID in use and start after it to avoid collisions
        base_id = self._get_max_id(sequence_json) + 1
        self.logger.debug(f"Starting dynamic ID generation at {base_id}")

        id_counter = [base_id]  # Use list so it can be modified in nested function

        # select_filters_for_task raises RuntimeError when allow_no_filter=False and no filter found
        filters_to_use = self.select_filters_for_task(task, allow_no_filter=False)
        assert filters_to_use is not None

        for filter_id, filter_info in filters_to_use.items():
            filter_name = filter_info["name"]
            focus_position = filter_info["focus_position"]

            # Create a deep copy of the triplet and update ALL nested IDs
            filter_triplet = json.loads(json.dumps(template_triplet))
            self._update_all_ids(filter_triplet, id_counter)

            # Update filter switch (first item in triplet)
            filter_triplet[0]["Filter"]["_name"] = filter_name
            filter_triplet[0]["Filter"]["_position"] = filter_id

            # Update focus position (second item in triplet)
            filter_triplet[1]["Position"] = focus_position

            # Exposure settings (third item) are already set from template

            # Add this triplet to the sequence
            new_items.extend(filter_triplet)

            self.logger.debug(f"Added filter {filter_name} (ID: {filter_id}) with focus position {focus_position}")

        # Update the items list
        tle_items.clear()
        tle_items.extend(new_items)

        # Convert back to JSON string
        template_str = json.dumps(sequence_json, indent=2)

        # POST the sequence

        self.logger.info("Posting NINA sequence")
        post_response = requests.post(f"{self.nina_api_path}{self.SEQUENCE_URL}load", json=sequence_json).json()
        if not post_response.get("Success"):
            self.logger.error(f"Failed to post sequence: {post_response.get('Error')}")
            raise RuntimeError("Failed to post NINA sequence")

        self.logger.info("Loaded sequence to NINA, starting sequence...")

        self._event_listener.sequence_finished.clear()
        self._event_listener.sequence_failed.clear()

        # Wait for IMAGE-SAVE WS events to know images are flushed to disk,
        # then query image-history once to get indices for download.
        expected_image_count = len(filters_to_use)
        matched_count = [0]
        images_ready = threading.Event()

        def _on_image_saved(stats: dict) -> None:
            # Called on the single WS listener thread — no concurrent writes to matched_count
            filename = stats.get("Filename", "")
            if task.id in filename:
                matched_count[0] += 1
                self.logger.info(f"IMAGE-SAVE: {filename} ({matched_count[0]}/{expected_image_count})")
                if matched_count[0] >= expected_image_count:
                    images_ready.set()

        prev_callback = self._event_listener.on_image_save
        self._event_listener.on_image_save = _on_image_saved

        try:
            start_response = requests.get(
                f"{self.nina_api_path}{self.SEQUENCE_URL}start?skipValidation=true"
            ).json()  # TODO: try and fix validation issues
            if not start_response.get("Success"):
                self.logger.error(f"Failed to start sequence: {start_response.get('Error')}")
                raise RuntimeError("Failed to start NINA sequence")

            deadline = time.time() + self.SEQUENCE_TIMEOUT_MINUTES * 60
            while not self._event_listener.sequence_finished.is_set():
                if self._event_listener.sequence_failed.is_set():
                    err = self._event_listener.last_sequence_error or {}
                    entity = err.get("Entity", "unknown")
                    error_msg = err.get("Error", "unknown error")
                    self.logger.error(f"NINA sequence entity failed: {entity} — {error_msg}")
                    raise RuntimeError(f"NINA sequence failed: {entity} — {error_msg}")
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.logger.error(
                        f"NINA sequence did not complete within timeout of {self.SEQUENCE_TIMEOUT_MINUTES} minutes"
                    )
                    raise RuntimeError("NINA sequence timeout")
                self._event_listener.sequence_finished.wait(timeout=min(remaining, 2.0))

            self.logger.info("NINA sequence completed, waiting for images to save...")
            images_ready.wait(timeout=self.INFO_QUERY_TIMEOUT)
        finally:
            self._event_listener.on_image_save = prev_callback

        images_to_download = self._find_task_images(task.id, expected_image_count)

        if not images_to_download:
            self.logger.error(
                f"No images matching task {task.id} found. "
                "Ensure NINA is configured to include Sequence Title in image filenames "
                "under Options > Imaging > Image File Pattern."
            )
            raise RuntimeError(f"No matching images found for task {task.id}")

        self.logger.info(f"Downloading {len(images_to_download)} images")

        filepaths = []
        for image_index in images_to_download:
            self.logger.debug("Retrieving image from NINA...")
            image_response = requests.get(
                f"{self.nina_api_path}/image/{image_index}",
                params={"raw_fits": "true"},
            )

            if image_response.status_code != 200:
                self.logger.error(f"Failed to retrieve image: HTTP {image_response.status_code}")
                raise RuntimeError("Failed to retrieve image from NINA")

            image_data = image_response.json()
            if not image_data.get("Success"):
                self.logger.error(f"Failed to get image: {image_data.get('Error')}")
                raise RuntimeError(f"Failed to get image from NINA: {image_data.get('Error')}")

            # Decode base64 FITS data and save to file
            fits_base64 = image_data["Response"]
            fits_bytes = base64.b64decode(fits_base64)

            # Save the FITS file
            filepath = str(self.images_dir / f"citra_task_{task.id}_image_{image_index}.fits")
            filepaths.append(filepath)

            with open(filepath, "wb") as f:
                f.write(fits_bytes)

            self.logger.info(f"Saved FITS image to {filepath}")

        return filepaths
