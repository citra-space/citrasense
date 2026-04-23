import threading
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    type: str
    status: str
    creationEpoch: str
    updateEpoch: str
    taskStart: str
    taskStop: str
    userId: str
    username: str
    satelliteId: str
    satelliteName: str
    telescopeId: str
    telescopeName: str
    groundStationId: str
    groundStationName: str
    sensor_type: str = "telescope"
    sensor_id: str = ""
    assigned_filter_name: str | None = None

    # Local execution state (not from API, never sent to server)
    local_status_msg: str | None = None
    retry_scheduled_time: float | None = None  # Unix timestamp when retry will execute (None if not retrying)
    is_being_executed: bool = False  # True when a worker is actively executing this task

    # Thread safety for status fields (not included in __init__)
    _status_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        sensor_type = data.get("sensorType", "")
        sensor_id = data.get("sensorId", "")
        if not sensor_type:
            if data.get("antennaId"):
                sensor_type = "rf"
                sensor_id = sensor_id or data.get("antennaId", "")
            else:
                sensor_type = "telescope"
                sensor_id = sensor_id or data.get("telescopeId", "")

        return cls(
            id=str(data.get("id", "")),
            type=data.get("type", ""),
            status=str(data.get("status", "")),
            creationEpoch=data.get("creationEpoch", ""),
            updateEpoch=data.get("updateEpoch", ""),
            taskStart=data.get("taskStart", ""),
            taskStop=data.get("taskStop", ""),
            userId=data.get("userId", ""),
            username=data.get("username", ""),
            satelliteId=data.get("satelliteId", ""),
            satelliteName=data.get("satelliteName", ""),
            telescopeId=data.get("telescopeId", ""),
            telescopeName=data.get("telescopeName", ""),
            groundStationId=data.get("groundStationId", ""),
            groundStationName=data.get("groundStationName", ""),
            sensor_type=sensor_type,
            sensor_id=sensor_id,
            assigned_filter_name=data.get("assignedFilterName"),
        )

    def set_status_msg(self, msg: str | None):
        """Thread-safe setter for local_status_msg."""
        with self._status_lock:
            self.local_status_msg = msg

    def get_status_msg(self) -> str | None:
        """Thread-safe getter for local_status_msg."""
        with self._status_lock:
            return self.local_status_msg

    def set_retry_time(self, timestamp: float | None):
        """Thread-safe setter for retry_scheduled_time."""
        with self._status_lock:
            self.retry_scheduled_time = timestamp

    def get_retry_time(self) -> float | None:
        """Thread-safe getter for retry_scheduled_time."""
        with self._status_lock:
            return self.retry_scheduled_time

    def set_executing(self, executing: bool):
        """Thread-safe setter for is_being_executed."""
        with self._status_lock:
            self.is_being_executed = executing

    def get_executing(self) -> bool:
        """Thread-safe getter for is_being_executed."""
        with self._status_lock:
            return self.is_being_executed

    def get_status_info(self) -> tuple[str | None, float | None, bool]:
        """Thread-safe getter for all status fields at once."""
        with self._status_lock:
            return (self.local_status_msg, self.retry_scheduled_time, self.is_being_executed)

    def __repr__(self):
        label = self.satelliteName or self.sensor_id or self.id[:8]
        return f"<Task {self.id[:8]} {self.type} [{self.sensor_type}] '{label}' {self.status}>"
