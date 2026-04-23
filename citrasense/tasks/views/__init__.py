"""Per-modality task views — typed accessors for sensor-specific Task fields."""

from citrasense.tasks.views.radar_task_view import RadarTaskView
from citrasense.tasks.views.rf_task_view import RfTaskView
from citrasense.tasks.views.telescope_task_view import TelescopeTaskView

__all__ = [
    "RadarTaskView",
    "RfTaskView",
    "TelescopeTaskView",
]
