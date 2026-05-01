"""Allsky-camera runtime builder.

Allsky sensors have no Citra API record and no ground-station
lookup.  The capture loop starts inside
:meth:`SensorRuntime._ensure_queues_started` once the connect
worker flips this runtime to ``connected``, so
:meth:`connect_post_wiring` is a no-op today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from citrasense.sensors.runtime_builder import BuildContext
from citrasense.sensors.sensor_runtime import SensorRuntime

if TYPE_CHECKING:
    from citrasense.sensors.abstract_sensor import AbstractSensor


class AllskyRuntimeBuilder:
    """Build an allsky :class:`SensorRuntime` in ``init_state="pending"``."""

    def __init__(self, ctx: BuildContext) -> None:
        self.ctx = ctx

    def build(self, sensor: AbstractSensor) -> SensorRuntime:
        from citrasense.sensors.allsky.allsky_camera_sensor import AllskyCameraSensor

        ctx = self.ctx
        assert isinstance(sensor, AllskyCameraSensor)

        ctx.logger.info("Building runtime for allsky sensor %s", sensor.sensor_id)

        sensor.preview_bus = ctx.preview_bus

        runtime = SensorRuntime(
            sensor,
            logger=ctx.logger,
            settings=ctx.settings,
            api_client=ctx.api_client,
            hardware_adapter=None,
            ground_station=ctx.ground_station,
            preview_bus=ctx.preview_bus,
            task_index=ctx.task_index,
            safety_monitor=ctx.safety_monitor,
            sensor_bus=ctx.sensor_bus,
        )

        if ctx.init_state_callback_factory is not None:
            runtime.on_init_state_change = ctx.init_state_callback_factory(sensor.sensor_id)

        ctx.task_dispatcher.register_runtime(runtime)
        ctx.logger.info("Allsky sensor %s registered (init_state=pending)", sensor.sensor_id)
        return runtime

    def connect_post_wiring(self, runtime: SensorRuntime) -> None:
        """No post-connect wiring required for allsky sensors today."""
        return None
