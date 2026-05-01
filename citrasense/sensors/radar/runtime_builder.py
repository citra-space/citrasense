"""Passive-radar runtime builder.

Radar sensors don't talk to the Citra tasks API — there's no
``citra_record`` to fetch, no ground-station lookup.  The
sensor-side ``citra_antenna_id`` is the only backend identifier and
it's already loaded from the config.  ``connect_post_wiring`` is a
no-op today because there's no live-adapter wiring beyond what
``adapter.connect()`` does itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from citrasense.sensors.runtime_builder import BuildContext
from citrasense.sensors.sensor_runtime import SensorRuntime

if TYPE_CHECKING:
    from citrasense.sensors.abstract_sensor import AbstractSensor


class RadarRuntimeBuilder:
    """Build a passive-radar :class:`SensorRuntime` in ``init_state="pending"``."""

    def __init__(self, ctx: BuildContext) -> None:
        self.ctx = ctx

    def build(self, sensor: AbstractSensor) -> SensorRuntime:
        from citrasense.sensors.radar.passive_radar_sensor import PassiveRadarSensor

        ctx = self.ctx
        assert isinstance(sensor, PassiveRadarSensor)

        ctx.logger.info("Building runtime for radar sensor %s", sensor.sensor_id)

        # Wire toast callback so staleness / error / announce events
        # reach the operator.  Best-effort — a missing web server is
        # fine (headless testing).
        if ctx.web_server:
            sensor.on_toast = ctx.web_server.send_toast
            # Live-broadcast every detection to browser clients via the
            # web loop.  ``send_radar_detection`` is a thread-safe
            # bridge (``run_coroutine_threadsafe``); the NATS asyncio
            # thread never awaits here.
            sensor.on_detection_broadcast = ctx.web_server.send_radar_detection

        runtime = SensorRuntime(
            sensor,
            logger=ctx.logger,
            settings=ctx.settings,
            api_client=ctx.api_client,
            hardware_adapter=None,
            ground_station=ctx.ground_station,
            task_index=ctx.task_index,
            safety_monitor=ctx.safety_monitor,
            sensor_bus=ctx.sensor_bus,
        )

        if ctx.init_state_callback_factory is not None:
            runtime.on_init_state_change = ctx.init_state_callback_factory(sensor.sensor_id)

        ctx.task_dispatcher.register_runtime(runtime)
        ctx.logger.info("Radar sensor %s registered (init_state=pending)", sensor.sensor_id)
        return runtime

    def connect_post_wiring(self, runtime: SensorRuntime) -> None:
        """No post-connect wiring required for radar sensors today."""
        return None
