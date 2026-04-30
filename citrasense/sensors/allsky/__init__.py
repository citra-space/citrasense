"""Allsky camera sensor package.

Home of :class:`AllskyCameraSensor` — a streaming sensor that runs a wide-FOV
camera (USB or Raspberry Pi HQ) in a periodic-capture loop and pushes JPEG
previews to :class:`~citrasense.sensors.preview_bus.PreviewBus` for the web
UI.

No Citra task flow, no upload, no processing pipeline (yet) — those land in
follow-up issues.  The runtime queues still build per the
:class:`~citrasense.sensors.sensor_runtime.SensorRuntime` contract so a
future satellite-detection pipeline can plug in without restructuring.
"""
