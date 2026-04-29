"""Radar sensor package.

Home of :class:`PassiveRadarSensor` — the first streaming sensor — and the
:class:`DetectionSource` protocol that isolates its transport (NATS today,
potentially HTTP/SSE in the future) from the rest of the class.

All NATS imports are confined to :mod:`citrasense.sensors.radar.nats_detection_source`;
the sensor itself depends only on :class:`DetectionSource`.
"""
