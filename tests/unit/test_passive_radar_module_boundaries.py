"""Enforce the acceptance criterion from issue #307 that the sensor
module itself stays transport-agnostic.

``PassiveRadarSensor`` is allowed to *import* a NATS-backed
``DetectionSource`` from its factory (``from_config``) — that's a
single-responsibility concession so one place builds the NATS wiring.
Everywhere else, the sensor must talk to the transport only via the
abstract :class:`DetectionSource` protocol.

This test `grep`s the sensor module's source and asserts every mention
of the substring ``nats`` is one of a small, explicit set of allowed
references (factory import + the module-level docstring).
"""

from __future__ import annotations

import re
from pathlib import Path

import citrasense.sensors.radar.passive_radar_sensor as target_module

_SOURCE_PATH = Path(target_module.__file__)


def _iter_nats_lines() -> list[tuple[int, str]]:
    lines = _SOURCE_PATH.read_text().splitlines()
    return [(idx + 1, line) for idx, line in enumerate(lines) if re.search(r"nats", line, re.I)]


def test_nats_only_appears_inside_known_allowlist():
    allowlist = (
        # The class / factory reference — the ONE seam that picks the
        # NATS transport.  These are the lines that MUST keep working.
        "NatsDetectionSource",
        "nats_detection_source",
        # Docstrings + comments that document the design; cheap to
        # enumerate and they're the obvious place to describe the
        # NATS-backed v1 transport.
        "NATS",
        "nats://",
        "nats_url",
        '"nats"',  # appears in the module docstring describing this rule
    )
    offenders: list[tuple[int, str]] = []
    for lineno, line in _iter_nats_lines():
        if any(token in line for token in allowlist):
            continue
        offenders.append((lineno, line.strip()))
    assert offenders == [], (
        "passive_radar_sensor.py mentions 'nats' outside the factory / docstring"
        f" allowlist — move it to NatsDetectionSource:\n{offenders!r}"
    )


def test_passive_radar_sensor_does_not_import_nats_py():
    """The nats-py package (``from nats.aio.client import ...``) must
    live only in ``nats_detection_source.py`` so a future HTTP/SSE
    transport can be dropped in without churning the sensor."""
    text = _SOURCE_PATH.read_text()
    assert "from nats." not in text
    assert "import nats" not in text


def test_passive_radar_sensor_depends_on_detection_source_protocol():
    text = _SOURCE_PATH.read_text()
    # The module must reference the protocol by name — either as a
    # type annotation or as an import — so swapping implementations is
    # a one-line change in ``from_config``.
    assert "DetectionSource" in text
