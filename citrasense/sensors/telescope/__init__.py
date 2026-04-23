"""Telescope-backed sensor (phase 1 thin wrapper).

The concrete implementation lives in
:mod:`citrasense.sensors.telescope.telescope_sensor`; the surrounding
``hardware/`` tree (adapters, device stack, autofocus, ...) stays where it is
for phase 1 and is reached via ``TelescopeSensor.adapter``.
"""

from citrasense.sensors.telescope.telescope_sensor import TelescopeSensor

__all__ = ["TelescopeSensor"]
