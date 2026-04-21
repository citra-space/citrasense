"""Low-level astronomical helpers shared across citrasense layers.

Lives below ``hardware/``, ``web/``, and ``api/`` in the dependency graph so
every layer that needs sidereal time or a keplemon Observatory can import
from a single source of truth.
"""
