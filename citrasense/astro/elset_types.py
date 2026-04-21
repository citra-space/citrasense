"""Elset type string constants used across API client and hardware adapters.

The literal string values match soicat's ``CitraElsetType`` value enum and the
on-wire format of the ``types=`` query parameter on
``GET /satellites/{id}/elsets``.
"""

from __future__ import annotations

CLASSIC_SGP4_ELSET_TYPES: tuple[str, ...] = (
    "SGP4 with Kozai mean motion",
    "SGP4 with Brouwer mean motion",
)
"""Elset types that classic SGP4 propagators (NINA, Orbitals, PlaneWave) can handle.

Hand a TLE of any other theory (``SGP4-XP with Brouwer mean motion``,
``Osculating``, ...) to a classic-SGP4 propagator and it will either silently
mis-propagate (XP) or fail (osculating). Adapters that delegate propagation to
such software should return this tuple from ``select_elset_types()`` so
``fetch_satellite()`` constrains the server-side ranking accordingly.
"""
