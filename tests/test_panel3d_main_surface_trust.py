# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
The trust of the 3D panel method takes the spanwise count of the main surface.

The main surface is the surface with the largest projected area
(``Aircraft.main_surface_index``). Before the fix, the trust took the count
of the first surface in the list, so a coarse tail listed before the wing
gave the coarse-mesh warning of the tail.
"""

from __future__ import annotations

import numpy as np

import ventorum as vt


def _aircraft(tail_first: bool) -> vt.Aircraft:
    tail = vt.LiftingSurface(
        name="tail", semi_span=1.0, n_panels=6, position=np.array([4.0, 0.0, 0.8]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    wing = vt.LiftingSurface(
        name="wing", semi_span=4.0, n_panels=12,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    return vt.Aircraft(surfaces=[tail, wing] if tail_first else [wing, tail])


def test_panel3d_trust_uses_the_main_surface_count():
    """The coarse-mesh warning follows the wing count (12), whatever the order of the surfaces."""
    st = vt.SolverSettings(solver_type="panel3d")
    first = vt.analyze(_aircraft(tail_first=True), alpha_deg=4.0, settings=st).totals.trust
    second = vt.analyze(_aircraft(tail_first=False), alpha_deg=4.0, settings=st).totals.trust
    for trust in (first, second):
        assert not any("Coarse spanwise mesh" in w for w in trust.warnings), trust.warnings
    assert first.score == second.score
    assert first.rating == second.rating
