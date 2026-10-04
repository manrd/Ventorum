# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Induced drag of a wing and tail when the wing wake passes near the tail (task T-0013).

The Trefftz-plane normal wash uses the same cross-surface core as the near
field, so the induced drag of this case does not depend on the tail mesh.
"""

from __future__ import annotations

import numpy as np

import ventorum as vt

TAIL_PANELS = (6, 8, 12, 16, 24)


def _aircraft(tail_panels: int, tail_z: float) -> vt.Aircraft:
    """Wing (5 deg dihedral) and tail; at alpha 2 deg the wing wake passes near the tail for tail_z 0.2 m."""
    wing = vt.LiftingSurface(
        name="wing", semi_span=4.0, n_panels=16, dihedral=np.radians(5.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.6)])
    tail = vt.LiftingSurface(
        name="tail", semi_span=1.2, n_panels=tail_panels, position=np.array([3.5, 0.0, tail_z]),
        sections=[vt.WingSection(y_frac=0.0, chord=0.5), vt.WingSection(y_frac=1.0, chord=0.4)])
    return vt.Aircraft(surfaces=[wing, tail])


def _cdi_spread(tail_z: float) -> float:
    cdi = []
    for n in TAIL_PANELS:
        totals = vt.analyze(
            _aircraft(n, tail_z),
            condition=vt.FlightCondition(V_inf=50.0, alpha=np.radians(2.0)),
            settings=vt.SolverSettings(solver_type="vlm"),
        ).totals
        cdi.append(totals.CDi)
    cdi = np.array(cdi)
    return float((cdi.max() - cdi.min()) / np.mean(np.abs(cdi)))


def test_tail_mesh_spread():
    """With the tail in the wing wake, CDi over 6 to 24 tail panels varies by less than 15 %."""
    assert _cdi_spread(tail_z=0.2) < 0.15


def test_tail_mesh_spread_out_of_wake():
    """With the tail well above the wing wake, CDi does not depend on the tail mesh."""
    assert _cdi_spread(tail_z=0.6) < 1e-3
