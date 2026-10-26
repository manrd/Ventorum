# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Trust warning on an ill-conditioned mean-line vortex lattice.

On a lattice on the cambered section mean line, straight horseshoe legs
leave the curved mean surface. With narrow strips they pass close to
control points of other strips, and the solve can be near-singular: with
the default settings a NACA 6409 wing gave CL 7.74 at alpha 0 (about 0.50
on wider strips). The legs now follow the mean surface (the node chains of
``VortexLattice.leg_next``), and the default settings give smooth strip
loads. The trust score keeps the rule as a diagnostic: it warns when the
strip lift oscillates between neighbouring strips.
"""

from __future__ import annotations

import numpy as np
import pytest

from ventorum.core.datatypes import (
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    WingSection,
)
from ventorum.core.trust import _MEAN_LINE_STRIP_JUMP, mean_line_strip_jump
from ventorum.solvers.factory import make_solver

_KEY = "Cambered mean-line lattice"


def _wing(shape: str | None, controls=None) -> LiftingSurface:
    return LiftingSurface(
        name="w", semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape=shape), WingSection(y_frac=1.0, chord=1.0, shape=shape)],
        controls=controls or [],
    )


def _solve(wing, settings):
    return make_solver("vlm").solve(wing, FlightCondition(V_inf=30.0, alpha=np.radians(2.0)), settings)


def _warned(res) -> bool:
    return any(_KEY in w for w in res.totals.trust.warnings)


@pytest.mark.parametrize("shape", ["6409", "4412"])
def test_default_settings_on_strong_camber_do_not_warn(shape):
    """The default mesh of a strongly cambered wing gives smooth strip loads and no warning."""
    res = _solve(_wing(shape), SolverSettings())
    lattice = res.details["lattice"]
    assert lattice.leg_next is not None
    assert mean_line_strip_jump(lattice, res.spanwise) < _MEAN_LINE_STRIP_JUMP
    assert not _warned(res)


@pytest.mark.parametrize("shape", ["6409", "4412"])
def test_straight_legs_on_strong_camber_warn(monkeypatch, shape):
    """With the former straight legs (no node chains) the default mesh oscillates and gets the warning."""
    from ventorum.geometry import lattice as mesh_mod
    from ventorum.geometry import lattice_cache

    orig = mesh_mod._leg_nodes

    def straight(a, b, a_te, b_te, hinge_nodes, chain_strips, n_chord, *nodes):
        return orig(a, b, a_te, b_te, hinge_nodes, chain_strips[:0], n_chord)

    monkeypatch.setattr(mesh_mod, "_leg_nodes", straight)
    lattice_cache.clear()
    try:
        res = _solve(_wing(shape), SolverSettings())
    finally:
        lattice_cache.clear()
    lattice = res.details["lattice"]
    assert lattice.leg_next is None
    assert mean_line_strip_jump(lattice, res.spanwise) > _MEAN_LINE_STRIP_JUMP
    assert _warned(res)


@pytest.mark.parametrize("shape", ["6409", "4412", "2412"])
def test_wide_strips_do_not_warn(shape):
    """Sixteen strips per semi-span give smooth strip loads and no warning."""
    res = _solve(_wing(shape), SolverSettings(n_panels=16, n_chord=8))
    assert mean_line_strip_jump(res.details["lattice"], res.spanwise) < _MEAN_LINE_STRIP_JUMP
    assert not _warned(res)


def test_uncambered_wing_has_no_value():
    """Without a cambered mean line the rule does not apply."""
    res = _solve(_wing("0012"), SolverSettings())
    assert mean_line_strip_jump(res.details["lattice"], res.spanwise) is None
    assert not _warned(res)


def test_flap_edges_are_not_counted():
    """A deflected flap changes the strip lift by design: its strips are left out."""
    flap = ControlSurface("flap", 0.2, 0.6, 0.75, np.radians(15.0))
    res = _solve(_wing("2412", [flap]), SolverSettings(n_panels=16, n_chord=8))
    assert mean_line_strip_jump(res.details["lattice"], res.spanwise) < _MEAN_LINE_STRIP_JUMP
    assert not _warned(res)
