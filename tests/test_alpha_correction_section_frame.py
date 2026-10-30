"""Section frame of the alpha-correction coupling on non-planar strips.

A strip correction turns the onset in the section plane of the strip
(the plane normal to its span axis), and the panel coupling measures the
strip lift along the section lift direction. A rigid rotation of a
one-sided wing together with its onset therefore does not change the
corrections or the loads. A vertical fin in sideslip is a planar wing
turned by 90 deg about the x axis, so its polar must act on its side
force.
"""

import warnings

import numpy as np
import pytest

from ventorum.core.datatypes import (
    Aircraft, FlightCondition, LiftingSurface, LinearAirfoil, SolverSettings, WingSection,
)
from ventorum.solvers.factory import make_solver
from ventorum.viscous.alpha_correction import solve_viscous

#: Section angle [rad] of the planar reference wing.
ANGLE = np.radians(6.0)

#: Polar with a slope far below the inviscid section slope [1/rad].
POLAR = LinearAirfoil(a0=4.0)

_SOLVER = {"vlm": "vlm", "llt": "linear", "panel3d": "panel3d"}


def _wing(dihedral_deg, symmetric=False):
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape="0012"),
            WingSection(1.0, 1.0, x_le=-0.25, shape="0012")]
    return Aircraft(surfaces=[LiftingSurface(
        name="Wing", semi_span=4.0, sections=secs, dihedral=np.radians(dihedral_deg),
        is_symmetric=symmetric, spacing="uniform",
    )])


def _settings(model, n_span=6, n_chord=8):
    if model == "panel3d":
        return SolverSettings(solver_type="panel3d", n_panels=n_span, n_chord=n_chord,
                              spacing="uniform", panel_backend="numpy", panel_cache=False)
    return SolverSettings(n_panels=12, n_chord=4, spacing="uniform")


def _rotated_condition(dihedral_deg):
    """Return the onset of the planar wing at ANGLE turned with the wing about x.

    The rotation by the dihedral angle G maps the onset
    (cos a, 0, sin a) to (cos a, -sin a sin G, sin a cos G).
    """
    g = np.radians(dihedral_deg)
    beta = np.arcsin(np.sin(ANGLE) * np.sin(g))
    alpha = np.arctan2(np.sin(ANGLE) * np.cos(g), np.cos(ANGLE))
    return FlightCondition(V_inf=30.0, alpha=alpha, beta=beta)


def _force(totals):
    """Return the force magnitude in the lift and side-force axes [-]."""
    return float(np.hypot(totals.CL, totals.CY))


@pytest.mark.parametrize("model", ["vlm", "llt", "panel3d"])
def test_rigid_rotation_keeps_the_correction(model):
    """Dihedral 30 and 60 deg and a vertical fin give the planar corrections and loads."""
    settings = _settings(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        planar = solve_viscous(_wing(0.0), _rotated_condition(0.0), settings, POLAR, model=model, tol=1e-12)
        planar_inv = make_solver(_SOLVER[model]).solve(_wing(0.0), _rotated_condition(0.0), settings)
        ratio = _force(planar.totals) / _force(planar_inv.totals)
        assert planar.converged
        # The polar lowers the lift.
        assert ratio < 0.8
        for dihedral in (30.0, 60.0, 90.0):
            cond = _rotated_condition(dihedral)
            res = solve_viscous(_wing(dihedral), cond, settings, POLAR, model=model, tol=1e-12)
            inv = make_solver(_SOLVER[model]).solve(_wing(dihedral), cond, settings)
            assert res.converged
            np.testing.assert_allclose(res.details["delta"], planar.details["delta"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(res.details["alpha_eff"], planar.details["alpha_eff"], rtol=0, atol=1e-9)
            assert _force(res.totals) / _force(inv.totals) == pytest.approx(ratio, rel=1e-9)
            if model == "panel3d":
                np.testing.assert_allclose(res.details["cl_inviscid"], planar.details["cl_inviscid"],
                                           rtol=0, atol=1e-9)


@pytest.mark.parametrize("model", ["vlm", "panel3d"])
def test_vertical_fin_side_force_follows_the_polar(model):
    """A vertical fin in sideslip: the polar changes the side force."""
    settings = _settings(model)
    cond = FlightCondition(V_inf=30.0, alpha=0.0, beta=ANGLE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_viscous(_wing(90.0), cond, settings, POLAR, model=model, tol=1e-12)
        inv = make_solver(_SOLVER[model]).solve(_wing(90.0), cond, settings)
    assert res.converged
    assert abs(res.totals.CY) < 0.8 * abs(inv.totals.CY)
    # The strip lift of the coupling: the section lift of the fin. The
    # spanwise Cl of the panel result is on the global lift axis. The 3D
    # panel coupling reads the polar relative to the converged 2D section
    # (target = polar - converged + same-mesh section lift), so there the
    # polar value and the target are checked separately.
    if model == "panel3d":
        np.testing.assert_allclose(res.details["cl_polar"], POLAR.Cl(res.details["alpha_eff"]),
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(res.details["cl_inviscid"], res.details["cl_viscous"], rtol=0, atol=1e-9)
    else:
        cl = np.concatenate([sw.Cl for sw in res.spanwise])
        np.testing.assert_allclose(cl, POLAR.Cl(res.details["alpha_eff"]), rtol=0, atol=1e-9)


def test_dihedral_change_agrees_between_panel_and_lattice():
    """Symmetric dihedral wing: the relative lift change of 3DP V1 follows VLM V1.

    The relative change ``1 - CL_visc / CL_inv`` grows with the dihedral
    because the dihedral lowers the inviscid lift. The 3DP and VLM growth
    factors must agree. A strip lift on the global lift axis and an onset
    change about the y axis make the 3DP factor fall with the cosine of
    the dihedral instead. The tolerance 0.08 comes from the data: on these
    meshes the factors differ by 0.016 at 30 deg and 0.063 at 60 deg, and
    the difference falls when the panel mesh is refined (0.041 at 60 deg
    with 12 x 24 panels).
    """
    cond = FlightCondition(V_inf=30.0, alpha=ANGLE)
    change = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for model in ("vlm", "panel3d"):
            settings = (_settings(model, n_span=8, n_chord=16) if model == "panel3d"
                        else SolverSettings(n_panels=40, n_chord=12, spacing="uniform"))
            for dihedral in (0.0, 30.0, 60.0):
                wing = _wing(dihedral, symmetric=True)
                res = solve_viscous(wing, cond, settings, POLAR, model=model, tol=1e-10)
                inv = make_solver(_SOLVER[model]).solve(wing, cond, settings)
                assert res.converged
                change[model, dihedral] = 1.0 - res.totals.CL / inv.totals.CL
    for dihedral in (30.0, 60.0):
        growth = {m: change[m, dihedral] / change[m, 0.0] for m in ("vlm", "panel3d")}
        print(f"dihedral {dihedral:g} deg: growth VLM {growth['vlm']:.4f}, 3DP {growth['panel3d']:.4f}")
        assert growth["panel3d"] == pytest.approx(growth["vlm"], abs=0.08)


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_dihedral_half_mesh_matches_full_mesh(kutta):
    """The mirrored onset change of a dihedral strip on a half mesh gives the full-mesh result."""
    from dataclasses import replace

    cond = FlightCondition(V_inf=30.0, alpha=ANGLE)
    settings = replace(_settings("panel3d", n_span=4, n_chord=8), panel_kutta=kutta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        half = solve_viscous(_wing(30.0, True), cond, settings, POLAR, model="panel3d", tol=1e-11)
        full = solve_viscous(_wing(30.0, True), cond, replace(settings, use_symmetry=False), POLAR,
                             model="panel3d", tol=1e-11)
    assert half.symmetry_used and not full.symmetry_used
    assert half.converged and full.converged
    np.testing.assert_allclose(half.details["delta"], full.details["delta"], rtol=0, atol=1e-11)
    for key in ("CL", "CDi", "Cm"):
        assert getattr(half.totals, key) == pytest.approx(getattr(full.totals, key), abs=1e-11)
