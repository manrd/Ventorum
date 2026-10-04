# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the pitching moment of the classical Fourier solver.

The moment is ``r x F`` of the section forces about the reference point.
The force of a section has a lift part normal to the free stream and an
induced-drag part along it, so a wing above or below the reference point
gets a moment from both parts (independent review round 5, finding A2: the
moment used only the x arms before).
"""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt

ALPHA = np.radians(8.0)


def _wing(z: float = 0.0) -> vt.LiftingSurface:
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                   vt.WingSection(y_frac=1.0, chord=1.0)])
    w.position = np.array([0.0, 0.0, z])
    return w


def _cm(geometry, solver: str) -> float:
    r = vt.analyze(geometry, vt.FlightCondition(V_inf=30.0, alpha=ALPHA),
                   vt.SolverSettings(solver_type=solver, n_panels=40))
    return r.totals.Cm


@pytest.mark.parametrize("dz", [0.5, 1.0])
def test_moment_z_offset_matches_llt(dz):
    """A wing above the reference point: Fourier Cm follows the linear lifting line.

    The difference between the two models at z = 0 is about 1e-3. With the
    wing at z = dz the difference must stay at that size; before the fix
    the Fourier Cm did not change with z (an error of 0.037 per 0.5 m).
    """
    base = abs(_cm(_wing(), "fourier") - _cm(_wing(), "linear"))
    diff = abs(_cm(_wing(dz), "fourier") - _cm(_wing(dz), "linear"))
    assert diff < 2.0 * base + 1e-4
    # The z arm changes the moment (lift tilted forward by alpha).
    assert _cm(_wing(dz), "fourier") < _cm(_wing(), "fourier") - 0.05 * dz


def test_moment_reference_point_below_wing():
    ac = vt.Aircraft(surfaces=[_wing()], ref_point=np.array([0.25, 0.0, -1.0]))
    cm_f, cm_l = _cm(ac, "fourier"), _cm(ac.clone(), "linear")
    assert cm_f == pytest.approx(cm_l, abs=2e-3)
    assert abs(cm_f) > 0.05


def test_moment_transfer():
    """M2 = M1 + (p1 - p2) x F holds to round-off between two reference points."""
    def aircraft(ref):
        w = vt.LiftingSurface(semi_span=4.0, sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=1.0, chord=0.6, twist=np.radians(-3.0), x_le=0.15)])
        w.position = np.array([0.0, 0.0, 0.3])
        return vt.Aircraft(surfaces=[w], ref_point=np.array(ref, dtype=float))

    p1, p2 = np.array([0.25, 0.0, 0.0]), np.array([0.6, 0.0, -0.8])
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(7.0))
    st = vt.SolverSettings(solver_type="fourier", n_panels=40)
    r1, r2 = vt.analyze(aircraft(p1), cond, st), vt.analyze(aircraft(p2), cond, st)
    t = r1.totals
    ca, sa = np.cos(cond.alpha), np.sin(cond.alpha)
    # Force coefficient in body axes (x aft, z up): lift normal to the free stream, drag along it.
    fx = t.CL * (-sa) + t.CDi * ca
    fz = t.CL * ca + t.CDi * sa
    ref = aircraft(p1)
    ref.compute_reference_values()
    d = p1 - p2
    assert r2.totals.Cm == pytest.approx(t.Cm + (d[2] * fx - d[0] * fz) / ref.c_ref, abs=1e-12)


def test_sweep_equals_single_solve():
    w = _wing(0.5)
    st = vt.SolverSettings(solver_type="fourier", n_panels=40)
    alphas = np.radians([2.0, 8.0])
    sweep = vt.FourierSolver().solve_sweep(vt.Aircraft(surfaces=[w]), vt.FlightCondition(V_inf=30.0), st, alphas)
    for a, r in zip(alphas, sweep):
        single = vt.analyze(_wing(0.5), vt.FlightCondition(V_inf=30.0, alpha=float(a)), st)
        assert r.totals.Cm == pytest.approx(single.totals.Cm, rel=1e-12, abs=1e-15)
