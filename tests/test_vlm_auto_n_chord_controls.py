"""Automatic chordwise panel count of the vortex lattice with a deflected control.

The flap effectiveness of the lattice converges at first order with
n_chord (about 40/n_chord % low). With a control (any deflection, zero
included), the automatic count is at least 16, so that all cases of a
control derivative use one mesh. A smaller count that the user sets with a
deflected control gets a trust note.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.trust import control_chord_note
from ventorum.geometry.controls import flap_effectiveness
from ventorum.solvers.lattice_base import CONTROL_MIN_AUTO_N_CHORD, DEFAULT_N_CHORD

# Glauert's flap effectiveness for a hinge at 0.75 c [-].
TAU_075 = 0.6090


def _wing(delta: float, *, with_control: bool = True) -> vt.Aircraft:
    """Rectangular wing of aspect ratio 40 (chord 1 m), NACA 0012, full-span flap with the hinge at 0.75 c."""
    controls = [vt.ControlSurface("flap", 0.0, 1.0, 0.75, delta)] if with_control else []
    return vt.Aircraft(surfaces=[vt.LiftingSurface(
        name="wing", semi_span=20.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0, shape="0012") for e in (0.0, 1.0)],
        controls=controls,
    )], ref_point=np.array([0.25, 0.0, 0.0]))


def _cl(ac: vt.Aircraft, alpha_deg: float, settings: vt.SolverSettings) -> tuple[float, vt.SolverResult]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = vt.VortexLatticeSolver().solve(ac, vt.FlightCondition(V_inf=30.0, alpha=math.radians(alpha_deg)), settings)
    return float(r.totals.CL), r


def _notes(r: vt.SolverResult) -> list[str]:
    return list(r.details["notes"])


def test_glauert_tau_value():
    assert flap_effectiveness(0.75) == pytest.approx(TAU_075, abs=5e-5)


def test_auto_n_chord_flap_effectiveness_within_3_percent():
    """Automatic n_chord with a deflected flap: tau within 3 % of Glauert's 0.6090."""
    settings = vt.SolverSettings(n_panels=20)
    d = math.radians(2.0)
    cl_p, r_p = _cl(_wing(d), 0.0, settings)
    cl_m, _ = _cl(_wing(-d), 0.0, settings)
    cl_a, _ = _cl(_wing(d), 2.0, settings)
    assert r_p.details["lattice"].n_chord >= CONTROL_MIN_AUTO_N_CHORD
    cl_delta = (cl_p - cl_m) / (2.0 * d)
    cl_alpha = (cl_a - cl_p) / math.radians(2.0)
    tau = cl_delta / cl_alpha
    assert abs(tau - TAU_075) / TAU_075 < 0.03, tau
    assert not any("flap effectiveness about" in n for n in _notes(r_p))


def test_resolve_n_chord_rules():
    solver = vt.VortexLatticeSolver()
    st = vt.SolverSettings(n_panels=20)
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)
    assert solver.resolve_n_chord(_wing(0.05), st, cond, None) == CONTROL_MIN_AUTO_N_CHORD
    # A control at zero deflection gets the same count (one mesh for control derivatives).
    assert solver.resolve_n_chord(_wing(0.0), st, cond, None) == CONTROL_MIN_AUTO_N_CHORD
    # No control: the default count, as before.
    assert solver.resolve_n_chord(_wing(0.0, with_control=False), st, cond, None) == DEFAULT_N_CHORD
    # A count that the user sets is used as given.
    assert solver.resolve_n_chord(_wing(0.05), vt.SolverSettings(n_panels=20, n_chord=4), cond, None) == 4
    # In ground effect the deflected control keeps the minimum of 16 (gap rule gives less here).
    cond_ge = vt.FlightCondition(V_inf=30.0, alpha=0.0, h=2.0)
    assert solver.resolve_n_chord(_wing(0.0, with_control=False), st, cond_ge, None) == DEFAULT_N_CHORD
    assert solver.resolve_n_chord(_wing(0.0), st, cond_ge, None) == CONTROL_MIN_AUTO_N_CHORD
    assert solver.resolve_n_chord(_wing(0.05), st, cond_ge, None) == CONTROL_MIN_AUTO_N_CHORD
    # Near the ground the gap rule can ask for more, up to the maximum.
    cond_low = vt.FlightCondition(V_inf=30.0, alpha=0.0, h=0.04)
    assert solver.resolve_n_chord(_wing(0.05), st, cond_low, None) > CONTROL_MIN_AUTO_N_CHORD


def test_user_n_chord_4_gives_note():
    """n_chord 4 set by the user with a deflected flap: the note states about 10 %."""
    settings = vt.SolverSettings(n_panels=20, n_chord=4)
    _, r = _cl(_wing(math.radians(5.0)), 2.0, settings)
    note = control_chord_note(4)
    assert "flap effectiveness about 10 % low (first-order chordwise convergence)" in note
    assert note in _notes(r)
    assert note in r.totals.trust.warnings
    with pytest.warns(RuntimeWarning, match="flap effectiveness about 10 % low"):
        vt.VortexLatticeSolver().solve(_wing(math.radians(5.0)), vt.FlightCondition(V_inf=30.0, alpha=0.0), settings)


def test_note_does_not_change_the_score():
    """The note is a warning only: the same score with and without it."""
    from ventorum.core.trust import evaluate_aerodynamic_trust

    base = dict(AR=40.0, CL=0.3, CDi=0.001, n_panels=40, max_sweep_rad=0.0, solver_type="vlm", n_chord=4)
    a = evaluate_aerodynamic_trust(**base)
    b = evaluate_aerodynamic_trust(**base, notes=[control_chord_note(4)])
    assert a.score == b.score


@pytest.mark.parametrize("n_chord", [1, 16])
def test_no_note_at_or_above_minimum_or_section_mode(n_chord):
    settings = vt.SolverSettings(n_panels=20, n_chord=n_chord)
    _, r = _cl(_wing(math.radians(5.0)), 2.0, settings)
    assert not any("flap effectiveness about" in n for n in _notes(r))


def test_no_note_without_deflection():
    settings = vt.SolverSettings(n_panels=20, n_chord=4)
    for ac in (_wing(0.0), _wing(0.0, with_control=False)):
        _, r = _cl(ac, 2.0, settings)
        assert not any("flap effectiveness about" in n for n in _notes(r))


def _small_wing() -> vt.Aircraft:
    """Rectangular wing of aspect ratio 8 with a part-span flap at zero deflection."""
    return vt.Aircraft(surfaces=[vt.LiftingSurface(
        name="wing", semi_span=4.0,
        sections=[vt.WingSection(y_frac=e, chord=1.0) for e in (0.0, 1.0)],
        controls=[vt.ControlSurface("flap", 0.2, 0.6, 0.75, 0.0)],
    )], ref_point=np.array([0.25, 0.0, 0.0]))


def test_control_derivatives_use_one_mesh_at_zero_deflection(monkeypatch):
    """At zero base deflection, the base, alpha-step and deflection-step solves have the same n_chord."""
    seen = []
    original = vt.analyze

    def recording(*args, **kwargs):
        r = original(*args, **kwargs)
        seen.append(int(r.details["lattice"].n_chord))
        return r

    monkeypatch.setattr(vt, "analyze", recording)
    cond = vt.FlightCondition(V_inf=30.0, alpha=math.radians(3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vt.control_derivatives(_small_wing(), cond, vt.SolverSettings(solver_type="vlm", n_panels=12))
    # Base, two deflection steps and two alpha steps.
    assert len(seen) == 5
    assert set(seen) == {CONTROL_MIN_AUTO_N_CHORD}


def test_derivative_matrix_control_column_equals_control_derivatives():
    """The control column of the derivative matrix equals a standalone control_derivatives call."""
    cond = vt.FlightCondition(V_inf=30.0, alpha=math.radians(3.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dm = vt.derivative_matrix(_small_wing(), cond, st, groups=("static", "control"))
        cd = vt.control_derivatives(_small_wing(), cond, st, axes="body", hinge_alpha=False).derivatives["flap"]
    col = dm.matrix[:, list(dm.columns).index("delta:flap")]
    expected = np.array([cd.CL, cd.CD, cd.CY, cd.Cl, cd.Cm, cd.Cn])
    np.testing.assert_allclose(col, expected, rtol=1e-12, atol=1e-14)
    assert dm.base.details["lattice"].n_chord == CONTROL_MIN_AUTO_N_CHORD


def test_case_without_controls_unchanged():
    """A case without a deflected control keeps n_chord 4 and the same result as n_chord 4 set by hand."""
    ac = _wing(0.0, with_control=False)
    cl_auto, r_auto = _cl(ac, 3.0, vt.SolverSettings(n_panels=20))
    cl_set, _ = _cl(ac, 3.0, vt.SolverSettings(n_panels=20, n_chord=DEFAULT_N_CHORD))
    assert r_auto.details["lattice"].n_chord == DEFAULT_N_CHORD
    assert cl_auto == cl_set
