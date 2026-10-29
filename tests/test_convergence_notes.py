# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Trust notes and warnings of the measured mesh-convergence limits.

Each note or warning has one case that gives it and one case that does not.
The notes do not change the score: the score, the rating and the penalty
factors stay the same with and without them.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core import trust as T
from ventorum.reference.analytic import wing_sections_elliptic

JUNCTION = "Joined surfaces with a fold"
ROOT = "Root dihedral (fold of"
KINK = "Lifting line at a winglet kink"
TIP = "Finite tip chord"
RATES = "Body rates: the near-field lateral values"
TERMS = "Fourier lifting line with"
BIAS = "Chordwise bias of the panel method"
GULL = "Dihedral break of"


def _wing(semi_span=4.0, chord=1.0, dihedral_deg=0.0, sections=None, name="wing"):
    secs = sections or [vt.WingSection(y_frac=0.0, chord=chord), vt.WingSection(y_frac=1.0, chord=chord)]
    return vt.LiftingSurface(name=name, semi_span=semi_span, sections=secs, dihedral=math.radians(dihedral_deg))


def _winglet_aircraft():
    wing = _wing()
    tip = vt.LiftingSurface(
        name="winglet", semi_span=0.8, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                 vt.WingSection(y_frac=1.0, chord=0.5)],
        dihedral=math.radians(90.0), position=np.array([0.0, 4.0, 0.0]), is_symmetric=False)
    return vt.Aircraft(surfaces=[wing, tip, tip.mirrored()])


def _elliptic_wing():
    return _wing(sections=wing_sections_elliptic(1.0, n_sections=41))


def _gull_wing():
    z1 = 1.4 * math.tan(math.radians(20.0))
    z2 = z1 + 2.6 * math.tan(math.radians(-5.0))
    secs = [vt.WingSection(y_frac=0.0, chord=1.0, z_le=0.0), vt.WingSection(y_frac=0.35, chord=1.0, z_le=z1),
            vt.WingSection(y_frac=1.0, chord=1.0, z_le=z2)]
    return _wing(sections=secs)


def _solve(geometry, solver, n_panels=12, **cond):
    ac = geometry if isinstance(geometry, vt.Aircraft) else vt.Aircraft(surfaces=[geometry])
    fc = vt.FlightCondition(V_inf=40.0, alpha=math.radians(4.0), **cond)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vt.analyze(ac, fc, vt.SolverSettings(solver_type=solver, n_panels=n_panels))


def _has(result, prefix):
    return any(w.startswith(prefix) for w in result.totals.trust.warnings)


def _assert_score_unchanged(result, prefixes):
    """The trust score without the new notes equals the score of the result."""
    tr = result.totals.trust
    kept = [w for w in tr.warnings if not any(w.startswith(p) for p in prefixes)]
    assert len(kept) < len(tr.warnings)
    ref = T.evaluate_aerodynamic_trust(AR=8.0, CL=0.3, CDi=0.005, notes=kept, solver_type=result.solver_type)
    new = T.evaluate_aerodynamic_trust(AR=8.0, CL=0.3, CDi=0.005, notes=tr.warnings, solver_type=result.solver_type)
    assert (new.score, new.rating, new.factors) == (ref.score, ref.rating, ref.factors)


def test_vlm_winglet_junction_note():
    res = _solve(_winglet_aircraft(), "vlm")
    assert _has(res, JUNCTION)
    assert not _has(res, KINK)
    _assert_score_unchanged(res, (JUNCTION,))
    assert not _has(_solve(_wing(), "vlm"), JUNCTION)


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_root_dihedral_note(solver):
    res = _solve(_wing(dihedral_deg=10.0), solver)
    assert _has(res, ROOT)
    assert "fold of 20 deg" in next(w for w in res.totals.trust.warnings if w.startswith(ROOT))
    assert not _has(_solve(_wing(dihedral_deg=2.0), solver), ROOT)
    assert not _has(_solve(_wing(), solver), ROOT)


def test_lifting_line_winglet_kink_warning():
    res = _solve(_winglet_aircraft(), "linear")
    assert _has(res, KINK)
    assert not _has(res, JUNCTION)
    _assert_score_unchanged(res, (KINK,))
    assert not _has(_solve(_wing(), "linear"), KINK)


def test_lifting_line_finite_tip_note():
    res = _solve(_wing(), "linear")
    assert _has(res, TIP)
    assert "tip chord 100 %" in next(w for w in res.totals.trust.warnings if w.startswith(TIP))
    assert not _has(_solve(_elliptic_wing(), "linear", n_panels=20), TIP)
    assert not _has(_solve(_wing(), "vlm"), TIP)


def test_body_rate_note():
    res = _solve(_wing(), "vlm", p=0.2)
    assert _has(res, RATES)
    _assert_score_unchanged(res, (RATES,))
    assert not _has(_solve(_wing(), "vlm"), RATES)


def test_fourier_notes():
    res = _solve(_wing(), "fourier", n_panels=20)
    assert _has(res, TIP)
    assert not _has(res, TERMS)
    assert not _has(_solve(_elliptic_wing(), "fourier", n_panels=20), TIP)
    # Aspect ratio 25: fewer than 50 terms give the warning, 50 terms do not.
    high = _wing(semi_span=12.5)
    res = _solve(high, "fourier", n_panels=20)
    assert _has(res, TERMS)
    _assert_score_unchanged(res, (TERMS, TIP))
    assert not _has(_solve(high, "fourier", n_panels=50), TERMS)


def test_fourier_terms_rule_limits():
    assert T.fourier_convergence_notes(None, 20.0, 10) == []
    assert T.fourier_convergence_notes(None, 20.5, 49)
    assert T.fourier_convergence_notes(None, 20.5, 50) == []


@pytest.mark.slow
def test_panel3d_notes():
    settings = vt.SolverSettings(solver_type="panel3d", n_panels=6, n_chord=10)
    fc = vt.FlightCondition(V_inf=40.0, alpha=math.radians(4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = vt.analyze(vt.Aircraft(surfaces=[_wing()]), fc, settings)
        gull = vt.analyze(vt.Aircraft(surfaces=[_gull_wing()]), fc, settings)
    bias = next(w for w in flat.totals.trust.warnings if w.startswith(BIAS))
    assert "with 10 chordwise panels the lift is probably 5.5 to 8.5 % low" in bias
    assert "panel_estimate_mesh_error=True" in bias
    assert not _has(flat, GULL)
    assert _has(gull, GULL) and _has(gull, BIAS)
    _assert_score_unchanged(gull, (GULL, BIAS))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        vt.analyze(vt.Aircraft(surfaces=[_gull_wing()]), fc, settings)
    assert not any(str(w.message).startswith((BIAS, GULL)) for w in rec)
    assert not _has(_solve(_wing(), "vlm"), BIAS)


def test_panel3d_note_function():
    assert T.panel3d_convergence_notes([_wing()], None) == []
    notes = T.panel3d_convergence_notes([_wing()], 40)
    assert len(notes) == 1 and "1.4 to 2.1 % low" in notes[0]
    assert T.surface_dihedral_break_deg(_gull_wing()) == pytest.approx(25.0)
    assert T.surface_dihedral_break_deg(_wing(dihedral_deg=10.0)) == 0.0


def test_fold_angles_of_lattices():
    win = _solve(_winglet_aircraft(), "vlm").details["lattice"]
    folds = T.lattice_fold_angles(win)
    assert folds["junction"] == pytest.approx(90.0)
    assert folds["root"] == pytest.approx(0.0, abs=1e-9)
    dih = _solve(_wing(dihedral_deg=10.0), "vlm").details["lattice"]
    assert T.lattice_fold_angles(dih)["root"] == pytest.approx(20.0)
    assert T.lattice_fold_angles(None) == {"root": 0.0, "junction": 0.0, "interior": 0.0}
    assert T.lattice_tip_chord_ratio(win, 0) is None  # both tips of the wing are joined
    assert T.lattice_tip_chord_ratio(win, 1) == pytest.approx(0.5)


def _runtime_warnings(geometry, solver, n_panels=12, **cond):
    """Return the texts of the run-time warnings of one solve."""
    ac = geometry if isinstance(geometry, vt.Aircraft) else vt.Aircraft(surfaces=[geometry])
    fc = vt.FlightCondition(V_inf=40.0, alpha=math.radians(4.0), **cond)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        vt.analyze(ac, fc, vt.SolverSettings(solver_type=solver, n_panels=n_panels))
    return [str(w.message) for w in rec]


def test_notes_are_not_runtime_warnings_and_warnings_are():
    """The notes stay in the trust score; the two warnings are also run-time warnings."""
    msgs = _runtime_warnings(_wing(dihedral_deg=10.0), "linear", p=0.2)
    assert not any(m.startswith((ROOT, TIP, RATES)) for m in msgs)
    assert not any(m.startswith(JUNCTION) for m in _runtime_warnings(_winglet_aircraft(), "vlm"))
    assert any(m.startswith(KINK) for m in _runtime_warnings(_winglet_aircraft(), "linear"))
    msgs = _runtime_warnings(_wing(semi_span=12.5), "fourier", n_panels=20)
    assert any(m.startswith(TERMS) for m in msgs)
    assert not any(m.startswith(TIP) for m in msgs)
