# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the cached parts of the per-case result build of the lattice solvers.

A sweep builds one result object per case. The parts of the build that
depend only on the lattice geometry, or only on the inputs of the
calibrated error model, are computed once and kept. These tests make sure
that the kept values are the values that a direct computation gives.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.core import trust as tr
from ventorum.solvers import lattice_base
from ventorum.solvers.core import direct_solve_info


def _direct_outcome(solver, AR, sweep_deg, taper, n_chord, n_span, h_over_c, controls):
    """Return the error-model outcome without the cache."""
    model = tr.predicted_error(solver, AR, sweep_qc_deg=sweep_deg, taper=taper, n_chord=n_chord,
                               n_span=n_span, h_over_c=h_over_c, controls=controls)
    if model is None:
        return None
    error, k, terms = model
    if k * error >= tr.ERROR_BAND_HIGH:
        warn, rec = tr._error_model_messages(solver, k, error, terms, AR, sweep_deg)
    else:
        warn, rec = [], []
    return tr.error_score(k * error), tuple(warn), tuple(rec)


@pytest.mark.parametrize("args", [
    ("linear", 8.0, 0.0, 1.0, 1, 80, None, ()),
    ("nonlinear", 4.0, 12.0, 0.4, 1, 40, None, ()),
    ("vlm", 8.0, 0.0, None, 4, 40, None, ()),
    ("vlm", 3.0, 30.0, None, 2, 8, 0.2, ((0.2, 0.75),)),
    ("vlm", 8.0, 0.0, None, 4, None, None, ()),
    ("panel3d", 8.0, 0.0, None, 12, 24, None, ()),
    ("linear", 0.0, 0.0, 1.0, 1, 80, None, ()),
])
def test_error_model_outcome_matches_direct(args):
    """The cached outcome equals the outcome of the model without the cache."""
    assert tr._error_model_outcome(*args) == _direct_outcome(*args)
    # The second call reads the cache and gives the same value.
    assert tr._error_model_outcome(*args) == _direct_outcome(*args)


def test_error_model_outcome_keeps_the_argument_types():
    """A numpy float and a Python float are different keys of the cache, with the same value."""
    a = tr._error_model_outcome("linear", 8.0, 0.0, 1.0, 1, 80, None, ())
    b = tr._error_model_outcome("linear", np.float64(8.0), 0.0, 1.0, 1, 80, None, ())
    assert a == b


@pytest.mark.parametrize("rho", [1.225, 0.7, 0.36392, 0.1])
def test_isa_speed_of_sound_cache(rho):
    """The cached ISA speed of sound equals the closed form and accepts a numpy value."""
    if rho == tr.RHO_SL:
        expected = tr.A_SL
    elif rho <= tr._ISA_RHO_TROPOPAUSE:
        expected = math.sqrt(tr._GAMMA_R_AIR * tr._ISA_T_TROPOPAUSE)
    else:
        expected = math.sqrt(tr._GAMMA_R_AIR * tr._ISA_T0 * (rho / tr.RHO_SL) ** (1.0 / tr._ISA_DENSITY_EXPONENT))
    assert tr.isa_speed_of_sound(rho) == pytest.approx(expected, rel=1e-15)
    assert tr.isa_speed_of_sound(np.float64(rho)) == tr.isa_speed_of_sound(rho)
    assert tr.isa_speed_of_sound(np.array(rho)) == tr.isa_speed_of_sound(rho)


@pytest.mark.parametrize("rho", [float("nan"), 0.0, -1.0])
def test_isa_speed_of_sound_invalid_density(rho):
    """A density that is not finite or not positive gives the sea-level value."""
    assert tr.isa_speed_of_sound(rho) == tr.A_SL


def test_direct_solve_info_converged_argument():
    """A batch can give the result of its finite test; without it the circulation is tested."""
    g = np.array([1.0, np.nan])
    assert direct_solve_info(g).converged is False
    assert direct_solve_info(np.ones(3)).converged is True
    assert direct_solve_info(None, converged=False).converged is False
    assert direct_solve_info(None, converged=True, symmetric=True).symmetric is True


def test_mean_line_lattice():
    """The mean-line test is True only for a lattice on a cambered mean line."""
    flat = vt.LiftingSurface(name="flat", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)])
    cambered = vt.LiftingSurface(name="cambered", semi_span=4.0, sections=[
        vt.WingSection(y_frac=0.0, chord=1.0, shape="2412"), vt.WingSection(y_frac=1.0, chord=1.0, shape="2412")])
    st = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=4)
    lat_flat = vt.analyze(flat, alpha_deg=2.0, settings=st).details["lattice"]
    lat_cambered = vt.analyze(cambered, alpha_deg=2.0, settings=st).details["lattice"]
    assert tr.mean_line_lattice(None) is False
    assert tr.mean_line_lattice(lat_flat) is False
    assert tr.mean_line_lattice(lat_cambered) is True


@pytest.mark.parametrize("solver, n_chord", [("linear", None), ("vlm", 4)])
def test_case_constants_once_per_lattice(monkeypatch, solver, n_chord):
    """A sweep computes the geometry-only values of the result build once, with the direct values."""
    calls = []
    direct = lattice_base.lattice_convergence_notes

    def spy(*args, **kwargs):
        calls.append(args)
        return direct(*args, **kwargs)

    monkeypatch.setattr(lattice_base, "lattice_convergence_notes", spy)
    # A semi-span that no other test uses, so the lattice cache has no lattice of it.
    wing = vt.LiftingSurface(name="cache_probe", semi_span=4.321, dihedral=0.15, sections=[
        vt.WingSection(y_frac=0.0, chord=1.3), vt.WingSection(y_frac=1.0, chord=0.7)])
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=n_chord)
    res = vt.analyze_sweep(wing, np.linspace(-2.0, 8.0, 11), settings=st)
    assert len(calls) == 1
    lattice = res[0].details["lattice"]
    notes, warns = direct(lattice, solver == "linear", 0)
    for r in res:
        assert r.details["lattice"] is lattice
        for m in warns + notes:
            assert m in r.details["notes"]
            assert m in r.totals.trust.warnings
