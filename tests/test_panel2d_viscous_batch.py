# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the batched viscous solver (torch, CPU and CUDA).

The batch runs the same algorithm and the same residual and Jacobian code
as the sequential solver. These tests use the tolerance 1e-10 on the
residual so that both solvers stop at the same discrete solution to
round-off level.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from ventorum.airfoil import airfoil
from ventorum.panel2d.viscous import solve_viscous
from ventorum.panel2d.viscous_batch import ViscousCase, solve_viscous_batch

TOL = 1e-10


def _cases():
    """Return the batch cases and the sequential sweeps they correspond to."""
    sweeps = []
    for code in ("0012", "2412"):
        coords = airfoil(code).repanel(n_panels=120).coords
        for re in (5e5, 2e6):
            sweeps.append((coords, re, None, [-2.0, 0.0, 2.0, 4.0]))
    coords = airfoil("0012").repanel(n_panels=120).coords
    sweeps.append((coords, 1e6, 0.5, [0.0, 2.0]))
    cases = [ViscousCase(coordinates=c, alpha=a, Re=re, height=h, tol=TOL)
             for c, re, h, alphas in sweeps for a in alphas]
    return cases, sweeps


def _sequential(sweeps):
    """Return the sequential results per case, in the case order of :func:`_cases`."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c, re, h, alphas in sweeps:
            r = solve_viscous(c, alphas, re, height=h, tol=TOL)
            for i in range(len(alphas)):
                out.append({"Cl": r.Cl[i], "Cd": r.Cd[i], "Cm": r.Cm[i], "xtr": r.transition_positions[i],
                            "converged": r.converged[i]})
    return out


@pytest.mark.slow
def test_batch_cpu_float64_equals_sequential():
    """torch float64 on the CPU equals the sequential numpy solver per case to 1e-8."""
    cases, sweeps = _cases()
    seq = _sequential(sweeps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bat = solve_viscous_batch(cases, device="cpu", dtype=torch.float64)
    assert len(bat) == len(cases)
    for s, b in zip(seq, bat):
        assert b.converged == s["converged"]
        assert abs(b.Cl - s["Cl"]) < 1e-8
        assert abs(b.Cd - s["Cd"]) < 1e-8
        assert abs(b.Cm - s["Cm"]) < 1e-8
        assert np.allclose(b.transition_positions, s["xtr"], atol=1e-8, rtol=0.0)
    assert np.all(bat.converged)


def test_batch_reports_failures():
    """A case that cannot converge is reported per case (converged False, reason, residual)."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    cases = [ViscousCase(coordinates=coords, alpha=2.0, Re=1e6, max_iter=2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_viscous_batch(cases, device="cpu", dtype=torch.float64)
    assert not r.converged[0]
    assert "no convergence" in r.reason[0]
    assert r.residual[0] > 1e-6
    # The loads and distributions of a failed case are NaN (not the last iterate).
    assert np.isnan(r.Cl[0]) and np.isnan(r.Cd[0]) and np.isnan(r.Cm[0])
    assert np.isnan(r.Cd_friction[0]) and np.isnan(r.Cd_pressure[0])
    assert np.all(np.isnan(r[0].Cp)) and np.all(np.isnan(r[0].theta))
    with pytest.raises(ValueError):
        solve_viscous_batch(cases, backend="numpy")


@pytest.mark.slow
def test_batch_work_counters_and_budget():
    """The batch reports the work per case (batched first attempt included) and applies the work budget."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    cases = [ViscousCase(coordinates=coords, alpha=a, Re=1e6) for a in (0.0, 1.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_viscous_batch(cases, device="cpu", dtype=torch.float64)
        seq = solve_viscous(coords, [0.0, 1.0], 1e6)
        cut = solve_viscous_batch([ViscousCase(coordinates=coords, alpha=1.0, Re=1e6, work_budget=3)],
                                  device="cpu", dtype=torch.float64)
    assert np.all(r.converged)
    assert np.array_equal(r.newton_iterations, seq.newton_iterations)
    assert np.all(r.linear_solves >= r.newton_iterations)
    assert not cut.converged[0] and "work budget" in cut.reason[0]
    assert cut.newton_iterations[0] <= 3
    with pytest.raises(ValueError):
        solve_viscous_batch([ViscousCase(coordinates=coords, alpha=1.0, Re=1e6, stall_factor=2.0)])


@pytest.mark.gpu
@pytest.mark.slow
def test_batch_cuda():
    """CUDA float64 equals the sequential solver to 1e-8; float32 stays within a measured floor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cases, sweeps = _cases()
    seq = _sequential(sweeps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b64 = solve_viscous_batch(cases, device="cuda", dtype=torch.float64)
    for s, b in zip(seq, b64):
        assert abs(b.Cl - s["Cl"]) < 1e-8
        assert abs(b.Cd - s["Cd"]) < 1e-8
    cases32 = [ViscousCase(coordinates=c.coordinates, alpha=c.alpha, Re=c.Re, height=c.height, tol=1e-6)
               for c in cases]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b32 = solve_viscous_batch(cases32, device="cuda", dtype=torch.float32)
    ok = b32.converged
    d_cl = np.abs(b32.Cl[ok] - np.array([s["Cl"] for s in seq])[ok])
    d_cd = np.abs(b32.Cd[ok] / np.array([s["Cd"] for s in seq])[ok] - 1.0)
    print(f"\n[batch] float32 CUDA: {int(ok.sum())} of {len(ok)} converged; largest |dCl| {d_cl.max():.2e}, "
          f"largest relative dCd {d_cd.max():.2e}; residuals of the others {b32.residual[~ok]}")
    assert ok.sum() >= 0.8 * len(ok)
    assert d_cl.max() < 1e-4
    assert d_cd.max() < 1e-3


def test_batch_reference_chord_is_checked():
    """ViscousCase takes the reference chord of solve_viscous; a value that is not positive raises ValueError."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    with pytest.raises(ValueError, match="chord_ref"):
        solve_viscous_batch([ViscousCase(coordinates=coords, alpha=0.0, Re=1e6, chord_ref=-1.0)])
    case = ViscousCase(coordinates=coords, alpha=2.0, Re=1e6, max_iter=1, chord_ref=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_viscous_batch([case], device="cpu", dtype=torch.float64)
    assert not r.converged[0] and np.isnan(r.Cl[0])


@pytest.mark.slow
def test_batch_takes_shapes_and_reference_values():
    """ViscousCase takes a flapped shape (undeflected reference) and explicit chord_ref and moment_ref.

    The shape cases, the point-array cases with the undeflected chord and quarter-chord point given, and
    the shape cases with another moment point form separate sequences; each equals the sequential solver
    to 1e-8 (torch float64 on the CPU).
    """
    flapped = airfoil("0012", n_points=121).with_flap(x_hinge=0.75, deflection=3.0)
    alphas = [0.0, 2.0]
    cases = [ViscousCase(coordinates=flapped, alpha=a, Re=1e6, tol=TOL) for a in alphas]
    ref = flapped.reference_geometry()
    cases += [ViscousCase(coordinates=flapped.coords, alpha=a, Re=1e6, tol=TOL, chord_ref=ref.chord_ref,
                          moment_ref=ref.quarter_chord) for a in alphas]
    cases += [ViscousCase(coordinates=flapped, alpha=a, Re=1e6, tol=TOL, moment_ref=(0.5, 0.1)) for a in alphas]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bat = solve_viscous_batch(cases, device="cpu", dtype=torch.float64)
        seq = solve_viscous(flapped, alphas, 1e6, tol=TOL)
        moved = solve_viscous(flapped, alphas, 1e6, tol=TOL, moment_ref=(0.5, 0.1))
    assert np.all(bat.converged) and np.all(seq.converged)
    for k, ref in ((0, seq), (2, seq), (4, moved)):
        for i in range(2):
            b = bat[k + i]
            assert abs(b.Cl - ref.Cl[i]) < 1e-8 and abs(b.Cd - ref.Cd[i]) < 1e-8
            assert abs(b.Cm - ref.Cm[i]) < 1e-8
    assert np.max(np.abs(moved.Cm - seq.Cm)) > 1e-3


def test_batch_moment_point_is_checked():
    """A ViscousCase moment point that is not two finite numbers raises ValueError."""
    coords = airfoil("0012").repanel(n_panels=80).coords
    with pytest.raises(ValueError, match="moment_ref"):
        solve_viscous_batch([ViscousCase(coordinates=coords, alpha=0.0, Re=1e6, moment_ref=(0.25,))])
