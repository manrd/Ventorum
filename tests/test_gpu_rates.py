# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests of cases with body rates on the GPU pipelines (:mod:`ventorum.gpu`).

The tests compare the GPU pipeline results against the CPU reference path for
cases with body angular rates p, q and r.

Tolerances:
* float64: 1e-10 relative error on coefficients, circulation and derivatives.
* float32: 1e-4 on coefficients, 1e-3 on rate derivatives.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.solvers.factory import make_solver


def _need_gpu():
    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")
    if os.environ.get("VENTORUM_DEVICE", "").strip().lower() == "cpu":
        pytest.skip("VENTORUM_DEVICE=cpu selects CPU device.")


@pytest.fixture
def gpu_device():
    """Select the GPU device and restore settings after the test."""
    _need_gpu()
    old_dev = gpu.get_device()
    old_prec = gpu._precision
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


def _wing(AR: float = 6.0, chord: float = 1.0, twist_deg: float = 0.0) -> vt.Aircraft:
    """Rectangular wing with linear airfoil."""
    af = vt.LinearAirfoil(a0=2.0 * np.pi, Cd0=0.008, Cm0=-0.04)
    s = [vt.LiftingSurface(
        name="wing",
        semi_span=0.5 * AR * chord,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord, x_le=-0.25 * chord, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=chord, x_le=-0.25 * chord, airfoil=af, twist=np.radians(twist_deg)),
        ],
    )]
    ac = vt.Aircraft(surfaces=s)
    ac.compute_reference_values()
    return ac


def _assert_parity(cpu_res, gpu_res, tol: float):
    """Check that GPU result matches CPU result within tolerance."""
    assert gpu_res.details["device"] == "gpu"
    assert cpu_res.converged == gpu_res.converged

    cl_scale = max(float(abs(cpu_res.totals.CL)), 1e-3)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn", "CDi_nearfield"):
        a = getattr(cpu_res.totals, name)
        b = getattr(gpu_res.totals, name)
        if a is not None and b is not None:
            scale = max(float(abs(a)), cl_scale if name not in ("CDi", "CDi_nearfield") else 1e-3)
            diff = abs(float(a) - float(b))
            assert diff <= tol * scale, f"{name}: diff={diff:g}, scale={scale:g}"

    if cpu_res.totals.CD_total is not None and gpu_res.totals.CD_total is not None:
        diff = abs(float(cpu_res.totals.CD_total) - float(gpu_res.totals.CD_total))
        assert diff <= tol * cl_scale

    ga = np.asarray(cpu_res.details["gamma"], dtype=float)
    gb = np.asarray(gpu_res.details["gamma"], dtype=float)
    g_scale = max(float(np.max(np.abs(ga))), 1e-6)
    assert np.max(np.abs(ga - gb)) <= 50.0 * tol * g_scale

    assert len(cpu_res.spanwise) == len(gpu_res.spanwise)
    for sw_c, sw_g in zip(cpu_res.spanwise, gpu_res.spanwise):
        for field in ("Cl", "gamma", "Cm_section"):
            vc = getattr(sw_c, field, None)
            vg = getattr(sw_g, field, None)
            if vc is not None and vg is not None:
                sc = max(float(np.max(np.abs(vc))), 1e-6)
                assert np.max(np.abs(vc - vg)) <= 50.0 * tol * sc


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("rate_kind", ["p", "q", "r", "combined"])
def test_gpu_rate_parity_float64_free_air(gpu_device, solver, rate_kind):
    """GPU float64 matches CPU float64 within 1e-10 relative error."""
    rates = {
        "p": {"p": 0.3},
        "q": {"q": 0.2},
        "r": {"r": 0.25},
        "combined": {"p": 0.15, "q": 0.1, "r": -0.2},
    }[rate_kind]

    ac = _wing(AR=6.0, chord=1.0)
    fc = vt.FlightCondition(V_inf=25.0, alpha=np.radians(3.0), **rates)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)

    s_inst = make_solver(solver)
    lattice = s_inst.build(ac, st, fc, None, ac.ref_point)

    gpu.set_device("cpu")
    cpu_res = s_inst.solve_batch(lattice, [fc], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point, _cpu_only=True)[0]

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    gpu_res = s_inst.solve_batch(lattice, [fc], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)[0]

    _assert_parity(cpu_res, gpu_res, 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("rate_kind", ["p", "q", "combined"])
def test_gpu_rate_parity_float32_free_air(gpu_device, solver, rate_kind):
    """GPU float32 matches CPU float64 within 1e-4 on coefficients."""
    rates = {
        "p": {"p": 0.25},
        "q": {"q": 0.15},
        "combined": {"p": 0.1, "q": 0.1, "r": 0.1},
    }[rate_kind]

    ac = _wing(AR=6.0, chord=1.0)
    fc = vt.FlightCondition(V_inf=25.0, alpha=np.radians(2.0), **rates)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)

    s_inst = make_solver(solver)
    lattice = s_inst.build(ac, st, fc, None, ac.ref_point)

    gpu.set_device("cpu")
    cpu_res = s_inst.solve_batch(lattice, [fc], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point, _cpu_only=True)[0]

    gpu.set_device("gpu")
    gpu.set_precision("float32")
    gpu_res = s_inst.solve_batch(lattice, [fc], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)[0]

    _assert_parity(cpu_res, gpu_res, 1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_gpu_rate_parity_ground_effect(gpu_device, solver):
    """GPU matches CPU in ground effect with body rates."""
    ac = _wing(AR=6.0, chord=1.0)
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0), h=1.5, p=0.1, q=0.15, r=-0.1)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)

    s_inst = make_solver(solver)
    lattice = s_inst.build(ac, st, fc, None, ac.ref_point)

    gpu.set_device("cpu")
    cpu_res = s_inst.solve_batch(lattice, [fc], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point, _cpu_only=True)[0]

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    gpu_res = s_inst.solve_batch(lattice, [fc], st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)[0]

    _assert_parity(cpu_res, gpu_res, 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_gpu_rate_batch_sweep_mixed_rates(gpu_device, solver):
    """Batch solve with a mix of rate-free and rated cases."""
    ac = _wing(AR=6.0, chord=1.0)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)
    conds = [
        vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0)),
        vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0), p=0.2),
        vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0), q=0.15),
        vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0), r=-0.1),
        vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0), p=0.1, q=0.1, r=0.1),
    ]

    s_inst = make_solver(solver)
    lattice = s_inst.build(ac, st, conds[0], None, ac.ref_point)

    gpu.set_device("cpu")
    cpu_batch = s_inst.solve_batch(lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point, _cpu_only=True)

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    gpu_batch = s_inst.solve_batch(lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)

    assert len(cpu_batch) == len(gpu_batch)
    for cr, gr in zip(cpu_batch, gpu_batch):
        _assert_parity(cr, gr, 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_rate_derivatives_runs_on_gpu(gpu_device, solver):
    """rate_derivatives runs on the GPU and reports device='gpu'."""
    ac = _wing(AR=6.0, chord=1.0)
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0))
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)

    gpu.set_device("cpu")
    cpu_rd = vt.rate_derivatives(ac, fc, st)
    assert cpu_rd.device == "cpu"

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    gpu_rd = vt.rate_derivatives(ac, fc, st)
    assert gpu_rd.device == "gpu"
    assert gpu_rd.precision == "float64"

    # Parity within float64 tolerance 1e-10
    for key, val in cpu_rd.derivatives.items():
        ref_val = val
        new_val = gpu_rd.derivatives[key]
        diff = abs(ref_val - new_val)
        scale = max(abs(ref_val), 1e-3)
        assert diff <= 1e-10 * scale, f"{key}: diff={diff}, scale={scale}"

    # float32 check
    gpu.set_precision("float32")
    gpu_rd32 = vt.rate_derivatives(ac, fc, st)
    assert gpu_rd32.device == "gpu"
    assert gpu_rd32.precision == "float32"
    for key, val in cpu_rd.derivatives.items():
        ref_val = val
        new_val = gpu_rd32.derivatives[key]
        diff = abs(ref_val - new_val)
        scale = max(abs(ref_val), 1e-3)
        assert diff <= 1e-3 * scale, f"{key}: diff={diff}, scale={scale}"


@pytest.mark.gpu
def test_derivative_matrix_reports_gpu(gpu_device):
    """derivative_matrix reports device='gpu' when running on GPU."""
    ac = _wing(AR=6.0, chord=1.0)
    fc = vt.FlightCondition(V_inf=20.0, alpha=np.radians(2.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=2)

    gpu.set_device("gpu")
    gpu.set_precision("float64")
    dm = vt.derivative_matrix(ac, fc, st, groups=("rate",))
    assert dm.device == "gpu"
