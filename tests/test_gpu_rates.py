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


def _aircraft(kind: str, af=None) -> vt.Aircraft:
    """Wing, wing and tail (``"tail"``), or wing, tail and a single-side fin (``"fin"``)."""
    if af is None:
        af = vt.LinearAirfoil(a0=2.0 * np.pi, Cd0=0.008, Cm0=-0.04)
    s = [vt.LiftingSurface(
        name="wing", semi_span=5.0, sweep_le=np.radians(5.0), dihedral=np.radians(3.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.25, airfoil=af),
                  vt.WingSection(y_frac=1.0, chord=0.9, airfoil=af, twist=np.radians(-2.0))],
    )]
    if kind in ("tail", "fin"):
        s.append(vt.LiftingSurface(
            name="tail", semi_span=1.5, position=np.array([3.5, 0.0, 0.5]),
            sections=[vt.WingSection(y_frac=0.0, chord=0.6, airfoil=af),
                      vt.WingSection(y_frac=1.0, chord=0.4, airfoil=af)],
        ))
    if kind == "fin":
        s.append(vt.LiftingSurface(
            name="fin", semi_span=1.2, is_symmetric=False, dihedral=np.radians(90.0),
            position=np.array([3.6, 0.0, 0.5]),
            sections=[vt.WingSection(y_frac=0.0, chord=0.7, airfoil=af),
                      vt.WingSection(y_frac=1.0, chord=0.45, airfoil=af)],
        ))
    ac = vt.Aircraft(surfaces=s)
    ac.compute_reference_values()
    return ac


def _polar() -> vt.TabulatedAirfoil:
    """Tabulated polar for the nonlinear lifting line (linear lift, quadratic drag)."""
    a = np.radians(np.arange(-10.0, 25.0, 1.0))
    return vt.TabulatedAirfoil(alpha=a, Cl_data=2.0 * np.pi * a,
                               Cd_data=0.008 + 0.01 * (a / np.radians(10.0)) ** 2,
                               Cm_data=-0.05 + 0.01 * a)


def _cpu_gpu(solver: str, ac, conds, st, precision: str = "float64"):
    """Solve *conds* on the CPU reference path and on the GPU; return both result lists."""
    s_inst = make_solver(solver)
    lattice = s_inst.build(ac, st, conds[0], None, ac.ref_point)
    gpu.set_device("cpu")
    cpu = s_inst.solve_batch(lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point,
                             _cpu_only=True)
    gpu.set_device("gpu")
    gpu.set_precision(precision)
    res = s_inst.solve_batch(lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)
    assert len(cpu) == len(res)
    return cpu, res


def _assert_parity(cpu_res, gpu_res, tol: float, field_tol: float | None = None):
    """Check that GPU result matches CPU result within tolerance.

    *tol* applies to the coefficients. *field_tol* applies to the
    circulation, the strip values and the strip forces (relative to their
    maximum); it is *tol* when not given.
    """
    field_tol = tol if field_tol is None else field_tol
    assert gpu_res.details["device"] == "gpu"
    assert cpu_res.converged == gpu_res.converged

    cl_scale = max(float(abs(cpu_res.totals.CL)), 1e-3)
    for name in ("CL", "CDi", "Cm", "CY", "Cl", "Cn", "CDi_nearfield", "CY_farfield", "Cn_trefftz"):
        a = getattr(cpu_res.totals, name)
        b = getattr(gpu_res.totals, name)
        assert (a is None) == (b is None), name
        if a is not None:
            scale = max(float(abs(a)), cl_scale if name not in ("CDi", "CDi_nearfield") else 1e-3)
            diff = abs(float(a) - float(b))
            assert diff <= tol * scale, f"{name}: diff={diff:g}, scale={scale:g}"

    if cpu_res.totals.CD_total is not None and gpu_res.totals.CD_total is not None:
        diff = abs(float(cpu_res.totals.CD_total) - float(gpu_res.totals.CD_total))
        assert diff <= tol * cl_scale

    ga = np.asarray(cpu_res.details["gamma"], dtype=float)
    gb = np.asarray(gpu_res.details["gamma"], dtype=float)
    g_scale = max(float(np.max(np.abs(ga))), 1e-6)
    assert np.max(np.abs(ga - gb)) <= field_tol * g_scale, "gamma"

    fa = np.asarray(cpu_res.details["loads"].strip_force, dtype=float)
    fb = np.asarray(gpu_res.details["loads"].strip_force, dtype=float)
    f_scale = max(float(np.max(np.abs(fa))), 1e-6)
    assert np.max(np.abs(fa - fb)) <= field_tol * f_scale, "strip_force"

    assert len(cpu_res.spanwise) == len(gpu_res.spanwise)
    for sw_c, sw_g in zip(cpu_res.spanwise, gpu_res.spanwise):
        for field in ("Cl", "gamma", "Cm_section", "Cd_i"):
            vc = getattr(sw_c, field, None)
            vg = getattr(sw_g, field, None)
            if vc is not None and vg is not None:
                sc = max(float(np.max(np.abs(vc))), 1e-6)
                assert np.max(np.abs(vc - vg)) <= field_tol * sc, field


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

    _assert_parity(cpu_res, gpu_res, 1e-4, field_tol=5e-3)


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


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("h", [None, 1.5])
def test_gpu_rate_parity_wing_tail(gpu_device, solver, h):
    """Wing and tail with body rates, in free air and in ground effect: GPU float64 matches CPU to 1e-10."""
    af = _polar() if solver == "nonlinear" else None
    ac = _aircraft("tail", af)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)
    conds = [vt.FlightCondition(V_inf=25.0, alpha=np.radians(3.0), h=h, p=0.2, q=0.1, r=-0.1)]
    cpu, res = _cpu_gpu(solver, ac, conds, st)
    _assert_parity(cpu[0], res[0], 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("h", [None, 1.5])
def test_gpu_rate_parity_single_side_fin(gpu_device, solver, h):
    """A non-symmetric geometry (single-side fin) with body rates: GPU float64 matches CPU to 1e-10.

    The yaw rate on the fin gives a side force, so the far-field lateral
    diagnostics ``CY_farfield`` and ``Cn_trefftz`` are not zero.
    """
    af = _polar() if solver == "nonlinear" else None
    ac = _aircraft("fin", af)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)
    conds = [vt.FlightCondition(V_inf=25.0, alpha=np.radians(2.0), h=h, p=0.1, q=0.05, r=0.2)]
    cpu, res = _cpu_gpu(solver, ac, conds, st)
    assert abs(cpu[0].totals.CY_farfield) > 1e-4
    assert abs(cpu[0].totals.Cn_trefftz) > 1e-6
    _assert_parity(cpu[0], res[0], 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("h", [None, 1.5])
@pytest.mark.parametrize("kind", ["lateral", "pitch"])
def test_gpu_rate_parity_nonlinear_batch(gpu_device, h, kind):
    """The nonlinear lifting line with a batch of rated cases: GPU float64 matches CPU to 1e-10.

    All cases of one batch have the same symmetry (roll or yaw rate in each
    case, or pitch rate only), so they form one GPU group. A batch that mixes
    symmetric and non-symmetric cases needs the cases in order for the
    continuation, and the CPU path solves it.
    """
    ac = _aircraft("tail", _polar())
    st = vt.SolverSettings(solver_type="nonlinear", n_panels=16, n_chord=1)
    if kind == "lateral":
        rates = ((0.2, 0.0, 0.0), (0.0, 0.0, -0.1), (0.1, 0.1, 0.1), (-0.15, 0.05, 0.2), (0.05, -0.1, 0.0))
    else:
        rates = ((0.0, 0.05, 0.0), (0.0, 0.1, 0.0), (0.0, 0.15, 0.0), (0.0, -0.1, 0.0), (0.0, 0.2, 0.0))
    conds = [vt.FlightCondition(V_inf=25.0, alpha=np.radians(a), h=h, p=p, q=q, r=r)
             for a, (p, q, r) in zip((0.0, 2.0, 4.0, 6.0, 8.0), rates)]
    cpu, res = _cpu_gpu("nonlinear", ac, conds, st)
    for cr, gr in zip(cpu, res):
        _assert_parity(cr, gr, 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_gpu_rate_parity_ground_batch(gpu_device, solver):
    """A batch in ground effect with different heights and rates: GPU float64 matches CPU to 1e-10."""
    ac = _aircraft("tail")
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)
    conds = [
        vt.FlightCondition(V_inf=20.0, alpha=np.radians(3.0), h=h, p=p, q=q, r=r)
        for h, p, q, r in ((1.5, 0.0, 0.0, 0.0), (2.0, 0.2, 0.0, 0.0), (2.5, 0.0, 0.15, 0.0),
                           (3.0, 0.1, 0.1, -0.1))
    ]
    cpu, res = _cpu_gpu(solver, ac, conds, st)
    for cr, gr in zip(cpu, res):
        _assert_parity(cr, gr, 1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_rate_free_pipeline_passes_no_omega(gpu_device, monkeypatch, solver):
    """A rate-free batch gives omega=None to the engine, so no device sync tests the rates per solve."""
    from ventorum.gpu import engine

    seen = []
    for name in ("llt_terms", "vlm_solve", "loads"):
        orig = getattr(engine, name)

        def spy(*args, _orig=orig, _name=name, **kwargs):
            seen.append((_name, kwargs.get("omega")))
            return _orig(*args, **kwargs)

        monkeypatch.setattr(engine, name, spy)
    af = _polar() if solver == "nonlinear" else None
    ac = _aircraft("tail", af)
    st = vt.SolverSettings(solver_type=solver, n_panels=16, n_chord=2 if solver == "vlm" else 1)
    s_inst = make_solver(solver)
    conds = [vt.FlightCondition(V_inf=25.0, alpha=np.radians(a)) for a in (0.0, 2.0, 4.0)]
    lattice = s_inst.build(ac, st, conds[0], None, ac.ref_point)
    gpu.set_device("gpu")
    gpu.set_precision("float64")
    res = s_inst.solve_batch(lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)
    assert all(r.details["device"] == "gpu" for r in res)
    assert "loads" in {n for n, _ in seen}
    assert all(om is None for _, om in seen)
    seen.clear()
    rated = [vt.FlightCondition(V_inf=25.0, alpha=np.radians(2.0), q=0.1)]
    res = s_inst.solve_batch(lattice, rated, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=ac.ref_point)
    assert res[0].details["device"] == "gpu"
    assert seen and all(om is not None for _, om in seen)
