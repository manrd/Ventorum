# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Verification tests for the Prandtl-Glauert compressibility correction in the VLM."""

from __future__ import annotations

import math
import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.loads import compute_loads, trefftz_induced_drag
from ventorum.aero.system import (
    UnknownMap,
    freestream_direction,
    lift_direction,
    panel_targets,
    side_direction,
)
from ventorum.aero.vortex import HorseshoeSet, induced_velocity, influence_matrix
from ventorum.core.constants import MACH_LIMIT_INCOMPRESSIBLE
from ventorum.core.datatypes import FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.core.errors import ValidityError
from ventorum.sensitivity.errors import GradientScopeError


def _make_wing_tail():
    """Create a wing and horizontal tail geometry."""
    wing = LiftingSurface(
        name="MainWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.5),
            WingSection(y_frac=1.0, chord=0.9),
        ],
        sweep_le=np.radians(15.0),
        dihedral=np.radians(2.0),
    )
    tail = LiftingSurface(
        name="Tail",
        semi_span=1.8,
        sections=[
            WingSection(y_frac=0.0, chord=0.8),
            WingSection(y_frac=1.0, chord=0.5),
        ],
        position=np.array([4.5, 0.0, 0.3]),
    )
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


def test_flight_condition_mach_validation():
    """Check input validation of the FlightCondition.mach field."""
    cond_default = FlightCondition()
    assert cond_default.mach is None

    cond_zero = FlightCondition(mach=0.0)
    assert cond_zero.mach == 0.0

    cond_valid = FlightCondition(mach=0.65)
    assert cond_valid.mach == 0.65

    cond_cloned = cond_valid.clone()
    assert cond_cloned.mach == 0.65

    with pytest.raises(ValueError, match="mach"):
        FlightCondition(mach=-0.05)

    with pytest.raises(ValueError, match="mach"):
        FlightCondition(mach=1.0)

    with pytest.raises(ValueError, match="mach"):
        FlightCondition(mach=1.2)


def test_solver_refusals_with_mach():
    """Check that lifting line and Fourier solvers refuse mach > 0 with ValidityError."""
    wing = LiftingSurface(
        name="StraightWing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(4.0), mach=0.5)

    for solver_name in ("linear", "nonlinear", "fourier"):
        with pytest.raises(ValidityError) as exc_info:
            vt.analyze(wing, condition=cond, solver=solver_name)
        msg = str(exc_info.value).lower()
        assert "vlm" in msg


def test_ground_effect_refusal_with_mach():
    """Check that ground effect refuses mach > 0 with an error."""
    wing = LiftingSurface(
        name="WingGE",
        semi_span=3.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), h=1.5, mach=0.5)

    with pytest.raises((ValidityError, ValueError)) as exc_info:
        vt.analyze(wing, condition=cond, solver="vlm")
    assert "ground" in str(exc_info.value).lower()


def test_gpu_fallback_with_mach():
    """Check that GPU dispatch runs on the GPU and agrees with the CPU when mach > 0."""
    from ventorum import gpu

    wing = LiftingSurface(
        name="GPUWing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), mach=0.5)
    res_cpu = vt.analyze(wing, condition=cond, solver="vlm")
    if not gpu.available():
        dispatch = res_cpu.details.get("dispatch", {})
        assert dispatch.get("device") == "cpu"
        return
    old_dev = gpu.get_device()
    old_prec = gpu.get_precision()
    try:
        gpu.set_device("gpu")
        gpu.set_precision("float64")
        res_gpu = vt.analyze(wing, condition=cond, solver="vlm")
        dispatch = res_gpu.details.get("dispatch", {})
        assert dispatch.get("device") == "gpu"
        assert np.isclose(res_gpu.totals.CL, res_cpu.totals.CL, rtol=1e-10)
        assert np.isclose(res_gpu.totals.CDi, res_cpu.totals.CDi, rtol=1e-10)
        assert np.allclose(res_gpu.details["gamma"], res_cpu.details["gamma"], rtol=1e-10)
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)


def test_gradient_mach_scope():
    """Check the gradient scope with mach: VLM in free air works, ground and lifting lines refuse."""
    from ventorum import gpu
    from ventorum.sensitivity import check_gradients, gradients

    wing = LiftingSurface(
        name="GradWing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), mach=0.5)
    settings = SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)

    # VLM in free air at mach > 0 gives gradients, including the mach input.
    old = gpu._precision
    gpu.set_precision("float64")
    try:
        res = gradients(wing, cond, settings, outputs=("CL",), wrt=("alpha", "mach"))
        assert np.isfinite(float(res.jacobian["CL"]["alpha"]))
        assert np.isfinite(float(res.jacobian["CL"]["mach"]))
        out = check_gradients(wing, cond, settings, outputs=("CL",),
                              wrt=("alpha", "mach"), rtol=1e-6)
        assert out["pass"], out["reasons"]
    finally:
        gpu.set_precision(old)

    # Ground effect keeps refusing mach > 0.
    cond_ge = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), h=1.5, mach=0.5)
    with pytest.raises(GradientScopeError):
        gradients(wing, cond_ge, settings)

    with pytest.raises(GradientScopeError):
        check_gradients(wing, cond_ge, settings, wrt=("alpha",))

    # Lifting lines keep refusing mach > 0.
    for solver_name in ("linear", "nonlinear"):
        st = SolverSettings(solver_type=solver_name)
        with pytest.raises(GradientScopeError):
            gradients(wing, cond, st)

        with pytest.raises(GradientScopeError):
            check_gradients(wing, cond, st, wrt=("alpha",))

    # The mach input needs a Mach number in the condition.
    cond_none = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), mach=None)
    with pytest.raises(GradientScopeError):
        gradients(wing, cond_none, settings, outputs=("CL",), wrt=("mach",))


def test_trust_score_compressibility():
    """Check trust score penalties and warning text with mach."""
    wing = LiftingSurface(
        name="TrustWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    # 1. Mach 0.5: envelope penalty applies, text names Prandtl-Glauert
    cond_05 = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), mach=0.5)
    res_05 = vt.analyze(wing, condition=cond_05, solver="vlm")
    trust_05 = res_05.totals.trust
    assert trust_05 is not None
    assert trust_05.factors["compressibility"] > 0.0
    pen_expected = 0.60 + 0.30 * min(1.0, (0.5 - MACH_LIMIT_INCOMPRESSIBLE) / 0.35)
    assert np.isclose(trust_05.factors["compressibility"], pen_expected)
    assert any("prandtl-glauert" in w.lower() for w in trust_05.warnings)

    # 2. Mach 0.75: refused (the linear rule has no meaning near the critical
    # Mach number; the solve gave a result rated only by a trust warning before).
    cond_075 = FlightCondition(V_inf=40.0, alpha=np.radians(3.0), mach=0.75)
    with pytest.raises(ValidityError, match="0.7"):
        vt.analyze(wing, condition=cond_075, solver="vlm")


def test_mach_none_and_zero_equality():
    """Check that mach=None and mach=0.0 give results equal bit for bit to today."""
    aircraft = _make_wing_tail()
    cond_none = FlightCondition(
        V_inf=45.0,
        alpha=np.radians(4.0),
        beta=np.radians(2.5),
        mach=None,
    )
    cond_zero = FlightCondition(
        V_inf=45.0,
        alpha=np.radians(4.0),
        beta=np.radians(2.5),
        mach=0.0,
    )
    settings = SolverSettings(solver_type="vlm", n_panels=20, n_chord=2)

    res_none = vt.analyze(aircraft, condition=cond_none, settings=settings)
    res_zero = vt.analyze(aircraft, condition=cond_zero, settings=settings)

    for field in ("CL", "CDi", "CDp", "CD_total", "Cl", "Cm", "Cn", "CY", "CDi_nearfield", "e"):
        v_none = getattr(res_none.totals, field)
        v_zero = getattr(res_zero.totals, field)
        if v_none is None or v_zero is None:
            assert v_none is None and v_zero is None
        else:
            assert np.array_equal(np.asarray(v_none), np.asarray(v_zero))

    assert np.array_equal(res_none.details["gamma"], res_zero.details["gamma"])
    assert res_none.totals.trust.score == res_zero.totals.trust.score


def test_discrete_identity():
    """Check the discrete identity against a hand-stretched lattice and Step 2/Step 4."""
    wing = LiftingSurface(
        name="SweptWing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.2),
            WingSection(y_frac=1.0, chord=0.8),
        ],
        sweep_le=np.radians(10.0),
        dihedral=np.radians(3.0),
    )
    aircraft = vt.Aircraft(name="SweptAircraft", surfaces=[wing])
    aircraft.compute_reference_values()
    mach = 0.55
    beta_pg = math.sqrt(1.0 - mach ** 2)
    cond = FlightCondition(
        V_inf=45.0,
        alpha=np.radians(3.5),
        beta=np.radians(1.5),
        mach=mach,
    )
    settings = SolverSettings(solver_type="vlm", n_panels=16, n_chord=1)
    solver = vt.VortexLatticeSolver()
    lattice = solver.build(aircraft, settings, cond)
    ref_point = np.asarray(aircraft.moment_reference(), dtype=float)

    res = solver.solve_lattice(
        lattice, cond, settings,
        aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
        ref_point=ref_point,
    )
    gamma_solver = res.details["gamma"]

    # Hand-built stretched lattice in free-stream axes
    d = freestream_direction(cond.alpha, cond.beta)
    lift = lift_direction(cond.alpha)
    side = side_direction(cond.alpha, cond.beta)
    R = np.vstack([d, side, lift])

    def to_stretched_fs(pts):
        pts_fs = (pts - ref_point) @ R.T
        pts_s = pts_fs.copy()
        pts_s[:, 0] /= beta_pg
        return pts_s

    a_s = to_stretched_fs(lattice.a)
    b_s = to_stretched_fs(lattice.b)
    a_te_s = to_stretched_fs(lattice.a_te)
    b_te_s = to_stretched_fs(lattice.b_te)
    cp_s = to_stretched_fs(lattice.cp)

    n_p = lattice.n_panels
    umap = UnknownMap(unknown_panels=np.arange(n_p), panel_column=np.arange(n_p), symmetric=False)
    w_fs = np.array([1.0, 0.0, 0.0])
    from ventorum.aero.system import _core_group
    group = _core_group(lattice)[lattice.panel_strip]
    src_s = HorseshoeSet(
        a=a_s, b=b_s, a_te=a_te_s, b_te=b_te_s,
        wake_dir=np.tile(w_fs, (n_p, 1)),
        rc=lattice.rc,
        sign=np.ones(n_p),
        column=umap.panel_column.copy(),
        group=group,
    )
    tg = panel_targets(lattice, umap.unknown_panels)

    # Step 2: normal rule
    n_phys = lattice.normal_bc[umap.unknown_panels]
    n_fs = n_phys @ R.T
    n_star = n_fs.copy()
    n_star[:, 0] /= beta_pg

    A_hand = influence_matrix(cp_s[umap.unknown_panels], n_star, src_s, umap.n, tg)
    rhs_hand = -cond.V_inf * (n_phys @ d)

    gamma_hand = np.linalg.solve(A_hand, rhs_hand)
    diff_gamma = np.max(np.abs(gamma_solver - gamma_hand))
    assert diff_gamma < 1e-12

    # Step 4: hand-checked near-field force on panel 0 bound segment
    mid_0 = lattice.force_points[0:1]
    mid_0_s = to_stretched_fs(mid_0)
    tgt_0 = panel_targets(lattice, np.array([0]))
    v_s_fs = induced_velocity(mid_0_s, src_s, gamma_hand, tgt_0)[0]
    v_phys_fs = v_s_fs.copy()
    v_phys_fs[0] /= beta_pg
    v_phys_geo = v_phys_fs @ R

    l_vec_0 = lattice.b[0] - lattice.a[0]
    F_bound_hand = cond.rho * gamma_hand[0] * np.cross(cond.V_inf * d + v_phys_geo, l_vec_0)

    # Solver's bound force on panel 0
    v_ind_solver = res.details["loads"].extras.get("v_ind")
    if v_ind_solver is None:
        # Re-evaluate with compute_loads to inspect
        loads_res = compute_loads(
            lattice, gamma_solver, cond,
            aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
            ref_point=ref_point, leg_forces=False,
        )
        F_bound_solver = loads_res.strip_force[0]
        # For a 1-chord lattice without leg forces, panel 0 is strip 0
        diff_F = np.max(np.abs(F_bound_hand - F_bound_solver))
        assert diff_F < 1e-12


def test_finite_wing_helmbold_diederich():
    """Check aspect ratio 8 wing lift slope against Helmbold-Diederich within 3%."""
    wing = LiftingSurface(
        name="AR8Wing",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    aircraft = vt.Aircraft(name="AR8Aircraft", surfaces=[wing])
    settings = SolverSettings(solver_type="vlm", n_panels=24, n_chord=4)
    AR = 8.0

    # Mach 0 base slope
    c0_1 = FlightCondition(V_inf=50.0, alpha=np.radians(1.0), mach=0.0)
    c0_2 = FlightCondition(V_inf=50.0, alpha=np.radians(2.0), mach=0.0)
    r0_1 = vt.analyze(aircraft, condition=c0_1, settings=settings)
    r0_2 = vt.analyze(aircraft, condition=c0_2, settings=settings)
    cl_alpha_0 = (r0_2.totals.CL - r0_1.totals.CL) / np.radians(1.0)

    hd_0 = 2.0 * np.pi * AR / (2.0 + math.sqrt(AR ** 2 + 4.0))

    measured_data = []
    for m in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        c_1 = FlightCondition(V_inf=50.0, alpha=np.radians(1.0), mach=m)
        c_2 = FlightCondition(V_inf=50.0, alpha=np.radians(2.0), mach=m)
        r_1 = vt.analyze(aircraft, condition=c_1, settings=settings)
        r_2 = vt.analyze(aircraft, condition=c_2, settings=settings)
        cla_m = (r_2.totals.CL - r_1.totals.CL) / np.radians(1.0)
        ratio_solver = cla_m / cl_alpha_0

        beta_m = math.sqrt(1.0 - m ** 2)
        hd_m = 2.0 * np.pi * AR / (2.0 + math.sqrt(AR ** 2 * beta_m ** 2 + 4.0))
        ratio_hd = hd_m / hd_0
        err = abs(ratio_solver - ratio_hd) / ratio_hd
        measured_data.append((m, cla_m, ratio_solver, ratio_hd, err))
        assert err < 0.03, f"Mach {m}: solver ratio {ratio_solver:.4f} vs HD {ratio_hd:.4f} (err {err*100:.2f}%)"


def test_trefftz_induced_drag():
    """Check that Trefftz plane induced drag uses circulation without explicit 1/beta."""
    wing = LiftingSurface(
        name="DragWing",
        semi_span=5.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0),
            WingSection(y_frac=1.0, chord=0.7),
        ],
    )
    aircraft = vt.Aircraft(name="DragAircraft", surfaces=[wing])
    mach = 0.45
    cond = FlightCondition(V_inf=50.0, alpha=np.radians(4.0), mach=mach)
    settings = SolverSettings(solver_type="vlm", n_panels=20, n_chord=2)
    res = vt.analyze(aircraft, condition=cond, settings=settings)

    lattice = res.details["lattice"]
    gamma = res.details["gamma"]
    strip_gamma = gamma.reshape(lattice.n_strips, lattice.n_chord).sum(axis=1)

    d = freestream_direction(cond.alpha, cond.beta)
    Di_direct, _, _ = trefftz_induced_drag(lattice, strip_gamma, d, cond.rho)
    qS = 0.5 * cond.rho * cond.V_inf ** 2 * aircraft.S_ref
    CDi_direct = Di_direct / qS
    assert np.isclose(res.totals.CDi, CDi_direct, rtol=1e-12)
