# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of control surfaces: definition, deflection and effects in solvers.

Tests verify:
1. Thin-airfoil flap theory formulas and tables (tau and dCm_ddelta).
2. Zero deflection gives bit-identical results to wings without controls.
3. Section mode in lifting line equals an equivalent shifted airfoil model.
4. Chordwise VLM flap lift ratio approaches thin-airfoil tau with n_chord.
5. Sign conventions of lift and pitching moment for flap deflection.
6. Aileron signs and proof that asymmetric controls do not fold symmetry.
7. Rudder signs (side force CY < 0 and yawing moment Cn > 0).
8. Twin rudders on mirrored vertical fins double the side force.
9. Span limits snap to strip edges without altering section breaks.
10. Chordwise hinge edge snapping on control strips only.
11. Cache key sensitivity to all control surface fields.
12. Aircraft.set_deflection linking controls across surfaces by name.
13. Input validation refusing invalid control definitions.
14. Fourier solver refusing non-zero control surface deflections.
15. Aircraft JSON serialization round-trip preserving controls.
16. Control surfaces operating reliably in ground effect.
17. GPU vs CPU parity for deflected control surfaces across solvers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.geometry import controls as ctrl_mod
from ventorum.geometry import lattice_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Thin-airfoil table check
# ═══════════════════════════════════════════════════════════════════════════════

def test_flap_effectiveness_matches_thin_airfoil_table():
    """Prove that flap effectiveness tau and moment derivative match thin-airfoil theory."""
    # Values from Glauert (1926) / Katz & Plotkin (2001) as given in the card table
    table = [
        # (hinge_x_c, tau, dCm_ddelta)
        (0.6, 0.7477845036, -0.5878775383),
        (0.7, 0.6607459491, -0.6415605973),
        (0.75, 0.6089977810, -0.6495190528),
        (0.8, 0.5498151442, -0.6400000000),
        (0.9, 0.3958186964, -0.5400000000),
    ]
    for h, tau_ref, dcm_ref in table:
        tau_val = ctrl_mod.flap_effectiveness(h)
        dcm_val = ctrl_mod.flap_moment_derivative(h)
        assert abs(tau_val - tau_ref) <= 1e-9, f"tau mismatch at h={h}: {tau_val} vs {tau_ref}"
        assert abs(dcm_val - dcm_ref) <= 1e-9, f"dCm mismatch at h={h}: {dcm_val} vs {dcm_ref}"

    # Asymptotic limits near the edges
    tau_near_0 = ctrl_mod.flap_effectiveness(1e-14)
    tau_near_1 = ctrl_mod.flap_effectiveness(1.0 - 1e-14)
    assert abs(tau_near_0 - 1.0) <= 1e-6
    assert abs(tau_near_1 - 0.0) <= 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Zero deflection gives the exact same bits
# ═══════════════════════════════════════════════════════════════════════════════

def test_zero_deflection_full_span_gives_the_same_bits():
    """Prove that a zero-deflection control surface produces bit-identical results."""
    lattice_cache.clear()
    wing_clean = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        n_panels=8,
    )
    wing_ctrl = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)],
        n_panels=8,
    )

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    solvers = [
        ("vlm", vt.VortexLatticeSolver(), vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=4, chord_spacing="uniform")),
        ("linear", vt.LinearLLTSolver(), vt.SolverSettings(solver_type="linear", n_panels=8)),
        ("nonlinear", vt.NonlinearSolver(), vt.SolverSettings(solver_type="nonlinear", n_panels=8)),
    ]

    for name, s_inst, st in solvers:
        lattice_cache.clear()
        r_clean = s_inst.solve(vt.Aircraft(surfaces=[wing_clean]), cond, st)
        lattice_cache.clear()
        r_ctrl = s_inst.solve(vt.Aircraft(surfaces=[wing_ctrl]), cond, st)

        assert r_clean.totals.CL == r_ctrl.totals.CL, f"{name}: CL not bit-identical"
        assert r_clean.totals.CDi == r_ctrl.totals.CDi, f"{name}: CDi not bit-identical"
        assert r_clean.totals.Cm == r_ctrl.totals.Cm, f"{name}: Cm not bit-identical"
        assert r_clean.totals.Cl == r_ctrl.totals.Cl, f"{name}: Cl not bit-identical"
        assert r_clean.totals.Cn == r_ctrl.totals.Cn, f"{name}: Cn not bit-identical"
        assert r_clean.totals.CY == r_ctrl.totals.CY, f"{name}: CY not bit-identical"
        assert np.array_equal(r_clean.details["gamma"], r_ctrl.details["gamma"]), f"{name}: gamma not bit-identical"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Section mode equals shifted airfoil
# ═══════════════════════════════════════════════════════════════════════════════

def test_section_mode_equals_shifted_airfoil():
    """Prove that section mode deflection in lifting lines equals an analytically shifted airfoil."""
    lattice_cache.clear()
    h = 0.7
    delta = np.radians(4.0)
    tau = ctrl_mod.flap_effectiveness(h)
    dcm = ctrl_mod.flap_moment_derivative(h)

    # 1. Linear lifting line
    af_lin = vt.LinearAirfoil(name="base", a0=2.0 * np.pi, alpha_L0=0.0, Cm0=-0.02, Cd0=0.008)
    wing_ctrl = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0, airfoil=af_lin), vt.WingSection(1.0, 1.0, airfoil=af_lin)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=h, deflection=delta)],
        n_panels=10,
    )

    af_shifted_lin = vt.LinearAirfoil(
        name="shifted",
        a0=af_lin.a0,
        alpha_L0=float(af_lin.alpha_L0 - tau * delta),
        Cd0=af_lin.Cd0,
        Cm0=float(af_lin.Cm0 + dcm * delta),
    )
    wing_shifted_lin = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0, airfoil=af_shifted_lin), vt.WingSection(1.0, 1.0, airfoil=af_shifted_lin)],
        n_panels=10,
    )

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(3.0))
    st_lin = vt.SolverSettings(solver_type="linear", n_panels=10)

    r_ctrl = vt.LinearLLTSolver().solve(wing_ctrl, cond, st_lin)
    r_shift = vt.LinearLLTSolver().solve(wing_shifted_lin, cond, st_lin)

    for field_name in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
        v_c = getattr(r_ctrl.totals, field_name)
        v_s = getattr(r_shift.totals, field_name)
        if abs(v_s) > 1e-12:
            assert abs(v_c - v_s) / abs(v_s) <= 1e-12, f"Linear {field_name} mismatch: {v_c} vs {v_s}"
        else:
            assert abs(v_c - v_s) <= 1e-14, f"Linear {field_name} mismatch near zero: {v_c} vs {v_s}"

    # 2. Nonlinear lifting line with TabulatedAirfoil
    alpha_tab = np.radians(np.arange(-10.0, 20.1, 2.0))
    cl_tab = 2.0 * np.pi * alpha_tab
    cd_tab = 0.008 + 0.01 * (alpha_tab / np.radians(10)) ** 2
    cm_tab = -0.05 + 0.0 * alpha_tab

    af_tab = vt.TabulatedAirfoil(name="tab_base", alpha=alpha_tab, Cl_data=cl_tab, Cd_data=cd_tab, Cm_data=cm_tab)
    wing_ctrl_tab = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0, airfoil=af_tab), vt.WingSection(1.0, 1.0, airfoil=af_tab)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=h, deflection=delta)],
        n_panels=10,
    )

    af_shifted_tab = vt.TabulatedAirfoil(
        name="tab_shifted",
        alpha=alpha_tab - tau * delta,
        Cl_data=cl_tab.copy(),
        Cd_data=cd_tab.copy(),
        Cm_data=cm_tab + dcm * delta,
    )
    wing_shifted_tab = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0, airfoil=af_shifted_tab), vt.WingSection(1.0, 1.0, airfoil=af_shifted_tab)],
        n_panels=10,
    )

    st_nl = vt.SolverSettings(solver_type="nonlinear", n_panels=10)
    r_ctrl_nl = vt.NonlinearSolver().solve(wing_ctrl_tab, cond, st_nl)
    r_shift_nl = vt.NonlinearSolver().solve(wing_shifted_tab, cond, st_nl)

    for field_name in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
        v_c = getattr(r_ctrl_nl.totals, field_name)
        v_s = getattr(r_shift_nl.totals, field_name)
        if abs(v_s) > 1e-12:
            assert abs(v_c - v_s) / abs(v_s) <= 1e-12, f"Nonlinear {field_name} mismatch: {v_c} vs {v_s}"
        else:
            assert abs(v_c - v_s) <= 1e-14, f"Nonlinear {field_name} mismatch near zero: {v_c} vs {v_s}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VLM flap ratio approaches tau
# ═══════════════════════════════════════════════════════════════════════════════

def test_vlm_flap_ratio_approaches_tau():
    """Prove that VLM (dCL/ddelta)/(dCL/dalpha) converges towards thin-airfoil tau with n_chord."""
    lattice_cache.clear()
    h = 0.75
    tau_target = ctrl_mod.flap_effectiveness(h)
    step = np.radians(0.5)

    ratios: dict[int, float] = {}
    errors: dict[int, float] = {}

    for nc in (4, 8, 16):
        lattice_cache.clear()
        st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=nc, chord_spacing="uniform")

        # dCL/ddelta at alpha=0
        cond_0 = vt.FlightCondition(V_inf=30.0, alpha=0.0)
        wing_p = vt.LiftingSurface(
            semi_span=4.0,
            sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
            controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=h, deflection=step)],
            n_panels=20,
        )
        wing_m = vt.LiftingSurface(
            semi_span=4.0,
            sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
            controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=h, deflection=-step)],
            n_panels=20,
        )
        cl_delta_p = vt.VortexLatticeSolver().solve(wing_p, cond_0, st).totals.CL
        cl_delta_m = vt.VortexLatticeSolver().solve(wing_m, cond_0, st).totals.CL
        dcl_ddelta = (cl_delta_p - cl_delta_m) / (2.0 * step)

        # dCL/dalpha at delta=0
        wing_base = vt.LiftingSurface(
            semi_span=4.0,
            sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
            controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=h, deflection=0.0)],
            n_panels=20,
        )
        cond_p = vt.FlightCondition(V_inf=30.0, alpha=step)
        cond_m = vt.FlightCondition(V_inf=30.0, alpha=-step)
        cl_a_p = vt.VortexLatticeSolver().solve(wing_base, cond_p, st).totals.CL
        cl_a_m = vt.VortexLatticeSolver().solve(wing_base, cond_m, st).totals.CL
        dcl_dalpha = (cl_a_p - cl_a_m) / (2.0 * step)

        ratio = dcl_ddelta / dcl_dalpha
        ratios[nc] = ratio
        errors[nc] = abs(ratio - tau_target) / tau_target

    print(f"\nFlap ratios: n_chord 4: {ratios[4]:.6f}, n_chord 8: {ratios[8]:.6f}, n_chord 16: {ratios[16]:.6f} (tau={tau_target:.6f})")

    # Ratio at n_chord=16 within 3% of tau(0.75)
    assert errors[16] <= 0.03, f"n_chord=16 error {errors[16]*100:.2f}% exceeds 3%"
    # Error at 16 not larger than at 4
    assert errors[16] <= errors[4] + 1e-12, f"Error at 16 ({errors[16]}) is larger than at 4 ({errors[4]})"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Flap signs
# ═══════════════════════════════════════════════════════════════════════════════

def test_flap_signs():
    """Prove that positive flap deflection increases CL and decreases Cm (nose down)."""
    lattice_cache.clear()
    wing_0 = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=0.0)],
        n_panels=10,
    )
    wing_def = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(5.0))],
        n_panels=10,
    )
    ac_0 = vt.Aircraft(surfaces=[wing_0], ref_point=np.zeros(3))
    ac_def = vt.Aircraft(surfaces=[wing_def], ref_point=np.zeros(3))
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0)

    # VLM
    st_vlm = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4, chord_spacing="uniform")
    r0_vlm = vt.VortexLatticeSolver().solve(ac_0, cond, st_vlm)
    rdef_vlm = vt.VortexLatticeSolver().solve(ac_def, cond, st_vlm)

    print(f"\nVLM flap: CL={rdef_vlm.totals.CL:.6f} (base {r0_vlm.totals.CL:.6f}), Cm={rdef_vlm.totals.Cm:.6f} (base {r0_vlm.totals.Cm:.6f})")
    assert rdef_vlm.totals.CL > r0_vlm.totals.CL, "VLM flap must increase CL"
    assert rdef_vlm.totals.Cm < r0_vlm.totals.Cm, "VLM flap must decrease Cm (nose down)"

    # Linear LLT
    st_lin = vt.SolverSettings(solver_type="linear", n_panels=10)
    r0_lin = vt.LinearLLTSolver().solve(ac_0, cond, st_lin)
    rdef_lin = vt.LinearLLTSolver().solve(ac_def, cond, st_lin)

    print(f"Linear LLT flap: CL={rdef_lin.totals.CL:.6f} (base {r0_lin.totals.CL:.6f}), Cm={rdef_lin.totals.Cm:.6f} (base {r0_lin.totals.Cm:.6f})")
    assert rdef_lin.totals.CL > r0_lin.totals.CL, "Linear LLT flap must increase CL"
    assert rdef_lin.totals.Cm < r0_lin.totals.Cm, "Linear LLT flap must decrease Cm (nose down)"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Aileron signs and no symmetry folding
# ═══════════════════════════════════════════════════════════════════════════════

def test_aileron_signs_and_no_symmetry_fold():
    """Prove that positive aileron produces negative roll moment and disables symmetry folding."""
    lattice_cache.clear()
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        is_symmetric=True,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(5.0), symmetric=False)],
        n_panels=10,
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.zeros(3))
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    # 1. VLM
    st_sym = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4, use_symmetry=True)
    st_nosym = vt.SolverSettings(solver_type="vlm", n_panels=10, n_chord=4, use_symmetry=False)

    lat = vt.build_lattice(ac, st_sym)
    assert not lat.can_fold_symmetry(), "Aileron must disable symmetry folding"

    r_sym = vt.VortexLatticeSolver().solve(ac, cond, st_sym)
    r_nosym = vt.VortexLatticeSolver().solve(ac, cond, st_nosym)

    print(f"\nAileron VLM: Cl={r_sym.totals.Cl:.6f}, CL={r_sym.totals.CL:.6f}")
    assert r_sym.totals.Cl < 0.0, f"Aileron must give Cl < 0, got {r_sym.totals.Cl}"

    for coef in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
        vs = getattr(r_sym.totals, coef)
        vns = getattr(r_nosym.totals, coef)
        if abs(vns) > 1e-12:
            assert abs(vs - vns) / abs(vns) <= 1e-12, f"VLM symmetry fold mismatch on {coef}"
        else:
            assert abs(vs - vns) <= 1e-14, f"VLM symmetry fold mismatch on near-zero {coef}"

    # 2. Linear LLT
    st_lin_sym = vt.SolverSettings(solver_type="linear", n_panels=10, use_symmetry=True)
    st_lin_nosym = vt.SolverSettings(solver_type="linear", n_panels=10, use_symmetry=False)

    r_lin_sym = vt.LinearLLTSolver().solve(ac, cond, st_lin_sym)
    r_lin_nosym = vt.LinearLLTSolver().solve(ac, cond, st_lin_nosym)

    print(f"Aileron Linear: Cl={r_lin_sym.totals.Cl:.6f}, CL={r_lin_sym.totals.CL:.6f}")
    assert r_lin_sym.totals.Cl < 0.0, f"Linear aileron must give Cl < 0, got {r_lin_sym.totals.Cl}"

    for coef in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
        vs = getattr(r_lin_sym.totals, coef)
        vns = getattr(r_lin_nosym.totals, coef)
        if abs(vns) > 1e-12:
            assert abs(vs - vns) / abs(vns) <= 1e-12, f"Linear symmetry fold mismatch on {coef}"
        else:
            assert abs(vs - vns) <= 1e-14, f"Linear symmetry fold mismatch on near-zero {coef}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Rudder signs
# ═══════════════════════════════════════════════════════════════════════════════

def test_rudder_signs():
    """Prove that positive rudder deflection produces negative side force CY and positive yawing moment Cn."""
    lattice_cache.clear()
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        is_symmetric=True,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        n_panels=8,
    )
    fin = vt.LiftingSurface(
        name="fin",
        semi_span=2.0,
        is_symmetric=False,
        dihedral=np.pi / 2.0,
        position=np.array([4.0, 0.0, 0.0]),
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.5)],
        controls=[vt.ControlSurface(name="rudder", eta_start=0.0, eta_end=1.0, hinge_x_c=0.7, deflection=np.radians(5.0))],
        n_panels=6,
    )
    ac = vt.Aircraft(surfaces=[wing, fin], ref_point=np.zeros(3))
    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0, beta=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=4)

    res = vt.VortexLatticeSolver().solve(ac, cond, st)
    print(f"\nRudder VLM: CY={res.totals.CY:.6f}, Cn={res.totals.Cn:.6f}")
    assert res.totals.CY < 0.0, f"Rudder must produce CY < 0, got {res.totals.CY}"
    assert res.totals.Cn > 0.0, f"Rudder must produce Cn > 0, got {res.totals.Cn}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Twin rudders with mirror copy
# ═══════════════════════════════════════════════════════════════════════════════

def test_twin_rudders_with_mirror_copy():
    """Prove that twin rudders with mirrored copy roughly double the yaw/side force response."""
    lattice_cache.clear()
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        is_symmetric=True,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        n_panels=8,
    )
    fin_right = vt.LiftingSurface(
        name="fin_r",
        semi_span=2.0,
        is_symmetric=False,
        dihedral=np.pi / 2.0,
        position=np.array([4.0, 1.5, 0.0]),
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 0.6)],
        controls=[vt.ControlSurface(name="rudder", eta_start=0.0, eta_end=1.0, hinge_x_c=0.7, deflection=np.radians(5.0), symmetric=False)],
        n_panels=6,
    )

    cond = vt.FlightCondition(V_inf=30.0, alpha=0.0, beta=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=2)

    # 1. Single fin with deflected vs undeflected rudder
    ac_one_def = vt.Aircraft(surfaces=[wing, fin_right])
    ac_one_base = vt.Aircraft(surfaces=[wing, fin_right.clone()])
    ac_one_base.surfaces[1].controls[0].deflection = 0.0

    cy_one_def = vt.VortexLatticeSolver().solve(ac_one_def, cond, st).totals.CY
    cy_one_base = vt.VortexLatticeSolver().solve(ac_one_base, cond, st).totals.CY
    delta_cy_one = cy_one_def - cy_one_base

    # 2. Twin fins
    fin_left = fin_right.mirrored(name="fin_l")
    ac_twin_def = vt.Aircraft(surfaces=[wing, fin_right, fin_left])

    ac_twin_base = vt.Aircraft(surfaces=[wing, fin_right.clone(), fin_left.clone()])
    ac_twin_base.surfaces[1].controls[0].deflection = 0.0
    ac_twin_base.surfaces[2].controls[0].deflection = 0.0

    cy_twin_def = vt.VortexLatticeSolver().solve(ac_twin_def, cond, st).totals.CY
    cy_twin_base = vt.VortexLatticeSolver().solve(ac_twin_base, cond, st).totals.CY
    delta_cy_twin = cy_twin_def - cy_twin_base

    ratio = delta_cy_twin / delta_cy_one
    print(f"\nTwin rudder CY change: single={delta_cy_one:.6f}, twin={delta_cy_twin:.6f}, ratio={ratio:.3f}")

    assert np.sign(delta_cy_twin) == np.sign(delta_cy_one), "Twin rudders must maintain sign"
    assert 1.5 <= ratio <= 2.5, f"Twin rudder ratio {ratio:.3f} outside [1.5, 2.5]"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Span limits snap to strip edges
# ═══════════════════════════════════════════════════════════════════════════════

def test_span_limits_snap_to_strip_edges():
    """Prove that control span limits snap to strip edges and preserve existing section break edges."""
    lattice_cache.clear()
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.43, eta_end=0.77)],
        n_panels=20,
        spacing="uniform",
    )
    ac = vt.Aircraft(surfaces=[wing])
    lat = vt.build_lattice(ac)

    assert len(lat.control_info) == 1
    info = lat.control_info[0]
    assert abs(info["eta_start_eff"] - 0.43) <= 1e-12, f"eta_start_eff {info['eta_start_eff']} not 0.43"
    assert abs(info["eta_end_eff"] - 0.77) <= 1e-12, f"eta_end_eff {info['eta_end_eff']} not 0.77"

    # Section break preservation test
    wing_kink = vt.LiftingSurface(
        name="kink_wing",
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 2.0), vt.WingSection(0.5, 1.5), vt.WingSection(1.0, 1.0)],
        n_panels=20,
        spacing="uniform",
    )
    lat_kink_clean = vt.build_lattice(vt.Aircraft(surfaces=[wing_kink]))

    wing_kink_ctrl = wing_kink.clone()
    wing_kink_ctrl.controls = [vt.ControlSurface(name="ctrl", eta_start=0.22, eta_end=0.44)]
    lat_kink_ctrl = vt.build_lattice(vt.Aircraft(surfaces=[wing_kink_ctrl]))

    # Kink is at eta=0.5
    # Check that an edge exists at 0.5 in both lattices at the exact same location
    edge_eta_clean = lat_kink_clean.surfaces[0].edge_le[:, 1] / wing_kink.semi_span
    edge_eta_ctrl = lat_kink_ctrl.surfaces[0].edge_le[:, 1] / wing_kink.semi_span
    assert np.any(np.isclose(edge_eta_clean, 0.5, atol=1e-12))
    assert np.any(np.isclose(edge_eta_ctrl, 0.5, atol=1e-12))
    kink_clean = edge_eta_clean[np.argmin(np.abs(edge_eta_clean - 0.5))]
    kink_ctrl = edge_eta_ctrl[np.argmin(np.abs(edge_eta_ctrl - 0.5))]
    assert abs(kink_clean - kink_ctrl) <= 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Hinge snaps on control strips only
# ═══════════════════════════════════════════════════════════════════════════════

def test_hinge_snaps_on_control_strips_only():
    """Prove that interior chordwise hinge snapping applies exclusively to strips on the control."""
    lattice_cache.clear()
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.8, hinge_x_c=0.7)],
        n_panels=10,
        spacing="uniform",
    )
    ac = vt.Aircraft(surfaces=[wing])
    lat = vt.build_lattice(ac, n_chord=4, chord_spacing="uniform")

    info = lat.control_info[0]
    assert info["hinge_x_c_eff"] == 0.7

    # Find strips on the control and strips outside
    ctrl_strips = [s for s in range(lat.n_strips) if 0.2 - 1e-9 <= abs(lat.eta[s]) <= 0.8 + 1e-9]
    other_strips = [s for s in range(lat.n_strips) if s not in ctrl_strips]

    # For control strips, panel lengths of chord=1.0 at k=2 and k=3 are 0.2 and 0.3
    for s in ctrl_strips:
        p_len_k2 = lat.panel_length[s * 4 + 2]
        p_len_k3 = lat.panel_length[s * 4 + 3]
        assert abs(p_len_k2 - 0.2) <= 1e-12, f"Strip {s} k=2 panel length {p_len_k2} != 0.2"
        assert abs(p_len_k3 - 0.3) <= 1e-12, f"Strip {s} k=3 panel length {p_len_k3} != 0.3"

    for s in other_strips:
        for k in range(4):
            p_len = lat.panel_length[s * 4 + k]
            assert abs(p_len - 0.25) <= 1e-12, f"Non-control strip {s} k={k} panel length {p_len} != 0.25"

    # Edge length rule rejection: n_chord=2, hinge_x_c=0.97
    wing2 = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.97)],
        n_panels=8,
    )
    lat2 = vt.build_lattice(vt.Aircraft(surfaces=[wing2]), n_chord=2)
    assert lat2.control_info[0]["hinge_x_c_eff"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Cache key follows every control field
# ═══════════════════════════════════════════════════════════════════════════════

def test_cache_key_follows_every_control_field():
    """Prove that every public field of a ControlSurface changes the cache fingerprint."""
    wing = vt.LiftingSurface(
        semi_span=4.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", eta_start=0.2, eta_end=0.8, hinge_x_c=0.75, deflection=0.05, symmetric=True)],
        n_panels=8,
    )
    ac = vt.Aircraft(surfaces=[wing])

    fp_base = lattice_cache.surfaces_fingerprint(ac)

    # 1. name
    wing.controls[0].name = "aileron"
    fp_name = lattice_cache.surfaces_fingerprint(ac)
    assert fp_name != fp_base
    wing.controls[0].name = "flap"
    assert lattice_cache.surfaces_fingerprint(ac) == fp_base

    # 2. eta_start
    wing.controls[0].eta_start = 0.1
    assert lattice_cache.surfaces_fingerprint(ac) != fp_base
    wing.controls[0].eta_start = 0.2
    assert lattice_cache.surfaces_fingerprint(ac) == fp_base

    # 3. eta_end
    wing.controls[0].eta_end = 0.9
    assert lattice_cache.surfaces_fingerprint(ac) != fp_base
    wing.controls[0].eta_end = 0.8
    assert lattice_cache.surfaces_fingerprint(ac) == fp_base

    # 4. hinge_x_c
    wing.controls[0].hinge_x_c = 0.70
    assert lattice_cache.surfaces_fingerprint(ac) != fp_base
    wing.controls[0].hinge_x_c = 0.75
    assert lattice_cache.surfaces_fingerprint(ac) == fp_base

    # 5. deflection
    wing.controls[0].deflection = 0.10
    assert lattice_cache.surfaces_fingerprint(ac) != fp_base
    wing.controls[0].deflection = 0.05
    assert lattice_cache.surfaces_fingerprint(ac) == fp_base

    # 6. symmetric
    wing.controls[0].symmetric = False
    assert lattice_cache.surfaces_fingerprint(ac) != fp_base
    wing.controls[0].symmetric = True
    assert lattice_cache.surfaces_fingerprint(ac) == fp_base


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Aircraft.set_deflection
# ═══════════════════════════════════════════════════════════════════════════════

def test_set_deflection_links_controls_by_name():
    """Prove that Aircraft.set_deflection updates all controls with matching name across surfaces."""
    s1 = vt.LiftingSurface(name="tail_r", controls=[vt.ControlSurface(name="elevator", deflection=0.0)])
    s2 = vt.LiftingSurface(name="tail_l", controls=[vt.ControlSurface(name="elevator", deflection=0.0)])
    ac = vt.Aircraft(surfaces=[s1, s2])

    assert ac.control_names() == ["elevator"]
    count = ac.set_deflection("elevator", np.radians(2.5))
    assert count == 2
    assert abs(s1.controls[0].deflection - np.radians(2.5)) <= 1e-12
    assert abs(s2.controls[0].deflection - np.radians(2.5)) <= 1e-12

    with pytest.raises(ValueError, match="elevator"):
        ac.set_deflection("rudder", 0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Invalid controls are refused
# ═══════════════════════════════════════════════════════════════════════════════

def test_invalid_controls_are_refused():
    """Prove that validate_controls catches and refuses each rule with the field name in the error."""
    # 1. Empty name
    s = vt.LiftingSurface(controls=[vt.ControlSurface(name="")])
    with pytest.raises(ValueError, match="name"):
        ctrl_mod.validate_controls(s)

    # 2. Non-finite value
    s = vt.LiftingSurface(controls=[vt.ControlSurface(deflection=np.nan)])
    with pytest.raises(ValueError, match="deflection"):
        ctrl_mod.validate_controls(s)

    # 3. Span limits
    s = vt.LiftingSurface(controls=[vt.ControlSurface(eta_start=-0.1)])
    with pytest.raises(ValueError, match="eta_start"):
        ctrl_mod.validate_controls(s)

    s = vt.LiftingSurface(controls=[vt.ControlSurface(eta_start=0.8, eta_end=0.5)])
    with pytest.raises(ValueError, match="eta_start|eta_end"):
        ctrl_mod.validate_controls(s)

    s = vt.LiftingSurface(controls=[vt.ControlSurface(eta_end=1.1)])
    with pytest.raises(ValueError, match="eta_end"):
        ctrl_mod.validate_controls(s)

    # 4. Hinge position
    s = vt.LiftingSurface(controls=[vt.ControlSurface(hinge_x_c=0.0)])
    with pytest.raises(ValueError, match="hinge_x_c"):
        ctrl_mod.validate_controls(s)

    s = vt.LiftingSurface(controls=[vt.ControlSurface(hinge_x_c=1.0)])
    with pytest.raises(ValueError, match="hinge_x_c"):
        ctrl_mod.validate_controls(s)

    # 5. Deflection > 30 deg
    s = vt.LiftingSurface(controls=[vt.ControlSurface(deflection=np.radians(35.0))])
    with pytest.raises(ValueError, match="deflection"):
        ctrl_mod.validate_controls(s)

    # 6. Overlapping controls
    s = vt.LiftingSurface(controls=[
        vt.ControlSurface(name="c1", eta_start=0.2, eta_end=0.6),
        vt.ControlSurface(name="c2", eta_start=0.5, eta_end=0.9),
    ])
    with pytest.raises(ValueError, match="overlap|eta_start|eta_end"):
        ctrl_mod.validate_controls(s)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Fourier solver refuses deflection
# ═══════════════════════════════════════════════════════════════════════════════

def test_fourier_refuses_deflection():
    """Prove that the classical Fourier solver refuses non-zero control deflections."""
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", deflection=0.0)],
        n_panels=8,
    )
    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0))
    st = vt.SolverSettings(solver_type="fourier", n_panels=8)

    # Zero deflection runs successfully
    res = vt.FourierSolver().solve(wing, cond, st)
    assert np.isfinite(res.totals.CL)

    # Deflection != 0 raises ValueError naming other solvers
    wing.controls[0].deflection = np.radians(2.0)
    with pytest.raises(ValueError, match="vlm|linear|nonlinear"):
        vt.FourierSolver().solve(wing, cond, st)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. JSON round-trip
# ═══════════════════════════════════════════════════════════════════════════════

def test_json_round_trip_keeps_controls():
    """Prove that aircraft JSON serialization and deserialization preserves control surface configurations."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=6.0,
        sections=[vt.WingSection(0.0, 1.5), vt.WingSection(1.0, 0.8)],
        controls=[
            vt.ControlSurface(name="flap", eta_start=0.1, eta_end=0.5, hinge_x_c=0.75, deflection=0.1, symmetric=True),
            vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=0.95, hinge_x_c=0.70, deflection=-0.05, symmetric=False),
        ],
    )
    ac = vt.Aircraft(name="test_plane", surfaces=[wing])

    json_str = vt.aircraft_to_json(ac)
    ac_loaded = vt.aircraft_from_json(json_str)

    assert len(ac_loaded.surfaces[0].controls) == 2
    for orig, loaded in zip(wing.controls, ac_loaded.surfaces[0].controls):
        assert orig.name == loaded.name
        assert abs(orig.eta_start - loaded.eta_start) <= 1e-12
        assert abs(orig.eta_end - loaded.eta_end) <= 1e-12
        assert abs(orig.hinge_x_c - loaded.hinge_x_c) <= 1e-12
        assert abs(orig.deflection - loaded.deflection) <= 1e-12
        assert orig.symmetric == loaded.symmetric


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Control in ground effect runs
# ═══════════════════════════════════════════════════════════════════════════════

def test_control_in_ground_effect_runs():
    """Prove that VLM analysis with a deflected flap converges to finite results in ground effect."""
    wing_0 = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", deflection=0.0)],
        n_panels=8,
    )
    wing_def = vt.LiftingSurface(
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.0), vt.WingSection(1.0, 1.0)],
        controls=[vt.ControlSurface(name="flap", deflection=np.radians(4.0))],
        n_panels=8,
    )

    cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(2.0), h=1.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=4)

    r0 = vt.VortexLatticeSolver().solve(wing_0, cond, st)
    rdef = vt.VortexLatticeSolver().solve(wing_def, cond, st)

    assert np.isfinite(rdef.totals.CL)
    assert np.isfinite(rdef.totals.CDi)
    assert rdef.totals.CL > r0.totals.CL, "Deflected flap must increase CL in ground effect"


# ═══════════════════════════════════════════════════════════════════════════════
# 17. GPU parity test
# ═══════════════════════════════════════════════════════════════════════════════

def _need_gpu():
    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")


@pytest.fixture
def gpu_device():
    _need_gpu()
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("solver_type", ["vlm", "linear", "nonlinear"])
@pytest.mark.parametrize("control_type", ["flap", "aileron"])
def test_gpu_control_surfaces_equal_cpu(gpu_device, precision, solver_type, control_type):
    """Prove that GPU sweep results with deflected controls match CPU results within precision tolerance."""
    tol = 1e-11 if precision == "float64" else 2e-5

    af = vt.LinearAirfoil(Cd0=0.008, Cm0=-0.04) if solver_type != "nonlinear" else vt.TabulatedAirfoil(
        alpha=np.radians(np.arange(-10.0, 20.1, 2.0)),
        Cl_data=2.0 * np.pi * np.radians(np.arange(-10.0, 20.1, 2.0)),
        Cd_data=0.008 + 0.01 * (np.radians(np.arange(-10.0, 20.1, 2.0)) / np.radians(10)) ** 2,
        Cm_data=-0.04 + 0.0 * np.radians(np.arange(-10.0, 20.1, 2.0)),
    )

    if control_type == "flap":
        ctrl = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(4.0), symmetric=True)
    else:
        ctrl = vt.ControlSurface(name="aileron", eta_start=0.5, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(4.0), symmetric=False)

    wing = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[vt.WingSection(0.0, 1.2, airfoil=af), vt.WingSection(1.0, 0.8, airfoil=af)],
        controls=[ctrl],
        n_panels=8,
    )
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()

    solver_cls = {
        "vlm": vt.VortexLatticeSolver,
        "linear": vt.LinearLLTSolver,
        "nonlinear": vt.NonlinearSolver,
    }[solver_type]
    solver = solver_cls()

    st = vt.SolverSettings(solver_type=solver_type, n_panels=8, n_chord=4 if solver_type == "vlm" else 1)
    alphas = np.radians(np.linspace(-4.0, 8.0, 9))
    cond = vt.FlightCondition(V_inf=30.0)

    gpu.set_device("cpu")
    gpu.set_precision(precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_cpu = solver.solve_sweep(ac, cond, st, alphas)

    gpu.set_device("gpu")
    gpu.set_precision(precision)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_gpu = solver.solve_sweep(ac, cond, st, alphas)

    cl_scale = max(float(np.max(np.abs([r.totals.CL for r in r_cpu]))), 1e-3)
    for name in ("CL", "CDi", "Cm", "Cl", "Cn", "CY"):
        a = np.array([getattr(r.totals, name) or 0.0 for r in r_cpu])
        b = np.array([getattr(r.totals, name) or 0.0 for r in r_gpu])
        scale = max(float(np.max(np.abs(a))), cl_scale if name not in ("CDi",) else 1e-3)
        diff = np.max(np.abs(a - b))
        assert diff <= tol * scale, f"{solver_type} {control_type} {precision} {name}: diff {diff:.3e} > {tol * scale:.3e}"
