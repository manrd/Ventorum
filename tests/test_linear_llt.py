# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Tests of the linear lifting line (LinearLLTSolver, Phillips and Snyder).

They check the section data (a0, alpha_L0, Cd0), sweep, dihedral and
sideslip, a wing with a tail, ground effect, the symmetry fold, sweeps
with a prebuilt system, tabulated airfoils, the top-level API and
independent instances. The tight checks against theory are in
test_analytical_verification.py.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.solvers.linear import LinearLLTSolver, LinearSolver


def test_linear_llt_elliptic_wing_exact():
    """Verify LinearLLTSolver matches the exact Lanchester–Prandtl lifting-line result on elliptic wings."""
    AR = 8.0
    b = 12.0
    S = b ** 2 / AR
    c0 = 4.0 * S / (np.pi * b)

    # Elliptic planform with a straight quarter-chord line (the lifting line of
    # classical theory). With a straight leading edge the quarter-chord line is
    # curved, which acts like local sweep.
    y_fracs = np.linspace(0.0, 1.0, 201)
    chords = c0 * np.sqrt(np.maximum(0.0, 1.0 - y_fracs ** 2))
    chords[-1] = 1e-3 * c0  # regularize tip

    sections = [
        vt.WingSection(y_frac=yf, chord=c, x_le=-0.25 * c,
                        airfoil=vt.LinearAirfoil(a0=2.0 * np.pi, alpha_L0=0.0))
        for yf, c in zip(y_fracs, chords)
    ]
    wing = vt.LiftingSurface(semi_span=b / 2.0, sections=sections)
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()

    alpha_deg = 5.0
    alpha_rad = np.radians(alpha_deg)
    cond = vt.FlightCondition(V_inf=45.0, alpha=alpha_rad)
    sett = vt.SolverSettings(solver_type="linear", n_panels=40)

    res = LinearLLTSolver().solve(ac, cond, sett)

    # Exact Lanchester–Prandtl values (with the aspect ratio of the discretised planform)
    AR_geo = ac.b_ref ** 2 / ac.S_ref
    a0 = 2.0 * np.pi
    CL_exact = a0 * np.sin(alpha_rad) / (1.0 + a0 / (np.pi * AR_geo))
    CDi_exact = CL_exact ** 2 / (np.pi * AR_geo)

    # FourierSolver analytical comparison
    res_fourier = vt.FourierSolver().solve(ac, cond, sett)

    assert abs(res.totals.CL - CL_exact) / CL_exact < 0.003
    assert abs(res.totals.CDi - CDi_exact) / CDi_exact < 0.006
    assert abs(res.totals.CL - res_fourier.totals.CL) / res_fourier.totals.CL < 0.003
    assert abs(res.totals.e - 1.0) < 0.003
    assert res.solver_type == "linear"
    assert res.converged is True


def test_linear_llt_airfoil_properties():
    """Verify that arbitrary section lift-curve slope and zero-lift angle are respected."""
    a0_val = 5.85
    alpha_L0_deg = -3.5
    alpha_L0_rad = np.radians(alpha_L0_deg)
    afoil = vt.LinearAirfoil(a0=a0_val, alpha_L0=alpha_L0_rad, Cd0=0.008)

    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            # Straight, unswept quarter-chord line: the Fourier solver
            # refuses a swept one.
            vt.WingSection(y_frac=0.0, chord=1.2, airfoil=afoil, x_le=-0.25 * 1.2),
            vt.WingSection(y_frac=1.0, chord=0.8, airfoil=afoil, x_le=-0.25 * 0.8),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(2.0))
    sett = vt.SolverSettings(solver_type="linear", n_panels=40, spacing="cosine")

    res_linear = LinearSolver().solve(ac, cond, sett)
    res_fourier = vt.FourierSolver().solve(ac, cond, sett)

    # Same wing, same section data: CL and CDi agree with the Fourier
    # solution (measured: 0.035 % and 0.034 %).
    assert abs(res_linear.totals.CL - res_fourier.totals.CL) / res_fourier.totals.CL < 0.004
    assert abs(res_linear.totals.CDi - res_fourier.totals.CDi) / res_fourier.totals.CDi < 0.004
    # Profile drag should be accounted for from Cd0
    assert res_linear.totals.CDp is not None
    assert np.isclose(res_linear.totals.CDp, 0.008, rtol=0.05)


def test_linear_llt_swept_wing():
    """With sweep the lifting line is not grid convergent: it must say so.

    The vortex lattice handles sweep; its swept/straight lift-slope ratio is
    compared with the Helmbold-Diederich (DATCOM) estimate.
    """
    sweep = np.radians(45.0)
    wing_swept = vt.LiftingSurface(
        semi_span=2.5,
        sweep_le=sweep,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    wing_straight = vt.LiftingSurface(
        semi_span=2.5,
        sweep_le=0.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )

    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(4.0))
    sett = vt.SolverSettings(solver_type="linear", n_panels=30)

    with pytest.warns(RuntimeWarning, match="not grid convergent"):
        res_swept = LinearLLTSolver().solve(wing_swept, cond, sett)
    assert res_swept.totals.trust.factors["sweep"] > 0.5

    def hd(AR, lam_half):  # Helmbold-Diederich, a0 = 2 pi
        return 2 * np.pi * AR / (2 + np.sqrt(AR * AR * (1 + np.tan(lam_half) ** 2) + 4))

    vlm = vt.SolverSettings(solver_type="vlm", n_panels=30)
    r_sw = vt.analyze(wing_swept, settings=vlm, alpha_deg=4.0).totals.CL
    r_st = vt.analyze(wing_straight, settings=vlm, alpha_deg=4.0).totals.CL
    ratio_ref = hd(5.0, sweep) / hd(5.0, 0.0)
    assert abs((r_sw / r_st) / ratio_ref - 1.0) < 0.03


def test_linear_llt_dihedral_and_sideslip():
    """Verify dihedral lateral stability derivative Cl_beta is negative (restoring roll)."""
    dih = np.radians(10.0)
    wing = vt.LiftingSurface(
        semi_span=3.285,
        dihedral=dih,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    sett = vt.SolverSettings(solver_type="linear", n_panels=25, use_symmetry=False)

    cond_sym = vt.FlightCondition(V_inf=50.0, alpha=np.radians(2.0), beta=0.0)
    cond_beta = vt.FlightCondition(V_inf=50.0, alpha=np.radians(2.0), beta=np.radians(5.0))

    res_sym = LinearLLTSolver().solve(ac, cond_sym, sett)
    res_beta = LinearLLTSolver().solve(ac, cond_beta, sett)

    # Symmetry at beta = 0: measured |Cl| = 7.5e-18 today, so the tolerance is the
    # floor of decision 1 (1e-12).
    assert abs(res_sym.totals.Cl) < 1e-12
    # Positive sideslip (wind from starboard) on a dihedral wing gives a restoring
    # left roll: the right wing rises, so Cl < 0 with the AGENTS.md convention
    # (Cl > 0 right wing down). Measured Cl = -0.0136 today.
    assert res_beta.totals.Cl < -0.005


def test_linear_llt_multi_surface_downwash():
    """Verify multi-surface coupling: tail in wing wake experiences downwash."""
    wing = vt.LiftingSurface(
        name="MainWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=1.0, chord=0.8),
        ],
    )
    # Tail positioned 3.5 m aft along x
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=1.8,
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.6, x_le=3.5),
            vt.WingSection(y_frac=1.0, chord=0.4, x_le=3.5),
        ],
    )
    ac = vt.Aircraft(name="WingTail", surfaces=[wing, tail])
    cond = vt.FlightCondition(V_inf=40.0, alpha=np.radians(3.0))
    sett = vt.SolverSettings(solver_type="linear", n_panels=25)

    res = LinearLLTSolver().solve(ac, cond, sett)
    assert len(res.spanwise) == 2
    # Both surfaces produce positive lift
    assert res.spanwise[0].local_lift.sum() > 0
    assert res.spanwise[1].local_lift.sum() > 0
    assert res.totals.CL > 0


def test_linear_llt_ground_effect():
    """Verify ground effect increases lift and decreases induced drag."""
    wing = vt.LiftingSurface(
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    sett = vt.SolverSettings(solver_type="linear", n_panels=25)

    cond_free = vt.FlightCondition(V_inf=45.0, alpha=np.radians(4.0), h=None)
    cond_ge = vt.FlightCondition(V_inf=45.0, alpha=np.radians(4.0), h=1.5)
    cond_low = vt.FlightCondition(V_inf=45.0, alpha=np.radians(4.0), h=0.4)

    res_free = LinearLLTSolver().solve(ac, cond_free, sett)
    with pytest.warns(RuntimeWarning, match="under-predicts"):
        res_ge = LinearLLTSolver().solve(ac, cond_ge, sett)

    assert res_ge.totals.CL > res_free.totals.CL
    assert res_ge.totals.CDi < res_free.totals.CDi
    assert res_ge.totals.e > res_free.totals.e
    # Below h_min/c = 1 the lifting line is refused.
    with pytest.raises(ValueError, match="not valid in ground effect"):
        LinearLLTSolver().solve(ac, cond_low, sett)


def test_linear_llt_symmetry_plane_equivalence():
    """Verify half-mesh symmetry plane produces identical results to full mesh (< 1e-12)."""
    wing = vt.LiftingSurface(
        semi_span=6.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.6, airfoil=vt.LinearAirfoil(a0=5.9, alpha_L0=np.radians(-2.0))),
            vt.WingSection(y_frac=1.0, chord=0.8, airfoil=vt.LinearAirfoil(a0=6.0, alpha_L0=np.radians(-1.0))),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=55.0, alpha=np.radians(3.5))

    sett_sym = vt.SolverSettings(solver_type="linear", n_panels=30, use_symmetry=True)
    sett_full = vt.SolverSettings(solver_type="linear", n_panels=30, use_symmetry=False)

    res_sym = LinearLLTSolver().solve(ac, cond, sett_sym)
    res_full = LinearLLTSolver().solve(ac, cond, sett_full)

    np.testing.assert_allclose(res_sym.totals.CL, res_full.totals.CL, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(res_sym.totals.CDi, res_full.totals.CDi, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(res_sym.totals.Cm, res_full.totals.Cm, rtol=1e-11, atol=1e-12)
    assert res_sym.symmetry_used is True
    assert res_full.symmetry_used is False


def test_linear_llt_sweep_precomputed_equivalence():
    """Verify LinearLLTSolver.solve_sweep matches serial individual solves."""
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.4, twist=0.0),
            vt.WingSection(y_frac=1.0, chord=0.7, twist=np.radians(-2.0)),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=48.0)
    sett = vt.SolverSettings(solver_type="linear", n_panels=25)
    alphas = np.radians(np.linspace(-2.0, 8.0, 6))

    solver = LinearLLTSolver()
    res_sweep = solver.solve_sweep(ac, cond, sett, alphas)

    for a_rad, res_fast in zip(alphas, res_sweep):
        c = vt.FlightCondition(V_inf=48.0, alpha=a_rad)
        res_serial = solver.solve(ac, c, sett)
        np.testing.assert_allclose(res_fast.totals.CL, res_serial.totals.CL, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(res_fast.totals.CDi, res_serial.totals.CDi, rtol=1e-10, atol=1e-12)


def test_linear_llt_tabulated_airfoil_extraction():
    """Verify TabulatedAirfoil linear property extraction in LinearLLTSolver."""
    alpha_polar = np.radians(np.linspace(-10.0, 15.0, 26))
    cl_polar = 2.0 * np.pi * (alpha_polar - np.radians(-2.0))
    cd_polar = 0.006 + 0.02 * cl_polar ** 2
    tab_af = vt.TabulatedAirfoil(name="Synthetic", alpha=alpha_polar, Cl_data=cl_polar, Cd_data=cd_polar)

    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, airfoil=tab_af),
            vt.WingSection(y_frac=1.0, chord=0.6, airfoil=tab_af),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=50.0, alpha=np.radians(2.0))
    sett = vt.SolverSettings(solver_type="linear", n_panels=25)

    res = LinearLLTSolver().solve(ac, cond, sett)
    assert res.totals.CL > 0.3
    assert res.totals.CDi > 0.0


def test_linear_llt_top_level_analyze_api():
    """Verify vt.analyze() works seamlessly with solver='linear' and solver='linear_llt'."""
    wing = vt.LiftingSurface(
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=0.8),
        ],
    )
    res1 = vt.analyze(wing, alpha_deg=3.0, solver="linear", n_panels=20)
    res2 = vt.analyze(wing, alpha_deg=3.0, solver="linear_llt", n_panels=20)

    assert res1.solver_type == "linear"
    assert res2.solver_type == "linear"
    np.testing.assert_allclose(res1.totals.CL, res2.totals.CL, rtol=1e-14)


def test_linear_llt_multi_instance():
    """Verify concurrent multi-instance Ventorum execution with solver='linear'."""
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=1.0, chord=0.6),
        ],
    )
    inst1 = vt.Ventorum(name="Case1", geometry=wing, solver="linear", alpha_deg=2.0)
    inst2 = vt.Ventorum(name="Case2", geometry=wing, solver="linear", alpha_deg=6.0)

    res1 = inst1.run()
    res2 = inst2.run()

    assert res1.solver_type == "linear"
    assert res2.solver_type == "linear"
    assert res2.totals.CL > res1.totals.CL
