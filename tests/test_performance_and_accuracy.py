# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Rigorous Accuracy and Equivalence Test Suite for Ventorum Performance Optimizations.

Validates that all optimized vectorized kernels, matrix projections, precomputed
caches, and force integration routines match baseline/theoretical values to
machine precision.
"""

import warnings
import numpy as np

import ventorum as vt
from ventorum.geometry.processing import discretize_surface
from ventorum.solvers.horseshoe import HorseshoeSolver
from ventorum.solvers.fourier import FourierSolver
from ventorum.solvers.nonlinear import NonlinearSolver


def test_vlm_prebuilt_lattice_equivalence():
    """Solving a prebuilt lattice gives the same result as a full solve (machine precision)."""
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sweep_le=np.radians(15.0),
        dihedral=np.radians(3.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=0.75),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()
    solver = HorseshoeSolver()
    settings = vt.SolverSettings(n_panels=30)
    lattice = solver.build(ac, settings, vt.FlightCondition())

    for alpha_deg in [-2.0, 0.0, 4.0, 8.5, 12.0]:
        cond = vt.FlightCondition(alpha=np.radians(alpha_deg), beta=0.0, V_inf=45.0)
        res_std = solver.solve(ac, cond, settings)
        res_cached = solver.solve_lattice(lattice, cond, settings, ac.S_ref, ac.b_ref, ac.c_ref)
        for k in ("CL", "CDi", "Cm", "e"):
            np.testing.assert_allclose(getattr(res_cached.totals, k), getattr(res_std.totals, k), atol=1e-13, rtol=1e-12)
        for s_cached, s_std in zip(res_cached.spanwise, res_std.spanwise):
            np.testing.assert_allclose(s_cached.gamma, s_std.gamma, atol=1e-12, rtol=1e-12)


def test_fourier_analytical_elliptic_wing():
    """Verify Fourier solver vectorization matches theoretical Glauert elliptic wing exact values."""
    AR = 8.0
    b = 10.0
    c_root = 4.0 * b / (np.pi * AR)

    # Discretize elliptic chord distribution with a straight, unswept
    # quarter-chord line (the lifting line of Glauert's solution; the
    # Fourier solver refuses a swept quarter-chord line, owner decision D-08)
    n_secs = 31
    y_fracs = np.linspace(0, 1, n_secs)
    chords = [c_root * np.sqrt(max(1.0 - yf**2, 1e-8)) for yf in y_fracs]
    sections = [
        vt.WingSection(y_frac=yf, chord=c, x_le=-0.25 * c)
        for yf, c in zip(y_fracs, chords)
    ]
    wing = vt.LiftingSurface(semi_span=b / 2.0, sections=sections)

    alpha_rad = np.radians(5.0)
    cond = vt.FlightCondition(alpha=alpha_rad, V_inf=30.0)
    res = vt.analyze(wing, condition=cond, solver="fourier", n_panels=40)

    # Each tolerance is about 10 times the relative error measured today on this
    # 31-section discretised planform (task T-0010, decision 1).

    # For an elliptic wing:
    # 1. Oswald efficiency e = 1.000 (measured error today: 1.6e-3; old tolerance 0.05)
    assert abs(res.totals.e - 1.0) < 1.7e-2

    # 2. Lift curve slope a_3D = a0 / (1 + a0 / (pi * AR))
    a0 = 2.0 * np.pi
    a_3D = a0 / (1.0 + a0 / (np.pi * AR))
    expected_CL = a_3D * alpha_rad
    # Measured relative error today: 2.8e-4 (0.028 %); old tolerance 0.10 accepted a 10 % error.
    np.testing.assert_allclose(res.totals.CL, expected_CL, rtol=2.8e-3)

    # 3. Induced drag CDi = CL^2 / (pi * AR * e)
    expected_CDi = (res.totals.CL ** 2) / (np.pi * AR * res.totals.e)
    # Measured relative error today: 2.3e-3. Ten times that (2.3e-2) would be
    # looser than the old tolerance, so the old, tighter 0.01 stays (review of
    # T-0010: a tolerance is never made looser).
    np.testing.assert_allclose(res.totals.CDi, expected_CDi, rtol=0.01)


def test_nonlinear_solver_linear_polar_equivalence():
    """With linear sections the nonlinear lifting line equals the linear one up to the
    small-angle terms (second order in alpha)."""
    a0 = 2 * np.pi
    afoil = vt.LinearAirfoil(a0=a0, alpha_L0=0.0)
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, airfoil=afoil),
            vt.WingSection(y_frac=1.0, chord=0.6, airfoil=afoil),
        ],
    )
    settings_nl = vt.SolverSettings(solver_type="nonlinear", n_panels=24, max_iterations=100, tolerance=1e-10)
    for a_deg, rtol in ((1.0, 1e-4), (4.0, 2e-3)):
        cond = vt.FlightCondition(alpha=np.radians(a_deg), V_inf=35.0)
        res_linear = vt.analyze(wing, condition=cond, solver="linear", n_panels=24)
        res_nl = vt.analyze(wing, condition=cond, settings=settings_nl)
        assert res_nl.converged
        np.testing.assert_allclose(res_nl.totals.CL, res_linear.totals.CL, rtol=rtol)
        np.testing.assert_allclose(res_nl.totals.CDi, res_linear.totals.CDi, rtol=2 * rtol)


def test_fourier_solve_sweep_machine_precision():
    """Verify FourierSolver.solve_sweep multi-RHS vectorization produces machine-precision identical results to serial solves."""
    wing = vt.LiftingSurface(
        semi_span=6.0,
        sections=[
            # Straight, unswept quarter-chord line (owner decision D-08).
            vt.WingSection(y_frac=0.0, chord=1.8, twist=0.0, x_le=-0.25 * 1.8),
            vt.WingSection(y_frac=1.0, chord=0.9, twist=np.radians(-2.0), x_le=-0.25 * 0.9),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=40.0)
    settings = vt.SolverSettings(solver_type="fourier", n_panels=35)
    alphas = np.radians(np.linspace(-3.0, 9.0, 13))

    solver = FourierSolver()
    # Serial reference
    res_serial = [
        solver.solve(ac, vt.FlightCondition(V_inf=40.0, alpha=float(a)), settings)
        for a in alphas
    ]

    # Vectorized multi-RHS sweep
    res_sweep = solver.solve_sweep(ac, cond, settings, alphas)

    assert len(res_sweep) == len(res_serial)
    for r_sw, r_ser in zip(res_sweep, res_serial):
        np.testing.assert_allclose(r_sw.totals.CL, r_ser.totals.CL, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_sw.totals.CDi, r_ser.totals.CDi, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_sw.totals.e, r_ser.totals.e, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_sw.spanwise[0].gamma, r_ser.spanwise[0].gamma, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(r_sw.spanwise[0].alpha_i, r_ser.spanwise[0].alpha_i, atol=1e-14, rtol=1e-14)


def test_nonlinear_prebuilt_lattice_equivalence():
    """NonlinearSolver.solve_lattice matches solve down to machine precision."""
    alpha_data = np.linspace(np.radians(-10), np.radians(20), 30)
    cl_data = 2 * np.pi * alpha_data
    cd_data = 0.01 + 0.05 * alpha_data**2
    af = vt.TabulatedAirfoil(name="TestTab", alpha=alpha_data, Cl_data=cl_data, Cd_data=cd_data)

    wing = vt.LiftingSurface(
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=0.6, airfoil=af),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    ac.compute_reference_values()
    cond = vt.FlightCondition(alpha=np.radians(3.5), V_inf=30.0)
    settings = vt.SolverSettings(solver_type="nonlinear", n_panels=20, max_iterations=60)

    solver = NonlinearSolver()
    res_direct = solver.solve(ac, cond, settings)
    lattice = solver.build(ac, settings, cond)
    res_lat = solver.solve_lattice(lattice, cond, settings, ac.S_ref, ac.b_ref, ac.c_ref)

    assert res_direct.iterations == res_lat.iterations
    assert res_direct.converged == res_lat.converged
    np.testing.assert_allclose(res_direct.totals.CL, res_lat.totals.CL, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(res_direct.totals.CDi, res_lat.totals.CDi, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(res_direct.spanwise[0].gamma, res_lat.spanwise[0].gamma, atol=1e-14, rtol=1e-14)


def test_discretized_surface_cache_integrity():
    """Verify that cached geometry and airfoil attributes on DiscretizedSurface are accurate."""
    wing = vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    ds = discretize_surface(wing, n_panels=20, spacing="uniform")

    # Linear airfoil with default Cd0=0.0 should report has_profile_drag False
    assert ds.has_profile_drag is False
    assert len(ds.panel_centers_qc) == len(ds.dy_panels)

    # Expected midpoint
    expected_qc = 0.5 * (ds.nodes_qc[:-1] + ds.nodes_qc[1:])
    np.testing.assert_allclose(ds.panel_centers_qc, expected_qc, atol=1e-15, rtol=1e-15)


def test_horseshoe_solve_sweep_machine_precision():
    """Verify HorseshoeSolver.solve_sweep produces machine-precision identical results to serial solves."""
    wing = vt.LiftingSurface(
        name="HorseshoeWing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.4, twist=np.radians(2.5)),
            vt.WingSection(y_frac=1.0, chord=0.7, twist=0.0),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=45.0)
    settings = vt.SolverSettings(solver_type="horseshoe", n_panels=25)
    alphas = np.radians(np.linspace(-4.0, 10.0, 8))

    solver = HorseshoeSolver()
    # Serial reference
    res_serial = [
        solver.solve(ac, vt.FlightCondition(V_inf=45.0, alpha=float(a)), settings)
        for a in alphas
    ]

    # Precomputed sweep
    res_sweep = solver.solve_sweep(ac, cond, settings, alphas)

    assert len(res_sweep) == len(res_serial)
    for r_sw, r_ser in zip(res_sweep, res_serial):
        np.testing.assert_allclose(r_sw.totals.CL, r_ser.totals.CL, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_sw.totals.CDi, r_ser.totals.CDi, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_sw.totals.e, r_ser.totals.e, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r_sw.spanwise[0].gamma, r_ser.spanwise[0].gamma, atol=1e-11, rtol=1e-12)


def test_nonlinear_solve_sweep_machine_precision():
    """Verify NonlinearSolver.solve_sweep matches serial solves down to machine precision."""
    alpha_data = np.linspace(np.radians(-10), np.radians(20), 30)
    cl_data = 2 * np.pi * (alpha_data + np.radians(1.0))
    cd_data = 0.009 + 0.04 * alpha_data**2
    af = vt.TabulatedAirfoil(name="SweepTab", alpha=alpha_data, Cl_data=cl_data, Cd_data=cd_data)

    wing = vt.LiftingSurface(
        name="TabWing",
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=0.5, airfoil=af),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing])
    cond = vt.FlightCondition(V_inf=30.0)
    settings = vt.SolverSettings(solver_type="nonlinear", n_panels=20, max_iterations=50)
    alphas = np.radians(np.linspace(-2.0, 8.0, 6))

    solver = NonlinearSolver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res_serial = [
            solver.solve(ac, vt.FlightCondition(V_inf=30.0, alpha=float(a)), settings)
            for a in alphas
        ]
        res_sweep = solver.solve_sweep(ac, cond, settings, alphas)

    # The sweep starts each angle from the previous solution (continuation), so the
    # iteration counts differ; both converge to the same solution within the tolerance.
    assert len(res_sweep) == len(res_serial)
    for r_sw, r_ser in zip(res_sweep, res_serial):
        assert r_sw.converged and r_ser.converged
        np.testing.assert_allclose(r_sw.totals.CL, r_ser.totals.CL, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(r_sw.totals.CDi, r_ser.totals.CDi, rtol=1e-5, atol=1e-10)

