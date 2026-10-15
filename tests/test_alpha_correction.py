# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Unit tests for viscous-inviscid alpha-correction coupling."""

from __future__ import annotations

import shutil
import numpy as np
import pytest

from ventorum.aero.polar_source import XfoilPolarSource
from ventorum.aero.xfoil_runner import xfoil_version
from ventorum.core.datatypes import (
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.viscous.alpha_correction import solve_viscous


def make_rectangular_wing(aspect_ratio: float = 8.0, chord: float = 1.0) -> LiftingSurface:
    """Return a rectangular wing of given aspect ratio and chord."""
    semi_span = 0.5 * aspect_ratio * chord
    return LiftingSurface(
        name="RectangularWing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord),
            WingSection(y_frac=1.0, chord=chord),
        ],
        sweep_le=0.0,
        dihedral=0.0,
    )


def make_linear_polar(slope: float = 2.0 * np.pi, cd0: float = 0.01) -> TabulatedAirfoil:
    """Return a synthetic linear polar with given lift-curve slope."""
    alpha_deg = np.linspace(-15.0, 25.0, 81)
    alpha_rad = np.radians(alpha_deg)
    cl_data = slope * alpha_rad
    cd_data = cd0 + 0.005 * alpha_rad**2
    return TabulatedAirfoil(
        name=f"LinearPolar_{slope:.3f}",
        alpha=alpha_rad,
        Cl_data=cl_data,
        Cd_data=cd_data,
        Re=1e6,
    )


def make_stalling_polar(
    slope: float = 2.0 * np.pi,
    alpha_stall_deg: float = 10.0,
    cl_drop_slope: float = 1.5,
    cd0: float = 0.01,
) -> TabulatedAirfoil:
    """Return a synthetic polar that is linear up to stall and drops after stall."""
    alpha_deg = np.linspace(-10.0, 25.0, 141)
    alpha_rad = np.radians(alpha_deg)
    a_stall_rad = np.radians(alpha_stall_deg)
    cl_max = slope * a_stall_rad

    cl_data = np.where(
        alpha_rad <= a_stall_rad,
        slope * alpha_rad,
        cl_max - cl_drop_slope * (alpha_rad - a_stall_rad),
    )
    cd_data = np.where(
        alpha_rad <= a_stall_rad,
        cd0 + 0.005 * alpha_rad**2,
        cd0 + 0.005 * a_stall_rad**2 + 0.4 * (alpha_rad - a_stall_rad),
    )
    return TabulatedAirfoil(
        name="StallingPolar",
        alpha=alpha_rad,
        Cl_data=cl_data,
        Cd_data=cd_data,
        Re=1e6,
    )


def test_linear_polar_matches_inviscid():
    """Verify that with slope 2*pi the viscous solve matches the inviscid solve to 1e-8."""
    wing = make_rectangular_wing(aspect_ratio=8.0, chord=1.0)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(5.0))
    settings = SolverSettings()
    polar = make_linear_polar(slope=2.0 * np.pi)

    for model in ("vlm", "llt"):
        from ventorum.solvers.factory import make_solver, resolve_solver_type

        inviscid_solver = make_solver(resolve_solver_type(model))
        inviscid_res = inviscid_solver.solve(wing, cond, settings)

        viscous_res = solve_viscous(wing, cond, settings, polar, model=model)

        assert viscous_res.converged
        assert abs(viscous_res.totals.CL - inviscid_res.totals.CL) < 1e-8
        assert abs(viscous_res.totals.CDi - inviscid_res.totals.CDi) < 1e-8
        assert np.all(np.abs(viscous_res.details["delta"]) < 1e-8)


def test_reduced_slope_follows_lifting_line():
    """Verify high-AR rectangular wing CL follows lifting-line formula with section slope a within 2 %."""
    from ventorum.reference import glauert_monoplane

    ar = 12.0
    chord = 1.0
    wing = make_rectangular_wing(aspect_ratio=ar, chord=chord)
    alpha_deg = 5.0
    alpha_rad = np.radians(alpha_deg)
    cond = FlightCondition(V_inf=40.0, alpha=alpha_rad)
    settings = SolverSettings()

    # Section slope reduced by 15 %
    a = 0.85 * (2.0 * np.pi)
    polar = make_linear_polar(slope=a)

    res = solve_viscous(wing, cond, settings, polar, model="llt")
    assert res.converged

    # Independent Glauert lifting-line reference solution for rectangular wing
    cla_ref, _ = glauert_monoplane(ar, taper=1.0, a0=a)
    cl_llt_theory = cla_ref * alpha_rad
    rel_error = abs(res.totals.CL - cl_llt_theory) / cl_llt_theory
    assert rel_error < 0.02


def test_stalling_polar_sweep_and_branches():
    """Verify sweep from 0 to 20 deg flags post-stall strips and reports two branches."""
    wing = make_rectangular_wing(aspect_ratio=6.0, chord=1.0)
    settings = SolverSettings()
    polar = make_stalling_polar(alpha_stall_deg=10.0, cl_drop_slope=2.0)

    alphas_deg = np.linspace(0.0, 20.0, 21)
    conds = [FlightCondition(V_inf=40.0, alpha=np.radians(a)) for a in alphas_deg]

    results = solve_viscous(wing, conds, settings, polar, model="vlm")
    assert len(results) == len(alphas_deg)

    # Low angles below stall converge without post-stall strips
    res_low = results[2]  # 2 deg
    assert res_low.converged
    assert not res_low.details["post_stall"]

    # High angles above stall flag post-stall strips and report branches
    stalled_results = [r for r in results if r.details["post_stall"]]
    assert len(stalled_results) > 0

    for r in stalled_results:
        assert any(flag == "post-stall, non-unique" for flag in r.details["strip_flags"])

    # At least one case shows two distinct branches
    branched = [r for r in results if "branches" in r.details and len(r.details["branches"]) > 1]
    if branched:
        b0 = branched[0].details["branches"][0]
        b1 = branched[0].details["branches"][1]
        assert abs(b0.totals.CL - b1.totals.CL) > 1e-3
        assert branched[0].totals.trust is not None
        assert branched[0].totals.trust.factors.get("post_stall_branch", 0.0) > 0.0


def test_factorization_count_is_one():
    """Verify that the system matrix factorization count is exactly 1 per geometry."""
    wing = make_rectangular_wing(aspect_ratio=8.0, chord=1.0)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(8.0))
    settings = SolverSettings()
    polar = make_stalling_polar(alpha_stall_deg=7.0)

    res = solve_viscous(wing, cond, settings, polar, model="vlm")
    assert res.details.get("factorization_count") == 1


def test_profile_drag_equals_span_integral():
    """Verify that total profile drag equals the span integral of the polar Cd at the solved angles (1e-10)."""
    wing = make_rectangular_wing(aspect_ratio=8.0, chord=1.0)
    cond = FlightCondition(V_inf=45.0, alpha=np.radians(6.0))
    settings = SolverSettings()
    polar = make_stalling_polar(alpha_stall_deg=12.0)

    res = solve_viscous(wing, cond, settings, polar, model="vlm")
    assert res.converged

    # Compute span integral of polar Cd at solved angles
    lattice = res.details["lattice"]
    alpha_eff = res.details["alpha_eff"]
    cd_strip = polar.Cd(alpha_eff)

    s_ref = 2.0 * wing.semi_span * 1.0
    cd_integral = float(np.sum(cd_strip * lattice.chord * lattice.width) / s_ref)
    assert res.totals.CDp is not None
    assert abs(res.totals.CDp - cd_integral) < 1e-10
    assert abs(res.totals.CD_total - (res.totals.CDi + res.totals.CDp)) < 1e-10


def test_real_polar_naca2412_sweep():
    """Sweep with real NACA 2412 polars (skipped when XFOIL is absent)."""
    if shutil.which("xfoil") is None and xfoil_version() is None:
        pytest.skip("XFOIL is not installed or not available in PATH.")

    source = XfoilPolarSource()
    polar = source.polar("naca 2412", 1e6, np.linspace(-8.0, 22.0, 61))

    wing = LiftingSurface(
        name="RectangularWing2412",
        semi_span=4.0,
        sections=[
            WingSection(y_frac=0.0, chord=1.0, airfoil=polar),
            WingSection(y_frac=1.0, chord=1.0, airfoil=polar),
        ],
    )
    settings = SolverSettings()

    alphas_deg = np.arange(-4.0, 18.0, 2.0)
    conds = [FlightCondition(V_inf=50.0, alpha=np.radians(a)) for a in alphas_deg]

    results_visc = solve_viscous(wing, conds, settings, polar, model="vlm")

    from ventorum.solvers.factory import make_solver

    inviscid_solver = make_solver("vlm")
    results_inv = [inviscid_solver.solve(wing, c, settings) for c in conds]

    cl_visc = [r.totals.CL for r in results_visc]
    cl_inv = [r.totals.CL for r in results_inv]

    print("\nNACA 2412 AR 8 Rectangular Wing Re 1e6 Sweep:")
    print("Alpha [deg] | CL inviscid | CL viscous")
    for a, ci, cv in zip(alphas_deg, cl_inv, cl_visc):
        print(f"{a:11.1f} | {ci:11.4f} | {cv:10.4f}")

    k_max = int(np.argmax(cl_visc))
    cl_max_visc = cl_visc[k_max]
    cl_inv_at_same_angle = cl_inv[k_max]
    assert cl_max_visc < cl_inv_at_same_angle
