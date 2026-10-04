# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Verification against closed-form results and independent solutions.

Every test compares with an external reference (a formula or an independent
implementation of the same theory) with a stated tolerance. A test that only
checks that the code runs does not belong here.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import build_sources, make_ground_plane, make_unknown_map
from ventorum.aero.vortex import induced_velocity
from ventorum.geometry.lattice import build_lattice
from ventorum.reference import (
    circular_wing_cl_alpha,
    elliptic_wing_cl_alpha,
    glauert_monoplane,
    helmbold_cl_alpha,
    wing_sections_elliptic,
)
from ventorum.solvers.lattice_base import assemble_system_matrix

# Task T-0010, decision 4: no fixture hides warnings here. A test in this
# module must not emit a warning; a test that expects one uses pytest.warns,
# and pytest.warns overrides this mark for that test.
pytestmark = pytest.mark.filterwarnings("error")


def _elliptic(AR: float, straight_line: str = "quarter_chord", airfoil=None) -> vt.Aircraft:
    b = 1.0
    c0 = 4.0 * b / (np.pi * AR)
    secs = wing_sections_elliptic(c0, straight_line=straight_line)
    if airfoil is not None:
        for s in secs:
            s.airfoil = airfoil
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=b / 2.0, sections=secs)])
    ac.compute_reference_values()
    return ac


def _tapered(AR: float, taper: float, straight_qc: bool = True, **surf_kw) -> vt.Aircraft:
    b = 1.0
    c_r = 2.0 * b / (AR * (1.0 + taper))
    c_t = taper * c_r
    if straight_qc:
        secs = [vt.WingSection(0.0, c_r, x_le=-0.25 * c_r), vt.WingSection(1.0, c_t, x_le=-0.25 * c_t)]
    else:
        secs = [vt.WingSection(0.0, c_r), vt.WingSection(1.0, c_t)]
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=b / 2.0, sections=secs, **surf_kw)])
    ac.compute_reference_values()
    return ac


def _ar(ac: vt.Aircraft) -> float:
    return ac.b_ref ** 2 / ac.S_ref


def _sweep_warning(solver: str):
    """Context manager that expects the sweep warning of the lifting-line solvers.

    The lifting line is not grid convergent with sweep and warns about it.
    The VLM is grid convergent with sweep and must stay silent.
    """
    if solver in ("linear", "nonlinear"):
        return pytest.warns(RuntimeWarning, match="not grid convergent")
    return contextlib.nullcontext()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Elliptic wing (Lanchester-Prandtl theory): e = 1 and CL_alpha = a0 / (1 + a0 / (pi AR))
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver", ["linear", "nonlinear", "fourier"])
@pytest.mark.parametrize("a0_factor", [1.0, 0.9])
def test_elliptic_wing_lifting_line_matches_prandtl(solver, a0_factor):
    a0 = a0_factor * 2.0 * np.pi
    ac = _elliptic(8.0, airfoil=vt.LinearAirfoil(a0=a0))
    alpha = np.radians(2.0)
    res = vt.analyze(ac, alpha_deg=2.0, solver=solver, n_panels=40)
    cla = res.totals.CL / alpha
    assert abs(res.totals.e - 1.0) < 0.005
    assert abs(cla / elliptic_wing_cl_alpha(_ar(ac), a0) - 1.0) < 0.005


def test_elliptic_wing_vlm_has_elliptic_loading():
    ac = _elliptic(8.0)
    res = vt.analyze(ac, alpha_deg=2.0, solver="vlm", n_panels=30)
    assert abs(res.totals.e - 1.0) < 0.005


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Glauert monoplane equation (independent solution), rectangular and tapered
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("AR", [4.0, 6.0, 10.0])
@pytest.mark.parametrize("taper", [1.0, 0.5, 0.25])
def test_lifting_line_matches_glauert_monoplane(AR, taper):
    cla_ref, e_ref = glauert_monoplane(AR, taper)
    ac = _tapered(AR, taper, straight_qc=True)
    alpha = np.radians(2.0)
    four = vt.analyze(ac, alpha_deg=2.0, solver="fourier", n_panels=60).totals
    ps = vt.analyze(ac, alpha_deg=2.0, solver="linear", n_panels=40).totals
    assert abs(four.CL / alpha / cla_ref - 1.0) < 1e-3
    assert abs(four.e / e_ref - 1.0) < 1e-3
    # The Phillips & Snyder system is linear in sin(alpha).
    assert abs(ps.CL / np.sin(alpha) / cla_ref - 1.0) < 2e-3
    assert abs(ps.e / e_ref - 1.0) < 2e-3


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Lifting-surface checks for the VLM
# ═══════════════════════════════════════════════════════════════════════════════

def test_vlm_circular_wing_matches_kinner():
    """Flat circular wing (AR = 4/pi): exact lifting-surface CL_alpha = 1.790/rad."""
    b = 1.0
    secs = wing_sections_elliptic(b, straight_line="mid_chord", n_sections=401)
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=b / 2.0, sections=secs)])
    # Small angle and the linear (body-axis) wake: Kinner's result is linear theory.
    sett = vt.SolverSettings(solver_type="vlm", n_panels=30, n_chord=16, wake_alignment="body")
    res = vt.analyze(ac, settings=sett, alpha_deg=0.5)
    cla = res.totals.CL / np.sin(np.radians(0.5))
    assert abs(cla / circular_wing_cl_alpha() - 1.0) < 0.002


@pytest.mark.parametrize("AR", [1.0, 2.0, 6.0, 20.0])
def test_vlm_elliptic_wing_lift_slope_between_slender_wing_and_lifting_line(AR):
    """Helmbold's formula is approximate (measured agreement 1 to 4 %); the VLM must
    also stay below lifting-line theory and above zero-aspect-ratio slender-wing theory
    at low aspect ratio."""
    ac = _elliptic(AR)
    res = vt.analyze(ac, settings=vt.SolverSettings(solver_type="vlm", n_panels=30, n_chord=8), alpha_deg=1.0)
    cla = res.totals.CL / np.sin(np.radians(1.0))
    ar = _ar(ac)
    assert abs(cla / helmbold_cl_alpha(ar) - 1.0) < 0.05
    assert cla < elliptic_wing_cl_alpha(ar)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Camber and section pitching moment
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver,sweep_deg", [("vlm", 0.0), ("vlm", 35.0), ("linear", 0.0)])
def test_camber_shifts_the_zero_lift_angle_exactly(solver, sweep_deg):
    """With a body-axis wake the system matrix does not depend on alpha, so the
    circulation is exactly proportional to the right-hand side. On a swept
    wing the zero-lift angle must act in the streamwise section (not times
    cos(sweep))."""
    aL0 = np.radians(-3.0)
    af = vt.LinearAirfoil(a0=0.95 * 2.0 * np.pi, alpha_L0=aL0)
    w = vt.LiftingSurface(semi_span=3.0, sweep_le=np.radians(sweep_deg),
                           sections=[vt.WingSection(0, 1.0, airfoil=af), vt.WingSection(1, 1.0, airfoil=af)])
    sett = vt.SolverSettings(solver_type=solver, n_panels=20, wake_alignment="body")
    ratios = []
    for a_deg in (0.0, 2.0, 5.0):
        a = np.radians(a_deg)
        g = vt.analyze(w, settings=sett, alpha_deg=a_deg).spanwise[0].gamma
        rhs = np.sin(a - aL0) if solver == "vlm" else (np.sin(a) - aL0)
        ratios.append(g / rhs)
    np.testing.assert_allclose(ratios[1], ratios[0], rtol=1e-10)
    np.testing.assert_allclose(ratios[2], ratios[0], rtol=1e-10)


@pytest.mark.parametrize("sweep_deg", [0.0, 30.0, 45.0])
def test_camber_equals_incidence_on_swept_vlm_wing(sweep_deg):
    """alpha_L0 = -2 deg at alpha = 0 against a flat wing at alpha = 2 deg.

    Body-axis wake, planar wing: both have the same right-hand side; the
    camber tilts the boundary-condition normal by 2 deg, and the induced
    velocity is normal to the plane, so the matrix of the cambered wing is
    cos(2 deg) times the other. Hence gamma_camber * cos(2 deg) = gamma_flat
    exactly, for every sweep (a camber turn about the swept strip axis gave
    an extra cos(sweep) factor). The lift differs slightly because the force
    uses the local velocity, which turns with alpha."""
    def wing(af):
        return vt.LiftingSurface(semi_span=4.0, sweep_le=np.radians(sweep_deg),
                                  sections=[vt.WingSection(0, 1.0, airfoil=af), vt.WingSection(1, 1.0, airfoil=af)])
    sett = vt.SolverSettings(solver_type="vlm", n_panels=16, n_chord=4, wake_alignment="body")
    r_camber = vt.analyze(wing(vt.LinearAirfoil(alpha_L0=np.radians(-2.0))), settings=sett, alpha_deg=0.0)
    r_flat = vt.analyze(wing(vt.LinearAirfoil()), settings=sett, alpha_deg=2.0)
    np.testing.assert_allclose(r_camber.spanwise[0].gamma * np.cos(np.radians(2.0)), r_flat.spanwise[0].gamma, rtol=1e-9)
    assert abs(r_camber.totals.CL / r_flat.totals.CL - 1.0) < 2e-3


@pytest.mark.parametrize("solver", ["vlm", "linear", "fourier"])
def test_section_moment_at_zero_lift_equals_cm0(solver):
    aL0, cm0 = np.radians(-4.0), -0.093
    af = vt.LinearAirfoil(alpha_L0=aL0, Cm0=cm0)
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0, airfoil=af), vt.WingSection(1, 1.0, airfoil=af)])
    # Zero lift: alpha = alpha_L0 (VLM, Fourier) or sin(alpha) = alpha_L0 (lifting line).
    a = aL0 if solver != "linear" else np.arcsin(aL0)
    sett = vt.SolverSettings(solver_type=solver, n_panels=20, wake_alignment="body")
    res = vt.analyze(w, condition=vt.FlightCondition(V_inf=30.0, alpha=a), settings=sett)
    assert abs(res.totals.CL) < 1e-10
    assert abs(res.totals.Cm - cm0) < 1e-10


def test_lifting_line_moment_about_quarter_chord_is_cm0():
    """A straight unswept lifting line puts all lift on the quarter-chord line."""
    cm0 = -0.071
    af = vt.LinearAirfoil(alpha_L0=np.radians(-2.0), Cm0=cm0)
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0, airfoil=af), vt.WingSection(1, 1.0, airfoil=af)])
    for solver in ("linear", "nonlinear"):
        res = vt.LinearLLTSolver().solve(
            w, vt.FlightCondition(alpha=np.radians(5.0)),
            vt.SolverSettings(solver_type=solver, n_panels=20), ref_point=np.array([0.25, 0.0, 0.0]),
        ) if solver == "linear" else vt.NonlinearSolver().solve(
            w, vt.FlightCondition(alpha=np.radians(5.0)),
            vt.SolverSettings(solver_type=solver, n_panels=20), ref_point=np.array([0.25, 0.0, 0.0]),
        )
        assert abs(res.totals.Cm - cm0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Sign conventions (standard aircraft axes)
# ═══════════════════════════════════════════════════════════════════════════════

def _derivs(ac, alpha_deg=4.0, wake="freestream", solver="vlm"):
    sett = vt.SolverSettings(solver_type=solver, n_panels=20, wake_alignment=wake)
    t = [vt.analyze(ac, condition=vt.FlightCondition(V_inf=40.0, alpha=np.radians(alpha_deg), beta=np.radians(b)),
                     settings=sett).totals for b in (-1.0, 1.0)]
    db = np.radians(2.0)
    return (t[1].Cl - t[0].Cl) / db, (t[1].Cn - t[0].Cn) / db, (t[1].CY - t[0].CY) / db


def test_dihedral_gives_negative_cl_beta_and_is_antisymmetric():
    def wing(dih):
        return vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=5.0, dihedral=np.radians(dih),
                                                         sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])])
    clb_pos, _, _ = _derivs(wing(5.0), wake="body")
    clb_neg, _, _ = _derivs(wing(-5.0), wake="body")
    clb_0, _, _ = _derivs(wing(0.0), wake="body")
    assert clb_pos < -0.05          # stable dihedral effect
    # The dihedral term changes sign with the dihedral. (In the VLM the flat wing
    # has a Cl_beta of its own, from the force on the on-surface chordwise legs.)
    d_pos, d_neg = clb_pos - clb_0, clb_neg - clb_0
    assert d_pos < 0.0 < d_neg
    assert abs(d_pos + d_neg) < 2e-2 * abs(d_pos)
    # The lifting line has forces on the bound vortex only: no flat-wing term with a body-axis wake.
    assert abs(_derivs(wing(0.0), wake="body", solver="linear")[0]) < 1e-10
    for solver in ("vlm", "linear"):
        assert _derivs(wing(5.0), solver=solver)[0] < 0.0


def test_leg_forces_converge_and_close_the_drag_balance():
    """The force on the on-surface trailing legs does not depend on the chordwise
    panel count, and with it the near-field drag agrees with the Trefftz-plane
    drag in sideslip."""
    ac = vt.Aircraft(surfaces=[vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])])
    clb = []
    for nc in (4, 8):
        sett = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=nc, wake_alignment="body")
        t = [vt.analyze(ac, condition=vt.FlightCondition(alpha=np.radians(5.0), beta=np.radians(b)), settings=sett).totals
             for b in (-2.0, 2.0)]
        clb.append((t[1].Cl - t[0].Cl) / np.radians(4.0))
    assert abs(clb[1] / clb[0] - 1.0) < 0.02
    t = vt.analyze(ac, condition=vt.FlightCondition(alpha=np.radians(5.0), beta=np.radians(10.0)),
                    settings=vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=4)).totals
    assert abs(t.CDi_nearfield / t.CDi - 1.0) < 0.01


def test_aft_fin_gives_weathercock_stability():
    wing = vt.LiftingSurface(name="wing", semi_span=5.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    fin = vt.LiftingSurface(name="fin", semi_span=1.2, dihedral=np.radians(90.0), is_symmetric=False,
                             position=np.array([5.0, 0.0, 0.2]), sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 0.6)])
    clb, cnb, cyb = _derivs(vt.Aircraft(surfaces=[wing, fin]), alpha_deg=2.0)
    assert cnb > 0.0   # nose turns into the wind
    assert cyb < 0.0   # side force away from the wind
    assert clb < 0.0   # fin above the moment reference


def test_aft_tail_gives_negative_pitch_stiffness():
    wing = vt.LiftingSurface(name="wing", semi_span=5.0, sections=[vt.WingSection(0, 1.2), vt.WingSection(1, 0.8)])
    tail = vt.LiftingSurface(name="tail", semi_span=1.6, position=np.array([5.0, 0.0, 0.3]),
                              sections=[vt.WingSection(0, 0.7), vt.WingSection(1, 0.5)])
    ac = vt.Aircraft(surfaces=[wing, tail])
    t = [vt.analyze(ac, alpha_deg=a, n_panels=20).totals for a in (2.0, 4.0)]
    cm_alpha = (t[1].Cm - t[0].Cm) / np.radians(2.0)
    assert cm_alpha < 0.0  # moments about the origin (wing root leading edge)


def test_vtail_and_fin_geometry_conventions():
    vtail = vt.LiftingSurface(semi_span=0.42, dihedral=np.radians(-38.0),
                            sections=[vt.WingSection(0, 0.14), vt.WingSection(1, 0.10)])
    ds = vt.discretize_surface(vtail, n_panels=8)
    tip = ds.nodes_qc[-1]
    assert abs(tip[1] - 0.42 * np.cos(np.radians(38.0))) < 1e-12
    assert abs(np.degrees(np.arctan2(-tip[2], tip[1])) - 38.0) < 1e-9
    fin = vt.LiftingSurface(semi_span=1.0, dihedral=np.radians(90.0), is_symmetric=False,
                             sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 0.5)])
    d2 = vt.discretize_surface(fin, n_panels=8)
    assert abs(d2.nodes_qc[-1, 1]) < 1e-12 and abs(d2.nodes_qc[-1, 2] - 1.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Mesh convergence and conditioning
# ═══════════════════════════════════════════════════════════════════════════════

def _monotone_half(gamma: np.ndarray) -> bool:
    half = gamma[len(gamma) // 2:]
    d = np.diff(half)
    return bool(np.all(d <= 1e-12 * np.max(np.abs(half))))


@pytest.mark.parametrize("solver", ["vlm", "linear"])
@pytest.mark.parametrize("spacing", ["auto", "cosine", "half-cosine"])
def test_mesh_convergence_and_smooth_circulation(solver, spacing):
    w = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    out = {}
    for n in (20, 80, 160):
        sett = vt.SolverSettings(solver_type=solver, n_panels=n, spacing=spacing)
        res = vt.analyze(w, settings=sett, alpha_deg=5.0)
        out[n] = res
        g = res.spanwise[0].gamma
        if spacing == "cosine":
            # Root clustering leaves a small documented dip next to the root.
            half = g[len(g) // 2:]
            assert (half.max() - half[0]) / half.max() < 5e-3
            assert np.all(np.diff(half[np.argmax(half):]) <= 1e-12 * half.max())
        else:
            assert _monotone_half(g), f"oscillating circulation at n={n}"
    cl80, cl160 = out[80].totals.CL, out[160].totals.CL
    e80, e160 = out[80].totals.e, out[160].totals.e
    assert abs(cl80 / cl160 - 1.0) < 1e-3
    assert abs(e80 / e160 - 1.0) < 1e-3
    cond = np.linalg.cond(assemble_system_matrix(w, vt.FlightCondition(alpha=np.radians(5.0)),
                                                 vt.SolverSettings(solver_type=solver, n_panels=160, spacing=spacing)))
    assert cond < 1e6


@pytest.mark.parametrize("sweep_deg", [0.0, 30.0, 45.0])
def test_vlm_converges_for_swept_wings(sweep_deg):
    w = vt.LiftingSurface(semi_span=4.0, sweep_le=np.radians(sweep_deg),
                           sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    cl = [vt.analyze(w, settings=vt.SolverSettings(solver_type="vlm", n_panels=n, n_chord=1), alpha_deg=2.0).totals.CL
          for n in (20, 80)]
    assert abs(cl[0] / cl[1] - 1.0) < 1e-3


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Symmetry plane: the folded solve equals the full solve
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver", ["vlm", "linear", "nonlinear"])
def test_symmetry_fold_equals_full_solve(solver):
    w = vt.LiftingSurface(semi_span=4.0, sweep_le=np.radians(10.0), dihedral=np.radians(4.0),
                           sections=[vt.WingSection(0, 1.3, twist=np.radians(1.0)), vt.WingSection(1, 0.6, twist=np.radians(-2.0))])
    with _sweep_warning(solver):
        r_sym = vt.analyze(w, alpha_deg=5.0, solver=solver, n_panels=16, use_symmetry=True)
    with _sweep_warning(solver):
        r_full = vt.analyze(w, alpha_deg=5.0, solver=solver, n_panels=16, use_symmetry=False)
    assert r_sym.symmetry_used and not r_full.symmetry_used
    for k in ("CL", "CDi", "Cm"):
        np.testing.assert_allclose(getattr(r_sym.totals, k), getattr(r_full.totals, k), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(r_sym.spanwise[0].gamma, r_full.spanwise[0].gamma, rtol=1e-9, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Ground effect: image method and limits
# ═══════════════════════════════════════════════════════════════════════════════

def test_ground_plane_has_zero_normal_velocity():
    """The images cancel the velocity normal to a tilted and banked ground."""
    w = vt.LiftingSurface(semi_span=2.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 0.7)])
    ac = vt.Aircraft(surfaces=[w])
    cond = vt.FlightCondition(V_inf=1.0, alpha=np.radians(5.0), phi=np.radians(3.0))
    lat = build_lattice(ac, vt.SolverSettings(n_panels=10), collocation="vlm", n_chord=3)
    gp = make_ground_plane(lat, 0.4, cond.alpha, 0.0, cond.phi, height_ref="min")
    d = vt.freestream_direction(cond.alpha, 0.0)
    umap = make_unknown_map(lat, use_symmetry=False)
    src = build_sources(lat, d, umap, gp)
    rng = np.random.default_rng(1)
    gamma = rng.normal(size=lat.n_panels)
    # Points on the ground plane under the wing.
    pts = rng.uniform([-0.5, -2.5, -1.0], [2.0, 2.5, 1.0], size=(40, 3))
    pts = gp.reflect_points(pts)  # reflect then average: lands exactly on the plane
    pts = 0.5 * (pts + gp.reflect_points(pts))
    v = induced_velocity(pts, src, gamma)
    assert np.max(np.abs(v @ gp.normal)) < 1e-12 * max(1.0, np.max(np.abs(v)))
    assert abs(d @ gp.normal) < 1e-14  # free stream parallel to the ground


def test_ground_effect_recovers_free_air_far_from_ground():
    w = vt.LiftingSurface(semi_span=3.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    sett = vt.SolverSettings(n_panels=16, n_chord=4)
    free = vt.analyze(w, settings=sett, alpha_deg=4.0).totals
    far = vt.analyze_ground_effect(w, h=1.0e4, alpha_deg=4.0, settings=sett, compute_strike_limit=False)
    assert abs(far.CL / free.CL - 1.0) < 1e-4
    assert abs(far.CDi / free.CDi - 1.0) < 1e-3


def test_ground_effect_bank_moment_is_restoring_and_limits_are_enforced():
    w = vt.LiftingSurface(semi_span=3.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    res = vt.analyze_ground_effect(w, h=0.5, alpha_deg=4.0, phi_deg=2.0, n_panels=16)
    assert res.Cl < 0.0                     # right wing down -> left rolling moment
    assert res.h_tip_right < res.h_tip_left
    with pytest.raises(ValueError, match="strike"):
        vt.analyze_ground_effect(w, h=0.2, alpha_deg=0.0, phi_deg=10.0, n_panels=16)
    with pytest.raises(ValueError, match="not valid in ground effect"):
        vt.analyze_ground_effect(w, h=0.5, alpha_deg=4.0, n_panels=16, solver="linear")
    # The chordwise panel count follows the gap.
    near = vt.analyze_ground_effect(w, h=0.1, alpha_deg=2.0, n_panels=12, compute_strike_limit=False)
    assert near.n_chord >= int(np.ceil(1.0 / near.h_min))


def test_ground_is_wings_level_in_sideslip():
    """phi = 0 means wings level also with sideslip: equal tip clearances."""
    from ventorum.aero.system import freestream_direction, ground_normal
    for a_deg, b_deg in ((5.0, 5.0), (8.0, 10.0)):
        a, b = np.radians(a_deg), np.radians(b_deg)
        k = ground_normal(a, b, 0.0)
        assert abs(k[1]) < 1e-15 and abs(k @ freestream_direction(a, b)) < 1e-15
    wing = vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    r = vt.analyze_ground_effect(wing, h=0.4, alpha_deg=8.0, beta_deg=10.0, phi_deg=0.0,
                                  settings=vt.SolverSettings(n_panels=12), compute_strike_limit=False)
    assert abs(r.h_tip_left - r.h_tip_right) < 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Nonlinear lifting line
# ═══════════════════════════════════════════════════════════════════════════════

def test_nonlinear_equals_linear_for_linear_sections_at_small_angles():
    w = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 0.6)])
    lin = vt.analyze(w, alpha_deg=1.0, solver="linear", n_panels=24).totals
    nl = vt.analyze(w, alpha_deg=1.0, solver="nonlinear", n_panels=24).totals
    assert abs(nl.CL / lin.CL - 1.0) < 1e-4  # difference is second order in alpha


def test_tabulated_linear_polar_equals_linear_airfoil():
    al = np.radians(np.linspace(-20, 20, 81))
    a0, aL0 = 5.9, np.radians(-2.0)
    tab = vt.TabulatedAirfoil(alpha=al, Cl_data=a0 * (al - aL0), Cd_data=np.full_like(al, 0.01))
    lin = vt.LinearAirfoil(a0=a0, alpha_L0=aL0, Cd0=0.01)

    def mk(af):
        return vt.LiftingSurface(semi_span=4.0, sections=[vt.WingSection(0, 1.0, airfoil=af), vt.WingSection(1, 1.0, airfoil=af)])
    r1 = vt.analyze(mk(tab), alpha_deg=3.0, solver="nonlinear", n_panels=20).totals
    r2 = vt.analyze(mk(lin), alpha_deg=3.0, solver="nonlinear", n_panels=20).totals
    assert abs(r1.CL / r2.CL - 1.0) < 1e-12
    assert abs(r1.CDp - 0.01) < 1e-12


def test_nonlinear_wing_stalls_below_section_maximum():
    al = np.radians(np.arange(-10, 31, 0.5))
    cl = np.minimum(2 * np.pi * al, np.where(al < np.radians(11), 2 * np.pi * al, 1.2 - 2.0 * (al - np.radians(11))))
    tab = vt.TabulatedAirfoil(alpha=al, Cl_data=np.clip(cl, -1.2, None), Cd_data=0.008 + 0.02 * al ** 2)
    w = vt.LiftingSurface(semi_span=3.0, sections=[vt.WingSection(0, 1.0, airfoil=tab), vt.WingSection(1, 1.0, airfoil=tab)])
    # "auto" keeps the VLM (linear part of the polar) and says so.
    with pytest.warns(RuntimeWarning, match="linear part"):
        assert vt.analyze(w, alpha_deg=4.0, n_panels=12).solver_type == "vlm"
    cls = []
    for a in (8.0, 12.0, 14.0, 16.0, 20.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = vt.analyze(w, alpha_deg=a, n_panels=20, solver="nonlinear")
        if a <= 16.0:
            assert r.converged
        else:
            # Far past the maximum lift the lifting line may have no solution;
            # it must then say so.
            assert r.converged or r.totals.trust.rating != "HIGH"
        cls.append(r.totals.CL)
    assert max(cls) < 1.2                         # the wing cannot exceed the section maximum
    assert cls[-1] < max(cls)                     # lift falls after the maximum


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Mirrored surfaces and entry-point consistency
# ═══════════════════════════════════════════════════════════════════════════════

def _twin_fins():
    af = vt.LinearAirfoil(alpha_L0=np.radians(-2.0), Cm0=-0.05)
    wing = vt.LiftingSurface(name="wing", semi_span=4.0, sections=[vt.WingSection(0, 1.0), vt.WingSection(1, 1.0)])
    fin = vt.LiftingSurface(name="fin", semi_span=1.0, dihedral=np.radians(70.0), is_symmetric=False,
                             position=np.array([4.0, 1.0, 0.0]),
                             sections=[vt.WingSection(0, 0.8, airfoil=af),
                                       vt.WingSection(1, 0.5, twist=np.radians(-3.0), airfoil=af)])
    return wing, fin


def test_symmetric_surface_off_the_centre_plane_is_refused():
    wing, fin = _twin_fins()
    fin.is_symmetric = True
    with pytest.raises(ValueError, match="not on the plane"):
        vt.analyze(vt.Aircraft(surfaces=[wing, fin]), alpha_deg=4.0, n_panels=8)
    with pytest.raises(ValueError):
        fin.mirrored()


def test_mirrored_surface_is_an_exact_mirror_image():
    """Canted, cambered and twisted twin fins: no lateral force or moment at
    beta = 0, and exactly opposite values at +/- beta."""
    wing, fin = _twin_fins()
    ac = vt.Aircraft(surfaces=[wing, fin, fin.mirrored()])
    sett = vt.SolverSettings(n_panels=10)
    t0 = vt.analyze(ac, settings=sett, alpha_deg=4.0).totals
    assert max(abs(t0.CY), abs(t0.Cl), abs(t0.Cn)) < 1e-12
    tp, tm = (vt.analyze(ac, condition=vt.FlightCondition(alpha=np.radians(4.0), beta=np.radians(b)), settings=sett).totals
              for b in (3.0, -3.0))
    for name in ("CY", "Cl", "Cn"):
        assert abs(getattr(tp, name) + getattr(tm, name)) < 1e-10
    assert tp.Cn > 0.0 and tp.CY < 0.0           # fins aft: weathercock stability
    # The mirror flag survives the JSON round trip.
    from ventorum.core.datatypes import aircraft_from_json, aircraft_to_json
    back = aircraft_from_json(aircraft_to_json(ac))
    assert [s.mirror_y for s in back.surfaces] == [False, False, True]


@pytest.mark.parametrize("solver", ["vlm", "linear"])
def test_analyze_and_analyze_sweep_agree(solver):
    wing = vt.LiftingSurface(semi_span=5.0, sections=[vt.WingSection(0, 2.0), vt.WingSection(1, 1.0, twist=-0.05)])
    with _sweep_warning(solver):
        r1 = vt.analyze(wing, alpha_deg=5.0, solver=solver, n_panels=20)
    with _sweep_warning(solver):
        r2 = vt.analyze_sweep(wing, alpha_deg_range=np.array([5.0]), solver=solver, n_panels=20)[0]
    assert abs(r1.totals.CL - r2.totals.CL) < 1e-12
    assert abs(r1.totals.CDi - r2.totals.CDi) < 1e-12


def _straight(name, semi_span, n, **kw):
    return vt.LiftingSurface(
        name=name, semi_span=semi_span, n_panels=n, spacing="uniform",
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)], **kw,
    )


@pytest.mark.parametrize("dihedral_deg", [0.0, 35.0])
def test_surfaces_that_share_an_edge_equal_one_surface(dihedral_deg):
    """A surface cut into parts that share their edges (a split wing, a V-tail
    made of two mirrored halves) is the same vortex sheet: the result must
    equal the result of the one-piece surface to round-off."""
    cond = vt.FlightCondition(V_inf=10.0, alpha=np.radians(5.0))
    st = vt.SolverSettings(n_panels=40)
    dih = np.radians(dihedral_deg)
    one = vt.Aircraft(surfaces=[_straight("one", 4.0, 40, dihedral=dih)], S_ref=8.0, b_ref=8.0, c_ref=1.0)
    half = _straight("half", 4.0, 40, dihedral=dih, is_symmetric=False)
    halves = vt.Aircraft(surfaces=[half, half.mirrored()], S_ref=8.0, b_ref=8.0, c_ref=1.0)
    inner = _straight("inner", 2.0, 20, dihedral=dih)
    tip = np.array([0.0, 2.0 * np.cos(dih), 2.0 * np.sin(dih)])
    outer = _straight("outer", 2.0, 20, dihedral=dih, is_symmetric=False, position=tip)
    split = vt.Aircraft(surfaces=[inner, outer, outer.mirrored()], S_ref=8.0, b_ref=8.0, c_ref=1.0)
    ref = vt.analyze(one, condition=cond, settings=st).totals
    for ac in (halves, split):
        t = vt.analyze(ac, condition=cond, settings=st).totals
        assert t.CL == pytest.approx(ref.CL, rel=1e-9)
        assert t.CDi == pytest.approx(ref.CDi, rel=1e-9)
        assert t.Cm == pytest.approx(ref.Cm, rel=1e-9, abs=1e-12)


def test_core_groups_join_only_surfaces_that_share_an_edge():
    wing = _straight("wing", 4.0, 20)
    winglet = _straight("winglet", 0.5, 4, is_symmetric=False, dihedral=np.radians(80.0),
                        position=np.array([0.0, 4.0, 0.0]))
    tail = _straight("tail", 1.5, 8, position=np.array([5.0, 0.0, 0.3]))
    ac = vt.Aircraft(surfaces=[wing, winglet, winglet.mirrored(), tail])
    lat = build_lattice(ac, vt.SolverSettings(n_panels=20), collocation="vlm")
    group = {s.name: lat.strip_core_group[s.strips][0] for s in lat.surfaces}
    names = [s.name for s in lat.surfaces]
    assert len({group[n] for n in names if n != "tail"}) == 1
    assert group["tail"] != group["wing"]
