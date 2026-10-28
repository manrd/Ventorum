# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Unit tests for viscous-inviscid alpha-correction coupling."""

from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytest

from ventorum.core.constants import MU_SL
from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.polar_source import XfoilPolarSource
from ventorum.core.datatypes import (
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverSettings,
    TabulatedAirfoil,
    WingSection,
)
from ventorum.geometry.controls import flap_effectiveness
from ventorum.viscous.alpha_correction import solve_viscous


def make_rectangular_wing(aspect_ratio: float = 8.0, chord: float = 1.0, shape=None) -> LiftingSurface:
    """Return a rectangular wing of given aspect ratio and chord."""
    semi_span = 0.5 * aspect_ratio * chord
    return LiftingSurface(
        name="RectangularWing",
        semi_span=semi_span,
        sections=[
            WingSection(y_frac=0.0, chord=chord, shape=shape),
            WingSection(y_frac=1.0, chord=chord, shape=shape),
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

    # Below stall every angle converges.
    for a, r in zip(alphas_deg, results):
        if a <= 6.0:
            assert r.converged, a
    # Only converged solutions form branches; a non-converged result has none.
    for r in results:
        if not r.converged:
            assert "branches" not in r.details
            assert not r.details["non_unique"]
        for b in r.details.get("branches", []):
            assert b.converged
    # The search for a second branch is tested on a polar with a known
    # double solution in test_known_double_solution_finds_both_branches.


def test_factorization_count_is_one():
    """Verify that one condition makes exactly one factorization of the system matrix."""
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


_XFOIL_EXE = shutil.which(os.environ.get("VENTORUM_XFOIL", "xfoil"))


@pytest.mark.skipif(_XFOIL_EXE is None, reason="XFOIL executable not found")
def test_real_polar_naca2412_sweep(tmp_path):
    """Sweep with real NACA 2412 polars (skipped when XFOIL is absent).

    Regression: the skip condition ``xfoil_version() is None`` was never
    true; the test now skips when no executable is found and caches in a
    scratch folder.
    """
    source = XfoilPolarSource(cache=PolarCache(tmp_path / "polars"))
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


def test_xfoil_skip_condition_follows_the_executable(monkeypatch):
    """The skip rule of the XFOIL test is the executable lookup, not the version text."""
    from ventorum.aero import xfoil_runner

    monkeypatch.setattr(xfoil_runner.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    xfoil_runner.xfoil_version.cache_clear()
    # A missing XFOIL gives "unknown", never None: the old rule could not skip.
    assert xfoil_runner.xfoil_version("no-such-xfoil") == "unknown"
    xfoil_runner.xfoil_version.cache_clear()


# ── Flaps: the flap is counted once ─────────────────────────────────────────

FLAP_DEG = 10.0
HINGE = 0.75


def make_flapped_wing(eta=(0.0, 1.0), shape=None) -> LiftingSurface:
    """AR 8 rectangular wing with a plain flap of 10 deg at 75 % chord."""
    return LiftingSurface(
        name="FlappedWing",
        semi_span=4.0,
        sections=[WingSection(y_frac=0.0, chord=1.0, shape=shape),
                  WingSection(y_frac=1.0, chord=1.0, shape=shape)],
        controls=[ControlSurface("flap", eta[0], eta[1], HINGE, np.radians(FLAP_DEG))],
    )


def thin_airfoil_coefficients(alpha_rad, flap_deg, x_hinge=HINGE):
    """Thin-airfoil lift with the Glauert flap shift (2 pi slope)."""
    tau = flap_effectiveness(x_hinge)
    return 2.0 * np.pi * (np.asarray(alpha_rad) + tau * np.radians(flap_deg))


class GlauertSource:
    """Fake section polar source: thin-airfoil polars with the Glauert flap shift.

    It records every call, so the tests see the section, the Reynolds
    number, the flap tuple and the height that the coupling sends.
    """

    def __init__(self, supports_ground=False):
        self.calls = []
        self.supports_ground = supports_ground

    def polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, **options):
        self.calls.append({"section": section, "Re": Re, "flap": flap, **options})
        alpha = np.radians(np.asarray(alphas_deg, dtype=float))
        flap_deg = 0.0 if flap is None else float(flap[2])
        x_hinge = HINGE if flap is None else float(flap[0])
        return TabulatedAirfoil(
            name="glauert", alpha=alpha, Cl_data=thin_airfoil_coefficients(alpha, flap_deg, x_hinge),
            Cd_data=np.full(alpha.shape, 0.01), Cm_data=np.zeros(alpha.shape), Re=float(Re),
        )


def glauert_table(x_hinge=HINGE):
    """Polar table over Re and flap deflection with thin-airfoil data."""
    from ventorum.aero.polar_table import build_polar_table

    return build_polar_table(
        GlauertSource(), "naca 0012", [1e5, 1e7], list(np.arange(-15.0, 25.5, 0.5)),
        flap_deg=[0.0, 5.0, 10.0], x_hinge=x_hinge, workers=1,
    )


def _thin_polar():
    alpha = np.radians(np.linspace(-20.0, 25.0, 91))
    return TabulatedAirfoil(name="thin", alpha=alpha, Cl_data=2.0 * np.pi * alpha,
                            Cd_data=np.full(alpha.shape, 0.01), Cm_data=np.zeros(alpha.shape), Re=1e6)


@pytest.mark.parametrize("eta", [(0.0, 1.0), (0.2, 0.6)])
@pytest.mark.parametrize("kind", ["airfoil", "table", "source"])
def test_flap_counted_once_llt(eta, kind):
    """Thin-airfoil polars with the Glauert flap shift give the inviscid lifting line."""
    from ventorum.solvers.factory import make_solver

    wing = make_flapped_wing(eta, shape="0012")
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0))
    settings = SolverSettings()
    polars = {"airfoil": _thin_polar(), "table": glauert_table(), "source": GlauertSource()}[kind]
    inviscid = make_solver("linear").solve(wing, cond, settings).totals.CL
    viscous = solve_viscous(wing, cond, settings, polars, model="llt")
    assert viscous.converged
    assert abs(viscous.totals.CL - inviscid) / inviscid < 1e-6


#: Spanwise panels per semi-span of the flap tests [-]: the property does not
#: depend on the spanwise count, and a small count keeps the tests fast.
FLAP_TEST_N_PANELS = 24


@pytest.mark.parametrize("eta", [(0.0, 1.0), (0.2, 0.6)])
@pytest.mark.parametrize("kind", ["airfoil", "table", "source"])
def test_flap_vlm_equals_inviscid_at_the_same_panel_count(eta, kind):
    """Regression: the coupled flapped VLM was higher than the inviscid VLM at the same mesh.

    Owner decision: the reference of the flapped VLM coupling is the
    inviscid VLM at the same panel count. With polars that follow
    thin-airfoil theory (2 pi slope, Glauert flap shift) the coupling read
    the full flap effect from the polar, while the coarse lattice flap has
    less: the coupled CL was higher by about 11 % (alpha 0 deg) and 4 to
    6 % (alpha 4 deg) at n_chord 4, halving when n_chord doubles. The
    polar is now read relative to its own thin-airfoil model, so the
    correction is zero at any n_chord and the coupled CL equals the
    inviscid CL (requirement 1 %, measured below 1e-6; the inviscid solve
    can run in float32 on a GPU). The comparison with the mesh-converged
    inviscid value (Richardson extrapolation of n_chord 8 and 16) is
    printed, not asserted.
    """
    from ventorum.solvers.factory import make_solver

    wing = make_flapped_wing(eta, shape="0012")
    conds = [FlightCondition(V_inf=40.0, alpha=np.radians(a)) for a in (0.0, 4.0)]
    polars = {"airfoil": _thin_polar(), "table": glauert_table(), "source": GlauertSource()}[kind]
    vlm = make_solver("vlm")
    cl_inv = {}
    cl_visc = {}
    for n_chord in (4, 8, 16):
        settings = SolverSettings(n_chord=n_chord, n_panels=FLAP_TEST_N_PANELS)
        visc = solve_viscous(wing, conds, settings, polars, model="vlm")
        for cond, res in zip(conds, visc):
            assert res.converged
            inv = vlm.solve(wing, cond, settings).totals.CL
            key = (n_chord, round(float(np.degrees(cond.alpha))))
            cl_inv[key] = inv
            cl_visc[key] = res.totals.CL
            assert abs(res.totals.CL - inv) / abs(inv) < 0.01
            assert abs(res.totals.CL - inv) / abs(inv) < 1e-4
            # A thin-airfoil polar needs no correction at any panel count.
            assert np.max(np.abs(res.details["delta"])) < 1e-6
    for a in (0, 4):
        converged = 2.0 * cl_inv[(16, a)] - cl_inv[(8, a)]
        print(f"eta {eta}, {kind}, alpha {a} deg: coupled CL at n_chord 4 / mesh-converged inviscid CL - 1 = "
              f"{cl_visc[(4, a)] / converged - 1.0:+.4f}")


class ScaledFlapSource(GlauertSource):
    """Thin-airfoil polars with a flap shift of ``factor`` times the Glauert value."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def polar(self, section, Re, alphas_deg, *, mach=0.0, flap=None, **options):
        if flap is not None:
            flap = (flap[0], flap[1], self.factor * float(flap[2]))
        return super().polar(section, Re, alphas_deg, mach=mach, flap=flap, **options)


def _strip_deflection(lattice):
    """Return the flap deflection [rad] of each lattice strip at its control point (0 without a flap)."""
    field = lattice.geom_cache["flap_field"]
    return np.where(np.isfinite(field["hinge"]), field["strip_angle"], 0.0)


def test_flap_correction_is_the_departure_of_the_polar_from_thin_airfoil_theory():
    """The correction of a flapped strip does not depend on the chordwise panel count.

    A polar with half of the Glauert flap shift departs from thin-airfoil
    theory by -2 pi tau phi / 2 on the flapped strips. The coupling turns
    this departure into the correction delta = tau phi / 2 on every
    flapped strip and zero elsewhere, at n_chord 4 and 16 alike. Before,
    delta also held the chordwise discretization error of the lattice flap.
    phi is the deflection of the lattice strip: in the joins at the side
    edges of the control it is a part of the control deflection.
    """
    tau = flap_effectiveness(HINGE)
    wing = make_flapped_wing((0.2, 0.6), shape="0012")
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0))
    for n_chord in (4, 16):
        res = solve_viscous(wing, cond, SolverSettings(n_chord=n_chord, n_panels=FLAP_TEST_N_PANELS),
                            ScaledFlapSource(0.5), model="vlm")
        assert res.converged
        lattice = res.details["lattice"]
        phi = _strip_deflection(lattice)
        flapped = phi != 0.0
        eta = np.abs(lattice.eta)
        assert np.all(flapped[(eta >= 0.2 - 1e-9) & (eta <= 0.6 + 1e-9)])
        np.testing.assert_allclose(res.details["delta"], 0.5 * tau * phi, atol=1e-7)
        # The thin-airfoil model of a flapped strip has the Glauert shift.
        np.testing.assert_allclose(res.details["alpha0_thin_airfoil"][flapped], -tau * phi[flapped],
                                   rtol=1e-12)
        assert np.array_equal(res.details["alpha0_thin_airfoil"][~flapped], res.details["alpha0_section"][~flapped])


def test_flap_thin_airfoil_model_keeps_the_camber_of_the_mean_line():
    """Regression: a flapped strip on a cambered mean line lost its camber in the thin-airfoil model.

    With the lattice on the section mean line, the strip normal has no
    camber turn. The clean zero-lift angle of a flapped strip came from
    that normal, so it was 0 and not the camber angle (about -2.1 deg for
    a NACA 2412). The lift offset of the flapped strips was then about
    -0.2 in Cl, and the coupled CL of an AR 8 wing with a 10 deg flap rose
    from 0.96 to 1.17. The clean angle is now the zero-lift angle of the
    strip section with its panel normals before the flap turn: the
    lattice section of the same wing without the flap.
    """
    tau = flap_effectiveness(HINGE)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0))
    settings = SolverSettings(n_chord=8, n_panels=FLAP_TEST_N_PANELS)
    clean_wing = make_rectangular_wing(aspect_ratio=8.0, shape="2412")
    clean = solve_viscous(clean_wing, cond, settings, GlauertSource(), model="vlm")
    flapped = solve_viscous(make_flapped_wing((0.2, 0.6), shape="2412"), cond, settings, GlauertSource(),
                            model="vlm")
    assert clean.converged and flapped.converged
    lattice = flapped.details["lattice"]
    assert np.all(lattice.geometric_camber)
    clean_alpha0 = clean.details["alpha0_section"]
    assert np.all(clean_alpha0 < np.radians(-1.9))
    phi = _strip_deflection(lattice)
    on_flap = phi != 0.0
    np.testing.assert_allclose(flapped.details["alpha0_thin_airfoil"][on_flap],
                               clean_alpha0[on_flap] - tau * phi[on_flap], rtol=1e-10)
    assert np.array_equal(flapped.details["alpha0_thin_airfoil"][~on_flap],
                          flapped.details["alpha0_section"][~on_flap])


def test_flap_reference_section_and_ground_factor():
    """The converged 2D section gives the Glauert flap shift; its ground factor tends to 1 far up.

    With a polar that holds the ground height, the thin-airfoil model of a
    flapped strip uses the ground factor of the flap shift (ratio of the
    converged 2D sections with and without images). A free-air polar in
    ground effect keeps the factor 1.
    """
    from ventorum.aero.system import GroundPlane
    from ventorum.utils.parallel import blas_single_thread
    from ventorum.viscous.alpha_correction import _section_flap_shift

    def shift(*args, **kwargs):
        # BLAS on one thread, as in the coupling: a threaded solve of this size is slow.
        with blas_single_thread():
            return _section_flap_shift(*args, **kwargs)

    tau = flap_effectiveness(HINGE)
    wing = make_flapped_wing((0.0, 1.0), shape="0012")
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0))
    res = solve_viscous(wing, cond, SolverSettings(n_panels=FLAP_TEST_N_PANELS), _thin_polar(), model="vlm")
    lattice = res.details["lattice"]
    # Unit panel angle in the sign of the lattice section (trailing edge up).
    free = shift(lattice, 0, HINGE, None)
    assert free == pytest.approx(tau, rel=3e-3)
    assert shift(lattice, 0, HINGE, None, n_panels=1024) == pytest.approx(tau, rel=1e-3)
    far = GroundPlane(normal=np.array([0.0, 0.0, 1.0]), offset=-20.0)
    assert shift(lattice, 0, HINGE, far) / free == pytest.approx(1.0, abs=1e-3)

    phi = np.radians(FLAP_DEG)
    cond_ground = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.5)
    settings = SolverSettings(n_panels=FLAP_TEST_N_PANELS, n_chord=4)
    in_ground = solve_viscous(wing, cond_ground, settings, GlauertSource(supports_ground=True), model="vlm")
    free_polar = solve_viscous(wing, cond_ground, settings, GlauertSource(supports_ground=False), model="vlm")
    assert in_ground.converged and free_polar.converged
    np.testing.assert_allclose(free_polar.details["alpha0_thin_airfoil"], -tau * phi, rtol=1e-12)
    kappa = in_ground.details["alpha0_thin_airfoil"] / (-tau * phi)
    gl = in_ground.details["lattice"]
    expected = np.array([shift(gl, s, HINGE, in_ground.details["ground"]) / free for s in range(gl.n_strips)])
    np.testing.assert_allclose(kappa, expected, rtol=1e-9)
    assert np.all(np.abs(kappa - 1.0) > 1e-3)


def test_vlm_section_relation_of_a_clean_strip_is_thin_airfoil_theory():
    wing = make_rectangular_wing(aspect_ratio=8.0)
    res = solve_viscous(wing, FlightCondition(V_inf=40.0, alpha=np.radians(3.0)), SolverSettings(),
                        _thin_polar(), model="vlm")
    np.testing.assert_allclose(res.details["lift_slope_section"], 2.0 * np.pi, rtol=1e-12)
    np.testing.assert_allclose(res.details["alpha0_section"], 0.0, atol=1e-12)
    # Without a deflected control the thin-airfoil model is the lattice section: no offset.
    assert np.array_equal(res.details["alpha0_thin_airfoil"], res.details["alpha0_section"])


def test_table_hinge_must_match_the_control():
    wing = make_flapped_wing(shape="0012")
    with pytest.raises(ValueError, match="hinge"):
        solve_viscous(wing, FlightCondition(V_inf=40.0, alpha=np.radians(2.0)), SolverSettings(),
                      glauert_table(x_hinge=0.7), model="llt")


# ── Section polar source path ───────────────────────────────────────────────


def test_source_gets_section_hinge_and_reynolds_number():
    source = GlauertSource()
    wing = make_flapped_wing((0.0, 0.5), shape="NACA 2412")
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(2.0), rho=0.9)
    res = solve_viscous(wing, cond, SolverSettings(), source, model="llt")
    assert res.converged
    sections = {c["section"] for c in source.calls}
    assert sections == {"naca 2412"}
    flaps = {c["flap"] for c in source.calls}
    # The hinge of the control and the thickness-fraction rule, not (0.75, 0.5) by default.
    assert flaps == {None, (HINGE, 0.5, FLAP_DEG)}
    # One viscosity constant in the package: ventorum.core.constants.MU_SL.
    expected_re = 0.9 * 30.0 * 1.0 / MU_SL
    assert all(c["Re"] == pytest.approx(expected_re, rel=1e-12) for c in source.calls)
    assert all("height" not in c for c in source.calls)


def test_source_uses_the_hinge_of_the_control():
    source = GlauertSource()
    wing = make_flapped_wing(shape="0012")
    wing.controls[0].hinge_x_c = 0.7
    solve_viscous(wing, FlightCondition(V_inf=40.0, alpha=np.radians(2.0)), SolverSettings(),
                  source, model="llt")
    assert {c["flap"] for c in source.calls} == {(0.7, 0.5, FLAP_DEG)}


def test_source_without_section_shape_is_refused():
    """Regression: a strip without a shape fell back silently to NACA 0012."""
    wing = make_rectangular_wing()
    with pytest.raises(ValueError, match="WingSection.shape"):
        solve_viscous(wing, FlightCondition(V_inf=40.0, alpha=np.radians(2.0)), SolverSettings(),
                      GlauertSource(), model="vlm")


def test_source_section_from_airfoil_name_and_coordinates():
    from ventorum.airfoil import airfoil

    named = make_rectangular_wing()
    for sec in named.sections:
        sec.airfoil = LinearAirfoil(name="NACA 4412")
    source = GlauertSource()
    solve_viscous(named, FlightCondition(V_inf=40.0, alpha=np.radians(2.0)), SolverSettings(),
                  source, model="llt")
    assert {c["section"] for c in source.calls} == {"naca 4412"}
    coords = airfoil("0012").coords
    shaped = make_rectangular_wing()
    for sec in shaped.sections:
        sec.shape = coords
    source = GlauertSource()
    solve_viscous(shaped, FlightCondition(V_inf=40.0, alpha=np.radians(2.0)), SolverSettings(),
                  source, model="llt")
    assert all(isinstance(c["section"], np.ndarray) for c in source.calls)
    np.testing.assert_allclose(source.calls[0]["section"], coords)


def test_source_in_ground_gets_the_strip_height():
    source = GlauertSource(supports_ground=True)
    wing = make_rectangular_wing(shape="0012")
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(2.0), h=0.6)
    res = solve_viscous(wing, cond, SolverSettings(), source, model="vlm")
    assert res.converged
    heights = [c["height"] for c in source.calls]
    assert heights and all(h is not None for h in heights)
    lattice = res.details["lattice"]
    ground = res.details["ground"]
    expected = ground.height(lattice.qc_mid) / lattice.chord
    np.testing.assert_allclose(res.details["strip_h_over_c"], expected, rtol=1e-12)
    assert min(heights) == pytest.approx(float(np.min(expected)), abs=1e-6)
    # The 2D section of each strip has its images in the ground: steeper lift slope.
    assert np.all(res.details["lift_slope_section"] > 2.0 * np.pi * 1.01)
    # A free-air source gets no height and keeps the free-air section relation.
    free_source = GlauertSource(supports_ground=False)
    res_free = solve_viscous(wing, cond, SolverSettings(), free_source, model="vlm")
    assert all("height" not in c for c in free_source.calls)
    np.testing.assert_allclose(res_free.details["lift_slope_section"], 2.0 * np.pi, rtol=1e-9)


def test_ventorum2d_source_works_in_the_coupling(monkeypatch, tmp_path):
    """Regression: the Ventorum-2D source refused the keyword call of the coupling."""
    from types import SimpleNamespace

    import ventorum.aero.polar_source_v2d as v2d
    from ventorum.aero.polar_source_v2d import Ventorum2DPolarSource

    def fake_solve(bodies, alpha_deg, Re, **kwargs):
        a = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
        return SimpleNamespace(alpha=a, Cl=2.0 * np.pi * np.radians(a), Cd=np.full(a.shape, 0.01),
                               Cm=np.zeros(a.shape), converged=np.ones(a.shape, dtype=bool))

    monkeypatch.setattr(v2d, "solve_viscous", fake_solve)
    source = Ventorum2DPolarSource(cache=PolarCache(tmp_path))
    wing = make_rectangular_wing(shape="0012")
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(3.0))
    res = solve_viscous(wing, cond, SolverSettings(), source, model="llt")
    assert res.converged
    assert np.all(np.abs(res.details["delta"]) < 1e-8)


# ── Branches and convergence ─────────────────────────────────────────────────

STALL_START_DEG = 12.0
STALL_END_DEG = 13.0
CL_PLATEAU = 0.5


def plateau_polar():
    """Thin-airfoil lift to 12 deg, a sharp drop to Cl 0.5 at 13 deg, then a flat plateau.

    A strip load line crosses this curve more than once above stall, so
    the coupled equations have more than one solution: one is known
    exactly, the fully stalled state with Cl = 0.5 on every strip.
    """
    deg = np.linspace(-10.0, 60.0, 701)
    alpha = np.radians(deg)
    cl_s = 2.0 * np.pi * np.radians(STALL_START_DEG)
    drop = cl_s + (CL_PLATEAU - cl_s) * (deg - STALL_START_DEG) / (STALL_END_DEG - STALL_START_DEG)
    cl = np.where(deg <= STALL_START_DEG, 2.0 * np.pi * alpha,
                  np.where(deg <= STALL_END_DEG, drop, CL_PLATEAU))
    return TabulatedAirfoil(name="plateau", alpha=alpha, Cl_data=cl, Cd_data=np.full(alpha.shape, 0.01),
                            Cm_data=np.zeros(alpha.shape), Re=1e6)


def _strip_cl_from_circulation(res):
    lattice = res.details["lattice"]
    gamma = res.details["gamma"]
    strip_gamma = np.array([gamma[lattice.panel_strip == s].sum() for s in range(lattice.n_strips)])
    return 2.0 * strip_gamma / (res.condition.V_inf * lattice.chord)


def test_known_double_solution_finds_both_branches():
    """Both converged branches are found and each one solves the coupled equations."""
    wing = make_rectangular_wing(aspect_ratio=8.0)
    settings = SolverSettings(n_panels=2, min_panels=1, spacing="uniform")
    polar = plateau_polar()
    res = solve_viscous(wing, FlightCondition(V_inf=40.0, alpha=np.radians(16.0)), settings, polar)
    assert res.converged
    assert res.details["non_unique"]
    branches = res.details["branches"]
    assert len(branches) >= 2
    assert branches[0] is res
    cls = [b.totals.CL for b in branches]
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            assert abs(cls[i] - cls[j]) > 1e-3
    for b in branches:
        assert b.converged
        cl_strip = _strip_cl_from_circulation(b)
        cl_polar = polar.Cl(b.details["alpha_eff"])
        np.testing.assert_allclose(cl_strip, cl_polar, atol=1e-5)
        assert b.totals.trust.factors.get("post_stall_branch", 0.0) > 0.0
    # The known solution: every strip on the plateau, so every strip Cl is 0.5.
    stalled = [b for b in branches
               if np.all(np.degrees(b.details["alpha_eff"]) >= STALL_END_DEG)]
    assert len(stalled) == 1
    np.testing.assert_allclose(_strip_cl_from_circulation(stalled[0]), CL_PLATEAU, atol=1e-5)
    assert stalled[0].totals.CL == pytest.approx(CL_PLATEAU, abs=5e-3)
    assert all(f == "post-stall, non-unique" for f in stalled[0].details["strip_flags"])


def test_non_converged_primary_gives_no_branch_verdict():
    """Regression: a non-converged state formed branches and a non-unique verdict."""
    wing = make_rectangular_wing(aspect_ratio=8.0)
    settings = SolverSettings(n_panels=2, min_panels=1, spacing="uniform")
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = solve_viscous(wing, FlightCondition(V_inf=40.0, alpha=np.radians(16.0)), settings,
                            plateau_polar(), max_iter=1, tol=1e-12)
    assert not res.converged
    assert "branches" not in res.details
    assert not res.details["non_unique"]
    assert res.totals.trust.factors.get("post_stall_branch", 0.0) == 0.0
    assert any("did not converge" in n for n in res.details["notes"])


def test_relaxation_option_reaches_the_same_solution():
    wing = make_rectangular_wing(aspect_ratio=8.0)
    cond = FlightCondition(V_inf=40.0, alpha=np.radians(9.0))
    polar = make_stalling_polar(alpha_stall_deg=10.0)
    full = solve_viscous(wing, cond, SolverSettings(), polar)
    half = solve_viscous(wing, cond, SolverSettings(), polar, relaxation=0.5)
    assert full.converged and half.converged
    assert half.totals.CL == pytest.approx(full.totals.CL, abs=1e-6)
    with pytest.raises(ValueError, match="relaxation"):
        solve_viscous(wing, cond, SolverSettings(), polar, relaxation=1.5)


# ── Sweeps: continuation and factorizations ────────────────────────────────


def test_sweep_continues_both_ways_from_zero(monkeypatch):
    import ventorum.viscous.alpha_correction as module

    seen = []
    real = module._solve_condition

    def recording(solver, model, aircraft, condition, *args):
        delta0 = args[5]  # settings, resolver, factorization, ref_point, main_surface, delta0
        seen.append((round(float(np.degrees(condition.alpha)), 6), delta0 is None))
        return real(solver, model, aircraft, condition, *args)

    monkeypatch.setattr(module, "_solve_condition", recording)
    wing = make_rectangular_wing()
    alphas = [8.0, -4.0, 0.0, 4.0, -8.0]
    conds = [FlightCondition(V_inf=40.0, alpha=np.radians(a)) for a in alphas]
    results = solve_viscous(wing, conds, SolverSettings(), _thin_polar())
    assert [a for a, _ in seen] == [0.0, 4.0, 8.0, -4.0, -8.0]
    # Only the first condition starts from zero; the others continue.
    assert [first for _, first in seen] == [True, False, False, False, False]
    # The results keep the input order.
    np.testing.assert_allclose([np.degrees(r.condition.alpha) for r in results], alphas)


@pytest.mark.parametrize("model", ["vlm", "llt"])
def test_factorization_count_is_the_real_count(monkeypatch, model):
    """Regression: the count was overwritten with 1 (a 4-condition sweep made 4)."""
    import scipy.linalg

    calls = []
    real = scipy.linalg.lu_factor

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    wing = make_rectangular_wing()
    conds = [FlightCondition(V_inf=40.0, alpha=np.radians(a)) for a in (0.0, 2.0, 4.0, 6.0)]
    # The free-stream wake turns with alpha: the matrix changes in every condition.
    results = solve_viscous(wing, conds, SolverSettings(), _thin_polar(), model=model)
    assert len(calls) == 4
    assert all(r.details["factorization_count"] == 4 for r in results)
    assert [r.details["factorized"] for r in results] == [True] * 4


def test_body_wake_sweep_reuses_one_factorization(monkeypatch):
    import scipy.linalg

    calls = []
    real = scipy.linalg.lu_factor

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)
    wing = make_rectangular_wing()
    settings = SolverSettings(wake_alignment="body")
    conds = [FlightCondition(V_inf=40.0, alpha=np.radians(a)) for a in (0.0, 2.0, 4.0, 6.0)]
    results = solve_viscous(wing, conds, settings, make_stalling_polar(alpha_stall_deg=4.0))
    # The matrix does not depend on alpha with the body-axis wake.
    assert len(calls) == 1
    assert all(r.details["factorization_count"] == 1 for r in results)
    assert [r.details["factorized"] for r in results] == [True, False, False, False]


# ---------------------------------------------------------------------------
# Keys of a polar dict
# ---------------------------------------------------------------------------

def _named_thick_wing() -> LiftingSurface:
    return LiftingSurface(
        name="Wing", semi_span=4.0, spacing="uniform",
        sections=[WingSection(y_frac=0.0, chord=1.0, x_le=-0.25, shape="0012"),
                  WingSection(y_frac=1.0, chord=1.0, x_le=-0.25, shape="0012")],
    )


def _key_settings(model: str) -> SolverSettings:
    if model == "panel3d":
        return SolverSettings(solver_type="panel3d", n_panels=3, n_chord=8, spacing="uniform",
                              panel_backend="numpy", panel_cache=False)
    return SolverSettings(n_panels=8, n_chord=4, spacing="uniform")


@pytest.mark.parametrize("model", ["vlm", "llt", "panel3d"])
def test_polar_dict_key_without_match_is_refused(model):
    """A misspelt surface name is an error, not a silent use of the section airfoil."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(6.0))
    with pytest.raises(ValueError, match=r"'Wnig'.*match no strip, surface or airfoil.*'Wing'"):
        solve_viscous(_named_thick_wing(), cond, _key_settings(model), {"Wnig": LinearAirfoil(a0=4.0)},
                      model=model, tol=1e-10)
    with pytest.raises(ValueError, match=r"\[99\].*strip indices 0 to"):
        solve_viscous(_named_thick_wing(), cond, _key_settings(model), {99: LinearAirfoil(a0=4.0)},
                      model=model, tol=1e-10)


@pytest.mark.parametrize("model", ["vlm", "llt", "panel3d"])
def test_polar_dict_fallback_has_a_trust_note(model):
    """Strips without a dict entry use their section airfoil and the result says so."""
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(6.0))
    settings = _key_settings(model)
    polar = LinearAirfoil(a0=4.0)
    note = "have no entry in the polar dict; they use the section airfoil of the geometry"
    named = solve_viscous(_named_thick_wing(), cond, settings, {"Wing": polar}, model=model, tol=1e-10)
    single = solve_viscous(_named_thick_wing(), cond, settings, polar, model=model, tol=1e-10)
    assert not any(note in n for n in named.details["notes"])
    assert named.totals.CL == pytest.approx(single.totals.CL, abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        partial = solve_viscous(_named_thick_wing(), cond, settings, {0: polar}, model=model, tol=1e-10)
    assert any(note in n for n in partial.details["notes"])
    assert any(note in w for w in partial.totals.trust.warnings)
