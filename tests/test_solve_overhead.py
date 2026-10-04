# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parity tests for solve overhead optimizations.

Tests verify:
1. The vector cross product helper matches numpy.cross bit for bit.
2. The geometry cache in compute_loads gives identical results to uncached loads.
3. Angle-of-attack sweeps match single solve results.
4. The airfoil blender fast path returns identical objects or equal values.
"""

from __future__ import annotations

import bisect
import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.loads import _cross, compute_loads
from ventorum.aero.system import ground_plane_from_condition
from ventorum.geometry.lattice import (
    _AirfoilBlender,
    _sorted_sections,
    airfoil_linear_properties,
)


def _wing_and_tail() -> vt.Aircraft:
    """Return a two-surface aircraft for testing."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=4.0,
        n_panels=8,
        dihedral=np.radians(5.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=1.0, chord=0.6),
        ],
    )
    tail = vt.LiftingSurface(
        name="tail",
        semi_span=1.2,
        n_panels=4,
        position=np.array([3.5, 0.0, 0.2]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.5),
            vt.WingSection(y_frac=1.0, chord=0.4),
        ],
    )
    return vt.Aircraft(surfaces=[wing, tail])


def _tapered_wing(
    af_root: vt.AirfoilType | None = None,
    af_tip: vt.AirfoilType | None = None,
) -> vt.Aircraft:
    """Return a single tapered wing aircraft."""
    root = af_root if af_root is not None else vt.LinearAirfoil()
    tip = af_tip if af_tip is not None else root
    surf = vt.LiftingSurface(
        name="wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5, airfoil=root),
            vt.WingSection(y_frac=1.0, chord=0.75, airfoil=tip),
        ],
    )
    return vt.Aircraft(surfaces=[surf])


@pytest.fixture(autouse=True)
def _restore_kernel_backend():
    """Restore the kernel backend after each test."""
    old = vt.aero.vortex._kernel_backend
    yield
    vt.aero.vortex.set_kernel_backend(old)


def test_cross_helper_equals_numpy():
    """Verify that _cross gives the same bits as numpy.cross for float64 inputs."""
    rng = np.random.default_rng(42)

    # Shape (50, 3) against (50, 3)
    u_50 = rng.standard_normal((50, 3))
    v_50 = rng.standard_normal((50, 3))
    assert np.array_equal(_cross(u_50, v_50), np.cross(u_50, v_50))

    # Shape (3,) against (50, 3)
    u_3 = rng.standard_normal(3)
    assert np.array_equal(_cross(u_3, v_50), np.cross(u_3, v_50))

    # Shape (50, 3) against (3,)
    assert np.array_equal(_cross(u_50, u_3), np.cross(u_50, u_3))

    # Shape (3,) against (3,)
    v_3 = rng.standard_normal(3)
    assert np.array_equal(_cross(u_3, v_3), np.cross(u_3, v_3))


@pytest.mark.parametrize(
    "case_id",
    [
        "vlm_1chord",
        "vlm_4chord",
        "llt_linear",
        "sideslip_3deg",
        "ground_effect",
        "wing_and_tail",
        "blended_airfoils",
    ],
)
def test_cached_loads_equal_uncached(case_id: str):
    """Verify that cached loads equal uncached loads exactly."""
    vt.aero.vortex.set_kernel_backend("numpy")
    if case_id == "vlm_1chord":
        ac = _tapered_wing()
        solver = vt.VortexLatticeSolver()
        st = vt.SolverSettings(n_panels=20, n_chord=1)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        leg_forces = True
    elif case_id == "vlm_4chord":
        ac = _tapered_wing()
        solver = vt.VortexLatticeSolver()
        st = vt.SolverSettings(n_panels=20, n_chord=4)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        leg_forces = True
    elif case_id == "llt_linear":
        ac = _tapered_wing()
        solver = vt.LinearLLTSolver()
        st = vt.SolverSettings(n_panels=20, n_chord=1)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        leg_forces = False
    elif case_id == "sideslip_3deg":
        ac = _tapered_wing()
        solver = vt.VortexLatticeSolver()
        st = vt.SolverSettings(n_panels=20, n_chord=1)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))
        leg_forces = True
    elif case_id == "ground_effect":
        ac = _tapered_wing()
        ac.compute_reference_values()
        solver = vt.VortexLatticeSolver()
        st = vt.SolverSettings(n_panels=20, n_chord=1)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0), h=0.5 * ac.c_ref)
        leg_forces = True
    elif case_id == "wing_and_tail":
        ac = _wing_and_tail()
        solver = vt.VortexLatticeSolver()
        st = vt.SolverSettings(n_panels=8, n_chord=1)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        leg_forces = True
    elif case_id == "blended_airfoils":
        af_root = vt.LinearAirfoil(name="root", a0=6.28, alpha_L0=0.0, Cd0=0.01, Cm0=-0.05)
        af_tip = vt.LinearAirfoil(name="tip", a0=5.80, alpha_L0=-0.02, Cd0=0.015, Cm0=-0.02)
        ac = _tapered_wing(af_root=af_root, af_tip=af_tip)
        solver = vt.VortexLatticeSolver()
        st = vt.SolverSettings(n_panels=20, n_chord=1)
        cond = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
        leg_forces = True
    else:
        raise ValueError(f"Unknown case {case_id}")

    ac.compute_reference_values()
    rp = ac.moment_reference()
    ground = None if cond.h is None else ground_plane_from_condition(
        solver.build(ac, st, cond, None, rp), cond, rp
    )
    res = solver.solve(ac, cond, st, ground=ground, ref_point=rp)
    lat = res.details["lattice"]
    gamma = res.details["gamma"]
    alpha_eff = res.details["loads"].alpha_eff
    wake_dir = res.details["wake_dir"]

    # First evaluate uncached loads with kernel_cache = None
    lat.kernel_cache = None
    uncached = compute_loads(
        lat,
        gamma,
        cond,
        ac.S_ref,
        ac.b_ref,
        ac.c_ref,
        ground=ground,
        ref_point=rp,
        alpha_eff_strip=alpha_eff,
        wake_dir=wake_dir,
        leg_forces=leg_forces,
    )

    # Next evaluate with kernel_cache = {} twice (second call uses cache)
    lat.kernel_cache = {}
    compute_loads(
        lat,
        gamma,
        cond,
        ac.S_ref,
        ac.b_ref,
        ac.c_ref,
        ground=ground,
        ref_point=rp,
        alpha_eff_strip=alpha_eff,
        wake_dir=wake_dir,
        leg_forces=leg_forces,
    )
    cached = compute_loads(
        lat,
        gamma,
        cond,
        ac.S_ref,
        ac.b_ref,
        ac.c_ref,
        ground=ground,
        ref_point=rp,
        alpha_eff_strip=alpha_eff,
        wake_dir=wake_dir,
        leg_forces=leg_forces,
    )

    # Compare totals
    assert cached.totals.CL == uncached.totals.CL
    assert cached.totals.CDi == uncached.totals.CDi
    assert cached.totals.CD_total == uncached.totals.CD_total
    assert cached.totals.CY == uncached.totals.CY
    assert cached.totals.Cl == uncached.totals.Cl
    assert cached.totals.Cm == uncached.totals.Cm
    assert cached.totals.Cn == uncached.totals.Cn

    # Compare strip arrays
    assert np.array_equal(cached.strip_gamma, uncached.strip_gamma)
    assert np.array_equal(cached.strip_force, uncached.strip_force)
    assert np.array_equal(cached.alpha_eff, uncached.alpha_eff)

    assert len(cached.spanwise) == len(uncached.spanwise)
    for c_sp, u_sp in zip(cached.spanwise, uncached.spanwise):
        assert np.array_equal(c_sp.gamma, u_sp.gamma)
        assert np.array_equal(c_sp.Cl, u_sp.Cl)
        assert np.array_equal(c_sp.Cd_i, u_sp.Cd_i)
        assert np.array_equal(c_sp.alpha_eff, u_sp.alpha_eff)
        assert np.array_equal(c_sp.alpha_i, u_sp.alpha_i)
        assert np.array_equal(c_sp.local_lift, u_sp.local_lift)
        if c_sp.Cd_profile is not None:
            assert np.array_equal(c_sp.Cd_profile, u_sp.Cd_profile)
        if c_sp.chord is not None:
            assert np.array_equal(c_sp.chord, u_sp.chord)
        if c_sp.Cm_section is not None:
            assert np.array_equal(c_sp.Cm_section, u_sp.Cm_section)


def test_sweep_equals_single_solves():
    """Verify that an alpha sweep matches single solve results."""
    ac = _tapered_wing()
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=1)
    alphas = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    sweep = vt.analyze_sweep(ac, alphas, settings=st, n_jobs=1)
    for a, r in zip(alphas, sweep):
        single = vt.analyze(ac, alpha_deg=float(a), settings=st)
        for name in ("CL", "CDi", "Cm"):
            ref = getattr(single.totals, name)
            got = getattr(r.totals, name)
            # The existing sweep test in test_kernels.py uses this tolerance.
            # The difference comes from the vortex kernel cache.
            assert got == pytest.approx(ref, rel=1e-11, abs=1e-14)


def test_blender_fast_path():
    """Verify that _AirfoilBlender fast path returns identical objects."""
    # Shared airfoil case: must return the exact same object
    shared_af = vt.LinearAirfoil(name="shared", a0=6.0, alpha_L0=0.0, Cd0=0.01, Cm0=0.0)
    surf_shared = vt.LiftingSurface(
        sections=[
            vt.WingSection(y_frac=0.0, airfoil=shared_af),
            vt.WingSection(y_frac=0.5, airfoil=shared_af),
            vt.WingSection(y_frac=1.0, airfoil=shared_af),
        ]
    )
    blender_shared = _AirfoilBlender(_sorted_sections(surf_shared))
    for eta in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        assert blender_shared.at(eta) is shared_af

    # Two distinct airfoils case: compare with reference general path
    af1 = vt.LinearAirfoil(name="root", a0=6.28, alpha_L0=0.0, Cd0=0.01, Cm0=-0.05)
    af2 = vt.LinearAirfoil(name="tip", a0=5.80, alpha_L0=-0.02, Cd0=0.015, Cm0=-0.02)
    secs_two = [
        vt.WingSection(y_frac=0.0, airfoil=af1),
        vt.WingSection(y_frac=1.0, airfoil=af2),
    ]
    blender_two = _AirfoilBlender(secs_two)

    def _general_path_reference(sections: list[vt.WingSection], eta: float) -> vt.AirfoilType:
        fr = [float(s.y_frac) for s in sections]
        if eta <= fr[0]:
            return sections[0].airfoil
        if eta >= fr[-1]:
            return sections[-1].airfoil
        j = bisect.bisect_right(fr, eta) - 1
        j = min(max(j, 0), len(fr) - 2)
        s1, s2 = sections[j].airfoil, sections[j + 1].airfoil
        span = fr[j + 1] - fr[j]
        w = 0.0 if span <= 0 else float((eta - fr[j]) / span)
        if s1 is s2 or w <= 1e-12:
            return s1
        if w >= 1.0 - 1e-12:
            return s2
        if isinstance(s1, vt.LinearAirfoil) and isinstance(s2, vt.LinearAirfoil):
            return vt.LinearAirfoil(
                name=f"{s1.name}|{s2.name}",
                a0=(1 - w) * s1.a0 + w * s2.a0,
                alpha_L0=(1 - w) * s1.alpha_L0 + w * s2.alpha_L0,
                Cd0=(1 - w) * s1.Cd0 + w * s2.Cd0,
                Cm0=(1 - w) * s1.Cm0 + w * s2.Cm0,
            )
        return s1 if w < 0.5 else s2

    for eta in [0.0, 0.2, 0.4, 0.5, 0.7, 1.0]:
        got = blender_two.at(eta)
        ref = _general_path_reference(secs_two, eta)
        props_got = airfoil_linear_properties(got)
        props_ref = airfoil_linear_properties(ref)
        assert np.allclose(props_got, props_ref)
