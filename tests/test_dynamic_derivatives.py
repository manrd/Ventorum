# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests for the public dynamic_derivatives interface and DynamicDerivativesResult.

Covers k -> 0 parity against steady solvers (ventorum.analyze, rate_derivatives,
height_derivatives) at default ring settings, free-air identities, high-aspect-ratio
Theodorsen limits, finite-frequency ground-effect stability, axes transforms,
JSON-safe to_dict() output, every input refusal, and every trust rule.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.errors import GroundStrikeError, InputError
from ventorum.unsteady import DynamicDerivativesResult, dynamic_derivatives


def _rect_wing(
    semi_span: float = 3.0,
    chord: float = 1.0,
    n_panels: int = 8,
    ref_x: float = 0.25,
) -> vt.Aircraft:
    """Build a flat rectangular wing with a chosen moment reference x [m]."""
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=semi_span,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=chord),
            vt.WingSection(y_frac=1.0, chord=chord),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([ref_x * chord, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _tapered_wing(root_chord: float = 1.5, tip_chord: float = 0.5, n_panels: int = 8) -> vt.Aircraft:
    """Build a strongly tapered wing whose trailing-edge ratio leaves [0.8, 1.25]."""
    wing = vt.LiftingSurface(
        name="tapered",
        semi_span=3.0,
        n_panels=n_panels,
        sections=[
            vt.WingSection(y_frac=0.0, chord=root_chord),
            vt.WingSection(y_frac=1.0, chord=tip_chord),
        ],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.35, 0.0, 0.0]))
    ac.compute_reference_values()
    return ac


def _steady_alpha_slopes(
    ac: vt.Aircraft,
    cond: vt.FlightCondition,
    st: vt.SolverSettings,
    d_alpha: float = 1.0e-4,
) -> tuple[float, float]:
    """Return central-difference (CL_alpha, Cm_alpha) [1/rad] from ventorum.analyze."""
    c_p = vt.FlightCondition(
        V_inf=cond.V_inf,
        rho=cond.rho,
        alpha=cond.alpha + d_alpha,
        beta=cond.beta,
        phi=cond.phi,
        h=cond.h,
    )
    c_m = vt.FlightCondition(
        V_inf=cond.V_inf,
        rho=cond.rho,
        alpha=cond.alpha - d_alpha,
        beta=cond.beta,
        phi=cond.phi,
        h=cond.h,
    )
    rp = vt.analyze(ac, c_p, st)
    rm = vt.analyze(ac, c_m, st)
    cl_a = (rp.totals.CL - rm.totals.CL) / (2.0 * d_alpha)
    cm_a = (rp.totals.Cm - rm.totals.Cm) / (2.0 * d_alpha)
    return float(cl_a), float(cm_a)


# ---------------------------------------------------------------------------
# 1. Exports and basic return contract
# ---------------------------------------------------------------------------


def test_public_exports_and_default_k0():
    """dynamic_derivatives and DynamicDerivativesResult are exported from ventorum and ventorum.unsteady."""
    assert vt.dynamic_derivatives is dynamic_derivatives
    assert vt.DynamicDerivativesResult is DynamicDerivativesResult
    assert "dynamic_derivatives" in vt.unsteady.__all__
    assert "DynamicDerivativesResult" in vt.unsteady.__all__

    # Passing a LiftingSurface directly works with default k=(0.0,).
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=6,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0))
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8)
    res = vt.dynamic_derivatives(wing, cond, st)

    assert isinstance(res, DynamicDerivativesResult)
    assert np.array_equal(res.k, np.array([0.0]))
    assert res.axes == "body"
    assert res.n_chord == 8
    assert res.in_ground_effect is False
    assert res.h_m is None
    for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        for suffix in ("alpha", "q", "alpha_dot", "h", "h_dot", "h_dot_ground"):
            key = f"{coeff}_{suffix}"
            assert key in res.k0
            assert key in res.derivatives
            assert res.derivatives[key].shape == (1,)
            assert math.isclose(res.derivatives[key][0], res.k0[key], rel_tol=0.0, abs_tol=1e-15)
        assert res.H_theta[coeff].shape == (1,)
        assert res.H_z[coeff].shape == (1,)
        assert res.H_pureq[coeff].shape == (1,)


# ---------------------------------------------------------------------------
# 2. k -> 0 limits against ventorum.analyze, rate_derivatives and height_derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h_ground", [None, 0.5])
def test_k0_matches_steady_solvers_default_settings(h_ground):
    """At default ring settings (0.25 TE offset and Ventorum core), k -> 0 matches steady VLM within 5e-3 rel.

    Tolerance matches the default-settings check in tests/test_unsteady_harmonic.py
    (5e-3 relative, or 5e-3 absolute when a reference value is near zero, such as
    Cm_alpha about the quarter chord).
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(
        solver_type="vlm",
        n_panels=6,
        n_chord=4,
        spacing="half-cosine",
        chord_spacing="uniform",
    )
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=h_ground)

    cl_a_ref, cm_a_ref = _steady_alpha_slopes(ac, cond, st)
    rd = vt.rate_derivatives(ac, cond, st, step=0.01, axes="body")
    res = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.1))

    tol = 5.0e-3
    assert abs(res.k0["CL_alpha"] - cl_a_ref) <= tol * max(abs(cl_a_ref), 1.0)
    assert abs(res.k0["Cm_alpha"] - cm_a_ref) <= tol * max(abs(cm_a_ref), 1.0)
    assert abs(res.k0["CL_q"] - rd.derivatives["CL_q"]) <= tol * max(abs(rd.derivatives["CL_q"]), 1.0)
    assert abs(res.k0["Cm_q"] - rd.derivatives["Cm_q"]) <= tol * max(abs(rd.derivatives["Cm_q"]), 1.0)

    if h_ground is not None:
        hd = vt.height_derivatives(ac, cond, st, h_step=1.0e-4, alpha_step_deg=1.0e-3, axes="body")
        assert abs(res.k0["CL_h"] - hd.derivatives["CL_h"]) <= tol * max(abs(hd.derivatives["CL_h"]), 1.0)
        assert abs(res.k0["Cm_h"] - hd.derivatives["Cm_h"]) <= tol * max(abs(hd.derivatives["Cm_h"]), 1.0)
        assert abs(res.CL0 - hd.base.totals.CL) <= tol * max(abs(hd.base.totals.CL), 1.0)
        assert abs(res.Cm0 - hd.base.totals.Cm) <= tol * max(abs(hd.base.totals.Cm), 1.0)


def test_loaded_ground_effect_finite_k_q_and_alpha_dot_smooth():
    """In loaded ground effect, C_q(k) and C_alpha_dot(k) stay finite and approach k -> 0 smoothly."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8, spacing="half-cosine")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(5.0), h=0.5)

    res = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.01, 0.05, 0.2))
    assert res.k0["CL_h"] < -0.1
    # At small k = 0.01, CL_q and Cm_q must stay within 1 % of their exact k -> 0 limits (no 1/k^2 blow-up).
    assert abs(res.derivatives["CL_q"][1] - res.k0["CL_q"]) <= 1.0e-2 * abs(res.k0["CL_q"])
    assert abs(res.derivatives["Cm_q"][1] - res.k0["Cm_q"]) <= 1.0e-2 * abs(res.k0["Cm_q"])
    assert abs(res.derivatives["CL_alpha_dot"][1] - res.k0["CL_alpha_dot"]) <= 2.0e-2 * max(
        abs(res.k0["CL_alpha_dot"]), 1.0,
    )
    assert abs(res.derivatives["Cm_alpha_dot"][1] - res.k0["Cm_alpha_dot"]) <= 2.0e-2 * max(
        abs(res.k0["Cm_alpha_dot"]), 1.0,
    )


# ---------------------------------------------------------------------------
# 3. Free-air identities and symmetry
# ---------------------------------------------------------------------------


def test_free_air_height_and_heave_rate_identities():
    """In free air at alpha = 0, C_h = 0 to round-off and C_h_dot + C_alpha = 0 at k -> 0.

    Tolerance: C_h <= 1e-14 (exact zero); C_h_dot + C_alpha <= 1e-4 * max(|C_alpha|, 1)
    at the small-k probe k = 1e-3 used for the k -> 0 heave-rate limit.
    """
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=8)
    st = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=8, spacing="half-cosine")
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=0.0)

    res = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.05, 0.2))
    for coeff in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
        assert abs(res.k0[f"{coeff}_h"]) <= 1.0e-14
        assert abs(res.derivatives[f"{coeff}_h"][0]) <= 1.0e-14
        # Ground part equals C_h_dot + C_alpha identically.
        assert np.allclose(
            res.derivatives[f"{coeff}_h_dot_ground"],
            res.derivatives[f"{coeff}_h_dot"] + res.derivatives[f"{coeff}_alpha"],
            atol=1.0e-14,
            rtol=0.0,
        )
        scale = max(abs(res.k0[f"{coeff}_alpha"]), 1.0)
        assert abs(res.k0[f"{coeff}_h_dot"] + res.k0[f"{coeff}_alpha"]) <= 1.0e-4 * scale
        assert abs(res.k0[f"{coeff}_h_dot_ground"]) <= 1.0e-4 * scale

    # Symmetric wing in symmetric flight: lateral coefficients vanish to round-off.
    for lat_c in ("CY", "Cl", "Cn"):
        for suffix in ("alpha", "q", "alpha_dot", "h", "h_dot", "h_dot_ground"):
            assert np.max(np.abs(res.derivatives[f"{lat_c}_{suffix}"])) <= 1.0e-12


def test_stability_axes_option():
    """axes='stability' keeps forces and Cm unchanged and rotates (Cl, Cn) by alpha about y."""
    wing = vt.LiftingSurface(
        name="asym",
        semi_span=3.0,
        n_panels=6,
        is_symmetric=False,
        dihedral=np.radians(10.0),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac = vt.Aircraft(surfaces=[wing], ref_point=np.array([0.25, 0.0, 0.0]))
    ac.compute_reference_values()
    alpha = np.radians(8.0)
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=alpha)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8)

    r_body = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.2), axes="body")
    r_stab = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.2), axes="stability")
    rd_stab = vt.rate_derivatives(ac, cond, st, step=0.01, axes="stability")

    ca, sa = math.cos(alpha), math.sin(alpha)
    for suffix in ("alpha", "q", "alpha_dot", "h", "h_dot", "h_dot_ground"):
        for c in ("CL", "CD", "CY", "Cm"):
            assert np.allclose(r_stab.derivatives[f"{c}_{suffix}"], r_body.derivatives[f"{c}_{suffix}"], atol=1e-13)
        cl_exp = ca * r_body.derivatives[f"Cl_{suffix}"] + sa * r_body.derivatives[f"Cn_{suffix}"]
        cn_exp = -sa * r_body.derivatives[f"Cl_{suffix}"] + ca * r_body.derivatives[f"Cn_{suffix}"]
        assert np.allclose(r_stab.derivatives[f"Cl_{suffix}"], cl_exp, atol=1e-13)
        assert np.allclose(r_stab.derivatives[f"Cn_{suffix}"], cn_exp, atol=1e-13)

    # Stability-axis pitch-rate moment derivatives match rate_derivatives(..., axes='stability') within 5e-3.
    for c in ("Cl", "Cm", "Cn"):
        ref = rd_stab.derivatives[f"{c}_q"]
        assert abs(r_stab.k0[f"{c}_q"] - ref) <= 5.0e-3 * max(abs(ref), 1.0)


# ---------------------------------------------------------------------------
# 4. High aspect ratio (AR 40 rectangular) against Theodorsen (1935)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_high_aspect_ratio_ar40_matches_theodorsen_thin_airfoil():
    """AR 40 rectangular wing at k -> 0 matches Theodorsen (1935, NACA Report 496) thin-airfoil limits.

    Reference values for a 2D flat plate pitching about the quarter chord (a = -1/2)
    with rates normalised by c / (2 V) (Theodorsen 1935, NACA Report 496):
    * Cm_q = -pi / 4, Cm_alpha_dot = -pi / 4, Cm_q + Cm_alpha_dot = -pi / 2
      (independent of Theodorsen's circulation function C(k) about c/4);
    * CL_q = CL_alpha -> 2 pi (quasi-steady three-quarter-chord strip identity;
      finite AR 40 reduces both CL_q and CL_alpha by about 7.5 % from 2 pi).

    Tolerances: 5 % relative on Cm_q and Cm_alpha_dot against -pi / 4;
    2 % relative on Cm_q + Cm_alpha_dot against -pi / 2;
    1 % relative on CL_q against CL_alpha and 10 % relative against 2 pi.
    """
    ac = _rect_wing(semi_span=20.0, chord=1.0, n_panels=20, ref_x=0.25)
    st = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=8, spacing="half-cosine", chord_spacing="uniform")
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    res = vt.dynamic_derivatives(ac, cond, st, k=(0.0,), wake_chords=10.0)

    cm_q_ref = -math.pi / 4.0
    cm_adot_ref = -math.pi / 4.0
    cm_sum_ref = -math.pi / 2.0

    assert abs(res.k0["Cm_q"] - cm_q_ref) <= 0.05 * abs(cm_q_ref)
    assert abs(res.k0["Cm_alpha_dot"] - cm_adot_ref) <= 0.05 * abs(cm_adot_ref)
    assert abs((res.k0["Cm_q"] + res.k0["Cm_alpha_dot"]) - cm_sum_ref) <= 0.02 * abs(cm_sum_ref)
    assert abs(res.k0["CL_q"] - res.k0["CL_alpha"]) <= 0.01 * abs(res.k0["CL_alpha"])
    assert abs(res.k0["CL_q"] - 2.0 * math.pi) <= 0.10 * (2.0 * math.pi)


# ---------------------------------------------------------------------------
# 5. JSON-safe to_dict()
# ---------------------------------------------------------------------------


def test_to_dict_is_strict_json_safe():
    """DynamicDerivativesResult.to_dict() produces strict JSON with no NaN, Inf or complex objects."""
    ac = _rect_wing(semi_span=3.0, chord=1.0, n_panels=6)
    st = vt.SolverSettings(solver_type="vlm", n_panels=6, n_chord=8)
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0), h=0.5)

    res = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.05, 0.2))
    data = res.to_dict()
    encoded = json.dumps(data, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["k"] == [0.0, 0.05, 0.2]
    assert decoded["axes"] == "body"
    assert decoded["in_ground_effect"] is True
    assert decoded["h_m"] == pytest.approx(0.5)
    assert decoded["CL0"] == pytest.approx(res.CL0)
    assert decoded["Cm0"] == pytest.approx(res.Cm0)
    assert decoded["k0"]["CL_alpha"] == pytest.approx(res.k0["CL_alpha"])
    assert len(decoded["derivatives"]["Cm_q"]) == 3
    assert len(decoded["responses"]["H_theta"]["CL"]["re"]) == 3
    assert len(decoded["responses"]["H_theta"]["CL"]["im"]) == 3
    assert decoded["mesh"]["n_chord"] == 8
    assert decoded["trust"]["rating"] == res.base.totals.trust.rating


# ---------------------------------------------------------------------------
# 6. Input checks (item 2)
# ---------------------------------------------------------------------------


def test_input_checks_k_wake_axes_solver_mach():
    """Invalid k, wake_chords, axes, solver_type and Mach > 0.3 raise InputError."""
    ac = _rect_wing(n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4)
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)

    for bad_k in ([-0.1], [0.1, -0.05], [np.nan], [np.inf], [-np.inf], []):
        with pytest.raises(InputError, match="k"):
            vt.dynamic_derivatives(ac, cond, st, k=bad_k)

    for bad_wake in (0.0, -2.0, np.nan, np.inf):
        with pytest.raises(InputError, match="wake_chords"):
            vt.dynamic_derivatives(ac, cond, st, wake_chords=bad_wake)

    for bad_axes in ("wind", "invalid", "all"):
        with pytest.raises(InputError, match="axes"):
            vt.dynamic_derivatives(ac, cond, st, axes=bad_axes)

    for bad_solver in ("linear", "nonlinear", "fourier"):
        st_bad = vt.SolverSettings(solver_type=bad_solver, n_panels=4)
        with pytest.raises(InputError, match="vlm"):
            vt.dynamic_derivatives(ac, cond, st_bad)

    # Mach above the incompressible limit (0.3): both explicit mach and high V_inf.
    cond_mach_04 = vt.FlightCondition(V_inf=50.0, rho=1.225, alpha=0.0, mach=0.4)
    with pytest.raises(InputError, match="Mach|mach"):
        vt.dynamic_derivatives(ac, cond_mach_04, st)

    cond_mach_08 = vt.FlightCondition(V_inf=50.0, rho=1.225, alpha=0.0, mach=0.8)
    with pytest.raises(InputError, match="Mach|mach"):
        vt.dynamic_derivatives(ac, cond_mach_08, st)

    cond_fast = vt.FlightCondition(V_inf=120.0, rho=1.225, alpha=0.0)
    with pytest.raises(InputError, match="Mach|mach"):
        vt.dynamic_derivatives(ac, cond_fast, st)

def _assert_finite_k0(res):
    """Assert that the k -> 0 lift derivatives of a result are finite numbers."""
    assert res.k0, "the result has no k -> 0 derivatives"
    assert all(math.isfinite(v) for v in res.k0.values())
    json.dumps(res.to_dict(), allow_nan=False)


def test_cambered_flapped_and_joined_lattices_run_and_ground_strike_refused():
    """Cambered, flapped and joined lattices give finite derivatives; ground strike raises GroundStrikeError."""
    cond = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=0.0)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4)

    # Cambered mean-line leg chain with n_chord > 1.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w_cam = vt.LiftingSurface(
            name="cam",
            semi_span=3.0,
            n_panels=4,
            sections=[
                vt.WingSection(y_frac=0.0, chord=1.0, shape="4412"),
                vt.WingSection(y_frac=1.0, chord=1.0, shape="4412"),
            ],
        )
        ac_cam = vt.Aircraft(surfaces=[w_cam])
        _assert_finite_k0(vt.dynamic_derivatives(ac_cam, cond, st))

    # Deflected chordwise flap.
    fl = vt.ControlSurface(name="flap", eta_start=0.0, eta_end=1.0, hinge_x_c=0.75, deflection=np.radians(10.0))
    w_fl = vt.LiftingSurface(
        name="flapped",
        semi_span=3.0,
        n_panels=4,
        controls=[fl],
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    ac_fl = vt.Aircraft(surfaces=[w_fl])
    _assert_finite_k0(vt.dynamic_derivatives(ac_fl, cond, st))

    # Surface junction (wing + winglet).
    wing = vt.LiftingSurface(
        name="wing",
        semi_span=3.0,
        n_panels=6,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    winglet = vt.LiftingSurface(
        name="winglet",
        semi_span=0.5,
        n_panels=4,
        is_symmetric=False,
        dihedral=np.radians(90.0),
        position=np.array([0.0, 3.0, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=0.5)],
    )
    ac_j = vt.Aircraft(surfaces=[wing, winglet])
    _assert_finite_k0(vt.dynamic_derivatives(ac_j, cond, st))

    # Ground strike.
    ac_rect = _rect_wing(n_panels=4)
    cond_strike = vt.FlightCondition(V_inf=20.0, rho=1.225, alpha=np.radians(20.0), h=0.05)
    with pytest.raises(GroundStrikeError):
        vt.dynamic_derivatives(ac_rect, cond_strike, st)


# ---------------------------------------------------------------------------
# 7. Trust notes (item 3 and owner decision 3)
# ---------------------------------------------------------------------------


def test_trust_notes_all_four_rules():
    """Every unsteady trust rule fires when violated and stays silent when satisfied."""
    ac_rect = _rect_wing(n_panels=8)
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0))

    # Clean case: n_chord = 8, uniform chord (dx_TE/dx_w = 1), k <= 1, wake_chords >= 5.
    st_clean = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=8)
    r_clean = vt.dynamic_derivatives(ac_rect, cond, st_clean, k=(0.0, 0.2), wake_chords=5.0)
    assert r_clean.trust_notes == []
    assert r_clean.trust is not None
    assert r_clean.trust.score == r_clean.base.totals.trust.score
    assert r_clean.trust.rating == r_clean.base.totals.trust.rating

    # Rule 1: n_chord < 8 warning (steady trust score/rating stays unchanged).
    st_coarse = vt.SolverSettings(solver_type="vlm", n_panels=8, n_chord=4)
    r_coarse = vt.dynamic_derivatives(ac_rect, cond, st_coarse, k=(0.0,), wake_chords=5.0)
    assert any("below 8" in msg for msg in r_coarse.trust_notes)
    assert any("below 8" in msg for msg in r_coarse.trust.warnings)
    assert r_coarse.trust.score == r_coarse.base.totals.trust.score
    assert r_coarse.trust.rating == r_coarse.base.totals.trust.rating

    # Rule 2: trailing-edge length ratio dx_TE,s / dx_w outside [0.8, 1.25].
    ac_tap = _tapered_wing(root_chord=1.5, tip_chord=0.5, n_panels=8)
    r_tap = vt.dynamic_derivatives(ac_tap, cond, st_clean, k=(0.0,), wake_chords=5.0)
    assert any("dx_TE" in msg and "0.8" in msg and "1.25" in msg for msg in r_tap.trust_notes)

    # Rule 3: k > 1.
    r_highk = vt.dynamic_derivatives(ac_rect, cond, st_clean, k=(0.0, 1.5), wake_chords=5.0)
    assert any("1.0" in msg and "verified" in msg for msg in r_highk.trust_notes)

    # Rule 4: wake_chords < 5.
    r_shortw = vt.dynamic_derivatives(ac_rect, cond, st_clean, k=(0.0,), wake_chords=3.0)
    assert any("5 chords" in msg for msg in r_shortw.trust_notes)


def test_dynamic_derivatives_records_device_precision_and_fallback_note(monkeypatch):
    """DynamicDerivativesResult records device and precision (also in to_dict) and fallback notes."""
    from ventorum import gpu

    ac = _rect_wing(n_panels=4)
    st = vt.SolverSettings(solver_type="vlm", n_panels=4, n_chord=4)
    cond = vt.FlightCondition(V_inf=30.0, rho=1.225, alpha=np.radians(4.0))

    old_dev = gpu.get_device()
    try:
        gpu.set_device("cpu")
        r_cpu = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.1))
        assert r_cpu.device == "cpu"
        assert r_cpu.precision == "float64"
        d_cpu = r_cpu.to_dict()
        assert d_cpu["device"] == "cpu"
        assert d_cpu["precision"] == "float64"
        assert not any("GPU unavailable" in msg for msg in r_cpu.notes)

        monkeypatch.setattr(gpu, "get_device", lambda: "gpu")
        monkeypatch.setattr(gpu, "available", lambda: False)
        monkeypatch.setattr(gpu, "unavailable_reason", lambda: "PyTorch finds no CUDA GPU.")
        r_fb = vt.dynamic_derivatives(ac, cond, st, k=(0.0, 0.1))
        assert r_fb.device == "cpu"
        assert r_fb.precision == "float64"
        assert any("GPU unavailable" in msg and "CPU" in msg for msg in r_fb.notes)
        assert r_fb.trust is not None
        assert any("GPU unavailable" in msg and "CPU" in msg for msg in r_fb.trust.warnings)
        assert r_fb.trust.score == r_cpu.trust.score
        assert r_fb.trust.rating == r_cpu.trust.rating
        assert float(r_fb.k0["CL_alpha"]).hex() == float(r_cpu.k0["CL_alpha"]).hex()
    finally:
        gpu.set_device(old_dev)
