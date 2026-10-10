# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Comprehensive test suite for the Ventorum Ground Effect Pipeline.

Verifies:
1. Ground plane: parallel to the free stream, wings level at zero bank, bank lowers the right wing.
2. Physical symmetry: zero roll moment at phi=0, restoring moment at phi>0 (standard sign: Cl < 0).
3. Roll restoring moment grows as height h decreases.
4. Asymmetric induced drag causing induced yaw Cn(phi).
5. Monotonic lift rise and induced drag suppression as h/c -> 0.
6. Ground clearance tracking, ground strike detection, and critical strike bank limits.
7. Multidimensional parameter sweeps and aerodynamic stability derivatives.
8. Multi-surface aircraft configurations (wing + horizontal tail) in ground effect.
9. Tabulated nonlinear airfoil support in ground effect.
10. Multi-worker parallel sweep execution consistency.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import freestream_direction, ground_normal
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.ground_effect.solver import find_bank_strike_limit
from ventorum.ground_effect import (
    analyze_ground_effect,
    GroundEffectSweep,
)


def _mesh(n_panels: int, n_chord: int) -> vt.SolverSettings:
    """Small mesh with a fixed number of chordwise panels (fast tests)."""
    return vt.SolverSettings(n_panels=n_panels, n_chord=n_chord)


@pytest.fixture
def rectangular_wing():
    """Standard AR=10 rectangular wing with chord=1.0 m, span=10.0 m."""
    return vt.LiftingSurface(
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )


@pytest.fixture
def wig_aircraft(rectangular_wing):
    """Complete aircraft with main wing and horizontal stabilizer."""
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=1.5,
        position=np.array([3.5, 0.0, 0.5]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.6),
            vt.WingSection(y_frac=1.0, chord=0.4),
        ],
    )
    ac = vt.Aircraft(name="WIG_Craft", surfaces=[rectangular_wing, tail])
    ac.compute_reference_values()
    return ac


# ─── 1. Ground plane orientation ──────────────────────────────────────────────

def test_ground_plane_orientation():
    """The ground is parallel to the free stream; zero bank is wings level; positive
    bank lowers the right wing; positive alpha puts the trailing edge lower."""
    for a in (0.0, 4.0, -2.0):
        for b in (0.0, 3.0, -5.0):
            for p in (0.0, 2.0, -4.0):
                k = ground_normal(np.radians(a), np.radians(b), np.radians(p))
                assert abs(np.linalg.norm(k) - 1.0) < 1e-14
                assert abs(k @ freestream_direction(np.radians(a), np.radians(b))) < 1e-14
                if p == 0.0:
                    assert abs(k[1]) < 1e-14
    k = ground_normal(0.0, 0.0, np.radians(5.0))
    assert np.array([0.0, 5.0, 0.0]) @ k < np.array([0.0, -5.0, 0.0]) @ k   # right tip lower
    k = ground_normal(np.radians(5.0), 0.0, 0.0)
    assert np.array([1.0, 0.0, 0.0]) @ k < np.array([0.0, 0.0, 0.0]) @ k     # x aft: trailing edge lower


# ─── 2. Physical Symmetry and Restoring Roll Moments ──────────────────────────

def test_zero_roll_symmetry(rectangular_wing):
    """Verify that a symmetric wing at zero roll produces exactly zero rolling moment."""
    res = analyze_ground_effect(rectangular_wing, h=0.5, alpha_deg=4.0, phi_deg=0.0, n_panels=40)
    assert abs(res.Cl) < 1e-12
    assert abs(res.CY) < 1e-12
    assert abs(res.Cn) < 1e-12


def test_positive_roll_restoring_moment(rectangular_wing):
    """Banking in ground effect creates a bank-restoring rolling moment.

    Right wing down (phi > 0) puts it nearer to the ground: more lift on the
    right wing, which rolls the aircraft back to the left. In the standard
    convention (Cl > 0 right wing down) a restoring moment is Cl < 0.
    """
    res_rolled = analyze_ground_effect(rectangular_wing, h=0.5, alpha_deg=4.0, phi_deg=2.0, n_panels=40)
    assert res_rolled.Cl < -0.001
    assert res_rolled.Mx < 0.0
    assert res_rolled.Cl_body < 0.0


def test_roll_restoring_increases_near_ground(rectangular_wing):
    """Verify that roll restoring moment Cl increases monotonically as height decreases."""
    phi = 2.0
    alpha = 4.0
    heights = [2.0, 1.0, 0.5, 0.25]
    cl_moments = []

    for h in heights:
        res = analyze_ground_effect(rectangular_wing, h=h, alpha_deg=alpha, phi_deg=phi, settings=_mesh(30, 4))
        cl_moments.append(res.Cl)

    # The restoring moment (-Cl) grows monotonically as h decreases (closer to ground)
    for i in range(len(cl_moments) - 1):
        assert cl_moments[i+1] < cl_moments[i] < 0.0


# ─── 3. Height Sweep and Ground Effect Augmentation ───────────────────────────

def test_height_sweep_monotonic_lift_rise(rectangular_wing):
    """Verify classical ground effect: CL rises and CDi decreases monotonically as h -> 0."""
    heights = [5.0, 2.0, 1.0, 0.5, 0.2]
    cls = []
    cdis = []
    lds = []

    for h in heights:
        res = analyze_ground_effect(rectangular_wing, h=h, alpha_deg=4.0, phi_deg=0.0, settings=_mesh(30, 4))
        cls.append(res.CL)
        cdis.append(res.CDi)
        lds.append(res.L_over_D)

    # Lift monotonically rises as ground is approached
    for i in range(len(cls) - 1):
        assert cls[i+1] > cls[i]

    # Induced drag factor (CDi / CL^2 = 1 / (pi * AR * e)) monotonically decreases as ground is approached
    cdi_factors = [cdi / (cl**2) for cdi, cl in zip(cdis, cls)]
    for i in range(len(cdi_factors) - 1):
        assert cdi_factors[i+1] < cdi_factors[i]

    # L/D at h=0.5 and h=0.2 is much higher than free air
    assert lds[-1] > lds[0] * 1.5


def test_span_efficiency_in_ground_effect(rectangular_wing):
    """Verify Oswald efficiency factor e exceeds 1.0 in ground effect without artificial clipping."""
    res_near = analyze_ground_effect(rectangular_wing, h=0.2, alpha_deg=4.0, phi_deg=0.0, settings=_mesh(30, 4))
    # At h/c = 0.2 on AR=10, theoretical span efficiency e exceeds 1.5
    assert res_near.e > 1.5


# ─── 4. Ground Clearance and Strike Limits ─────────────────────────────────────

def test_ground_clearance_tracking(rectangular_wing):
    """Verify minimum clearance and tip clearance accuracy."""
    h_ref = 1.0
    phi = 3.0  # ~3 degrees bank on b=10m: tip drop = 5.0 * sin(3 deg) = 0.261 m
    res = analyze_ground_effect(rectangular_wing, h=h_ref, alpha_deg=0.0, phi_deg=phi, n_panels=20)

    expected_tip_drop = 5.0 * np.sin(np.radians(phi))
    assert abs((h_ref - res.h_tip_right) - expected_tip_drop) < 0.05
    assert abs((res.h_tip_left - h_ref) - expected_tip_drop) < 0.05
    assert res.h_min > 0.0


def test_strike_detection_and_limit(rectangular_wing):
    """Verify the critical bank limit, and that a strike gives no result."""
    # At h = 0.5 m, b = 10 m (semi-span = 5 m):
    # Tip strikes ground at phi_crit = arcsin(0.5 / 5.0) = 5.74 degrees
    ac = vt.Aircraft(surfaces=[rectangular_wing])
    sett = vt.SolverSettings(n_panels=10)
    expected_phi = np.degrees(np.arcsin(0.5 / 5.0))
    crit_phi = find_bank_strike_limit(ac, sett, 0.5, 0.0, 0.0, np.zeros(3), "ref")
    assert abs(crit_phi - expected_phi) < 0.05
    # The 'min' convention gives the same limit here (all edges at the same height at alpha = 0).
    assert abs(find_bank_strike_limit(ac, sett, 0.5, 0.0, 0.0, np.zeros(3), "min") - expected_phi) < 0.05

    # Safe at 3 degrees; the bank limit of the result is the exact geometric value
    res_safe = analyze_ground_effect(rectangular_wing, h=0.5, alpha_deg=0.0, phi_deg=3.0, n_panels=20)
    assert res_safe.h_min > 0.0
    assert abs(res_safe.phi_strike_limit - expected_phi) < 0.05

    # Strike at 8 degrees (> 5.74 degrees): refused
    with pytest.raises(GroundStrikeError, match="strike"):
        analyze_ground_effect(rectangular_wing, h=0.5, alpha_deg=0.0, phi_deg=8.0, n_panels=20)


# ─── 5. Parameter Sweeps and Stability Derivatives ────────────────────────────

def test_sweep_engine_and_derivatives(rectangular_wing):
    """Verify multidimensional sweep execution and stability derivatives."""
    sweep = GroundEffectSweep(rectangular_wing, n_jobs=1, settings=_mesh(20, 3))
    heights = [0.3, 0.6, 1.2]
    alphas = [0.0, 4.0, 8.0]
    phis = [0.0, 2.0]

    sweep_res = sweep.run_sweep(heights=heights, alphas_deg=alphas, phis_deg=phis)

    assert sweep_res.grid_shape == (3, 3, 2)
    assert len(sweep_res.results) == 18
    assert sweep_res.CL.shape == (3, 3, 2)

    # Compute stability derivatives
    derivs = sweep_res.compute_stability_derivatives()
    for key in ("CL_alpha", "Cl_phi", "CL_h", "Cm_h", "x_h", "x_alpha", "irodov_margin"):
        assert key in derivs

    # CL_alpha should be higher at lower heights
    assert derivs["CL_alpha"][0] > derivs["CL_alpha"][-1]

    # Roll stiffness: Cl_phi < 0 (restoring, standard sign) and stronger near ground
    assert derivs["Cl_phi"][0] < derivs["Cl_phi"][-1] < 0.0

    # Lift rises as the height falls: CL_h < 0
    assert np.all(derivs["CL_h"][:, 1] < 0.0)


# ─── 6. Multi-Surface Configuration (WIG Craft) ───────────────────────────────

def test_wig_multisurface(wig_aircraft):
    """Verify ground effect pipeline with multi-surface aircraft (wing + tail)."""
    res = analyze_ground_effect(wig_aircraft, h=0.8, alpha_deg=3.0, phi_deg=1.5, n_panels=30)
    assert len(res.transformed_surfaces) == 2
    assert len(res.spanwise) == 2
    assert res.CL > 0.0
    assert res.Cl < 0.0  # Restoring rolling moment (standard sign)


# ─── 7. Tabulated Nonlinear Airfoil Support ────────────────────────────────────

def test_tabulated_airfoil_ground_effect():
    """Verify ground effect with tabulated airfoil (Nonlinear solver)."""
    alphas = np.radians(np.linspace(-10, 20, 61))
    cls = 2.0 * np.pi * alphas
    cds = 0.01 + 0.02 * alphas**2
    af = vt.TabulatedAirfoil(name="NACA0012_Tab", alpha=alphas, Cl_data=cls, Cd_data=cds)

    wing = vt.LiftingSurface(
        semi_span=4.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0, airfoil=af),
            vt.WingSection(y_frac=1.0, chord=1.0, airfoil=af),
        ],
    )
    # The nonlinear lifting line is refused below h_min/c = 1.
    with pytest.raises(ValidityError, match="not valid in ground effect"):
        analyze_ground_effect(wing, h=0.6, alpha_deg=3.0, phi_deg=1.0, n_panels=20, solver="nonlinear")

    # The default VLM uses the linear part of the polar (with a warning) and the polar drag.
    with pytest.warns(RuntimeWarning, match="linear part"):
        res = analyze_ground_effect(wing, h=0.6, alpha_deg=3.0, phi_deg=1.0, n_panels=20)
    assert res.CL > 0.0
    assert res.CDp is not None and res.CDp > 0.0
    assert res.Cl < 0.0

    # Higher up the nonlinear lifting line is allowed.
    res_hi = analyze_ground_effect(wing, h=2.5, alpha_deg=3.0, phi_deg=1.0, n_panels=20, solver="nonlinear")
    assert res_hi.solver_type == "nonlinear"
    assert res_hi.CDp is not None and res_hi.CDp > 0.0


# ─── 8. Multi-Worker Parallel Sweep ───────────────────────────────────────────

def test_parallel_sweep_consistency(rectangular_wing):
    """Verify that multi-worker execution matches serial execution bitwise."""
    heights = [0.5, 1.0]
    alphas = [2.0, 4.0]
    phis = [0.0, 2.0]

    sweep_ser = GroundEffectSweep(rectangular_wing, n_jobs=1, n_panels=20)
    sweep_par = GroundEffectSweep(rectangular_wing, n_jobs=2, n_panels=20)

    res_ser = sweep_ser.run_sweep(heights=heights, alphas_deg=alphas, phis_deg=phis)
    res_par = sweep_par.run_sweep(heights=heights, alphas_deg=alphas, phis_deg=phis)

    np.testing.assert_allclose(res_par.CL, res_ser.CL, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(res_par.CDi, res_ser.CDi, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(res_par.Cl, res_ser.Cl, atol=1e-12, rtol=1e-12)


# ─── 9. Regression tests ─────────────────────────────────────────────────────

@pytest.fixture
def small_wing():
    """AR = 6 rectangular wing, chord 1 m, span 6 m (small and fast)."""
    return vt.LiftingSurface(
        semi_span=3.0,
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )


@pytest.mark.parametrize("bad_h", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_height_must_be_finite_and_positive(small_wing, bad_h):
    """NaN, infinite, zero and negative heights are refused (single point and sweep)."""
    with pytest.raises(ValueError, match="finite number larger than 0"):
        analyze_ground_effect(small_wing, h=bad_h, settings=_mesh(12, 2))
    with pytest.raises(ValueError, match="finite number larger than 0"):
        GroundEffectSweep(small_wing, settings=_mesh(12, 2)).run_sweep([0.5, bad_h], [4.0])


def test_sweep_refuses_non_finite_angles(small_wing):
    with pytest.raises(ValueError, match="alphas_deg"):
        GroundEffectSweep(small_wing, settings=_mesh(12, 2)).run_sweep([0.5], [float("nan")])


def test_mirrored_half_wings_tip_clearance(small_wing):
    """Two mirrored half wings give the same tip clearances as one symmetric wing."""
    half = vt.LiftingSurface(
        name="Half", semi_span=2.5, is_symmetric=False, position=np.array([0.0, 0.5, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    pair = vt.Aircraft(name="Pair", surfaces=[half, half.mirrored()])
    sett = _mesh(12, 2)
    r_pair = analyze_ground_effect(pair, h=1.0, alpha_deg=2.0, phi_deg=4.0, settings=sett,
                                   compute_strike_limit=False)
    r_sym = analyze_ground_effect(small_wing, h=1.0, alpha_deg=2.0, phi_deg=4.0, settings=sett,
                                  compute_strike_limit=False)
    assert r_pair.h_tip_left == pytest.approx(r_sym.h_tip_left, abs=1e-9)
    assert r_pair.h_tip_right == pytest.approx(r_sym.h_tip_right, abs=1e-9)
    assert r_pair.h_tip_left > 1.0 > r_pair.h_tip_right  # right wing down
    assert r_pair.reference_surface_indices == [0, 1]


def test_fourier_in_ground_effect_raises_validity_error(small_wing):
    with pytest.raises(ValidityError, match="Fourier"):
        analyze_ground_effect(small_wing, h=1.0, settings=_mesh(12, 2), solver="fourier")


def test_sweep_fixes_n_chord_from_lowest_height(small_wing):
    """Without n_chord in the settings, all heights use the count of the lowest height."""
    sw = GroundEffectSweep(small_wing, settings=vt.SolverSettings(n_panels=12)).run_sweep(
        [0.2, 0.5, 1.5], [4.0], compute_strike_limit=False)
    counts = {sw.result_at(i, 0, 0).n_chord for i in range(3)}
    low_alone = analyze_ground_effect(small_wing, h=0.2, alpha_deg=4.0, settings=vt.SolverSettings(n_panels=12),
                                      compute_strike_limit=False)
    assert counts == {low_alone.n_chord} and sw.n_chord == low_alone.n_chord
    assert low_alone.n_chord > 4  # the automatic count of the highest height alone is smaller
    # An n_chord that the user gives is kept.
    sw2 = GroundEffectSweep(small_wing, settings=_mesh(12, 3)).run_sweep([0.2, 1.5], [4.0], compute_strike_limit=False)
    assert sw2.n_chord == 3 and {sw2.result_at(i, 0, 0).n_chord for i in range(2)} == {3}


def test_strike_limit_found_flag(small_wing):
    """A limit equal to the search limit (60 deg) means no contact up to it: the flag is False."""
    sett = _mesh(12, 2)
    low = analyze_ground_effect(small_wing, h=0.5, alpha_deg=0.0, settings=sett)
    high = analyze_ground_effect(small_wing, h=10.0, alpha_deg=0.0, settings=sett)
    assert low.strike_limit_found is True and low.phi_strike_limit < 60.0
    assert high.strike_limit_found is False and high.phi_strike_limit == 60.0
    assert "search limit" in high.summary()
    assert "strike_limit_found" in high.to_dict()
    none = analyze_ground_effect(small_wing, h=0.5, settings=sett, compute_strike_limit=False)
    assert none.phi_strike_limit is None and none.strike_limit_found is None

    sw = GroundEffectSweep(small_wing, settings=sett).run_sweep([0.5, 10.0], [0.0])
    assert sw.strike_limit_found[:, 0, 0].tolist() == [True, False]
    assert sw.result_at(1, 0, 0).strike_limit_found is False


def test_removed_dead_fields(small_wing):
    """GroundEffectResult has no is_strike field (a strike raises); the sweep has no 'geometry' alias."""
    res = analyze_ground_effect(small_wing, h=1.0, settings=_mesh(12, 2), compute_strike_limit=False)
    assert not hasattr(res, "is_strike") and "is_strike" not in res.to_dict()
    assert "STRIKE" not in res.summary()
    assert not hasattr(GroundEffectSweep(small_wing), "geometry")


# ─── 10. Plot regression tests ────────────────────────────────────────────────

@pytest.fixture
def small_sweep(small_wing):
    return GroundEffectSweep(small_wing, settings=_mesh(12, 2)).run_sweep([0.5, 1.0, 10.0], [2.0, 4.0])


def test_matrix_plot_checks_grid_and_metric(small_wing, small_sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ventorum.ground_effect import plot_ground_effect_matrix

    fig = plot_ground_effect_matrix(small_sweep, metric="L_over_D")
    plt.close(fig)
    one_h = GroundEffectSweep(small_wing, settings=_mesh(12, 2)).run_sweep(
        [0.5], [2.0, 4.0], compute_strike_limit=False)
    with pytest.raises(ValueError, match="at least 2 heights and 2 angles"):
        plot_ground_effect_matrix(one_h)
    with pytest.raises(ValueError, match="at least 2 heights and 2 angles"):
        plot_ground_effect_matrix(GroundEffectSweep(small_wing, settings=_mesh(12, 2)).run_sweep(
            [0.5, 1.0], [4.0], compute_strike_limit=False))
    with pytest.raises(ValueError, match="Use one of"):
        plot_ground_effect_matrix(small_sweep, metric="__class__")


def test_clearance_envelope_without_limits(small_wing, small_sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ventorum.ground_effect import plot_clearance_envelope

    nolim = GroundEffectSweep(small_wing, settings=_mesh(12, 2)).run_sweep(
        [0.5, 1.0], [4.0], compute_strike_limit=False)
    with pytest.raises(ValueError, match="compute_strike_limit=True"):
        plot_clearance_envelope(nolim)
    # Partly NaN limits are gaps, not 0 deg.
    small_sweep.phi_strike_limit[1] = np.nan
    fig = plot_clearance_envelope(small_sweep)
    y = fig.axes[0].lines[0].get_ydata()
    assert np.isnan(y[1]) and y[0] > 0.0
    plt.close(fig)


def test_plots_do_not_change_global_rcparams(small_wing, small_sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ventorum.ground_effect import (
        plot_asymmetric_distributions, plot_clearance_envelope, plot_ground_effect_matrix,
        plot_height_sweep, plot_pitch_stability, plot_roll_effect,
    )

    before = {k: plt.rcParams[k] for k in ("font.size", "axes.labelsize", "grid.alpha", "grid.linestyle")}
    res = small_sweep.result_at(0, 1, 0)
    for f in (plot_height_sweep(small_sweep), plot_roll_effect(small_sweep), plot_pitch_stability(small_sweep),
              plot_ground_effect_matrix(small_sweep), plot_clearance_envelope(small_sweep),
              plot_asymmetric_distributions(res)):
        plt.close(f)
    assert {k: plt.rcParams[k] for k in before} == before


def test_asymmetric_distributions_show_both_mirrored_halves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ventorum.ground_effect import plot_asymmetric_distributions

    half = vt.LiftingSurface(
        name="Half", semi_span=2.5, is_symmetric=False, position=np.array([0.0, 0.5, 0.0]),
        sections=[vt.WingSection(y_frac=0.0, chord=1.0), vt.WingSection(y_frac=1.0, chord=1.0)],
    )
    pair = vt.Aircraft(name="Pair", surfaces=[half, half.mirrored()])
    sett = _mesh(12, 2)
    banked = analyze_ground_effect(pair, h=1.0, alpha_deg=4.0, phi_deg=4.0, settings=sett, compute_strike_limit=False)
    level = analyze_ground_effect(pair, h=1.0, alpha_deg=4.0, settings=sett, compute_strike_limit=False)
    fig = plot_asymmetric_distributions(banked, level)
    for ax in fig.axes:
        x = ax.lines[0].get_xdata()
        assert x.min() < -0.5 and x.max() > 0.5
        assert len(x) == 24
    # The clearance is smaller on the lowered right wing.
    x, z = fig.axes[3].lines[0].get_xdata(), fig.axes[3].lines[0].get_ydata()
    assert z[np.argmax(x)] < z[np.argmin(x)]
    plt.close(fig)


# ── Tests for sweep_roll and sweep_alpha (public function kept, with a test) ────────

def test_sweep_roll_kept(rectangular_wing):
    """Test that sweep_roll runs and returns results (a public function that is kept, with a test)."""
    from ventorum.ground_effect.sweep import sweep_roll

    heights = [1.0, 2.0]  # Higher heights to avoid ground strike at phi=5 deg
    phis = [0.0, 2.0, 5.0]
    res = sweep_roll(rectangular_wing, phis_deg=phis, heights=heights, n_panels=10, V_inf=50.0)
    assert res.grid_shape == (len(heights), 1, len(phis))
    assert np.all(np.isfinite(res.CL[0, 0, :]))
    # Roll moment should be restoring (Cl < 0 for phi > 0)
    assert res.Cl[0, 0, 1] < 0
    assert res.Cl[0, 0, 2] < res.Cl[0, 0, 1]


def test_sweep_alpha_kept(rectangular_wing):
    """Test that sweep_alpha runs and returns results (a public function that is kept, with a test)."""
    from ventorum.ground_effect.sweep import sweep_alpha

    heights = [0.5, 1.0, 2.0]
    alphas = [0.0, 4.0, 8.0]
    res = sweep_alpha(rectangular_wing, alphas_deg=alphas, heights=heights, n_panels=10, V_inf=50.0)
    assert res.grid_shape == (len(heights), len(alphas), 1)
    assert np.all(np.isfinite(res.CL[:, 0, 0]))
    # CL should increase with alpha
    assert res.CL[0, 1, 0] > res.CL[0, 0, 0]
    assert res.CL[0, 2, 0] > res.CL[0, 1, 0]


# ---------------------------------------------------------------------------
# One chordwise mesh for all angles of a sweep in ground effect (review 5, A1)
# ---------------------------------------------------------------------------

def _ge_rect_wing():
    w = vt.LiftingSurface(semi_span=3.0, sections=[vt.WingSection(y_frac=0.0, chord=1.0),
                                                   vt.WingSection(y_frac=1.0, chord=1.0)])
    return vt.Aircraft(surfaces=[w], ref_point=np.array([0.25, 0.0, 0.0]))


def _ge_sweep(alphas_deg):
    from ventorum.solvers.factory import make_solver

    settings = vt.SolverSettings(solver_type="vlm", n_panels=20)
    return make_solver("vlm").solve_sweep(_ge_rect_wing(), vt.FlightCondition(V_inf=30.0, h=0.255),
                                          settings, np.radians(alphas_deg))


def test_sweep_ge_one_mesh():
    """At h = 0.255 m the automatic n_chord (ceil(c/h_min)) is 5 at 2 deg and 6 at 6 deg.

    The sweep must use one mesh (the count of the lowest gap) for every
    angle, so dCm/dalpha has no jump at the change of the count.
    """
    al = np.arange(2.0, 6.01, 0.25)
    single = [vt.analyze(_ge_rect_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), h=0.255),
                         vt.SolverSettings(solver_type="vlm", n_panels=20)).details["lattice"].n_chord
              for a in (al[0], al[-1])]
    assert single[0] < single[1]  # the case crosses a change of the automatic count
    rs = _ge_sweep(al)
    counts = {r.details["lattice"].n_chord for r in rs}
    assert counts == {single[1]}
    cl = np.array([r.totals.CL for r in rs])
    cm = np.array([r.totals.Cm for r in rs])
    slope = np.diff(cm) / np.diff(cl)
    # Smooth: the slope changes by about the same amount in each interval.
    # A mesh change in the sweep gave a step about 20 times the usual change.
    step = np.abs(np.diff(slope))
    assert np.max(step) < 2.0 * np.median(step)


def test_sweep_ge_matches_single_solve_on_pinned_mesh():
    """Each angle of the sweep equals a single solve with the pinned n_chord."""
    al = np.array([2.0, 6.0])
    rs = _ge_sweep(al)
    n = rs[0].details["lattice"].n_chord
    for a, r in zip(al, rs):
        ref = vt.analyze(_ge_rect_wing(), vt.FlightCondition(V_inf=30.0, alpha=np.radians(a), h=0.255),
                         vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=n))
        assert r.totals.CL == pytest.approx(ref.totals.CL, rel=1e-12, abs=1e-14)
        assert r.totals.Cm == pytest.approx(ref.totals.Cm, rel=1e-12, abs=1e-14)


def test_sweep_ge_keeps_given_n_chord():
    from ventorum.solvers.factory import make_solver

    settings = vt.SolverSettings(solver_type="vlm", n_panels=20, n_chord=3)
    rs = make_solver("vlm").solve_sweep(_ge_rect_wing(), vt.FlightCondition(V_inf=30.0, h=0.255),
                                        settings, np.radians([2.0, 6.0]))
    assert {r.details["lattice"].n_chord for r in rs} == {3}
    assert settings.n_chord == 3


@pytest.mark.gpu
def test_sweep_ge_one_mesh_gpu():
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    try:
        gpu.set_device("gpu")
        gpu.set_precision("float64")
        rs = _ge_sweep(np.array([2.0, 6.0]))
        gpu.set_device("cpu")
        cpu = _ge_sweep(np.array([2.0, 6.0]))
    finally:
        gpu.set_device(old_dev)
        gpu.set_precision(old_prec)
    assert {r.details["device"] for r in rs} == {"gpu"}
    assert len({r.details["lattice"].n_chord for r in rs}) == 1
    for g, c in zip(rs, cpu):
        assert g.totals.Cm == pytest.approx(c.totals.Cm, rel=1e-10, abs=1e-13)
