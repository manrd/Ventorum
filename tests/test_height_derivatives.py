# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Height derivatives of one flight state in ground effect, and the Irodov criterion.

Checks:

* each derivative equals direct central differences of single solves on the
  same mesh, at a banked and side-slipped state;
* the Irodov centres and margin equal those of the grid tool
  (:class:`~ventorum.ground_effect.GroundEffectSweep`) for the same steps;
* the pivot: the alpha derivatives about two pivots obey the chain rule with
  the height derivative;
* symmetry at zero bank and sideslip, the free-air limit, the axes
  transform with the reference lengths, the lifting lines, the refusals and
  the GPU path.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ventorum as vt
from ventorum.aero.system import ground_normal
from ventorum.core.axes import rotation_body_to_stability, rotation_body_to_wind
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.ground_effect import GroundEffectSweep

C = 1.0
REF = np.array([0.3, 0.0, 0.0])
SETT = vt.SolverSettings(n_panels=12, n_chord=4)


def _wing(semi_span: float = 1.5) -> vt.LiftingSurface:
    return vt.LiftingSurface(name="wing", semi_span=semi_span,
                             sections=[vt.WingSection(0.0, C), vt.WingSection(1.0, C)])


def _ac(ref=REF, semi_span: float = 1.5) -> vt.Aircraft:
    b = 2.0 * semi_span
    return vt.Aircraft(surfaces=[_wing(semi_span)], S_ref=b * C, b_ref=b, c_ref=C,
                       ref_point=np.asarray(ref, dtype=float))


def _cond(alpha_deg=4.0, h=0.25, phi_deg=0.0, beta_deg=0.0, **kw) -> vt.FlightCondition:
    return vt.FlightCondition(V_inf=30.0, alpha=np.radians(alpha_deg), beta=np.radians(beta_deg),
                              phi=np.radians(phi_deg), h=h, **kw)


def _coeffs(ac, cond, sett=SETT) -> np.ndarray:
    t = vt.analyze(ac.clone(), cond, sett).totals
    return np.array([t.CL, t.CDi, t.CY, t.Cl, t.Cm, t.Cn])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Against direct differences and the grid tool
# ═══════════════════════════════════════════════════════════════════════════════


def test_each_derivative_equals_direct_differences():
    """Banked and side-slipped state: the batch equals single solves on the same mesh (round-off)."""
    import dataclasses

    base = _cond(alpha_deg=4.0, h=0.3, phi_deg=2.0, beta_deg=1.5)
    res = vt.height_derivatives(_ac(), base, SETT)
    assert res.n_solves == 9 and res.n_chord == 4 and res.device == "cpu"
    steps = {"alpha": np.radians(res.steps["alpha_deg"]), "beta": np.radians(res.steps["beta_deg"]),
             "phi": np.radians(res.steps["phi_deg"]), "h": res.steps["h_m"]}
    assert steps["h"] == pytest.approx(0.05 * res.h_min_m)
    for j, name in enumerate(("alpha", "beta", "phi", "h")):
        d = steps[name]
        cp = _coeffs(_ac(), dataclasses.replace(base, **{name: getattr(base, name) + d}))
        cm = _coeffs(_ac(), dataclasses.replace(base, **{name: getattr(base, name) - d}))
        den = 2.0 * d / C if name == "h" else 2.0 * d
        assert np.allclose(res.matrix[:, j], (cp - cm) / den, rtol=1e-9, atol=1e-10), name
    assert res.derivatives["Cm_h"] == res.matrix[4, 3]
    assert res.derivatives["Cl_phi"] == res.matrix[3, 2]


def test_irodov_equals_the_grid_tool():
    """Same steps and mesh: the single-state values equal the centre of a 3 x 3 grid."""
    h0, a0 = 0.25, 4.0
    res = vt.height_derivatives(_ac(), _cond(alpha_deg=a0, h=h0), SETT)
    dh, da = res.steps["h_m"], res.steps["alpha_deg"]
    sw = GroundEffectSweep(_ac(), settings=SETT, backend="serial").run_sweep(
        [h0 - dh, h0, h0 + dh], [a0 - da, a0, a0 + da], [0.0], ref_point=REF, height_ref="ref",
        compute_strike_limit=False)
    d = sw.compute_stability_derivatives()
    assert res.irodov["x_alpha_m"] == pytest.approx(d["x_alpha"][1, 1], abs=1e-9)
    assert res.irodov["x_h_m"] == pytest.approx(d["x_h"][1, 1], abs=1e-9)
    assert res.irodov["margin"] == pytest.approx(d["irodov_margin"][1, 1], abs=1e-9)
    assert res.derivatives["CL_h"] == pytest.approx(d["CL_h"][1, 1], rel=1e-9)
    assert res.derivatives["Cm_h"] == pytest.approx(d["Cm_h"][1, 1], rel=1e-9)
    assert res.irodov["verdict"] == ("stable" if res.irodov["margin"] > 0 else "unstable")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pivot, symmetry, physics, free-air limit
# ═══════════════════════════════════════════════════════════════════════════════


def test_alpha_derivative_obeys_the_chain_rule_between_two_pivots():
    """CL_alpha|P2 = CL_alpha|P1 + CL_h d(h_P1/c)/dalpha|P2; CL_h does not depend on the pivot."""
    a0, h1 = 4.0, 0.3
    p1, p2 = REF, np.array([-0.2, 0.0, 0.05])
    k = ground_normal(np.radians(a0))
    h2 = h1 + float((p2 - p1) @ k)           # the same flight state, height given at P2
    kw = {"h_step": 0.01, "alpha_step_deg": 0.1}
    r1 = vt.height_derivatives(_ac(p1), _cond(alpha_deg=a0, h=h1), SETT, **kw)
    r2 = vt.height_derivatives(_ac(p2), _cond(alpha_deg=a0, h=h2), SETT, **kw)
    assert r2.base.totals.CL == pytest.approx(r1.base.totals.CL, rel=1e-10)
    assert r2.derivatives["CL_h"] == pytest.approx(r1.derivatives["CL_h"], rel=1e-4)
    e = 1e-6
    dk = (ground_normal(np.radians(a0) + e) - ground_normal(np.radians(a0) - e)) / (2.0 * e)
    dh1 = float((p1 - p2) @ dk) / C
    expected = r1.derivatives["CL_alpha"] + r1.derivatives["CL_h"] * dh1
    assert abs(r1.derivatives["CL_h"] * dh1) > 0.05            # the pivot term is not small
    assert r2.derivatives["CL_alpha"] == pytest.approx(expected, abs=2e-4)


def test_height_centre_is_a_fixed_point_of_the_geometry():
    """x_h is measured from the pivot: moving the pivot aft by dx moves x_h forward by about dx."""
    a0, h1, dx = 2.0, 0.3, 0.25
    p2 = REF + np.array([dx, 0.0, 0.0])
    h2 = h1 + float((p2 - REF) @ ground_normal(np.radians(a0)))
    r1 = vt.height_derivatives(_ac(REF), _cond(alpha_deg=a0, h=h1), SETT)
    r2 = vt.height_derivatives(_ac(p2), _cond(alpha_deg=a0, h=h2), SETT)
    # Not exact: the moment transfer uses the normal force, the height centre uses the lift.
    assert r2.irodov["x_h_m"] == pytest.approx(r1.irodov["x_h_m"] - dx, abs=0.01 * dx)


def test_symmetry_at_zero_bank_and_sideslip():
    res = vt.height_derivatives(_ac(), _cond(alpha_deg=4.0, h=0.3), SETT)
    m = res.matrix
    lat, lon = [2, 3, 5], [0, 1, 4]
    assert np.max(np.abs(m[np.ix_(lat, [0, 3])])) < 1e-10      # alpha and h give no lateral loads
    assert np.max(np.abs(m[np.ix_(lon, [1, 2])])) < 1e-10      # beta and phi give no longitudinal change


def test_ground_effect_signs_and_trend_with_height():
    """Near the ground the lift falls with height and the bank gives a restoring rolling moment."""
    low = vt.height_derivatives(_ac(), _cond(h=0.2), SETT).derivatives
    high = vt.height_derivatives(_ac(), _cond(h=0.8), SETT).derivatives
    assert low["CL_h"] < high["CL_h"] < 0.0
    assert low["Cl_phi"] < high["Cl_phi"] < 0.0
    assert low["CL_alpha"] > high["CL_alpha"]


def test_free_air_limit():
    """At h/b = 30 the height and bank columns vanish and the alpha column is the free-air one."""
    sett = vt.SolverSettings(n_panels=12, n_chord=4)
    res = vt.height_derivatives(_ac(), _cond(alpha_deg=4.0, h=90.0), sett)
    assert np.max(np.abs(res.matrix[:, 2:])) < 2e-4
    d = math.radians(0.5)
    free = _cond(alpha_deg=4.0, h=None)
    import dataclasses

    cp = _coeffs(_ac(), dataclasses.replace(free, alpha=free.alpha + d), sett)
    cm = _coeffs(_ac(), dataclasses.replace(free, alpha=free.alpha - d), sett)
    assert np.allclose(res.matrix[:, 0], (cp - cm) / (2.0 * d), atol=2e-4)


def test_body_rates_are_kept_in_every_case():
    res = vt.height_derivatives(_ac(), _cond(h=0.3, q=0.2), SETT)
    assert res.base.condition.q == 0.2
    plain = vt.height_derivatives(_ac(), _cond(h=0.3), SETT)
    assert res.derivatives["Cm_h"] != plain.derivatives["Cm_h"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Axes
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("axes", ["stability", "wind"])
def test_axes_turn_the_dimensional_moments(axes):
    """b != c and sideslip: the moment rows are the dimensional moments turned by R, scaled back."""
    ac = _ac(semi_span=1.5)                 # b = 3, c = 1
    cond = _cond(alpha_deg=4.0, h=0.3, phi_deg=1.0, beta_deg=3.0)
    body = vt.height_derivatives(ac, cond, SETT)
    other = vt.height_derivatives(ac, cond, SETT, axes=axes)
    R = (rotation_body_to_stability(cond.alpha) if axes == "stability"
         else rotation_body_to_wind(cond.alpha, cond.beta))
    b, c = 3.0, C
    dim = body.matrix[3:] * np.array([[b], [c], [b]])
    expected = (R @ dim) / np.array([[b], [c], [b]])
    assert np.allclose(other.matrix[3:], expected, rtol=1e-12, atol=1e-14)
    assert np.array_equal(other.matrix[:3], body.matrix[:3])
    assert other.irodov == body.irodov            # Irodov uses the body Cm in every set
    assert other.axes == axes


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Solvers, refusals, output
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.filterwarnings("ignore:Lifting line in ground effect:RuntimeWarning")
@pytest.mark.parametrize("solver", ["linear", "nonlinear"])
def test_lifting_lines_above_their_limit(solver):
    sett = vt.SolverSettings(solver_type=solver, n_panels=16)
    res = vt.height_derivatives(_ac(), _cond(alpha_deg=4.0, h=1.6), sett)
    assert res.n_chord == 1
    assert res.derivatives["CL_h"] < 0.0 and res.derivatives["Cl_phi"] < 0.0


def test_lifting_line_refused_below_its_limit():
    with pytest.raises(ValidityError):
        vt.height_derivatives(_ac(), _cond(h=0.5), vt.SolverSettings(solver_type="linear", n_panels=16))


def test_automatic_chordwise_count_is_that_of_the_smallest_gap():
    """All cases use the largest automatic count of the nine cases (no mesh change in the differences)."""
    import dataclasses

    base = _cond(h=0.12, phi_deg=1.0)
    res = vt.height_derivatives(_ac(), base, vt.SolverSettings(n_panels=12, n_chord=None))
    st = res.steps
    cases = [base]
    for name, d in (("alpha", np.radians(st["alpha_deg"])), ("beta", np.radians(st["beta_deg"])),
                    ("phi", np.radians(st["phi_deg"])), ("h", st["h_m"])):
        cases += [dataclasses.replace(base, **{name: getattr(base, name) + s * d}) for s in (1.0, -1.0)]
    counts = [vt.analyze(_ac(), c, vt.SolverSettings(n_panels=12, n_chord=None)).details["lattice"].n_chord
              for c in cases]
    assert res.n_chord == max(counts) > min(counts)


def test_free_air_is_refused():
    with pytest.raises(ValueError, match="condition.h"):
        vt.height_derivatives(_ac(), _cond(h=None), SETT)


def test_fourier_is_refused():
    with pytest.raises(ValueError, match="Fourier"):
        vt.height_derivatives(_ac(), _cond(h=0.5), vt.SolverSettings(solver_type="fourier", n_panels=16))


@pytest.mark.parametrize("kw, match", [
    ({"h_step": 0.0}, "h_step"), ({"h_step": 0.6}, "h_step"), ({"alpha_step_deg": -1.0}, "alpha_step_deg"),
    ({"beta_step_deg": 6.0}, "beta_step_deg"), ({"phi_step_deg": float("nan")}, "phi_step_deg"),
    ({"axes": "earth"}, "axes"),
])
def test_invalid_inputs_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        vt.height_derivatives(_ac(), _cond(h=0.3), SETT, **kw)


def test_strike_of_a_stepped_case_names_the_case():
    with pytest.raises(GroundStrikeError, match="phi"):
        vt.height_derivatives(_ac(), _cond(alpha_deg=0.0, h=0.05), SETT, phi_step_deg=5.0)


def test_strike_of_the_base_state():
    with pytest.raises(GroundStrikeError, match="base state"):
        vt.height_derivatives(_ac(), _cond(alpha_deg=4.0, h=0.01), SETT)


def test_geometry_is_not_changed():
    ac = _ac()
    before = ac.ref_point.copy()
    vt.height_derivatives(ac, _cond(h=0.3), SETT)
    assert np.array_equal(ac.ref_point, before)


def test_lifting_surface_input():
    res = vt.height_derivatives(_wing(), _cond(h=0.5), SETT)
    assert np.array_equal(res.ref_point, np.zeros(3))


def test_to_dict_is_json_ready():
    out = vt.height_derivatives(_ac(), _cond(h=0.3), SETT).to_dict()
    json.dumps(out, allow_nan=False)
    assert set(out["irodov"]) == {"x_alpha_m", "x_h_m", "margin", "verdict"}
    assert out["device"] == "cpu" and out["axes"] == "body" and len(out["derivatives"]) == 24


@pytest.mark.gpu
def test_gpu_agrees_with_cpu():
    """The selected device runs the cases (D-26); float32 differences stay at the round-off floor."""
    from ventorum import gpu

    if not gpu.available():
        pytest.skip(f"GPU pipeline unavailable: {gpu.unavailable_reason()}")
    cond = _cond(alpha_deg=4.0, h=0.3, phi_deg=2.0, beta_deg=1.0)
    gpu.set_device("cpu")
    cpu = vt.height_derivatives(_ac(), cond, SETT)
    gpu.set_device("gpu")
    try:
        on_gpu = vt.height_derivatives(_ac(), cond, SETT)
    finally:
        gpu.set_device("cpu")
    assert on_gpu.device == "gpu"
    assert np.allclose(on_gpu.matrix, cpu.matrix, rtol=1e-3, atol=1e-3)
    assert on_gpu.irodov["margin"] == pytest.approx(cpu.irodov["margin"], abs=2e-3)
