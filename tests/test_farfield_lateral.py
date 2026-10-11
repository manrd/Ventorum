# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Tests of the far-field lateral diagnostics (CY_farfield, Cn_trefftz) and the near/far trust warning."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum import gpu
from ventorum.aero.system import freestream_direction, lift_direction
from ventorum.core.axes import transform_moments
from ventorum.core.trust import (
    NEAR_FAR_FLOOR_CN,
    NEAR_FAR_FLOOR_CY,
    near_far_lateral_warnings,
)
from ventorum.solvers.control_derivatives import control_derivatives

SOLVERS = ("vlm", "linear", "nonlinear")


def _wing(dihedral_deg=0.0, controls=None, semi_span=4.0, chord=1.0):
    af = vt.LinearAirfoil()
    return vt.LiftingSurface(
        name="w", semi_span=semi_span, is_symmetric=True, dihedral=math.radians(dihedral_deg),
        sections=[vt.WingSection(0.0, chord, airfoil=af), vt.WingSection(1.0, chord, airfoil=af)],
        controls=list(controls or []),
    )


def _settings(solver):
    return vt.SolverSettings(solver_type=solver, n_panels=24, spacing="cosine",
                             n_chord=4 if solver == "vlm" else None)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _cn_stab(res):
    t, c = res.totals, res.condition
    return transform_moments(t.Cl, t.Cm, t.Cn, c.alpha, c.beta, "stability", b_ref=res.b_ref, c_ref=res.c_ref)["Cn"]


def _near_far(res):
    """The warnings of the near/far rule for a result (near field in the axes of the far field)."""
    t = res.totals
    return near_far_lateral_warnings(t.CY, t.CY_farfield, _cn_stab(res), t.Cn_trefftz)


def _lateral_warnings(res, name):
    return [w for w in res.totals.trust.warnings if w.startswith(f"Near field and far field do not agree for {name} ")]


def _independent_cn(res, S_ref, b_ref):
    """Sum of the strip induced drag times its lateral arm, from the Trefftz strip data of the result."""
    lat = res.details["lattice"]
    loads = res.details["loads"]
    c = res.condition
    wd = np.asarray(res.details["wake_dir"], dtype=float)
    wd = wd / np.linalg.norm(wd)
    rp = res.details["ref_point"]
    rp = np.zeros(3) if rp is None else rp
    gam = loads.strip_gamma
    w_n = loads.extras["trefftz_normalwash"]
    total = 0.0
    side = np.cross(lift_direction(c.alpha), freestream_direction(c.alpha, c.beta))
    rp_t = rp - np.dot(rp, wd) * wd
    for i in range(lat.n_strips):
        pl = lat.te_left[i] - np.dot(lat.te_left[i], wd) * wd
        pr = lat.te_right[i] - np.dot(lat.te_right[i], wd) * wd
        ds = pr - pl
        drag = -0.5 * c.rho * gam[i] * w_n[i] * math.sqrt(np.dot(ds, ds))
        point = pl + lat.cp_frac[i] * ds
        total += drag * np.dot(point - rp_t, side)
    return total / (0.5 * c.rho * c.V_inf ** 2 * S_ref * b_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Symmetric cases
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver", SOLVERS)
def test_symmetric_case_zero(solver):
    """A symmetric wing at alpha and beta 0 gives zero far-field lateral values."""
    c = vt.FlightCondition(V_inf=20.0, alpha=math.radians(5.0))
    res = _quiet(vt.analyze, _wing(5.0), c, _settings(solver))
    assert res.totals.CY_farfield is not None and res.totals.Cn_trefftz is not None
    assert abs(res.totals.CY_farfield) <= 1e-14
    assert abs(res.totals.Cn_trefftz) <= 1e-14
    assert not _lateral_warnings(res, "CY") and not _lateral_warnings(res, "Cn")


def test_fourier_gives_none():
    """The Fourier solver has no Trefftz trace: both fields are None."""
    c = vt.FlightCondition(V_inf=20.0, alpha=math.radians(5.0))
    res = _quiet(vt.analyze, _wing(), c, vt.SolverSettings(solver_type="fourier", n_panels=24))
    assert res.totals.CY_farfield is None and res.totals.Cn_trefftz is None
    clone = res.totals.clone()
    assert clone.CY_farfield is None and clone.Cn_trefftz is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Sideslip on a wing with dihedral
# ═══════════════════════════════════════════════════════════════════════════════

DIHEDRAL_COND = vt.FlightCondition(V_inf=20.0, alpha=math.radians(4.0), beta=math.radians(5.0))


@pytest.mark.parametrize("solver", SOLVERS)
def test_dihedral_sideslip_cy(solver):
    """10 deg dihedral, beta 5 deg, alpha 4 deg: the far field and the near field agree by the near/far rule for CY."""
    res = _quiet(vt.analyze, _wing(10.0), DIHEDRAL_COND, _settings(solver))
    t = res.totals
    assert t.CY < 0.0 and t.CY_farfield < 0.0   # dihedral effect: side force to the left for wind from the right
    diff = abs(t.CY - t.CY_farfield)
    assert not (diff > 0.05 * abs(t.CY_farfield) and diff > NEAR_FAR_FLOOR_CY)
    assert not _lateral_warnings(res, "CY")
    # The clone keeps the fields.
    clone = t.clone()
    assert clone.CY_farfield == t.CY_farfield and clone.Cn_trefftz == t.Cn_trefftz


@pytest.mark.parametrize("solver", [
    pytest.param("vlm", marks=pytest.mark.xfail(
        strict=True, reason="Known difference, not resolved: the VLM near-field CY is about 9 % below the far field "
        "at alpha 4 deg (the difference grows with the spanwise panels; at alpha 0 the two agree to 0.3 %).")),
    "linear",
    "nonlinear",
])
def test_dihedral_sideslip_cy_within_5_percent(solver):
    """The card value: CY_farfield and the near-field CY agree within 5 % (relative)."""
    res = _quiet(vt.analyze, _wing(10.0), DIHEDRAL_COND, _settings(solver))
    t = res.totals
    assert abs(t.CY - t.CY_farfield) <= 0.05 * abs(t.CY_farfield)


def test_dihedral_sideslip_cy_at_zero_alpha_agrees():
    """At alpha 0 the near-field and the far-field side forces of the VLM agree closely (1 %)."""
    c = vt.FlightCondition(V_inf=20.0, alpha=0.0, beta=math.radians(5.0))
    res = _quiet(vt.analyze, _wing(10.0), c, _settings("vlm"))
    t = res.totals
    assert t.CY == pytest.approx(t.CY_farfield, rel=0.01)


@pytest.mark.parametrize("solver", SOLVERS)
def test_cn_trefftz_matches_drag_moment(solver):
    """Cn_trefftz equals an independent sum of the strip induced drag times the lateral arm."""
    ac = vt.Aircraft(name="a", surfaces=[_wing(10.0)], ref_point=np.array([0.3, 0.0, 0.1]))
    ac.compute_reference_values()
    res = _quiet(vt.analyze, ac, DIHEDRAL_COND, _settings(solver))
    ref = _independent_cn(res, ac.S_ref, ac.b_ref)
    assert abs(ref) > 1e-5
    assert res.totals.Cn_trefftz == pytest.approx(ref, rel=1e-12)


def test_cn_trefftz_sign_more_drag_on_the_right_is_nose_right():
    """An antisymmetric aileron (right trailing edge down) loads the right wing: adverse yaw, Cn_trefftz > 0."""
    w = _wing(0.0, [vt.ControlSurface("ail", 0.6, 1.0, 0.75, math.radians(5.0), symmetric=False)])
    res = _quiet(vt.analyze, w, vt.FlightCondition(V_inf=20.0, alpha=math.radians(3.0)), _settings("vlm"))
    assert res.totals.Cl < 0.0        # right wing up
    assert res.totals.Cn_trefftz > 0.0
    assert _cn_stab(res) > 0.0        # the near field also gives adverse yaw in stability axes


def test_near_field_cn_in_stability_axes_equals_transform_moments():
    """The closed form of the trust comparison equals transform_moments (body to stability axes)."""
    from ventorum.solvers.lattice_base import LatticeSolver

    ac = vt.Aircraft(name="a", surfaces=[_wing(10.0)], ref_point=np.array([0.3, 0.0, 0.1]))
    for cond in (DIHEDRAL_COND, vt.FlightCondition(V_inf=20.0, alpha=math.radians(-7.0), beta=math.radians(-12.0))):
        res = _quiet(vt.analyze, ac, cond, _settings("vlm"))
        got = LatticeSolver._cn_stability(res.totals, cond, res.b_ref, res.c_ref)
        assert got == pytest.approx(_cn_stab(res), rel=1e-14, abs=1e-17)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Aileron
# ═══════════════════════════════════════════════════════════════════════════════

AILERON = [vt.ControlSurface("ail", 0.6, 1.0, 0.75, math.radians(5.0), symmetric=False)]
AILERON_COND = vt.FlightCondition(V_inf=20.0, alpha=math.radians(3.0))


def _aileron_result():
    # AR 8 flat wing: span 8 m, chord 1 m.
    return _quiet(vt.analyze, _wing(0.0, AILERON), AILERON_COND, _settings("vlm"))


def test_aileron_far_field():
    """AR 8 flat wing, antisymmetric aileron 5 deg, alpha 3 deg: near-field CY about 0; the trust follows the rule."""
    res = _aileron_result()
    t = res.totals
    assert abs(t.CY) <= 1e-6
    assert t.CY_farfield is not None and math.isfinite(t.CY_farfield)
    assert t.Cn_trefftz is not None and math.isfinite(t.Cn_trefftz)
    expected = _near_far(res)
    got = [w for w in t.trust.warnings if w.startswith("Near field and far field do not agree")]
    assert got == expected


@pytest.mark.xfail(strict=True, reason=(
    "Known limit, not resolved: the Trefftz trace of the induced drag does not include the "
    "trailing-edge displacement of a deflected control (the controls turn the panel normals). Adding it would "
    "change the induced drag of every deflected case, so it is not added; the flat trace gives no side force."))
def test_aileron_far_field_side_force_sign():
    """Thin-wing theory with the real deflected trailing edge: the side force is not 0 and is positive.

    The steps of the trailing-edge trace at the inboard ends of the
    ailerons (right edge down, left edge up) carry the local circulation;
    both give a force to the right for a right aileron with the trailing
    edge down.
    """
    t = _aileron_result().totals
    assert abs(t.CY_farfield) > 1e-6
    assert t.CY_farfield > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. The warning rule
# ═══════════════════════════════════════════════════════════════════════════════

def test_warning_rule():
    """The near/far rule: warn when |near - far| > 5 % |far| and |near - far| > the floor (0.001 CY, 0.0002 Cn)."""
    assert NEAR_FAR_FLOOR_CY == 0.001 and NEAR_FAR_FLOOR_CN == 0.0002
    # Large values: the 5 % limit decides.
    assert near_far_lateral_warnings(CY=0.104, CY_farfield=0.1) == []          # 4 %
    w = near_far_lateral_warnings(CY=0.106, CY_farfield=0.1)                   # 6 %
    assert len(w) == 1 and "CY" in w[0] and "0.106" in w[0] and "0.1" in w[0] and "side-edge" in w[0]
    assert "difference 0.006" in w[0]
    # Small values: the floor decides.
    assert near_far_lateral_warnings(CY=0.0009, CY_farfield=0.0) == []
    assert len(near_far_lateral_warnings(CY=0.0011, CY_farfield=0.0)) == 1
    assert near_far_lateral_warnings(CY=-0.0005, CY_farfield=0.0004) == []     # 0.0009 < floor
    assert len(near_far_lateral_warnings(CY=-0.0006, CY_farfield=0.0005)) == 1
    # Near zero both ways.
    assert near_far_lateral_warnings(CY=0.0, CY_farfield=0.0, Cn=0.0, Cn_trefftz=0.0) == []
    assert near_far_lateral_warnings(Cn=0.00019, Cn_trefftz=0.0) == []
    w = near_far_lateral_warnings(Cn=0.00021, Cn_trefftz=0.0)
    assert len(w) == 1 and "Cn" in w[0] and "Cn_trefftz" in w[0] and "stability axes" in w[0]
    assert near_far_lateral_warnings(Cn=0.0102, Cn_trefftz=0.01) == []          # 2 %
    assert len(near_far_lateral_warnings(Cn=0.0110, Cn_trefftz=0.01)) == 1     # 10 %, 0.001 > floor
    # Exactly at a limit: no warning (strict inequality).
    assert near_far_lateral_warnings(CY=0.002, CY_farfield=0.001) == []        # diff = floor
    # Both coefficients at once; None and non-finite values give no warning.
    assert len(near_far_lateral_warnings(CY=-0.0009, CY_farfield=0.0009, Cn=0.02, Cn_trefftz=0.01)) == 2
    assert near_far_lateral_warnings(CY=0.2, CY_farfield=None, Cn=None, Cn_trefftz=0.01) == []
    assert near_far_lateral_warnings(CY=float("nan"), CY_farfield=0.1) == []


def test_cn_compared_only_without_side_force():
    """Regression test: with a side force the Cn pair is not compared.

    The Trefftz plane has no chordwise position, so Cn_trefftz misses the yaw of a
    side force ahead of or behind the reference point. A dihedral wing in sideslip
    gave a false Cn warning.
    """
    w = near_far_lateral_warnings(CY=0.05, CY_farfield=0.05, Cn=-1.4e-4, Cn_trefftz=5.5e-4)
    assert w == []
    w = near_far_lateral_warnings(CY=0.0005, CY_farfield=0.0008, Cn=-1.4e-4, Cn_trefftz=5.5e-4)
    assert len(w) == 1 and "Cn" in w[0]


def test_warning_does_not_change_the_score():
    """The comparison adds a warning only: the score and the factors stay the same."""
    from ventorum.core.trust import evaluate_aerodynamic_trust

    base = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.008, solver_type="vlm", n_panels=24)
    warned = evaluate_aerodynamic_trust(AR=8.0, CL=0.4, CDi=0.008, solver_type="vlm", n_panels=24,
                                        CY=0.2, CY_farfield=0.1, Cn_stability=0.0, Cn_trefftz=0.0)
    assert warned.score == base.score and warned.factors == base.factors
    assert len(warned.warnings) == len(base.warnings) + 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Ground effect
# ═══════════════════════════════════════════════════════════════════════════════

def test_ground_effect_images():
    """h/c 0.3, beta 3 deg: finite values that use the image wake; beta 0 (the folded solve) gives zeros."""
    w = _wing(5.0)
    ac = vt.Aircraft(name="a", surfaces=[w])
    ac.compute_reference_values()
    st = _settings("vlm")
    c = vt.FlightCondition(V_inf=20.0, alpha=math.radians(3.0), beta=math.radians(3.0), h=0.3)
    res = _quiet(vt.analyze, ac, c, st)
    t = res.totals
    assert res.details["ground"] is not None
    assert math.isfinite(t.CY_farfield) and math.isfinite(t.Cn_trefftz)
    assert t.CY_farfield != 0.0 and t.Cn_trefftz != 0.0
    # The drag distribution of the Trefftz plane (with the images) gives Cn_trefftz.
    assert t.Cn_trefftz == pytest.approx(_independent_cn(res, ac.S_ref, ac.b_ref), rel=1e-12)
    free = _quiet(vt.analyze, ac, vt.FlightCondition(V_inf=20.0, alpha=c.alpha, beta=c.beta), st)
    assert abs(t.Cn_trefftz - free.totals.Cn_trefftz) > 1e-3 * abs(free.totals.Cn_trefftz)
    sym = _quiet(vt.analyze, ac, vt.FlightCondition(V_inf=20.0, alpha=c.alpha, h=0.3), st)
    assert sym.totals.CY_farfield == 0.0 and sym.totals.Cn_trefftz == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Batch and single solves
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver_cls", [vt.VortexLatticeSolver, vt.LinearLLTSolver, vt.NonlinearLLTSolver])
@pytest.mark.parametrize("h", [None, 2.0])
def test_batch_and_single_agree(solver_cls, h):
    """A sweep and single solves give the same far-field fields (1e-12)."""
    solver = solver_cls()
    st = _settings(solver.name)
    alphas = np.radians([-2.0, 0.0, 3.0, 6.0])
    base = vt.FlightCondition(V_inf=20.0, beta=math.radians(3.0), h=h)
    ac = vt.Aircraft(name="a", surfaces=[_wing(10.0)], ref_point=np.array([0.25, 0.0, 0.0]))
    if solver.name == "nonlinear" and h is None:
        # The sweep starts each angle from the angle before it (continuation),
        # so its circulation differs from a single solve at the solver
        # tolerance. The batch without continuation is the same as single solves.
        ac.compute_reference_values()
        rp = ac.moment_reference()
        lattice = solver.build(ac, st, base, None, rp)
        conds = [vt.FlightCondition(V_inf=20.0, alpha=float(a), beta=base.beta) for a in alphas]
        sweep = _quiet(solver.solve_batch, lattice, conds, st, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=rp,
                       continuation=False)
    else:
        sweep = _quiet(solver.solve_sweep, ac, base, st, alphas)
    for a, r in zip(alphas, sweep):
        single = _quiet(solver.solve, ac, vt.FlightCondition(V_inf=20.0, alpha=float(a), beta=base.beta, h=h), st)
        for name in ("CY_farfield", "Cn_trefftz"):
            x, y = getattr(r.totals, name), getattr(single.totals, name)
            assert abs(x - y) <= 1e-12 * max(abs(y), 1e-3), (name, a, x, y)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GPU
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def gpu_device():
    """Run the test with the GPU device selected; restore the settings after it."""
    if not gpu.available():
        pytest.skip(f"The GPU pipelines cannot run: {gpu.unavailable_reason()}")
    old_dev, old_prec = gpu.get_device(), gpu._precision
    yield
    gpu.set_device(old_dev)
    gpu.set_precision(old_prec)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("solver_cls", [vt.VortexLatticeSolver, vt.LinearLLTSolver, vt.NonlinearLLTSolver])
@pytest.mark.parametrize("h,beta_deg", [(None, 4.0), (2.0, 3.0), (None, 0.0)])
def test_gpu_fields(gpu_device, precision, solver_cls, h, beta_deg):
    """GPU fields agree with the CPU float64 values: 1e-4 relative or 1e-6 absolute."""
    solver = solver_cls()
    st = _settings(solver.name)
    alphas = np.radians(np.linspace(-2.0, 8.0, 6))
    ac = vt.Aircraft(name="a", surfaces=[_wing(10.0, AILERON)], ref_point=np.array([0.25, 0.0, 0.0]))
    cond = vt.FlightCondition(V_inf=20.0, beta=math.radians(beta_deg), h=h)
    gpu.set_device("cpu")
    ref = _quiet(solver.solve_sweep, ac, cond, st, alphas)
    gpu.set_device("gpu")
    gpu.set_precision(precision)
    new = _quiet(solver.solve_sweep, ac, cond, st, alphas)
    for r, s in zip(ref, new):
        assert s.details["device"] == "gpu"
        for name in ("CY_farfield", "Cn_trefftz"):
            a, b = getattr(r.totals, name), getattr(s.totals, name)
            assert b is not None
            assert abs(a - b) <= max(1e-4 * abs(a), 1e-6), (name, a, b)
        assert _lateral_warnings(r, "CY") == _lateral_warnings(s, "CY") or precision == "float32"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Control derivatives and the agent output
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("axes", ["body", "stability"])
def test_control_derivative_far_field(axes):
    """The far-field derivative fields exist, are finite and equal central differences of the field values."""
    w = _wing(10.0, [vt.ControlSurface("ail", 0.6, 1.0, 0.75, 0.0, symmetric=False)])
    cond = vt.FlightCondition(V_inf=20.0, alpha=math.radians(3.0), beta=math.radians(3.0))
    st = _settings("vlm")
    out = _quiet(control_derivatives, w, cond, st, step_deg=1.0, axes=axes, hinge_alpha=False)
    d = out.derivatives["ail"]
    assert math.isfinite(d.CY_farfield) and math.isfinite(d.Cn_trefftz)
    h = math.radians(1.0)
    vals = {}
    for sgn in (1.0, -1.0):
        ws = _wing(10.0, [vt.ControlSurface("ail", 0.6, 1.0, 0.75, sgn * h, symmetric=False)])
        vals[sgn] = _quiet(vt.analyze, ws, cond, st).totals
    assert d.CY_farfield == pytest.approx((vals[1.0].CY_farfield - vals[-1.0].CY_farfield) / (2 * h), abs=1e-10)
    assert d.Cn_trefftz == pytest.approx((vals[1.0].Cn_trefftz - vals[-1.0].Cn_trefftz) / (2 * h), abs=1e-10)
    js = d.to_dict()
    assert js["CY_farfield_delta_per_rad"] == d.CY_farfield
    assert js["Cn_trefftz_delta_per_rad"] == d.Cn_trefftz
    assert js["CY_delta_per_rad"] == d.CY   # the near-field value stays


def test_agent_wing_analysis_gives_the_fields():
    """The totals of the agent tool output include the two far-field fields."""
    from ventorum.agent import wing_analysis

    p = wing_analysis({"span_m": 8.0, "root_chord_m": 1.0, "tip_chord_m": 1.0, "dihedral_deg": 10.0},
                      {"V_inf_m_s": 20.0, "alpha_deg": 4.0, "beta_deg": 5.0},
                      {"n_panels": 16}, "summary")
    m = p["metrics"]
    assert "CY_farfield" in m and "Cn_trefftz" in m
    assert m["CY_farfield"] is not None and m["CY_farfield"] < 0.0
