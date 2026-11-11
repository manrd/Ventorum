# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Side force of the trailing vortex sheet in the Trefftz plane of the 3D panel solver (``CY_farfield``).

Checks:

* the value equals an independent Kutta-Joukowski sum over the wake strip
  traces (body-axis wake, so the Trefftz plane is the y-z plane); the
  same sum along the lift direction gives the lift of the wing, which
  fixes the sign of the strip circulation;
* the sign (wind axes, positive to the right) and the antisymmetry in the
  sideslip; a planar wing and a symmetric flow on a half mesh give 0;
* parity with the vortex-lattice ``CY_farfield`` on a wing with dihedral
  in sideslip and with yaw and roll rates;
* every path: Morino batch and single condition, pressure Kutta, half and
  full mesh, the viscous coupling, GPU float64 and float32.

The formula is in the theory manual (docs/theory/ventorum_3dp.md, far-field
induced drag) and in docs/theory/forces_and_moments.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.post import compute_trefftz_side_force
from ventorum.panel3d.solver import PanelSolver

V_INF = 30.0
ALPHA = np.radians(4.0)
#: Stability-axis rate of a nondimensional rate r_hat = r b / (2 V) = 0.01 [rad/s] (span 8 m).
RATE = 0.01 * 2.0 * V_INF / 8.0

#: Relative agreement of the 3D panel and the vortex-lattice CY_farfield [-].
#: The 3D panel wing has a NACA 0006 section and the lattice is thin; the
#: thickness adds about 4 % to the lift slope. On the 16 x 16 mesh against
#: the lattice 40 x 8 (wake along the body x axis), 10 deg dihedral:
#: beta 5 deg -9.109e-03 against -8.748e-03 (+4.1 %), yaw rate +3.490e-04
#: against +3.391e-04 (+2.9 %), roll rate -3.126e-03 against -2.990e-03
#: (+4.5 %).
PARITY_TOLERANCE = 0.10


@pytest.fixture(autouse=True)
def _cpu():
    """Select the CPU, and restore the previous device and precision after the test."""
    from ventorum import gpu

    old = gpu.get_device()
    old_prec = gpu._precision
    gpu.set_device("cpu")
    yield
    gpu.set_device(old)
    gpu.set_precision(old_prec)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _aircraft(dihedral_deg: float = 0.0) -> Aircraft:
    """Rectangular NACA 0006 wing, span 8 m, chord 1 m, quarter chord on the origin."""
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape="0006"), WingSection(1.0, 1.0, x_le=-0.25, shape="0006")]
    return Aircraft(surfaces=[LiftingSurface(semi_span=4.0, sections=secs, dihedral=np.radians(dihedral_deg))])


def _beta(deg: float = 5.0) -> FlightCondition:
    return FlightCondition(V_inf=V_INF, alpha=ALPHA, beta=np.radians(deg))


def _yaw(sign: float = 1.0) -> FlightCondition:
    """Stability-axis yaw rate: body rates p = -r_s sin(alpha), r = r_s cos(alpha) [rad/s]."""
    return FlightCondition(V_inf=V_INF, alpha=ALPHA, p=-sign * RATE * np.sin(ALPHA), r=sign * RATE * np.cos(ALPHA))


def _roll(sign: float = 1.0) -> FlightCondition:
    return FlightCondition(V_inf=V_INF, alpha=ALPHA, p=sign * RATE)


def _panel(n: int = 16, **kw) -> SolverSettings:
    return SolverSettings(solver_type="panel3d", n_panels=n, n_chord=n, **kw)


def _lattice(**kw) -> SolverSettings:
    return SolverSettings(solver_type="vlm", n_panels=40, n_chord=8, wake_alignment="body", **kw)


def _cy(aircraft, condition, settings) -> float:
    return _quiet(vt.analyze, aircraft, condition=condition, settings=settings).totals.CY_farfield


def _solver(half: bool = True, kutta: str = "morino", dihedral_deg: float = 10.0, **kw) -> PanelSolver:
    """Small NACA 0006 wing solver (span 8 m, chord 1 m), reference point at the quarter chord."""
    surface = LiftingSurface(
        name="Wing", semi_span=4.0, is_symmetric=True, dihedral=np.radians(dihedral_deg),
        sections=[WingSection(0.0, 1.0, x_le=-0.25, shape="0006"),
                  WingSection(1.0, 1.0, x_le=-0.25, shape="0006")],
    )
    mesh = mesh_lifting_surface(surface, airfoil("0006").coords, n_chord=8, n_span=8, half=half)
    return PanelSolver(mesh, S_ref=8.0, b_ref=8.0, c_ref=1.0, ref_point=np.zeros(3), kutta=kutta, **kw)


def _independent_forces(solver: PanelSolver, condition: FlightCondition) -> tuple[float, float]:
    """Far-field side force and lift from the wake strips, written out again (body-axis wake).

    The strip trace goes from the end with -Gamma to the end with +Gamma
    (along n_w x d, n_w the strip normal on the upper side) and is projected
    on the y-z plane. With the doublet sign of the solver the
    Kutta-Joukowski circulation of the strip is -Gamma:
    ``F = rho V (-Gamma) (V_hat x ds)``.
    """
    key, _, _ = solver._config_key(condition, None)
    cfg = solver._configs[key]
    mu = solver._reconstruct_full_mu(cfg.unit_mu, solver._condition_coeffs(condition))
    wake = cfg.full_wake
    gamma = mu[wake.wake_up] - mu[wake.wake_lo]
    pl, pr = np.array(wake.p_left, float), np.array(wake.p_right, float)
    first = np.array([idx[0] for idx in wake.strip_panel_indices], dtype=int)
    quad = np.asarray(wake.panels, dtype=float)[first]
    n_w = np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    flip = np.sum((pr - pl) * np.cross(n_w, [1.0, 0.0, 0.0]), axis=1) < 0.0
    ds = np.where(flip[:, None], pl - pr, pr - pl)
    ds[:, 0] = 0.0
    a, b = condition.alpha, condition.beta
    v_hat = np.array([np.cos(a) * np.cos(b), -np.sin(b), np.sin(a) * np.cos(b)])
    lift = np.array([-np.sin(a), 0.0, np.cos(a)])
    side = np.cross(lift, v_hat)
    force = condition.rho * condition.V_inf * np.sum(-gamma[:, None] * np.cross(v_hat[None, :], ds), axis=0)
    q_s = 0.5 * condition.rho * condition.V_inf**2 * solver.S_ref
    return float(force @ side / q_s), float(force @ lift / q_s)


# ---------------------------------------------------------------------------
# 1. Formula and sign of the circulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("half", [True, False])
@pytest.mark.parametrize("condition", [_beta(), _yaw(), _roll()], ids=["sideslip", "yaw_rate", "roll_rate"])
def test_value_equals_an_independent_kutta_joukowski_sum(half, condition):
    """Batch, single-condition and public paths give the Kutta-Joukowski sum of the wake strips."""
    solver = _solver(half=half)
    res = solver.solve(condition)
    cy, _ = _independent_forces(solver, condition)
    assert res.CY_farfield == pytest.approx(cy, rel=1e-10)
    assert abs(res.CY_farfield) > 1e-5
    key, _, _ = solver._config_key(condition, None)
    cfg = solver._configs[key]
    single = solver._evaluate_result(cfg, condition)
    assert single.CY_farfield == pytest.approx(res.CY_farfield, rel=1e-12)
    mu = solver._reconstruct_full_mu(cfg.unit_mu, solver._condition_coeffs(condition))
    value = compute_trefftz_side_force(cfg.full_wake, mu, condition, solver.S_ref)
    assert value == pytest.approx(res.CY_farfield, rel=1e-12)


@pytest.mark.parametrize("dihedral_deg", [0.0, 10.0])
def test_the_same_sum_along_the_lift_direction_gives_the_lift(dihedral_deg):
    """The Kutta-Joukowski sum with circulation -Gamma gives the lift of the wing (sign check).

    On the 8 x 8 mesh the far-field lift is within 2 % of the pressure lift
    (0.9 % on the planar wing).
    """
    solver = _solver(half=False, dihedral_deg=dihedral_deg)
    cond = FlightCondition(V_inf=V_INF, alpha=ALPHA)
    res = solver.solve(cond)
    _, cl_far = _independent_forces(solver, cond)
    assert res.CL > 0.2
    assert cl_far == pytest.approx(res.CL, rel=0.02)


# ---------------------------------------------------------------------------
# 2. Sign, antisymmetry and the symmetric case
# ---------------------------------------------------------------------------

def test_sign_and_antisymmetry():
    """Dihedral wing: sideslip beta > 0 (wind from the right) gives a side force to the left, CY_farfield < 0.

    A roll rate p > 0 gives CY_farfield < 0 and a yaw rate r > 0 gives
    CY_farfield > 0. The opposite onset gives the opposite value. The
    lattice has the same signs.
    """
    solver = _solver()
    beta, beta_back = solver.solve(_beta()).CY_farfield, solver.solve(_beta(-5.0)).CY_farfield
    roll, roll_back = solver.solve(_roll()).CY_farfield, solver.solve(_roll(-1.0)).CY_farfield
    yaw, yaw_back = solver.solve(_yaw()).CY_farfield, solver.solve(_yaw(-1.0)).CY_farfield
    assert beta < 0.0 and roll < 0.0 and yaw > 0.0
    assert beta_back == pytest.approx(-beta, rel=1e-10)
    assert roll_back == pytest.approx(-roll, rel=1e-10)
    assert yaw_back == pytest.approx(-yaw, rel=1e-10)
    ac = _aircraft(dihedral_deg=10.0)
    lattice = [_cy(ac, c, _lattice()) for c in (_beta(), _roll(), _yaw())]
    assert lattice[0] < 0.0 and lattice[1] < 0.0 and lattice[2] > 0.0


def test_planar_wing_gives_zero():
    """A planar wing has no trace component along z: no far-field side force in sideslip or with rates."""
    solver = _solver(dihedral_deg=0.0)
    for cond in (_beta(), _yaw(), _roll()):
        assert abs(solver.solve(cond).CY_farfield) < 1e-14


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_symmetric_flow_gives_zero(kutta):
    """No asymmetric onset on a half mesh: 0 exactly; on a full mesh: round-off (about 1e-13 of CL)."""
    cond = FlightCondition(V_inf=V_INF, alpha=ALPHA, q=0.1)
    assert _solver(kutta=kutta).solve(cond).CY_farfield == 0.0
    assert abs(_solver(half=False, kutta=kutta).solve(cond).CY_farfield) < 1e-12


def test_pressure_kutta_value_equals_the_sum_of_its_wake():
    """Pressure Kutta: the value is the Kutta-Joukowski sum of the corrected wake strengths."""
    solver = _solver(half=False, kutta="pressure")
    cond = _beta()
    res = solver.solve(cond)
    key, _, _ = solver._config_key(cond, None)
    cfg = solver._configs[key]
    value = compute_trefftz_side_force(cfg.full_wake, None, cond, solver.S_ref, wake_gamma=res.mu_w)
    assert res.CY_farfield == pytest.approx(value, rel=1e-12)
    morino = _solver(half=False).solve(cond).CY_farfield
    assert res.CY_farfield == pytest.approx(morino, rel=0.1)


def test_totals_carry_the_value():
    """The factory fills totals.CY_farfield, and the clone and the dictionary keep it."""
    res = _quiet(vt.analyze, _aircraft(dihedral_deg=10.0), _beta(), _panel(8))
    t = res.totals
    assert t.CY_farfield is not None and t.CY_farfield < 0.0
    assert t.clone().CY_farfield == t.CY_farfield
    assert t.to_dict()["CY_farfield"] == t.CY_farfield
    assert res.details["panel_result"].CY_farfield == t.CY_farfield


# ---------------------------------------------------------------------------
# 3. Parity with the vortex lattice
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("kutta", ["morino", "pressure"])
@pytest.mark.parametrize("condition", [_beta(), _yaw(), _roll()], ids=["sideslip", "yaw_rate", "roll_rate"])
def test_agrees_with_the_vortex_lattice(kutta, condition):
    """Wing with 10 deg dihedral (NACA 0006), alpha 4 deg: the 3D panel and lattice CY_farfield agree.

    The lattice wake is along the body x axis, as the free-air 3D panel
    wake.
    """
    ac = _aircraft(dihedral_deg=10.0)
    panel = _cy(ac, condition, _panel(panel_kutta=kutta))
    lattice = _cy(ac, condition, _lattice())
    assert np.sign(panel) == np.sign(lattice)
    assert panel == pytest.approx(lattice, rel=PARITY_TOLERANCE)


# ---------------------------------------------------------------------------
# 4. Viscous coupling
# ---------------------------------------------------------------------------

def _viscous_wing() -> LiftingSurface:
    return LiftingSurface(
        name="Wing", semi_span=4.0, spacing="uniform", dihedral=np.radians(10.0),
        sections=[WingSection(y_frac=0, chord=1, shape="0012"), WingSection(y_frac=1, chord=1, shape="0012")],
    )


def _viscous_settings() -> SolverSettings:
    return SolverSettings(n_panels=3, n_chord=10, spacing="uniform", panel_backend="numpy", panel_cache=False)


@pytest.mark.parametrize("condition", [_beta(), _yaw()], ids=["sideslip", "yaw_rate"])
def test_viscous_coupling_gives_the_inviscid_value_for_the_identity_polar(condition):
    """viscous_panel3d fills Cn_trefftz and CY_farfield from its coupled wake.

    With the converged inviscid 2D section as the polar of each strip (the
    identity polar) the coupling makes no correction, and both values
    equal those of the inviscid solve. A polar with a lower lift slope
    gives smaller values of the same sign.
    """
    from ventorum.core.datatypes import LinearAirfoil
    from ventorum.solvers.factory import make_solver
    from ventorum.viscous import alpha_correction as module
    from ventorum.viscous.alpha_correction import solve_viscous

    captured = {}
    original = module._PanelCoupling.references

    def record(self, ground, uses_height, curves):
        refs = original(self, ground, uses_height, curves)
        captured["refs"] = list(self.full(np.array(refs, dtype=object)))
        return refs

    class _Identity(LinearAirfoil):
        def __init__(self, curve):
            super().__init__(a0=2 * np.pi)
            self.curve = curve

        def Cl(self, angle):
            return np.vectorize(lambda a: self.curve.evaluate(a)[0], otypes=[float])(np.asarray(angle, dtype=float))

    wing, settings = _viscous_wing(), _viscous_settings()
    module._PanelCoupling.references = record
    try:
        low = _quiet(solve_viscous, wing, condition, settings, LinearAirfoil(a0=5.0), model="panel3d", tol=1e-10)
        polars = {i: _Identity(curve) for i, curve in enumerate(captured["refs"])}
        res = _quiet(solve_viscous, wing, condition, settings, polars, model="panel3d", tol=1e-10)
    finally:
        module._PanelCoupling.references = original
    inv = _quiet(make_solver("panel3d").solve, wing, condition, settings)
    assert res.solver_type == "viscous_panel3d" and res.converged
    assert np.max(np.abs(res.details["delta"])) < 1e-9
    for key in ("Cn_trefftz", "CY_farfield"):
        assert getattr(res.totals, key) is not None
        assert getattr(res.totals, key) == pytest.approx(getattr(inv.totals, key), rel=1e-9, abs=1e-14)
        assert getattr(res.details["panel_result"], key) == getattr(res.totals, key)
        assert np.sign(getattr(low.totals, key)) == np.sign(getattr(inv.totals, key))
        assert abs(getattr(low.totals, key)) < abs(getattr(inv.totals, key))


def test_viscous_symmetric_flow_gives_zero():
    """viscous_panel3d on a half mesh with a symmetric flow: both values are 0 exactly."""
    from ventorum.core.datatypes import LinearAirfoil
    from ventorum.viscous.alpha_correction import solve_viscous

    res = _quiet(solve_viscous, _viscous_wing(), FlightCondition(V_inf=V_INF, alpha=ALPHA), _viscous_settings(),
                 LinearAirfoil(a0=5.0), model="panel3d", tol=1e-10)
    assert res.symmetry_used
    assert res.totals.Cn_trefftz == 0.0 and res.totals.CY_farfield == 0.0


# ---------------------------------------------------------------------------
# 6. GPU
# ---------------------------------------------------------------------------

def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


@pytest.mark.gpu
@pytest.mark.parametrize("half", [True, False])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_gpu_matches_numpy(dtype, half):
    """GPU batch and single-condition paths match numpy (float64 to 1e-10, float32 to 2e-3).

    The GPU gives the strip circulations in float64 (from the float64 unit
    doublets on a half mesh, from the solve precision on a full mesh), and
    the host computes the side force. The float32 floor: the float32
    doublets of a full-mesh solve are not symmetric to rounding, which gives
    about 1.2e-7 in CY_farfield of the symmetric flow (float32 rounding of
    the circulation, of order 1e-7 of CL = 0.33).
    """
    torch = _need_cuda()
    conditions = [_beta(), _yaw(), _roll(), FlightCondition(V_inf=V_INF, alpha=ALPHA)]
    ref = _solver(half=half, backend="numpy").solve_conditions(conditions)
    gpu = _solver(half=half, backend="gpu", dtype=getattr(torch, dtype))
    rel, floor = (1e-10, 1e-14) if dtype == "float64" else (2e-3, 1e-6)
    for cond, a, b in zip(conditions, gpu.solve_conditions(conditions), ref):
        assert a.CY_farfield == pytest.approx(b.CY_farfield, rel=rel, abs=floor)
        key, _, _ = gpu._config_key(cond, None)
        single = gpu._evaluate_result(gpu._configs[key], cond)
        assert single.CY_farfield == pytest.approx(b.CY_farfield, rel=rel, abs=floor)
