# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Yawing moment of the Trefftz-plane induced drag of the 3D panel solver (``Cn_trefftz``).

Checks:

* the value equals an independent sum ``sum D'_s y_s / (q S b)`` of the
  strip induced drag of the wake (body-axis wake, so the lateral arm is
  the strip midpoint y);
* the sign (AVL convention, ``Cn > 0`` nose right) and the antisymmetry in
  the rate; a symmetric flow on a half mesh gives 0 exactly, as in the
  vortex lattice;
* parity with the vortex-lattice ``Cn_trefftz`` (stability axes) on a thin
  planar wing in sideslip and with yaw and roll rates, and on a wing with
  dihedral in sideslip;
* every path: Morino batch and single condition, pressure Kutta, half and
  full mesh, GPU float64 and float32.

The formula is in the theory manual (docs/theory/ventorum_3dp.md, far-field
induced drag) and in docs/theory/forces_and_moments.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings, WingSection
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.post import compute_trefftz_yawing_moment
from ventorum.panel3d.solver import PanelSolver

V_INF = 30.0
ALPHA = np.radians(4.0)
#: Stability-axis yaw rate of a nondimensional rate r_hat = r b / (2 V) = 0.01 [rad/s].
R_STAB = 0.01 * 2.0 * V_INF / 8.0

#: Relative agreement of the 3D panel and the vortex-lattice Cn_trefftz [-].
#: The 3D panel wing has a NACA 0006 section and the lattice is thin; the
#: thickness adds about 4 % to the lift slope and the roll damping. On the
#: 16 x 16 mesh against the lattice 40 x 8: yaw rate -1.690e-05 against
#: -1.628e-05 (+3.8 %), roll rate +2.422e-04 against +2.335e-04 (+3.8 %),
#: 10 deg dihedral at beta 5 deg +5.645e-04 against +5.472e-04 (+3.2 %).
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


def _aircraft(shape: str = "0006", dihedral_deg: float = 0.0) -> Aircraft:
    """Rectangular wing, span 8 m, chord 1 m, quarter chord on the origin."""
    secs = [WingSection(0.0, 1.0, x_le=-0.25, shape=shape), WingSection(1.0, 1.0, x_le=-0.25, shape=shape)]
    return Aircraft(surfaces=[LiftingSurface(semi_span=4.0, sections=secs, dihedral=np.radians(dihedral_deg))])


def _yaw(sign: float = 1.0) -> FlightCondition:
    """Stability-axis yaw rate: body rates p = -r_s sin(alpha), r = r_s cos(alpha) [rad/s]."""
    return FlightCondition(V_inf=V_INF, alpha=ALPHA, p=-sign * R_STAB * np.sin(ALPHA),
                           r=sign * R_STAB * np.cos(ALPHA))


def _roll() -> FlightCondition:
    return FlightCondition(V_inf=V_INF, alpha=ALPHA, p=R_STAB)


def _beta() -> FlightCondition:
    return FlightCondition(V_inf=V_INF, alpha=ALPHA, beta=np.radians(5.0))


def _panel(n: int = 16, **kw) -> SolverSettings:
    return SolverSettings(solver_type="panel3d", n_panels=n, n_chord=n, **kw)


def _lattice(**kw) -> SolverSettings:
    return SolverSettings(solver_type="vlm", n_panels=40, n_chord=8, **kw)


def _cn(aircraft, condition, settings) -> float:
    return _quiet(vt.analyze, aircraft, condition=condition, settings=settings).totals.Cn_trefftz


def _solver(half: bool = True, kutta: str = "morino", ref_point=(0.0, 0.0, 0.0), **kw) -> PanelSolver:
    """Small NACA 0006 wing solver (span 8 m, chord 1 m), reference point at the quarter chord."""
    surface = LiftingSurface(
        name="Wing", semi_span=4.0, is_symmetric=True,
        sections=[WingSection(0.0, 1.0, x_le=-0.25, shape="0006"),
                  WingSection(1.0, 1.0, x_le=-0.25, shape="0006")],
    )
    from ventorum.airfoil import airfoil

    mesh = mesh_lifting_surface(surface, airfoil("0006").coords, n_chord=8, n_span=8, half=half)
    return PanelSolver(mesh, S_ref=8.0, b_ref=8.0, c_ref=1.0, ref_point=np.asarray(ref_point, dtype=float),
                       kutta=kutta, **kw)


def _independent_cn(solver: PanelSolver, condition: FlightCondition) -> tuple[float, float]:
    """Trefftz drag and yawing moment from the wake strips, written out again (body-axis wake, free air).

    Each strip end carries a 2D point vortex in the y-z plane; the strip
    drag is ``D' = -0.5 rho Gamma w_n l`` and the yawing moment in
    stability axes (beta = 0) is ``sum D' y / (q S b)``.
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
    pl[flip], pr[flip] = np.array(wake.p_right, float)[flip], np.array(wake.p_left, float)[flip]
    length = np.hypot(pr[:, 1] - pl[:, 1], pr[:, 2] - pl[:, 2])
    mid = 0.5 * (pl + pr)
    normal = np.column_stack([-(pr[:, 2] - pl[:, 2]), pr[:, 1] - pl[:, 1]]) / length[:, None]
    rc2 = (0.01 * length.mean()) ** 2
    w_n = np.zeros(len(gamma))
    for vertex, sign in ((pr, 1.0), (pl, -1.0)):
        dy = mid[:, None, 1] - vertex[None, :, 1]
        dz = mid[:, None, 2] - vertex[None, :, 2]
        den = 2.0 * np.pi * (dy**2 + dz**2 + rc2)
        w_n += np.sum(sign * gamma[None, :] * (-dz * normal[:, 0, None] + dy * normal[:, 1, None]) / den, axis=1)
    drag = -0.5 * condition.rho * gamma * w_n * length
    q_s = 0.5 * condition.rho * condition.V_inf**2 * solver.S_ref
    return float(drag.sum() / q_s), float(np.sum(drag * mid[:, 1]) / (q_s * solver.b_ref))


# ---------------------------------------------------------------------------
# 1. Formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("half", [True, False])
@pytest.mark.parametrize("condition", [_yaw(), _roll()], ids=["yaw_rate", "roll_rate"])
def test_value_equals_an_independent_strip_sum(half, condition):
    """Batch and single-condition paths give sum D' y / (q S b) of the wake strips."""
    solver = _solver(half=half)
    res = solver.solve(condition)
    cdi, cn = _independent_cn(solver, condition)
    assert res.CDi_trefftz == pytest.approx(cdi, rel=1e-10)
    assert res.Cn_trefftz == pytest.approx(cn, rel=1e-10)
    assert abs(res.Cn_trefftz) > 1e-6
    key, _, _ = solver._config_key(condition, None)
    single = solver._evaluate_result(solver._configs[key], condition)
    assert single.Cn_trefftz == pytest.approx(res.Cn_trefftz, rel=1e-12)
    # The public function gives the same value.
    cfg = solver._configs[key]
    mu = solver._reconstruct_full_mu(cfg.unit_mu, solver._condition_coeffs(condition))
    value = compute_trefftz_yawing_moment(cfg.full_wake, mu, condition, solver.S_ref, solver.b_ref,
                                          ref_point=solver.ref_point)
    assert value == pytest.approx(res.Cn_trefftz, rel=1e-12)


def test_reference_point_moves_the_arm_in_the_trefftz_plane_only():
    """Body-axis wake: the Trefftz plane is y-z (same doublets, other reference points).

    A shift of the reference point along x changes nothing. A shift
    ``y_ref`` along y changes every lateral arm by ``-y_ref``, so
    ``Cn_trefftz`` changes by ``-y_ref CDi / b``.
    """
    solver = _solver()
    cond = _yaw()
    res = solver.solve(cond)
    key, _, _ = solver._config_key(cond, None)
    cfg = solver._configs[key]
    mu = solver._reconstruct_full_mu(cfg.unit_mu, solver._condition_coeffs(cond))

    def cn(ref_point):
        return compute_trefftz_yawing_moment(cfg.full_wake, mu, cond, solver.S_ref, solver.b_ref,
                                             ref_point=np.array(ref_point))

    assert cn((0.7, 0.0, 0.0)) == pytest.approx(res.Cn_trefftz, rel=1e-12)
    assert cn((0.0, 0.5, 0.0)) == pytest.approx(res.Cn_trefftz - 0.5 * res.CDi_trefftz / 8.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. Sign, antisymmetry and the symmetric case
# ---------------------------------------------------------------------------

def test_sign_and_antisymmetry():
    """Roll rate p > 0 (right wing down) loads the right wing: more drag there, Cn_trefftz > 0.

    A yaw rate r > 0 loads the advancing left wing: Cn_trefftz < 0 (yaw
    damping). The opposite rate gives the opposite value. The lattice has
    the same signs.
    """
    solver = _solver()
    roll = solver.solve(_roll()).Cn_trefftz
    yaw, yaw_back = solver.solve(_yaw()).Cn_trefftz, solver.solve(_yaw(-1.0)).Cn_trefftz
    assert roll > 0.0 and yaw < 0.0
    assert yaw_back == pytest.approx(-yaw, rel=1e-10)
    ac = _aircraft()
    assert _cn(ac, _roll(), _lattice()) > 0.0 and _cn(ac, _yaw(), _lattice()) < 0.0


@pytest.mark.parametrize("kutta", ["morino", "pressure"])
def test_symmetric_flow_gives_zero(kutta):
    """No asymmetric onset on a half mesh: 0 exactly; on a full mesh: round-off."""
    cond = FlightCondition(V_inf=V_INF, alpha=ALPHA, q=0.1)
    assert _solver(kutta=kutta).solve(cond).Cn_trefftz == 0.0
    assert abs(_solver(half=False, kutta=kutta).solve(cond).Cn_trefftz) < 1e-14


def test_totals_carry_the_value():
    """The factory fills totals.Cn_trefftz, and the clone and the dictionary keep it."""
    res = _quiet(vt.analyze, _aircraft(), _yaw(), _panel(8))
    t = res.totals
    assert t.Cn_trefftz is not None and t.Cn_trefftz < 0.0
    assert t.clone().Cn_trefftz == t.Cn_trefftz
    assert t.to_dict()["Cn_trefftz"] == t.Cn_trefftz
    assert t.CY_farfield is None        # the far-field side force is not computed for the 3D panel solver


# ---------------------------------------------------------------------------
# 3. Parity with the vortex lattice
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("kutta", ["morino", "pressure"])
@pytest.mark.parametrize("condition", [_yaw(), _roll()], ids=["yaw_rate", "roll_rate"])
def test_rates_agree_with_the_vortex_lattice(kutta, condition):
    """Thin planar wing (NACA 0006), alpha 4 deg: the 3D panel and lattice Cn_trefftz agree."""
    ac = _aircraft()
    panel = _cn(ac, condition, _panel(panel_kutta=kutta))
    lattice = _cn(ac, condition, _lattice())
    assert np.sign(panel) == np.sign(lattice)
    assert panel == pytest.approx(lattice, rel=PARITY_TOLERANCE)


@pytest.mark.slow
def test_sideslip_agrees_with_the_vortex_lattice():
    """Sideslip, beta 5 deg, with the lattice wake along the body x axis (the free-air 3D panel wake).

    On a planar wing the side onset lies in the plane of the wing: the
    circulation and the Trefftz drag stay symmetric and both solvers give
    0. With 10 deg dihedral the side onset has a normal component on each
    wing half, and the two values agree.
    """
    planar = _aircraft()
    assert abs(_cn(planar, _beta(), _panel())) < 1e-9
    assert abs(_cn(planar, _beta(), _lattice(wake_alignment="body"))) < 1e-9
    dihedral = _aircraft(dihedral_deg=10.0)
    panel = _cn(dihedral, _beta(), _panel())
    lattice = _cn(dihedral, _beta(), _lattice(wake_alignment="body"))
    assert panel > 0.0 and lattice > 0.0
    assert panel == pytest.approx(lattice, rel=PARITY_TOLERANCE)


# ---------------------------------------------------------------------------
# 4. GPU
# ---------------------------------------------------------------------------

def _need_cuda():
    """Skip the test when a CUDA GPU, PyTorch or Warp is missing."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    return torch


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_gpu_matches_numpy(dtype):
    """Gpu batch and single-condition paths match numpy (float64 to 1e-10, float32 to its floor).

    The float32 floor: the float32 doublets of a full-mesh solve are not
    symmetric to rounding, which gives about 1e-8 in Cn_trefftz.
    """
    torch = _need_cuda()
    conditions = [_yaw(), _roll(), _beta(), FlightCondition(V_inf=V_INF, alpha=ALPHA)]
    ref = _solver(backend="numpy").solve_conditions(conditions)
    gpu = _solver(backend="gpu", dtype=getattr(torch, dtype))
    rel, floor = (1e-10, 1e-14) if dtype == "float64" else (2e-3, 5e-8)
    for cond, a, b in zip(conditions, gpu.solve_conditions(conditions), ref):
        assert a.Cn_trefftz == pytest.approx(b.Cn_trefftz, rel=rel, abs=floor)
        key, _, _ = gpu._config_key(cond, None)
        single = gpu._evaluate_result(gpu._configs[key], cond)
        assert single.Cn_trefftz == pytest.approx(b.Cn_trefftz, rel=rel, abs=floor)
