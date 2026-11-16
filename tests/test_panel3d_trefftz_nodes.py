"""Trailing vortices of the 3D panel Trefftz plane, merged per trace node.

The strip ends at one trace node are one vortex with the net circulation
jump, and the node has one core radius from the shortest adjacent strip.
Two vortices of different cores at one node do not cancel: next to a
narrow strip (a control-surface edge of the mesh builder) the old rule
gave a false wash proportional to the full strip circulation.

Checks:

* a strip split into two strips of width ratio 0.01 with equal
  circulation gives the wash and the drag of the unsplit strip;
* a flapped rectangular wing gives a smooth CDi_trefftz with the spanwise
  panel count;
* an aileron case gives the vortex-lattice Cn_trefftz;
* the GPU path gives the CPU values on a flapped and an aileron mesh.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ventorum as vt
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft,
    ControlSurface,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
    WingSection,
)
from ventorum.panel3d.mesh import mesh_lifting_surface
from ventorum.panel3d.post import (
    TREFFTZ_CORE_FRACTION,
    _trefftz_batch,
    _trefftz_factors,
    _trefftz_strip_wash,
    compute_trefftz_induced_drag,
)
from ventorum.panel3d.solver import PanelSolver
from ventorum.panel3d.wake import PanelWake

V_INF = 30.0
ALPHA = np.radians(3.0)
S_REF = 7.2
B_REF = 6.0
C_REF = 1.2


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


def _strip_wake(edges_y: np.ndarray) -> PanelWake:
    """Return a planar wake (z = 0, x = 1 m) with strips between the given y stations [m]."""
    n = len(edges_y) - 1
    pl = np.column_stack([np.ones(n), edges_y[:-1], np.zeros(n)])
    pr = np.column_stack([np.ones(n), edges_y[1:], np.zeros(n)])
    return PanelWake(
        panels=np.empty((0, 4, 3)),
        wake_up=np.arange(n),
        wake_lo=np.arange(n),
        strip_panel_indices=[],
        p_left=pl,
        p_right=pr,
        wake_edges=np.zeros((n, 4), dtype=int),
        cp_frac=np.full(n, 0.5),
    )


def _per_strip_core_wash(wake: PanelWake, gamma: np.ndarray) -> np.ndarray:
    """Normal wash [m/s] of the previous rule: one core per strip, no merge of the ends."""
    pl, pr = wake.p_left, wake.p_right
    length = np.linalg.norm(pr - pl, axis=1)
    q = pl + wake.cp_frac[:, None] * (pr - pl)
    pv = np.vstack([pr, pl])
    gv = np.concatenate([gamma, -gamma])
    rc2 = (TREFFTZ_CORE_FRACTION * np.concatenate([length, length])) ** 2
    dy = q[:, None, 1] - pv[None, :, 1]
    dz = q[:, None, 2] - pv[None, :, 2]
    den = 2.0 * np.pi * (dy**2 + dz**2 + rc2[None, :])
    # Planar wake along y: the in-plane normal is z, so w_n = v_z.
    return np.sum(gv[None, :] * dy / den, axis=1)


def _split_sheet(gamma_of_y) -> tuple:
    """Return the unsplit and the split planar sheet (16 strips, span 8 m) and their circulations.

    Strip 10 is split into two strips of width ratio 0.01 (w and 100 w),
    each with the circulation of the unsplit strip [m^2/s].
    """
    base = np.linspace(-4.0, 4.0, 17)
    k = 10
    w = (base[k + 1] - base[k]) / 101.0
    split = np.insert(base, k + 1, base[k] + w)
    gamma_m = gamma_of_y(0.5 * (base[:-1] + base[1:]))
    gamma_s = np.insert(gamma_m, k + 1, gamma_m[k])
    return k, base, split, gamma_m, gamma_s


def _cdi(wake, gamma) -> float:
    return compute_trefftz_induced_drag(wake, np.zeros(0), V_INF, 1.225, 8.0, wake_gamma=gamma)


def _old_rule_cdi(edges, gamma) -> float:
    wash = _per_strip_core_wash(_strip_wake(edges), gamma)
    return -0.5 * 1.225 * float(np.sum(gamma * wash * np.diff(edges))) / (0.5 * 1.225 * V_INF**2 * 8.0)


def test_split_strip_of_width_ratio_one_percent_equals_one_strip():
    """A strip split into widths w and 100 w with equal circulation acts as one strip.

    Uniform circulation: only the two tip nodes carry a circulation jump,
    and their cores do not change. The interior nodes, also the node of
    the split, carry no net circulation, so the merged rule leaves no
    vortex there. The wash is then the same to round-off at every
    evaluation point. The drag changes only because the split moves the
    evaluation points of the drag sum (a quadrature change, 1.7e-6
    relative here). The previous rule (one core per strip end) leaves a
    vortex pair of different cores at the split node: its drag is 34 % too
    large.
    """
    k, base, split, gamma_m, gamma_s = _split_sheet(np.ones_like)
    merged_wake, split_wake = _strip_wake(base), _strip_wake(split)
    _, wn_m, _ = _trefftz_strip_wash(merged_wake, np.zeros(0), None, None, gamma_m)
    _, wn_s, _ = _trefftz_strip_wash(split_wake, np.zeros(0), None, None, gamma_s)
    scale = float(np.max(np.abs(wn_m)))
    others_s = np.delete(np.arange(len(gamma_s)), [k, k + 1])
    others_m = np.delete(np.arange(len(gamma_m)), [k])
    np.testing.assert_allclose(wn_s[others_s], wn_m[others_m], rtol=0.0, atol=1e-13 * scale)

    # The wash of the two split strips: the unsplit sheet evaluated at the same points.
    for idx in (k, k + 1):
        probe = _strip_wake(base)
        probe.cp_frac[k] = (0.5 * (split[idx] + split[idx + 1]) - base[k]) / (base[k + 1] - base[k])
        _, wn_p, _ = _trefftz_strip_wash(probe, np.zeros(0), None, None, gamma_m)
        assert wn_s[idx] == pytest.approx(wn_p[k], rel=0.0, abs=1e-13 * scale)

    cdi_m, cdi_s = _cdi(merged_wake, gamma_m), _cdi(split_wake, gamma_s)
    assert cdi_s == pytest.approx(cdi_m, rel=1e-5)

    # The batch factors give the drag of the single path.
    cdi_batch = _trefftz_batch(gamma_s[None, :], np.array([V_INF]), np.array([1.225]), 8.0,
                               _trefftz_factors(split_wake))[0]
    assert cdi_batch == pytest.approx(cdi_s, rel=1e-12)

    # The previous rule fails this check.
    assert abs(_old_rule_cdi(split, gamma_s) - cdi_m) > 0.2 * cdi_m


def test_split_strip_next_to_a_circulation_jump():
    """Elliptic step circulation: the split strip next to a jump changes the drag by less than 2 %.

    The narrow strip has its evaluation point at w / 2 from the node at its
    outer end, where the circulation jumps. That node gets the core of the
    narrow strip (0.01 w), so the narrow strip has a large wash, and its
    drag term is of the order of the jump times its circulation. This is a
    term of the midpoint quadrature, not of the merge (1.3 % here). The
    previous rule gives 61 %. The wash of the other strips changes by less
    than 1e-4 of the largest wash: the cores of the two outer nodes are
    smaller.
    """
    k, base, split, gamma_m, gamma_s = _split_sheet(lambda y: np.sqrt(1.0 - (y / 4.0) ** 2))
    merged_wake, split_wake = _strip_wake(base), _strip_wake(split)
    _, wn_m, _ = _trefftz_strip_wash(merged_wake, np.zeros(0), None, None, gamma_m)
    _, wn_s, _ = _trefftz_strip_wash(split_wake, np.zeros(0), None, None, gamma_s)
    scale = float(np.max(np.abs(wn_m)))
    others_s = np.delete(np.arange(len(gamma_s)), [k, k + 1])
    others_m = np.delete(np.arange(len(gamma_m)), [k])
    np.testing.assert_allclose(wn_s[others_s], wn_m[others_m], rtol=0.0, atol=1e-4 * scale)
    cdi_m = _cdi(merged_wake, gamma_m)
    assert _cdi(split_wake, gamma_s) == pytest.approx(cdi_m, rel=0.02)
    assert abs(_old_rule_cdi(split, gamma_s) - cdi_m) > 0.3 * cdi_m


def test_equal_neighbour_strips_change_only_at_round_off():
    """Equal strip widths: one core per node is the old core, so only round-off changes."""
    edges = np.linspace(-4.0, 4.0, 25)
    wake = _strip_wake(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.sqrt(1.0 - (mid / 4.0) ** 2)
    _, wn, _ = _trefftz_strip_wash(wake, np.zeros(0), None, None, gamma)
    np.testing.assert_allclose(wn, _per_strip_core_wash(wake, gamma), rtol=1e-12, atol=1e-14)


def _wing(controls) -> LiftingSurface:
    """Rectangular NACA 0012 wing, semi-span 3 m, chord 1.2 m."""
    surf = LiftingSurface(
        name="Wing", semi_span=3.0, is_symmetric=True,
        sections=[WingSection(0.0, C_REF, shape="0012"), WingSection(1.0, C_REF, shape="0012")],
    )
    surf.controls = list(controls)
    return surf


def _flap() -> list:
    """Plain flap from 30 to 70 percent semi-span, hinge at 75 percent chord, 10 deg down."""
    return [ControlSurface(name="flap", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75,
                           deflection=np.radians(10.0), symmetric=True)]


def _aileron() -> list:
    """Aileron from 30 to 70 percent semi-span, hinge at 75 percent chord, +10 / -10 deg."""
    return [ControlSurface(name="aileron", eta_start=0.3, eta_end=0.7, hinge_x_c=0.75,
                           deflection=np.radians(10.0), symmetric=False)]


def _panel_solver(controls, n_span: int, **kw) -> PanelSolver:
    mesh = mesh_lifting_surface(_wing(controls), airfoil("0012").coords, n_chord=10, n_span=n_span)
    return PanelSolver(mesh, S_ref=S_REF, b_ref=B_REF, c_ref=C_REF, **kw)


#: CDi_trefftz [-] of the flapped wing with the merged rule at n_span 12,
#: 24 and 48 (n_chord 10). The vortex lattice (n_panels 48, n_chord 10)
#: gives about 0.0130. The previous rule gave 0.013008, 0.026902 and
#: 0.020929: the sliver strips at the flap edges made a false wash.
FLAP_CDI = {12: 0.012956, 24: 0.012590, 48: 0.012559}


@pytest.mark.slow
def test_flapped_wing_drag_is_smooth_with_the_span_count():
    """Flapped wing, alpha 3 deg: CDi_trefftz within 3 % of the merged values, steps below 5 %."""
    cond = FlightCondition(V_inf=V_INF, alpha=ALPHA)
    values = []
    for n_span, ref in FLAP_CDI.items():
        cdi = _panel_solver(_flap(), n_span).solve(cond).CDi_trefftz
        assert cdi == pytest.approx(ref, rel=0.03)
        values.append(cdi)
    steps = np.abs(np.diff(values)) / np.asarray(values[1:])
    assert np.all(steps < 0.05), steps


@pytest.mark.slow
def test_aileron_yaw_agrees_with_the_vortex_lattice():
    """Aileron +10 / -10 deg, alpha 3 deg: Cn_trefftz within 15 % of the lattice value.

    The previous rule gave about twice the lattice value (3.07e-3) and a
    CDi_trefftz of 0.0142 (lattice 0.0083).
    """
    cond = FlightCondition(V_inf=V_INF, alpha=ALPHA)
    solver = _panel_solver(_aileron(), 24)
    panel = solver.solve(cond)
    ac = Aircraft(surfaces=[_wing(_aileron())], ref_point=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lattice = vt.analyze(ac, condition=cond,
                             settings=SolverSettings(solver_type="vlm", n_panels=48, n_chord=10)).totals
    assert lattice.Cn_trefftz > 0.0
    assert panel.Cn_trefftz == pytest.approx(lattice.Cn_trefftz, rel=0.15)
    # The single-condition path gives the batch value.
    key, _, _ = solver._config_key(cond, None)
    single = solver._evaluate_result(solver._configs[key], cond)
    assert single.CDi_trefftz == pytest.approx(panel.CDi_trefftz, rel=1e-10)
    assert single.Cn_trefftz == pytest.approx(panel.Cn_trefftz, rel=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("controls", [_flap, _aileron], ids=["flap", "aileron"])
def test_gpu_matches_cpu_with_controls(controls):
    """GPU float64 gives the CPU CDi_trefftz and Cn_trefftz of a mesh with sliver strips."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is not available.")
    conditions = [FlightCondition(V_inf=V_INF, alpha=ALPHA),
                  FlightCondition(V_inf=V_INF, alpha=ALPHA, beta=np.radians(3.0))]
    ref = _panel_solver(controls(), 12, backend="numpy").solve_conditions(conditions)
    gpu = _panel_solver(controls(), 12, backend="gpu", dtype="float64")
    for cond, a, b in zip(conditions, gpu.solve_conditions(conditions), ref):
        assert a.CDi_trefftz == pytest.approx(b.CDi_trefftz, rel=1e-10, abs=1e-14)
        assert a.Cn_trefftz == pytest.approx(b.Cn_trefftz, rel=1e-10, abs=1e-14)
        key, _, _ = gpu._config_key(cond, None)
        single = gpu._evaluate_result(gpu._configs[key], cond)
        assert single.CDi_trefftz == pytest.approx(b.CDi_trefftz, rel=1e-10, abs=1e-14)
