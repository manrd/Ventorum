"""Tests for the VLM evaluation point and per-strip core radius in 3DP Trefftz drag."""

from __future__ import annotations

import numpy as np
import pytest

import ventorum as vt
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    WingSection,
)
from ventorum.geometry.lattice import CORE_RADIUS_FRACTION
from ventorum.panel3d.images import reconstruct_full_mesh
from ventorum.panel3d.mesh import PanelMesh, mesh_lifting_surface
from ventorum.panel3d.post import (
    TREFFTZ_CORE_FRACTION,
    _trefftz_batch,
    _trefftz_factors,
    compute_trefftz_induced_drag,
    compute_trefftz_yawing_moment,
)
from ventorum.panel3d.solver import PanelSolver
from ventorum.panel3d.wake import build_wake


def _rect_wing(semi_span: float = 4.0, chord: float = 1.0) -> LiftingSurface:
    """Return a rectangular NACA 0012 symmetric wing [m]."""
    sections = [
        WingSection(y_frac=0.0, chord=chord, shape="0012"),
        WingSection(y_frac=1.0, chord=chord, shape="0012"),
    ]
    return LiftingSurface(name="Wing", semi_span=semi_span, sections=sections, is_symmetric=True)


def _old_rule_trefftz_cdi(
    wake,
    gamma: np.ndarray,
    V_inf: float,
    rho: float,
    S_ref: float,
    *,
    ground: tuple[np.ndarray, float] | None = None,
) -> float:
    """Compute Trefftz CDi [-] using the previous midpoint and global-mean core rule."""
    pl = np.array(wake.p_left, dtype=float, copy=True)
    pr = np.array(wake.p_right, dtype=float, copy=True)
    d = np.array([1.0, 0.0, 0.0], dtype=float)
    if len(wake.panels):
        first = np.fromiter((idx[0] for idx in wake.strip_panel_indices), dtype=int)
        quad = np.asarray(wake.panels)[first]
        nw = np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
        flip = np.sum((pr - pl) * np.cross(nw, d), axis=1) < 0.0
        pl[flip], pr[flip] = wake.p_right[flip], wake.p_left[flip]
    segment = pr - pl
    length = np.linalg.norm(segment, axis=-1)
    safe = np.maximum(length, 1.0e-12)
    q = (0.5 * (pl + pr))[:, 1:]
    seg = segment[:, 1:]
    vertices = np.vstack([pr, pl])
    gv = np.concatenate([gamma, -gamma])
    if ground is not None:
        normal, offset = ground
        height = vertices @ normal - offset
        vertices = np.vstack([vertices, vertices - 2.0 * height[:, None] * normal])
        gv = np.concatenate([gv, -gv])
    pv = vertices[:, 1:]
    nq = np.column_stack([-seg[:, 1], seg[:, 0]]) / safe[:, None]
    rq = q[:, None, :] - pv[None, :, :]
    rc_sq = (TREFFTZ_CORE_FRACTION * float(np.mean(safe))) ** 2
    denom = 2.0 * np.pi * (rq[..., 0] ** 2 + rq[..., 1] ** 2 + rc_sq)
    vy = -rq[..., 1] / denom
    vz = rq[..., 0] / denom
    wash = np.sum(gv[None, :] * vy, axis=-1) * nq[:, 0] + np.sum(gv[None, :] * vz, axis=-1) * nq[:, 1]
    drag = -0.5 * rho * float(np.sum(gamma * wash * length))
    q_inf = 0.5 * rho * V_inf**2 * S_ref
    return float(drag / q_inf)


def _elliptic_wake_and_gamma(n_half: int, semi_span: float = 4.0) -> tuple:
    """Return full wake and elliptic strip circulations evaluated at cp_frac."""
    surf = _rect_wing(semi_span=semi_span, chord=1.0)
    half = mesh_lifting_surface(surf, airfoil("0012").coords, n_span=n_half, n_chord=4, spacing="auto", half=True)
    full = reconstruct_full_mesh(half)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(full.panels), 1))
    wake = build_wake(full.nodes, full.panels, full.wake_edges, normals, cp_frac=full.wake_cp_frac)
    assert wake.cp_frac is not None
    q = wake.p_left + wake.cp_frac[:, None] * (wake.p_right - wake.p_left)
    eta = q[:, 1] / semi_span
    gamma = np.sqrt(np.maximum(1.0 - eta**2, 0.0))
    return wake, gamma


def test_elliptic_load_spanwise_convergence() -> None:
    """Elliptic load converges at second order with the VLM rule and below 1.2 with midpoint."""
    semi_span = 4.0
    s_ref = 2.0 * semi_span * 1.0
    ar = (2.0 * semi_span) ** 2 / s_ref
    v_inf = 1.0
    rho = 1.0
    gamma_0 = 1.0
    cl_exact = np.pi * semi_span * gamma_0 / (v_inf * s_ref)
    cdi_exact = cl_exact**2 / (np.pi * ar)

    cdis_new: list[float] = []
    cdis_old: list[float] = []
    errs_new: list[float] = []
    for n_half in (20, 40, 80):
        wake, gamma = _elliptic_wake_and_gamma(n_half, semi_span=semi_span)
        factors = _trefftz_factors(wake)
        cdi_new = float(
            _trefftz_batch(
                gamma[None, :], np.array([v_inf]), np.array([rho]), s_ref, factors
            )[0]
        )
        cdis_new.append(cdi_new)
        errs_new.append(abs(cdi_new - cdi_exact) / cdi_exact)

        # Previous rule evaluated at the geometric midpoint with global mean core.
        mid = 0.5 * (wake.p_left + wake.p_right)
        gamma_mid = gamma_0 * np.sqrt(np.maximum(1.0 - (mid[:, 1] / semi_span) ** 2, 0.0))
        cdi_old = _old_rule_trefftz_cdi(wake, gamma_mid, v_inf, rho, s_ref)
        cdis_old.append(cdi_old)

    order_new = np.log(abs(cdis_new[1] - cdis_new[0]) / abs(cdis_new[2] - cdis_new[1])) / np.log(2.0)
    order_old = np.log(abs(cdis_old[1] - cdis_old[0]) / abs(cdis_old[2] - cdis_old[1])) / np.log(2.0)
    assert order_new >= 1.9
    assert errs_new[1] < 5.0e-4
    assert order_old < 1.2


@pytest.mark.slow
def test_solved_wing_spanwise_order_and_cl() -> None:
    """Solved AR 8 NACA 0012 at 4 deg keeps bit-identical CL and spanwise order >= 1.1."""
    # Pre-change CPU CL values from the exact same Morino solve at n_chord=20.
    pinned_cl = {
        20: 0.33778004568886427,
        40: 0.33701279298982434,
        80: 0.3366197562777946,
    }
    surf = _rect_wing(semi_span=4.0, chord=1.0)
    ac = Aircraft(surfaces=[surf])
    ac.compute_reference_values()
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))

    cdis: list[float] = []
    for n_span in (20, 40, 80):
        sett = vt.SolverSettings(
            solver_type="panel3d",
            n_chord=20,
            n_panels=n_span,
            panel_kutta="morino",
        )
        res = vt.analyze(ac, condition=cond, settings=sett)
        assert res.totals.CL == pinned_cl[n_span]
        cdis.append(float(res.totals.CDi))

    order = np.log(abs(cdis[1] - cdis[0]) / abs(cdis[2] - cdis[1])) / np.log(2.0)
    assert order >= 1.1


def test_fallback_user_mesh_without_cp_frac() -> None:
    """User-built PanelMesh without wake_cp_frac falls back to 0.5 and per-strip core."""
    surf = _rect_wing(semi_span=4.0, chord=1.0)
    built = mesh_lifting_surface(surf, airfoil("0012").coords, n_span=12, n_chord=6, spacing="auto", half=False)
    user_mesh = PanelMesh(
        nodes=built.nodes.copy(),
        panels=built.panels.copy(),
        component=built.component.copy(),
        kind=built.kind.copy(),
        wake_edges=built.wake_edges.copy(),
        half=False,
        tip=built.tip,
    )
    assert user_mesh.wake_cp_frac is None

    solver = PanelSolver(user_mesh, S_ref=8.0, b_ref=8.0, c_ref=1.0)
    wake = solver._full_wake
    assert wake.cp_frac is None

    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
    res = solver.solve(cond)
    gamma = res.mu[wake.wake_up] - res.mu[wake.wake_lo]

    cdi_old = _old_rule_trefftz_cdi(wake, gamma, float(cond.V_inf), float(cond.rho), solver.S_ref)
    rel_diff = abs(res.CDi_trefftz - cdi_old) / abs(cdi_old)
    assert rel_diff < 1.0e-3

    # Verify the fallback point is exact 0.5 by checking per-strip core at 0.5 directly.
    assert TREFFTZ_CORE_FRACTION == CORE_RADIUS_FRACTION
    pl = wake.p_left
    pr = wake.p_right
    seg_len = np.linalg.norm(pr - pl, axis=-1)
    assert np.min(seg_len) < np.max(seg_len)  # spacing="auto" has varying strip lengths


def test_symmetric_wing_zero_cn_and_mirrored_cp_frac() -> None:
    """Symmetric wing at beta=0 gives zero Cn_trefftz and mirrored cp_frac equals 1 - cp_frac."""
    sections = [
        WingSection(y_frac=0.0, chord=1.2, shape="0012"),
        WingSection(y_frac=1.0, chord=0.6, shape="0012"),
    ]
    surf = LiftingSurface(
        name="Wing",
        semi_span=4.0,
        sections=sections,
        sweep_le=np.radians(20.0),
        is_symmetric=True,
    )
    half = mesh_lifting_surface(surf, airfoil("0012").coords, n_span=16, n_chord=6, spacing="auto", half=True)
    full = reconstruct_full_mesh(half)
    assert full.wake_cp_frac is not None
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(full.panels), 1))
    wake = build_wake(full.nodes, full.panels, full.wake_edges, normals, cp_frac=full.wake_cp_frac)
    assert wake.cp_frac is not None

    n_half = len(half.wake_edges)
    np.testing.assert_allclose(wake.cp_frac[n_half:], 1.0 - wake.cp_frac[:n_half], atol=1.0e-14)

    cond = FlightCondition(V_inf=30.0, alpha=np.radians(5.0), beta=0.0)
    solver_half = PanelSolver(half, S_ref=7.2, b_ref=8.0, c_ref=0.9)
    res_half = solver_half.solve(cond)
    assert res_half.Cn_trefftz == 0.0

    solver_full = PanelSolver(full, S_ref=7.2, b_ref=8.0, c_ref=0.9)
    res_full = solver_full.solve(cond)
    assert abs(res_full.Cn_trefftz) < 1.0e-14

    # Direct Trefftz yaw evaluation with symmetric circulation must also cancel to round-off.
    cn_direct = compute_trefftz_yawing_moment(
        solver_full._full_wake,
        res_full.mu,
        cond,
        solver_full.S_ref,
        solver_full.b_ref,
    )
    assert abs(cn_direct) < 1.0e-14


def test_ground_effect_trefftz_rule() -> None:
    """Ground-effect Trefftz drag at h/b=0.2 uses cp_frac and per-strip core consistently."""
    semi_span = 4.0
    span = 2.0 * semi_span
    s_ref = span * 1.0
    v_inf = 1.0
    rho = 1.0
    height = 0.2 * span
    ground = (np.array([0.0, 0.0, 1.0]), -height)

    # Fine converged ground-effect reference with elliptic circulation at h/b = 0.2.
    wake_ref, gamma_ref = _elliptic_wake_and_gamma(160, semi_span=semi_span)
    factors_ref = _trefftz_factors(wake_ref, ground=ground)
    cdi_ref = float(
        _trefftz_batch(
            gamma_ref[None, :],
            np.array([v_inf]),
            np.array([rho]),
            s_ref,
            factors_ref,
            ground=True,
        )[0]
    )

    wake_40, gamma_40 = _elliptic_wake_and_gamma(20, semi_span=semi_span)
    factors_40 = _trefftz_factors(wake_40, ground=ground)
    cdi_batch = float(
        _trefftz_batch(
            gamma_40[None, :],
            np.array([v_inf]),
            np.array([rho]),
            s_ref,
            factors_40,
            ground=True,
        )[0]
    )
    cdi_direct = compute_trefftz_induced_drag(
        wake_40,
        np.zeros(0),
        v_inf,
        rho,
        s_ref,
        ground=ground,
        wake_gamma=gamma_40,
    )
    # _trefftz_factors + _trefftz_batch must match _trefftz_strip_wash ground branch.
    assert abs(cdi_batch - cdi_direct) / abs(cdi_direct) < 1.0e-13

    mid_40 = 0.5 * (wake_40.p_left + wake_40.p_right)
    gamma_mid_40 = np.sqrt(np.maximum(1.0 - (mid_40[:, 1] / semi_span) ** 2, 0.0))
    cdi_old = _old_rule_trefftz_cdi(wake_40, gamma_mid_40, v_inf, rho, s_ref, ground=ground)

    err_new = abs(cdi_direct - cdi_ref) / cdi_ref
    err_old = abs(cdi_old - cdi_ref) / cdi_ref
    assert err_new < 1.0e-3
    assert err_old > 5.0 * err_new


@pytest.mark.gpu
def test_gpu_trefftz_parity() -> None:
    """GPU backend matches CPU CDi_trefftz and Cn_trefftz in free air and sideslip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available")

    surf = _rect_wing(semi_span=4.0, chord=1.0)
    half = mesh_lifting_surface(surf, airfoil("0012").coords, n_span=12, n_chord=6, spacing="auto")
    full = reconstruct_full_mesh(half)
    cond = FlightCondition(V_inf=30.0, alpha=np.radians(4.0), beta=np.radians(3.0))

    res_cpu = PanelSolver(full, S_ref=8.0, b_ref=8.0, c_ref=1.0, backend="numpy").solve(cond)
    res_gpu = PanelSolver(
        full, S_ref=8.0, b_ref=8.0, c_ref=1.0, backend="gpu", dtype="float64"
    ).solve(cond)

    np.testing.assert_allclose(res_gpu.CDi_trefftz, res_cpu.CDi_trefftz, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(res_gpu.Cn_trefftz, res_cpu.Cn_trefftz, rtol=1.0e-10, atol=1.0e-12)
