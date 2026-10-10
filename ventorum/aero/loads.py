# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Forces and moments from a solved vortex lattice.

* Lift, side force and moments: Kutta-Joukowski force on every bound vortex,
  ``F = rho * Gamma * (V_inf + v) x l``, with ``v`` the velocity induced by
  all other vortices (and ground images) at the middle of the bound vortex.
* Induced drag: Trefftz plane far downstream (the wake is straight and
  parallel to the free stream). This is the reference value ``CDi``. The
  drag component of the surface forces is kept as ``CDi_nearfield``.
* Profile drag: section polar ``Cd(alpha_eff)`` on every strip.
* Section pitching moment: ``Cm(alpha_eff)`` about the quarter chord
  (``Cm0`` for a linear airfoil), added as a pure moment.

Reported moments use the standard aircraft convention (body axes x forward,
y right, z down): ``Cl > 0`` right wing down, ``Cm > 0`` nose up,
``Cn > 0`` nose right. The geometry axes are x aft, y right, z up, so
``Cl = -Mx/(q S b)``, ``Cm = My/(q S c)``, ``Cn = -Mz/(q S b)``.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge University
  Press, 2001: the Kutta-Joukowski force on a bound vortex and the induced
  drag in the Trefftz plane.
* M. Drela, "Flight Vehicle Aerodynamics", MIT Press, 2014: the Trefftz-plane induced drag and the near-field forces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.aero.system import (
    GroundPlane,
    UnknownMap,
    build_sources,
    freestream_direction,
    is_symmetric_condition,
    lift_direction,
    panel_targets,
)
from ventorum.aero.vortex import induced_velocity, trefftz_normalwash_prepared, trefftz_prepare
from ventorum.core.datatypes import (
    FlightCondition,
    IntegratedResult,
    LinearAirfoil,
    SpanwiseResult,
)
from ventorum.geometry.lattice import (
    CORE_RADIUS_FRACTION,
    VortexLattice,
    section_cd,
    section_cm,
)
from ventorum.utils.vec import cross3


@dataclass
class LoadsResult:
    """Everything the solvers need to build a :class:`SolverResult`."""

    totals: IntegratedResult
    spanwise: list[SpanwiseResult]
    strip_gamma: np.ndarray
    strip_force: np.ndarray
    alpha_eff: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


_cross = cross3  # shared helper (same formula as numpy.cross)


def group_strips_by_airfoil(airfoils: list) -> list[tuple[Any, np.ndarray]]:
    """Strip indices grouped by airfoil object (identity)."""
    groups: dict[int, tuple[Any, list[int]]] = {}
    for i, af in enumerate(airfoils):
        key = id(af)
        if key not in groups:
            groups[key] = (af, [])
        groups[key][1].append(i)
    return [(af, np.asarray(idx, dtype=int)) for af, idx in groups.values()]


def trefftz_induced_drag(
    lattice: VortexLattice,
    strip_gamma: np.ndarray,
    wake_dir: np.ndarray,
    rho: float,
    ground: GroundPlane | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Induced drag from the Trefftz plane.

    Parameters
    ----------
    lattice : VortexLattice
        The lattice.
    strip_gamma : (n_strips,)
        Circulation of each strip [m^2/s].
    wake_dir : (3,)
        Unit vector of the wake direction [-].
    rho : float
        Air density [kg/m^3].
    ground : GroundPlane or None
        Ground plane (image wake), or None in free air.

    Returns
    -------
    D_i : float
        Induced drag [N].
    w_n : (n_strips,)
        Normal wash at each strip in the Trefftz plane [m/s] (negative is
        downwash for a lifting strip).
    seg_len : (n_strips,)
        Length of each strip projected on the Trefftz plane [m].
    """
    D_i, w_n, seg_len = trefftz_induced_drag_batch(
        lattice, np.asarray(strip_gamma, dtype=float)[None, :], np.asarray(wake_dir, dtype=float)[None, :],
        np.array([float(rho)]), [ground])
    return float(D_i[0]), w_n[0], seg_len[0]


_FORCE_FULL_TREFFTZ = False


def trefftz_induced_drag_batch(
    lattice: VortexLattice,
    strip_gammas: np.ndarray,
    wake_dirs: np.ndarray,
    rho: np.ndarray,
    grounds: list[GroundPlane | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Induced drag from the Trefftz plane for K cases on the same lattice.

    Each case gets the same result, to the last bit, as
    :func:`trefftz_induced_drag` with its own inputs.

    Parameters
    ----------
    strip_gammas : (K, n_strips)
        Circulation of every strip [m^2/s].
    wake_dirs : (K, 3)
        Wake direction of each case (the normal of its Trefftz plane).
    rho : (K,)
        Air density [kg/m^3].
    grounds : list of GroundPlane or None
        Ground plane of each case.

    Returns
    -------
    D_i : (K,)
        Induced drag [N].
    w_n : (K, n_strips)
        Normal wash in the Trefftz plane [m/s].
    seg_len : (K, n_strips)
        Length of each strip projected on the Trefftz plane [m].
    """
    K, n_s = strip_gammas.shape
    te_l, te_r = lattice.te_left, lattice.te_right
    d = np.empty((K, 3))
    for k in range(K):
        dk = np.asarray(wake_dirs[k], dtype=float)
        d[k] = dk / np.linalg.norm(dk)
    # Projection on the plane: p - (p . d) d.
    dot_l = np.empty((K, n_s))
    dot_r = np.empty((K, n_s))
    for k in range(K):
        dot_l[k] = te_l @ d[k]
        dot_r[k] = te_r @ d[k]
    pl = te_l - dot_l[:, :, None] * d[:, None, :]
    pr = te_r - dot_r[:, :, None] * d[:, None, :]
    seg = pr - pl
    seg_len = np.linalg.norm(seg, axis=-1)
    # Evaluation point at the same span fraction as the control point.
    q = pl + lattice.cp_frac[:, None] * seg
    qn = _cross(d[:, None, :], seg)
    qn /= np.maximum(np.linalg.norm(qn, axis=-1, keepdims=True), 1e-300)
    w_n = np.empty((K, n_s))
    can_fold = lattice.can_fold_symmetry()
    right = np.flatnonzero(lattice.strip_is_right)
    left = np.flatnonzero(~lattice.strip_is_right)
    mirror = lattice.strip_mirror
    has_half = bool(can_fold and right.size > 0 and left.size > 0)
    pos = np.full(n_s, -1, dtype=np.int64)
    if has_half:
        pos[right] = np.arange(right.size)

    for k in range(K):
        sg = strip_gammas[k]
        p_vort = np.vstack([pr[k], pl[k]])
        g_vort = np.concatenate([sg, -sg])
        ground = grounds[k]
        if ground is not None:
            pts = ground.reflect_points(np.vstack([te_r, te_l]))
            p_img = pts - (pts @ d[k])[:, None] * d[k]
            p_vort = np.vstack([p_vort, p_img])
            g_vort = np.concatenate([g_vort, -g_vort])

        use_half = False
        if has_half and not _FORCE_FULL_TREFFTZ:
            case_is_sym = (wake_dirs[k][1] == 0.0) and (ground is None or abs(ground.normal[1]) <= 1e-12)
            if case_is_sym:
                sg_max = max(float(np.max(np.abs(sg))), 1e-300)
                if _close(sg[left], sg[mirror[left]], 1e-12 * sg_max):
                    use_half = True

        if use_half:
            prep_r = _trefftz_core_data(lattice, ground is not None, right_only=True)
            wn_r = trefftz_normalwash_prepared(q[k][right], qn[k][right], p_vort, g_vort, d[k], prep_r)
            w_n[k, right] = wn_r
            w_n[k, left] = wn_r[pos[mirror[left]]]
        else:
            w_n[k] = trefftz_normalwash_prepared(q[k], qn[k], p_vort, g_vort, d[k],
                                                 _trefftz_core_data(lattice, ground is not None))
    D_i = -0.5 * rho * np.sum(strip_gammas * w_n * seg_len, axis=1)
    return D_i, w_n, seg_len


def _trefftz_core_data(lattice: VortexLattice, with_images: bool, right_only: bool = False) -> tuple:
    """Core data of the Trefftz-plane vortices (lattice-only, cached on the lattice).

    The vortices are the right and left trailing-edge points of every strip
    (and their ground images when *with_images*). Cross-surface core: the
    same rule and size as the near field (panel_targets), so that a wake
    that passes near another surface gives the same regularised velocity in
    both places. One panel of each strip gives the core group and the core
    radius of that strip.
    """
    key = ("trefftz_core_data", with_images, right_only)
    prep = lattice.geom_cache.get(key)
    if prep is not None:
        return prep
    base_key = ("trefftz_core_data", with_images)
    base_prep = lattice.geom_cache.get(base_key)
    if base_prep is None:
        rc = CORE_RADIUS_FRACTION * lattice.width
        first_panel = np.empty(lattice.n_strips, dtype=np.int64)
        first_panel[lattice.panel_strip[::-1]] = np.arange(lattice.n_panels)[::-1]
        targets = panel_targets(lattice, first_panel)
        copies = 4 if with_images else 2
        rc_vort = np.concatenate([rc] * copies)
        g_src = np.concatenate([targets.group] * copies)
        base_prep = trefftz_prepare(lattice.n_strips, rc_vort.size, rc_vort, group=g_src, targets=targets)
        lattice.geom_cache[base_key] = base_prep
        lattice.geom_cache[("trefftz_core_data", with_images, False)] = base_prep
    if not right_only:
        return base_prep
    rc2_arr, src_group, tg_group, tg_rc2, use_tg = base_prep
    right = np.flatnonzero(lattice.strip_is_right)
    prep = (
        rc2_arr,
        src_group,
        np.ascontiguousarray(tg_group[right]),
        np.ascontiguousarray(tg_rc2[right]),
        use_tg,
    )
    lattice.geom_cache[key] = prep
    return prep


_MIRROR = np.array([1.0, -1.0, 1.0])


def _close(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    """Return True if ``|a - b| <= atol`` everywhere (``np.allclose`` with rtol 0, finite data).

    A NaN gives False, as in ``np.allclose``. It avoids the temporary arrays of
    ``np.isclose``, which cost more than the comparison for small arrays.
    """
    return bool(np.all(np.abs(a - b) <= atol))


def _symmetric_fold(
    lattice: VortexLattice,
    gamma_panel: np.ndarray,
    points: list[np.ndarray],
    swap: list[int],
    leg_forces: bool | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the panels to evaluate and the mirror map, or None if the case is not symmetric.

    *points* holds arrays of evaluation points per panel (for example the
    force points and the two leg mid-points). On a symmetric case the point
    set ``points[k]`` of a left panel is the mirror image of the point set
    ``points[swap[k]]`` of its mirror panel. The function checks this, and
    that the circulation is symmetric, before it allows the fold.
    """
    cache = lattice.geom_cache
    has_legs = bool(leg_forces if leg_forces is not None else (len(points) > 1))
    key = ("symmetric_fold_geometry", has_legs)
    geom = cache.get(key)

    if geom is None:
        if not lattice.can_fold_symmetry():
            cache[key] = False
            return None
        mirror = lattice.panel_mirror
        left = np.flatnonzero(~lattice.strip_is_right[lattice.panel_strip])
        if left.size == 0:
            cache[key] = False
            return None
        scale = max(float(np.max(np.abs(lattice.all_points()))), 1.0)
        for k, pk in enumerate(points):
            if not _close(pk[left], points[swap[k]][mirror[left]] * _MIRROR, 1e-12 * scale):
                cache[key] = False
                return None
        right = np.flatnonzero(lattice.strip_is_right[lattice.panel_strip])
        geom = (right, left)
        cache[key] = geom
    elif geom is False:
        return None

    right, left = geom
    mirror = lattice.panel_mirror
    g = np.asarray(gamma_panel, dtype=float)
    gmax = max(float(np.max(np.abs(g))), 1e-300)
    if not _close(g[left], g[mirror[left]], 1e-12 * gmax):
        return None
    return right, left


def compute_loads(
    lattice: VortexLattice,
    gamma_panel: np.ndarray,
    condition: FlightCondition,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ground: GroundPlane | None = None,
    ref_point: np.ndarray | None = None,
    alpha_eff_strip: np.ndarray | None = None,
    wake_dir: np.ndarray | None = None,
    leg_forces: bool | None = None,
    v_control: np.ndarray | None = None,
) -> LoadsResult:
    """Integrate forces and moments of a solved lattice.

    Parameters
    ----------
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m] (rolling and yawing moments).
    c_ref : float
        Reference chord [m] (pitching moment).
    ref_point : (3,) or None
        Moment reference point [m]. Default: the origin.
    gamma_panel : (n_panels,)
        Circulation of every panel horseshoe [m^2/s].
    alpha_eff_strip : (n_strips,) or None
        Effective section angle of attack from the solver. If *None*, it is
        found from the strip lift: ``alpha_L0 + Cl / a0``.
    leg_forces : bool or None
        Apply the Kutta-Joukowski force also to the trailing legs that lie on
        the surface (from the bound vortex to the trailing edge). These legs
        are bound vorticity of the lifting surface; their force is not zero
        in sideslip, on non-planar surfaces and near the ground. Default:
        True for the vortex-lattice method, False for the lifting line (whose
        model has no chordwise extent; forces on the bound vortex only, as in
        Phillips and Snyder).
    v_control : (n_panels, 3) or None
        Induced velocity at the control points [m/s] from the solver. On a
        lifting-line lattice without leg forces the force points are the
        control points, so this replaces the kernel evaluation of the loads.

    Notes
    -----
    This is :func:`compute_loads_batch` with one case.
    """
    return compute_loads_batch(
        lattice, np.asarray(gamma_panel, dtype=float)[None, :], [condition], S_ref, b_ref, c_ref,
        grounds=[ground], ref_point=ref_point,
        alpha_eff_strips=None if alpha_eff_strip is None else np.asarray(alpha_eff_strip, dtype=float)[None, :],
        wake_dirs=None if wake_dir is None else np.asarray(wake_dir, dtype=float)[None, :],
        leg_forces=leg_forces,
        v_controls=None if v_control is None else np.asarray(v_control, dtype=float)[None, :, :],
    )[0]


def compute_loads_batch(
    lattice: VortexLattice,
    gamma_panels: np.ndarray,
    conditions: list[FlightCondition],
    S_ref: float,
    b_ref: float,
    c_ref: float,
    grounds: list[GroundPlane | None] | None = None,
    ref_point: np.ndarray | None = None,
    alpha_eff_strips: np.ndarray | None = None,
    wake_dirs: np.ndarray | None = None,
    leg_forces: bool | None = None,
    v_controls: np.ndarray | None = None,
) -> list[LoadsResult]:
    """Integrate forces and moments of K solved cases on the same lattice.

    The cases (for example the angles of a sweep) are computed together:
    each array operation acts on all cases at once. Each case gets the same
    result, to the last bit, as :func:`compute_loads` with its own inputs
    (the operations on one case are the same, in the same order).

    Parameters
    ----------
    lattice : VortexLattice
        The lattice of all cases.
    gamma_panels : (K, n_panels)
        Circulation of every panel horseshoe [m^2/s], one row per case.
    conditions : list of FlightCondition
        Flight condition of each case (K items).
    S_ref, b_ref, c_ref : float
        Reference area [m^2], span [m] and chord [m].
    grounds : list of GroundPlane or None, optional
        Ground plane of each case (None items for free air). None means
        free air for all cases.
    ref_point : numpy.ndarray or None, optional
        Moment reference point [m], shape (3,). None is the origin.
    alpha_eff_strips : (K, n_strips) or None
        Effective section angle of attack [rad] from the solver, or None
        (see :func:`compute_loads`).
    wake_dirs : (K, 3) or None
        Unit wake direction of each case. None is the free-stream direction.
    leg_forces : bool or None
        See :func:`compute_loads`.
    v_controls : (K, n_panels, 3) or None
        Induced velocity at the control points [m/s] from the solver (see
        ``v_control`` in :func:`compute_loads`).

    Returns
    -------
    list of LoadsResult
        One result per case.
    """
    K = len(conditions)
    G = np.asarray(gamma_panels, dtype=float).reshape(K, -1)
    grounds = [None] * K if grounds is None else list(grounds)
    V = np.array([float(c.V_inf) for c in conditions])
    rho = np.array([float(c.rho) for c in conditions])
    q_inf = 0.5 * rho * V * V
    D = np.array([freestream_direction(c.alpha, c.beta) for c in conditions]).reshape(K, 3)
    WD = D if wake_dirs is None else np.asarray(wake_dirs, dtype=float).reshape(K, 3)
    rp = np.zeros(3) if ref_point is None else np.asarray(ref_point, dtype=float)
    n_s, n_c = lattice.n_strips, lattice.n_chord
    strip_gamma = G.reshape(K, n_s, n_c).sum(axis=2)

    # --- near-field Kutta-Joukowski forces ------------------------------------
    mid = lattice.force_points
    n_p = lattice.n_panels
    if leg_forces is None:
        leg_forces = lattice.collocation == "vlm"
    use_leg = bool(leg_forces)
    geom_key = ("loads_geom", use_leg)
    cached_geom = lattice.geom_cache.get(geom_key)
    if cached_geom is None:
        points = [mid]
        swap = [0]
        if use_leg:
            leg_a_mid = 0.5 * (lattice.a_te + lattice.a)
            leg_b_mid = 0.5 * (lattice.b + lattice.b_te)
            points += [leg_a_mid, leg_b_mid]
            swap += [2, 1]  # the a-leg of a left panel mirrors the b-leg of its mirror panel
            leg_start = np.vstack([lattice.a_te, lattice.b])
            leg_end = np.vstack([lattice.a, lattice.b_te])
            leg_mid = np.vstack([points[1], points[2]])
            leg_vec = leg_end - leg_start
        else:
            leg_mid = leg_vec = None
        l_vec = lattice.b - lattice.a
        cached_geom = (points, swap, l_vec, leg_mid, leg_vec)
        lattice.geom_cache[geom_key] = cached_geom
    points, swap, l_vec, leg_mid, leg_vec = cached_geom

    reuse = v_controls is not None and not use_leg and lattice.collocation == "llt"
    if reuse:
        v_all = [np.asarray(v_controls, dtype=float).reshape(K, n_p, 3)]
    else:
        v_all = [np.empty((K, n_p, 3)) for _ in points]
        for k in range(K):
            _near_field_velocity(lattice, G[k], conditions[k], grounds[k], WD[k], points, swap,
                                 use_leg, [vk[k] for vk in v_all])
    v_ind = v_all[0]
    Vd = V[:, None, None] * D[:, None, :]
    F_panel = rho[:, None, None] * G[:, :, None] * _cross(Vd + v_ind, l_vec)
    M_geo = np.sum(_cross(mid - rp, F_panel), axis=1)
    if use_leg:
        v_leg = np.concatenate([v_all[1], v_all[2]], axis=1)
        g_leg = np.concatenate([G, G], axis=1)
        F_leg = rho[:, None, None] * g_leg[:, :, None] * _cross(Vd + v_leg, leg_vec)
        M_geo += np.sum(_cross(leg_mid - rp, F_leg), axis=1)
        F_panel = F_panel + F_leg[:, :n_p] + F_leg[:, n_p:]
    F_strip = F_panel.reshape(K, n_s, n_c, 3).sum(axis=2)
    F_near = F_panel.sum(axis=1)

    # --- section quantities ---------------------------------------------------
    chord = lattice.chord
    width = lattice.width
    Cl = 2.0 * strip_gamma / (V[:, None] * chord)
    alpha_geom = np.empty((K, n_s))
    for k in range(K):
        alpha_geom[k] = np.arctan2(lattice.normal @ D[k], lattice.chord_dir @ D[k])
    if alpha_eff_strips is None:
        alpha_eff = lattice.alpha_L0 + Cl / np.maximum(lattice.a0, 1e-9)
    else:
        alpha_eff = np.asarray(alpha_eff_strips, dtype=float).reshape(K, n_s)
    alpha_i = alpha_geom - alpha_eff

    lin_afs, lin_strips, lin_index, other_groups = _airfoil_split(lattice)
    Cd_p = np.zeros((K, n_s))
    Cm_s = np.zeros((K, n_s))
    if lin_afs:
        # Linear airfoils: constant coefficients, written for all their strips at once.
        # The values are read from the airfoil objects on every call.
        Cd_p[:, lin_strips] = np.array([af.Cd0 for af in lin_afs], dtype=float)[lin_index]
        Cm_s[:, lin_strips] = np.array([af.Cm0 for af in lin_afs], dtype=float)[lin_index]
    for af, idx in other_groups:
        a_idx = alpha_eff[:, idx].ravel()
        Cd_p[:, idx] = section_cd(af, a_idx).reshape(K, idx.size)
        Cm_s[:, idx] = section_cm(af, a_idx).reshape(K, idx.size)
    has_profile = np.any(Cd_p != 0.0, axis=1)

    # Profile drag along the free stream, applied at the strip quarter chord.
    D_p = q_inf[:, None] * chord * width * Cd_p
    F_prof = D_p[:, :, None] * D[:, None, :]
    M_geo += np.sum(_cross(lattice.qc_mid - rp, F_prof), axis=1)

    # Section pitching moment about the axis n_lift x d (nose towards the lift side).
    n_lift = _cross(D[:, None, :], lattice.dl)
    n_lift /= np.maximum(np.linalg.norm(n_lift, axis=-1, keepdims=True), 1e-300)
    axis = _cross(n_lift, D[:, None, :])
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-300)
    M_geo += np.sum((q_inf[:, None] * chord ** 2 * width * Cm_s)[:, :, None] * axis, axis=1)

    # --- Trefftz plane ----------------------------------------------------------
    D_i, w_n, seg_len = trefftz_induced_drag_batch(lattice, strip_gamma, WD, rho, grounds)
    Cd_i = -strip_gamma * w_n * seg_len / (V[:, None] * V[:, None] * chord * width)

    # --- totals and spanwise output per case ------------------------------------
    F_prof_sum = F_prof.sum(axis=1)
    L_dir = np.array([lift_direction(c.alpha) for c in conditions]).reshape(K, 3)
    S_dir = _cross(L_dir, D)   # side_direction of each case, row by row
    AR = b_ref ** 2 / S_ref
    out: list[LoadsResult] = []
    for k in range(K):
        qS = float(q_inf[k]) * S_ref
        F_total = F_near[k] + F_prof_sum[k]
        CL = float(F_total @ L_dir[k]) / qS
        CY = float(F_total @ S_dir[k]) / qS
        CDi = float(D_i[k]) / qS
        CDi_near = float(F_near[k] @ D[k]) / qS
        hp = bool(has_profile[k])
        CDp = float(D_p[k].sum()) / qS if hp else None
        CD_total = CDi + (CDp if CDp is not None else 0.0)
        M = M_geo[k]
        e = CL ** 2 / (np.pi * AR * CDi) if CDi > 1e-12 else float("nan")
        sg = strip_gamma[k]
        rVk = float(rho[k]) * float(V[k])
        out.append(loads_result(
            lattice, CL=CL, CDi=CDi, CDp=CDp, CD_total=CD_total if hp else None, e=e, AR=AR,
            Cl=-float(M[0]) / (qS * b_ref), Cm=float(M[1]) / (qS * c_ref), Cn=-float(M[2]) / (qS * b_ref),
            CY=CY, CDi_nearfield=CDi_near, strip_gamma=sg, Cl_strip=Cl[k], Cd_i=Cd_i[k],
            Cd_profile=Cd_p[k] if hp else None, alpha_eff=alpha_eff[k], alpha_i=alpha_i[k],
            local_lift_factor=rVk, Cm_section=Cm_s[k], strip_force=F_strip[k], trefftz_normalwash=w_n[k],
            force_total=F_total, moment_total=M,
        ))
    return out


def loads_result(
    lattice: VortexLattice,
    *,
    CL: float,
    CDi: float,
    CDp: float | None,
    CD_total: float | None,
    e: float,
    AR: float,
    Cl: float,
    Cm: float,
    Cn: float,
    CY: float,
    CDi_nearfield: float,
    strip_gamma: np.ndarray,
    Cl_strip: np.ndarray,
    Cd_i: np.ndarray,
    Cd_profile: np.ndarray | None,
    alpha_eff: np.ndarray,
    alpha_i: np.ndarray,
    local_lift_factor: float,
    Cm_section: np.ndarray,
    strip_force: np.ndarray,
    trefftz_normalwash: np.ndarray,
    force_total: np.ndarray,
    moment_total: np.ndarray,
) -> LoadsResult:
    """Return the :class:`LoadsResult` of one case from its integrated values and its strip arrays.

    The CPU loads (:func:`compute_loads_batch`) and the GPU pipelines
    (:mod:`ventorum.gpu`) make their result objects here. The strip arrays
    (shape (n_strips,)) are split per surface; ``local_lift_factor`` is
    ``rho * V_inf`` [kg/(m^2 s)], so the local lift is ``rho V Gamma`` [N/m].
    """
    totals = IntegratedResult(
        CL=CL,
        CDi=CDi,
        CDp=CDp,
        CD_total=CD_total,
        e=e,
        AR=AR,
        Cl=Cl,
        Cm=Cm,
        Cn=Cn,
        CY=CY,
        CDi_nearfield=CDi_nearfield,
    )
    chord = lattice.chord
    spanwise: list[SpanwiseResult] = []
    for surf in lattice.surfaces:
        sl = surf.strips
        spanwise.append(SpanwiseResult(
            y=lattice.qc_mid[sl, 1].copy(),
            gamma=strip_gamma[sl].copy(),
            Cl=Cl_strip[sl].copy(),
            Cd_i=Cd_i[sl].copy(),
            Cd_profile=Cd_profile[sl].copy() if Cd_profile is not None else None,
            alpha_eff=alpha_eff[sl].copy(),
            alpha_i=alpha_i[sl].copy(),
            local_lift=local_lift_factor * strip_gamma[sl],
            surface_name=surf.name,
            chord=chord[sl].copy(),
            Cm_section=Cm_section[sl].copy(),
        ))
    return LoadsResult(
        totals=totals,
        spanwise=spanwise,
        strip_gamma=strip_gamma,
        strip_force=strip_force,
        alpha_eff=alpha_eff,
        extras={
            "trefftz_normalwash": trefftz_normalwash,
            "force_total_body": force_total,
            "moment_total_geometry_axes": moment_total,
        },
    )


def _load_points(lattice: VortexLattice, eval_panels: np.ndarray, points: list[np.ndarray]) -> tuple:
    """Return the evaluation points of the loads, their targets, and the repeats among the points.

    Returns ``(P, tgt, repeat)``. *P* stacks the point sets set by set. With
    one core group (one surface, or surfaces joined into one group) the
    target core radius never acts (it acts only between different groups),
    so the velocity at a point depends only on the point. A point that occurs
    more than once (the leg mid-point of a panel is the leg mid-point of its
    neighbour) is then evaluated once: *repeat* = ``(keep, inverse)`` with
    ``P[keep][inverse]`` equal to *P* (see :func:`induced_velocity`). With
    more than one group, or no repeated point, *repeat* is None.
    """
    P = np.vstack([pk[eval_panels] for pk in points])
    tgt = panel_targets(lattice, np.tile(eval_panels, len(points)))
    if np.unique(tgt.group).size > 1 or np.unique(_core_group_of(lattice)).size > 1:
        return P, tgt, None
    P_u, first, inverse = np.unique(P, axis=0, return_index=True, return_inverse=True)
    if P_u.shape[0] == P.shape[0]:
        return P, tgt, None
    order = np.argsort(first)   # keep the first-occurrence order of the points
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return P, tgt, (first[order], rank[inverse.ravel()])


def _core_group_of(lattice: VortexLattice) -> np.ndarray:
    """Core group of every strip (the surface index when the lattice has no joined groups)."""
    g = lattice.strip_core_group
    return lattice.strip_surface if g is None else g


def _airfoil_split(lattice: VortexLattice) -> tuple:
    """Return the strips grouped by airfoil: the linear airfoils together, the others per airfoil.

    Returns ``(lin_afs, lin_strips, lin_index, other_groups)``: the linear
    airfoil objects, their strips, the index into *lin_afs* of each of those
    strips, and ``(airfoil, strips)`` for every other airfoil. Lattice-only,
    cached on the lattice.
    """
    split = lattice.geom_cache.get("airfoil_split")
    if split is None:
        groups = lattice.geom_cache.get("airfoil_groups")
        if groups is None:
            groups = group_strips_by_airfoil(lattice.airfoils)
            lattice.geom_cache["airfoil_groups"] = groups
        lin = [(af, idx) for af, idx in groups if isinstance(af, LinearAirfoil)]
        other = [(af, idx) for af, idx in groups if not isinstance(af, LinearAirfoil)]
        lin_afs = [af for af, _ in lin]
        lin_strips = np.concatenate([idx for _, idx in lin]) if lin else np.empty(0, dtype=int)
        lin_index = np.concatenate([np.full(idx.size, i) for i, (_, idx) in enumerate(lin)]) if lin             else np.empty(0, dtype=int)
        split = (lin_afs, lin_strips, lin_index, other)
        lattice.geom_cache["airfoil_split"] = split
    return split


def _near_field_velocity(
    lattice: VortexLattice,
    gamma_panel: np.ndarray,
    condition: FlightCondition,
    ground: GroundPlane | None,
    wake_dir: np.ndarray,
    points: list[np.ndarray],
    swap: list[int],
    leg_forces: bool,
    out: list[np.ndarray],
) -> None:
    """Write the induced velocity at each point set of one case into *out* (one kernel evaluation).

    On a symmetric case only the right panels are evaluated; the left panels
    get the mirror image of their mirror panel.
    """
    n_p = lattice.n_panels
    if not is_symmetric_condition(condition, ground):
        fold = None
    else:
        fold = _symmetric_fold(lattice, gamma_panel, points, swap, leg_forces=leg_forces)
    eval_panels = np.arange(n_p) if fold is None else fold[0]
    n_e = eval_panels.size
    full_map = lattice.geom_cache.get("loads_full_map")
    if full_map is None:
        full_map = UnknownMap(
            unknown_panels=np.arange(n_p),
            panel_column=np.arange(n_p),
            symmetric=False,
        )
        lattice.geom_cache["loads_full_map"] = full_map
    src = build_sources(lattice, wake_dir, full_map, ground)
    p_tgt_key = ("loads_points", fold is not None, bool(leg_forces))
    p_tgt = lattice.geom_cache.get(p_tgt_key)
    if p_tgt is None:
        p_tgt = _load_points(lattice, eval_panels, points)
        lattice.geom_cache[p_tgt_key] = p_tgt
    P, tgt, repeat = p_tgt
    cache = lattice.kernel_cache if ground is None else None
    v_eval = induced_velocity(P, src, gamma_panel, tgt, cache=cache,
                              cache_key=("loads", fold is not None, bool(leg_forces)), repeat=repeat)
    for k in range(len(points)):
        out[k][eval_panels] = v_eval[k * n_e:(k + 1) * n_e]
    if fold is not None:
        left = fold[1]
        mirror = lattice.panel_mirror[left]
        for k in range(len(points)):
            out[k][left] = out[swap[k]][mirror] * _MIRROR
