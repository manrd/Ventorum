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
    side_direction,
)
from ventorum.aero.vortex import induced_velocity, trefftz_normalwash
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
    d = np.asarray(wake_dir, dtype=float)
    d = d / np.linalg.norm(d)

    def project(p: np.ndarray) -> np.ndarray:
        return p - (p @ d)[:, None] * d

    pl = project(lattice.te_left)
    pr = project(lattice.te_right)
    seg = pr - pl
    seg_len = np.linalg.norm(seg, axis=1)
    # Evaluation point at the same span fraction as the control point.
    q = pl + lattice.cp_frac[:, None] * seg
    qn = _cross(d, seg)
    qn /= np.maximum(np.linalg.norm(qn, axis=1, keepdims=True), 1e-300)

    rc = CORE_RADIUS_FRACTION * lattice.width
    p_vort = np.vstack([pr, pl])
    g_vort = np.concatenate([strip_gamma, -strip_gamma])
    rc_vort = np.concatenate([rc, rc])
    # Cross-surface core: the same rule and size as the near field
    # (panel_targets), so that a wake that passes near another surface gives
    # the same regularised velocity in both places. One panel of each strip
    # gives the core group and the core radius of that strip.
    if "trefftz_strip_core" in lattice.geom_cache:
        targets, g_src = lattice.geom_cache["trefftz_strip_core"]
    else:
        first_panel = np.empty(lattice.n_strips, dtype=np.int64)
        first_panel[lattice.panel_strip[::-1]] = np.arange(lattice.n_panels)[::-1]
        targets = panel_targets(lattice, first_panel)
        g_src = np.concatenate([targets.group, targets.group])
        lattice.geom_cache["trefftz_strip_core"] = (targets, g_src)
    if ground is not None:
        p_img = project(ground.reflect_points(np.vstack([lattice.te_right, lattice.te_left])))
        p_vort = np.vstack([p_vort, p_img])
        g_vort = np.concatenate([g_vort, -g_vort])
        rc_vort = np.concatenate([rc_vort, rc_vort])
        g_src = np.concatenate([g_src, g_src])

    w_n = trefftz_normalwash(q, qn, p_vort, g_vort, d, rc_vort, group=g_src, targets=targets)
    D_i = -0.5 * rho * float(np.sum(strip_gamma * w_n * seg_len))
    return D_i, w_n, seg_len


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
    """
    V = float(condition.V_inf)
    rho = float(condition.rho)
    q_inf = 0.5 * rho * V * V
    d = freestream_direction(condition.alpha, condition.beta)
    wd = d if wake_dir is None else np.asarray(wake_dir, dtype=float)
    rp = np.zeros(3) if ref_point is None else np.asarray(ref_point, dtype=float)
    n_s, n_c = lattice.n_strips, lattice.n_chord

    gamma_panel = np.asarray(gamma_panel, dtype=float)
    strip_gamma = gamma_panel.reshape(n_s, n_c).sum(axis=1)

    # --- near-field Kutta-Joukowski forces ------------------------------------
    mid = lattice.force_points
    n_p = lattice.n_panels
    if leg_forces is None:
        leg_forces = lattice.collocation == "vlm"
    # Evaluation points per panel: the force point and, with leg forces, the
    # mid-points of the legs a_te -> a and b -> b_te (they carry the panel
    # circulation in the same sense as in the horseshoe kernel).
    points = [mid]
    swap = [0]
    if leg_forces:
        leg_a_mid = 0.5 * (lattice.a_te + lattice.a)
        leg_b_mid = 0.5 * (lattice.b + lattice.b_te)
        points += [leg_a_mid, leg_b_mid]
        swap += [2, 1]   # the a-leg of a left panel mirrors the b-leg of its mirror panel
    reuse = v_control is not None and not leg_forces and lattice.collocation == "llt"
    if reuse or not is_symmetric_condition(condition, ground):
        fold = None
    else:
        fold = _symmetric_fold(lattice, gamma_panel, points, swap, leg_forces=leg_forces)
    if fold is None:
        eval_panels = np.arange(n_p)
    else:
        eval_panels = fold[0]
    n_e = eval_panels.size
    if reuse:
        v_all = [np.asarray(v_control, dtype=float)]
    else:
        full_map = UnknownMap(
            unknown_panels=np.arange(n_p),
            panel_column=np.arange(n_p),
            symmetric=False,
        )
        src = build_sources(lattice, wd, full_map, ground)
        P = np.vstack([pk[eval_panels] for pk in points])
        tgt = panel_targets(lattice, np.tile(eval_panels, len(points)))
        cache = lattice.kernel_cache if ground is None else None
        v_eval = induced_velocity(P, src, gamma_panel, tgt, cache=cache,
                                  cache_key=("loads", fold is not None, bool(leg_forces)))
        v_all = []
        for k in range(len(points)):
            vk = np.empty((n_p, 3))
            vk[eval_panels] = v_eval[k * n_e:(k + 1) * n_e]
            v_all.append(vk)
    if fold is not None:
        left = fold[1]
        mirror = lattice.panel_mirror[left]
        for k in range(len(points)):
            v_all[k][left] = v_all[swap[k]][mirror] * _MIRROR
    v_ind = v_all[0]
    l_vec = lattice.b - lattice.a
    F_panel = rho * gamma_panel[:, None] * _cross(V * d[None, :] + v_ind, l_vec)
    M_geo = np.sum(_cross(mid - rp, F_panel), axis=0)
    if leg_forces:
        leg_start = np.vstack([lattice.a_te, lattice.b])
        leg_end = np.vstack([lattice.a, lattice.b_te])
        leg_mid = np.vstack([points[1], points[2]])
        v_leg = np.vstack([v_all[1], v_all[2]])
        g_leg = np.concatenate([gamma_panel, gamma_panel])
        F_leg = rho * g_leg[:, None] * _cross(V * d[None, :] + v_leg, leg_end - leg_start)
        M_geo += np.sum(_cross(leg_mid - rp, F_leg), axis=0)
        F_panel = F_panel + F_leg[:n_p] + F_leg[n_p:]
    F_strip = F_panel.reshape(n_s, n_c, 3).sum(axis=1)
    F_near = F_panel.sum(axis=0)

    # --- section quantities ---------------------------------------------------
    chord = lattice.chord
    width = lattice.width
    Cl = 2.0 * strip_gamma / (V * chord)
    alpha_geom = np.arctan2(lattice.normal @ d, lattice.chord_dir @ d)
    if alpha_eff_strip is None:
        alpha_eff = lattice.alpha_L0 + Cl / np.maximum(lattice.a0, 1e-9)
    else:
        alpha_eff = np.asarray(alpha_eff_strip, dtype=float)
    alpha_i = alpha_geom - alpha_eff

    groups = lattice.geom_cache.get("airfoil_groups")
    if groups is None:
        groups = group_strips_by_airfoil(lattice.airfoils)
        lattice.geom_cache["airfoil_groups"] = groups
    Cd_p = np.zeros(n_s)
    Cm_s = np.zeros(n_s)
    for af, idx in groups:
        if isinstance(af, LinearAirfoil):
            Cd_p[idx] = af.Cd0
            Cm_s[idx] = af.Cm0
        else:
            Cd_p[idx] = section_cd(af, alpha_eff[idx])
            Cm_s[idx] = section_cm(af, alpha_eff[idx])
    has_profile = bool(np.any(Cd_p != 0.0))

    # Profile drag along the free stream, applied at the strip quarter chord.
    D_p = q_inf * chord * width * Cd_p
    F_prof = D_p[:, None] * d[None, :]
    M_geo += np.sum(_cross(lattice.qc_mid - rp, F_prof), axis=0)

    # Section pitching moment about the axis n_lift x d (nose towards the lift side).
    n_lift = _cross(d[None, :], lattice.dl)
    n_lift /= np.maximum(np.linalg.norm(n_lift, axis=1, keepdims=True), 1e-300)
    axis = _cross(n_lift, d[None, :])
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-300)
    M_geo += np.sum((q_inf * chord ** 2 * width * Cm_s)[:, None] * axis, axis=0)

    # --- Trefftz plane ----------------------------------------------------------
    D_i, w_n, seg_len = trefftz_induced_drag(lattice, strip_gamma, wd, rho, ground)
    Cd_i = -strip_gamma * w_n * seg_len / (V * V * chord * width)

    # --- totals -------------------------------------------------------------------
    qS = q_inf * S_ref
    F_total = F_near + F_prof.sum(axis=0)
    CL = float(F_total @ lift_direction(condition.alpha)) / qS
    CY = float(F_total @ side_direction(condition.alpha, condition.beta)) / qS
    CDi = D_i / qS
    CDi_near = float(F_near @ d) / qS
    CDp = float(D_p.sum()) / qS if has_profile else None
    CD_total = CDi + (CDp if CDp is not None else 0.0)
    Cl_roll = -float(M_geo[0]) / (qS * b_ref)
    Cm = float(M_geo[1]) / (qS * c_ref)
    Cn = -float(M_geo[2]) / (qS * b_ref)
    AR = b_ref ** 2 / S_ref
    e = CL ** 2 / (np.pi * AR * CDi) if CDi > 1e-12 else float("nan")

    totals = IntegratedResult(
        CL=CL,
        CDi=CDi,
        CDp=CDp,
        CD_total=CD_total if has_profile else None,
        e=e,
        AR=AR,
        Cl=Cl_roll,
        Cm=Cm,
        Cn=Cn,
        CY=CY,
        CDi_nearfield=CDi_near,
    )

    # --- per-surface spanwise output -----------------------------------------------
    spanwise: list[SpanwiseResult] = []
    for surf in lattice.surfaces:
        sl = surf.strips
        spanwise.append(SpanwiseResult(
            y=lattice.qc_mid[sl, 1].copy(),
            gamma=strip_gamma[sl].copy(),
            Cl=Cl[sl].copy(),
            Cd_i=Cd_i[sl].copy(),
            Cd_profile=Cd_p[sl].copy() if has_profile else None,
            alpha_eff=alpha_eff[sl].copy(),
            alpha_i=alpha_i[sl].copy(),
            local_lift=rho * V * strip_gamma[sl],
            surface_name=surf.name,
            chord=chord[sl].copy(),
            Cm_section=Cm_s[sl].copy(),
        ))

    return LoadsResult(
        totals=totals,
        spanwise=spanwise,
        strip_gamma=strip_gamma,
        strip_force=F_strip,
        alpha_eff=alpha_eff,
        extras={
            "trefftz_normalwash": w_n,
            "force_total_body": F_total,
            "moment_total_geometry_axes": M_geo,
        },
    )
