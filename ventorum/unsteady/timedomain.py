# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Prescribed-wake time-domain vortex-ring lattice solver.

This module marches a rigid :class:`ventorum.unsteady.rings.RingLattice` along
a prescribed heave and pitch path in incompressible flow, in free air or over a
flat ground plane. The wake is prescribed: at each time step a new row of
vortex rings leaves the trailing-edge closure with the trailing-edge ring
strength of the previous step, and all shed rows move with the free stream
without induced-velocity deformation.

Formulation
-----------
1. **Frames and rigid-body motion.** Geometry axes are x aft, y right, z up.
   The mean flight path lies along ``V_0 = V_inf * d_0`` [m/s] (parallel to the
   ground when flat-ground effect is active). The wing moves rigidly relative
   to this mean-flight frame: heave ``z(t)`` [m] translates ``ref_point`` along
   the vertical direction ``e_h`` (``lift_direction(alpha)`` in free air or
   ``ground.normal`` in ground effect), and pitch ``theta(t)`` [rad] rotates
   the wing nose up about the body ``+y`` axis through ``ref_point``.
2. **Time step and wake shedding.** The default time step is
   ``dt = dx_TE / V_inf`` [s], where ``dx_TE`` [m] is the strip-width-weighted
   mean trailing-edge panel length (Katz and Plotkin 2001, chapter 13). A note
   is added when a user ``dt`` differs from ``dx_TE / V_inf`` by more than
   25 %. Starting impulsively from rest of the wake (zero wake rows at
   ``t = 0``; Katz and Plotkin 2001, section 13.11), each step ``n >= 1`` sheds
   one wake row behind every strip using the trailing-edge closure
   ``te_extension_fraction`` and junction topology of
   :mod:`ventorum.unsteady.harmonic`. Its strength equals the trailing-edge
   bound ring strength of step ``n - 1`` (unsteady Kutta condition; Katz and
   Plotkin 2001, section 13.8).
3. **Linear system and ground effect.** At each step, the impermeability
   condition at the moved collocation points determines the bound vortex-ring
   circulations. In free air (and for a stationary wing in ground effect), the
   ``N x N`` bound influence matrix is invariant and LU-factorised once
   (``O(N^3)`` setup, ``O(N^2)`` back-substitution per step). In ground effect
   with wing motion, the mirror images of the bound rings and wake move with
   the wing, so the image part of the bound influence matrix is recomputed and
   the ``N x N`` system is refactorised at every moving step (``O(N^3)`` linear
   algebra plus ``O(N^2)`` image-ring kernel work per step).
4. **Unsteady loads.** At each step, aerodynamic loads combine:

   * Kutta-Joukowski loads on all bound spanwise segments and chordwise legs
     with the leg-force rules of :func:`ventorum.aero.loads.leg_force_rules`
     and the full local velocity (onset minus surface velocity plus induced
     velocity of bound rings, wake rings and ground images);
   * apparent-mass loads ``rho * (dGamma / dt) * area_vec`` over the physical
     surface area of each bound ring, using a second-order backward difference
     for ``dGamma / dt`` at steps ``n >= 2`` and a first-order backward
     difference at step ``n = 1`` (zero at the initial instant ``n = 0``);
   * the trailing-edge boundary load correction of
     :mod:`ventorum.unsteady.harmonic`, ``Delta gamma_TE,s = f_TE,s *
     (Gamma_w1,s - Gamma_TE,s)`` with ``f_TE,s = dx_TE,s / (2 * (dx_TE,s +
     dx_w))`` applied at the physical trailing edge for ``n >= 1``.

   Force coefficients ``(CL, CD, CY)`` are projected onto the mean wind axes
   and moment coefficients ``(Cl, Cm, Cn)`` are reported in the body axes of
   the mean state about the instantaneous reference point.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13 (unsteady vortex-ring lattice, wake
  shedding, impulsive start, unsteady Bernoulli loads and method of images).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg as sl

from ventorum.aero.loads import leg_force_rules, leg_segments, leg_strength
from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    lift_direction,
    side_direction,
)
from ventorum.aero.vortex_ring import ring_normal_influence, ring_velocity
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import VortexLattice
from ventorum.unsteady.harmonic import (
    _eval_pair_cores,
    _ey_cross,
    _physical_ring_areas,
    _rings_to_horseshoe,
    _rot_y,
    _slice_rc_rows,
)
from ventorum.unsteady.rings import (
    RingLattice,
    _ring_pair_cores,
    build_ring_lattice,
)
from ventorum.utils.vec import cross3

# Chunk sizes for bounded temporary memory in wake and load kernel calls.
_WAKE_ROW_CHUNK = 32
_LOAD_POINT_CHUNK = 256
# Memory cap [B] for caching reusable velocity influence tensors.
_MAX_CACHE_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class _TimeDomainResponse:
    """Internal result record of the prescribed-wake time-domain ring solver.

    Force coefficients ``(CL, CD, CY)`` are along the mean wind axes (lift,
    free stream, side). Moment coefficients ``(Cl, Cm, Cn)`` are in body axes
    of the mean state (``Cl > 0`` right wing down, ``Cm > 0`` nose up,
    ``Cn > 0`` nose right).

    Attributes
    ----------
    time : numpy.ndarray
        Time instants ``t[n] = n * dt`` [s], shape ``(n_steps + 1,)``.
    z : numpy.ndarray
        Prescribed heave displacement of ``ref_point`` along ``e_h`` [m],
        shape ``(n_steps + 1,)``.
    theta : numpy.ndarray
        Prescribed pitch angle about ``ref_point`` [rad] (positive nose up),
        shape ``(n_steps + 1,)``.
    z_dot : numpy.ndarray
        Heave velocity [m/s] at each time instant, shape ``(n_steps + 1,)``.
    theta_dot : numpy.ndarray
        Pitch angular rate [rad/s] at each time instant, shape
        ``(n_steps + 1,)``.
    CL : numpy.ndarray
        Lift coefficient per step [-], shape ``(n_steps + 1,)``.
    CD : numpy.ndarray
        Near-field drag coefficient per step [-], shape ``(n_steps + 1,)``.
    CY : numpy.ndarray
        Side-force coefficient per step [-], shape ``(n_steps + 1,)``.
    Cl : numpy.ndarray
        Rolling-moment coefficient per step [-], shape ``(n_steps + 1,)``.
    Cm : numpy.ndarray
        Pitching-moment coefficient per step [-], shape ``(n_steps + 1,)``.
    Cn : numpy.ndarray
        Yawing-moment coefficient per step [-], shape ``(n_steps + 1,)``.
    strip_cl : numpy.ndarray
        Per-strip section lift coefficient per step [-], shape
        ``(n_steps + 1, n_strips)``.
    gamma_bound : numpy.ndarray
        Bound vortex-ring circulations at the last step [m^2/s], shape
        ``(n_panels,)``.
    gamma_wake : numpy.ndarray
        Wake vortex-ring circulations at the last step [m^2/s], shape
        ``(n_steps, n_strips)``. Row ``0`` is the newest wake row behind the
        trailing edge and row ``n_steps - 1`` is the oldest wake row.
    wake_corners : numpy.ndarray
        Wake vortex-ring corners at the last step in the mean-flight frame [m],
        shape ``(n_steps, n_strips, 4, 3)``, ordered matching ``gamma_wake``.
    dt : float
        Time step [s].
    dt_default : float
        Default convective time step ``dx_te_mean / V_inf`` [s].
    dx_w : float
        Wake row convective length ``V_inf * dt`` [m].
    dx_te_mean : float
        Strip-width-weighted mean trailing-edge panel length [m].
    n_steps : int
        Number of time steps marched (total samples is ``n_steps + 1``).
    n_panels : int
        Number of bound vortex rings.
    n_strips : int
        Number of spanwise strips.
    n_chord : int
        Number of chordwise panels per strip.
    te_correction : bool
        True when the trailing-edge boundary load correction is included.
    c_ref : float
        Reference chord [m].
    b_ref : float
        Reference span [m].
    S_ref : float
        Reference area [m^2].
    notes : list of str
        Diagnostic notes (for example when ``dt`` differs from ``dx_TE / V_inf``
        by more than 25 %).
    """

    time: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    z_dot: np.ndarray
    theta_dot: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CY: np.ndarray
    Cl: np.ndarray
    Cm: np.ndarray
    Cn: np.ndarray
    strip_cl: np.ndarray
    gamma_bound: np.ndarray
    gamma_wake: np.ndarray
    wake_corners: np.ndarray
    dt: float
    dt_default: float
    dx_w: float
    dx_te_mean: float
    n_steps: int
    n_panels: int
    n_strips: int
    n_chord: int
    te_correction: bool
    c_ref: float
    b_ref: float
    S_ref: float
    notes: list[str] = field(default_factory=list)


def _tile_wake_rc(rc_wake: np.ndarray | float, n_chunk: int) -> np.ndarray | float:
    """Repeat per-strip wake core radii *rc_wake* [m] across *n_chunk* wake rows."""
    if np.ndim(rc_wake) == 0:
        return rc_wake
    arr = np.asarray(rc_wake, dtype=np.float64)
    if arr.ndim == 1:
        return np.tile(arr, n_chunk)
    return np.tile(arr, (1, n_chunk))


def _sample_motion(
    motion: Sequence[float] | np.ndarray | Callable[[float], float],
    time: np.ndarray,
    dt: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample motion *motion* and its time derivative at *time* [s].

    Callable motions are differentiated with tiny finite differences on the
    continuous function (right-sided second-order at ``t = 0``, central for
    ``t > 0``). Array motions of length ``n_steps + 1`` are differentiated with
    second-order finite differences on the time grid (first-order when
    ``n_steps == 1``).
    """
    n_pts = int(time.size)
    if callable(motion):
        vals = np.empty(n_pts, dtype=np.float64)
        dots = np.empty(n_pts, dtype=np.float64)
        eps_base = 1.0e-7 * dt
        for i, ti in enumerate(time):
            tf = float(ti)
            v = float(motion(tf))
            vals[i] = v
            if i == 0:
                h = eps_base
                v1 = float(motion(tf + h))
                v2 = float(motion(tf + 2.0 * h))
                dots[i] = (-3.0 * v + 4.0 * v1 - v2) / (2.0 * h)
            else:
                h = min(eps_base, 0.25 * tf)
                vp = float(motion(tf + h))
                vm = float(motion(tf - h))
                dots[i] = (vp - vm) / (2.0 * h)
        if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(dots)):
            raise InputError(f"Motion {name} returned non-finite values.")
        return vals, dots

    try:
        arr = np.asarray(motion, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"Motion {name} must be a 1-D array of length {n_pts} or a callable.") from exc
    if arr.ndim != 1 or arr.shape[0] != n_pts:
        raise InputError(
            f"Motion array {name} must have length n_steps + 1 = {n_pts}, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise InputError(f"Motion array {name} must contain only finite numbers.")

    dots = np.empty(n_pts, dtype=np.float64)
    if n_pts == 2:
        slope = (arr[1] - arr[0]) / dt
        dots[0] = slope
        dots[1] = slope
    else:
        dots[0] = (-3.0 * arr[0] + 4.0 * arr[1] - arr[2]) / (2.0 * dt)
        dots[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
        dots[-1] = (3.0 * arr[-1] - 4.0 * arr[-2] + arr[-3]) / (2.0 * dt)
    return arr.copy(), dots


def _move_wing_geometry(
    rings: RingLattice,
    ref_point: np.ndarray,
    x_ref: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return moved ``(cp, normal_bc, corners, te_left, te_right, side_nodes)`` in mean-flight axes."""
    if theta == 0.0:
        shift = x_ref - ref_point
        cp = rings.cp + shift[None, :]
        normal_bc = rings.normal_bc
        corners = rings.corners + shift[None, None, :]
        te_left = rings.te_left + shift[None, :]
        te_right = rings.te_right + shift[None, :]
        side_nodes = None if rings.side_nodes is None else rings.side_nodes + shift[None, None, None, :]
        return cp, normal_bc, corners, te_left, te_right, side_nodes

    cp = x_ref[None, :] + _rot_y(rings.cp - ref_point[None, :], theta)
    normal_bc = _rot_y(rings.normal_bc, theta)
    corners = x_ref[None, None, :] + _rot_y(rings.corners - ref_point[None, None, :], theta)
    te_left = x_ref[None, :] + _rot_y(rings.te_left - ref_point[None, :], theta)
    te_right = x_ref[None, :] + _rot_y(rings.te_right - ref_point[None, :], theta)
    side_nodes = (
        None
        if rings.side_nodes is None
        else x_ref[None, None, None, :] + _rot_y(rings.side_nodes - ref_point[None, None, None, :], theta)
    )
    return cp, normal_bc, corners, te_left, te_right, side_nodes


def _deduplicate_load_points(
    P_load_b: np.ndarray,
    rc_load_ring: np.ndarray | float,
    rc_load_wake: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(unique_idx, inv_idx)`` merging coincident load points when target cores do not vary."""
    n_eval = P_load_b.shape[0]
    if np.ndim(rc_load_ring) == 2 or np.ndim(rc_load_wake) == 2 or n_eval == 0:
        idx = np.arange(n_eval, dtype=np.int64)
        return idx, idx

    unique_idx: list[int] = []
    inv_idx = np.empty(n_eval, dtype=np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    scale = max(float(np.max(np.abs(P_load_b))), 1.0)
    tol = 1.0e-13 * scale
    inv_grid = 1.0 / (1.0e-11 * scale)

    for i in range(n_eval):
        pt = P_load_b[i]
        key = (int(round(pt[0] * inv_grid)), int(round(pt[1] * inv_grid)), int(round(pt[2] * inv_grid)))
        found = -1
        cand = buckets.get(key)
        if cand is not None:
            for u_pos in cand:
                rep = P_load_b[unique_idx[u_pos]]
                if abs(pt[0] - rep[0]) <= tol and abs(pt[1] - rep[1]) <= tol and abs(pt[2] - rep[2]) <= tol:
                    found = u_pos
                    break
        if found >= 0:
            inv_idx[i] = found
        else:
            u_pos = len(unique_idx)
            unique_idx.append(i)
            inv_idx[i] = u_pos
            buckets.setdefault(key, []).append(u_pos)

    return np.asarray(unique_idx, dtype=np.int64), inv_idx


def _bound_velocity_at(
    P: np.ndarray,
    corners: np.ndarray,
    side_nodes: np.ndarray | None,
    side_counts: np.ndarray | None,
    gamma: np.ndarray,
    rc_ring: np.ndarray | float,
    *,
    ground: GroundPlane | None,
    backend: str | None,
    image_only: bool = False,
) -> np.ndarray:
    """Return induced velocity [m/s] at *P* from bound rings (and optional ground images)."""
    m = P.shape[0]
    v_out = np.zeros((m, 3), dtype=np.float64)
    img_corners = None if ground is None else ground.reflect_points(corners)
    img_sn = None if (ground is None or side_nodes is None) else ground.reflect_points(side_nodes)
    for i0 in range(0, m, _LOAD_POINT_CHUNK):
        i1 = min(m, i0 + _LOAD_POINT_CHUNK)
        Pc = P[i0:i1]
        rc_c = _slice_rc_rows(rc_ring, i0, i1)
        if not image_only:
            vr = ring_velocity(
                Pc, corners, rc_c, side_nodes=side_nodes, side_counts=side_counts, backend=backend,
            )
            v_out[i0:i1] += vr.swapaxes(1, 2) @ gamma
        if img_corners is not None:
            vr_i = ring_velocity(
                Pc, img_corners, rc_c, side_nodes=img_sn, side_counts=side_counts, backend=backend,
            )
            v_out[i0:i1] -= vr_i.swapaxes(1, 2) @ gamma
    return v_out


def _wake_normal_rhs(
    cp: np.ndarray,
    normal_bc: np.ndarray,
    wake_corners: np.ndarray,
    wake_gamma: np.ndarray,
    n_active: int,
    rc_wake: np.ndarray | float,
    *,
    ground: GroundPlane | None,
    backend: str | None,
) -> np.ndarray:
    """Return wake normal velocity [m/s] at *cp* from the first *n_active* wake rows."""
    vn = np.zeros(cp.shape[0], dtype=np.float64)
    for r0 in range(0, n_active, _WAKE_ROW_CHUNK):
        r1 = min(n_active, r0 + _WAKE_ROW_CHUNK)
        n_chunk = r1 - r0
        wc = wake_corners[r0:r1].reshape(-1, 4, 3)
        wg = wake_gamma[r0:r1].ravel()
        rc_c = _tile_wake_rc(rc_wake, n_chunk)
        vn += ring_normal_influence(cp, normal_bc, wc, rc_c, backend=backend) @ wg
        if ground is not None:
            wc_img = ground.reflect_points(wc)
            vn -= ring_normal_influence(cp, normal_bc, wc_img, rc_c, backend=backend) @ wg
    return vn


def _wake_velocity_at(
    P: np.ndarray,
    wake_corners: np.ndarray,
    wake_gamma: np.ndarray,
    n_active: int,
    rc_wake: np.ndarray | float,
    *,
    ground: GroundPlane | None,
    backend: str | None,
) -> np.ndarray:
    """Return wake induced velocity [m/s] at *P* from the first *n_active* wake rows."""
    m = P.shape[0]
    v_out = np.zeros((m, 3), dtype=np.float64)
    for r0 in range(0, n_active, _WAKE_ROW_CHUNK):
        r1 = min(n_active, r0 + _WAKE_ROW_CHUNK)
        n_chunk = r1 - r0
        wc = wake_corners[r0:r1].reshape(-1, 4, 3)
        wg = wake_gamma[r0:r1].ravel()
        wc_img = None if ground is None else ground.reflect_points(wc)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc = P[i0:i1]
            rc_c = _tile_wake_rc(_slice_rc_rows(rc_wake, i0, i1), n_chunk)
            vw = ring_velocity(Pc, wc, rc_c, backend=backend)
            v_out[i0:i1] += vw.swapaxes(1, 2) @ wg
            if wc_img is not None:
                vw_i = ring_velocity(Pc, wc_img, rc_c, backend=backend)
                v_out[i0:i1] -= vw_i.swapaxes(1, 2) @ wg
    return v_out


def _solve_time_rings(
    lattice: VortexLattice,
    condition: FlightCondition,
    *,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ref_point: np.ndarray | None = None,
    z: Sequence[float] | np.ndarray | Callable[[float], float],
    theta: Sequence[float] | np.ndarray | Callable[[float], float],
    n_steps: int,
    dt: float | None = None,
    ground: GroundPlane | None = None,
    wake_dir: np.ndarray | None = None,
    te_extension_fraction: float = 0.25,
    rc: np.ndarray | float | None = None,
    use_cross_surface_core: bool = True,
    te_correction: bool = True,
    wake_alignment: str = "freestream",
    backend: str | None = None,
) -> _TimeDomainResponse:
    """March the prescribed-wake time-domain vortex-ring solver for *n_steps* steps.

    In free air (and for stationary motion in ground effect), the ``N x N``
    bound influence matrix is LU-factorised once (``O(N^3)`` setup, ``O(N^2)``
    solve per step). In ground effect with wing motion, the mirror image of the
    wing moves relative to the real wing at every step, so the image bound
    influence matrix is recomputed and the ``N x N`` system is refactorised at
    each moving step (``O(N^3)`` per step).

    Parameters
    ----------
    lattice : VortexLattice
        Built vortex-lattice mesh (``collocation="vlm"``).
    condition : FlightCondition
        Mean flight state (``V_inf`` [m/s], ``rho`` [kg/m^3], ``alpha``,
        ``beta`` [rad]).
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray or None, optional
        Moment reference and pitch pivot point [m], shape ``(3,)``. None uses
        the origin.
    z : sequence of float, numpy.ndarray or callable
        Prescribed heave displacement of ``ref_point`` [m] (positive up along
        ``e_h``), either an array of length ``n_steps + 1`` or a callable
        ``z(t)`` of time ``t`` [s].
    theta : sequence of float, numpy.ndarray or callable
        Prescribed pitch angle about ``ref_point`` [rad] (positive nose up),
        either an array of length ``n_steps + 1`` or a callable ``theta(t)``
        of time ``t`` [s].
    n_steps : int
        Number of time steps to march (must be ``>= 1``). Produces
        ``n_steps + 1`` load samples at ``t = 0, dt, ..., n_steps * dt``.
    dt : float or None, optional
        Time step [s]. None uses the default ``dx_TE / V_inf`` [s], where
        ``dx_TE`` is the strip-width-weighted mean trailing-edge panel length.
        A diagnostic note is added when *dt* differs from ``dx_TE / V_inf`` by
        more than 25 %.
    ground : GroundPlane or None, optional
        Flat ground plane in mean-flight axes, or None for free air.
    wake_dir : numpy.ndarray or None, optional
        Unit wake convection direction [-], shape ``(3,)`` or ``(n_strips, 3)``.
        None uses the free stream (or body x-axis in free air when
        ``wake_alignment="body"``).
    te_extension_fraction : float, optional
        Trailing-edge ring closure distance aft of the trailing edge as a
        fraction of the trailing-edge panel length [-]. Default ``0.25``.
    rc : numpy.ndarray or float or None, optional
        Core radius override [m]. None uses ``lattice.rc`` (with cross-surface
        target core when *use_cross_surface_core* is True).
    use_cross_surface_core : bool, optional
        Add cross-surface target cores between different core groups when *rc*
        is None. Default True.
    te_correction : bool, optional
        Apply the trailing-edge boundary load correction using the first wake
        row as the shed ring. Default True.
    wake_alignment : {'freestream', 'body'}, optional
        Wake convection direction rule when *wake_dir* is None. Default
        ``'freestream'``.
    backend : {'auto', 'numba', 'numpy'} or None, optional
        Ring-kernel backend override. None follows
        :mod:`ventorum.aero.vortex`.

    Returns
    -------
    _TimeDomainResponse
        Time history of motion and load coefficients, final bound and wake ring
        strengths and wake geometry, notes and mesh metadata.

    Raises
    ------
    InputError
        If *n_steps*, *dt*, *z*, *theta*, *lattice* or reference scalars are
        invalid, or if any node of the wing reaches or crosses *ground* at any
        step.
    """
    # 1. Validate scalar inputs and reference point.
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise InputError(f"n_steps must be an integer >= 1, got {n_steps!r}.")
    n_steps_int = int(n_steps)
    if n_steps_int < 1:
        raise InputError(f"n_steps must be >= 1, got {n_steps_int}.")

    for name, val in (
        ("V_inf", condition.V_inf),
        ("rho", condition.rho),
        ("S_ref", S_ref),
        ("b_ref", b_ref),
        ("c_ref", c_ref),
    ):
        v = float(val)
        if not math.isfinite(v) or v <= 0.0:
            raise InputError(f"{name} must be finite and > 0, got {val!r}.")
    if wake_alignment not in ("freestream", "body"):
        raise InputError(f"wake_alignment must be 'freestream' or 'body', got {wake_alignment!r}.")

    rp = np.zeros(3, dtype=np.float64) if ref_point is None else np.asarray(ref_point, dtype=np.float64)
    if rp.shape != (3,) or not np.all(np.isfinite(rp)):
        raise InputError(f"ref_point must be a finite 3-vector [m], got {ref_point!r}.")

    V_inf = float(condition.V_inf)
    rho = float(condition.rho)
    S_ref_f = float(S_ref)
    b_ref_f = float(b_ref)
    c_ref_f = float(c_ref)
    q_inf = 0.5 * rho * V_inf * V_inf
    qS = q_inf * S_ref_f

    # 2. Mean-flight axes and vortex-ring lattice.
    d_0 = freestream_direction(float(condition.alpha), float(condition.beta))
    L_0 = lift_direction(float(condition.alpha))
    S_0 = side_direction(float(condition.alpha), float(condition.beta))
    V_0 = V_inf * d_0
    e_h = L_0 if ground is None else np.asarray(ground.normal, dtype=np.float64)

    if wake_dir is None:
        if wake_alignment == "body" and ground is None:
            wd_in = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            wd_in = d_0
    else:
        wd_in = np.asarray(wake_dir, dtype=np.float64)

    rings = build_ring_lattice(
        lattice,
        wake_dir=wd_in,
        te_extension_fraction=te_extension_fraction,
    )
    n_p = rings.n_panels
    n_s = rings.n_strips
    te_idx = rings.te_indices

    # 3. Time step and diagnostic note.
    dx_te = np.asarray(lattice.panel_length[te_idx], dtype=np.float64)
    dx_te_mean = float(np.sum(rings.strip_width * dx_te) / np.sum(rings.strip_width))
    if not math.isfinite(dx_te_mean) or dx_te_mean <= 0.0:
        raise InputError(f"Invalid mean trailing-edge panel length dx_te_mean={dx_te_mean!r}.")
    dt_default = dx_te_mean / V_inf

    notes: list[str] = []
    if dt is None:
        dt_val = dt_default
    else:
        dt_val = float(dt)
        if not math.isfinite(dt_val) or dt_val <= 0.0:
            raise InputError(f"dt must be finite and > 0, got {dt!r}.")
        if abs(dt_val - dt_default) > 0.25 * dt_default:
            notes.append(
                f"Time step dt = {dt_val:.6g} s differs from the default dx_TE / V_inf = "
                f"{dt_default:.6g} s by more than 25 % (ratio {dt_val / dt_default:.3g})."
            )

    dx_w = V_inf * dt_val
    f_te = dx_te / (2.0 * (dx_te + dx_w))
    wake_conv = dx_w * rings.wake_dir  # (n_s, 3)

    # 4. Sample prescribed motion and validate ground clearance at every step.
    time = np.arange(n_steps_int + 1, dtype=np.float64) * dt_val
    z_arr, z_dot = _sample_motion(z, time, dt_val, "z")
    th_arr, th_dot = _sample_motion(theta, time, dt_val, "theta")

    if ground is not None:
        wing_pts_rel = lattice.all_points() - rp[None, :]
        for step in range(n_steps_int + 1):
            x_ref_s = rp + z_arr[step] * e_h
            th_s = float(th_arr[step])
            pts_s = x_ref_s[None, :] + (wing_pts_rel if th_s == 0.0 else _rot_y(wing_pts_rel, th_s))
            h_min = float(np.min(ground.height(pts_s)))
            if h_min <= 0.0:
                raise InputError(
                    f"Wing node reaches or crosses the ground plane at step {step} "
                    f"(t = {time[step]:.6g} s, minimum height {h_min:.4g} m)."
                )

    # 5. Core radii and body-frame load geometry (exact match with harmonic.py).
    rc_ring, rc_wake = _ring_pair_cores(rings, rc, use_cross_surface_core)

    mid_b = np.asarray(lattice.force_points, dtype=np.float64)
    l_vec_b = np.asarray(lattice.b - lattice.a, dtype=np.float64)
    leg_start, leg_end = leg_segments(lattice)
    leg_mid_b = 0.5 * (leg_start + leg_end)
    leg_vec_b = leg_end - leg_start
    leg_normal_b, leg_scale, _ = leg_force_rules(lattice)
    n_leg_sets = leg_mid_b.shape[0] // n_p

    P_load_b = np.vstack([mid_b, leg_mid_b])
    eval_panels = np.tile(np.arange(n_p, dtype=np.int64), 1 + n_leg_sets)
    eval_strips = rings.panel_strip[eval_panels]
    seg_vec_b = np.vstack([l_vec_b, leg_vec_b])
    seg_normal_b = np.vstack([np.zeros((n_p, 3), dtype=np.float64), leg_normal_b])
    seg_scale = np.concatenate([np.ones(n_p, dtype=np.float64), leg_scale])
    arm_b = P_load_b - rp[None, :]

    rc_load_ring, rc_load_wake = _eval_pair_cores(rings, eval_panels, rc, use_cross_surface_core)
    load_unique_idx, load_inv_idx = _deduplicate_load_points(P_load_b, rc_load_ring, rc_load_wake)
    P_load_u_b = P_load_b[load_unique_idx]
    arm_u_b = arm_b[load_unique_idx]
    n_eval_u = P_load_u_b.shape[0]
    rc_load_u_ring = rc_load_ring[load_unique_idx] if np.ndim(rc_load_ring) == 2 else rc_load_ring
    rc_load_u_wake = rc_load_wake[load_unique_idx] if np.ndim(rc_load_wake) == 2 else rc_load_wake

    te_l_phys = np.asarray(lattice.te_left, dtype=np.float64)
    te_r_phys = np.asarray(lattice.te_right, dtype=np.float64)
    l_te_b = te_r_phys - te_l_phys
    p_te_b = te_l_phys + np.asarray(lattice.cp_frac, dtype=np.float64)[:, None] * l_te_b
    arm_te_b = p_te_b - rp[None, :]
    rc_te_ring, rc_te_wake = _eval_pair_cores(rings, te_idx, rc, use_cross_surface_core)

    area_vec_b, ring_centroid_b, area_moment_vec_b = _physical_ring_areas(lattice, rings, ref_point=rp)
    arm_ring_b = ring_centroid_b - rp[None, :]
    strip_area = np.asarray(lattice.chord, dtype=np.float64) * np.asarray(lattice.width, dtype=np.float64)

    # 6. Precompute invariant real bound influence matrix and optional velocity caches.
    A_bb_real = ring_normal_influence(
        rings.cp,
        rings.normal_bc,
        rings.corners,
        rc_ring,
        side_nodes=rings.side_nodes,
        side_counts=rings.side_counts,
        backend=backend,
    )
    lu_Abb = sl.lu_factor(A_bb_real) if ground is None else None

    cache_bound_load = (n_eval_u * n_p * 24) <= _MAX_CACHE_BYTES
    V_bound_load_b = (
        ring_velocity(
            P_load_u_b,
            rings.corners,
            rc_load_u_ring,
            side_nodes=rings.side_nodes,
            side_counts=rings.side_counts,
            backend=backend,
        )
        if cache_bound_load
        else None
    )
    V_bound_te_b = (
        ring_velocity(
            p_te_b,
            rings.corners,
            rc_te_ring,
            side_nodes=rings.side_nodes,
            side_counts=rings.side_counts,
            backend=backend,
        )
        if te_correction and (n_s * n_p * 24) <= _MAX_CACHE_BYTES
        else None
    )

    # Stationary-wing detection: constant (z, theta) allows O(n_steps) wake-row reuse.
    is_stationary = bool(np.all(z_arr == z_arr[0]) and np.all(th_arr == th_arr[0]))
    A_wake_stat = (
        np.empty((n_steps_int, n_p, n_s), dtype=np.float64)
        if is_stationary and (n_steps_int * n_p * n_s * 8) <= _MAX_CACHE_BYTES
        else None
    )
    V_wake_load_stat = (
        np.empty((n_steps_int, n_eval_u, n_s, 3), dtype=np.float64)
        if is_stationary and (n_steps_int * n_eval_u * n_s * 24) <= _MAX_CACHE_BYTES
        else None
    )
    V_wake_te_stat = (
        np.empty((n_steps_int, n_s, n_s, 3), dtype=np.float64)
        if is_stationary and te_correction and (n_steps_int * n_s * n_s * 24) <= _MAX_CACHE_BYTES
        else None
    )
    V_bound_img_load_stat: np.ndarray | None = None
    V_bound_img_te_stat: np.ndarray | None = None

    # 7. Preallocate wake buffers (length n_steps) and output time series (length n_steps + 1).
    wake_corners_earth = np.empty((n_steps_int, n_s, 4, 3), dtype=np.float64)
    wake_corners_cur = np.empty((n_steps_int, n_s, 4, 3), dtype=np.float64)
    wake_gamma_buf = np.empty((n_steps_int, n_s), dtype=np.float64)

    CL = np.empty(n_steps_int + 1, dtype=np.float64)
    CD = np.empty(n_steps_int + 1, dtype=np.float64)
    CY = np.empty(n_steps_int + 1, dtype=np.float64)
    Cl = np.empty(n_steps_int + 1, dtype=np.float64)
    Cm = np.empty(n_steps_int + 1, dtype=np.float64)
    Cn = np.empty(n_steps_int + 1, dtype=np.float64)
    strip_cl = np.empty((n_steps_int + 1, n_s), dtype=np.float64)

    prev_base_l: np.ndarray | None = None
    prev_base_r: np.ndarray | None = None
    prev_gamma: np.ndarray | None = None
    prev2_gamma: np.ndarray | None = None
    prev_z_ge: float | None = None
    prev_th_ge: float | None = None

    # 8. Time-marching loop (step = 0 .. n_steps).
    for step in range(n_steps_int + 1):
        zn = float(z_arr[step])
        th = float(th_arr[step])
        zd = float(z_dot[step])
        thd = float(th_dot[step])
        x_ref = rp + zn * e_h

        cp, normal_bc, corners, te_left, te_right, side_nodes = _move_wing_geometry(rings, rp, x_ref, th)

        # Earth-frame trailing-edge closure corners for drift-free convection.
        shift_conv = float(step) * wake_conv
        base_l = te_left - shift_conv
        base_r = te_right - shift_conv

        # Append the newly shed wake row at step >= 1 (Kutta condition: Gamma_TE of previous step).
        if step > 0:
            row_idx = step - 1
            wake_corners_earth[row_idx, :, 0, :] = base_l
            wake_corners_earth[row_idx, :, 1, :] = base_r
            wake_corners_earth[row_idx, :, 2, :] = prev_base_r
            wake_corners_earth[row_idx, :, 3, :] = prev_base_l
            wake_gamma_buf[row_idx] = prev_gamma[te_idx]
            wake_corners_cur[:step] = wake_corners_earth[:step] + shift_conv[None, :, None, :]

        # Bound linear system (refactorise in ground effect when wing pose changes).
        if ground is not None:
            pose_changed = (prev_z_ge is None) or (zn != prev_z_ge) or (th != prev_th_ge)
            if pose_changed:
                img_corners = ground.reflect_points(corners)
                img_sn = None if side_nodes is None else ground.reflect_points(side_nodes)
                A_img = ring_normal_influence(
                    cp,
                    normal_bc,
                    img_corners,
                    rc_ring,
                    side_nodes=img_sn,
                    side_counts=rings.side_counts,
                    backend=backend,
                )
                lu_Abb = sl.lu_factor(A_bb_real - A_img)
                prev_z_ge = zn
                prev_th_ge = th

        # Impermeability right-hand side: (V_surf - V_0 - v_wake) . n_bc.
        v_surf_cp = zd * e_h[None, :] + thd * _ey_cross(cp - x_ref[None, :])
        rhs = np.sum((v_surf_cp - V_0[None, :]) * normal_bc, axis=1)

        if step > 0:
            if A_wake_stat is not None:
                # Stationary wing: row of age (step - 1) is the oldest row (index 0 in wake_corners_cur).
                age = step - 1
                wc_age = wake_corners_cur[0]  # (n_s, 4, 3)
                blk = ring_normal_influence(cp, normal_bc, wc_age, rc_wake, backend=backend)
                if ground is not None:
                    blk -= ring_normal_influence(
                        cp, normal_bc, ground.reflect_points(wc_age), rc_wake, backend=backend,
                    )
                A_wake_stat[age] = blk
                gamma_by_age = wake_gamma_buf[:step][::-1]
                rhs -= np.einsum("rms,rs->m", A_wake_stat[:step], gamma_by_age)
            else:
                rhs -= _wake_normal_rhs(
                    cp,
                    normal_bc,
                    wake_corners_cur,
                    wake_gamma_buf,
                    step,
                    rc_wake,
                    ground=ground,
                    backend=backend,
                )

        gamma_n = sl.lu_solve(lu_Abb, rhs)

        # Unsteady loads: moved body-frame load geometry.
        if th == 0.0:
            arm_all = arm_b
            arm_u = arm_u_b
            seg_vec = seg_vec_b
            seg_normal = seg_normal_b
            arm_te = arm_te_b
            l_te = l_te_b
            area_vec = area_vec_b
            arm_ring = arm_ring_b
            area_moment_vec = area_moment_vec_b
        else:
            arm_all = _rot_y(arm_b, th)
            arm_u = arm_all[load_unique_idx]
            seg_vec = _rot_y(seg_vec_b, th)
            seg_normal = _rot_y(seg_normal_b, th)
            arm_te = _rot_y(arm_te_b, th)
            l_te = _rot_y(l_te_b, th)
            area_vec = _rot_y(area_vec_b, th)
            arm_ring = _rot_y(arm_ring_b, th)
            area_moment_vec = None if area_moment_vec_b is None else _rot_y(area_moment_vec_b, th)
        P_load_u = x_ref[None, :] + arm_u
        p_te = x_ref[None, :] + arm_te

        # Induced velocity at deduplicated Kutta-Joukowski load points.
        if V_bound_load_b is not None:
            v_real_b = V_bound_load_b.swapaxes(1, 2) @ gamma_n
            v_ind_u = v_real_b if th == 0.0 else _rot_y(v_real_b, th)
            if ground is not None:
                if is_stationary and cache_bound_load:
                    if V_bound_img_load_stat is None:
                        img_c = ground.reflect_points(corners)
                        img_s = None if side_nodes is None else ground.reflect_points(side_nodes)
                        V_bound_img_load_stat = ring_velocity(
                            P_load_u,
                            img_c,
                            rc_load_u_ring,
                            side_nodes=img_s,
                            side_counts=rings.side_counts,
                            backend=backend,
                        )
                    v_ind_u = v_ind_u - (V_bound_img_load_stat.swapaxes(1, 2) @ gamma_n)
                else:
                    v_ind_u = v_ind_u + _bound_velocity_at(
                        P_load_u,
                        corners,
                        side_nodes,
                        rings.side_counts,
                        gamma_n,
                        rc_load_u_ring,
                        ground=ground,
                        backend=backend,
                        image_only=True,
                    )
        else:
            v_ind_u = _bound_velocity_at(
                P_load_u,
                corners,
                side_nodes,
                rings.side_counts,
                gamma_n,
                rc_load_u_ring,
                ground=ground,
                backend=backend,
                image_only=False,
            )

        if step > 0:
            if V_wake_load_stat is not None:
                age = step - 1
                wc_age = wake_corners_cur[0]
                v_row = ring_velocity(P_load_u, wc_age, rc_load_u_wake, backend=backend)
                if ground is not None:
                    v_row -= ring_velocity(P_load_u, ground.reflect_points(wc_age), rc_load_u_wake, backend=backend)
                V_wake_load_stat[age] = v_row
                gamma_by_age = wake_gamma_buf[:step][::-1]
                v_ind_u = v_ind_u + np.einsum("rmsi,rs->mi", V_wake_load_stat[:step], gamma_by_age)
            else:
                v_ind_u = v_ind_u + _wake_velocity_at(
                    P_load_u,
                    wake_corners_cur,
                    wake_gamma_buf,
                    step,
                    rc_load_u_wake,
                    ground=ground,
                    backend=backend,
                )

        v_ind_load = v_ind_u[load_inv_idx]

        # Kutta-Joukowski force and moment.
        v_surf_load = zd * e_h[None, :] + thd * _ey_cross(arm_all)
        V_tot = V_0[None, :] - v_surf_load + v_ind_load
        V_eff = V_tot - np.sum(V_tot * seg_normal, axis=1, keepdims=True) * seg_normal

        g_hs = _rings_to_horseshoe(gamma_n, rings.n_chord)
        g_leg = leg_strength(lattice, g_hs)
        g_all = np.concatenate([g_hs] + [g_leg] * n_leg_sets)

        F_seg = (rho * g_all * seg_scale)[:, None] * cross3(V_eff, seg_vec)
        F_tot = np.sum(F_seg, axis=0)
        M_tot = np.sum(cross3(arm_all, F_seg), axis=0)
        s_lift = np.zeros(n_s, dtype=np.float64)
        np.add.at(s_lift, eval_strips, F_seg @ L_0)

        # Apparent-mass load: rho * (dGamma / dt) * area_vec.
        if step >= 2:
            dG_dt = (3.0 * gamma_n - 4.0 * prev_gamma + prev2_gamma) / (2.0 * dt_val)
        elif step == 1:
            dG_dt = (gamma_n - prev_gamma) / dt_val
        else:
            dG_dt = None

        if dG_dt is not None:
            coef_am = rho * dG_dt
            F_am = coef_am[:, None] * area_vec
            F_tot = F_tot + np.sum(F_am, axis=0)
            if area_moment_vec is not None:
                M_tot = M_tot + np.sum(coef_am[:, None] * area_moment_vec, axis=0)
            else:
                M_tot = M_tot + np.sum(cross3(arm_ring, F_am), axis=0)
            np.add.at(s_lift, rings.panel_strip, F_am @ L_0)

        # Trailing-edge boundary load correction with the first wake row as shed ring.
        if te_correction and step >= 1:
            if V_bound_te_b is not None:
                v_te_real = V_bound_te_b.swapaxes(1, 2) @ gamma_n
                v_ind_te = v_te_real if th == 0.0 else _rot_y(v_te_real, th)
                if ground is not None:
                    if is_stationary:
                        if V_bound_img_te_stat is None:
                            img_c = ground.reflect_points(corners)
                            img_s = None if side_nodes is None else ground.reflect_points(side_nodes)
                            V_bound_img_te_stat = ring_velocity(
                                p_te,
                                img_c,
                                rc_te_ring,
                                side_nodes=img_s,
                                side_counts=rings.side_counts,
                                backend=backend,
                            )
                        v_ind_te = v_ind_te - (V_bound_img_te_stat.swapaxes(1, 2) @ gamma_n)
                    else:
                        v_ind_te = v_ind_te + _bound_velocity_at(
                            p_te,
                            corners,
                            side_nodes,
                            rings.side_counts,
                            gamma_n,
                            rc_te_ring,
                            ground=ground,
                            backend=backend,
                            image_only=True,
                        )
            else:
                v_ind_te = _bound_velocity_at(
                    p_te,
                    corners,
                    side_nodes,
                    rings.side_counts,
                    gamma_n,
                    rc_te_ring,
                    ground=ground,
                    backend=backend,
                    image_only=False,
                )

            if V_wake_te_stat is not None:
                age = step - 1
                wc_age = wake_corners_cur[0]
                v_row_te = ring_velocity(p_te, wc_age, rc_te_wake, backend=backend)
                if ground is not None:
                    v_row_te -= ring_velocity(p_te, ground.reflect_points(wc_age), rc_te_wake, backend=backend)
                V_wake_te_stat[age] = v_row_te
                gamma_by_age = wake_gamma_buf[:step][::-1]
                v_ind_te = v_ind_te + np.einsum("rmsi,rs->mi", V_wake_te_stat[:step], gamma_by_age)
            else:
                v_ind_te = v_ind_te + _wake_velocity_at(
                    p_te,
                    wake_corners_cur,
                    wake_gamma_buf,
                    step,
                    rc_te_wake,
                    ground=ground,
                    backend=backend,
                )

            v_surf_te = zd * e_h[None, :] + thd * _ey_cross(arm_te)
            V_te = V_0[None, :] - v_surf_te + v_ind_te
            dg_te = f_te * (wake_gamma_buf[step - 1] - gamma_n[te_idx])
            F_te = (rho * dg_te)[:, None] * cross3(V_te, l_te)
            F_tot = F_tot + np.sum(F_te, axis=0)
            M_tot = M_tot + np.sum(cross3(arm_te, F_te), axis=0)
            s_lift = s_lift + (F_te @ L_0)

        CL[step] = float((F_tot @ L_0) / qS)
        CD[step] = float((F_tot @ d_0) / qS)
        CY[step] = float((F_tot @ S_0) / qS)
        Cl[step] = float(-M_tot[0] / (qS * b_ref_f))
        Cm[step] = float(M_tot[1] / (qS * c_ref_f))
        Cn[step] = float(-M_tot[2] / (qS * b_ref_f))
        strip_cl[step] = s_lift / (q_inf * strip_area)

        prev_base_l = base_l
        prev_base_r = base_r
        prev2_gamma = prev_gamma
        prev_gamma = gamma_n

    return _TimeDomainResponse(
        time=time,
        z=z_arr,
        theta=th_arr,
        z_dot=z_dot,
        theta_dot=th_dot,
        CL=CL,
        CD=CD,
        CY=CY,
        Cl=Cl,
        Cm=Cm,
        Cn=Cn,
        strip_cl=strip_cl,
        gamma_bound=prev_gamma.copy(),
        gamma_wake=wake_gamma_buf[::-1].copy(),
        wake_corners=wake_corners_cur[::-1].copy(),
        dt=dt_val,
        dt_default=dt_default,
        dx_w=dx_w,
        dx_te_mean=dx_te_mean,
        n_steps=n_steps_int,
        n_panels=n_p,
        n_strips=n_s,
        n_chord=rings.n_chord,
        te_correction=bool(te_correction),
        c_ref=c_ref_f,
        b_ref=b_ref_f,
        S_ref=S_ref_f,
        notes=notes,
    )
