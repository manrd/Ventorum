# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Frequency-domain harmonic vortex-ring lattice solver (longitudinal motions).

This module solves small-amplitude harmonic heave and pitch motions about a
steady mean flight state on a :class:`ventorum.unsteady.rings.RingLattice`,
in free air or in flat-ground effect.

Formulation
-----------
1. **Wake and time step.** Behind each trailing-edge strip ``s``, a flat
   discrete wake of ``N_wx`` vortex-ring rows marches along the strip wake
   direction starting at the trailing-edge ring closure. All strips share one
   row length ``dx_w = V_inf dt`` [m], chosen as the strip-width-weighted mean
   of the trailing-edge panel lengths:

       dx_w = sum_s (w_s dx_TE,s) / sum_s w_s,    dt = dx_w / V_inf.

   For a uniform-chord wing this equals the trailing-edge panel length exactly;
   for a tapered wing it avoids wake-row explosion at a narrow tip while
   recording the per-strip ratio ``dx_TE,s / dx_w`` for later trust checks.
   Wake row ``m`` (``1 .. N_wx``) carries strength
   ``Gamma_w,s,m = Gamma_TE,s exp(-i omega m dt)`` (Hall 1994; Katz and
   Plotkin 2001, chapter 13). After row ``N_wx``, the closed-form 3D
   semi-infinite harmonic far-wake tail (:mod:`ventorum.unsteady.tail`) closes
   the wake without truncation oscillation.

2. **Linear system and Woodbury solve.** The real bound influence matrix
   ``A_bb`` [1/m] (with ground images) and the real wake-row influence tensor
   ``A_bw`` [1/m] are assembled once per configuration and height. ``A_bb`` is
   LU-factorised once in float64. For each reduced frequency ``k = omega c_ref
   / (2 V_inf)`` [-], the system ``(A_bb + W(k) E_te^T) Gamma = rhs`` has a
   rank-``n_strips`` complex wake update in the trailing-edge columns and is
   solved via the Woodbury identity (Parenteau 2021, chapter 4). A direct
   complex LU path is retained for verification.

3. **Right-hand sides.** Unit heave is one reference chord upward along the
   vertical direction ``e_h`` (``ground.normal`` in ground effect,
   :func:`ventorum.aero.system.lift_direction` in free air), so heave responses
   are dimensionless per unit ``h / c_ref`` [-]. Unit pitch is one radian nose
   up about ``ref_point`` [m]. In ground effect (and in free air when the wake
   follows the free stream) with a lifting mean state ``Gamma_0``, the
   loaded-mean-state terms ``-(d(A_bb + A_bw,steady)/dh) Gamma_0`` and
   ``-(d(A_bb + A_bw,steady)/dtheta) Gamma_0`` are added by central finite
   differences of the steady influence operator (Parenteau 2021, chapter 4).

4. **Linearised loads.** Complex force coefficients ``(CL, CD, CY)`` along the
   mean wind axes and moment coefficients ``(Cl, Cm, Cn)`` in Ventorum body
   axes combine:

   * the linearised Kutta-Joukowski load on all bound spanwise segments and
     chordwise legs (using :func:`ventorum.aero.loads.leg_force_rules`), with
     both perturbation circulation times mean local velocity and mean
     circulation times perturbation velocity (motion velocity plus perturbation
     induced velocity);
   * the moved geometry of the mean load in pitch (rotation of the mean force
     and moment arm about ``ref_point``);
   * the apparent-mass load ``rho (i omega Gamma) A_p n_p`` integrated over the
     physical surface quadrilateral covered by each bound ring;
   * the trailing-edge boundary correction ``Delta gamma_TE,s = f_TE,s
     (Gamma_w1,s - Gamma_TE,s)`` with ``f_TE,s = dx_TE,s / (2 (dx_TE,s +
     dx_w))`` applied at the physical trailing edge with the mean local
     velocity. It vanishes identically at ``k = 0`` and restores second-order
     chordwise convergence of unsteady pitching-moment derivatives.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13 (unsteady vortex-ring lattice, wake
  shedding and unsteady Bernoulli loads).
* K. C. Hall, "Eigenanalysis of unsteady flows about airfoils, cascades, and
  wings", AIAA Journal 32(12), 1994, pp. 2426-2432 (harmonic wake phase
  relation; cited through Parenteau 2021).
* M. Parenteau, "A Non-Linear Frequency Domain Potential Flow Model for
  Compressible, Transonic and Viscous Aeroelastic Analyses", PhD thesis,
  Polytechnique Montreal, 2021, chapter 4 (frequency-domain vortex-lattice
  system, Woodbury update and loaded-mean-state terms; cited through study
  material).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg as sl

from ventorum.aero.loads import leg_force_rules, leg_segments, leg_strength
from ventorum.aero.system import (
    GroundPlane,
    check_ground_clearance,
    freestream_direction,
    lift_direction,
    make_unknown_map,
    side_direction,
)
from ventorum.aero.vortex_ring import (
    ring_normal_influence,
    ring_velocity,
    steady_wake_normal_influence,
    steady_wake_velocity,
)
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import CROSS_SURFACE_CORE_FRACTION, VortexLattice
from ventorum.solvers.core import _vlm_rhs
from ventorum.unsteady.rings import (
    RingGroundImage,
    RingLattice,
    _ring_pair_cores,
    build_ring_lattice,
    ground_image_rings,
)
from ventorum.unsteady.tail import harmonic_tail_normal_influence, harmonic_tail_velocity
from ventorum.utils.vec import cross3

# Default wake-row chunk size for batched ring kernel calls.
_WAKE_ROW_CHUNK = 16
# Default evaluation-point chunk size when contracting velocity tensors into load functionals.
_LOAD_POINT_CHUNK = 256
# Default central finite-difference step for pitch geometry derivatives [rad].
_DEFAULT_FD_STEP_THETA = 1.0e-5
# Default central finite-difference step fraction for heave geometry derivatives [-].
_DEFAULT_FD_STEP_H_FRAC = 1.0e-5


@dataclass(slots=True)
class _SteadyLoads:
    """Real aerodynamic load coefficients in Ventorum conventions [-].

    Force coefficients ``(CL, CD, CY)`` are along the mean wind axes (lift,
    free stream, side). Moment coefficients ``(Cl, Cm, Cn)`` are in body axes
    (``Cl > 0`` right wing down, ``Cm > 0`` nose up, ``Cn > 0`` nose right).
    """

    CL: float
    CD: float
    CY: float
    Cl: float
    Cm: float
    Cn: float


@dataclass(slots=True)
class _HarmonicMotionResponse:
    """Complex frequency-domain load coefficients for one unit motion.

    Each coefficient array has shape ``(n_k,)`` and dtype ``complex128`` [-].
    ``strip_cl`` has shape ``(n_k, n_strips)`` and holds the complex section
    lift coefficient of each strip [-].
    """

    CL: np.ndarray
    CD: np.ndarray
    CY: np.ndarray
    Cl: np.ndarray
    Cm: np.ndarray
    Cn: np.ndarray
    strip_cl: np.ndarray


@dataclass(slots=True)
class _HarmonicResponse:
    """Internal response of the harmonic vortex-ring lattice solver.

    Attributes
    ----------
    k : numpy.ndarray
        Evaluated reduced frequencies ``omega c_ref / (2 V_inf)`` [-], shape
        ``(n_k,)``.
    heave : _HarmonicMotionResponse
        Complex load response per unit nondimensional heave ``h / c_ref`` [-].
    pitch : _HarmonicMotionResponse
        Complex load response per unit pitch angle ``theta`` [1/rad] about
        ``ref_point``.
    k0_heave : _SteadyLoads
        Exact ``k -> 0`` limit of heave (height derivatives ``CL_h``, ``Cm_h``,
        ...) [-].
    k0_pitch : _SteadyLoads
        Exact ``k -> 0`` limit of pitch (static incidence derivatives
        ``CL_alpha``, ``Cm_alpha``, ...) [1/rad].
    k0_pitch_rate : _SteadyLoads
        Exact ``k -> 0`` limit of nondimensional pitch rate ``q c_ref / (2 V_inf)``
        (``CL_q``, ``Cm_q``, ...) [-].
    mean_state : _SteadyLoads
        Steady mean-state load coefficients ``(CL, CD, CY, Cl, Cm, Cn)`` [-].
    dt : float
        Wake time step ``dx_w / V_inf`` [s].
    dx_w : float
        Discrete wake row length [m].
    dx_te_ratio : numpy.ndarray
        Per-strip ratio ``dx_TE,s / dx_w`` [-], shape ``(n_strips,)``.
    wake_chords : float
        Requested discrete wake length in reference chords [-].
    wake_length : float
        Actual discrete wake length ``n_wake_rows * dx_w`` [m].
    n_wake_rows : int
        Number of discrete wake rows behind each strip.
    n_panels : int
        Number of bound vortex rings.
    n_strips : int
        Number of spanwise strips.
    n_chord : int
        Number of chordwise panels per strip.
    use_tail : bool
        True when the closed-form 3D far-wake tail is included.
    te_correction : bool
        True when the trailing-edge boundary load correction is included.
    c_ref : float
        Reference chord [m].
    b_ref : float
        Reference span [m].
    S_ref : float
        Reference area [m^2].
    """

    k: np.ndarray
    heave: _HarmonicMotionResponse
    pitch: _HarmonicMotionResponse
    k0_heave: _SteadyLoads
    k0_pitch: _SteadyLoads
    k0_pitch_rate: _SteadyLoads
    mean_state: _SteadyLoads
    dt: float
    dx_w: float
    dx_te_ratio: np.ndarray
    wake_chords: float
    wake_length: float
    n_wake_rows: int
    n_panels: int
    n_strips: int
    n_chord: int
    use_tail: bool
    te_correction: bool
    c_ref: float
    b_ref: float
    S_ref: float

    @property
    def CL0(self) -> float:
        """Return the steady mean lift coefficient [-]."""
        return self.mean_state.CL

    @property
    def Cm0(self) -> float:
        """Return the steady mean pitching-moment coefficient [-]."""
        return self.mean_state.Cm


def _rot_y(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 3-D vector(s) *vec* by *angle* [rad] about the body ``+y`` axis."""
    c, s = math.cos(angle), math.sin(angle)
    out = np.array(vec, dtype=np.float64, copy=True)
    x = out[..., 0].copy()
    z = out[..., 2].copy()
    out[..., 0] = c * x + s * z
    out[..., 2] = -s * x + c * z
    return out


def _ey_cross(vec: np.ndarray) -> np.ndarray:
    """Return ``(0, 1, 0) x vec`` for real or complex vector(s) of shape ``(..., 3)``."""
    v = np.asarray(vec)
    out = np.zeros_like(v)
    out[..., 0] = v[..., 2]
    out[..., 2] = -v[..., 0]
    return out


def _cross3_c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the 3-D cross product of real or complex arrays *a* and *b* along the last axis."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.result_type(a, b))
    out[..., 0] = ay * bz - az * by
    out[..., 1] = az * bx - ax * bz
    out[..., 2] = ax * by - ay * bx
    return out


def _rings_to_horseshoe(gamma_rings: np.ndarray, n_chord: int) -> np.ndarray:
    """Convert real or complex ring circulations [m^2/s] to horseshoe circulations [m^2/s]."""
    g = np.asarray(gamma_rings).reshape(-1, int(n_chord))
    out = np.empty_like(g)
    out[:, 0] = g[:, 0]
    if n_chord > 1:
        out[:, 1:] = g[:, 1:] - g[:, :-1]
    return out.ravel()


def _eval_pair_cores(
    rings: RingLattice,
    target_panels: np.ndarray,
    rc: np.ndarray | float | None,
    use_cross_surface_core: bool,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return ``(rc_rings, rc_wake)`` [m] for arbitrary evaluation points on *target_panels*."""
    te_idx = rings.te_indices
    if rc is not None:
        arr = np.asarray(rc, dtype=np.float64)
        if arr.ndim == 0:
            val = float(arr)
            return val, val
        if arr.ndim == 1:
            if arr.shape[0] != rings.n_panels:
                raise InputError(f"rc must have shape ({rings.n_panels},), got {arr.shape}.")
            return arr, arr[te_idx]
        raise InputError(f"rc for harmonic solves must be scalar or 1-D, got shape {arr.shape}.")

    rc_src = rings.rc
    if not use_cross_surface_core:
        return rc_src, rc_src[te_idx]

    src_group = rings.strip_core_group[rings.panel_strip]
    if np.all(src_group == src_group[0]):
        return rc_src, rc_src[te_idx]

    tg_strip = rings.panel_strip[target_panels]
    tg_group = rings.strip_core_group[tg_strip]
    tg_rc2 = (CROSS_SURFACE_CORE_FRACTION * rings.strip_width[tg_strip]) ** 2
    other = tg_group[:, None] != src_group[None, :]
    rc_mat = np.sqrt(rc_src[None, :] ** 2 + np.where(other, tg_rc2[:, None], 0.0))
    return rc_mat, rc_mat[:, te_idx]


def _slice_rc_rows(rc: np.ndarray | float, i0: int, i1: int) -> np.ndarray | float:
    """Slice *rc* along row axis when it is 2-D."""
    if np.ndim(rc) == 2:
        return np.asarray(rc, dtype=np.float64)[i0:i1]
    return rc


def _physical_ring_areas(lattice: VortexLattice, rings: RingLattice) -> tuple[np.ndarray, np.ndarray]:
    """Return physical surface area vectors [m^2] and centroids [m] of every bound ring.

    Each interior ring ``p`` spans the quadrilateral from its own bound segment
    ``(a[p], b[p])`` to the next bound segment ``(a[p + 1], b[p + 1])``. Each
    trailing-edge ring spans from ``(a[p], b[p])`` to the physical trailing
    edge ``(te_left[s], te_right[s])`` so the apparent-mass integral covers
    exactly the physical chord (length ``0.75 dx_TE`` on the last panel).
    """
    q0 = rings.corners[:, 0, :]
    q1 = rings.corners[:, 1, :]
    q2 = rings.corners[:, 2, :].copy()
    q3 = rings.corners[:, 3, :].copy()
    te_idx = rings.te_indices
    q2[te_idx] = np.asarray(lattice.te_right, dtype=np.float64)
    q3[te_idx] = np.asarray(lattice.te_left, dtype=np.float64)

    # Split each quadrilateral into two triangles for exact area and centroid on tapered panels.
    a1 = 0.5 * cross3(q2 - q0, q1 - q0)
    a2 = 0.5 * cross3(q3 - q0, q2 - q0)
    c1 = (q0 + q1 + q2) / 3.0
    c2 = (q0 + q2 + q3) / 3.0
    area_vec = a1 + a2
    m1 = np.linalg.norm(a1, axis=1)
    m2 = np.linalg.norm(a2, axis=1)
    centroid = (m1[:, None] * c1 + m2[:, None] * c2) / np.maximum((m1 + m2)[:, None], 1e-300)
    return area_vec, centroid


def _build_wake_row_corners(
    te_left: np.ndarray,
    te_right: np.ndarray,
    wake_dir: np.ndarray,
    row_start: int,
    row_stop: int,
    dx_w: float,
) -> np.ndarray:
    """Return wake ring corners [m] for rows ``row_start .. row_stop - 1``, shape ``(n_r * n_s, 4, 3)``."""
    m = np.arange(row_start, row_stop, dtype=np.float64)[:, None, None]
    d0 = (m * dx_w) * wake_dir[None, :, :]
    d1 = ((m + 1.0) * dx_w) * wake_dir[None, :, :]
    c0 = te_left[None, :, :] + d0
    c1 = te_right[None, :, :] + d0
    c2 = te_right[None, :, :] + d1
    c3 = te_left[None, :, :] + d1
    return np.stack([c0, c1, c2, c3], axis=2).reshape(-1, 4, 3)


def _steady_velocity_at(
    P: np.ndarray,
    rings: RingLattice,
    gamma_rings: np.ndarray,
    rc_ring: np.ndarray | float,
    rc_wake: np.ndarray | float,
    *,
    ground: GroundPlane | None = None,
    wake_dir: np.ndarray | None = None,
    image_only: bool = False,
) -> np.ndarray:
    """Return steady induced velocity [m/s] at *P* from *rings* with circulations *gamma_rings*."""
    wd = rings.wake_dir if wake_dir is None else wake_dir
    te_idx = rings.te_indices
    g_te = gamma_rings[te_idx]
    img: RingGroundImage | None = None if ground is None else ground_image_rings(rings, ground, wake_dir=wd)

    m = P.shape[0]
    v_out = np.zeros((m, 3), dtype=np.float64)
    for i0 in range(0, m, _LOAD_POINT_CHUNK):
        i1 = min(m, i0 + _LOAD_POINT_CHUNK)
        Pc = P[i0:i1]
        rc_r = _slice_rc_rows(rc_ring, i0, i1)
        rc_w = _slice_rc_rows(rc_wake, i0, i1)
        if not image_only:
            vr = ring_velocity(Pc, rings.corners, rc_r)
            vw = steady_wake_velocity(Pc, rings.te_left, rings.te_right, wd, rc_w)
            v_out[i0:i1] += np.einsum("mni,n->mi", vr, gamma_rings)
            v_out[i0:i1] += np.einsum("msi,s->mi", vw, g_te)
        if img is not None:
            vr_i = ring_velocity(Pc, img.corners, rc_r)
            vw_i = steady_wake_velocity(Pc, img.te_left, img.te_right, img.wake_dir, rc_w)
            v_out[i0:i1] += img.sign * np.einsum("mni,n->mi", vr_i, gamma_rings)
            v_out[i0:i1] += img.sign * np.einsum("msi,s->mi", vw_i, g_te)
    return v_out


def _build_velocity_load_weights(
    rho: float,
    gamma_0_eval: np.ndarray,
    seg_vec: np.ndarray,
    seg_normal: np.ndarray,
    seg_scale: np.ndarray,
    arm: np.ndarray,
    lift_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear load functional weights ``(W_fm, W_lift)`` for perturbation velocity.

    For any velocity perturbation ``v`` of shape ``(n_eval, 3)`` [m/s] at the
    segment force points, the integrated force and moment contribution is
    ``FM[c] = sum_q W_fm[c, q] . v[q]`` (``c = 0..2`` force [N], ``c = 3..5``
    moment [N m]) and the per-segment lift contribution is
    ``L_seg[q] = W_lift[q] . v[q]`` [N].
    """
    n_eval = seg_vec.shape[0]
    w = rho * gamma_0_eval * seg_scale
    nl = cross3(seg_normal, seg_vec)
    W_F = np.empty((3, n_eval, 3), dtype=np.float64)

    # Component a of (v - (v . n) n) x l equals v . ((l x e_a) - (n x l)_a n).
    lx, ly, lz = seg_vec[:, 0], seg_vec[:, 1], seg_vec[:, 2]
    l_x_e = (
        np.column_stack([np.zeros(n_eval), lz, -ly]),
        np.column_stack([-lz, np.zeros(n_eval), lx]),
        np.column_stack([ly, -lx, np.zeros(n_eval)]),
    )
    for a in range(3):
        W_F[a] = w[:, None] * (l_x_e[a] - nl[:, a : a + 1] * seg_normal)

    W_M = np.empty((3, n_eval, 3), dtype=np.float64)
    W_M[0] = arm[:, 1:2] * W_F[2] - arm[:, 2:3] * W_F[1]
    W_M[1] = arm[:, 2:3] * W_F[0] - arm[:, 0:1] * W_F[2]
    W_M[2] = arm[:, 0:1] * W_F[1] - arm[:, 1:2] * W_F[0]

    W_fm = np.concatenate([W_F, W_M], axis=0)
    W_lift = lift_dir[0] * W_F[0] + lift_dir[1] * W_F[1] + lift_dir[2] * W_F[2]
    return W_fm, W_lift


class _HarmonicRingSystem:
    """Precomputed real geometry and mean state of a harmonic vortex-ring lattice.

    Assembles the real bound matrix ``A_bb`` (and its real LU factors), the
    real discrete wake-row tensor ``A_bw``, the steady mean circulation
    ``Gamma_0``, and the loaded-mean-state geometry derivatives once per
    configuration and ground height. Solves each reduced frequency ``k`` via
    the Woodbury identity (or direct complex LU for tests).
    """

    def __init__(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        *,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        ground: GroundPlane | None = None,
        ref_point: np.ndarray | None = None,
        wake_dir: np.ndarray | None = None,
        wake_chords: float = 5.0,
        te_extension_fraction: float = 0.25,
        rc: np.ndarray | float | None = None,
        use_cross_surface_core: bool = True,
        use_tail: bool = True,
        te_correction: bool = True,
        fd_step_h: float | None = None,
        fd_step_theta: float | None = None,
        wake_alignment: str = "freestream",
    ) -> None:
        self._validate_inputs(condition, S_ref, b_ref, c_ref, wake_chords, wake_alignment)
        if ground is not None:
            check_ground_clearance(lattice, ground)

        self.lattice = lattice
        self.condition = condition
        self.S_ref = float(S_ref)
        self.b_ref = float(b_ref)
        self.c_ref = float(c_ref)
        self.V_inf = float(condition.V_inf)
        self.rho = float(condition.rho)
        self.q_inf = 0.5 * self.rho * self.V_inf * self.V_inf
        self.ground = ground
        self.ref_point = np.zeros(3, dtype=np.float64) if ref_point is None else np.asarray(ref_point, dtype=np.float64)
        if self.ref_point.shape != (3,) or not np.all(np.isfinite(self.ref_point)):
            raise InputError(f"ref_point must be a finite 3-vector [m], got {ref_point!r}.")

        self.wake_chords = float(wake_chords)
        self.use_tail = bool(use_tail)
        self.te_correction = bool(te_correction)
        self.use_cross_surface_core = bool(use_cross_surface_core)
        self.rc_override = rc

        # Mean-state wind axes and heave direction.
        self.d_0 = freestream_direction(float(condition.alpha), float(condition.beta))
        self.L_0 = lift_direction(float(condition.alpha))
        self.S_0 = side_direction(float(condition.alpha), float(condition.beta))
        self.V_0 = self.V_inf * self.d_0
        self.e_h = self.L_0 if ground is None else np.asarray(ground.normal, dtype=np.float64)

        if wake_dir is None:
            if wake_alignment == "body" and ground is None:
                wd_in = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                self._wake_rotates_with_alpha = False
            else:
                wd_in = self.d_0
                self._wake_rotates_with_alpha = True
        else:
            wd_in = np.asarray(wake_dir, dtype=np.float64)
            self._wake_rotates_with_alpha = (ground is not None) or (wake_alignment == "freestream")

        self.rings = build_ring_lattice(
            lattice,
            wake_dir=wd_in,
            te_extension_fraction=te_extension_fraction,
        )
        self.img: RingGroundImage | None = (
            None if ground is None else ground_image_rings(self.rings, ground, wake_dir=self.rings.wake_dir)
        )
        self.rc_ring, self.rc_wake = _ring_pair_cores(self.rings, rc, use_cross_surface_core)

        # Wake discretisation rule: strip-width-weighted mean TE panel length.
        te_idx = self.rings.te_indices
        self.dx_te = np.asarray(lattice.panel_length[te_idx], dtype=np.float64)
        self.dx_w = float(np.sum(self.rings.strip_width * self.dx_te) / np.sum(self.rings.strip_width))
        if not math.isfinite(self.dx_w) or self.dx_w <= 0.0:
            raise InputError(f"Invalid wake row length dx_w={self.dx_w!r} computed from lattice.")
        self.dt = self.dx_w / self.V_inf
        self.dx_te_ratio = self.dx_te / self.dx_w
        self.f_te = self.dx_te / (2.0 * (self.dx_te + self.dx_w))
        self.n_wake_rows = max(1, int(round(self.wake_chords * self.c_ref / self.dx_w)))
        self.wake_length = float(self.n_wake_rows * self.dx_w)

        # Far-wake tail start corners (real and ground image).
        self.tail_left = self.rings.te_left + self.wake_length * self.rings.wake_dir
        self.tail_right = self.rings.te_right + self.wake_length * self.rings.wake_dir
        if self.img is not None:
            self.img_tail_left = self.img.te_left + self.wake_length * self.img.wake_dir
            self.img_tail_right = self.img.te_right + self.wake_length * self.img.wake_dir
        else:
            self.img_tail_left = None
            self.img_tail_right = None

        # Assemble real A_bb, steady wake closure A_bw_steady, real LU of A_bb, and discrete wake tensor A_bw.
        self._assemble_system_matrices()

        # Precompute load geometry (bound segments, chordwise legs, physical ring areas, TE segments).
        self._prepare_load_geometry()

        # Solve steady mean state Gamma_0 and mean loads.
        self._solve_mean_state()

        # Precompute loaded-mean-state geometry derivatives and perturbation-velocity load functionals.
        self._prepare_loaded_terms(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)

        # Exact k -> 0 limits (heave, pitch, pitch rate).
        self.k0_heave, self.k0_pitch, self.k0_pitch_rate = self._compute_k0_limits()

    @staticmethod
    def _validate_inputs(
        condition: FlightCondition,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        wake_chords: float,
        wake_alignment: str,
    ) -> None:
        for name, val in (
            ("V_inf", condition.V_inf),
            ("rho", condition.rho),
            ("S_ref", S_ref),
            ("b_ref", b_ref),
            ("c_ref", c_ref),
            ("wake_chords", wake_chords),
        ):
            v = float(val)
            if not math.isfinite(v) or v <= 0.0:
                raise InputError(f"{name} must be finite and > 0, got {val!r}.")
        if wake_alignment not in ("freestream", "body"):
            raise InputError(f"wake_alignment must be 'freestream' or 'body', got {wake_alignment!r}.")

    def _assemble_system_matrices(self) -> None:
        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips
        te_idx = rings.te_indices

        A_bb = ring_normal_influence(rings.cp, rings.normal_bc, rings.corners, self.rc_ring)
        A_bw_steady = steady_wake_normal_influence(
            rings.cp, rings.normal_bc, rings.te_left, rings.te_right, rings.wake_dir, self.rc_wake,
        )
        if self.img is not None:
            A_bb += self.img.sign * ring_normal_influence(rings.cp, rings.normal_bc, self.img.corners, self.rc_ring)
            A_bw_steady += self.img.sign * steady_wake_normal_influence(
                rings.cp, rings.normal_bc, self.img.te_left, self.img.te_right, self.img.wake_dir, self.rc_wake,
            )

        self.A_bb = A_bb
        self.A_bw_steady = A_bw_steady
        self.A_steady = A_bb.copy()
        self.A_steady[:, te_idx] += A_bw_steady
        self.lu_Abb = sl.lu_factor(self.A_bb)
        self.lu_Asteady = sl.lu_factor(self.A_steady)

        # Assemble real wake-row influence tensor A_bw of shape (n_p, n_wake_rows, n_s).
        n_rows = self.n_wake_rows
        A_bw = np.empty((n_p, n_rows, n_s), dtype=np.float64)
        for m0 in range(0, n_rows, _WAKE_ROW_CHUNK):
            m1 = min(n_rows, m0 + _WAKE_ROW_CHUNK)
            n_chunk = m1 - m0
            rc_chunk = self._tile_wake_rc(self.rc_wake, n_chunk)
            w_corners = _build_wake_row_corners(rings.te_left, rings.te_right, rings.wake_dir, m0, m1, self.dx_w)
            blk = ring_normal_influence(rings.cp, rings.normal_bc, w_corners, rc_chunk)
            if self.img is not None:
                img_w = _build_wake_row_corners(
                    self.img.te_left, self.img.te_right, self.img.wake_dir, m0, m1, self.dx_w,
                )
                blk += self.img.sign * ring_normal_influence(rings.cp, rings.normal_bc, img_w, rc_chunk)
            A_bw[:, m0:m1, :] = blk.reshape(n_p, n_chunk, n_s)
        self.A_bw = A_bw

    @staticmethod
    def _tile_wake_rc(rc_wake: np.ndarray | float, n_chunk: int) -> np.ndarray | float:
        if np.ndim(rc_wake) == 0:
            return rc_wake
        arr = np.asarray(rc_wake, dtype=np.float64)
        if arr.ndim == 1:
            return np.tile(arr, n_chunk)
        return np.tile(arr, (1, n_chunk))

    def _prepare_load_geometry(self) -> None:
        lat = self.lattice
        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips

        self.mid = np.asarray(lat.force_points, dtype=np.float64)
        self.l_vec = np.asarray(lat.b - lat.a, dtype=np.float64)

        leg_start, leg_end = leg_segments(lat)
        self.leg_mid = 0.5 * (leg_start + leg_end)
        self.leg_vec = leg_end - leg_start
        self.leg_normal, self.leg_scale, _ = leg_force_rules(lat)
        self.n_leg_sets = self.leg_mid.shape[0] // n_p

        # Stacked Kutta-Joukowski evaluation points: bound force points followed by chordwise legs.
        self.P_load = np.vstack([self.mid, self.leg_mid])
        self.eval_panels = np.tile(np.arange(n_p, dtype=np.int64), 1 + self.n_leg_sets)
        self.eval_strips = rings.panel_strip[self.eval_panels]
        self.seg_vec_all = np.vstack([self.l_vec, self.leg_vec])
        self.seg_normal_all = np.vstack([np.zeros((n_p, 3), dtype=np.float64), self.leg_normal])
        self.seg_scale_all = np.concatenate([np.ones(n_p, dtype=np.float64), self.leg_scale])
        self.arm_all = self.P_load - self.ref_point[None, :]

        self.rc_load_ring, self.rc_load_wake = _eval_pair_cores(
            rings, self.eval_panels, self.rc_override, self.use_cross_surface_core,
        )

        # Physical trailing-edge segment for the trailing-edge boundary load correction.
        te_l_phys = np.asarray(lat.te_left, dtype=np.float64)
        te_r_phys = np.asarray(lat.te_right, dtype=np.float64)
        self.l_te = te_r_phys - te_l_phys
        self.p_te = te_l_phys + np.asarray(lat.cp_frac, dtype=np.float64)[:, None] * self.l_te
        self.arm_te = self.p_te - self.ref_point[None, :]
        self.rc_te_ring, self.rc_te_wake = _eval_pair_cores(
            rings, rings.te_indices, self.rc_override, self.use_cross_surface_core,
        )

        # Physical ring surface areas and centroids for apparent-mass loads.
        self.area_vec, self.ring_centroid = _physical_ring_areas(lat, rings)
        self.arm_ring = self.ring_centroid - self.ref_point[None, :]

        # Strip reference areas [m^2] for strip lift coefficient normalisation.
        self.strip_area = np.asarray(lat.chord, dtype=np.float64) * np.asarray(lat.width, dtype=np.float64)
        self._n_strips = n_s

    def _solve_mean_state(self) -> None:
        umap = make_unknown_map(self.lattice, False)
        rhs_0 = _vlm_rhs(self.lattice, umap, self.condition, self.ref_point)
        self.Gamma_0 = sl.lu_solve(self.lu_Asteady, rhs_0)
        self.has_mean_load = bool(np.max(np.abs(self.Gamma_0)) > 1e-15 * max(self.V_inf * self.c_ref, 1.0))

        self.gamma_hs_0 = _rings_to_horseshoe(self.Gamma_0, self.rings.n_chord)
        g_leg_base = leg_strength(self.lattice, self.gamma_hs_0)
        self.gamma_0_all = np.concatenate([self.gamma_hs_0] + [g_leg_base] * self.n_leg_sets)

        if self.has_mean_load:
            self.v_ind_load_0 = _steady_velocity_at(
                self.P_load, self.rings, self.Gamma_0, self.rc_load_ring, self.rc_load_wake,
                ground=self.ground, wake_dir=self.rings.wake_dir,
            )
            self.v_ind_te_0 = _steady_velocity_at(
                self.p_te, self.rings, self.Gamma_0, self.rc_te_ring, self.rc_te_wake,
                ground=self.ground, wake_dir=self.rings.wake_dir,
            )
        else:
            self.v_ind_load_0 = np.zeros_like(self.P_load)
            self.v_ind_te_0 = np.zeros_like(self.p_te)

        # Mean total local velocities on all KJ segments and physical TE segments.
        self.V_tot_0_all = self.V_0[None, :] + self.v_ind_load_0
        self.V_eff_0_all = (
            self.V_tot_0_all
            - np.sum(self.V_tot_0_all * self.seg_normal_all, axis=1, keepdims=True) * self.seg_normal_all
        )
        self.V_te_0 = self.V_0[None, :] + self.v_ind_te_0

        # Mean Kutta-Joukowski forces and moments (body axes).
        self.F_seg_0 = (self.rho * self.gamma_0_all * self.seg_scale_all)[:, None] * cross3(
            self.V_eff_0_all, self.seg_vec_all,
        )
        self.F_total_0 = np.sum(self.F_seg_0, axis=0)
        self.M_total_0 = np.sum(cross3(self.arm_all, self.F_seg_0), axis=0)
        self.mean_state = self._to_steady_loads(self.F_total_0, self.M_total_0)

    def _prepare_loaded_terms(
        self,
        *,
        fd_step_h: float | None,
        fd_step_theta: float | None,
    ) -> None:
        rings = self.rings
        n_p = rings.n_panels

        self.rhs_heave_geo = np.zeros(n_p, dtype=np.float64)
        self.rhs_pitch_geo = np.zeros(n_p, dtype=np.float64)
        self.dv_load_heave_geo = np.zeros_like(self.P_load)
        self.dv_load_pitch_geo = np.zeros_like(self.P_load)

        if not self.has_mean_load:
            self.W_fm = None
            self.W_lift = None
            self.FM_bound_mat = None
            self.SL_bound_mat = None
            self.FM_wake_tensor = None
            self.SL_wake_tensor = None
            self.FM_steady_wake_mat = None
            self.SL_steady_wake_mat = None
            return

        # 1. Precompute velocity-to-load linear functionals W_fm (6, n_eval, 3) and W_lift (n_eval, 3).
        self.W_fm, self.W_lift = _build_velocity_load_weights(
            self.rho, self.gamma_0_all, self.seg_vec_all, self.seg_normal_all,
            self.seg_scale_all, self.arm_all, self.L_0,
        )
        self._assemble_velocity_load_matrices()

        # 2. Central finite differences of the steady system and steady induced velocity.
        h_min = None
        if self.ground is not None:
            pts = np.vstack([self.lattice.all_points(), rings.corners.reshape(-1, 3)])
            h_min = float(np.min(self.ground.height(pts)))
            if h_min <= 0.0:
                h_min = float(np.min(self.ground.height(self.lattice.all_points())))

        # Heave loaded-mean-state terms (ground effect only; zero in free air).
        if self.ground is not None:
            eps_h = (
                _DEFAULT_FD_STEP_H_FRAC * min(h_min, self.c_ref)
                if fd_step_h is None
                else float(fd_step_h)
            )
            if not math.isfinite(eps_h) or eps_h <= 0.0 or eps_h >= 0.5 * h_min:
                raise InputError(f"fd_step_h must satisfy 0 < fd_step_h < 0.5 * h_min, got {eps_h!r}.")
            g_plus = GroundPlane(normal=self.ground.normal, offset=self.ground.offset - eps_h)
            g_minus = GroundPlane(normal=self.ground.normal, offset=self.ground.offset + eps_h)
            dA_dh_G0 = (self._image_steady_matvec(g_plus) - self._image_steady_matvec(g_minus)) / (2.0 * eps_h)
            self.rhs_heave_geo = -self.c_ref * dA_dh_G0

            vp = _steady_velocity_at(
                self.P_load, rings, self.Gamma_0, self.rc_load_ring, self.rc_load_wake,
                ground=g_plus, wake_dir=rings.wake_dir, image_only=True,
            )
            vm = _steady_velocity_at(
                self.P_load, rings, self.Gamma_0, self.rc_load_ring, self.rc_load_wake,
                ground=g_minus, wake_dir=rings.wake_dir, image_only=True,
            )
            self.dv_load_heave_geo = self.c_ref * (vp - vm) / (2.0 * eps_h)

        # Pitch loaded-mean-state terms (ground effect and/or free-stream-aligned wake).
        if self.ground is not None or self._wake_rotates_with_alpha:
            eps_th = _DEFAULT_FD_STEP_THETA if fd_step_theta is None else float(fd_step_theta)
            if self.ground is not None and fd_step_theta is None:
                max_lever = max(float(np.max(np.linalg.norm(self.lattice.all_points() - self.ref_point, axis=1))), 1e-6)
                eps_th = min(eps_th, 0.01 * h_min / max_lever)
            if not math.isfinite(eps_th) or eps_th <= 0.0:
                raise InputError(f"fd_step_theta must be finite and > 0, got {eps_th!r}.")

            g_tp, wd_tp = self._pitched_steady_state(+eps_th)
            g_tm, wd_tm = self._pitched_steady_state(-eps_th)
            dA_dth_G0 = (
                self._varying_steady_matvec(g_tp, wd_tp) - self._varying_steady_matvec(g_tm, wd_tm)
            ) / (2.0 * eps_th)
            self.rhs_pitch_geo = -dA_dth_G0

            vp_th = self._varying_steady_load_vel(g_tp, wd_tp)
            vm_th = self._varying_steady_load_vel(g_tm, wd_tm)
            self.dv_load_pitch_geo = (vp_th - vm_th) / (2.0 * eps_th)

    def _pitched_steady_state(self, d_theta: float) -> tuple[GroundPlane | None, np.ndarray]:
        """Return ``(ground, wake_dir)`` in body axes for aircraft pitch perturbation *d_theta* [rad]."""
        wd = _rot_y(self.rings.wake_dir, -d_theta) if self._wake_rotates_with_alpha else self.rings.wake_dir
        if self.ground is None:
            return None, wd
        k_new = _rot_y(self.ground.normal, -d_theta)
        h_ref = float(self.ref_point @ self.ground.normal - self.ground.offset)
        return GroundPlane(normal=k_new, offset=float(self.ref_point @ k_new - h_ref)), wd

    def _image_steady_matvec(self, ground: GroundPlane) -> np.ndarray:
        """Return ``A_img_steady @ Gamma_0`` for *ground* (real rings unchanged in heave)."""
        rings = self.rings
        img = ground_image_rings(rings, ground, wake_dir=rings.wake_dir)
        v = img.sign * (
            ring_normal_influence(rings.cp, rings.normal_bc, img.corners, self.rc_ring) @ self.Gamma_0
        )
        v += img.sign * (
            steady_wake_normal_influence(
                rings.cp, rings.normal_bc, img.te_left, img.te_right, img.wake_dir, self.rc_wake,
            ) @ self.Gamma_0[rings.te_indices]
        )
        return v

    def _varying_steady_matvec(self, ground: GroundPlane | None, wd: np.ndarray) -> np.ndarray:
        """Return ``(A_bw_real(wd) + A_img_steady(ground, wd)) @ Gamma_0`` (real bound rings invariant)."""
        rings = self.rings
        g_te = self.Gamma_0[rings.te_indices]
        v = np.zeros(rings.n_panels, dtype=np.float64)
        if self._wake_rotates_with_alpha:
            v += steady_wake_normal_influence(
                rings.cp, rings.normal_bc, rings.te_left, rings.te_right, wd, self.rc_wake,
            ) @ g_te
        if ground is not None:
            img = ground_image_rings(rings, ground, wake_dir=wd)
            v += img.sign * (
                ring_normal_influence(rings.cp, rings.normal_bc, img.corners, self.rc_ring) @ self.Gamma_0
            )
            v += img.sign * (
                steady_wake_normal_influence(
                    rings.cp, rings.normal_bc, img.te_left, img.te_right, img.wake_dir, self.rc_wake,
                ) @ g_te
            )
        return v

    def _varying_steady_load_vel(self, ground: GroundPlane | None, wd: np.ndarray) -> np.ndarray:
        """Return load-point velocity from real steady wake and ground images for pitch FD."""
        rings = self.rings
        g_te = self.Gamma_0[rings.te_indices]
        img = None if ground is None else ground_image_rings(rings, ground, wake_dir=wd)
        m = self.P_load.shape[0]
        v_out = np.zeros((m, 3), dtype=np.float64)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc = self.P_load[i0:i1]
            rc_r = _slice_rc_rows(self.rc_load_ring, i0, i1)
            rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
            if self._wake_rotates_with_alpha:
                vw = steady_wake_velocity(Pc, rings.te_left, rings.te_right, wd, rc_w)
                v_out[i0:i1] += np.einsum("msi,s->mi", vw, g_te)
            if img is not None:
                vr_i = ring_velocity(Pc, img.corners, rc_r)
                vw_i = steady_wake_velocity(Pc, img.te_left, img.te_right, img.wake_dir, rc_w)
                v_out[i0:i1] += img.sign * np.einsum("mni,n->mi", vr_i, self.Gamma_0)
                v_out[i0:i1] += img.sign * np.einsum("msi,s->mi", vw_i, g_te)
        return v_out

    def _contract_load_vel(
        self,
        v_chunk: np.ndarray,
        i0: int,
        i1: int,
        fm_out: np.ndarray,
        sl_out: np.ndarray,
    ) -> None:
        """Accumulate velocity chunk ``(chunk_m, n_src, 3)`` into ``fm_out (6, n_src)`` and ``sl_out (n_s, n_src)``."""
        fm_out += np.einsum("cmi,mni->cn", self.W_fm[:, i0:i1, :], v_chunk)
        seg_lift = np.einsum("mi,mni->mn", self.W_lift[i0:i1, :], v_chunk)
        np.add.at(sl_out, self.eval_strips[i0:i1], seg_lift)

    def _assemble_velocity_load_matrices(self) -> None:
        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips
        n_rows = self.n_wake_rows
        m = self.P_load.shape[0]

        FM_bound = np.zeros((6, n_p), dtype=np.float64)
        SL_bound = np.zeros((n_s, n_p), dtype=np.float64)
        FM_sw = np.zeros((6, n_s), dtype=np.float64)
        SL_sw = np.zeros((n_s, n_s), dtype=np.float64)

        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc = self.P_load[i0:i1]
            rc_r = _slice_rc_rows(self.rc_load_ring, i0, i1)
            rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)

            vr = ring_velocity(Pc, rings.corners, rc_r)
            vw = steady_wake_velocity(Pc, rings.te_left, rings.te_right, rings.wake_dir, rc_w)
            if self.img is not None:
                vr += self.img.sign * ring_velocity(Pc, self.img.corners, rc_r)
                vw += self.img.sign * steady_wake_velocity(
                    Pc, self.img.te_left, self.img.te_right, self.img.wake_dir, rc_w,
                )
            self._contract_load_vel(vr, i0, i1, FM_bound, SL_bound)
            self._contract_load_vel(vw, i0, i1, FM_sw, SL_sw)

        self.FM_bound_mat = FM_bound
        self.SL_bound_mat = SL_bound
        self.FM_steady_wake_mat = FM_sw
        self.SL_steady_wake_mat = SL_sw

        # Discrete wake-row contribution to integrated loads (6, n_rows, n_s) and strip lift (n_s, n_rows, n_s).
        FM_wake = np.zeros((6, n_rows, n_s), dtype=np.float64)
        SL_wake = np.zeros((n_s, n_rows, n_s), dtype=np.float64)
        for m0 in range(0, n_rows, _WAKE_ROW_CHUNK):
            m1 = min(n_rows, m0 + _WAKE_ROW_CHUNK)
            n_chunk = m1 - m0
            w_corners = _build_wake_row_corners(rings.te_left, rings.te_right, rings.wake_dir, m0, m1, self.dx_w)
            img_w = (
                None
                if self.img is None
                else _build_wake_row_corners(
                    self.img.te_left, self.img.te_right, self.img.wake_dir, m0, m1, self.dx_w,
                )
            )
            fm_blk = np.zeros((6, n_chunk * n_s), dtype=np.float64)
            sl_blk = np.zeros((n_s, n_chunk * n_s), dtype=np.float64)
            for i0 in range(0, m, _LOAD_POINT_CHUNK):
                i1 = min(m, i0 + _LOAD_POINT_CHUNK)
                Pc = self.P_load[i0:i1]
                rc_w = self._tile_wake_rc(_slice_rc_rows(self.rc_load_wake, i0, i1), n_chunk)
                vw = ring_velocity(Pc, w_corners, rc_w)
                if img_w is not None:
                    vw += self.img.sign * ring_velocity(Pc, img_w, rc_w)
                self._contract_load_vel(vw, i0, i1, fm_blk, sl_blk)
            FM_wake[:, m0:m1, :] = fm_blk.reshape(6, n_chunk, n_s)
            SL_wake[:, m0:m1, :] = sl_blk.reshape(n_s, n_chunk, n_s)

        self.FM_wake_tensor = FM_wake
        self.SL_wake_tensor = SL_wake

    def _tail_load_matrices(self, kappa: float, phase0: complex) -> tuple[np.ndarray, np.ndarray]:
        """Return complex tail perturbation-velocity load matrices ``(FM_tail, SL_tail)``."""
        n_s = self.rings.n_strips
        m = self.P_load.shape[0]
        FM_tail = np.zeros((6, n_s), dtype=np.complex128)
        SL_tail = np.zeros((n_s, n_s), dtype=np.complex128)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc = self.P_load[i0:i1]
            rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
            vt = harmonic_tail_velocity(Pc, self.tail_left, self.tail_right, self.rings.wake_dir, kappa, rc_w)
            if self.img is not None:
                vt += self.img.sign * harmonic_tail_velocity(
                    Pc, self.img_tail_left, self.img_tail_right, self.img.wake_dir, kappa, rc_w,
                )
            self._contract_load_vel(vt, i0, i1, FM_tail, SL_tail)
        FM_tail *= phase0
        SL_tail *= phase0
        return FM_tail, SL_tail

    def wake_matrix(self, k: float, *, use_tail: bool | None = None) -> np.ndarray:
        """Return complex wake normal-influence matrix ``W(k)`` [1/m], shape ``(n_panels, n_strips)``."""
        tail_on = self.use_tail if use_tail is None else bool(use_tail)
        if k == 0.0 and tail_on:
            return self.A_bw_steady.astype(np.complex128, copy=True)

        omega = 2.0 * k * self.V_inf / self.c_ref
        kappa = omega / self.V_inf
        lags = np.exp(-1j * omega * np.arange(1, self.n_wake_rows + 1, dtype=np.float64) * self.dt)
        W = np.tensordot(self.A_bw, lags, axes=([1], [0]))

        if tail_on:
            phase0 = np.exp(-1j * omega * (self.n_wake_rows + 0.5) * self.dt)
            W_t = harmonic_tail_normal_influence(
                self.rings.cp, self.rings.normal_bc,
                self.tail_left, self.tail_right, self.rings.wake_dir, kappa, self.rc_wake,
            )
            if self.img is not None:
                W_t += self.img.sign * harmonic_tail_normal_influence(
                    self.rings.cp, self.rings.normal_bc,
                    self.img_tail_left, self.img_tail_right, self.img.wake_dir, kappa, self.rc_wake,
                )
            W = W + phase0 * W_t
        return W

    def build_rhs(self, k: float) -> np.ndarray:
        """Return complex RHS columns ``[rhs_heave, rhs_pitch]`` [m/s], shape ``(n_panels, 2)``."""
        omega = 2.0 * k * self.V_inf / self.c_ref
        rings = self.rings
        n_bc = rings.normal_bc

        # Heave: displacement c_ref * e_h, velocity v_surf = i omega c_ref e_h.
        rhs_h = (1j * omega * self.c_ref) * (n_bc @ self.e_h) + self.rhs_heave_geo

        # Pitch: incidence change delta_U_alpha = -V_inf (e_y x d_0) plus rotation -i omega (e_y x (x - x_ref)).
        dU_alpha = -_ey_cross(self.V_0)
        rel_cp = rings.cp - self.ref_point[None, :]
        dU_rot = -(1j * omega) * _ey_cross(rel_cp)
        rhs_p = -(n_bc @ dU_alpha) - np.sum(n_bc * dU_rot, axis=1) + self.rhs_pitch_geo

        return np.column_stack([rhs_h, rhs_p]).astype(np.complex128)

    def _solve_linear(self, W: np.ndarray, rhs: np.ndarray, solve_method: str) -> np.ndarray:
        """Solve ``(A_bb + W E_te^T) Gamma = rhs`` by Woodbury or direct complex LU."""
        te_idx = self.rings.te_indices
        n_s = self.rings.n_strips
        if solve_method == "woodbury":
            Y = sl.lu_solve(self.lu_Abb, rhs.real) + 1j * sl.lu_solve(self.lu_Abb, rhs.imag)
            Z = sl.lu_solve(self.lu_Abb, W.real) + 1j * sl.lu_solve(self.lu_Abb, W.imag)
            C_cap = np.eye(n_s, dtype=np.complex128) + Z[te_idx, :]
            c_sol = sl.solve(C_cap, Y[te_idx, :])
            return Y - Z @ c_sol
        if solve_method == "direct":
            A_sys = self.A_bb.astype(np.complex128, copy=True)
            A_sys[:, te_idx] += W
            return sl.solve(A_sys, rhs)
        raise InputError(f"solve_method must be 'woodbury' or 'direct', got {solve_method!r}.")

    def solve_circulations(
        self,
        k: float,
        *,
        solve_method: str = "woodbury",
        use_tail: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rhs, Gamma)`` of shape ``(n_panels, 2)`` for heave and pitch at *k*."""
        W = self.wake_matrix(k, use_tail=use_tail)
        rhs = self.build_rhs(k)
        Gamma = self._solve_linear(W, rhs, solve_method)
        return rhs, Gamma

    def _evaluate_linearised_loads(
        self,
        Gamma_hat: np.ndarray,
        *,
        k: float,
        dU_load: np.ndarray,
        dv_geo_load: np.ndarray,
        rotate_mean_geometry: bool,
        FM_wake_k: np.ndarray | None,
        SL_wake_k: np.ndarray | None,
        include_apparent_mass: bool,
        include_te_correction: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute complex ``([CL, CD, CY, Cl, Cm, Cn], strip_cl)`` for one motion."""
        rings = self.rings
        omega = 2.0 * k * self.V_inf / self.c_ref
        n_s = self._n_strips

        # 1. Perturbation circulation times mean local velocity (bound + chordwise legs).
        g_hs = _rings_to_horseshoe(Gamma_hat, rings.n_chord)
        g_leg_base = leg_strength(self.lattice, g_hs)
        g_all = np.concatenate([g_hs] + [g_leg_base] * self.n_leg_sets)
        F_seg_dG = (self.rho * g_all * self.seg_scale_all)[:, None] * cross3(self.V_eff_0_all, self.seg_vec_all)
        F_tot = np.sum(F_seg_dG, axis=0).astype(np.complex128)
        M_tot = np.sum(_cross3_c(self.arm_all, F_seg_dG), axis=0).astype(np.complex128)
        strip_lift = np.zeros(n_s, dtype=np.complex128)
        np.add.at(strip_lift, self.eval_strips, F_seg_dG @ self.L_0)

        # 2. Mean circulation times perturbation velocity (motion onset + geometry FD + perturbation induced).
        if self.has_mean_load:
            dv_direct = dU_load + dv_geo_load
            FM_dV = np.einsum("cmi,mi->c", self.W_fm, dv_direct).astype(np.complex128)
            FM_dV += self.FM_bound_mat @ Gamma_hat
            if FM_wake_k is not None:
                FM_dV += FM_wake_k @ Gamma_hat[rings.te_indices]
            F_tot += FM_dV[:3]
            M_tot += FM_dV[3:]

            sl_direct = np.einsum("mi,mi->m", self.W_lift, dv_direct).astype(np.complex128)
            np.add.at(strip_lift, self.eval_strips, sl_direct)
            strip_lift += self.SL_bound_mat @ Gamma_hat
            if SL_wake_k is not None:
                strip_lift += SL_wake_k @ Gamma_hat[rings.te_indices]

        # 3. Moved geometry of the mean load in pitch (rotation of mean force and moment arm about ref_point).
        if rotate_mean_geometry and self.has_mean_load:
            F_rot = _ey_cross(self.F_total_0)
            M_rot = _ey_cross(self.M_total_0)
            F_tot += F_rot
            M_tot += M_rot
            # Per-strip rotation of mean force: sum_{q in strip} (e_y x F_seg_0[q]) . L_0.
            np.add.at(strip_lift, self.eval_strips, _ey_cross(self.F_seg_0) @ self.L_0)

        # 4. Apparent-mass load: rho * (i omega Gamma_hat) * physical ring area vector.
        if include_apparent_mass and omega != 0.0:
            F_am = (self.rho * 1j * omega * Gamma_hat)[:, None] * self.area_vec
            F_tot += np.sum(F_am, axis=0)
            M_tot += np.sum(_cross3_c(self.arm_ring, F_am), axis=0)
            np.add.at(strip_lift, rings.panel_strip, F_am @ self.L_0)

        # 5. Trailing-edge boundary load correction: Delta gamma_TE,s = f_TE,s (Gamma_w1,s - Gamma_TE,s).
        if include_te_correction and omega != 0.0:
            g_te = Gamma_hat[rings.te_indices]
            dg_te = self.f_te * g_te * (np.exp(-1j * omega * self.dt) - 1.0)
            F_te = (self.rho * dg_te)[:, None] * cross3(self.V_te_0, self.l_te)
            F_tot += np.sum(F_te, axis=0)
            M_tot += np.sum(_cross3_c(self.arm_te, F_te), axis=0)
            strip_lift += F_te @ self.L_0

        qS = self.q_inf * self.S_ref
        coeffs = np.array(
            [
                (F_tot @ self.L_0) / qS,
                (F_tot @ self.d_0) / qS,
                (F_tot @ self.S_0) / qS,
                -M_tot[0] / (qS * self.b_ref),
                M_tot[1] / (qS * self.c_ref),
                -M_tot[2] / (qS * self.b_ref),
            ],
            dtype=np.complex128,
        )
        strip_cl = strip_lift / (self.q_inf * self.strip_area)
        return coeffs, strip_cl

    def _to_steady_loads(self, F_tot: np.ndarray, M_tot: np.ndarray) -> _SteadyLoads:
        qS = self.q_inf * self.S_ref
        return _SteadyLoads(
            CL=float((F_tot @ self.L_0) / qS),
            CD=float((F_tot @ self.d_0) / qS),
            CY=float((F_tot @ self.S_0) / qS),
            Cl=float(-M_tot[0] / (qS * self.b_ref)),
            Cm=float(M_tot[1] / (qS * self.c_ref)),
            Cn=float(-M_tot[2] / (qS * self.b_ref)),
        )

    def _compute_k0_limits(self) -> tuple[_SteadyLoads, _SteadyLoads, _SteadyLoads]:
        """Compute exact real ``k -> 0`` limits for heave, pitch and pitch rate."""
        rings = self.rings
        n_bc = rings.normal_bc
        rel_cp = rings.cp - self.ref_point[None, :]

        # Right-hand sides at k = 0: heave geometry, pitch incidence + geometry, unit q_hat = q c/(2V) = 1.
        dU_alpha = -_ey_cross(self.V_0)
        q_dim = 2.0 * self.V_inf / self.c_ref
        dU_q_cp = -q_dim * _ey_cross(rel_cp)

        rhs_h0 = self.rhs_heave_geo
        rhs_p0 = -(n_bc @ dU_alpha) + self.rhs_pitch_geo
        rhs_q0 = -np.sum(n_bc * dU_q_cp, axis=1)

        G_0s = sl.lu_solve(self.lu_Asteady, np.column_stack([rhs_h0, rhs_p0, rhs_q0]))
        G_h0, G_p0, G_q0 = G_0s[:, 0], G_0s[:, 1], G_0s[:, 2]

        zero_v = np.zeros_like(self.P_load)
        dU_alpha_load = np.broadcast_to(dU_alpha[None, :], self.P_load.shape)
        dU_q_load = -q_dim * _ey_cross(self.arm_all)

        c_h0, _ = self._evaluate_linearised_loads(
            G_h0, k=0.0, dU_load=zero_v, dv_geo_load=self.dv_load_heave_geo,
            rotate_mean_geometry=False, FM_wake_k=self.FM_steady_wake_mat,
            SL_wake_k=self.SL_steady_wake_mat, include_apparent_mass=False, include_te_correction=False,
        )
        c_p0, _ = self._evaluate_linearised_loads(
            G_p0, k=0.0, dU_load=dU_alpha_load, dv_geo_load=self.dv_load_pitch_geo,
            rotate_mean_geometry=True, FM_wake_k=self.FM_steady_wake_mat,
            SL_wake_k=self.SL_steady_wake_mat, include_apparent_mass=False, include_te_correction=False,
        )
        c_q0, _ = self._evaluate_linearised_loads(
            G_q0, k=0.0, dU_load=dU_q_load, dv_geo_load=zero_v,
            rotate_mean_geometry=False, FM_wake_k=self.FM_steady_wake_mat,
            SL_wake_k=self.SL_steady_wake_mat, include_apparent_mass=False, include_te_correction=False,
        )
        return (
            _SteadyLoads(*(float(x.real) for x in c_h0)),
            _SteadyLoads(*(float(x.real) for x in c_p0)),
            _SteadyLoads(*(float(x.real) for x in c_q0)),
        )

    def solve(
        self,
        k_values: Sequence[float] | np.ndarray,
        *,
        solve_method: str = "woodbury",
        use_tail: bool | None = None,
        te_correction: bool | None = None,
    ) -> _HarmonicResponse:
        """Solve heave and pitch harmonic responses for all reduced frequencies *k_values*."""
        k_arr = np.atleast_1d(np.asarray(k_values, dtype=np.float64))
        if k_arr.ndim != 1 or k_arr.size == 0:
            raise InputError("k_values must be a non-empty 1-D sequence of reduced frequencies.")
        if not np.all(np.isfinite(k_arr)) or np.any(k_arr < 0.0):
            raise InputError(f"All reduced frequencies k must be finite and >= 0, got {k_values!r}.")
        if solve_method not in ("woodbury", "direct"):
            raise InputError(f"solve_method must be 'woodbury' or 'direct', got {solve_method!r}.")

        tail_on = self.use_tail if use_tail is None else bool(use_tail)
        te_on = self.te_correction if te_correction is None else bool(te_correction)
        n_k = k_arr.size
        n_s = self._n_strips

        heave_c = np.empty((n_k, 6), dtype=np.complex128)
        pitch_c = np.empty((n_k, 6), dtype=np.complex128)
        heave_scl = np.empty((n_k, n_s), dtype=np.complex128)
        pitch_scl = np.empty((n_k, n_s), dtype=np.complex128)

        dU_alpha_load = np.broadcast_to((-_ey_cross(self.V_0))[None, :], self.P_load.shape)
        ey_arm_load = _ey_cross(self.arm_all)

        for ik, k_val in enumerate(k_arr):
            kf = float(k_val)
            omega = 2.0 * kf * self.V_inf / self.c_ref
            kappa = omega / self.V_inf
            _, G_mat = self.solve_circulations(kf, solve_method=solve_method, use_tail=tail_on)
            G_heave, G_pitch = G_mat[:, 0], G_mat[:, 1]

            FM_wake_k = None
            SL_wake_k = None
            if self.has_mean_load:
                if kf == 0.0 and tail_on:
                    FM_wake_k = self.FM_steady_wake_mat.astype(np.complex128, copy=False)
                    SL_wake_k = self.SL_steady_wake_mat.astype(np.complex128, copy=False)
                else:
                    lags = np.exp(-1j * omega * np.arange(1, self.n_wake_rows + 1, dtype=np.float64) * self.dt)
                    FM_wake_k = np.tensordot(self.FM_wake_tensor, lags, axes=([1], [0]))
                    SL_wake_k = np.tensordot(self.SL_wake_tensor, lags, axes=([1], [0]))
                    if tail_on:
                        phase0 = complex(np.exp(-1j * omega * (self.n_wake_rows + 0.5) * self.dt))
                        fm_t, sl_t = self._tail_load_matrices(kappa, phase0)
                        FM_wake_k = FM_wake_k + fm_t
                        SL_wake_k = SL_wake_k + sl_t

            dU_heave_load = np.broadcast_to(
                ((-1j * omega * self.c_ref) * self.e_h)[None, :], self.P_load.shape,
            )
            dU_pitch_load = dU_alpha_load - (1j * omega) * ey_arm_load

            heave_c[ik], heave_scl[ik] = self._evaluate_linearised_loads(
                G_heave, k=kf, dU_load=dU_heave_load, dv_geo_load=self.dv_load_heave_geo,
                rotate_mean_geometry=False, FM_wake_k=FM_wake_k, SL_wake_k=SL_wake_k,
                include_apparent_mass=True, include_te_correction=te_on,
            )
            pitch_c[ik], pitch_scl[ik] = self._evaluate_linearised_loads(
                G_pitch, k=kf, dU_load=dU_pitch_load, dv_geo_load=self.dv_load_pitch_geo,
                rotate_mean_geometry=True, FM_wake_k=FM_wake_k, SL_wake_k=SL_wake_k,
                include_apparent_mass=True, include_te_correction=te_on,
            )

        return _HarmonicResponse(
            k=k_arr,
            heave=_HarmonicMotionResponse(
                CL=heave_c[:, 0], CD=heave_c[:, 1], CY=heave_c[:, 2],
                Cl=heave_c[:, 3], Cm=heave_c[:, 4], Cn=heave_c[:, 5],
                strip_cl=heave_scl,
            ),
            pitch=_HarmonicMotionResponse(
                CL=pitch_c[:, 0], CD=pitch_c[:, 1], CY=pitch_c[:, 2],
                Cl=pitch_c[:, 3], Cm=pitch_c[:, 4], Cn=pitch_c[:, 5],
                strip_cl=pitch_scl,
            ),
            k0_heave=self.k0_heave,
            k0_pitch=self.k0_pitch,
            k0_pitch_rate=self.k0_pitch_rate,
            mean_state=self.mean_state,
            dt=self.dt,
            dx_w=self.dx_w,
            dx_te_ratio=self.dx_te_ratio.copy(),
            wake_chords=self.wake_chords,
            wake_length=self.wake_length,
            n_wake_rows=self.n_wake_rows,
            n_panels=self.rings.n_panels,
            n_strips=self.rings.n_strips,
            n_chord=self.rings.n_chord,
            use_tail=tail_on,
            te_correction=te_on,
            c_ref=self.c_ref,
            b_ref=self.b_ref,
            S_ref=self.S_ref,
        )


def _solve_harmonic_rings(
    lattice: VortexLattice,
    condition: FlightCondition,
    k_values: Sequence[float] | np.ndarray,
    *,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ground: GroundPlane | None = None,
    ref_point: np.ndarray | None = None,
    wake_dir: np.ndarray | None = None,
    wake_chords: float = 5.0,
    te_extension_fraction: float = 0.25,
    rc: np.ndarray | float | None = None,
    use_cross_surface_core: bool = True,
    use_tail: bool = True,
    te_correction: bool = True,
    solve_method: str = "woodbury",
    fd_step_h: float | None = None,
    fd_step_theta: float | None = None,
    wake_alignment: str = "freestream",
) -> _HarmonicResponse:
    """Solve small-amplitude harmonic heave and pitch on a vortex-ring lattice.

    Parameters
    ----------
    lattice : VortexLattice
        Built vortex-lattice mesh (``collocation="vlm"``).
    condition : FlightCondition
        Steady mean flight state (``V_inf`` [m/s], ``rho`` [kg/m^3], ``alpha``,
        ``beta`` [rad]).
    k_values : sequence of float
        Reduced frequencies ``k = omega c_ref / (2 V_inf)`` [-], all finite
        and ``>= 0``.
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ground : GroundPlane or None, optional
        Flat ground plane in geometry axes, or None for free air.
    ref_point : numpy.ndarray or None, optional
        Moment reference and pitch pivot point [m], shape ``(3,)``. None uses
        the origin.
    wake_dir : numpy.ndarray or None, optional
        Unit wake direction [-], shape ``(3,)`` or ``(n_strips, 3)``. None uses
        the free stream (or body x-axis in free air when
        ``wake_alignment="body"``).
    wake_chords : float, optional
        Discrete wake length in reference chords [-]. Default ``5.0``.
    te_extension_fraction : float, optional
        Trailing-edge ring closure distance aft of the trailing edge as a
        fraction of the trailing-edge panel length [-]. Default ``0.25``.
    rc : numpy.ndarray or float or None, optional
        Core radius override [m]. None uses ``lattice.rc`` (with cross-surface
        target core when *use_cross_surface_core* is True).
    use_cross_surface_core : bool, optional
        Add cross-surface target cores between different core groups when *rc*
        is None. Default True.
    use_tail : bool, optional
        Append the closed-form 3D semi-infinite harmonic far-wake tail after
        the discrete wake rows. Default True.
    te_correction : bool, optional
        Apply the trailing-edge bound-vorticity load correction. Default True.
    solve_method : {'woodbury', 'direct'}, optional
        Linear solve path for each reduced frequency. Default ``'woodbury'``.
    fd_step_h : float or None, optional
        Override central finite-difference step [m] for loaded-mean-state heave
        geometry derivatives. None uses ``1e-5 * min(h_min, c_ref)``.
    fd_step_theta : float or None, optional
        Override central finite-difference step [rad] for loaded-mean-state
        pitch geometry derivatives. None uses ``1e-5`` rad (clamped near
        ground).
    wake_alignment : {'freestream', 'body'}, optional
        Steady wake alignment rule when *wake_dir* is None. Default
        ``'freestream'``.

    Returns
    -------
    _HarmonicResponse
        Complex heave and pitch load responses, exact ``k -> 0`` limits, mean
        state and wake discretisation metadata.

    Raises
    ------
    InputError
        If *k_values*, *wake_chords*, *solve_method* or the lattice is invalid.
    GroundStrikeError
        If the geometry touches or crosses *ground*.
    """
    # Validate k_values and solve_method upfront before assembling matrices.
    k_arr = np.atleast_1d(np.asarray(k_values, dtype=np.float64))
    if k_arr.ndim != 1 or k_arr.size == 0:
        raise InputError("k_values must be a non-empty 1-D sequence of reduced frequencies.")
    if not np.all(np.isfinite(k_arr)) or np.any(k_arr < 0.0):
        raise InputError(f"All reduced frequencies k must be finite and >= 0, got {k_values!r}.")
    if solve_method not in ("woodbury", "direct"):
        raise InputError(f"solve_method must be 'woodbury' or 'direct', got {solve_method!r}.")

    system = _HarmonicRingSystem(
        lattice,
        condition,
        S_ref=S_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        ground=ground,
        ref_point=ref_point,
        wake_dir=wake_dir,
        wake_chords=wake_chords,
        te_extension_fraction=te_extension_fraction,
        rc=rc,
        use_cross_surface_core=use_cross_surface_core,
        use_tail=use_tail,
        te_correction=te_correction,
        fd_step_h=fd_step_h,
        fd_step_theta=fd_step_theta,
        wake_alignment=wake_alignment,
    )
    return system.solve(k_arr, solve_method=solve_method, use_tail=use_tail, te_correction=te_correction)
