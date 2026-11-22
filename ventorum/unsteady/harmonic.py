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
   up about ``ref_point`` [m]. With a lifting mean state ``Gamma_0``, the
   loaded-mean-state terms combine the rigid ``k = 0`` geometry derivative
   (ground images and free-stream wake rotation) with the travelling-wave
   displacement of the steady wake (discrete wake rows and semi-infinite far
   wake) matching the linearised shedding geometry of
   :mod:`ventorum.unsteady.timedomain` (Parenteau 2021, chapter 4; Katz and
   Plotkin 2001, chapter 13).

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
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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
from ventorum.aero.vortex import segment_velocity
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
from ventorum.unsteady.tail import (
    displaced_steady_tail_normal_influence,
    displaced_steady_tail_velocity,
    harmonic_tail_normal_influence,
    harmonic_tail_velocity,
)
from ventorum.utils.vec import cross3

# Default wake-row chunk size for batched ring kernel calls.
_WAKE_ROW_CHUNK = 16
# Default evaluation-point chunk size when contracting velocity tensors into load functionals.
_LOAD_POINT_CHUNK = 256
# Default central finite-difference step for pitch geometry derivatives [rad].
_DEFAULT_FD_STEP_THETA = 1.0e-5
# Default central finite-difference step fraction for heave geometry derivatives [-].
_DEFAULT_FD_STEP_H_FRAC = 1.0e-5
# Cap on temporary GPU batched linear-solve tensors [B].
_GPU_SOLVE_BATCH_BYTES = 128 * 1024 ** 2

#: Result note added when the GPU runs out of memory twice and the harmonic solve falls back to the CPU.
NOTE_GPU_OUT_OF_MEMORY = (
    "GPU out of memory: the GPU ran out of memory twice, so this case was solved on the CPU."
)


def _note_gpu_unavailable(reason: str) -> str:
    """Return the CPU fallback note when a GPU solve was requested without CUDA."""
    detail = str(reason).strip()
    if detail and not detail.endswith("."):
        detail += "."
    return f"GPU unavailable: {detail} This case was solved on the CPU."


def _select_harmonic_device(
    n_p: int,
    n_rows: int,
    n_s: int,
    has_ground: bool,
) -> tuple[str, str, list[str]]:
    """Return ``(device, precision, fallback_notes)`` for the harmonic ring solver."""
    from ventorum import gpu
    from ventorum.utils.parallel import _local as _par_local

    setting = gpu.get_device()
    if setting == "cpu":
        return "cpu", "float64", []
    if setting == "auto" and getattr(_par_local, "worker_threads", None) is not None:
        return "cpu", "float64", []
    if setting == "gpu":
        if gpu.available():
            return "gpu", gpu.get_precision(), []
        reason = gpu.unavailable_reason() or "no CUDA GPU is available."
        return "cpu", "float64", [_note_gpu_unavailable(reason)]
    # setting == "auto"
    if not gpu.available():
        return "cpu", "float64", []
    work = float(n_p * n_p + n_p * n_rows * n_s) * (2.0 if has_ground else 1.0)
    use, _ = gpu.use_gpu(work, "vlm", n_cases=1, n_panels=None, return_reason=True)
    if use:
        return "gpu", gpu.get_precision(), []
    return "cpu", "float64", []


class _HarmonicGPUState:
    """Registered container for CUDA arrays held by one :class:`_HarmonicRingSystem`."""

    def __init__(self, precision: str) -> None:
        from ventorum.gpu import ring_kernels as RK

        self.precision = precision
        self.tensors: dict[str, Any] = {}
        RK.register_device_holder(self)

    def device_bytes(self) -> int:
        """Return total CUDA storage bytes held by this state [B]."""
        from ventorum.gpu import memory as gpu_memory

        return gpu_memory._tensor_bytes(self.tensors, set())

    def release_gpu(self) -> int:
        """Drop all CUDA tensor references and return freed bytes [B]."""
        freed = self.device_bytes()
        self.tensors.clear()
        return freed


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
    device : str
        Execution device: ``"cpu"`` or ``"gpu"``.
    precision : str
        Kernel arithmetic precision: ``"float64"`` or ``"float32"``.
    notes : list of str
        Device fallback notes (for example out-of-memory or unavailable CUDA).
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
    device: str = "cpu"
    precision: str = "float64"
    notes: list[str] = field(default_factory=list)

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


def _physical_ring_areas(
    lattice: VortexLattice,
    rings: RingLattice,
    ref_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return physical surface area vectors [m^2], centroids [m] and optional moment vectors [m^3].

    Each interior ring ``p`` spans from its own bound segment ``(a[p], b[p])``
    to the next bound segment ``(a[p + 1], b[p + 1])``. Each trailing-edge ring
    spans from ``(a[p], b[p])`` to the physical trailing edge
    ``(te_left[s], te_right[s])`` so the apparent-mass integral covers exactly
    the physical chord (length ``0.75 dx_TE`` on the last panel).

    When *rings* carries bent side polylines (camber or flap hinge nodes), each
    bent ring is triangulated across its intermediate physical nodes so the
    apparent-mass force and moment integrate over the bent surface rather than
    cutting straight across the camber or hinge. When no ring has intermediate
    side nodes, the third return value is None and the four-corner path is
    unchanged.
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

    if not rings.has_side_nodes:
        return area_vec, centroid, None

    rp = np.zeros(3, dtype=np.float64) if ref_point is None else np.asarray(ref_point, dtype=np.float64)
    area_moment = cross3(centroid - rp[None, :], area_vec)
    sn = rings.side_nodes
    sc = rings.side_counts
    te_mask = rings.te_mask
    n_p = rings.n_panels
    for p in range(n_p):
        cr = int(sc[p, 0])
        cl = int(sc[p, 1])
        if cr == 0 and cl == 0:
            continue
        r_pts = [sn[p, 0, k] for k in range(cr)]
        l_pts = [sn[p, 1, k] for k in range(cl)]
        # Exclude the physical TE node from the intermediate list on extended TE rings.
        if te_mask[p]:
            if r_pts and np.allclose(r_pts[-1], q2[p], atol=1e-13, rtol=1e-13):
                r_pts.pop()
            if l_pts and np.allclose(l_pts[0], q3[p], atol=1e-13, rtol=1e-13):
                l_pts.pop(0)
        if not r_pts and not l_pts:
            continue

        # Order both side chains from front (bound) to physical rear.
        L = [q0[p], *l_pts[::-1], q3[p]]
        R = [q1[p], *r_pts, q2[p]]
        a_sum = np.zeros(3, dtype=np.float64)
        m_sum = np.zeros(3, dtype=np.float64)
        c_num = np.zeros(3, dtype=np.float64)
        w_sum = 0.0
        il, ir = 0, 0
        nl, nr = len(L) - 1, len(R) - 1
        while il < nl or ir < nr:
            if il < nl and ir < nr:
                tris = ((L[il], R[ir + 1], R[ir]), (L[il], L[il + 1], R[ir + 1]))
                il += 1
                ir += 1
            elif ir < nr:
                tris = ((L[il], R[ir + 1], R[ir]),)
                ir += 1
            else:
                tris = ((L[il], L[il + 1], R[ir]),)
                il += 1
            for t0, t1, t2 in tris:
                at = 0.5 * np.cross(t1 - t0, t2 - t0)
                ct = (t0 + t1 + t2) / 3.0
                mt = float(np.linalg.norm(at))
                a_sum += at
                m_sum += np.cross(ct - rp, at)
                c_num += mt * ct
                w_sum += mt
        area_vec[p] = a_sum
        centroid[p] = c_num / max(w_sum, 1e-300)
        area_moment[p] = m_sum

    return area_vec, centroid, area_moment


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
            vr = ring_velocity(
                Pc, rings.corners, rc_r, side_nodes=rings.side_nodes, side_counts=rings.side_counts,
            )
            vw = steady_wake_velocity(Pc, rings.te_left, rings.te_right, wd, rc_w)
            v_out[i0:i1] += np.einsum("mni,n->mi", vr, gamma_rings)
            v_out[i0:i1] += np.einsum("msi,s->mi", vw, g_te)
        if img is not None:
            vr_i = ring_velocity(
                Pc, img.corners, rc_r, side_nodes=img.side_nodes, side_counts=img.side_counts,
            )
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

        self.device, self.precision, self.notes = _select_harmonic_device(
            self.rings.n_panels, self.n_wake_rows, self.rings.n_strips, self.ground is not None,
        )
        self._gpu_state: _HarmonicGPUState | None = None
        self.lu_Abb = None
        self.lu_Asteady = None

        if self.device == "gpu":
            self._init_with_gpu_fallback(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)
        else:
            self._init_cpu(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)

    def _init_cpu(self, *, fd_step_h: float | None, fd_step_theta: float | None) -> None:
        """Assemble matrices, mean state, loaded terms and k -> 0 limits on the CPU."""
        self._assemble_system_matrices()
        self._prepare_load_geometry()
        self._solve_mean_state()
        self._prepare_loaded_terms(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)
        self.k0_heave, self.k0_pitch, self.k0_pitch_rate = self._compute_k0_limits()

    def _init_with_gpu_fallback(self, *, fd_step_h: float | None, fd_step_theta: float | None) -> None:
        """Run GPU initialisation with one OOM retry, falling back to CPU after two OOM errors."""
        from ventorum.gpu import memory as gpu_memory
        from ventorum.gpu.pipeline import is_out_of_memory

        message = ""
        for _attempt in range(2):
            try:
                self._init_gpu(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)
                return
            except Exception as exc:  # noqa: BLE001 - only out-of-memory errors are caught
                if not is_out_of_memory(exc):
                    raise
                message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            if self._gpu_state is not None:
                self._gpu_state.release_gpu()
                self._gpu_state = None
            gpu_memory.release()
        warnings.warn(
            f"The GPU ran out of memory twice ({message}). The harmonic solve falls back to the CPU.",
            RuntimeWarning,
            stacklevel=4,
        )
        self.device = "cpu"
        self.precision = "float64"
        if NOTE_GPU_OUT_OF_MEMORY not in self.notes:
            self.notes.append(NOTE_GPU_OUT_OF_MEMORY)
        self._init_cpu(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)

    def _init_gpu(self, *, fd_step_h: float | None, fd_step_theta: float | None) -> None:
        """Assemble matrices, mean state, loaded terms and k -> 0 limits on the GPU."""
        self._gpu_state = _HarmonicGPUState(self.precision)
        self._assemble_system_matrices_gpu()
        self._prepare_load_geometry()
        self._solve_mean_state_gpu()
        self._prepare_loaded_terms_gpu(fd_step_h=fd_step_h, fd_step_theta=fd_step_theta)
        self.k0_heave, self.k0_pitch, self.k0_pitch_rate = self._compute_k0_limits_gpu()

    def _assemble_system_matrices_gpu(self) -> None:
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips
        te_idx = rings.te_indices
        prec = self.precision
        t_dtype = engine.torch_dtype(prec)
        dev = engine._dev()
        st = self._gpu_state.tensors

        cp_g = engine._f64(rings.cp)
        nbc_g = engine._f64(rings.normal_bc)
        corners_g = engine._f64(rings.corners)
        sn_g = engine._f64(rings.side_nodes) if rings.has_side_nodes else None
        sc_g = engine._i32(rings.side_counts) if rings.has_side_nodes else None
        te_l_g = engine._f64(rings.te_left)
        te_r_g = engine._f64(rings.te_right)
        wd_g = engine._f64(rings.wake_dir)
        tail_l_g = engine._f64(self.tail_left)
        tail_r_g = engine._f64(self.tail_right)
        te_idx_g = torch.as_tensor(te_idx, dtype=torch.int64, device=dev)

        A_bb_g = RK.ring_normal_influence_gpu(
            cp_g, nbc_g, corners_g, self.rc_ring,
            side_nodes=sn_g, side_counts=sc_g, ground=self.ground, precision=prec, out_tensor=True,
        ).to(torch.float64)
        A_bw_steady_g = RK.steady_wake_normal_influence_gpu(
            cp_g, nbc_g, te_l_g, te_r_g, wd_g, self.rc_wake,
            ground=self.ground, precision=prec, out_tensor=True,
        ).to(torch.float64)

        A_steady_g = A_bb_g.clone()
        A_steady_g[:, te_idx_g] += A_bw_steady_g
        lu_Abb_g, piv_Abb_g = torch.linalg.lu_factor(A_bb_g)
        lu_Asteady_g, piv_Asteady_g = torch.linalg.lu_factor(A_steady_g)

        n_rows = self.n_wake_rows
        A_bw_g = torch.empty((n_p, n_rows, n_s), dtype=t_dtype, device=dev)
        for m0 in range(0, n_rows, _WAKE_ROW_CHUNK):
            m1 = min(n_rows, m0 + _WAKE_ROW_CHUNK)
            n_chunk = m1 - m0
            rc_chunk = self._tile_wake_rc(self.rc_wake, n_chunk)
            m_idx = torch.arange(m0, m1, dtype=torch.float64, device=dev)[:, None, None]
            d0 = (m_idx * self.dx_w) * wd_g[None, :, :]
            d1 = ((m_idx + 1.0) * self.dx_w) * wd_g[None, :, :]
            w_corners_g = torch.stack(
                [te_l_g[None, :, :] + d0, te_r_g[None, :, :] + d0, te_r_g[None, :, :] + d1, te_l_g[None, :, :] + d1],
                dim=2,
            ).reshape(-1, 4, 3)
            blk = RK.ring_normal_influence_gpu(
                cp_g, nbc_g, w_corners_g, rc_chunk, ground=self.ground, precision=prec, out_tensor=True,
            )
            A_bw_g[:, m0:m1, :] = blk.reshape(n_p, n_chunk, n_s)

        st["cp"] = cp_g
        st["normal_bc"] = nbc_g
        st["corners"] = corners_g
        if sn_g is not None:
            st["side_nodes"] = sn_g
            st["side_counts"] = sc_g
        st["te_left"] = te_l_g
        st["te_right"] = te_r_g
        st["wake_dir"] = wd_g
        st["tail_left"] = tail_l_g
        st["tail_right"] = tail_r_g
        st["te_idx"] = te_idx_g
        st["A_bb"] = A_bb_g
        st["A_bw_steady"] = A_bw_steady_g
        st["A_steady"] = A_steady_g
        st["lu_Abb"] = lu_Abb_g
        st["piv_Abb"] = piv_Abb_g
        st["lu_Asteady"] = lu_Asteady_g
        st["piv_Asteady"] = piv_Asteady_g
        st["A_bw"] = A_bw_g

        self.A_bb = A_bb_g.cpu().numpy()
        self.A_bw_steady = A_bw_steady_g.cpu().numpy()
        self.A_steady = A_steady_g.cpu().numpy()
        self.A_bw = A_bw_g.double().cpu().numpy()
        self.lu_Abb = None
        self.lu_Asteady = None

    def _steady_velocity_at_gpu(
        self,
        P: np.ndarray,
        gamma_rings_g,
        rc_ring: np.ndarray | float,
        rc_wake: np.ndarray | float,
        *,
        ground: GroundPlane | None = None,
        wake_dir=None,
        img_mode: int | None = None,
        precision: str | None = None,
    ) -> np.ndarray:
        """Return steady induced velocity [m/s] at *P* computed on the GPU."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        st = self._gpu_state.tensors
        prec = self.precision if precision is None else precision
        wd_g = st["wake_dir"] if wake_dir is None else engine._f64(wake_dir)
        g_te_g = gamma_rings_g[st["te_idx"]]
        corners_g = st["corners"]
        sn_g = st.get("side_nodes")
        sc_g = st.get("side_counts")
        te_l_g = st["te_left"]
        te_r_g = st["te_right"]
        m = P.shape[0]
        v_out_g = torch.zeros((m, 3), dtype=torch.float64, device=engine._dev())
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc_g = engine._f64(P[i0:i1])
            rc_r = _slice_rc_rows(rc_ring, i0, i1)
            rc_w = _slice_rc_rows(rc_wake, i0, i1)
            vr = RK.ring_velocity_gpu(
                Pc_g, corners_g, rc_r, side_nodes=sn_g, side_counts=sc_g,
                ground=ground, img_mode=img_mode, precision=prec, out_tensor=True,
            ).to(torch.float64)
            vw = RK.steady_wake_velocity_gpu(
                Pc_g, te_l_g, te_r_g, wd_g, rc_w,
                ground=ground, img_mode=img_mode, precision=prec, out_tensor=True,
            ).to(torch.float64)
            v_out_g[i0:i1] = torch.einsum("mni,n->mi", vr, gamma_rings_g) + torch.einsum("msi,s->mi", vw, g_te_g)
        return v_out_g.cpu().numpy()

    def _solve_mean_state_gpu(self) -> None:
        import torch

        from ventorum.gpu import engine

        st = self._gpu_state.tensors
        umap = make_unknown_map(self.lattice, False)
        rhs_0 = _vlm_rhs(self.lattice, umap, self.condition, self.ref_point)
        rhs_0_g = engine._f64(rhs_0)
        Gamma_0_g = torch.linalg.lu_solve(
            st["lu_Asteady"], st["piv_Asteady"], rhs_0_g.unsqueeze(-1),
        ).squeeze(-1)
        st["Gamma_0"] = Gamma_0_g
        self.Gamma_0 = Gamma_0_g.cpu().numpy()
        self.has_mean_load = bool(np.max(np.abs(self.Gamma_0)) > 1e-15 * max(self.V_inf * self.c_ref, 1.0))

        self.gamma_hs_0 = _rings_to_horseshoe(self.Gamma_0, self.rings.n_chord)
        g_leg_base = leg_strength(self.lattice, self.gamma_hs_0)
        self.gamma_0_all = np.concatenate([self.gamma_hs_0] + [g_leg_base] * self.n_leg_sets)

        if self.has_mean_load:
            self.v_ind_load_0 = self._steady_velocity_at_gpu(
                self.P_load, Gamma_0_g, self.rc_load_ring, self.rc_load_wake, ground=self.ground,
            )
            self.v_ind_te_0 = self._steady_velocity_at_gpu(
                self.p_te, Gamma_0_g, self.rc_te_ring, self.rc_te_wake, ground=self.ground,
            )
        else:
            self.v_ind_load_0 = np.zeros_like(self.P_load)
            self.v_ind_te_0 = np.zeros_like(self.p_te)

        self.V_tot_0_all = self.V_0[None, :] + self.v_ind_load_0
        self.V_eff_0_all = (
            self.V_tot_0_all
            - np.sum(self.V_tot_0_all * self.seg_normal_all, axis=1, keepdims=True) * self.seg_normal_all
        )
        self.V_te_0 = self.V_0[None, :] + self.v_ind_te_0
        self.F_seg_0 = (self.rho * self.gamma_0_all * self.seg_scale_all)[:, None] * cross3(
            self.V_eff_0_all, self.seg_vec_all,
        )
        self.F_total_0 = np.sum(self.F_seg_0, axis=0)
        self.M_total_0 = np.sum(cross3(self.arm_all, self.F_seg_0), axis=0)
        self.mean_state = self._to_steady_loads(self.F_total_0, self.M_total_0)

    def _contract_load_vel_gpu(self, v_chunk_g, i0: int, i1: int, fm_out_g, sl_out_g) -> None:
        """Accumulate CUDA velocity chunk ``(chunk_m, n_src, 3)`` into ``fm_out_g`` and ``sl_out_g``."""
        import torch

        st = self._gpu_state.tensors
        W_fm_c = st["W_fm"][:, i0:i1, :]
        W_lift_c = st["W_lift"][i0:i1, :]
        if v_chunk_g.is_complex():
            W_fm_c = W_fm_c.to(torch.complex128)
            W_lift_c = W_lift_c.to(torch.complex128)
        else:
            v_chunk_g = v_chunk_g.to(torch.float64)
        fm_out_g.add_(torch.einsum("cmi,mni->cn", W_fm_c, v_chunk_g))
        seg_lift = torch.einsum("mi,mni->mn", W_lift_c, v_chunk_g)
        sl_out_g.index_add_(0, st["eval_strips"][i0:i1], seg_lift)

    def _assemble_velocity_load_matrices_gpu(self) -> None:
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        st = self._gpu_state.tensors
        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips
        n_rows = self.n_wake_rows
        m = self.P_load.shape[0]
        prec = self.precision
        dev = engine._dev()

        P_load_g = engine._f64(self.P_load)
        st["P_load"] = P_load_g
        st["W_fm"] = engine._f64(self.W_fm)
        st["W_lift"] = engine._f64(self.W_lift)
        st["eval_strips"] = torch.as_tensor(self.eval_strips, dtype=torch.int64, device=dev)

        corners_g = st["corners"]
        sn_g = st.get("side_nodes")
        sc_g = st.get("side_counts")
        te_l_g = st["te_left"]
        te_r_g = st["te_right"]
        wd_g = st["wake_dir"]

        FM_bound_g = torch.zeros((6, n_p), dtype=torch.float64, device=dev)
        SL_bound_g = torch.zeros((n_s, n_p), dtype=torch.float64, device=dev)
        FM_sw_g = torch.zeros((6, n_s), dtype=torch.float64, device=dev)
        SL_sw_g = torch.zeros((n_s, n_s), dtype=torch.float64, device=dev)

        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc_g = P_load_g[i0:i1].contiguous()
            rc_r = _slice_rc_rows(self.rc_load_ring, i0, i1)
            rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
            vr = RK.ring_velocity_gpu(
                Pc_g, corners_g, rc_r, side_nodes=sn_g, side_counts=sc_g,
                ground=self.ground, precision=prec, out_tensor=True,
            )
            vw = RK.steady_wake_velocity_gpu(
                Pc_g, te_l_g, te_r_g, wd_g, rc_w,
                ground=self.ground, precision=prec, out_tensor=True,
            )
            self._contract_load_vel_gpu(vr, i0, i1, FM_bound_g, SL_bound_g)
            self._contract_load_vel_gpu(vw, i0, i1, FM_sw_g, SL_sw_g)

        self.FM_bound_mat = FM_bound_g.cpu().numpy()
        self.SL_bound_mat = SL_bound_g.cpu().numpy()
        self.FM_steady_wake_mat = FM_sw_g.cpu().numpy()
        self.SL_steady_wake_mat = SL_sw_g.cpu().numpy()

        FM_wake_g = torch.zeros((6, n_rows, n_s), dtype=torch.float64, device=dev)
        SL_wake_g = torch.zeros((n_s, n_rows, n_s), dtype=torch.float64, device=dev)
        for m0 in range(0, n_rows, _WAKE_ROW_CHUNK):
            m1 = min(n_rows, m0 + _WAKE_ROW_CHUNK)
            n_chunk = m1 - m0
            m_idx = torch.arange(m0, m1, dtype=torch.float64, device=dev)[:, None, None]
            d0 = (m_idx * self.dx_w) * wd_g[None, :, :]
            d1 = ((m_idx + 1.0) * self.dx_w) * wd_g[None, :, :]
            w_corners_g = torch.stack(
                [te_l_g[None, :, :] + d0, te_r_g[None, :, :] + d0, te_r_g[None, :, :] + d1, te_l_g[None, :, :] + d1],
                dim=2,
            ).reshape(-1, 4, 3)
            fm_blk = torch.zeros((6, n_chunk * n_s), dtype=torch.float64, device=dev)
            sl_blk = torch.zeros((n_s, n_chunk * n_s), dtype=torch.float64, device=dev)
            for i0 in range(0, m, _LOAD_POINT_CHUNK):
                i1 = min(m, i0 + _LOAD_POINT_CHUNK)
                Pc_g = P_load_g[i0:i1].contiguous()
                rc_w = self._tile_wake_rc(_slice_rc_rows(self.rc_load_wake, i0, i1), n_chunk)
                vw = RK.ring_velocity_gpu(Pc_g, w_corners_g, rc_w, ground=self.ground, precision=prec, out_tensor=True)
                self._contract_load_vel_gpu(vw, i0, i1, fm_blk, sl_blk)
            FM_wake_g[:, m0:m1, :] = fm_blk.reshape(6, n_chunk, n_s)
            SL_wake_g[:, m0:m1, :] = sl_blk.reshape(n_s, n_chunk, n_s)

        self.FM_wake_tensor = FM_wake_g.cpu().numpy()
        self.SL_wake_tensor = SL_wake_g.cpu().numpy()

    def _image_steady_matvec_gpu(self, ground: GroundPlane) -> np.ndarray:
        """Return ``A_img_steady @ Gamma_0`` on GPU in float64 (for central finite differences)."""
        from ventorum.gpu import ring_kernels as RK

        st = self._gpu_state.tensors
        Gamma_0_g = st["Gamma_0"]
        g_te_g = Gamma_0_g[st["te_idx"]]
        a_r = RK.ring_normal_influence_gpu(
            st["cp"], st["normal_bc"], st["corners"], self.rc_ring,
            side_nodes=st.get("side_nodes"), side_counts=st.get("side_counts"),
            ground=ground, img_mode=2, precision="float64", out_tensor=True,
        )
        a_w = RK.steady_wake_normal_influence_gpu(
            st["cp"], st["normal_bc"], st["te_left"], st["te_right"], st["wake_dir"], self.rc_wake,
            ground=ground, img_mode=2, precision="float64", out_tensor=True,
        )
        return (a_r @ Gamma_0_g + a_w @ g_te_g).cpu().numpy()

    def _varying_steady_matvec_gpu(self, ground: GroundPlane | None, wd: np.ndarray) -> np.ndarray:
        """Return ``(A_bw_real(wd) + A_img_steady(ground, wd)) @ Gamma_0`` on GPU in float64."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        st = self._gpu_state.tensors
        Gamma_0_g = st["Gamma_0"]
        g_te_g = Gamma_0_g[st["te_idx"]]
        wd_g = engine._f64(wd)
        v_g = torch.zeros(self.rings.n_panels, dtype=torch.float64, device=engine._dev())
        if self._wake_rotates_with_alpha:
            a_w = RK.steady_wake_normal_influence_gpu(
                st["cp"], st["normal_bc"], st["te_left"], st["te_right"], wd_g, self.rc_wake,
                ground=None, img_mode=0, precision="float64", out_tensor=True,
            )
            v_g = v_g + a_w @ g_te_g
        if ground is not None:
            a_ri = RK.ring_normal_influence_gpu(
                st["cp"], st["normal_bc"], st["corners"], self.rc_ring,
                side_nodes=st.get("side_nodes"), side_counts=st.get("side_counts"),
                ground=ground, img_mode=2, precision="float64", out_tensor=True,
            )
            a_wi = RK.steady_wake_normal_influence_gpu(
                st["cp"], st["normal_bc"], st["te_left"], st["te_right"], wd_g, self.rc_wake,
                ground=ground, img_mode=2, precision="float64", out_tensor=True,
            )
            v_g = v_g + a_ri @ Gamma_0_g + a_wi @ g_te_g
        return v_g.cpu().numpy()

    def _varying_steady_load_vel_gpu(self, ground: GroundPlane | None, wd: np.ndarray) -> np.ndarray:
        """Return load-point velocity from real steady wake and ground images on GPU in float64."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        st = self._gpu_state.tensors
        Gamma_0_g = st["Gamma_0"]
        g_te_g = Gamma_0_g[st["te_idx"]]
        wd_g = engine._f64(wd)
        P_load_g = st["P_load"]
        m = self.P_load.shape[0]
        v_out_g = torch.zeros((m, 3), dtype=torch.float64, device=engine._dev())
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc_g = P_load_g[i0:i1].contiguous()
            rc_r = _slice_rc_rows(self.rc_load_ring, i0, i1)
            rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
            if self._wake_rotates_with_alpha:
                vw = RK.steady_wake_velocity_gpu(
                    Pc_g, st["te_left"], st["te_right"], wd_g, rc_w,
                    ground=None, img_mode=0, precision="float64", out_tensor=True,
                )
                v_out_g[i0:i1] += torch.einsum("msi,s->mi", vw, g_te_g)
            if ground is not None:
                vr_i = RK.ring_velocity_gpu(
                    Pc_g, st["corners"], rc_r,
                    side_nodes=st.get("side_nodes"), side_counts=st.get("side_counts"),
                    ground=ground, img_mode=2, precision="float64", out_tensor=True,
                )
                vw_i = RK.steady_wake_velocity_gpu(
                    Pc_g, st["te_left"], st["te_right"], wd_g, rc_w,
                    ground=ground, img_mode=2, precision="float64", out_tensor=True,
                )
                v_out_g[i0:i1] += torch.einsum("mni,n->mi", vr_i, Gamma_0_g)
                v_out_g[i0:i1] += torch.einsum("msi,s->mi", vw_i, g_te_g)
        return v_out_g.cpu().numpy()

    def _init_travelling_wake_displacements(self) -> None:
        """Compute trailing-edge closure displacement vectors for heave and pitch."""
        rings = self.rings
        n_s = rings.n_strips
        u_h = np.ascontiguousarray(np.broadcast_to(self.c_ref * self.e_h, (n_s, 3)), dtype=np.float64)
        self.u_heave_L = u_h
        self.u_heave_R = u_h
        self.u_pitch_L = np.ascontiguousarray(_ey_cross(rings.te_left - self.ref_point[None, :]), dtype=np.float64)
        self.u_pitch_R = np.ascontiguousarray(_ey_cross(rings.te_right - self.ref_point[None, :]), dtype=np.float64)
        if self.ground is not None:
            self.u_heave_L_img = self.ground.reflect_directions(self.u_heave_L)
            self.u_heave_R_img = self.ground.reflect_directions(self.u_heave_R)
            self.u_pitch_L_img = self.ground.reflect_directions(self.u_pitch_L)
            self.u_pitch_R_img = self.ground.reflect_directions(self.u_pitch_R)
        else:
            self.u_heave_L_img = None
            self.u_heave_R_img = None
            self.u_pitch_L_img = None
            self.u_pitch_R_img = None

    def _assemble_travelling_wake_one_motion_gpu(
        self,
        u_L: np.ndarray,
        u_R: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(dA_wake, dFM_wake, dSL_wake)`` for one motion on the GPU in float64."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        st = self._gpu_state.tensors
        dev = engine._dev()
        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips
        n_rows = self.n_wake_rows
        m = self.P_load.shape[0]
        g_te_g = st["Gamma_0"][st["te_idx"]]
        te_l_g = st["te_left"]
        te_r_g = st["te_right"]
        wd_g = st["wake_dir"]
        cp_g = st["cp"]
        nbc_g = st["normal_bc"]
        P_load_g = st["P_load"]
        u_L_g = engine._f64(u_L)
        u_R_g = engine._f64(u_R)

        dA_lead_g = torch.zeros((n_p, n_rows), dtype=torch.float64, device=dev)
        dA_trail_g = torch.zeros((n_p, n_rows), dtype=torch.float64, device=dev)
        dFM_lead_g = torch.zeros((6, n_rows), dtype=torch.float64, device=dev)
        dFM_trail_g = torch.zeros((6, n_rows), dtype=torch.float64, device=dev)
        dSL_lead_g = torch.zeros((n_s, n_rows), dtype=torch.float64, device=dev)
        dSL_trail_g = torch.zeros((n_s, n_rows), dtype=torch.float64, device=dev)
        inv_2eps = 0.5 / eps

        for m0 in range(0, n_rows, _WAKE_ROW_CHUNK):
            m1 = min(n_rows, m0 + _WAKE_ROW_CHUNK)
            m_idx = torch.arange(m0, m1, dtype=torch.float64, device=dev)[:, None, None]
            d0 = (m_idx * self.dx_w) * wd_g[None, :, :]
            d1 = ((m_idx + 1.0) * self.dx_w) * wd_g[None, :, :]
            w_base = torch.stack(
                [te_l_g[None, :, :] + d0, te_r_g[None, :, :] + d0, te_r_g[None, :, :] + d1, te_l_g[None, :, :] + d1],
                dim=2,
            )  # (n_chunk, n_s, 4, 3)

            for is_lead in (True, False):
                r_off = 1 if (is_lead and m0 == 0) else 0
                r0 = m0 + r_off
                n_sub = m1 - r0
                if n_sub <= 0:
                    continue
                w_sub = w_base[r_off:].reshape(-1, 4, 3)
                du = torch.zeros((n_sub, n_s, 4, 3), dtype=torch.float64, device=dev)
                if is_lead:
                    du[:, :, 0, :] = u_L_g[None, :, :]
                    du[:, :, 1, :] = u_R_g[None, :, :]
                else:
                    du[:, :, 2, :] = u_R_g[None, :, :]
                    du[:, :, 3, :] = u_L_g[None, :, :]
                du_flat = du.reshape(-1, 4, 3)
                w_plus = (w_sub + eps * du_flat).contiguous()
                w_minus = (w_sub - eps * du_flat).contiguous()

                rc_sub = self._tile_wake_rc(self.rc_wake, n_sub)
                ap = RK.ring_normal_influence_gpu(
                    cp_g, nbc_g, w_plus, rc_sub, ground=self.ground, precision="float64", out_tensor=True,
                )
                am = RK.ring_normal_influence_gpu(
                    cp_g, nbc_g, w_minus, rc_sub, ground=self.ground, precision="float64", out_tensor=True,
                )
                da_blk = ((ap - am) * inv_2eps).reshape(n_p, n_sub, n_s) @ g_te_g
                if is_lead:
                    dA_lead_g[:, r0:m1] = da_blk
                else:
                    dA_trail_g[:, r0:m1] = da_blk

                fm_blk = torch.zeros((6, n_sub), dtype=torch.float64, device=dev)
                sl_blk = torch.zeros((n_s, n_sub), dtype=torch.float64, device=dev)
                for i0 in range(0, m, _LOAD_POINT_CHUNK):
                    i1 = min(m, i0 + _LOAD_POINT_CHUNK)
                    Pc_g = P_load_g[i0:i1].contiguous()
                    rc_l = self._tile_wake_rc(_slice_rc_rows(self.rc_load_wake, i0, i1), n_sub)
                    vp = RK.ring_velocity_gpu(
                        Pc_g, w_plus, rc_l, ground=self.ground, precision="float64", out_tensor=True,
                    )
                    vm = RK.ring_velocity_gpu(
                        Pc_g, w_minus, rc_l, ground=self.ground, precision="float64", out_tensor=True,
                    )
                    dv = ((vp - vm) * inv_2eps).reshape(i1 - i0, n_sub, n_s, 3)
                    dv_row = torch.einsum("mrsi,s->mri", dv, g_te_g)
                    self._contract_load_vel_gpu(dv_row, i0, i1, fm_blk, sl_blk)
                if is_lead:
                    dFM_lead_g[:, r0:m1] = fm_blk
                    dSL_lead_g[:, r0:m1] = sl_blk
                else:
                    dFM_trail_g[:, r0:m1] = fm_blk
                    dSL_trail_g[:, r0:m1] = sl_blk

        # Combine trailing edge of row j-1 and leading edge of row j at station j = 1 .. n_rows.
        dA_g = dA_trail_g.clone()
        dFM_g = dFM_trail_g.clone()
        dSL_g = dSL_trail_g.clone()
        if n_rows > 1:
            dA_g[:, :-1].add_(dA_lead_g[:, 1:])
            dFM_g[:, :-1].add_(dFM_lead_g[:, 1:])
            dSL_g[:, :-1].add_(dSL_lead_g[:, 1:])

        # Cancel the trailing spanwise segment (tail_right -> tail_left) of the last discrete row
        # by adding the matching opposite closing segment (tail_left -> tail_right).
        tl_p = (st["tail_left"] + eps * u_L_g).contiguous()
        tr_p = (st["tail_right"] + eps * u_R_g).contiguous()
        tl_m = (st["tail_left"] - eps * u_L_g).contiguous()
        tr_m = (st["tail_right"] - eps * u_R_g).contiguous()
        vsp = RK.segment_velocity_gpu(cp_g, tl_p, tr_p, self.rc_wake, ground=self.ground, out_tensor=True)
        vsm = RK.segment_velocity_gpu(cp_g, tl_m, tr_m, self.rc_wake, ground=self.ground, out_tensor=True)
        dA_g[:, -1].add_(torch.einsum("msi,mi,s->m", (vsp - vsm) * inv_2eps, nbc_g, g_te_g))

        fm_sp = torch.zeros((6, 1), dtype=torch.float64, device=dev)
        sl_sp = torch.zeros((n_s, 1), dtype=torch.float64, device=dev)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc_g = P_load_g[i0:i1].contiguous()
            rc_l = _slice_rc_rows(self.rc_load_wake, i0, i1)
            vsp_l = RK.segment_velocity_gpu(Pc_g, tl_p, tr_p, rc_l, ground=self.ground, out_tensor=True)
            vsm_l = RK.segment_velocity_gpu(Pc_g, tl_m, tr_m, rc_l, ground=self.ground, out_tensor=True)
            dv_sp = torch.einsum("msi,s->mi", (vsp_l - vsm_l) * inv_2eps, g_te_g).unsqueeze(1)
            self._contract_load_vel_gpu(dv_sp, i0, i1, fm_sp, sl_sp)
        dFM_g[:, -1].add_(fm_sp[:, 0])
        dSL_g[:, -1].add_(sl_sp[:, 0])

        return dA_g.cpu().numpy(), dFM_g.cpu().numpy(), dSL_g.cpu().numpy()

    def _prepare_loaded_terms_gpu(
        self,
        *,
        fd_step_h: float | None,
        fd_step_theta: float | None,
    ) -> None:
        from ventorum.gpu import engine

        rings = self.rings
        n_p = rings.n_panels
        self.rhs_heave_geo = np.zeros(n_p, dtype=np.float64)
        self.rhs_pitch_geo = np.zeros(n_p, dtype=np.float64)
        self.dv_load_heave_geo = np.zeros_like(self.P_load)
        self.dv_load_pitch_geo = np.zeros_like(self.P_load)
        self.dA_wake_heave = None
        self.dA_wake_pitch = None
        self.dFM_wake_heave = None
        self.dSL_wake_heave = None
        self.dFM_wake_pitch = None
        self.dSL_wake_pitch = None
        self.u_heave_L = None
        self.u_heave_R = None
        self.u_pitch_L = None
        self.u_pitch_R = None
        self.u_heave_L_img = None
        self.u_heave_R_img = None
        self.u_pitch_L_img = None
        self.u_pitch_R_img = None

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

        self.W_fm, self.W_lift = _build_velocity_load_weights(
            self.rho, self.gamma_0_all, self.seg_vec_all, self.seg_normal_all,
            self.seg_scale_all, self.arm_all, self.L_0,
        )
        self._assemble_velocity_load_matrices_gpu()
        self._init_travelling_wake_displacements()
        st = self._gpu_state.tensors
        st["u_heave_L"] = engine._f64(self.u_heave_L)
        st["u_heave_R"] = engine._f64(self.u_heave_R)
        st["u_pitch_L"] = engine._f64(self.u_pitch_L)
        st["u_pitch_R"] = engine._f64(self.u_pitch_R)

        h_min = None
        if self.ground is not None:
            pts_list = [self.lattice.all_points(), rings.corners.reshape(-1, 3)]
            if rings.side_nodes is not None:
                pts_list.append(rings.side_nodes.reshape(-1, 3))
            pts = np.vstack(pts_list)
            h_min = float(np.min(self.ground.height(pts)))
            if h_min <= 0.0:
                h_min = float(np.min(self.ground.height(self.lattice.all_points())))

        # Central finite differences use float64 GPU kernels so eps = 1e-5 does not amplify float32 noise.
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
            dA_dh_G0 = (self._image_steady_matvec_gpu(g_plus) - self._image_steady_matvec_gpu(g_minus)) / (2.0 * eps_h)
            self.rhs_heave_geo = -self.c_ref * dA_dh_G0

            Gamma_0_g = st["Gamma_0"]
            vp = self._steady_velocity_at_gpu(
                self.P_load, Gamma_0_g, self.rc_load_ring, self.rc_load_wake,
                ground=g_plus, img_mode=2, precision="float64",
            )
            vm = self._steady_velocity_at_gpu(
                self.P_load, Gamma_0_g, self.rc_load_ring, self.rc_load_wake,
                ground=g_minus, img_mode=2, precision="float64",
            )
            self.dv_load_heave_geo = self.c_ref * (vp - vm) / (2.0 * eps_h)
        else:
            eps_h = _DEFAULT_FD_STEP_H_FRAC * self.c_ref if fd_step_h is None else float(fd_step_h)
            if not math.isfinite(eps_h) or eps_h <= 0.0:
                raise InputError(f"fd_step_h must be finite and > 0, got {eps_h!r}.")

        eps_th = _DEFAULT_FD_STEP_THETA if fd_step_theta is None else float(fd_step_theta)
        if self.ground is not None and fd_step_theta is None:
            max_lever = max(float(np.max(np.linalg.norm(self.lattice.all_points() - self.ref_point, axis=1))), 1e-6)
            eps_th = min(eps_th, 0.01 * h_min / max_lever)
        if not math.isfinite(eps_th) or eps_th <= 0.0:
            raise InputError(f"fd_step_theta must be finite and > 0, got {eps_th!r}.")

        if self.ground is not None or self._wake_rotates_with_alpha:
            g_tp, wd_tp = self._pitched_steady_state(+eps_th)
            g_tm, wd_tm = self._pitched_steady_state(-eps_th)
            dA_dth_G0 = (
                self._varying_steady_matvec_gpu(g_tp, wd_tp) - self._varying_steady_matvec_gpu(g_tm, wd_tm)
            ) / (2.0 * eps_th)
            self.rhs_pitch_geo = -dA_dth_G0

            vp_th = self._varying_steady_load_vel_gpu(g_tp, wd_tp)
            vm_th = self._varying_steady_load_vel_gpu(g_tm, wd_tm)
            self.dv_load_pitch_geo = (vp_th - vm_th) / (2.0 * eps_th)

        self.dA_wake_heave, self.dFM_wake_heave, self.dSL_wake_heave = (
            self._assemble_travelling_wake_one_motion_gpu(self.u_heave_L, self.u_heave_R, eps_h / self.c_ref)
        )
        self.dA_wake_pitch, self.dFM_wake_pitch, self.dSL_wake_pitch = (
            self._assemble_travelling_wake_one_motion_gpu(self.u_pitch_L, self.u_pitch_R, eps_th)
        )

    def _compute_k0_limits_gpu(self) -> tuple[_SteadyLoads, _SteadyLoads, _SteadyLoads]:
        """Compute exact real ``k -> 0`` limits using the GPU LU of ``A_steady``."""
        import torch

        from ventorum.gpu import engine

        rings = self.rings
        n_bc = rings.normal_bc
        rel_cp = rings.cp - self.ref_point[None, :]
        dU_alpha = -_ey_cross(self.V_0)
        q_dim = 2.0 * self.V_inf / self.c_ref
        dU_q_cp = -q_dim * _ey_cross(rel_cp)

        rhs_h0 = self.rhs_heave_geo
        rhs_p0 = -(n_bc @ dU_alpha) + self.rhs_pitch_geo
        rhs_q0 = -np.sum(n_bc * dU_q_cp, axis=1)

        st = self._gpu_state.tensors
        rhs_mat_g = engine._f64(np.column_stack([rhs_h0, rhs_p0, rhs_q0]))
        G_0s = torch.linalg.lu_solve(st["lu_Asteady"], st["piv_Asteady"], rhs_mat_g).cpu().numpy()
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

    def _ensure_gpu_solve_state(self) -> dict[str, Any]:
        """Return active GPU tensors for solves, re-uploading from host if ``gpu.release()`` cleared them."""
        import torch

        from ventorum.gpu import engine

        if self._gpu_state is None:
            self._gpu_state = _HarmonicGPUState(self.precision)
        st = self._gpu_state.tensors
        if "lu_Abb" not in st:
            dev = engine._dev()
            t_dtype = engine.torch_dtype(self.precision)
            A_bb_g = engine._f64(self.A_bb)
            lu_Abb_g, piv_Abb_g = torch.linalg.lu_factor(A_bb_g)
            st["A_bb"] = A_bb_g
            st["lu_Abb"] = lu_Abb_g
            st["piv_Abb"] = piv_Abb_g
            st["A_bw_steady"] = engine._f64(self.A_bw_steady)
            st["A_bw"] = engine._f64(self.A_bw).to(t_dtype).contiguous()
            st["cp"] = engine._f64(self.rings.cp)
            st["normal_bc"] = engine._f64(self.rings.normal_bc)
            st["tail_left"] = engine._f64(self.tail_left)
            st["tail_right"] = engine._f64(self.tail_right)
            st["wake_dir"] = engine._f64(self.rings.wake_dir)
            st["te_idx"] = torch.as_tensor(self.rings.te_indices, dtype=torch.int64, device=dev)
            if self.has_mean_load and self.W_fm is not None:
                st["Gamma_0"] = engine._f64(self.Gamma_0)
                st["P_load"] = engine._f64(self.P_load)
                st["W_fm"] = engine._f64(self.W_fm)
                st["W_lift"] = engine._f64(self.W_lift)
                st["eval_strips"] = torch.as_tensor(self.eval_strips, dtype=torch.int64, device=dev)
                st["u_heave_L"] = engine._f64(self.u_heave_L)
                st["u_heave_R"] = engine._f64(self.u_heave_R)
                st["u_pitch_L"] = engine._f64(self.u_pitch_L)
                st["u_pitch_R"] = engine._f64(self.u_pitch_R)
        return st

    def _wake_matrices_batch_gpu(self, k_arr: np.ndarray, *, tail_on: bool):
        """Return batched complex wake influence ``W_batch`` [1/m] on GPU, shape ``(n_k, n_panels, n_strips)``."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        st = self._ensure_gpu_solve_state()
        dev = engine._dev()
        n_k = int(k_arr.size)
        n_p = self.rings.n_panels
        n_s = self.rings.n_strips
        n_rows = self.n_wake_rows

        omegas = 2.0 * k_arr * self.V_inf / self.c_ref
        row_idx = torch.arange(1, n_rows + 1, dtype=torch.float64, device=dev)
        omega_g = engine._f64(omegas)
        ang = -omega_g[:, None] * (row_idx[None, :] * self.dt)
        lags_re = torch.cos(ang)
        lags_im = torch.sin(ang)

        # Contract A_bw (n_p, n_rows, n_s) with lags (n_k, n_rows) via two float64 GEMMs.
        A_bw_2d = st["A_bw"].to(torch.float64).permute(1, 0, 2).reshape(n_rows, n_p * n_s)
        W_re = (lags_re @ A_bw_2d).reshape(n_k, n_p, n_s)
        W_im = (lags_im @ A_bw_2d).reshape(n_k, n_p, n_s)
        W_batch = torch.complex(W_re, W_im)

        if tail_on:
            A_bw_steady_c = st["A_bw_steady"].to(torch.complex128)
            for ik in range(n_k):
                kf = float(k_arr[ik])
                if kf == 0.0:
                    W_batch[ik] = A_bw_steady_c
                else:
                    omega = float(omegas[ik])
                    kappa = omega / self.V_inf
                    phase0 = complex(np.exp(-1j * omega * (n_rows + 0.5) * self.dt))
                    W_t = RK.harmonic_tail_normal_influence_gpu(
                        st["cp"], st["normal_bc"], st["tail_left"], st["tail_right"],
                        st["wake_dir"], kappa, self.rc_wake, ground=self.ground, out_tensor=True,
                    )
                    W_batch[ik] = W_batch[ik] + phase0 * W_t
        return W_batch

    def _solve_linear_batch_gpu(self, W_batch, rhs_batch, solve_method: str):
        """Solve batched ``(A_bb + W[ik] E_te^T) Gamma[ik] = rhs[ik]`` on GPU in complex128."""
        import torch

        from ventorum.gpu import engine

        st = self._ensure_gpu_solve_state()
        dev = engine._dev()
        n_k, n_p, n_s = int(W_batch.shape[0]), int(W_batch.shape[1]), int(W_batch.shape[2])
        n_r = int(rhs_batch.shape[2])
        te_idx_g = st["te_idx"]

        if solve_method == "woodbury":
            lu_Abb = st["lu_Abb"]
            piv_Abb = st["piv_Abb"]
            bytes_per_k = max(1, n_p * (2 * (n_s + n_r)) * 8)
            k_chunk = max(1, min(n_k, _GPU_SOLVE_BATCH_BYTES // bytes_per_k))
            eye_s = torch.eye(n_s, dtype=torch.complex128, device=dev).unsqueeze(0)
            out = torch.empty((n_k, n_p, n_r), dtype=torch.complex128, device=dev)
            for k0 in range(0, n_k, k_chunk):
                k1 = min(n_k, k0 + k_chunk)
                nc = k1 - k0
                Wc = W_batch[k0:k1]
                rc = rhs_batch[k0:k1]
                kw = nc * n_s
                kr = nc * n_r
                # Solve A_bb [W_re, W_im, rhs_re, rhs_im] for all k in the chunk in one real DTRSM call.
                B = torch.cat(
                    [
                        Wc.real.permute(1, 0, 2).reshape(n_p, kw),
                        Wc.imag.permute(1, 0, 2).reshape(n_p, kw),
                        rc.real.permute(1, 0, 2).reshape(n_p, kr),
                        rc.imag.permute(1, 0, 2).reshape(n_p, kr),
                    ],
                    dim=1,
                ).contiguous()
                sol = torch.linalg.lu_solve(lu_Abb, piv_Abb, B)
                Z = torch.complex(
                    sol[:, :kw].reshape(n_p, nc, n_s).permute(1, 0, 2),
                    sol[:, kw : 2 * kw].reshape(n_p, nc, n_s).permute(1, 0, 2),
                )
                Y = torch.complex(
                    sol[:, 2 * kw : 2 * kw + kr].reshape(n_p, nc, n_r).permute(1, 0, 2),
                    sol[:, 2 * kw + kr :].reshape(n_p, nc, n_r).permute(1, 0, 2),
                )
                C_cap = eye_s + Z[:, te_idx_g, :]
                c_sol = torch.linalg.solve(C_cap, Y[:, te_idx_g, :])
                out[k0:k1] = Y - torch.bmm(Z, c_sol)
            return out

        if solve_method == "direct":
            A_bb_c = st["A_bb"].to(torch.complex128)
            bytes_per_k = max(1, n_p * n_p * 16)
            k_chunk = max(1, min(n_k, _GPU_SOLVE_BATCH_BYTES // bytes_per_k))
            out = torch.empty((n_k, n_p, n_r), dtype=torch.complex128, device=dev)
            for k0 in range(0, n_k, k_chunk):
                k1 = min(n_k, k0 + k_chunk)
                nc = k1 - k0
                A_sys = A_bb_c.unsqueeze(0).expand(nc, n_p, n_p).clone()
                A_sys[:, :, te_idx_g] += W_batch[k0:k1]
                out[k0:k1] = torch.linalg.solve(A_sys, rhs_batch[k0:k1])
            return out

        raise InputError(f"solve_method must be 'woodbury' or 'direct', got {solve_method!r}.")

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

        A_bb = ring_normal_influence(
            rings.cp, rings.normal_bc, rings.corners, self.rc_ring,
            side_nodes=rings.side_nodes, side_counts=rings.side_counts,
        )
        A_bw_steady = steady_wake_normal_influence(
            rings.cp, rings.normal_bc, rings.te_left, rings.te_right, rings.wake_dir, self.rc_wake,
        )
        if self.img is not None:
            A_bb += self.img.sign * ring_normal_influence(
                rings.cp, rings.normal_bc, self.img.corners, self.rc_ring,
                side_nodes=self.img.side_nodes, side_counts=self.img.side_counts,
            )
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

        # Physical ring surface areas, centroids and bent moment vectors for apparent-mass loads.
        self.area_vec, self.ring_centroid, self.area_moment_vec = _physical_ring_areas(
            lat, rings, ref_point=self.ref_point,
        )
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

    def _assemble_travelling_wake_one_motion(
        self,
        u_L: np.ndarray,
        u_R: np.ndarray,
        u_L_img: np.ndarray | None,
        u_R_img: np.ndarray | None,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(dA_wake, dFM_wake, dSL_wake)`` for one motion on the CPU."""
        rings = self.rings
        n_p = rings.n_panels
        n_s = rings.n_strips
        n_rows = self.n_wake_rows
        m = self.P_load.shape[0]
        g_te = self.Gamma_0[rings.te_indices]
        inv_2eps = 0.5 / eps

        dA_lead = np.zeros((n_p, n_rows), dtype=np.float64)
        dA_trail = np.zeros((n_p, n_rows), dtype=np.float64)
        dFM_lead = np.zeros((6, n_rows), dtype=np.float64)
        dFM_trail = np.zeros((6, n_rows), dtype=np.float64)
        dSL_lead = np.zeros((n_s, n_rows), dtype=np.float64)
        dSL_trail = np.zeros((n_s, n_rows), dtype=np.float64)

        for m0 in range(0, n_rows, _WAKE_ROW_CHUNK):
            m1 = min(n_rows, m0 + _WAKE_ROW_CHUNK)
            n_chunk = m1 - m0
            w_base = _build_wake_row_corners(
                rings.te_left, rings.te_right, rings.wake_dir, m0, m1, self.dx_w,
            ).reshape(n_chunk, n_s, 4, 3)
            img_base = (
                None
                if self.img is None
                else _build_wake_row_corners(
                    self.img.te_left, self.img.te_right, self.img.wake_dir, m0, m1, self.dx_w,
                ).reshape(n_chunk, n_s, 4, 3)
            )

            for is_lead in (True, False):
                r_off = 1 if (is_lead and m0 == 0) else 0
                r0 = m0 + r_off
                n_sub = m1 - r0
                if n_sub <= 0:
                    continue
                w_sub = w_base[r_off:].reshape(-1, 4, 3)
                du = np.zeros((n_sub, n_s, 4, 3), dtype=np.float64)
                if is_lead:
                    du[:, :, 0, :] = u_L[None, :, :]
                    du[:, :, 1, :] = u_R[None, :, :]
                else:
                    du[:, :, 2, :] = u_R[None, :, :]
                    du[:, :, 3, :] = u_L[None, :, :]
                du_flat = du.reshape(-1, 4, 3)
                w_plus = w_sub + eps * du_flat
                w_minus = w_sub - eps * du_flat

                img_plus = None
                img_minus = None
                if img_base is not None:
                    img_sub = img_base[r_off:].reshape(-1, 4, 3)
                    du_i = np.zeros((n_sub, n_s, 4, 3), dtype=np.float64)
                    if is_lead:
                        du_i[:, :, 0, :] = u_L_img[None, :, :]
                        du_i[:, :, 1, :] = u_R_img[None, :, :]
                    else:
                        du_i[:, :, 2, :] = u_R_img[None, :, :]
                        du_i[:, :, 3, :] = u_L_img[None, :, :]
                    du_i_flat = du_i.reshape(-1, 4, 3)
                    img_plus = img_sub + eps * du_i_flat
                    img_minus = img_sub - eps * du_i_flat

                rc_sub = self._tile_wake_rc(self.rc_wake, n_sub)
                ap = ring_normal_influence(rings.cp, rings.normal_bc, w_plus, rc_sub)
                am = ring_normal_influence(rings.cp, rings.normal_bc, w_minus, rc_sub)
                if img_plus is not None:
                    ap += self.img.sign * ring_normal_influence(rings.cp, rings.normal_bc, img_plus, rc_sub)
                    am += self.img.sign * ring_normal_influence(rings.cp, rings.normal_bc, img_minus, rc_sub)
                da_blk = ((ap - am) * inv_2eps).reshape(n_p, n_sub, n_s) @ g_te
                if is_lead:
                    dA_lead[:, r0:m1] = da_blk
                else:
                    dA_trail[:, r0:m1] = da_blk

                fm_blk = np.zeros((6, n_sub), dtype=np.float64)
                sl_blk = np.zeros((n_s, n_sub), dtype=np.float64)
                for i0 in range(0, m, _LOAD_POINT_CHUNK):
                    i1 = min(m, i0 + _LOAD_POINT_CHUNK)
                    Pc = self.P_load[i0:i1]
                    rc_l = self._tile_wake_rc(_slice_rc_rows(self.rc_load_wake, i0, i1), n_sub)
                    vp = ring_velocity(Pc, w_plus, rc_l)
                    vm = ring_velocity(Pc, w_minus, rc_l)
                    if img_plus is not None:
                        vp += self.img.sign * ring_velocity(Pc, img_plus, rc_l)
                        vm += self.img.sign * ring_velocity(Pc, img_minus, rc_l)
                    dv = ((vp - vm) * inv_2eps).reshape(i1 - i0, n_sub, n_s, 3)
                    dv_row = np.einsum("mrsi,s->mri", dv, g_te)
                    self._contract_load_vel(dv_row, i0, i1, fm_blk, sl_blk)
                if is_lead:
                    dFM_lead[:, r0:m1] = fm_blk
                    dSL_lead[:, r0:m1] = sl_blk
                else:
                    dFM_trail[:, r0:m1] = fm_blk
                    dSL_trail[:, r0:m1] = sl_blk

        # Combine trailing edge of row j-1 and leading edge of row j at station j = 1 .. n_rows.
        dA = dA_trail.copy()
        dFM = dFM_trail.copy()
        dSL = dSL_trail.copy()
        if n_rows > 1:
            dA[:, :-1] += dA_lead[:, 1:]
            dFM[:, :-1] += dFM_lead[:, 1:]
            dSL[:, :-1] += dSL_lead[:, 1:]

        # Cancel the trailing spanwise segment (tail_right -> tail_left) of the last discrete row
        # by adding the matching opposite closing segment (tail_left -> tail_right).
        tl_p = self.tail_left + eps * u_L
        tr_p = self.tail_right + eps * u_R
        tl_m = self.tail_left - eps * u_L
        tr_m = self.tail_right - eps * u_R
        vsp = segment_velocity(rings.cp, tl_p, tr_p, self.rc_wake)
        vsm = segment_velocity(rings.cp, tl_m, tr_m, self.rc_wake)
        if self.img is not None:
            itl_p = self.img_tail_left + eps * u_L_img
            itr_p = self.img_tail_right + eps * u_R_img
            itl_m = self.img_tail_left - eps * u_L_img
            itr_m = self.img_tail_right - eps * u_R_img
            vsp += self.img.sign * segment_velocity(rings.cp, itl_p, itr_p, self.rc_wake)
            vsm += self.img.sign * segment_velocity(rings.cp, itl_m, itr_m, self.rc_wake)
        dA[:, -1] += np.einsum("msi,mi,s->m", (vsp - vsm) * inv_2eps, rings.normal_bc, g_te)

        fm_sp = np.zeros((6, 1), dtype=np.float64)
        sl_sp = np.zeros((n_s, 1), dtype=np.float64)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc = self.P_load[i0:i1]
            rc_l = _slice_rc_rows(self.rc_load_wake, i0, i1)
            vsp_l = segment_velocity(Pc, tl_p, tr_p, rc_l)
            vsm_l = segment_velocity(Pc, tl_m, tr_m, rc_l)
            if self.img is not None:
                vsp_l += self.img.sign * segment_velocity(Pc, itl_p, itr_p, rc_l)
                vsm_l += self.img.sign * segment_velocity(Pc, itl_m, itr_m, rc_l)
            dv_sp = np.einsum("msi,s->mi", (vsp_l - vsm_l) * inv_2eps, g_te)[:, None, :]
            self._contract_load_vel(dv_sp, i0, i1, fm_sp, sl_sp)
        dFM[:, -1] += fm_sp[:, 0]
        dSL[:, -1] += sl_sp[:, 0]

        return dA, dFM, dSL

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
        self.dA_wake_heave = None
        self.dA_wake_pitch = None
        self.dFM_wake_heave = None
        self.dSL_wake_heave = None
        self.dFM_wake_pitch = None
        self.dSL_wake_pitch = None
        self.u_heave_L = None
        self.u_heave_R = None
        self.u_pitch_L = None
        self.u_pitch_R = None
        self.u_heave_L_img = None
        self.u_heave_R_img = None
        self.u_pitch_L_img = None
        self.u_pitch_R_img = None

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
        self._init_travelling_wake_displacements()

        # 2. Central finite differences of the steady system and steady induced velocity.
        h_min = None
        if self.ground is not None:
            pts_list = [self.lattice.all_points(), rings.corners.reshape(-1, 3)]
            if rings.side_nodes is not None:
                pts_list.append(rings.side_nodes.reshape(-1, 3))
            pts = np.vstack(pts_list)
            h_min = float(np.min(self.ground.height(pts)))
            if h_min <= 0.0:
                h_min = float(np.min(self.ground.height(self.lattice.all_points())))

        # Heave rigid k = 0 loaded-mean-state terms (ground effect only; zero in free air).
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
        else:
            eps_h = _DEFAULT_FD_STEP_H_FRAC * self.c_ref if fd_step_h is None else float(fd_step_h)
            if not math.isfinite(eps_h) or eps_h <= 0.0:
                raise InputError(f"fd_step_h must be finite and > 0, got {eps_h!r}.")

        # Pitch rigid k = 0 loaded-mean-state terms (ground effect and/or free-stream-aligned wake).
        eps_th = _DEFAULT_FD_STEP_THETA if fd_step_theta is None else float(fd_step_theta)
        if self.ground is not None and fd_step_theta is None:
            max_lever = max(float(np.max(np.linalg.norm(self.lattice.all_points() - self.ref_point, axis=1))), 1e-6)
            eps_th = min(eps_th, 0.01 * h_min / max_lever)
        if not math.isfinite(eps_th) or eps_th <= 0.0:
            raise InputError(f"fd_step_theta must be finite and > 0, got {eps_th!r}.")

        if self.ground is not None or self._wake_rotates_with_alpha:
            g_tp, wd_tp = self._pitched_steady_state(+eps_th)
            g_tm, wd_tm = self._pitched_steady_state(-eps_th)
            dA_dth_G0 = (
                self._varying_steady_matvec(g_tp, wd_tp) - self._varying_steady_matvec(g_tm, wd_tm)
            ) / (2.0 * eps_th)
            self.rhs_pitch_geo = -dA_dth_G0

            vp_th = self._varying_steady_load_vel(g_tp, wd_tp)
            vm_th = self._varying_steady_load_vel(g_tm, wd_tm)
            self.dv_load_pitch_geo = (vp_th - vm_th) / (2.0 * eps_th)

        # 3. Precompute per-station travelling-wave discrete wake derivatives for heave and pitch.
        self.dA_wake_heave, self.dFM_wake_heave, self.dSL_wake_heave = (
            self._assemble_travelling_wake_one_motion(
                self.u_heave_L, self.u_heave_R, self.u_heave_L_img, self.u_heave_R_img, eps_h / self.c_ref,
            )
        )
        self.dA_wake_pitch, self.dFM_wake_pitch, self.dSL_wake_pitch = (
            self._assemble_travelling_wake_one_motion(
                self.u_pitch_L, self.u_pitch_R, self.u_pitch_L_img, self.u_pitch_R_img, eps_th,
            )
        )

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
            ring_normal_influence(
                rings.cp, rings.normal_bc, img.corners, self.rc_ring,
                side_nodes=img.side_nodes, side_counts=img.side_counts,
            ) @ self.Gamma_0
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
                ring_normal_influence(
                    rings.cp, rings.normal_bc, img.corners, self.rc_ring,
                    side_nodes=img.side_nodes, side_counts=img.side_counts,
                ) @ self.Gamma_0
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
                vr_i = ring_velocity(
                    Pc, img.corners, rc_r, side_nodes=img.side_nodes, side_counts=img.side_counts,
                )
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

            vr = ring_velocity(
                Pc, rings.corners, rc_r, side_nodes=rings.side_nodes, side_counts=rings.side_counts,
            )
            vw = steady_wake_velocity(Pc, rings.te_left, rings.te_right, rings.wake_dir, rc_w)
            if self.img is not None:
                vr += self.img.sign * ring_velocity(
                    Pc, self.img.corners, rc_r, side_nodes=self.img.side_nodes, side_counts=self.img.side_counts,
                )
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
        if self.device == "gpu" and self._gpu_state is not None and "W_fm" in self._gpu_state.tensors:
            import torch

            from ventorum.gpu import engine
            from ventorum.gpu import ring_kernels as RK

            st = self._gpu_state.tensors
            dev = engine._dev()
            n_s = self.rings.n_strips
            m = self.P_load.shape[0]
            FM_tail_g = torch.zeros((6, n_s), dtype=torch.complex128, device=dev)
            SL_tail_g = torch.zeros((n_s, n_s), dtype=torch.complex128, device=dev)
            P_load_g = st["P_load"]
            for i0 in range(0, m, _LOAD_POINT_CHUNK):
                i1 = min(m, i0 + _LOAD_POINT_CHUNK)
                Pc_g = P_load_g[i0:i1].contiguous()
                rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
                vt_g = RK.harmonic_tail_velocity_gpu(
                    Pc_g, st["tail_left"], st["tail_right"], st["wake_dir"], kappa, rc_w,
                    ground=self.ground, out_tensor=True,
                )
                self._contract_load_vel_gpu(vt_g, i0, i1, FM_tail_g, SL_tail_g)
            FM_tail_g.mul_(phase0)
            SL_tail_g.mul_(phase0)
            return FM_tail_g.cpu().numpy(), SL_tail_g.cpu().numpy()

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

        if self.device == "gpu" and self._gpu_state is not None and "A_bw" in self._gpu_state.tensors:
            W_g = self._wake_matrices_batch_gpu(np.array([float(k)], dtype=np.float64), tail_on=tail_on)[0]
            return W_g.cpu().numpy()

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

    def _travelling_wake_rhs_delta(self, k: float) -> tuple[np.ndarray, np.ndarray]:
        """Return complex travelling-minus-rigid steady-wake normal-velocity deltas ``(dn_h, dn_p)``."""
        n_p = self.rings.n_panels
        if not self.has_mean_load or k == 0.0:
            z = np.zeros(n_p, dtype=np.complex128)
            return z, z.copy()

        omega = 2.0 * k * self.V_inf / self.c_ref
        kappa = omega / self.V_inf
        phi = np.expm1(-1j * omega * np.arange(1, self.n_wake_rows + 1, dtype=np.float64) * self.dt)
        dn_h = (self.dA_wake_heave @ phi).astype(np.complex128)
        dn_p = (self.dA_wake_pitch @ phi).astype(np.complex128)

        if (
            self.device == "gpu"
            and self._gpu_state is not None
            and "cp" in self._gpu_state.tensors
            and "Gamma_0" in self._gpu_state.tensors
        ):
            import torch

            from ventorum.gpu import ring_kernels as RK

            st = self._gpu_state.tensors
            g_te_c = st["Gamma_0"][st["te_idx"]].to(torch.complex128)
            a_fh = RK.displaced_steady_tail_normal_influence_gpu(
                st["cp"], st["normal_bc"], st["tail_left"], st["tail_right"], st["wake_dir"],
                st["u_heave_L"], st["u_heave_R"], kappa, self.wake_length, self.rc_wake,
                ground=self.ground, out_tensor=True,
            )
            a_fp = RK.displaced_steady_tail_normal_influence_gpu(
                st["cp"], st["normal_bc"], st["tail_left"], st["tail_right"], st["wake_dir"],
                st["u_pitch_L"], st["u_pitch_R"], kappa, self.wake_length, self.rc_wake,
                ground=self.ground, out_tensor=True,
            )
            dn_h = dn_h + (a_fh @ g_te_c).cpu().numpy()
            dn_p = dn_p + (a_fp @ g_te_c).cpu().numpy()
            return dn_h, dn_p

        rings = self.rings
        g_te = self.Gamma_0[rings.te_indices]
        a_fh = displaced_steady_tail_normal_influence(
            rings.cp, rings.normal_bc, self.tail_left, self.tail_right, rings.wake_dir,
            self.u_heave_L, self.u_heave_R, kappa, self.wake_length, self.rc_wake,
        )
        a_fp = displaced_steady_tail_normal_influence(
            rings.cp, rings.normal_bc, self.tail_left, self.tail_right, rings.wake_dir,
            self.u_pitch_L, self.u_pitch_R, kappa, self.wake_length, self.rc_wake,
        )
        if self.img is not None:
            a_fh += self.img.sign * displaced_steady_tail_normal_influence(
                rings.cp, rings.normal_bc, self.img_tail_left, self.img_tail_right, self.img.wake_dir,
                self.u_heave_L_img, self.u_heave_R_img, kappa, self.wake_length, self.rc_wake,
            )
            a_fp += self.img.sign * displaced_steady_tail_normal_influence(
                rings.cp, rings.normal_bc, self.img_tail_left, self.img_tail_right, self.img.wake_dir,
                self.u_pitch_L_img, self.u_pitch_R_img, kappa, self.wake_length, self.rc_wake,
            )
        dn_h = dn_h + a_fh @ g_te
        dn_p = dn_p + a_fp @ g_te
        return dn_h, dn_p

    def _travelling_wake_load_delta(
        self,
        k: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return complex travelling-minus-rigid steady-wake load deltas ``(dFM_h, dSL_h, dFM_p, dSL_p)``."""
        n_s = self._n_strips
        if not self.has_mean_load or k == 0.0:
            z6 = np.zeros(6, dtype=np.complex128)
            zs = np.zeros(n_s, dtype=np.complex128)
            return z6, zs, z6.copy(), zs.copy()

        omega = 2.0 * k * self.V_inf / self.c_ref
        kappa = omega / self.V_inf
        phi = np.expm1(-1j * omega * np.arange(1, self.n_wake_rows + 1, dtype=np.float64) * self.dt)
        dFM_h = (self.dFM_wake_heave @ phi).astype(np.complex128)
        dSL_h = (self.dSL_wake_heave @ phi).astype(np.complex128)
        dFM_p = (self.dFM_wake_pitch @ phi).astype(np.complex128)
        dSL_p = (self.dSL_wake_pitch @ phi).astype(np.complex128)

        m = self.P_load.shape[0]
        if (
            self.device == "gpu"
            and self._gpu_state is not None
            and "W_fm" in self._gpu_state.tensors
            and "Gamma_0" in self._gpu_state.tensors
        ):
            import torch

            from ventorum.gpu import engine
            from ventorum.gpu import ring_kernels as RK

            st = self._gpu_state.tensors
            dev = engine._dev()
            g_te_c = st["Gamma_0"][st["te_idx"]].to(torch.complex128)
            P_load_g = st["P_load"]
            fm_fh = torch.zeros((6, 1), dtype=torch.complex128, device=dev)
            sl_fh = torch.zeros((n_s, 1), dtype=torch.complex128, device=dev)
            fm_fp = torch.zeros((6, 1), dtype=torch.complex128, device=dev)
            sl_fp = torch.zeros((n_s, 1), dtype=torch.complex128, device=dev)
            for i0 in range(0, m, _LOAD_POINT_CHUNK):
                i1 = min(m, i0 + _LOAD_POINT_CHUNK)
                Pc_g = P_load_g[i0:i1].contiguous()
                rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
                vh_g = RK.displaced_steady_tail_velocity_gpu(
                    Pc_g, st["tail_left"], st["tail_right"], st["wake_dir"],
                    st["u_heave_L"], st["u_heave_R"], kappa, self.wake_length, rc_w,
                    ground=self.ground, out_tensor=True,
                )
                vp_g = RK.displaced_steady_tail_velocity_gpu(
                    Pc_g, st["tail_left"], st["tail_right"], st["wake_dir"],
                    st["u_pitch_L"], st["u_pitch_R"], kappa, self.wake_length, rc_w,
                    ground=self.ground, out_tensor=True,
                )
                dvh = torch.einsum("msi,s->mi", vh_g, g_te_c).unsqueeze(1)
                dvp = torch.einsum("msi,s->mi", vp_g, g_te_c).unsqueeze(1)
                self._contract_load_vel_gpu(dvh, i0, i1, fm_fh, sl_fh)
                self._contract_load_vel_gpu(dvp, i0, i1, fm_fp, sl_fp)
            dFM_h = dFM_h + fm_fh[:, 0].cpu().numpy()
            dSL_h = dSL_h + sl_fh[:, 0].cpu().numpy()
            dFM_p = dFM_p + fm_fp[:, 0].cpu().numpy()
            dSL_p = dSL_p + sl_fp[:, 0].cpu().numpy()
            return dFM_h, dSL_h, dFM_p, dSL_p

        rings = self.rings
        g_te = self.Gamma_0[rings.te_indices]
        fm_fh = np.zeros((6, 1), dtype=np.complex128)
        sl_fh = np.zeros((n_s, 1), dtype=np.complex128)
        fm_fp = np.zeros((6, 1), dtype=np.complex128)
        sl_fp = np.zeros((n_s, 1), dtype=np.complex128)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc = self.P_load[i0:i1]
            rc_w = _slice_rc_rows(self.rc_load_wake, i0, i1)
            vh = displaced_steady_tail_velocity(
                Pc, self.tail_left, self.tail_right, rings.wake_dir,
                self.u_heave_L, self.u_heave_R, kappa, self.wake_length, rc_w,
            )
            vp = displaced_steady_tail_velocity(
                Pc, self.tail_left, self.tail_right, rings.wake_dir,
                self.u_pitch_L, self.u_pitch_R, kappa, self.wake_length, rc_w,
            )
            if self.img is not None:
                vh += self.img.sign * displaced_steady_tail_velocity(
                    Pc, self.img_tail_left, self.img_tail_right, self.img.wake_dir,
                    self.u_heave_L_img, self.u_heave_R_img, kappa, self.wake_length, rc_w,
                )
                vp += self.img.sign * displaced_steady_tail_velocity(
                    Pc, self.img_tail_left, self.img_tail_right, self.img.wake_dir,
                    self.u_pitch_L_img, self.u_pitch_R_img, kappa, self.wake_length, rc_w,
                )
            dvh = np.einsum("msi,s->mi", vh, g_te)[:, None, :]
            dvp = np.einsum("msi,s->mi", vp, g_te)[:, None, :]
            self._contract_load_vel(dvh, i0, i1, fm_fh, sl_fh)
            self._contract_load_vel(dvp, i0, i1, fm_fp, sl_fp)
        dFM_h = dFM_h + fm_fh[:, 0]
        dSL_h = dSL_h + sl_fh[:, 0]
        dFM_p = dFM_p + fm_fp[:, 0]
        dSL_p = dSL_p + sl_fp[:, 0]
        return dFM_h, dSL_h, dFM_p, dSL_p

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

        if self.has_mean_load and k != 0.0:
            dn_h, dn_p = self._travelling_wake_rhs_delta(k)
            rhs_h = rhs_h - dn_h
            rhs_p = rhs_p - dn_p

        return np.column_stack([rhs_h, rhs_p]).astype(np.complex128)

    def _solve_linear(self, W: np.ndarray, rhs: np.ndarray, solve_method: str) -> np.ndarray:
        """Solve ``(A_bb + W E_te^T) Gamma = rhs`` by Woodbury or direct complex LU."""
        if self.device == "gpu" and self._gpu_state is not None and "lu_Abb" in self._gpu_state.tensors:
            import torch

            from ventorum.gpu import engine

            dev = engine._dev()
            W_g = torch.as_tensor(np.ascontiguousarray(W, dtype=np.complex128), device=dev).unsqueeze(0)
            rhs_g = torch.as_tensor(np.ascontiguousarray(rhs, dtype=np.complex128), device=dev).unsqueeze(0)
            return self._solve_linear_batch_gpu(W_g, rhs_g, solve_method)[0].cpu().numpy()

        te_idx = self.rings.te_indices
        n_s = self.rings.n_strips
        if solve_method == "woodbury":
            if self.lu_Abb is None:
                self.lu_Abb = sl.lu_factor(self.A_bb)
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
        FM_extra: np.ndarray | None = None,
        SL_extra: np.ndarray | None = None,
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
            if FM_extra is not None:
                FM_dV += FM_extra
            F_tot += FM_dV[:3]
            M_tot += FM_dV[3:]

            sl_direct = np.einsum("mi,mi->m", self.W_lift, dv_direct).astype(np.complex128)
            np.add.at(strip_lift, self.eval_strips, sl_direct)
            strip_lift += self.SL_bound_mat @ Gamma_hat
            if SL_wake_k is not None:
                strip_lift += SL_wake_k @ Gamma_hat[rings.te_indices]
            if SL_extra is not None:
                strip_lift += SL_extra

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
            coef_am = self.rho * 1j * omega * Gamma_hat
            F_am = coef_am[:, None] * self.area_vec
            F_tot += np.sum(F_am, axis=0)
            if self.area_moment_vec is not None:
                M_tot += np.sum(coef_am[:, None] * self.area_moment_vec, axis=0)
            else:
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

        G_batch_all: np.ndarray | None = None
        if self.device == "gpu":
            from ventorum.gpu import memory as gpu_memory
            from ventorum.gpu.pipeline import is_out_of_memory

            message = ""
            for _attempt in range(2):
                try:
                    import torch

                    from ventorum.gpu import engine

                    W_batch_g = self._wake_matrices_batch_gpu(k_arr, tail_on=tail_on)
                    rhs_np = np.stack([self.build_rhs(float(kv)) for kv in k_arr], axis=0)
                    rhs_batch_g = torch.as_tensor(
                        np.ascontiguousarray(rhs_np, dtype=np.complex128), device=engine._dev(),
                    )
                    G_batch_all = self._solve_linear_batch_gpu(W_batch_g, rhs_batch_g, solve_method).cpu().numpy()
                    break
                except Exception as exc:  # noqa: BLE001 - only out-of-memory errors are caught
                    if not is_out_of_memory(exc):
                        raise
                    message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                if self._gpu_state is not None:
                    self._gpu_state.release_gpu()
                    self._gpu_state = None
                gpu_memory.release()
            if G_batch_all is None:
                warnings.warn(
                    f"The GPU ran out of memory twice ({message}). The harmonic solve falls back to the CPU.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.device = "cpu"
                self.precision = "float64"
                if NOTE_GPU_OUT_OF_MEMORY not in self.notes:
                    self.notes.append(NOTE_GPU_OUT_OF_MEMORY)

        for ik, k_val in enumerate(k_arr):
            kf = float(k_val)
            omega = 2.0 * kf * self.V_inf / self.c_ref
            kappa = omega / self.V_inf
            if G_batch_all is not None:
                G_mat = G_batch_all[ik]
            else:
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

            if self.has_mean_load and kf != 0.0:
                dFM_h, dSL_h, dFM_p, dSL_p = self._travelling_wake_load_delta(kf)
            else:
                dFM_h = dSL_h = dFM_p = dSL_p = None

            heave_c[ik], heave_scl[ik] = self._evaluate_linearised_loads(
                G_heave, k=kf, dU_load=dU_heave_load, dv_geo_load=self.dv_load_heave_geo,
                rotate_mean_geometry=False, FM_wake_k=FM_wake_k, SL_wake_k=SL_wake_k,
                include_apparent_mass=True, include_te_correction=te_on,
                FM_extra=dFM_h, SL_extra=dSL_h,
            )
            pitch_c[ik], pitch_scl[ik] = self._evaluate_linearised_loads(
                G_pitch, k=kf, dU_load=dU_pitch_load, dv_geo_load=self.dv_load_pitch_geo,
                rotate_mean_geometry=True, FM_wake_k=FM_wake_k, SL_wake_k=SL_wake_k,
                include_apparent_mass=True, include_te_correction=te_on,
                FM_extra=dFM_p, SL_extra=dSL_p,
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
            device=self.device,
            precision=self.precision,
            notes=list(self.notes),
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
