# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Prescribed-wake time-domain vortex-ring lattice solver.

This module marches a :class:`ventorum.unsteady.rings.RingLattice` along a
prescribed six-axis rigid-body path with optional moving control surfaces in
incompressible flow, in free air or over a flat ground plane. The wake is
prescribed: at each time step a new row of vortex rings leaves the
instantaneous trailing-edge closure with the trailing-edge ring strength of
the previous step, and all shed rows move with the free stream without
induced-velocity deformation.

Formulation
-----------
1. **Frames and six-axis rigid-body motion.** Geometry axes are x aft, y right,
   z up. The mean flight path lies along ``V_0 = V_inf * d_0`` [m/s] (parallel
   to the ground when flat-ground effect is active). Translations
   ``(x, y, z)`` [m] displace ``ref_point`` along the mean-flight basis
   ``(e_x, e_y, e_z)``, where ``e_x = d_0`` is aft along the mean flight path,
   ``e_z`` is ``lift_direction(alpha)`` in free air or ``ground.normal`` in
   ground effect, and ``e_y = e_z x e_x`` points right. Rotations
   ``(roll, pitch, yaw)`` [rad] about ``ref_point`` follow the intrinsic 3-2-1
   sequence of flight mechanics (yaw, then pitch, then roll, each about the
   axis of the frame after the previous rotation) with AVL signs: ``roll > 0``
   right wing down, ``pitch > 0`` nose up, ``yaw > 0`` nose right. The body
   velocity of each point is ``v_trans + omega x (p - x_ref)``, with ``omega``
   computed from the Euler-angle rates of the same 3-2-1 sequence (Etkin and
   Reid 1996, chapter 4). The legacy ``z`` and ``theta`` keywords remain
   supported for heave and pitch.
2. **Moving control surfaces.** Optional time-varying control deflections
   ``controls`` [rad] rotate the nodes of the rings aft of the hinge line about
   each strip hinge axis by the deflection minus the built lattice deflection,
   using the hinge geometry and sealed join weights of
   :mod:`ventorum.geometry.lattice`. Bent side polylines that cross the hinge
   keep their hinge node. The surface velocity of flap points includes the
   hinge rotation rate ``omega_hinge x (p - H)``. The trailing-edge closure
   and the first shed wake row follow the moved flap trailing edge.
3. **Time step and wake shedding.** The default time step is
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
4. **Linear system, moving controls and ground effect.** At each step, the
   impermeability condition at the moved collocation points determines the
   bound vortex-ring circulations.

   * Free air with rigid-body motion only (and stationary geometry in ground
     effect): the ``N x N`` bound influence matrix is invariant and
     LU-factorised once (``O(N^3)`` setup, ``O(N^2)`` back-substitution per
     step).
   * Moving control surfaces (free air or ground): only the ``M_c`` columns of
     the rings whose corners or side nodes move (flap rings and hinge-crossing
     rings) and the ``M_r`` rows of the moved flap collocation points are
     recomputed in the real bound influence matrix (``O(N (M_r + M_c))``
     kernel work per step where deflections change), followed by ``O(N^3)`` LU
     refactorisation.
   * Ground effect with wing or flap motion: the mirror images of the wing,
     deflected flap rings and wake follow the moved geometry, so the image
     bound influence matrix is recomputed (``O(N^2)`` kernel work) and the
     ``N x N`` system is refactorised (``O(N^3)``) at every step where pose or
     control deflections change.
5. **Unsteady loads.** At each step, aerodynamic loads combine:

   * Kutta-Joukowski loads on all bound spanwise segments and chordwise legs
     with the leg-force rules of :func:`ventorum.aero.loads.leg_force_rules`
     (rotated with the flap on flap legs) and the full local velocity (onset
     minus surface velocity plus induced velocity of bound rings, wake rings
     and ground images);
   * apparent-mass loads ``rho * (dGamma / dt) * area_vec`` over the physical
     surface area of each bound ring at the current step, using a second-order
     backward difference for ``dGamma / dt`` at steps ``n >= 2`` and a
     first-order backward difference at step ``n = 1`` (zero at ``n = 0``);
   * the trailing-edge boundary load correction of
     :mod:`ventorum.unsteady.harmonic`, ``Delta gamma_TE,s = f_TE,s *
     (Gamma_w1,s - Gamma_TE,s)`` with ``f_TE,s = dx_TE,s / (2 * (dx_TE,s +
     dx_w))`` applied at the moved physical trailing edge for ``n >= 1``.

   Force coefficients ``(CL, CD, CY)`` are projected onto the mean wind axes
   and moment coefficients ``(Cl, Cm, Cn)`` are reported in the body axes of
   the mean state about the instantaneous reference point.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapter 13 (unsteady vortex-ring lattice, wake
  shedding, impulsive start, unsteady Bernoulli loads and method of images).
* B. Etkin and L. D. Reid, "Dynamics of Flight: Stability and Control", 3rd
  ed., Wiley, 1996, chapter 4 (3-2-1 Euler rotation sequence and kinematic
  angular-velocity relation).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg as sl

from ventorum.aero.loads import leg_force_rules, leg_segments, leg_strength
from ventorum.aero.system import (
    GroundPlane,
    freestream_direction,
    lift_direction,
    side_direction,
)
from ventorum.aero.vortex import chain_ends, chain_position
from ventorum.aero.vortex_ring import ring_normal_influence, ring_velocity
from ventorum.core.datatypes import FlightCondition
from ventorum.core.errors import InputError
from ventorum.geometry.lattice import CLAF_MAX, CLAF_MIN, VortexLattice
from ventorum.unsteady.harmonic import (
    NOTE_GPU_OUT_OF_MEMORY,
    _eval_pair_cores,
    _note_gpu_unavailable,
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
# Allowed rigid-body motion axes in the ``motion`` mapping.
_MOTION_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _select_time_device(
    n_steps: int,
    n_bound: int,
    n_strips: int,
    has_ground: bool,
) -> tuple[str, str, list[str]]:
    """Return ``(device, precision, fallback_notes)`` for the time-domain ring solver."""
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
    n_wake = n_steps * n_strips
    work = float(n_steps * n_bound * n_wake) * (2.0 if has_ground else 1.0)
    use, _ = gpu.use_gpu(work, "vlm", n_cases=1, n_panels=None, return_reason=True)
    if use:
        return "gpu", gpu.get_precision(), []
    return "cpu", "float64", []


class _TimeDomainGPUState:
    """Registered container for CUDA arrays held by one :class:`_TimeDomainStepper`.

    The arrays hold the marching state (wake buffers, previous circulation),
    which exists only on the device. A release of all GPU data
    (``ventorum.gpu.release``, or the out-of-memory path of another solve)
    while the march runs must not lose that state. So a release copies the
    storages to the host and frees the device arrays; the next access to
    :attr:`tensors` uploads them again, with the same views, so the march
    continues with the same values. A release during a step (from a gust or
    control callable, or from another thread) waits until the step ends.
    The stepper frees the arrays at its end with ``release_gpu(force=True)``.
    """

    def __init__(self, precision: str) -> None:
        import threading

        from ventorum.gpu import ring_kernels as RK

        self.precision = precision
        self._tensors: dict[str, Any] = {}
        self._host: dict[str, Any] | None = None
        self._host_storages: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._depth = 0
        self._pending = False
        RK.register_device_holder(self)

    @property
    def tensors(self) -> dict[str, Any]:
        """Return the device arrays; upload them again after a release."""
        with self._lock:
            if self._host is not None:
                self._upload_from_host()
            return self._tensors

    def device_bytes(self) -> int:
        """Return total CUDA storage bytes held by this state [B]."""
        from ventorum.gpu import memory as gpu_memory

        return gpu_memory._tensor_bytes(self._tensors, set())

    def step_scope(self):
        """Return a context manager that marks one march step as running."""
        import contextlib

        @contextlib.contextmanager
        def _scope():
            with self._lock:
                self._depth += 1
            try:
                yield
            finally:
                with self._lock:
                    self._depth -= 1
                    if self._depth == 0 and self._pending:
                        self._pending = False
                        self._move_to_host()

        return _scope()

    def _move_to_host(self) -> int:
        """Copy the storages to the host, drop the device arrays and return freed bytes [B]."""
        import torch

        freed = self.device_bytes()
        storages: dict[int, Any] = {}
        host: dict[str, Any] = {}
        for key, val in self._tensors.items():
            if isinstance(val, torch.Tensor) and val.is_cuda:
                st = val.untyped_storage()
                ptr = int(st.data_ptr())
                if ptr not in storages:
                    storages[ptr] = st.cpu()
                host[key] = (ptr, val.dtype, int(val.storage_offset()), tuple(val.shape), tuple(val.stride()))
            else:
                host[key] = ("value", val)
        self._host = host
        self._host_storages = storages
        self._tensors = {}
        return freed

    def _upload_from_host(self) -> None:
        """Upload the host copies again and register the holder again."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        dev = engine._dev()
        dev_storages = {ptr: st.to(device=dev) for ptr, st in self._host_storages.items()}
        tensors: dict[str, Any] = {}
        assert self._host is not None
        for key, meta in self._host.items():
            if meta[0] == "value":
                tensors[key] = meta[1]
                continue
            ptr, dtype, offset, shape, stride = meta
            tensors[key] = torch.empty((0,), dtype=dtype, device=dev).set_(dev_storages[ptr], offset, shape, stride)
        self._tensors = tensors
        self._host = None
        self._host_storages = {}
        RK.register_device_holder(self)

    def release_gpu(self, *, force: bool = False) -> int:
        """Free the CUDA arrays and return the freed bytes [B].

        With *force* (the end of the march) the arrays and their host copies
        are dropped. Without it (a release of all GPU data from outside the
        march) the state moves to the host and comes back on the next use;
        during a step the move waits until the step ends and this returns 0.
        """
        with self._lock:
            if force:
                freed = self.device_bytes()
                self._tensors.clear()
                self._host = None
                self._host_storages = {}
                return freed
            if self._host is not None:
                return 0
            if self._depth > 0:
                from ventorum.gpu import ring_kernels as RK

                self._pending = True
                RK.register_device_holder(self)
                return 0
            return self._move_to_host()


def _to_gpu_rc(rc: np.ndarray | float) -> Any:
    """Upload array core radius *rc* [m] to CUDA float64; keep scalars as float."""
    if np.ndim(rc) == 0:
        return float(rc)
    from ventorum.gpu import engine

    return engine._f64(rc)


def _slice_rc_rows_gpu(rc_g: Any, i0: int, i1: int) -> Any:
    """Slice 2-D CUDA core-radius tensor *rc_g* [m] along the row axis."""
    if not isinstance(rc_g, float) and getattr(rc_g, "ndim", 0) == 2:
        return rc_g[i0:i1]
    return rc_g


def _tile_wake_rc_gpu(rc_g: Any, n_chunk: int) -> Any:
    """Repeat per-strip CUDA wake core radii *rc_g* [m] across *n_chunk* wake rows."""
    if isinstance(rc_g, float):
        return rc_g
    if rc_g.ndim == 1:
        return rc_g.repeat(n_chunk)
    return rc_g.repeat(1, n_chunk)


def _slice_rc_sub_gpu(rc_g: Any, rows_g: Any | None, cols_g: Any | None) -> Any:
    """Slice CUDA core-radius tensor *rc_g* [m] to *rows_g* and *cols_g*."""
    if isinstance(rc_g, float):
        return rc_g
    if rc_g.ndim == 1:
        return rc_g if cols_g is None else rc_g[cols_g]
    if rows_g is None and cols_g is None:
        return rc_g
    if rows_g is None:
        return rc_g[:, cols_g]
    if cols_g is None:
        return rc_g[rows_g, :]
    return rc_g[rows_g[:, None], cols_g[None, :]]


def _rotate_vecs_gpu(vec_g: Any, R_g: Any | None) -> Any:
    """Apply CUDA body-to-reference rotation matrix *R_g* to *vec_g*."""
    if R_g is None:
        return vec_g
    return vec_g @ R_g.T


def _move_wing_geometry_gpu(
    cp_b_g: Any,
    normal_bc_b_g: Any,
    corners_b_g: Any,
    te_left_b_g: Any,
    te_right_b_g: Any,
    side_nodes_b_g: Any | None,
    rp_g: Any,
    x_ref_g: Any,
    R_g: Any | None,
) -> tuple[Any, Any, Any, Any, Any, Any | None]:
    """Return moved CUDA tensors ``(cp, normal_bc, corners, te_left, te_right, side_nodes)``."""
    if R_g is None:
        shift = x_ref_g - rp_g
        cp_g = cp_b_g + shift[None, :]
        normal_bc_g = normal_bc_b_g
        corners_g = corners_b_g + shift[None, None, :]
        te_left_g = te_left_b_g + shift[None, :]
        te_right_g = te_right_b_g + shift[None, :]
        side_nodes_g = None if side_nodes_b_g is None else side_nodes_b_g + shift[None, None, None, :]
        return cp_g, normal_bc_g, corners_g, te_left_g, te_right_g, side_nodes_g

    rt = R_g.T
    cp_g = x_ref_g[None, :] + (cp_b_g - rp_g[None, :]) @ rt
    normal_bc_g = normal_bc_b_g @ rt
    corners_g = x_ref_g[None, None, :] + (corners_b_g - rp_g[None, None, :]) @ rt
    te_left_g = x_ref_g[None, :] + (te_left_b_g - rp_g[None, :]) @ rt
    te_right_g = x_ref_g[None, :] + (te_right_b_g - rp_g[None, :]) @ rt
    side_nodes_g = (
        None
        if side_nodes_b_g is None
        else x_ref_g[None, None, None, :] + (side_nodes_b_g - rp_g[None, None, None, :]) @ rt
    )
    return cp_g, normal_bc_g, corners_g, te_left_g, te_right_g, side_nodes_g


def _rings_to_horseshoe_gpu(gamma_rings_g: Any, n_chord: int) -> Any:
    """Convert CUDA ring circulations [m^2/s] to horseshoe circulations [m^2/s]."""
    import torch

    g = gamma_rings_g.reshape(-1, int(n_chord))
    if n_chord <= 1:
        return gamma_rings_g.clone()
    out = torch.empty_like(g)
    out[:, 0] = g[:, 0]
    out[:, 1:] = g[:, 1:] - g[:, :-1]
    return out.reshape(-1)


def _update_bound_influence_blocks_gpu(
    A_bb_real_g: Any,
    c_m_g: Any,
    r_m_g: Any,
    c_f_g: Any,
    cp_b_g: Any,
    normal_bc_b_g: Any,
    corners_b_g: Any,
    side_nodes_b_g: Any | None,
    side_counts_g: Any | None,
    rc_ring_g: Any,
    *,
    precision: str,
) -> None:
    """Update only the moving rows and columns of CUDA *A_bb_real_g* [1/m] in place."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    sn_cm = None if side_nodes_b_g is None else side_nodes_b_g[c_m_g]
    sc_cm = None if side_counts_g is None else side_counts_g[c_m_g]
    A_bb_real_g[:, c_m_g] = RK.ring_normal_influence_gpu(
        cp_b_g,
        normal_bc_b_g,
        corners_b_g[c_m_g],
        _slice_rc_sub_gpu(rc_ring_g, None, c_m_g),
        side_nodes=sn_cm,
        side_counts=sc_cm,
        ground=None,
        precision=precision,
        out_tensor=True,
    ).to(torch.float64)
    if int(r_m_g.numel()) > 0 and int(c_f_g.numel()) > 0:
        sn_cf = None if side_nodes_b_g is None else side_nodes_b_g[c_f_g]
        sc_cf = None if side_counts_g is None else side_counts_g[c_f_g]
        A_bb_real_g[r_m_g[:, None], c_f_g[None, :]] = RK.ring_normal_influence_gpu(
            cp_b_g[r_m_g],
            normal_bc_b_g[r_m_g],
            corners_b_g[c_f_g],
            _slice_rc_sub_gpu(rc_ring_g, r_m_g, c_f_g),
            side_nodes=sn_cf,
            side_counts=sc_cf,
            ground=None,
            precision=precision,
            out_tensor=True,
        ).to(torch.float64)


def _bound_velocity_at_gpu(
    P_g: Any,
    corners_g: Any,
    side_nodes_g: Any | None,
    side_counts_g: Any | None,
    gamma_g: Any,
    rc_ring_g: Any,
    *,
    ground: GroundPlane | None,
    precision: str,
    image_only: bool = False,
) -> Any:
    """Return CUDA induced velocity [m/s] at *P_g* from bound rings (and optional ground images)."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    m = int(P_g.shape[0])
    v_out_g = torch.zeros((m, 3), dtype=torch.float64, device=P_g.device)
    img_mode = 2 if image_only else (1 if ground is not None else 0)
    for i0 in range(0, m, _LOAD_POINT_CHUNK):
        i1 = min(m, i0 + _LOAD_POINT_CHUNK)
        Pc_g = P_g[i0:i1].contiguous()
        rc_c = _slice_rc_rows_gpu(rc_ring_g, i0, i1)
        vr = RK.ring_velocity_gpu(
            Pc_g,
            corners_g,
            rc_c,
            side_nodes=side_nodes_g,
            side_counts=side_counts_g,
            ground=ground,
            img_mode=img_mode,
            precision=precision,
            out_tensor=True,
        ).to(torch.float64)
        v_out_g[i0:i1] = torch.einsum("mni,n->mi", vr, gamma_g)
    return v_out_g


def _wake_normal_rhs_gpu(
    cp_g: Any,
    normal_bc_g: Any,
    wake_corners_g: Any,
    wake_gamma_g: Any,
    n_active: int,
    rc_wake_g: Any,
    *,
    ground: GroundPlane | None,
    precision: str,
) -> Any:
    """Return CUDA wake normal velocity [m/s] at *cp_g* from the first *n_active* wake rows."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    vn_g = torch.zeros(int(cp_g.shape[0]), dtype=torch.float64, device=cp_g.device)
    for r0 in range(0, n_active, _WAKE_ROW_CHUNK):
        r1 = min(n_active, r0 + _WAKE_ROW_CHUNK)
        n_chunk = r1 - r0
        wc = wake_corners_g[r0:r1].reshape(-1, 4, 3).contiguous()
        wg = wake_gamma_g[r0:r1].reshape(-1)
        rc_c = _tile_wake_rc_gpu(rc_wake_g, n_chunk)
        A_w = RK.ring_normal_influence_gpu(
            cp_g,
            normal_bc_g,
            wc,
            rc_c,
            ground=ground,
            precision=precision,
            out_tensor=True,
        ).to(torch.float64)
        vn_g = vn_g + A_w @ wg
    return vn_g


def _wake_velocity_at_gpu(
    P_g: Any,
    wake_corners_g: Any,
    wake_gamma_g: Any,
    n_active: int,
    rc_wake_g: Any,
    *,
    ground: GroundPlane | None,
    precision: str,
) -> Any:
    """Return CUDA wake induced velocity [m/s] at *P_g* from the first *n_active* wake rows."""
    import torch

    from ventorum.gpu import ring_kernels as RK

    m = int(P_g.shape[0])
    v_out_g = torch.zeros((m, 3), dtype=torch.float64, device=P_g.device)
    for r0 in range(0, n_active, _WAKE_ROW_CHUNK):
        r1 = min(n_active, r0 + _WAKE_ROW_CHUNK)
        n_chunk = r1 - r0
        wc = wake_corners_g[r0:r1].reshape(-1, 4, 3).contiguous()
        wg = wake_gamma_g[r0:r1].reshape(-1)
        for i0 in range(0, m, _LOAD_POINT_CHUNK):
            i1 = min(m, i0 + _LOAD_POINT_CHUNK)
            Pc_g = P_g[i0:i1].contiguous()
            rc_c = _tile_wake_rc_gpu(_slice_rc_rows_gpu(rc_wake_g, i0, i1), n_chunk)
            vw = RK.ring_velocity_gpu(
                Pc_g,
                wc,
                rc_c,
                ground=ground,
                precision=precision,
                out_tensor=True,
            ).to(torch.float64)
            v_out_g[i0:i1] = v_out_g[i0:i1] + torch.einsum("mni,n->mi", vw, wg)
    return v_out_g


def _advect_wake_nodes_rk4_gpu(
    wake_l_g: Any,
    wake_r_g: Any,
    n_lines: int,
    v_base_g: Any,
    gust_fn: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    dt: float,
    g_norm_g: Any | None,
) -> None:
    """Advect the first *n_lines* CUDA wake node lines from *t0* to ``t0 + dt`` [s] by RK4 in place.

    Evaluates the Python *gust_fn* on the host at each RK4 stage, copies the
    velocities to the device, and updates the device wake node buffers in place.
    """
    import torch

    from ventorum.gpu import engine

    n_s = int(v_base_g.shape[0])
    half = n_lines * n_s
    pts_g = torch.cat(
        [wake_l_g[:n_lines].reshape(half, 3), wake_r_g[:n_lines].reshape(half, 3)],
        dim=0,
    )
    v_base_tile_g = v_base_g.repeat(2 * n_lines, 1)

    def _vel(p_g: torch.Tensor, ts: float) -> torch.Tensor:
        vg_host = _eval_gust(gust_fn, p_g.cpu().numpy(), ts)
        vg_g = engine._f64(vg_host)
        if g_norm_g is not None:
            vg_g = vg_g - (vg_g @ g_norm_g)[:, None] * g_norm_g[None, :]
        return v_base_tile_g + vg_g

    half_dt = 0.5 * dt
    k1 = _vel(pts_g, t0)
    k2 = _vel(pts_g + half_dt * k1, t0 + half_dt)
    k3 = _vel(pts_g + half_dt * k2, t0 + half_dt)
    k4 = _vel(pts_g + dt * k3, t0 + dt)
    pts_new_g = pts_g + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    wake_l_g[:n_lines] = pts_new_g[:half].reshape(n_lines, n_s, 3)
    wake_r_g[:n_lines] = pts_new_g[half:].reshape(n_lines, n_s, 3)


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
        Prescribed heave displacement of ``ref_point`` along ``e_z`` [m],
        shape ``(n_steps + 1,)``.
    theta : numpy.ndarray
        Prescribed pitch angle about ``ref_point`` [rad] (positive nose up),
        shape ``(n_steps + 1,)``. Matches ``pitch``.
    z_dot : numpy.ndarray
        Heave velocity [m/s] at each time instant, shape ``(n_steps + 1,)``.
    theta_dot : numpy.ndarray
        Pitch angular rate [rad/s] at each time instant, shape
        ``(n_steps + 1,)``. Matches ``pitch_dot``.
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
    x : numpy.ndarray
        Prescribed streamwise displacement of ``ref_point`` along ``e_x`` [m],
        shape ``(n_steps + 1,)``.
    y : numpy.ndarray
        Prescribed lateral displacement of ``ref_point`` along ``e_y`` [m],
        shape ``(n_steps + 1,)``.
    roll : numpy.ndarray
        Prescribed roll angle about ``ref_point`` [rad] (positive right wing
        down), shape ``(n_steps + 1,)``.
    pitch : numpy.ndarray
        Prescribed pitch angle about ``ref_point`` [rad] (positive nose up),
        shape ``(n_steps + 1,)``.
    yaw : numpy.ndarray
        Prescribed yaw angle about ``ref_point`` [rad] (positive nose right),
        shape ``(n_steps + 1,)``.
    x_dot : numpy.ndarray
        Streamwise velocity along ``e_x`` [m/s], shape ``(n_steps + 1,)``.
    y_dot : numpy.ndarray
        Lateral velocity along ``e_y`` [m/s], shape ``(n_steps + 1,)``.
    roll_dot : numpy.ndarray
        Roll Euler-angle rate [rad/s], shape ``(n_steps + 1,)``.
    pitch_dot : numpy.ndarray
        Pitch Euler-angle rate [rad/s], shape ``(n_steps + 1,)``.
    yaw_dot : numpy.ndarray
        Yaw Euler-angle rate [rad/s], shape ``(n_steps + 1,)``.
    controls : dict of str to numpy.ndarray
        Prescribed control-surface deflection histories [rad], each of shape
        ``(n_steps + 1,)``.
    controls_dot : dict of str to numpy.ndarray
        Time derivatives of prescribed control-surface deflections [rad/s],
        each of shape ``(n_steps + 1,)``.
    gust_velocity : numpy.ndarray
        Gust air velocity evaluated at the instantaneous reference point in
        geometry axes [m/s], shape ``(n_steps + 1, 3)``. Zero when no gust is
        given.
    notes : list of str
        Diagnostic notes (for example when ``dt`` differs from ``dx_TE / V_inf``
        by more than 25 %, or gust, device fallback and ground-effect notes).
    device : str
        Execution device: ``"cpu"`` or ``"gpu"``.
    precision : str
        Kernel arithmetic precision: ``"float64"`` or ``"float32"``.
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
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    roll: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    pitch: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    yaw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    x_dot: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y_dot: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    roll_dot: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    pitch_dot: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    yaw_dot: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    controls: dict[str, np.ndarray] = field(default_factory=dict)
    controls_dot: dict[str, np.ndarray] = field(default_factory=dict)
    gust_velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    notes: list[str] = field(default_factory=list)
    device: str = "cpu"
    precision: str = "float64"


def _eval_gust(
    gust_fn: Callable[[np.ndarray, float], np.ndarray],
    points: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate *gust_fn* at *points* [m] and time *t* [s], validating shape and finiteness."""
    try:
        raw = gust_fn(points, float(t))
        arr = np.asarray(raw, dtype=np.float64)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"gust callable failed at t = {float(t):.6g} s: {exc}") from exc
    if arr.shape != points.shape:
        raise InputError(
            f"gust callable must return an array of shape {points.shape}, got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise InputError(f"gust callable returned non-finite values at t = {float(t):.6g} s.")
    return arr


def _advect_wake_nodes_rk4(
    wake_l: np.ndarray,
    wake_r: np.ndarray,
    n_lines: int,
    v_base: np.ndarray,
    gust_fn: Callable[[np.ndarray, float], np.ndarray],
    t0: float,
    dt: float,
    ground: GroundPlane | None,
) -> None:
    """Advect the first *n_lines* wake node lines from *t0* to ``t0 + dt`` [s] by RK4 in place.

    In ground effect, only the gust component parallel to *ground* convects
    wake nodes so that wake nodes do not cross the ground plane.
    """
    n_s = v_base.shape[0]
    half = n_lines * n_s
    pts = np.vstack([wake_l[:n_lines].reshape(half, 3), wake_r[:n_lines].reshape(half, 3)])
    v_base_tile = np.tile(v_base, (2 * n_lines, 1))
    g_norm = None if ground is None else np.asarray(ground.normal, dtype=np.float64)

    def _vel(p: np.ndarray, ts: float) -> np.ndarray:
        vg = _eval_gust(gust_fn, p, ts)
        if g_norm is not None:
            vg = vg - (vg @ g_norm)[:, None] * g_norm[None, :]
        return v_base_tile + vg

    half_dt = 0.5 * dt
    k1 = _vel(pts, t0)
    k2 = _vel(pts + half_dt * k1, t0 + half_dt)
    k3 = _vel(pts + half_dt * k2, t0 + half_dt)
    k4 = _vel(pts + dt * k3, t0 + dt)
    pts_new = pts + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    wake_l[:n_lines] = pts_new[:half].reshape(n_lines, n_s, 3)
    wake_r[:n_lines] = pts_new[half:].reshape(n_lines, n_s, 3)


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
    notes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample motion *motion* and its time derivative at *time* [s].

    Callable motions are differentiated with tiny finite differences on the
    continuous function (right-sided second-order at ``t = 0``, central for
    ``t > 0``). Array motions of length ``n_steps + 1`` are differentiated with
    second-order finite differences on the time grid (first-order when
    ``n_steps == 1``).

    A callable with a jump has no finite rate at the jump. The tiny difference
    then gives a rate of about ``jump / (2e-7 dt)`` when the jump is at a sample
    time, and no rate at all when the jump is between two sample times. To
    find such intervals, the function compares the change of the sampled value
    over each interval with the sampled rates (see :func:`_jump_points`). At
    the samples next to a jump, the rate is the grid rate of the array rule,
    so near a jump a callable gives the same rates as the array of its
    samples, and the rate integrates to the jump. A note gives the times when
    *notes* is given. A smooth callable that the time step resolves, or a
    continuous callable with a kink, passes the tests, and its rates do not
    change.
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
        picked = _jump_points(vals, dots, dt)
        if picked.size:
            dots[picked] = _grid_rates(vals, dt)[picked]
            if notes is not None:
                notes.append(
                    f"Motion {name}: the callable has a jump (or a change that the time step does not "
                    f"resolve) between t = {float(time[picked[0]]):.6g} s and t = {float(time[picked[-1]]):.6g} s. "
                    f"The rate at the {picked.size} samples there is the grid rate of the sampled values, "
                    "as for an array input."
                )
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

    return arr.copy(), _grid_rates(arr, dt)


def _grid_rates(arr: np.ndarray, dt: float) -> np.ndarray:
    """Return the time derivative of the samples *arr* on a uniform grid of step *dt* [s].

    Second-order finite differences: one-sided at the two ends, central
    inside; first-order when *arr* has two samples.
    """
    n_pts = int(arr.shape[0])
    dots = np.empty(n_pts, dtype=np.float64)
    if n_pts == 2:
        slope = (arr[1] - arr[0]) / dt
        dots[0] = slope
        dots[1] = slope
    else:
        dots[0] = (-3.0 * arr[0] + 4.0 * arr[1] - arr[2]) / (2.0 * dt)
        dots[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
        dots[-1] = (3.0 * arr[-1] - 4.0 * arr[-2] + arr[-3]) / (2.0 * dt)
    return dots


# Thresholds of the jump test of a sampled callable [-]. A sample rate is too
# large when ``|dot_i| dt`` exceeds _JUMP_RATE_FACTOR times the larger value
# change of its two intervals (a smooth or kinked input gives a ratio of
# about 1; a jump at a sample time gives about 5e6). An interval change is
# too large when ``|dv|`` exceeds _JUMP_CHANGE_FACTOR times ``dt`` times the
# larger end rate (a resolved smooth input gives about 1 + (omega dt)^2 / 8,
# a kink at most 1; a jump between sample times gives a very large ratio).
_JUMP_RATE_FACTOR = 100.0
_JUMP_CHANGE_FACTOR = 2.0


def _jump_points(vals: np.ndarray, dots: np.ndarray, dt: float) -> np.ndarray:
    """Return the sample indices whose rate must come from the grid because of a jump.

    Two tests find a jump in a sampled callable, each above a round-off floor
    of ``1e-9 max|vals|``:

    1. A jump at the sample time ``t_i`` gives a rate ``dot_i`` that the value
       changes of the two intervals at ``t_i`` cannot explain. The samples
       ``i - 1``, ``i`` and ``i + 1`` are returned.
    2. A jump inside the interval ``[t_(i-1), t_i]`` gives a value change that
       the end rates cannot explain. The samples ``i - 1`` and ``i`` are
       returned.

    A continuous input with a kink (for example a ramp) passes both tests.
    """
    n_pts = int(vals.size)
    if n_pts < 2:
        return np.zeros(0, dtype=np.int64)
    dv = np.abs(np.diff(vals))
    floor = 1.0e-9 * float(np.max(np.abs(vals)))
    dv_left = np.concatenate([[0.0], dv])
    dv_right = np.concatenate([dv, [0.0]])
    rate_bad = np.abs(dots) * dt > _JUMP_RATE_FACTOR * np.maximum(dv_left, dv_right) + floor
    end_rate = np.maximum(np.abs(dots[:-1]), np.abs(dots[1:]))
    change_bad = dv > _JUMP_CHANGE_FACTOR * dt * end_rate + floor
    picked = np.zeros(n_pts, dtype=bool)
    for i in np.flatnonzero(rate_bad):
        picked[max(i - 1, 0):min(i + 2, n_pts)] = True
    for i in np.flatnonzero(change_bad):
        picked[i:i + 2] = True
    return np.flatnonzero(picked).astype(np.int64)


def _rotation_matrix_321(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the intrinsic 3-2-1 (yaw, pitch, roll) body-to-reference rotation matrix [-].

    Geometry axes are x aft, y right, z up. Signs follow AVL: ``roll > 0``
    right wing down, ``pitch > 0`` nose up, ``yaw > 0`` nose right. Rotations
    are composed in the flight-mechanics 3-2-1 sequence: yaw about ``-z``,
    then pitch about the rotated ``+y``, then roll about the rotated ``-x``
    (Etkin and Reid 1996, chapter 4).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, -cy * sp * sr + sy * cr, cy * sp * cr + sy * sr],
            [-sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [-sp, -cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _angular_velocity_321(
    pitch: float,
    yaw: float,
    roll_dot: float,
    pitch_dot: float,
    yaw_dot: float,
) -> np.ndarray:
    """Return the body angular velocity [rad/s] in mean-flight geometry axes.

    Computed from the 3-2-1 (yaw, pitch, roll) Euler-angle rates of Etkin and
    Reid (1996, chapter 4), mapped to geometry axes (x aft, y right, z up) so
    that the rigid-body velocity of a point is ``v_trans + omega x (p - x_ref)``.
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            -roll_dot * cy * cp + pitch_dot * sy,
            roll_dot * sy * cp + pitch_dot * cy,
            roll_dot * sp - yaw_dot,
        ],
        dtype=np.float64,
    )


def _rotate_vecs(
    vec: np.ndarray,
    R: np.ndarray | None,
    pitch_only: float | None = None,
) -> np.ndarray:
    """Apply body-to-reference rotation *R* (or pure pitch *pitch_only* [rad]) to *vec*."""
    if R is None:
        return vec
    if pitch_only is not None:
        return _rot_y(vec, pitch_only)
    return vec @ R.T


def _move_wing_geometry(
    cp_b: np.ndarray,
    normal_bc_b: np.ndarray,
    corners_b: np.ndarray,
    te_left_b: np.ndarray,
    te_right_b: np.ndarray,
    side_nodes_b: np.ndarray | None,
    ref_point: np.ndarray,
    x_ref: np.ndarray,
    R: np.ndarray | None,
    pitch_only: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return moved ``(cp, normal_bc, corners, te_left, te_right, side_nodes)`` in mean-flight axes."""
    if R is None:
        shift = x_ref - ref_point
        cp = cp_b + shift[None, :]
        normal_bc = normal_bc_b
        corners = corners_b + shift[None, None, :]
        te_left = te_left_b + shift[None, :]
        te_right = te_right_b + shift[None, :]
        side_nodes = None if side_nodes_b is None else side_nodes_b + shift[None, None, None, :]
        return cp, normal_bc, corners, te_left, te_right, side_nodes

    cp = x_ref[None, :] + _rotate_vecs(cp_b - ref_point[None, :], R, pitch_only)
    normal_bc = _rotate_vecs(normal_bc_b, R, pitch_only)
    corners = x_ref[None, None, :] + _rotate_vecs(corners_b - ref_point[None, None, :], R, pitch_only)
    te_left = x_ref[None, :] + _rotate_vecs(te_left_b - ref_point[None, :], R, pitch_only)
    te_right = x_ref[None, :] + _rotate_vecs(te_right_b - ref_point[None, :], R, pitch_only)
    side_nodes = (
        None
        if side_nodes_b is None
        else x_ref[None, None, None, :] + _rotate_vecs(side_nodes_b - ref_point[None, None, None, :], R, pitch_only)
    )
    return cp, normal_bc, corners, te_left, te_right, side_nodes


def _unit_vec(v: np.ndarray) -> np.ndarray:
    """Return unit vector along *v*, falling back to ``(0, 1, 0)`` if degenerate."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _rodrigues_dirs(vecs: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 3-D vector(s) *vecs* by *angle* [rad] about unit *axis* (Rodrigues formula)."""
    c = math.cos(angle)
    s = math.sin(angle)
    if vecs.ndim == 1:
        k_dot_v = float(axis[0] * vecs[0] + axis[1] * vecs[1] + axis[2] * vecs[2])
        cx = axis[1] * vecs[2] - axis[2] * vecs[1]
        cy = axis[2] * vecs[0] - axis[0] * vecs[2]
        cz = axis[0] * vecs[1] - axis[1] * vecs[0]
        return np.array(
            [
                vecs[0] * c + cx * s + axis[0] * k_dot_v * (1.0 - c),
                vecs[1] * c + cy * s + axis[1] * k_dot_v * (1.0 - c),
                vecs[2] * c + cz * s + axis[2] * k_dot_v * (1.0 - c),
            ],
            dtype=np.float64,
        )
    k_dot_v = np.sum(vecs * axis[None, :], axis=1, keepdims=True)
    return vecs * c + cross3(axis[None, :], vecs) * s + axis[None, :] * k_dot_v * (1.0 - c)


def _rodrigues_pts(pts: np.ndarray, hinge: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point(s) *pts* [m] about the line through *hinge* [m] along unit *axis* by *angle* [rad]."""
    return hinge + _rodrigues_dirs(pts - hinge, axis, angle)


def _slice_rc_sub(
    rc: np.ndarray | float,
    rows: np.ndarray | None,
    cols: np.ndarray | None,
) -> np.ndarray | float:
    """Slice core radius *rc* [m] (scalar, 1-D or 2-D) to *rows* and *cols*."""
    nd = np.ndim(rc)
    if nd == 0:
        return rc
    arr = np.asarray(rc, dtype=np.float64)
    if nd == 1:
        return arr if cols is None else arr[cols]
    if rows is None and cols is None:
        return arr
    if rows is None:
        return arr[:, cols]
    if cols is None:
        return arr[rows, :]
    return arr[np.ix_(rows, cols)]


def _physical_ring_areas_from_arrays(
    corners: np.ndarray,
    te_left_phys: np.ndarray,
    te_right_phys: np.ndarray,
    te_mask: np.ndarray,
    side_nodes: np.ndarray | None,
    side_counts: np.ndarray | None,
    ref_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return physical surface area vectors [m^2], centroids [m] and optional moment vectors [m^3].

    Matches :func:`ventorum.unsteady.harmonic._physical_ring_areas` directly
    from moved ring and physical trailing-edge arrays.
    """
    q0 = corners[:, 0, :]
    q1 = corners[:, 1, :]
    q2 = corners[:, 2, :].copy()
    q3 = corners[:, 3, :].copy()
    te_idx = np.flatnonzero(te_mask)
    q2[te_idx] = te_right_phys
    q3[te_idx] = te_left_phys

    a1 = 0.5 * cross3(q2 - q0, q1 - q0)
    a2 = 0.5 * cross3(q3 - q0, q2 - q0)
    c1 = (q0 + q1 + q2) / 3.0
    c2 = (q0 + q2 + q3) / 3.0
    area_vec = a1 + a2
    m1 = np.linalg.norm(a1, axis=1)
    m2 = np.linalg.norm(a2, axis=1)
    centroid = (m1[:, None] * c1 + m2[:, None] * c2) / np.maximum((m1 + m2)[:, None], 1e-300)

    has_poly = (
        side_nodes is not None
        and side_counts is not None
        and bool(np.any(side_counts > 0))
    )
    if not has_poly:
        return area_vec, centroid, None

    area_moment = cross3(centroid - ref_point[None, :], area_vec)
    n_p = corners.shape[0]
    for p in range(n_p):
        cr = int(side_counts[p, 0])
        cl = int(side_counts[p, 1])
        if cr == 0 and cl == 0:
            continue
        r_pts = [side_nodes[p, 0, k] for k in range(cr)]
        l_pts = [side_nodes[p, 1, k] for k in range(cl)]
        if te_mask[p]:
            if r_pts and np.allclose(r_pts[-1], q2[p], atol=1e-13, rtol=1e-13):
                r_pts.pop()
            if l_pts and np.allclose(l_pts[0], q3[p], atol=1e-13, rtol=1e-13):
                l_pts.pop(0)
        if not r_pts and not l_pts:
            continue

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
                m_sum += np.cross(ct - ref_point, at)
                c_num += mt * ct
                w_sum += mt
        area_vec[p] = a_sum
        centroid[p] = c_num / max(w_sum, 1e-300)
        area_moment[p] = m_sum

    return area_vec, centroid, area_moment


@dataclass(slots=True)
class _ControlStripMap:
    """Hinge geometry and moving-node indices for one active flap strip."""

    strip: int
    k_hinge: int
    f_cp: float
    is_chain: bool
    H_l: np.ndarray
    H_r: np.ndarray
    H_f: np.ndarray
    ax_l: np.ndarray
    ax_r: np.ndarray
    ax_f: np.ndarray
    cp_rot_panels: tuple[int, ...]
    akink_rot_panels: tuple[int, ...]
    bkink_rot_panels: tuple[int, ...]
    sn_right_rot: tuple[tuple[int, int], ...]
    sn_left_rot: tuple[tuple[int, int], ...]
    leg_norm_left_rot: tuple[int, ...]
    leg_norm_right_rot: tuple[int, ...]


@dataclass(slots=True)
class _ControlRig:
    """Precomputed hinge map, deflection histories and moving-block indices for controls."""

    strips: list[_ControlStripMap]
    dd_edges: np.ndarray
    rate_edges: np.ndarray
    moving_row_panels: np.ndarray
    moving_col_panels: np.ndarray
    fixed_col_panels: np.ndarray
    deflections_vary: bool
    hinge_points: np.ndarray
    a0: np.ndarray
    b0: np.ndarray
    cp0: np.ndarray
    normal_bc0: np.ndarray
    corners0: np.ndarray
    te_left0: np.ndarray
    te_right0: np.ndarray
    te_l_phys0: np.ndarray
    te_r_phys0: np.ndarray
    side_nodes0: np.ndarray | None
    a_kink0: np.ndarray
    b_kink0: np.ndarray
    leg_normal0: np.ndarray
    ctrl_strip_coefs: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def edges_for_values(
        self,
        ctrl_vals: Mapping[str, float],
        ctrl_rates: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dd_step, rate_step)`` of shape ``(n_strips, 2)`` for control values and rates."""
        n_s = self.dd_edges.shape[1]
        dd_step = np.zeros((n_s, 2), dtype=np.float64)
        rate_step = np.zeros((n_s, 2), dtype=np.float64)
        for cname, (c_arr, d0_arr) in self.ctrl_strip_coefs.items():
            if cname not in ctrl_vals:
                continue
            val = float(ctrl_vals[cname])
            rate = float(ctrl_rates.get(cname, 0.0))
            dd = val - d0_arr
            dd_step[:, 0] += c_arr[:, 0] * dd
            dd_step[:, 1] += c_arr[:, 1] * dd
            rate_step[:, 0] += c_arr[:, 0] * rate
            rate_step[:, 1] += c_arr[:, 1] * rate
        return dd_step, rate_step


def _assemble_control_rig(
    lattice: VortexLattice,
    rings: RingLattice,
    leg_normal0: np.ndarray,
    active_strip_ids: list[int],
    dd_edges: np.ndarray,
    rate_edges: np.ndarray,
    *,
    deflections_vary: bool,
    ctrl_strip_coefs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> _ControlRig:
    """Assemble a :class:`_ControlRig` for *active_strip_ids*."""
    flap_field = lattice.geom_cache["flap_field"]
    flap_hinges = lattice.geom_cache["flap_hinges"]
    n_s = rings.n_strips
    n_c = rings.n_chord
    n_p = rings.n_panels
    hinge_of = np.asarray(flap_field["hinge"], dtype=np.float64)
    far = np.asarray(flap_field["far"], dtype=np.int64)
    hinge_off = flap_field.get("hinge_off")
    leg_mid_frac = lattice.geom_cache.get("leg_mid_frac")

    def _far_hinge_pt(q: int, side_q: int, h_val: float) -> np.ndarray | None:
        if q < 0:
            return None
        if np.isfinite(hinge_of[q]) and hinge_of[q] == h_val and q in flap_hinges:
            return np.asarray(flap_hinges[q][side_q], dtype=np.float64)
        sf = lattice.surfaces[int(lattice.strip_surface[q])]
        edge_idx = q - sf.strips.start + side_q
        return np.asarray(sf.edge_le[edge_idx] + h_val * (sf.edge_te[edge_idx] - sf.edge_le[edge_idx]), dtype=np.float64)

    panel_len = np.asarray(lattice.panel_length, dtype=np.float64).reshape(n_s, n_c)
    chords = np.asarray(lattice.chord, dtype=np.float64)
    a0_arr = np.asarray(lattice.a0, dtype=np.float64)
    cp_frac = np.asarray(lattice.cp_frac, dtype=np.float64)
    leg_next = None if lattice.leg_next is None else np.asarray(lattice.leg_next, dtype=bool)
    sc = rings.side_counts

    strip_maps: list[_ControlStripMap] = []
    row_set: set[int] = set()
    col_set: set[int] = set()
    hinge_pts_list: list[np.ndarray] = []

    for s in active_strip_ids:
        h = float(hinge_of[s])
        d_xi = panel_len[s] / max(float(chords[s]), 1e-300)
        xi_s = np.empty(n_c + 1, dtype=np.float64)
        xi_s[0] = 0.0
        xi_s[1:] = np.cumsum(d_xi)
        xi_s[-1] = 1.0
        kh = 1 + int(np.argmin(np.abs(xi_s[1:n_c] - h)))

        H_l = np.asarray(flap_hinges[s][0], dtype=np.float64)
        H_r = np.asarray(flap_hinges[s][1], dtype=np.float64)
        hinge_pts_list.append(H_l)
        hinge_pts_list.append(H_r)

        fl, fr = far[s, 0], far[s, 1]
        p_fl = _far_hinge_pt(int(fl[0]), int(fl[1]), h)
        p_fr = _far_hinge_pt(int(fr[0]), int(fr[1]), h)
        ax_l = _unit_vec(H_r - (p_fl if p_fl is not None else H_l))
        ax_r = _unit_vec((p_fr if p_fr is not None else H_r) - H_l)
        f_cp = float(cp_frac[s])
        H_f = H_l + f_cp * (H_r - H_l)
        ax_f = _unit_vec(ax_l + f_cp * (ax_r - ax_l))
        claf_s = float(np.clip(a0_arr[s] / (2.0 * np.pi), CLAF_MIN, CLAF_MAX))

        p0 = s * n_c
        is_chain = (leg_next is not None) and bool(np.any(leg_next[p0 : p0 + n_c]))
        cp_rot: list[int] = []
        akink_rot: list[int] = []
        bkink_rot: list[int] = []
        sn_r_rot: list[tuple[int, int]] = []
        sn_l_rot: list[tuple[int, int]] = []
        ln_l_rot: list[int] = []
        ln_r_rot: list[int] = []

        for k in range(n_c):
            p = p0 + k
            x0_k = float(xi_s[k])
            dx_k = float(xi_s[k + 1] - x0_k)
            xb_k = x0_k + 0.25 * dx_k
            xc_k = xb_k + 0.5 * dx_k * claf_s
            is_aft = k >= kh

            if xc_k > h:
                cp_rot.append(p)
                row_set.add(p)
            if is_aft:
                row_set.add(p)
            if k >= kh - 1:
                col_set.add(p)

            # Leg-force normal rotation matches ventorum.geometry.lattice.build_lattice.
            if is_aft:
                ln_l_rot.append(p)
                ln_r_rot.append(n_p + p)
                if is_chain:
                    ln_l_rot.append(2 * n_p + p)
                    ln_r_rot.append(3 * n_p + p)
            elif not (is_chain and k < n_c - 1 and (k + 1) < kh):
                ln_l_rot.append(2 * n_p + p)
                ln_r_rot.append(3 * n_p + p)

            if is_chain:
                if k < n_c - 1:
                    x1_k = float(xi_s[k + 1])
                    xm_k = 0.5 * (xb_k + x1_k + 0.25 * float(xi_s[k + 2] - x1_k))
                else:
                    xm_k = xb_k + 0.5 * dx_k
                xm_l = float(leg_mid_frac[p, 0]) if (leg_mid_frac is not None and np.isfinite(leg_mid_frac[p, 0])) else xm_k
                xm_r = float(leg_mid_frac[p, 1]) if (leg_mid_frac is not None and np.isfinite(leg_mid_frac[p, 1])) else xm_k
                is_hinge_l = (k == kh - 1) and (hinge_off is None or not bool(hinge_off[s, 0]))
                is_hinge_r = (k == kh - 1) and (hinge_off is None or not bool(hinge_off[s, 1]))
                rot_al = (not is_hinge_l) and (xm_l > h)
                rot_br = (not is_hinge_r) and (xm_r > h)
                if rot_al:
                    akink_rot.append(p)
                if rot_br:
                    bkink_rot.append(p)
                if sc is not None:
                    cr_p = int(sc[p, 0])
                    cl_p = int(sc[p, 1])
                    if cr_p >= 1 and rot_br:
                        sn_r_rot.append((p, 0))
                    if cr_p >= 2 and k == n_c - 1:
                        sn_r_rot.append((p, 1))
                    if cl_p == 1 and rot_al:
                        sn_l_rot.append((p, 0))
                    elif cl_p >= 2 and k == n_c - 1:
                        sn_l_rot.append((p, 0))
                        if rot_al:
                            sn_l_rot.append((p, 1))
            elif is_aft:
                akink_rot.append(p)
                bkink_rot.append(p)

        strip_maps.append(
            _ControlStripMap(
                strip=s,
                k_hinge=kh,
                f_cp=f_cp,
                is_chain=is_chain,
                H_l=H_l,
                H_r=H_r,
                H_f=H_f,
                ax_l=ax_l,
                ax_r=ax_r,
                ax_f=ax_f,
                cp_rot_panels=tuple(cp_rot),
                akink_rot_panels=tuple(akink_rot),
                bkink_rot_panels=tuple(bkink_rot),
                sn_right_rot=tuple(sn_r_rot),
                sn_left_rot=tuple(sn_l_rot),
                leg_norm_left_rot=tuple(ln_l_rot),
                leg_norm_right_rot=tuple(ln_r_rot),
            )
        )

    moving_row_panels = np.array(sorted(row_set), dtype=np.int64)
    moving_col_panels = np.array(sorted(col_set), dtype=np.int64)
    col_mask = np.ones(n_p, dtype=bool)
    col_mask[moving_col_panels] = False
    fixed_col_panels = np.flatnonzero(col_mask)

    return _ControlRig(
        strips=strip_maps,
        dd_edges=dd_edges,
        rate_edges=rate_edges,
        moving_row_panels=moving_row_panels,
        moving_col_panels=moving_col_panels,
        fixed_col_panels=fixed_col_panels,
        deflections_vary=deflections_vary,
        hinge_points=np.vstack(hinge_pts_list),
        a0=np.asarray(lattice.a, dtype=np.float64).copy(),
        b0=np.asarray(lattice.b, dtype=np.float64).copy(),
        cp0=rings.cp.copy(),
        normal_bc0=rings.normal_bc.copy(),
        corners0=rings.corners.copy(),
        te_left0=rings.te_left.copy(),
        te_right0=rings.te_right.copy(),
        te_l_phys0=np.asarray(lattice.te_left, dtype=np.float64).copy(),
        te_r_phys0=np.asarray(lattice.te_right, dtype=np.float64).copy(),
        side_nodes0=None if rings.side_nodes is None else rings.side_nodes.copy(),
        a_kink0=np.asarray(lattice.a_kink, dtype=np.float64).copy(),
        b_kink0=np.asarray(lattice.b_kink, dtype=np.float64).copy(),
        leg_normal0=leg_normal0.copy(),
        ctrl_strip_coefs={} if ctrl_strip_coefs is None else ctrl_strip_coefs,
    )


def _validate_control_names_and_extract_coefs(
    lattice: VortexLattice,
    rings: RingLattice,
    control_names: Sequence[str],
    *,
    increments: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Validate *control_names* against *lattice* and return ``(hinge_of, ctrl_strip_coefs, delta0_by_name)``.

    With *increments* the base deflection of every strip is 0, so a control
    value is an increment over the built deflection of each surface (not an
    absolute deflection).
    """
    info_list = list(getattr(lattice, "control_info", []))
    known_names = {str(entry["name"]) for entry in info_list}
    mode_by_name: dict[str, set[str]] = {}
    delta0_by_surf_name: dict[tuple[str, str], float] = {}
    delta0_by_name: dict[str, float] = {}
    for entry in info_list:
        cname = str(entry["name"])
        sname = str(entry["surface"])
        mode_by_name.setdefault(cname, set()).add(str(entry["mode"]))
        delta0_by_surf_name[(sname, cname)] = float(entry["deflection_rad"])
        delta0_by_name[cname] = float(entry["deflection_rad"])

    for raw_name in control_names:
        if not isinstance(raw_name, str) or raw_name not in known_names:
            raise InputError(
                f"Unknown control surface {raw_name!r}; available controls: {sorted(known_names)}."
            )
        if mode_by_name.get(raw_name) != {"chordwise"}:
            raise InputError(
                f"Control surface {raw_name!r} uses section mode; moving control surfaces "
                "require a chordwise lattice (n_chord >= 2)."
            )

    flap_field = lattice.geom_cache.get("flap_field")
    flap_hinges = lattice.geom_cache.get("flap_hinges")
    if flap_field is None or not flap_hinges or lattice.a_kink is None or lattice.b_kink is None:
        raise InputError(
            "Lattice does not hold chordwise flap hinge geometry; build with n_chord >= 2."
        )

    n_s = rings.n_strips
    hinge_of = np.asarray(flap_field["hinge"], dtype=np.float64)
    coef_map = flap_field["coef"]

    ctrl_strip_coefs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cname in control_names:
        c_arr_raw = coef_map.get(cname)
        c_arr = np.zeros((n_s, 2), dtype=np.float64)
        d0_arr = np.zeros(n_s, dtype=np.float64)
        if c_arr_raw is not None:
            for s in range(n_s):
                if not np.isfinite(hinge_of[s]):
                    continue
                surf_name = lattice.surfaces[int(lattice.strip_surface[s])].name
                c_arr[s, 0] = float(c_arr_raw[s, 0])
                c_arr[s, 1] = float(c_arr_raw[s, 1])
                if not increments:
                    d0_arr[s] = delta0_by_surf_name.get((surf_name, cname), delta0_by_name[cname])
        ctrl_strip_coefs[cname] = (c_arr, d0_arr)

    return hinge_of, ctrl_strip_coefs, delta0_by_name


def _build_dynamic_control_rig(
    lattice: VortexLattice,
    rings: RingLattice,
    leg_normal0: np.ndarray,
    control_names: Sequence[str],
    n_pts: int,
) -> tuple[_ControlRig | None, dict[str, float]]:
    """Build a :class:`_ControlRig` for step-by-step control evaluation over *n_pts* time points."""
    if not control_names:
        return None, {}
    hinge_of, ctrl_strip_coefs, delta0_by_name = _validate_control_names_and_extract_coefs(
        lattice, rings, control_names,
    )
    n_s = rings.n_strips
    active_strip_ids = [
        s
        for s in range(n_s)
        if np.isfinite(hinge_of[s])
        and any(
            ctrl_strip_coefs[cname][0][s, 0] != 0.0 or ctrl_strip_coefs[cname][0][s, 1] != 0.0
            for cname in control_names
        )
    ]
    if not active_strip_ids:
        return None, delta0_by_name
    dd_edges = np.zeros((n_pts, n_s, 2), dtype=np.float64)
    rate_edges = np.zeros((n_pts, n_s, 2), dtype=np.float64)
    rig = _assemble_control_rig(
        lattice,
        rings,
        leg_normal0,
        active_strip_ids,
        dd_edges,
        rate_edges,
        deflections_vary=True,
        ctrl_strip_coefs=ctrl_strip_coefs,
    )
    return rig, delta0_by_name


def _build_control_rig(
    lattice: VortexLattice,
    rings: RingLattice,
    leg_normal0: np.ndarray,
    controls: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None,
    time: np.ndarray,
    dt_val: float,
    notes: list[str] | None = None,
    *,
    increments: bool = False,
) -> tuple[_ControlRig | None, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate and sample *controls*, returning ``(rig, ctrl_arrs, ctrl_dots)``.

    With *increments* the values are increments over the built deflection of
    each surface (see :func:`_validate_control_names_and_extract_coefs`).
    """
    if controls is None:
        return None, {}, {}
    if not isinstance(controls, Mapping):
        raise InputError(f"controls must be a mapping from control name to deflection history, got {type(controls).__name__}.")
    if len(controls) == 0:
        return None, {}, {}

    control_names = list(controls.keys())
    hinge_of, ctrl_strip_coefs, _ = _validate_control_names_and_extract_coefs(
        lattice, rings, control_names, increments=increments,
    )

    ctrl_arrs: dict[str, np.ndarray] = {}
    ctrl_dots: dict[str, np.ndarray] = {}
    for raw_name, spec in controls.items():
        arr, dot = _sample_motion(spec, time, dt_val, f"controls[{raw_name!r}]", notes)
        ctrl_arrs[raw_name] = arr
        ctrl_dots[raw_name] = dot

    n_pts = int(time.size)
    n_s = rings.n_strips
    dd_edges = np.zeros((n_pts, n_s, 2), dtype=np.float64)
    rate_edges = np.zeros((n_pts, n_s, 2), dtype=np.float64)
    for s in range(n_s):
        if not np.isfinite(hinge_of[s]):
            continue
        for cname, arr in ctrl_arrs.items():
            c_arr, d0_arr = ctrl_strip_coefs[cname]
            cl = float(c_arr[s, 0])
            cr = float(c_arr[s, 1])
            if cl == 0.0 and cr == 0.0:
                continue
            dd = arr - float(d0_arr[s])
            dt_c = ctrl_dots[cname]
            dd_edges[:, s, 0] += cl * dd
            dd_edges[:, s, 1] += cr * dd
            rate_edges[:, s, 0] += cl * dt_c
            rate_edges[:, s, 1] += cr * dt_c

    active_strip_ids = [
        s
        for s in range(n_s)
        if np.isfinite(hinge_of[s])
        and (bool(np.any(dd_edges[:, s, :] != 0.0)) or bool(np.any(rate_edges[:, s, :] != 0.0)))
    ]
    if not active_strip_ids:
        return None, ctrl_arrs, ctrl_dots

    deflections_vary = any(
        not np.all(dd_edges[:, s, :] == dd_edges[0, s, :]) for s in active_strip_ids
    )
    rig = _assemble_control_rig(
        lattice,
        rings,
        leg_normal0,
        active_strip_ids,
        dd_edges,
        rate_edges,
        deflections_vary=deflections_vary,
        ctrl_strip_coefs=ctrl_strip_coefs,
    )
    return rig, ctrl_arrs, ctrl_dots


def _deform_body_geometry(
    rig: _ControlRig,
    lattice: VortexLattice,
    rings: RingLattice,
    dd_step: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return deformed body-frame geometry arrays for incremental deflections *dd_step* [rad]."""
    n_c = rings.n_chord
    a_b = rig.a0.copy()
    b_b = rig.b0.copy()
    cp_b = rig.cp0.copy()
    normal_bc_b = rig.normal_bc0.copy()
    corners_b = rig.corners0.copy()
    te_left_b = rig.te_left0.copy()
    te_right_b = rig.te_right0.copy()
    te_l_phys_b = rig.te_l_phys0.copy()
    te_r_phys_b = rig.te_r_phys0.copy()
    side_nodes_b = None if rig.side_nodes0 is None else rig.side_nodes0.copy()
    a_kink_b = rig.a_kink0.copy()
    b_kink_b = rig.b_kink0.copy()
    leg_normal_b = rig.leg_normal0.copy()

    for st in rig.strips:
        s = st.strip
        dd_l = float(dd_step[s, 0])
        dd_r = float(dd_step[s, 1])
        dd_f = dd_l + st.f_cp * (dd_r - dd_l)
        if dd_l == 0.0 and dd_r == 0.0 and dd_f == 0.0:
            continue

        p0 = s * n_c
        kh = st.k_hinge
        p_aft = slice(p0 + kh, p0 + n_c)

        if dd_l != 0.0:
            a_b[p_aft] = _rodrigues_pts(a_b[p_aft], st.H_l, st.ax_l, dd_l)
            te_l_phys_b[s] = _rodrigues_pts(te_l_phys_b[s], st.H_l, st.ax_l, dd_l)
            te_left_b[s] = _rodrigues_pts(te_left_b[s], st.H_l, st.ax_l, dd_l)
        if dd_r != 0.0:
            b_b[p_aft] = _rodrigues_pts(b_b[p_aft], st.H_r, st.ax_r, dd_r)
            te_r_phys_b[s] = _rodrigues_pts(te_r_phys_b[s], st.H_r, st.ax_r, dd_r)
            te_right_b[s] = _rodrigues_pts(te_right_b[s], st.H_r, st.ax_r, dd_r)

        corners_b[p_aft, 0] = a_b[p_aft]
        corners_b[p_aft, 1] = b_b[p_aft]
        for k in range(kh - 1, n_c - 1):
            p = p0 + k
            corners_b[p, 2] = b_b[p + 1]
            corners_b[p, 3] = a_b[p + 1]
        p_te = p0 + n_c - 1
        corners_b[p_te, 2] = te_right_b[s]
        corners_b[p_te, 3] = te_left_b[s]

        if dd_f != 0.0:
            for p in st.cp_rot_panels:
                cp_b[p] = _rodrigues_pts(cp_b[p], st.H_f, st.ax_f, dd_f)
            normal_bc_b[p_aft] = _rodrigues_dirs(normal_bc_b[p_aft], st.ax_f, dd_f)

        if not st.is_chain:
            a_kink_b[p_aft] = te_l_phys_b[s]
            b_kink_b[p_aft] = te_r_phys_b[s]
        else:
            if dd_l != 0.0:
                for p in st.akink_rot_panels:
                    a_kink_b[p] = _rodrigues_pts(a_kink_b[p], st.H_l, st.ax_l, dd_l)
            if dd_r != 0.0:
                for p in st.bkink_rot_panels:
                    b_kink_b[p] = _rodrigues_pts(b_kink_b[p], st.H_r, st.ax_r, dd_r)
            if side_nodes_b is not None:
                if dd_r != 0.0:
                    for p, slot in st.sn_right_rot:
                        side_nodes_b[p, 0, slot] = _rodrigues_pts(side_nodes_b[p, 0, slot], st.H_r, st.ax_r, dd_r)
                if dd_l != 0.0:
                    for p, slot in st.sn_left_rot:
                        side_nodes_b[p, 1, slot] = _rodrigues_pts(side_nodes_b[p, 1, slot], st.H_l, st.ax_l, dd_l)

        if dd_l != 0.0:
            for idx_row in st.leg_norm_left_rot:
                leg_normal_b[idx_row] = _rodrigues_dirs(leg_normal_b[idx_row], st.ax_l, dd_l)
        if dd_r != 0.0:
            for idx_row in st.leg_norm_right_rot:
                leg_normal_b[idx_row] = _rodrigues_dirs(leg_normal_b[idx_row], st.ax_r, dd_r)

    return (
        a_b,
        b_b,
        cp_b,
        normal_bc_b,
        corners_b,
        te_left_b,
        te_right_b,
        te_l_phys_b,
        te_r_phys_b,
        side_nodes_b,
        a_kink_b,
        b_kink_b,
        leg_normal_b,
    )


def _compute_flap_velocities(
    rig: _ControlRig,
    lattice: VortexLattice,
    rings: RingLattice,
    rate_step: np.ndarray,
    a_b: np.ndarray,
    b_b: np.ndarray,
    cp_b: np.ndarray,
    te_l_phys_b: np.ndarray,
    te_r_phys_b: np.ndarray,
    a_kink_b: np.ndarray,
    b_kink_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body-frame flap velocities ``(v_cp_b, v_load_b, v_te_b)`` [m/s] from hinge rates."""
    n_p = rings.n_panels
    n_s = rings.n_strips
    n_c = rings.n_chord
    v_cp_b = np.zeros((n_p, 3), dtype=np.float64)
    v_a_b = np.zeros((n_p, 3), dtype=np.float64)
    v_b_b = np.zeros((n_p, 3), dtype=np.float64)
    v_te_l_b = np.zeros((n_s, 3), dtype=np.float64)
    v_te_r_b = np.zeros((n_s, 3), dtype=np.float64)
    v_a_kink_b = np.zeros((n_p, 3), dtype=np.float64)
    v_b_kink_b = np.zeros((n_p, 3), dtype=np.float64)

    for st in rig.strips:
        s = st.strip
        r_l = float(rate_step[s, 0])
        r_r = float(rate_step[s, 1])
        r_f = r_l + st.f_cp * (r_r - r_l)
        if r_l == 0.0 and r_r == 0.0 and r_f == 0.0:
            continue

        p0 = s * n_c
        p_aft = slice(p0 + st.k_hinge, p0 + n_c)
        if r_f != 0.0:
            w_f = r_f * st.ax_f
            for p in st.cp_rot_panels:
                v_cp_b[p] = np.cross(w_f, cp_b[p] - st.H_f)
        if r_l != 0.0:
            w_l = r_l * st.ax_l
            v_a_b[p_aft] = cross3(w_l[None, :], a_b[p_aft] - st.H_l[None, :])
            v_te_l_b[s] = np.cross(w_l, te_l_phys_b[s] - st.H_l)
            if not st.is_chain:
                v_a_kink_b[p_aft] = v_te_l_b[s]
            else:
                for p in st.akink_rot_panels:
                    v_a_kink_b[p] = np.cross(w_l, a_kink_b[p] - st.H_l)
        if r_r != 0.0:
            w_r = r_r * st.ax_r
            v_b_b[p_aft] = cross3(w_r[None, :], b_b[p_aft] - st.H_r[None, :])
            v_te_r_b[s] = np.cross(w_r, te_r_phys_b[s] - st.H_r)
            if not st.is_chain:
                v_b_kink_b[p_aft] = v_te_r_b[s]
            else:
                for p in st.bkink_rot_panels:
                    v_b_kink_b[p] = np.cross(w_r, b_kink_b[p] - st.H_r)

    p_strip = rings.panel_strip
    f_p = np.asarray(lattice.cp_frac, dtype=np.float64)[p_strip][:, None]
    v_mid_b = (1.0 - f_p) * v_a_b + f_p * v_b_b
    v_a_te_p = v_te_l_b[p_strip]
    v_b_te_p = v_te_r_b[p_strip]
    if lattice.leg_next is None:
        v_a_e, v_b_e = v_a_te_p, v_b_te_p
    else:
        v_a_e, v_b_e = chain_ends(v_a_b, v_b_b, v_a_te_p, v_b_te_p, lattice.leg_next)
    v_leg_start = np.vstack([v_a_kink_b, v_b_b, v_a_e, v_b_kink_b])
    v_leg_end = np.vstack([v_a_b, v_b_kink_b, v_a_kink_b, v_b_e])
    v_load_b = np.vstack([v_mid_b, 0.5 * (v_leg_start + v_leg_end)])
    f_s = np.asarray(lattice.cp_frac, dtype=np.float64)[:, None]
    v_te_b = (1.0 - f_s) * v_te_l_b + f_s * v_te_r_b
    return v_cp_b, v_load_b, v_te_b


def _update_bound_influence_blocks(
    A_bb_real: np.ndarray,
    rig: _ControlRig,
    cp_b: np.ndarray,
    normal_bc_b: np.ndarray,
    corners_b: np.ndarray,
    side_nodes_b: np.ndarray | None,
    side_counts: np.ndarray | None,
    rc_ring: np.ndarray | float,
    *,
    backend: str | None,
) -> None:
    """Update only the moving rows and columns of *A_bb_real* [1/m] in place."""
    c_m = rig.moving_col_panels
    r_m = rig.moving_row_panels
    c_f = rig.fixed_col_panels
    sn_cm = None if side_nodes_b is None else side_nodes_b[c_m]
    sc_cm = None if side_counts is None else side_counts[c_m]
    A_bb_real[:, c_m] = ring_normal_influence(
        cp_b,
        normal_bc_b,
        corners_b[c_m],
        _slice_rc_sub(rc_ring, None, c_m),
        side_nodes=sn_cm,
        side_counts=sc_cm,
        backend=backend,
    )
    if r_m.size > 0 and c_f.size > 0:
        sn_cf = None if side_nodes_b is None else side_nodes_b[c_f]
        sc_cf = None if side_counts is None else side_counts[c_f]
        A_bb_real[np.ix_(r_m, c_f)] = ring_normal_influence(
            cp_b[r_m],
            normal_bc_b[r_m],
            corners_b[c_f],
            _slice_rc_sub(rc_ring, r_m, c_f),
            side_nodes=sn_cf,
            side_counts=sc_cf,
            backend=backend,
        )


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


def _resolve_six_axis_motion(
    z: Sequence[float] | np.ndarray | Callable[[float], float] | None,
    theta: Sequence[float] | np.ndarray | Callable[[float], float] | None,
    motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None,
    time: np.ndarray,
    dt_val: float,
    notes: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Validate and sample the six rigid-body motion axes and their time derivatives."""
    if motion is not None and not isinstance(motion, Mapping):
        raise InputError(f"motion must be a mapping from axis name to history, got {type(motion).__name__}.")
    m_map: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] = {} if motion is None else motion

    for key in m_map:
        if key not in _MOTION_AXES:
            raise InputError(f"Unknown motion axis {key!r}; allowed axes are {_MOTION_AXES}.")
    if z is not None and "z" in m_map:
        raise InputError("Motion axis 'z' cannot be set in both z and motion.")
    if theta is not None and "pitch" in m_map:
        raise InputError("Motion axis 'pitch' cannot be set in both theta and motion.")

    n_pts = int(time.size)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for axis in _MOTION_AXES:
        if axis == "z" and z is not None:
            out[axis] = _sample_motion(z, time, dt_val, "z", notes)
        elif axis == "pitch" and theta is not None:
            out[axis] = _sample_motion(theta, time, dt_val, "theta", notes)
        elif axis in m_map:
            out[axis] = _sample_motion(m_map[axis], time, dt_val, axis, notes)
        else:
            out[axis] = (np.zeros(n_pts, dtype=np.float64), np.zeros(n_pts, dtype=np.float64))
    return out


def _build_body_load_geometry(
    lattice: VortexLattice,
    rings: RingLattice,
    rp: np.ndarray,
    a_b: np.ndarray,
    b_b: np.ndarray,
    corners_b: np.ndarray,
    te_l_phys_b: np.ndarray,
    te_r_phys_b: np.ndarray,
    side_nodes_b: np.ndarray | None,
    a_kink_b: np.ndarray | None,
    b_kink_b: np.ndarray | None,
    leg_normal_b: np.ndarray,
    rc_load_ring: np.ndarray | float,
    rc_load_wake: np.ndarray | float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | float,
    np.ndarray | float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    """Build body-frame Kutta-Joukowski, TE-correction and apparent-mass geometry."""
    n_p = rings.n_panels
    p_strip = rings.panel_strip
    f_p = np.asarray(lattice.cp_frac, dtype=np.float64)[p_strip][:, None]
    mid_b = a_b + f_p * (b_b - a_b)
    l_vec_b = b_b - a_b

    a_te_p = te_l_phys_b[p_strip]
    b_te_p = te_r_phys_b[p_strip]
    if a_kink_b is None:
        leg_start = np.vstack([a_te_p, b_b])
        leg_end = np.vstack([a_b, b_te_p])
    else:
        if lattice.leg_next is None:
            a_e, b_e = a_te_p, b_te_p
        else:
            a_e, b_e = chain_ends(a_b, b_b, a_te_p, b_te_p, lattice.leg_next)
        leg_start = np.vstack([a_kink_b, b_b, a_e, b_kink_b])
        leg_end = np.vstack([a_b, b_kink_b, a_kink_b, b_e])

    leg_mid_b = 0.5 * (leg_start + leg_end)
    leg_vec_b = leg_end - leg_start
    P_load_b = np.vstack([mid_b, leg_mid_b])
    seg_vec_b = np.vstack([l_vec_b, leg_vec_b])
    seg_normal_b = np.vstack([np.zeros((n_p, 3), dtype=np.float64), leg_normal_b])
    arm_b = P_load_b - rp[None, :]

    load_unique_idx, load_inv_idx = _deduplicate_load_points(P_load_b, rc_load_ring, rc_load_wake)
    P_load_u_b = P_load_b[load_unique_idx]
    arm_u_b = arm_b[load_unique_idx]
    rc_load_u_ring = rc_load_ring[load_unique_idx] if np.ndim(rc_load_ring) == 2 else rc_load_ring
    rc_load_u_wake = rc_load_wake[load_unique_idx] if np.ndim(rc_load_wake) == 2 else rc_load_wake

    l_te_b = te_r_phys_b - te_l_phys_b
    p_te_b = te_l_phys_b + np.asarray(lattice.cp_frac, dtype=np.float64)[:, None] * l_te_b
    arm_te_b = p_te_b - rp[None, :]

    area_vec_b, ring_centroid_b, area_moment_vec_b = _physical_ring_areas_from_arrays(
        corners_b,
        te_l_phys_b,
        te_r_phys_b,
        rings.te_mask,
        side_nodes_b,
        rings.side_counts,
        rp,
    )
    arm_ring_b = ring_centroid_b - rp[None, :]
    return (
        P_load_u_b,
        arm_b,
        arm_u_b,
        seg_vec_b,
        seg_normal_b,
        load_unique_idx,
        load_inv_idx,
        rc_load_u_ring,
        rc_load_u_wake,
        p_te_b,
        l_te_b,
        arm_te_b,
        area_vec_b,
        arm_ring_b,
        area_moment_vec_b,
    )


@dataclass(slots=True)
class _StepLoads:
    """Aerodynamic loads and diagnostics of one time step of :class:`_TimeDomainStepper`.

    Attributes
    ----------
    CL, CD, CY : float
        Force coefficients along mean wind axes [-].
    Cl, Cm, Cn : float
        Moment coefficients in body axes of the mean state about the
        instantaneous reference point [-].
    strip_cl : numpy.ndarray
        Per-strip section lift coefficient [-], shape ``(n_strips,)``.
    gust_ref : numpy.ndarray
        Gust air velocity at the instantaneous reference point in geometry axes
        [m/s], shape ``(3,)``.
    F_tot : numpy.ndarray
        Dimensional aerodynamic force vector in mean geometry axes [N],
        shape ``(3,)``.
    M_tot : numpy.ndarray
        Dimensional aerodynamic moment vector about the instantaneous reference
        point in mean geometry axes [N m], shape ``(3,)``.
    """

    CL: float
    CD: float
    CY: float
    Cl: float
    Cm: float
    Cn: float
    strip_cl: np.ndarray
    gust_ref: np.ndarray
    F_tot: np.ndarray
    M_tot: np.ndarray


@dataclass(slots=True)
class _StepperState:
    """Saved mutable time-marching state of :class:`_TimeDomainStepper`."""

    step: int
    prev_base_l: Any | None
    prev_base_r: Any | None
    prev_gamma: Any | None
    prev2_gamma: Any | None
    wake_l_nodes: Any | None
    wake_r_nodes: Any | None


class _TimeDomainStepper:
    """Private single-step time-domain vortex-ring marcher with state save and restore.

    Holds the vortex-ring geometry, core radii, precomputed real bound
    influence matrix, optional stationary caches, and wake buffers for one
    time-domain run on the CPU or the GPU. Calling :meth:`advance` marches one
    step ``step`` from a given six-axis pose, pose rates and control-surface
    edge deflections and rates. Calling :meth:`save_state` before a step and
    :meth:`restore_state` inside sub-iterations rolls back the wake and bound
    circulation history (copying device buffers on the GPU without host round
    trips) so the step can be repeated with updated state iterates. On the GPU,
    a gust callable is evaluated on the host at every step and its velocities
    are copied to the device, so it adds a host cost per step.
    """

    def __init__(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        *,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        ref_point: np.ndarray | None = None,
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
        gust: Callable[[np.ndarray, float], np.ndarray] | None = None,
        ctrl_rig: _ControlRig | None = None,
        is_stationary: bool = False,
        m_axes: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        ctrl_arrs: dict[str, np.ndarray] | None = None,
        ctrl_dots: dict[str, np.ndarray] | None = None,
        build_Dynamic_controls: Sequence[str] | None = None,
    ) -> None:
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

        d_0 = freestream_direction(float(condition.alpha), float(condition.beta))
        L_0 = lift_direction(float(condition.alpha))
        S_0 = side_direction(float(condition.alpha), float(condition.beta))
        V_0 = V_inf * d_0
        e_x = d_0
        e_z = L_0 if ground is None else np.asarray(ground.normal, dtype=np.float64)
        if ground is None:
            e_y = S_0
        else:
            e_y_raw = cross3(e_z, e_x)
            e_y_norm = float(np.linalg.norm(e_y_raw))
            e_y = e_y_raw / e_y_norm if e_y_norm > 1e-12 else S_0

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
        wake_conv = dx_w * rings.wake_dir

        if gust is not None and not callable(gust):
            raise InputError(f"gust must be a callable gust(points, t) -> velocities or None, got {type(gust).__name__}.")
        gust_fn: Callable[[np.ndarray, float], np.ndarray] | None = None
        if gust is not None:
            bind_fn = getattr(gust, "_bind_flow", None)
            gust_fn = bind_fn(condition, ground) if callable(bind_fn) else gust
            if ground is not None:
                notes.append(
                    "In ground effect, wake rows convect using only the gust velocity component "
                    "parallel to the ground so that wake nodes do not cross the ground plane."
                )
                notes.append(
                    "In ground effect, the gust velocity field is applied as given (not reflected); "
                    "a gust with a normal component at the ground does not satisfy the ground "
                    "impermeability condition by itself."
                )
            if bool(getattr(gust, "_is_shear", False)):
                notes.append(
                    "Wind shear is rotational flow; the potential-flow vortex-ring method feels it "
                    "only as an onset velocity at the wing and wake nodes."
                )

        time = np.arange(n_steps_int + 1, dtype=np.float64) * dt_val
        leg_normal_0, leg_scale, _ = leg_force_rules(lattice)
        leg_start_0, leg_end_0 = leg_segments(lattice)
        n_leg_sets = leg_start_0.shape[0] // n_p

        delta0_by_name: dict[str, float] = {}
        if build_Dynamic_controls is not None:
            ctrl_rig, delta0_by_name = _build_dynamic_control_rig(
                lattice, rings, leg_normal_0, build_Dynamic_controls, n_steps_int + 1,
            )

        rc_ring, rc_wake = _ring_pair_cores(rings, rc, use_cross_surface_core)
        eval_panels = np.tile(np.arange(n_p, dtype=np.int64), 1 + n_leg_sets)
        eval_strips = rings.panel_strip[eval_panels]
        seg_scale = np.concatenate([np.ones(n_p, dtype=np.float64), leg_scale])
        rc_load_ring, rc_load_wake = _eval_pair_cores(rings, eval_panels, rc, use_cross_surface_core)
        rc_te_ring, rc_te_wake = _eval_pair_cores(rings, te_idx, rc, use_cross_surface_core)
        strip_area = np.asarray(lattice.chord, dtype=np.float64) * np.asarray(lattice.width, dtype=np.float64)

        a_b = np.asarray(lattice.a, dtype=np.float64)
        b_b = np.asarray(lattice.b, dtype=np.float64)
        cp_b = rings.cp
        normal_bc_b = rings.normal_bc
        corners_b = rings.corners
        te_left_b = rings.te_left
        te_right_b = rings.te_right
        te_l_phys_b = np.asarray(lattice.te_left, dtype=np.float64)
        te_r_phys_b = np.asarray(lattice.te_right, dtype=np.float64)
        side_nodes_b = rings.side_nodes
        a_kink_b = None if lattice.a_kink is None else np.asarray(lattice.a_kink, dtype=np.float64)
        b_kink_b = None if lattice.b_kink is None else np.asarray(lattice.b_kink, dtype=np.float64)
        leg_normal_b = leg_normal_0

        controls_vary = (ctrl_rig is not None) and ctrl_rig.deflections_vary
        if ctrl_rig is not None and not controls_vary:
            (
                a_b,
                b_b,
                cp_b,
                normal_bc_b,
                corners_b,
                te_left_b,
                te_right_b,
                te_l_phys_b,
                te_r_phys_b,
                side_nodes_b,
                a_kink_b,
                b_kink_b,
                leg_normal_b,
            ) = _deform_body_geometry(ctrl_rig, lattice, rings, ctrl_rig.dd_edges[0])

        if ctrl_rig is None:
            mid_b = np.asarray(lattice.force_points, dtype=np.float64)
            l_vec_b = b_b - a_b
            leg_mid_b = 0.5 * (leg_start_0 + leg_end_0)
            leg_vec_b = leg_end_0 - leg_start_0
            P_load_b = np.vstack([mid_b, leg_mid_b])
            seg_vec_b = np.vstack([l_vec_b, leg_vec_b])
            seg_normal_b = np.vstack([np.zeros((n_p, 3), dtype=np.float64), leg_normal_b])
            arm_b = P_load_b - rp[None, :]
            load_unique_idx, load_inv_idx = _deduplicate_load_points(P_load_b, rc_load_ring, rc_load_wake)
            P_load_u_b = P_load_b[load_unique_idx]
            arm_u_b = arm_b[load_unique_idx]
            rc_load_u_ring = rc_load_ring[load_unique_idx] if np.ndim(rc_load_ring) == 2 else rc_load_ring
            rc_load_u_wake = rc_load_wake[load_unique_idx] if np.ndim(rc_load_wake) == 2 else rc_load_wake
            l_te_b = te_r_phys_b - te_l_phys_b
            p_te_b = te_l_phys_b + np.asarray(lattice.cp_frac, dtype=np.float64)[:, None] * l_te_b
            arm_te_b = p_te_b - rp[None, :]
            area_vec_b, ring_centroid_b, area_moment_vec_b = _physical_ring_areas(lattice, rings, ref_point=rp)
            arm_ring_b = ring_centroid_b - rp[None, :]
        else:
            (
                P_load_u_b,
                arm_b,
                arm_u_b,
                seg_vec_b,
                seg_normal_b,
                load_unique_idx,
                load_inv_idx,
                rc_load_u_ring,
                rc_load_u_wake,
                p_te_b,
                l_te_b,
                arm_te_b,
                area_vec_b,
                arm_ring_b,
                area_moment_vec_b,
            ) = _build_body_load_geometry(
                lattice,
                rings,
                rp,
                a_b,
                b_b,
                corners_b,
                te_l_phys_b,
                te_r_phys_b,
                side_nodes_b,
                a_kink_b,
                b_kink_b,
                leg_normal_b,
                rc_load_ring,
                rc_load_wake,
            )

        n_eval_u = P_load_u_b.shape[0]
        cache_bound_load = (not controls_vary) and ((n_eval_u * n_p * 24) <= _MAX_CACHE_BYTES)
        is_stationary_eff = bool(is_stationary and (not controls_vary))
        is_stationary_wake = is_stationary_eff and (gust_fn is None)

        if backend in ("numpy", "numba"):
            device, precision, dev_notes = "cpu", "float64", []
        else:
            device, precision, dev_notes = _select_time_device(
                n_steps_int, n_p, n_s, ground is not None,
            )
        notes.extend(dev_notes)

        self.lattice = lattice
        self.condition = condition
        self.rings = rings
        self.ground = ground
        self.backend = backend
        self.te_correction = bool(te_correction)
        self.rp = rp
        self.V_inf = V_inf
        self.rho = rho
        self.S_ref = S_ref_f
        self.b_ref = b_ref_f
        self.c_ref = c_ref_f
        self.q_inf = q_inf
        self.qS = qS
        self.d_0 = d_0
        self.L_0 = L_0
        self.S_0 = S_0
        self.V_0 = V_0
        self.e_x = e_x
        self.e_y = e_y
        self.e_z = e_z
        self.n_steps = n_steps_int
        self.n_panels = n_p
        self.n_strips = n_s
        self.n_chord = rings.n_chord
        self.te_idx = te_idx
        self.dx_te_mean = dx_te_mean
        self.dt_default = dt_default
        self.dt = dt_val
        self.dx_w = dx_w
        self.f_te = f_te
        self.wake_conv = wake_conv
        self.time = time
        self.notes = notes
        self.device = device
        self.precision = precision
        self._gpu_state: _TimeDomainGPUState | None = None
        self.gust_fn = gust_fn
        self.leg_normal_0 = leg_normal_0
        self.n_leg_sets = n_leg_sets
        self.ctrl_rig = ctrl_rig
        self.controls_vary = controls_vary
        self.delta0_by_name = delta0_by_name
        self.rc_ring = rc_ring
        self.rc_wake = rc_wake
        self.eval_strips = eval_strips
        self.seg_scale = seg_scale
        self.rc_load_ring = rc_load_ring
        self.rc_load_wake = rc_load_wake
        self.rc_te_ring = rc_te_ring
        self.rc_te_wake = rc_te_wake
        self.strip_area = strip_area

        self.a_b = a_b
        self.b_b = b_b
        self.cp_b = cp_b
        self.normal_bc_b = normal_bc_b
        self.corners_b = corners_b
        self.te_left_b = te_left_b
        self.te_right_b = te_right_b
        self.te_l_phys_b = te_l_phys_b
        self.te_r_phys_b = te_r_phys_b
        self.side_nodes_b = side_nodes_b
        self.a_kink_b = a_kink_b
        self.b_kink_b = b_kink_b
        self.leg_normal_b = leg_normal_b

        self.P_load_u_b = P_load_u_b
        self.arm_b = arm_b
        self.arm_u_b = arm_u_b
        self.seg_vec_b = seg_vec_b
        self.seg_normal_b = seg_normal_b
        self.load_unique_idx = load_unique_idx
        self.load_inv_idx = load_inv_idx
        self.rc_load_u_ring = rc_load_u_ring
        self.rc_load_u_wake = rc_load_u_wake
        self.p_te_b = p_te_b
        self.l_te_b = l_te_b
        self.arm_te_b = arm_te_b
        self.area_vec_b = area_vec_b
        self.arm_ring_b = arm_ring_b
        self.area_moment_vec_b = area_moment_vec_b

        self.cache_bound_load = cache_bound_load
        self.is_stationary = is_stationary_eff
        self._is_stationary_wake = is_stationary_wake
        self.A_bb_real: np.ndarray | None = None
        self.lu_Abb: Any = None
        self.V_bound_load_b: np.ndarray | None = None
        self.V_bound_te_b: np.ndarray | None = None
        self.A_wake_stat: np.ndarray | None = None
        self.V_wake_load_stat: np.ndarray | None = None
        self.V_wake_te_stat: np.ndarray | None = None
        self.V_bound_img_load_stat: np.ndarray | None = None
        self.V_bound_img_te_stat: np.ndarray | None = None
        self.wake_corners_earth: np.ndarray | None = None
        self.wake_corners_cur: np.ndarray | None = None
        self.wake_gamma_buf: np.ndarray | None = None
        self.wake_l_nodes: np.ndarray | None = None
        self.wake_r_nodes: np.ndarray | None = None
        self.v_base_wake = V_inf * rings.wake_dir

        self.step = 0
        self.prev_base_l: np.ndarray | None = None
        self.prev_base_r: np.ndarray | None = None
        self.prev_gamma: np.ndarray | None = None
        self.prev2_gamma: np.ndarray | None = None
        self.prev_pose_ge: tuple[float, float, float, float, float, float] | None = None
        self.prev_dd: np.ndarray | None = None

        self.m_axes = m_axes
        self.ctrl_arrs = {} if ctrl_arrs is None else ctrl_arrs
        self.ctrl_dots = {} if ctrl_dots is None else ctrl_dots

        if ground is not None:
            if ctrl_rig is None:
                self._wing_pts_rel: np.ndarray | None = lattice.all_points() - rp[None, :]
                self._fixed_pts: np.ndarray | None = None
            else:
                self._wing_pts_rel = None
                le_pts = np.vstack([sf.edge_le for sf in lattice.surfaces])
                self._fixed_pts = np.vstack([le_pts, ctrl_rig.hinge_points])
            self._cached_pts_rel: np.ndarray | None = None
        else:
            self._wing_pts_rel = None
            self._fixed_pts = None
            self._cached_pts_rel = None

        if self.device == "gpu":
            self._init_with_gpu_fallback()
        else:
            self._init_cpu_matrices()

    def _init_cpu_matrices(self) -> None:
        """Assemble CPU bound influence matrix, caches and wake buffers."""
        rings = self.rings
        n_p = self.n_panels
        n_s = self.n_strips
        n_steps_int = self.n_steps
        n_eval_u = self.P_load_u_b.shape[0]
        backend = self.backend

        A_bb_real = ring_normal_influence(
            rings.cp,
            rings.normal_bc,
            rings.corners,
            self.rc_ring,
            side_nodes=rings.side_nodes,
            side_counts=rings.side_counts,
            backend=backend,
        )
        if self.ctrl_rig is not None and not self.controls_vary:
            _update_bound_influence_blocks(
                A_bb_real,
                self.ctrl_rig,
                self.cp_b,
                self.normal_bc_b,
                self.corners_b,
                self.side_nodes_b,
                rings.side_counts,
                self.rc_ring,
                backend=backend,
            )

        self.A_bb_real = A_bb_real
        self.lu_Abb = sl.lu_factor(A_bb_real) if (self.ground is None and not self.controls_vary) else None
        self.V_bound_load_b = (
            ring_velocity(
                self.P_load_u_b,
                self.corners_b,
                self.rc_load_u_ring,
                side_nodes=self.side_nodes_b,
                side_counts=rings.side_counts,
                backend=backend,
            )
            if self.cache_bound_load
            else None
        )
        self.V_bound_te_b = (
            ring_velocity(
                self.p_te_b,
                self.corners_b,
                self.rc_te_ring,
                side_nodes=self.side_nodes_b,
                side_counts=rings.side_counts,
                backend=backend,
            )
            if (not self.controls_vary) and self.te_correction and ((n_s * n_p * 24) <= _MAX_CACHE_BYTES)
            else None
        )
        is_sw = self._is_stationary_wake
        self.A_wake_stat = (
            np.empty((n_steps_int, n_p, n_s), dtype=np.float64)
            if is_sw and (n_steps_int * n_p * n_s * 8) <= _MAX_CACHE_BYTES
            else None
        )
        self.V_wake_load_stat = (
            np.empty((n_steps_int, n_eval_u, n_s, 3), dtype=np.float64)
            if is_sw and (n_steps_int * n_eval_u * n_s * 24) <= _MAX_CACHE_BYTES
            else None
        )
        self.V_wake_te_stat = (
            np.empty((n_steps_int, n_s, n_s, 3), dtype=np.float64)
            if is_sw and self.te_correction and (n_steps_int * n_s * n_s * 24) <= _MAX_CACHE_BYTES
            else None
        )
        self.V_bound_img_load_stat = None
        self.V_bound_img_te_stat = None

        self.wake_corners_earth = np.empty((n_steps_int, n_s, 4, 3), dtype=np.float64)
        self.wake_corners_cur = np.empty((n_steps_int, n_s, 4, 3), dtype=np.float64)
        self.wake_gamma_buf = np.empty((n_steps_int, n_s), dtype=np.float64)
        self.wake_l_nodes = (
            np.empty((n_steps_int + 1, n_s, 3), dtype=np.float64) if self.gust_fn is not None else None
        )
        self.wake_r_nodes = (
            np.empty((n_steps_int + 1, n_s, 3), dtype=np.float64) if self.gust_fn is not None else None
        )

    def _init_with_gpu_fallback(self) -> None:
        """Initialise GPU buffers with one OOM retry, falling back to CPU after two OOM errors."""
        from ventorum.gpu import memory as gpu_memory
        from ventorum.gpu.pipeline import is_out_of_memory

        message = ""
        for _attempt in range(2):
            try:
                self._init_gpu()
                return
            except Exception as exc:  # noqa: BLE001 - only out-of-memory errors are caught
                if not is_out_of_memory(exc):
                    self.release_gpu()
                    raise
                message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            self.release_gpu()
            gpu_memory.release()
        warnings.warn(
            f"The GPU ran out of memory twice ({message}). The time-domain march falls back to the CPU.",
            RuntimeWarning,
            stacklevel=4,
        )
        self.device = "cpu"
        self.precision = "float64"
        if NOTE_GPU_OUT_OF_MEMORY not in self.notes:
            self.notes.append(NOTE_GPU_OUT_OF_MEMORY)
        self._init_cpu_matrices()

    def _upload_body_geometry_gpu(self) -> None:
        """Upload current body-frame geometry and load arrays to the GPU state."""
        import torch

        from ventorum.gpu import engine

        assert self._gpu_state is not None
        st = self._gpu_state.tensors
        dev = engine._dev()
        st["cp_b"] = engine._f64(self.cp_b)
        st["normal_bc_b"] = engine._f64(self.normal_bc_b)
        st["corners_b"] = engine._f64(self.corners_b)
        st["te_left_b"] = engine._f64(self.te_left_b)
        st["te_right_b"] = engine._f64(self.te_right_b)
        if self.side_nodes_b is not None:
            st["side_nodes_b"] = engine._f64(self.side_nodes_b)
        else:
            st["side_nodes_b"] = None

        st["P_load_u_b"] = engine._f64(self.P_load_u_b)
        st["arm_b"] = engine._f64(self.arm_b)
        st["arm_u_b"] = engine._f64(self.arm_u_b)
        st["seg_vec_b"] = engine._f64(self.seg_vec_b)
        st["seg_normal_b"] = engine._f64(self.seg_normal_b)
        st["load_unique_idx"] = torch.as_tensor(self.load_unique_idx, dtype=torch.int64, device=dev)
        st["load_inv_idx"] = torch.as_tensor(self.load_inv_idx, dtype=torch.int64, device=dev)
        st["rc_load_u_ring"] = _to_gpu_rc(self.rc_load_u_ring)
        st["rc_load_u_wake"] = _to_gpu_rc(self.rc_load_u_wake)
        st["p_te_b"] = engine._f64(self.p_te_b)
        st["l_te_b"] = engine._f64(self.l_te_b)
        st["arm_te_b"] = engine._f64(self.arm_te_b)
        st["area_vec_b"] = engine._f64(self.area_vec_b)
        st["arm_ring_b"] = engine._f64(self.arm_ring_b)
        st["area_moment_vec_b"] = None if self.area_moment_vec_b is None else engine._f64(self.area_moment_vec_b)

    def _init_gpu(self) -> None:
        """Assemble GPU bound influence matrix, caches and device wake buffers."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        rings = self.rings
        n_p = self.n_panels
        n_s = self.n_strips
        n_steps_int = self.n_steps
        n_eval_u = self.P_load_u_b.shape[0]
        prec = self.precision
        t_dtype = engine.torch_dtype(prec)
        dev = engine._dev()

        self._gpu_state = _TimeDomainGPUState(prec)
        st = self._gpu_state.tensors

        st["rp"] = engine._f64(self.rp)
        st["V_0"] = engine._f64(self.V_0)
        st["L_0"] = engine._f64(self.L_0)
        st["f_te"] = engine._f64(self.f_te)
        st["wake_conv"] = engine._f64(self.wake_conv)
        st["v_base_wake"] = engine._f64(self.v_base_wake)
        st["te_idx"] = torch.as_tensor(self.te_idx, dtype=torch.int64, device=dev)
        st["eval_strips"] = torch.as_tensor(self.eval_strips, dtype=torch.int64, device=dev)
        st["panel_strip"] = torch.as_tensor(rings.panel_strip, dtype=torch.int64, device=dev)
        st["seg_scale"] = engine._f64(self.seg_scale)
        st["g_norm"] = None if self.ground is None else engine._f64(self.ground.normal)
        st["side_counts"] = (
            None
            if (self.side_nodes_b is None or rings.side_counts is None)
            else engine._i32(rings.side_counts)
        )
        st["rc_ring"] = _to_gpu_rc(self.rc_ring)
        st["rc_wake"] = _to_gpu_rc(self.rc_wake)
        st["rc_te_ring"] = _to_gpu_rc(self.rc_te_ring)
        st["rc_te_wake"] = _to_gpu_rc(self.rc_te_wake)

        if self.ctrl_rig is not None:
            st["moving_col_panels"] = torch.as_tensor(self.ctrl_rig.moving_col_panels, dtype=torch.int64, device=dev)
            st["moving_row_panels"] = torch.as_tensor(self.ctrl_rig.moving_row_panels, dtype=torch.int64, device=dev)
            st["fixed_col_panels"] = torch.as_tensor(self.ctrl_rig.fixed_col_panels, dtype=torch.int64, device=dev)

        chain_levels: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.lattice.leg_next is not None and bool(np.any(self.lattice.leg_next)):
            pos = chain_position(self.lattice.leg_next)
            for lvl in range(1, int(pos.max(initial=0)) + 1):
                dst = np.flatnonzero(pos == lvl)
                chain_levels.append(
                    (
                        torch.as_tensor(dst, dtype=torch.int64, device=dev),
                        torch.as_tensor(dst - 1, dtype=torch.int64, device=dev),
                    )
                )
        st["chain_levels"] = chain_levels

        self._upload_body_geometry_gpu()

        sc_g = st["side_counts"]
        A_bb_real_g = RK.ring_normal_influence_gpu(
            st["cp_b"],
            st["normal_bc_b"],
            st["corners_b"],
            st["rc_ring"],
            side_nodes=st["side_nodes_b"],
            side_counts=sc_g,
            ground=None,
            precision=prec,
            out_tensor=True,
        ).to(torch.float64)
        st["A_bb_real"] = A_bb_real_g
        if self.ground is None and not self.controls_vary:
            st["lu_Abb"], st["piv_Abb"] = torch.linalg.lu_factor(A_bb_real_g)
        else:
            st["lu_Abb"] = None
            st["piv_Abb"] = None

        if self.cache_bound_load:
            st["V_bound_load_b"] = RK.ring_velocity_gpu(
                st["P_load_u_b"],
                st["corners_b"],
                st["rc_load_u_ring"],
                side_nodes=st["side_nodes_b"],
                side_counts=sc_g,
                ground=None,
                precision=prec,
                out_tensor=True,
            )
        else:
            st["V_bound_load_b"] = None

        if (not self.controls_vary) and self.te_correction and ((n_s * n_p * 24) <= _MAX_CACHE_BYTES):
            st["V_bound_te_b"] = RK.ring_velocity_gpu(
                st["p_te_b"],
                st["corners_b"],
                st["rc_te_ring"],
                side_nodes=st["side_nodes_b"],
                side_counts=sc_g,
                ground=None,
                precision=prec,
                out_tensor=True,
            )
        else:
            st["V_bound_te_b"] = None

        is_sw = self._is_stationary_wake
        st["A_wake_stat"] = (
            torch.empty((n_steps_int, n_p, n_s), dtype=t_dtype, device=dev)
            if is_sw and (n_steps_int * n_p * n_s * 8) <= _MAX_CACHE_BYTES
            else None
        )
        st["V_wake_load_stat"] = (
            torch.empty((n_steps_int, n_eval_u, n_s, 3), dtype=t_dtype, device=dev)
            if is_sw and (n_steps_int * n_eval_u * n_s * 24) <= _MAX_CACHE_BYTES
            else None
        )
        st["V_wake_te_stat"] = (
            torch.empty((n_steps_int, n_s, n_s, 3), dtype=t_dtype, device=dev)
            if is_sw and self.te_correction and (n_steps_int * n_s * n_s * 24) <= _MAX_CACHE_BYTES
            else None
        )
        st["V_bound_img_load_stat"] = None
        st["V_bound_img_te_stat"] = None

        st["wake_corners_earth"] = (
            torch.empty((n_steps_int, n_s, 4, 3), dtype=torch.float64, device=dev)
            if self.gust_fn is None
            else None
        )
        st["wake_corners_cur"] = torch.empty((n_steps_int, n_s, 4, 3), dtype=torch.float64, device=dev)
        st["wake_gamma_buf"] = torch.empty((n_steps_int, n_s), dtype=torch.float64, device=dev)
        st["wake_l_nodes"] = (
            torch.empty((n_steps_int + 1, n_s, 3), dtype=torch.float64, device=dev)
            if self.gust_fn is not None
            else None
        )
        st["wake_r_nodes"] = (
            torch.empty((n_steps_int + 1, n_s, 3), dtype=torch.float64, device=dev)
            if self.gust_fn is not None
            else None
        )
        st["prev_base_l"] = None
        st["prev_base_r"] = None
        st["prev_gamma"] = None
        st["prev2_gamma"] = None

    def release_gpu(self) -> int:
        """Release all CUDA tensors held by this stepper and return freed bytes [B]."""
        if self._gpu_state is None:
            return 0
        from ventorum.gpu import ring_kernels as RK

        freed = self._gpu_state.release_gpu(force=True)
        RK.unregister_device_holder(self._gpu_state)
        self._gpu_state = None
        return freed

    @classmethod
    def from_prescribed(
        cls,
        lattice: VortexLattice,
        condition: FlightCondition,
        *,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        ref_point: np.ndarray | None = None,
        z: Sequence[float] | np.ndarray | Callable[[float], float] | None = None,
        theta: Sequence[float] | np.ndarray | Callable[[float], float] | None = None,
        motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
        controls: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
        gust: Callable[[np.ndarray, float], np.ndarray] | None = None,
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
        control_increments: bool = False,
    ) -> _TimeDomainStepper:
        """Build a :class:`_TimeDomainStepper` with pre-sampled prescribed motion and controls.

        With *control_increments* the values of *controls* are increments over
        the built deflection of each surface instead of absolute deflections.
        """
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
        d_0 = freestream_direction(float(condition.alpha), float(condition.beta))
        if wake_dir is None:
            wd_in = np.array([1.0, 0.0, 0.0], dtype=np.float64) if (wake_alignment == "body" and ground is None) else d_0
        else:
            wd_in = np.asarray(wake_dir, dtype=np.float64)
        rings = build_ring_lattice(lattice, wake_dir=wd_in, te_extension_fraction=te_extension_fraction)
        dx_te = np.asarray(lattice.panel_length[rings.te_indices], dtype=np.float64)
        dx_te_mean = float(np.sum(rings.strip_width * dx_te) / np.sum(rings.strip_width))
        if not math.isfinite(dx_te_mean) or dx_te_mean <= 0.0:
            raise InputError(f"Invalid mean trailing-edge panel length dx_te_mean={dx_te_mean!r}.")
        dt_default = dx_te_mean / V_inf
        if dt is None:
            dt_val = dt_default
        else:
            dt_val = float(dt)
            if not math.isfinite(dt_val) or dt_val <= 0.0:
                raise InputError(f"dt must be finite and > 0, got {dt!r}.")

        if gust is not None and not callable(gust):
            raise InputError(
                f"gust must be a callable gust(points, t) -> velocities or None, got {type(gust).__name__}."
            )

        time = np.arange(n_steps_int + 1, dtype=np.float64) * dt_val
        input_notes: list[str] = []
        m_axes = _resolve_six_axis_motion(z, theta, motion, time, dt_val, input_notes)
        leg_normal_0, _, _ = leg_force_rules(lattice)
        ctrl_rig, ctrl_arrs, ctrl_dots = _build_control_rig(
            lattice, rings, leg_normal_0, controls, time, dt_val, input_notes,
            increments=control_increments,
        )

        # Validate ground clearance at every step before constructing the stepper.
        if ground is not None:
            S_0 = side_direction(float(condition.alpha), float(condition.beta))
            e_x = d_0
            e_z = np.asarray(ground.normal, dtype=np.float64)
            e_y_raw = cross3(e_z, e_x)
            e_y_norm = float(np.linalg.norm(e_y_raw))
            e_y = e_y_raw / e_y_norm if e_y_norm > 1e-12 else S_0
            x_arr, _ = m_axes["x"]
            y_arr, _ = m_axes["y"]
            z_arr, _ = m_axes["z"]
            roll_arr, _ = m_axes["roll"]
            pitch_arr, _ = m_axes["pitch"]
            yaw_arr, _ = m_axes["yaw"]

            if ctrl_rig is None:
                wing_pts_rel = lattice.all_points() - rp[None, :]
                for step in range(n_steps_int + 1):
                    x_ref_s = rp + x_arr[step] * e_x + y_arr[step] * e_y + z_arr[step] * e_z
                    rn, pn, yn = float(roll_arr[step]), float(pitch_arr[step]), float(yaw_arr[step])
                    if rn == 0.0 and pn == 0.0 and yn == 0.0:
                        pts_s = x_ref_s[None, :] + wing_pts_rel
                    elif rn == 0.0 and yn == 0.0:
                        pts_s = x_ref_s[None, :] + _rot_y(wing_pts_rel, pn)
                    else:
                        R_s = _rotation_matrix_321(rn, pn, yn)
                        pts_s = x_ref_s[None, :] + wing_pts_rel @ R_s.T
                    h_min = float(np.min(ground.height(pts_s)))
                    if h_min <= 0.0:
                        raise InputError(
                            f"Wing node reaches or crosses the ground plane at step {step} "
                            f"(t = {time[step]:.6g} s, minimum height {h_min:.4g} m)."
                        )
            else:
                le_pts = np.vstack([sf.edge_le for sf in lattice.surfaces])
                fixed_pts = np.vstack([le_pts, ctrl_rig.hinge_points])
                cached_pts_rel: np.ndarray | None = None
                for step in range(n_steps_int + 1):
                    if cached_pts_rel is None or ctrl_rig.deflections_vary:
                        deformed_s = _deform_body_geometry(ctrl_rig, lattice, rings, ctrl_rig.dd_edges[step])
                        a_s, b_s, cp_s = deformed_s[0], deformed_s[1], deformed_s[2]
                        te_lp_s, te_rp_s = deformed_s[7], deformed_s[8]
                        pts_rel_s = np.vstack([fixed_pts, a_s, b_s, cp_s, te_lp_s, te_rp_s]) - rp[None, :]
                        if not ctrl_rig.deflections_vary:
                            cached_pts_rel = pts_rel_s
                    else:
                        pts_rel_s = cached_pts_rel
                    x_ref_s = rp + x_arr[step] * e_x + y_arr[step] * e_y + z_arr[step] * e_z
                    rn, pn, yn = float(roll_arr[step]), float(pitch_arr[step]), float(yaw_arr[step])
                    if rn == 0.0 and pn == 0.0 and yn == 0.0:
                        pts_s = x_ref_s[None, :] + pts_rel_s
                    elif rn == 0.0 and yn == 0.0:
                        pts_s = x_ref_s[None, :] + _rot_y(pts_rel_s, pn)
                    else:
                        R_s = _rotation_matrix_321(rn, pn, yn)
                        pts_s = x_ref_s[None, :] + pts_rel_s @ R_s.T
                    h_min = float(np.min(ground.height(pts_s)))
                    if h_min <= 0.0:
                        raise InputError(
                            f"Wing or flap node reaches or crosses the ground plane at step {step} "
                            f"(t = {time[step]:.6g} s, minimum height {h_min:.4g} m)."
                        )

        controls_vary = (ctrl_rig is not None) and ctrl_rig.deflections_vary
        is_stationary = bool(
            (not controls_vary)
            and all(np.all(m_axes[axis][0] == m_axes[axis][0][0]) for axis in _MOTION_AXES)
        )

        stepper = cls(
            lattice,
            condition,
            S_ref=S_ref,
            b_ref=b_ref,
            c_ref=c_ref,
            ref_point=rp,
            n_steps=n_steps_int,
            dt=dt,
            ground=ground,
            wake_dir=wake_dir,
            te_extension_fraction=te_extension_fraction,
            rc=rc,
            use_cross_surface_core=use_cross_surface_core,
            te_correction=te_correction,
            wake_alignment=wake_alignment,
            backend=backend,
            gust=gust,
            ctrl_rig=ctrl_rig,
            is_stationary=is_stationary,
            m_axes=m_axes,
            ctrl_arrs=ctrl_arrs,
            ctrl_dots=ctrl_dots,
        )
        stepper.notes.extend(input_notes)
        return stepper

    def min_ground_clearance(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        dd_step: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray] | None:
        """Return ``(h_min, lowest_point)`` [m] above ``ground``, or None in free air."""
        if self.ground is None:
            return None
        if self.ctrl_rig is None:
            assert self._wing_pts_rel is not None
            pts_rel = self._wing_pts_rel
        else:
            assert self._fixed_pts is not None
            if dd_step is None:
                dd_use = self.ctrl_rig.dd_edges[min(self.step, self.n_steps)]
            else:
                dd_use = dd_step
            if self._cached_pts_rel is None or self.ctrl_rig.deflections_vary or dd_step is not None:
                deformed_s = _deform_body_geometry(self.ctrl_rig, self.lattice, self.rings, dd_use)
                a_s, b_s, cp_s = deformed_s[0], deformed_s[1], deformed_s[2]
                te_lp_s, te_rp_s = deformed_s[7], deformed_s[8]
                pts_rel = np.vstack([self._fixed_pts, a_s, b_s, cp_s, te_lp_s, te_rp_s]) - self.rp[None, :]
                if not self.ctrl_rig.deflections_vary and dd_step is None:
                    self._cached_pts_rel = pts_rel
            else:
                pts_rel = self._cached_pts_rel

        x_ref_s = self.rp + float(x) * self.e_x + float(y) * self.e_y + float(z) * self.e_z
        rn, pn, yn = float(roll), float(pitch), float(yaw)
        if rn == 0.0 and pn == 0.0 and yn == 0.0:
            pts_s = x_ref_s[None, :] + pts_rel
        elif rn == 0.0 and yn == 0.0:
            pts_s = x_ref_s[None, :] + _rot_y(pts_rel, pn)
        else:
            R_s = _rotation_matrix_321(rn, pn, yn)
            pts_s = x_ref_s[None, :] + pts_rel @ R_s.T
        heights = self.ground.height(pts_s)
        idx_min = int(np.argmin(heights))
        return float(heights[idx_min]), pts_s[idx_min].copy()

    def save_state(self) -> _StepperState:
        """Return a snapshot of the mutable time-marching state before the current step."""
        s = self.step
        if self._gpu_state is not None:
            st = self._gpu_state.tensors
            wl = st.get("wake_l_nodes")
            wr = st.get("wake_r_nodes")
            pbl = st.get("prev_base_l")
            pbr = st.get("prev_base_r")
            pg = st.get("prev_gamma")
            p2g = st.get("prev2_gamma")
            return _StepperState(
                step=s,
                prev_base_l=None if pbl is None else pbl.clone(),
                prev_base_r=None if pbr is None else pbr.clone(),
                prev_gamma=None if pg is None else pg.clone(),
                prev2_gamma=None if p2g is None else p2g.clone(),
                wake_l_nodes=None if (wl is None or s == 0) else wl[:s].clone(),
                wake_r_nodes=None if (wr is None or s == 0) else wr[:s].clone(),
            )
        return _StepperState(
            step=s,
            prev_base_l=None if self.prev_base_l is None else self.prev_base_l.copy(),
            prev_base_r=None if self.prev_base_r is None else self.prev_base_r.copy(),
            prev_gamma=None if self.prev_gamma is None else self.prev_gamma.copy(),
            prev2_gamma=None if self.prev2_gamma is None else self.prev2_gamma.copy(),
            wake_l_nodes=(
                None if (self.wake_l_nodes is None or s == 0) else self.wake_l_nodes[:s].copy()
            ),
            wake_r_nodes=(
                None if (self.wake_r_nodes is None or s == 0) else self.wake_r_nodes[:s].copy()
            ),
        )

    def restore_state(self, saved: _StepperState) -> None:
        """Restore the time-marching state to *saved* so step ``saved.step`` can be repeated."""
        s = int(saved.step)
        self.step = s
        if self._gpu_state is not None:
            st = self._gpu_state.tensors
            st["prev_base_l"] = None if saved.prev_base_l is None else saved.prev_base_l.clone()
            st["prev_base_r"] = None if saved.prev_base_r is None else saved.prev_base_r.clone()
            st["prev_gamma"] = None if saved.prev_gamma is None else saved.prev_gamma.clone()
            st["prev2_gamma"] = None if saved.prev2_gamma is None else saved.prev2_gamma.clone()
            wl = st.get("wake_l_nodes")
            wr = st.get("wake_r_nodes")
            wcc = st["wake_corners_cur"]
            if wl is not None and saved.wake_l_nodes is not None:
                assert wr is not None and saved.wake_r_nodes is not None
                wl[:s] = saved.wake_l_nodes
                wr[:s] = saved.wake_r_nodes
                if s >= 2:
                    n_w = s - 1
                    wcc[:n_w, :, 0, :] = wl[1 : n_w + 1]
                    wcc[:n_w, :, 1, :] = wr[1 : n_w + 1]
                    wcc[:n_w, :, 2, :] = wr[:n_w]
                    wcc[:n_w, :, 3, :] = wl[:n_w]
            elif self.gust_fn is None and s >= 2:
                n_w = s - 1
                shift_prev_g = float(n_w) * st["wake_conv"]
                wcc[:n_w] = st["wake_corners_earth"][:n_w] + shift_prev_g[None, :, None, :]
            return
        self.prev_base_l = None if saved.prev_base_l is None else saved.prev_base_l.copy()
        self.prev_base_r = None if saved.prev_base_r is None else saved.prev_base_r.copy()
        self.prev_gamma = None if saved.prev_gamma is None else saved.prev_gamma.copy()
        self.prev2_gamma = None if saved.prev2_gamma is None else saved.prev2_gamma.copy()
        if self.wake_l_nodes is not None and saved.wake_l_nodes is not None:
            self.wake_l_nodes[:s] = saved.wake_l_nodes
            assert self.wake_r_nodes is not None and saved.wake_r_nodes is not None
            self.wake_r_nodes[:s] = saved.wake_r_nodes
            if s >= 2:
                n_w = s - 1
                assert self.wake_corners_cur is not None
                self.wake_corners_cur[:n_w, :, 0, :] = self.wake_l_nodes[1 : n_w + 1]
                self.wake_corners_cur[:n_w, :, 1, :] = self.wake_r_nodes[1 : n_w + 1]
                self.wake_corners_cur[:n_w, :, 2, :] = self.wake_r_nodes[:n_w]
                self.wake_corners_cur[:n_w, :, 3, :] = self.wake_l_nodes[:n_w]
        elif self.gust_fn is None and s >= 2:
            n_w = s - 1
            shift_prev = float(n_w) * self.wake_conv
            assert self.wake_corners_cur is not None and self.wake_corners_earth is not None
            self.wake_corners_cur[:n_w] = self.wake_corners_earth[:n_w] + shift_prev[None, :, None, :]

    def advance_prescribed(self, step: int | None = None) -> _StepLoads:
        """Advance one step using the pre-sampled prescribed motion and control histories."""
        assert self.m_axes is not None
        s = self.step if step is None else int(step)
        self.step = s
        dd_s = None if self.ctrl_rig is None else self.ctrl_rig.dd_edges[s]
        rate_s = None if self.ctrl_rig is None else self.ctrl_rig.rate_edges[s]
        return self.advance(
            x=float(self.m_axes["x"][0][s]),
            y=float(self.m_axes["y"][0][s]),
            z=float(self.m_axes["z"][0][s]),
            roll=float(self.m_axes["roll"][0][s]),
            pitch=float(self.m_axes["pitch"][0][s]),
            yaw=float(self.m_axes["yaw"][0][s]),
            x_dot=float(self.m_axes["x"][1][s]),
            y_dot=float(self.m_axes["y"][1][s]),
            z_dot=float(self.m_axes["z"][1][s]),
            roll_dot=float(self.m_axes["roll"][1][s]),
            pitch_dot=float(self.m_axes["pitch"][1][s]),
            yaw_dot=float(self.m_axes["yaw"][1][s]),
            dd_step=dd_s,
            rate_step=rate_s,
        )

    def advance(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        x_dot: float,
        y_dot: float,
        z_dot: float,
        roll_dot: float,
        pitch_dot: float,
        yaw_dot: float,
        dd_step: np.ndarray | None = None,
        rate_step: np.ndarray | None = None,
    ) -> _StepLoads:
        """Advance step ``self.step`` for the given pose, rates and control deflections."""
        if self._gpu_state is not None:
            return self._advance_gpu(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                x_dot=x_dot,
                y_dot=y_dot,
                z_dot=z_dot,
                roll_dot=roll_dot,
                pitch_dot=pitch_dot,
                yaw_dot=yaw_dot,
                dd_step=dd_step,
                rate_step=rate_step,
            )
        step = self.step
        xn, yn, zn = float(x), float(y), float(z)
        roll_n, pitch_n, yaw_n = float(roll), float(pitch), float(yaw)
        xd, yd, zd = float(x_dot), float(y_dot), float(z_dot)
        roll_d, pitch_d, yaw_d = float(roll_dot), float(pitch_dot), float(yaw_dot)

        rp = self.rp
        e_x, e_y, e_z = self.e_x, self.e_y, self.e_z
        rings = self.rings
        lattice = self.lattice
        ground = self.ground
        backend = self.backend
        ctrl_rig = self.ctrl_rig

        x_ref = rp + xn * e_x + yn * e_y + zn * e_z
        v_trans = xd * e_x + yd * e_y + zd * e_z
        omega_g = _angular_velocity_321(pitch_n, yaw_n, roll_d, pitch_d, yaw_d)

        if roll_n == 0.0 and pitch_n == 0.0 and yaw_n == 0.0:
            R: np.ndarray | None = None
            pitch_only: float | None = None
        elif roll_n == 0.0 and yaw_n == 0.0:
            R = _rotation_matrix_321(0.0, pitch_n, 0.0)
            pitch_only = pitch_n
        else:
            R = _rotation_matrix_321(roll_n, pitch_n, yaw_n)
            pitch_only = None

        ctrl_changed = False
        if self.controls_vary and ctrl_rig is not None:
            dd_use = ctrl_rig.dd_edges[step] if dd_step is None else dd_step
            if self.prev_dd is None or not np.array_equal(dd_use, self.prev_dd):
                (
                    self.a_b,
                    self.b_b,
                    self.cp_b,
                    self.normal_bc_b,
                    self.corners_b,
                    self.te_left_b,
                    self.te_right_b,
                    self.te_l_phys_b,
                    self.te_r_phys_b,
                    self.side_nodes_b,
                    self.a_kink_b,
                    self.b_kink_b,
                    self.leg_normal_b,
                ) = _deform_body_geometry(ctrl_rig, lattice, rings, dd_use)
                (
                    self.P_load_u_b,
                    self.arm_b,
                    self.arm_u_b,
                    self.seg_vec_b,
                    self.seg_normal_b,
                    self.load_unique_idx,
                    self.load_inv_idx,
                    self.rc_load_u_ring,
                    self.rc_load_u_wake,
                    self.p_te_b,
                    self.l_te_b,
                    self.arm_te_b,
                    self.area_vec_b,
                    self.arm_ring_b,
                    self.area_moment_vec_b,
                ) = _build_body_load_geometry(
                    lattice,
                    rings,
                    rp,
                    self.a_b,
                    self.b_b,
                    self.corners_b,
                    self.te_l_phys_b,
                    self.te_r_phys_b,
                    self.side_nodes_b,
                    self.a_kink_b,
                    self.b_kink_b,
                    self.leg_normal_b,
                    self.rc_load_ring,
                    self.rc_load_wake,
                )
                _update_bound_influence_blocks(
                    self.A_bb_real,
                    ctrl_rig,
                    self.cp_b,
                    self.normal_bc_b,
                    self.corners_b,
                    self.side_nodes_b,
                    rings.side_counts,
                    self.rc_ring,
                    backend=backend,
                )
                if ground is None:
                    self.lu_Abb = sl.lu_factor(self.A_bb_real)
                self.prev_dd = dd_use.copy()
                self.prev_pose_ge = None
                ctrl_changed = True

        cp, normal_bc, corners, te_left, te_right, side_nodes = _move_wing_geometry(
            self.cp_b,
            self.normal_bc_b,
            self.corners_b,
            self.te_left_b,
            self.te_right_b,
            self.side_nodes_b,
            rp,
            x_ref,
            R,
            pitch_only,
        )

        if ctrl_rig is not None:
            rate_use = ctrl_rig.rate_edges[step] if rate_step is None else rate_step
            v_fl_cp_b, v_fl_load_b, v_fl_te_b = _compute_flap_velocities(
                ctrl_rig,
                lattice,
                rings,
                rate_use,
                self.a_b,
                self.b_b,
                self.cp_b,
                self.te_l_phys_b,
                self.te_r_phys_b,
                self.a_kink_b,
                self.b_kink_b,
            )
            v_fl_cp = _rotate_vecs(v_fl_cp_b, R, pitch_only)
            v_fl_load = _rotate_vecs(v_fl_load_b, R, pitch_only)
            v_fl_te = _rotate_vecs(v_fl_te_b, R, pitch_only)
        else:
            v_fl_cp = None
            v_fl_load = None
            v_fl_te = None

        t_now = float(self.time[step])
        gust_fn = self.gust_fn
        if gust_fn is None:
            gust_ref_step = np.zeros(3, dtype=np.float64)
            shift_conv = float(step) * self.wake_conv
            base_l = te_left - shift_conv
            base_r = te_right - shift_conv

            if step > 0:
                row_idx = step - 1
                self.wake_corners_earth[row_idx, :, 0, :] = base_l
                self.wake_corners_earth[row_idx, :, 1, :] = base_r
                self.wake_corners_earth[row_idx, :, 2, :] = self.prev_base_r
                self.wake_corners_earth[row_idx, :, 3, :] = self.prev_base_l
                self.wake_gamma_buf[row_idx] = self.prev_gamma[self.te_idx]
                self.wake_corners_cur[:step] = self.wake_corners_earth[:step] + shift_conv[None, :, None, :]
        else:
            assert self.wake_l_nodes is not None and self.wake_r_nodes is not None
            gust_ref_step = _eval_gust(gust_fn, x_ref[None, :], t_now)[0]
            if step == 0:
                self.wake_l_nodes[0] = te_left
                self.wake_r_nodes[0] = te_right
            else:
                _advect_wake_nodes_rk4(
                    self.wake_l_nodes,
                    self.wake_r_nodes,
                    step,
                    self.v_base_wake,
                    gust_fn,
                    float(self.time[step - 1]),
                    self.dt,
                    ground,
                )
                self.wake_l_nodes[step] = te_left
                self.wake_r_nodes[step] = te_right
                self.wake_corners_cur[:step, :, 0, :] = self.wake_l_nodes[1 : step + 1]
                self.wake_corners_cur[:step, :, 1, :] = self.wake_r_nodes[1 : step + 1]
                self.wake_corners_cur[:step, :, 2, :] = self.wake_r_nodes[:step]
                self.wake_corners_cur[:step, :, 3, :] = self.wake_l_nodes[:step]
                self.wake_gamma_buf[step - 1] = self.prev_gamma[self.te_idx]
            base_l = None
            base_r = None

        if ground is not None:
            pose_key = (xn, yn, zn, roll_n, pitch_n, yaw_n)
            pose_changed = (self.prev_pose_ge is None) or (pose_key != self.prev_pose_ge) or ctrl_changed
            if pose_changed:
                img_corners = ground.reflect_points(corners)
                img_sn = None if side_nodes is None else ground.reflect_points(side_nodes)
                A_img = ring_normal_influence(
                    cp,
                    normal_bc,
                    img_corners,
                    self.rc_ring,
                    side_nodes=img_sn,
                    side_counts=rings.side_counts,
                    backend=backend,
                )
                self.lu_Abb = sl.lu_factor(self.A_bb_real - A_img)
                self.prev_pose_ge = pose_key

        v_surf_cp = v_trans[None, :] + cross3(omega_g[None, :], cp - x_ref[None, :])
        if v_fl_cp is not None:
            v_surf_cp = v_surf_cp + v_fl_cp
        v_onset_cp = self.V_0[None, :] if gust_fn is None else self.V_0[None, :] + _eval_gust(gust_fn, cp, t_now)
        rhs = np.sum((v_surf_cp - v_onset_cp) * normal_bc, axis=1)

        if step > 0:
            if self.A_wake_stat is not None:
                age = step - 1
                wc_age = self.wake_corners_cur[0]
                blk = ring_normal_influence(cp, normal_bc, wc_age, self.rc_wake, backend=backend)
                if ground is not None:
                    blk -= ring_normal_influence(
                        cp, normal_bc, ground.reflect_points(wc_age), self.rc_wake, backend=backend,
                    )
                self.A_wake_stat[age] = blk
                gamma_by_age = self.wake_gamma_buf[:step][::-1]
                rhs -= np.einsum("rms,rs->m", self.A_wake_stat[:step], gamma_by_age)
            else:
                rhs -= _wake_normal_rhs(
                    cp,
                    normal_bc,
                    self.wake_corners_cur,
                    self.wake_gamma_buf,
                    step,
                    self.rc_wake,
                    ground=ground,
                    backend=backend,
                )

        gamma_n = sl.lu_solve(self.lu_Abb, rhs)

        if R is None:
            arm_all = self.arm_b
            arm_u = self.arm_u_b
            seg_vec = self.seg_vec_b
            seg_normal = self.seg_normal_b
            arm_te = self.arm_te_b
            l_te = self.l_te_b
            area_vec = self.area_vec_b
            arm_ring = self.arm_ring_b
            area_moment_vec = self.area_moment_vec_b
        else:
            arm_all = _rotate_vecs(self.arm_b, R, pitch_only)
            arm_u = arm_all[self.load_unique_idx]
            seg_vec = _rotate_vecs(self.seg_vec_b, R, pitch_only)
            seg_normal = _rotate_vecs(self.seg_normal_b, R, pitch_only)
            arm_te = _rotate_vecs(self.arm_te_b, R, pitch_only)
            l_te = _rotate_vecs(self.l_te_b, R, pitch_only)
            area_vec = _rotate_vecs(self.area_vec_b, R, pitch_only)
            arm_ring = _rotate_vecs(self.arm_ring_b, R, pitch_only)
            area_moment_vec = (
                None if self.area_moment_vec_b is None else _rotate_vecs(self.area_moment_vec_b, R, pitch_only)
            )
        P_load_u = x_ref[None, :] + arm_u
        p_te = x_ref[None, :] + arm_te

        if self.V_bound_load_b is not None:
            v_real_b = self.V_bound_load_b.swapaxes(1, 2) @ gamma_n
            v_ind_u = _rotate_vecs(v_real_b, R, pitch_only)
            if ground is not None:
                if self.is_stationary and self.cache_bound_load:
                    if self.V_bound_img_load_stat is None:
                        img_c = ground.reflect_points(corners)
                        img_s = None if side_nodes is None else ground.reflect_points(side_nodes)
                        self.V_bound_img_load_stat = ring_velocity(
                            P_load_u,
                            img_c,
                            self.rc_load_u_ring,
                            side_nodes=img_s,
                            side_counts=rings.side_counts,
                            backend=backend,
                        )
                    v_ind_u = v_ind_u - (self.V_bound_img_load_stat.swapaxes(1, 2) @ gamma_n)
                else:
                    v_ind_u = v_ind_u + _bound_velocity_at(
                        P_load_u,
                        corners,
                        side_nodes,
                        rings.side_counts,
                        gamma_n,
                        self.rc_load_u_ring,
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
                self.rc_load_u_ring,
                ground=ground,
                backend=backend,
                image_only=False,
            )

        if step > 0:
            if self.V_wake_load_stat is not None:
                age = step - 1
                wc_age = self.wake_corners_cur[0]
                v_row = ring_velocity(P_load_u, wc_age, self.rc_load_u_wake, backend=backend)
                if ground is not None:
                    v_row -= ring_velocity(
                        P_load_u, ground.reflect_points(wc_age), self.rc_load_u_wake, backend=backend,
                    )
                self.V_wake_load_stat[age] = v_row
                gamma_by_age = self.wake_gamma_buf[:step][::-1]
                v_ind_u = v_ind_u + np.einsum("rmsi,rs->mi", self.V_wake_load_stat[:step], gamma_by_age)
            else:
                v_ind_u = v_ind_u + _wake_velocity_at(
                    P_load_u,
                    self.wake_corners_cur,
                    self.wake_gamma_buf,
                    step,
                    self.rc_load_u_wake,
                    ground=ground,
                    backend=backend,
                )

        v_ind_load = v_ind_u[self.load_inv_idx]

        v_surf_load = v_trans[None, :] + cross3(omega_g[None, :], arm_all)
        if v_fl_load is not None:
            v_surf_load = v_surf_load + v_fl_load
        if gust_fn is None:
            v_onset_load = self.V_0[None, :]
        else:
            v_onset_load = self.V_0[None, :] + _eval_gust(gust_fn, P_load_u, t_now)[self.load_inv_idx]
        V_tot = v_onset_load - v_surf_load + v_ind_load
        V_eff = V_tot - np.sum(V_tot * seg_normal, axis=1, keepdims=True) * seg_normal

        g_hs = _rings_to_horseshoe(gamma_n, rings.n_chord)
        g_leg = leg_strength(lattice, g_hs)
        g_all = np.concatenate([g_hs] + [g_leg] * self.n_leg_sets)

        F_seg = (self.rho * g_all * self.seg_scale)[:, None] * cross3(V_eff, seg_vec)
        F_tot = np.sum(F_seg, axis=0)
        M_tot = np.sum(cross3(arm_all, F_seg), axis=0)
        s_lift = np.zeros(self.n_strips, dtype=np.float64)
        np.add.at(s_lift, self.eval_strips, F_seg @ self.L_0)

        if step >= 2:
            dG_dt = (3.0 * gamma_n - 4.0 * self.prev_gamma + self.prev2_gamma) / (2.0 * self.dt)
        elif step == 1:
            dG_dt = (gamma_n - self.prev_gamma) / self.dt
        else:
            dG_dt = None

        if dG_dt is not None:
            coef_am = self.rho * dG_dt
            F_am = coef_am[:, None] * area_vec
            F_tot = F_tot + np.sum(F_am, axis=0)
            if area_moment_vec is not None:
                M_tot = M_tot + np.sum(coef_am[:, None] * area_moment_vec, axis=0)
            else:
                M_tot = M_tot + np.sum(cross3(arm_ring, F_am), axis=0)
            np.add.at(s_lift, rings.panel_strip, F_am @ self.L_0)

        if self.te_correction and step >= 1:
            if self.V_bound_te_b is not None:
                v_te_real = self.V_bound_te_b.swapaxes(1, 2) @ gamma_n
                v_ind_te = _rotate_vecs(v_te_real, R, pitch_only)
                if ground is not None:
                    if self.is_stationary:
                        if self.V_bound_img_te_stat is None:
                            img_c = ground.reflect_points(corners)
                            img_s = None if side_nodes is None else ground.reflect_points(side_nodes)
                            self.V_bound_img_te_stat = ring_velocity(
                                p_te,
                                img_c,
                                self.rc_te_ring,
                                side_nodes=img_s,
                                side_counts=rings.side_counts,
                                backend=backend,
                            )
                        v_ind_te = v_ind_te - (self.V_bound_img_te_stat.swapaxes(1, 2) @ gamma_n)
                    else:
                        v_ind_te = v_ind_te + _bound_velocity_at(
                            p_te,
                            corners,
                            side_nodes,
                            rings.side_counts,
                            gamma_n,
                            self.rc_te_ring,
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
                    self.rc_te_ring,
                    ground=ground,
                    backend=backend,
                    image_only=False,
                )

            if self.V_wake_te_stat is not None:
                age = step - 1
                wc_age = self.wake_corners_cur[0]
                v_row_te = ring_velocity(p_te, wc_age, self.rc_te_wake, backend=backend)
                if ground is not None:
                    v_row_te -= ring_velocity(
                        p_te, ground.reflect_points(wc_age), self.rc_te_wake, backend=backend,
                    )
                self.V_wake_te_stat[age] = v_row_te
                gamma_by_age = self.wake_gamma_buf[:step][::-1]
                v_ind_te = v_ind_te + np.einsum("rmsi,rs->mi", self.V_wake_te_stat[:step], gamma_by_age)
            else:
                v_ind_te = v_ind_te + _wake_velocity_at(
                    p_te,
                    self.wake_corners_cur,
                    self.wake_gamma_buf,
                    step,
                    self.rc_te_wake,
                    ground=ground,
                    backend=backend,
                )

            v_surf_te = v_trans[None, :] + cross3(omega_g[None, :], arm_te)
            if v_fl_te is not None:
                v_surf_te = v_surf_te + v_fl_te
            v_onset_te = self.V_0[None, :] if gust_fn is None else self.V_0[None, :] + _eval_gust(gust_fn, p_te, t_now)
            V_te = v_onset_te - v_surf_te + v_ind_te
            dg_te = self.f_te * (self.wake_gamma_buf[step - 1] - gamma_n[self.te_idx])
            F_te = (self.rho * dg_te)[:, None] * cross3(V_te, l_te)
            F_tot = F_tot + np.sum(F_te, axis=0)
            M_tot = M_tot + np.sum(cross3(arm_te, F_te), axis=0)
            s_lift = s_lift + (F_te @ self.L_0)

        CL_step = float((F_tot @ self.L_0) / self.qS)
        CD_step = float((F_tot @ self.d_0) / self.qS)
        CY_step = float((F_tot @ self.S_0) / self.qS)
        Cl_step = float(-M_tot[0] / (self.qS * self.b_ref))
        Cm_step = float(M_tot[1] / (self.qS * self.c_ref))
        Cn_step = float(-M_tot[2] / (self.qS * self.b_ref))
        strip_cl_step = s_lift / (self.q_inf * self.strip_area)

        self.prev_base_l = base_l
        self.prev_base_r = base_r
        self.prev2_gamma = self.prev_gamma
        self.prev_gamma = gamma_n
        self.step = step + 1

        return _StepLoads(
            CL=CL_step,
            CD=CD_step,
            CY=CY_step,
            Cl=Cl_step,
            Cm=Cm_step,
            Cn=Cn_step,
            strip_cl=strip_cl_step,
            gust_ref=gust_ref_step,
            F_tot=F_tot,
            M_tot=M_tot,
        )

    def _advance_gpu(self, **kwargs: Any) -> _StepLoads:
        """Advance one step on the GPU; a release of the GPU data waits until the step ends."""
        assert self._gpu_state is not None
        with self._gpu_state.step_scope():
            return self._advance_gpu_step(**kwargs)

    def _advance_gpu_step(
        self,
        *,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        x_dot: float,
        y_dot: float,
        z_dot: float,
        roll_dot: float,
        pitch_dot: float,
        yaw_dot: float,
        dd_step: np.ndarray | None = None,
        rate_step: np.ndarray | None = None,
    ) -> _StepLoads:
        """Advance step ``self.step`` on the GPU using Warp ring kernels and PyTorch LU."""
        import torch

        from ventorum.gpu import engine
        from ventorum.gpu import ring_kernels as RK

        assert self._gpu_state is not None
        st = self._gpu_state.tensors
        dev = engine._dev()
        prec = self.precision

        step = self.step
        xn, yn, zn = float(x), float(y), float(z)
        roll_n, pitch_n, yaw_n = float(roll), float(pitch), float(yaw)
        xd, yd, zd = float(x_dot), float(y_dot), float(z_dot)
        roll_d, pitch_d, yaw_d = float(roll_dot), float(pitch_dot), float(yaw_dot)

        rp = self.rp
        e_x, e_y, e_z = self.e_x, self.e_y, self.e_z
        rings = self.rings
        lattice = self.lattice
        ground = self.ground
        ctrl_rig = self.ctrl_rig

        x_ref = rp + xn * e_x + yn * e_y + zn * e_z
        v_trans = xd * e_x + yd * e_y + zd * e_z
        omega_g = _angular_velocity_321(pitch_n, yaw_n, roll_d, pitch_d, yaw_d)

        if roll_n == 0.0 and pitch_n == 0.0 and yaw_n == 0.0:
            R: np.ndarray | None = None
            R_g: torch.Tensor | None = None
        else:
            R = _rotation_matrix_321(roll_n, pitch_n, yaw_n)
            R_g = engine._f64(R)

        x_ref_g = engine._f64(x_ref)
        v_trans_g = engine._f64(v_trans)
        omega_g_g = engine._f64(omega_g)
        sc_g = st["side_counts"]

        ctrl_changed = False
        if self.controls_vary and ctrl_rig is not None:
            dd_use = ctrl_rig.dd_edges[step] if dd_step is None else dd_step
            if self.prev_dd is None or not np.array_equal(dd_use, self.prev_dd):
                (
                    self.a_b,
                    self.b_b,
                    self.cp_b,
                    self.normal_bc_b,
                    self.corners_b,
                    self.te_left_b,
                    self.te_right_b,
                    self.te_l_phys_b,
                    self.te_r_phys_b,
                    self.side_nodes_b,
                    self.a_kink_b,
                    self.b_kink_b,
                    self.leg_normal_b,
                ) = _deform_body_geometry(ctrl_rig, lattice, rings, dd_use)
                (
                    self.P_load_u_b,
                    self.arm_b,
                    self.arm_u_b,
                    self.seg_vec_b,
                    self.seg_normal_b,
                    self.load_unique_idx,
                    self.load_inv_idx,
                    self.rc_load_u_ring,
                    self.rc_load_u_wake,
                    self.p_te_b,
                    self.l_te_b,
                    self.arm_te_b,
                    self.area_vec_b,
                    self.arm_ring_b,
                    self.area_moment_vec_b,
                ) = _build_body_load_geometry(
                    lattice,
                    rings,
                    rp,
                    self.a_b,
                    self.b_b,
                    self.corners_b,
                    self.te_l_phys_b,
                    self.te_r_phys_b,
                    self.side_nodes_b,
                    self.a_kink_b,
                    self.b_kink_b,
                    self.leg_normal_b,
                    self.rc_load_ring,
                    self.rc_load_wake,
                )
                self._upload_body_geometry_gpu()
                _update_bound_influence_blocks_gpu(
                    st["A_bb_real"],
                    st["moving_col_panels"],
                    st["moving_row_panels"],
                    st["fixed_col_panels"],
                    st["cp_b"],
                    st["normal_bc_b"],
                    st["corners_b"],
                    st["side_nodes_b"],
                    sc_g,
                    st["rc_ring"],
                    precision=prec,
                )
                if ground is None:
                    st["lu_Abb"], st["piv_Abb"] = torch.linalg.lu_factor(st["A_bb_real"])
                self.prev_dd = dd_use.copy()
                self.prev_pose_ge = None
                ctrl_changed = True

        cp_g, normal_bc_g, corners_g, te_left_g, te_right_g, side_nodes_g = _move_wing_geometry_gpu(
            st["cp_b"],
            st["normal_bc_b"],
            st["corners_b"],
            st["te_left_b"],
            st["te_right_b"],
            st["side_nodes_b"],
            st["rp"],
            x_ref_g,
            R_g,
        )

        if ctrl_rig is not None:
            rate_use = ctrl_rig.rate_edges[step] if rate_step is None else rate_step
            v_fl_cp_b, v_fl_load_b, v_fl_te_b = _compute_flap_velocities(
                ctrl_rig,
                lattice,
                rings,
                rate_use,
                self.a_b,
                self.b_b,
                self.cp_b,
                self.te_l_phys_b,
                self.te_r_phys_b,
                self.a_kink_b,
                self.b_kink_b,
            )
            v_fl_cp_g: torch.Tensor | None = _rotate_vecs_gpu(engine._f64(v_fl_cp_b), R_g)
            v_fl_load_g: torch.Tensor | None = _rotate_vecs_gpu(engine._f64(v_fl_load_b), R_g)
            v_fl_te_g: torch.Tensor | None = _rotate_vecs_gpu(engine._f64(v_fl_te_b), R_g)
        else:
            v_fl_cp_g = None
            v_fl_load_g = None
            v_fl_te_g = None

        t_now = float(self.time[step])
        gust_fn = self.gust_fn
        if gust_fn is None:
            gust_ref_step = np.zeros(3, dtype=np.float64)
            shift_conv_g = float(step) * st["wake_conv"]
            base_l_g: torch.Tensor | None = te_left_g - shift_conv_g
            base_r_g: torch.Tensor | None = te_right_g - shift_conv_g

            if step > 0:
                row_idx = step - 1
                wce = st["wake_corners_earth"]
                wce[row_idx, :, 0, :] = base_l_g
                wce[row_idx, :, 1, :] = base_r_g
                wce[row_idx, :, 2, :] = st["prev_base_r"]
                wce[row_idx, :, 3, :] = st["prev_base_l"]
                st["wake_gamma_buf"][row_idx] = st["prev_gamma"][st["te_idx"]]
                st["wake_corners_cur"][:step] = wce[:step] + shift_conv_g[None, :, None, :]
        else:
            wl_g = st["wake_l_nodes"]
            wr_g = st["wake_r_nodes"]
            gust_ref_step = _eval_gust(gust_fn, x_ref[None, :], t_now)[0]
            if step == 0:
                wl_g[0] = te_left_g
                wr_g[0] = te_right_g
            else:
                _advect_wake_nodes_rk4_gpu(
                    wl_g,
                    wr_g,
                    step,
                    st["v_base_wake"],
                    gust_fn,
                    float(self.time[step - 1]),
                    self.dt,
                    st["g_norm"],
                )
                wl_g[step] = te_left_g
                wr_g[step] = te_right_g
                wcc = st["wake_corners_cur"]
                wcc[:step, :, 0, :] = wl_g[1 : step + 1]
                wcc[:step, :, 1, :] = wr_g[1 : step + 1]
                wcc[:step, :, 2, :] = wr_g[:step]
                wcc[:step, :, 3, :] = wl_g[:step]
                st["wake_gamma_buf"][step - 1] = st["prev_gamma"][st["te_idx"]]
            base_l_g = None
            base_r_g = None

        if ground is not None:
            pose_key = (xn, yn, zn, roll_n, pitch_n, yaw_n)
            pose_changed = (self.prev_pose_ge is None) or (pose_key != self.prev_pose_ge) or ctrl_changed
            if pose_changed:
                A_img_signed = RK.ring_normal_influence_gpu(
                    cp_g,
                    normal_bc_g,
                    corners_g,
                    st["rc_ring"],
                    side_nodes=side_nodes_g,
                    side_counts=sc_g,
                    ground=ground,
                    img_mode=2,
                    precision=prec,
                    out_tensor=True,
                ).to(torch.float64)
                st["lu_Abb"], st["piv_Abb"] = torch.linalg.lu_factor(st["A_bb_real"] + A_img_signed)
                self.prev_pose_ge = pose_key

        v_surf_cp_g = v_trans_g[None, :] + torch.linalg.cross(
            omega_g_g[None, :].expand_as(cp_g), cp_g - x_ref_g[None, :], dim=-1,
        )
        if v_fl_cp_g is not None:
            v_surf_cp_g = v_surf_cp_g + v_fl_cp_g
        if gust_fn is None:
            v_onset_cp_g = st["V_0"][None, :]
        else:
            v_onset_cp_g = st["V_0"][None, :] + engine._f64(_eval_gust(gust_fn, cp_g.cpu().numpy(), t_now))
        rhs_g = torch.sum((v_surf_cp_g - v_onset_cp_g) * normal_bc_g, dim=1)

        if step > 0:
            A_wake_stat_g = st["A_wake_stat"]
            if A_wake_stat_g is not None:
                age = step - 1
                wc_age_g = st["wake_corners_cur"][0]
                blk = RK.ring_normal_influence_gpu(
                    cp_g,
                    normal_bc_g,
                    wc_age_g,
                    st["rc_wake"],
                    ground=ground,
                    precision=prec,
                    out_tensor=True,
                )
                A_wake_stat_g[age] = blk
                gamma_by_age_g = torch.flip(st["wake_gamma_buf"][:step], dims=(0,))
                rhs_g = rhs_g - torch.einsum("rms,rs->m", A_wake_stat_g[:step].to(torch.float64), gamma_by_age_g)
            else:
                rhs_g = rhs_g - _wake_normal_rhs_gpu(
                    cp_g,
                    normal_bc_g,
                    st["wake_corners_cur"],
                    st["wake_gamma_buf"],
                    step,
                    st["rc_wake"],
                    ground=ground,
                    precision=prec,
                )

        gamma_n_g = torch.linalg.lu_solve(
            st["lu_Abb"], st["piv_Abb"], rhs_g.unsqueeze(-1),
        ).squeeze(-1)

        if R_g is None:
            arm_all_g = st["arm_b"]
            arm_u_g = st["arm_u_b"]
            seg_vec_g = st["seg_vec_b"]
            seg_normal_g = st["seg_normal_b"]
            arm_te_g = st["arm_te_b"]
            l_te_g = st["l_te_b"]
            area_vec_g = st["area_vec_b"]
            arm_ring_g = st["arm_ring_b"]
            area_moment_vec_g = st["area_moment_vec_b"]
        else:
            arm_all_g = _rotate_vecs_gpu(st["arm_b"], R_g)
            arm_u_g = arm_all_g[st["load_unique_idx"]]
            seg_vec_g = _rotate_vecs_gpu(st["seg_vec_b"], R_g)
            seg_normal_g = _rotate_vecs_gpu(st["seg_normal_b"], R_g)
            arm_te_g = _rotate_vecs_gpu(st["arm_te_b"], R_g)
            l_te_g = _rotate_vecs_gpu(st["l_te_b"], R_g)
            area_vec_g = _rotate_vecs_gpu(st["area_vec_b"], R_g)
            arm_ring_g = _rotate_vecs_gpu(st["arm_ring_b"], R_g)
            area_moment_vec_g = (
                None
                if st["area_moment_vec_b"] is None
                else _rotate_vecs_gpu(st["area_moment_vec_b"], R_g)
            )
        P_load_u_g = x_ref_g[None, :] + arm_u_g
        p_te_g = x_ref_g[None, :] + arm_te_g

        V_bound_load_b_g = st["V_bound_load_b"]
        if V_bound_load_b_g is not None:
            v_real_b_g = torch.einsum("mni,n->mi", V_bound_load_b_g.to(torch.float64), gamma_n_g)
            v_ind_u_g = _rotate_vecs_gpu(v_real_b_g, R_g)
            if ground is not None:
                if self.is_stationary and self.cache_bound_load:
                    if st["V_bound_img_load_stat"] is None:
                        st["V_bound_img_load_stat"] = RK.ring_velocity_gpu(
                            P_load_u_g,
                            corners_g,
                            st["rc_load_u_ring"],
                            side_nodes=side_nodes_g,
                            side_counts=sc_g,
                            ground=ground,
                            img_mode=2,
                            precision=prec,
                            out_tensor=True,
                        )
                    v_ind_u_g = v_ind_u_g + torch.einsum(
                        "mni,n->mi", st["V_bound_img_load_stat"].to(torch.float64), gamma_n_g,
                    )
                else:
                    v_ind_u_g = v_ind_u_g + _bound_velocity_at_gpu(
                        P_load_u_g,
                        corners_g,
                        side_nodes_g,
                        sc_g,
                        gamma_n_g,
                        st["rc_load_u_ring"],
                        ground=ground,
                        precision=prec,
                        image_only=True,
                    )
        else:
            v_ind_u_g = _bound_velocity_at_gpu(
                P_load_u_g,
                corners_g,
                side_nodes_g,
                sc_g,
                gamma_n_g,
                st["rc_load_u_ring"],
                ground=ground,
                precision=prec,
                image_only=False,
            )

        if step > 0:
            V_wake_load_stat_g = st["V_wake_load_stat"]
            if V_wake_load_stat_g is not None:
                age = step - 1
                wc_age_g = st["wake_corners_cur"][0]
                v_row = RK.ring_velocity_gpu(
                    P_load_u_g,
                    wc_age_g,
                    st["rc_load_u_wake"],
                    ground=ground,
                    precision=prec,
                    out_tensor=True,
                )
                V_wake_load_stat_g[age] = v_row
                gamma_by_age_g = torch.flip(st["wake_gamma_buf"][:step], dims=(0,))
                v_ind_u_g = v_ind_u_g + torch.einsum(
                    "rmsi,rs->mi", V_wake_load_stat_g[:step].to(torch.float64), gamma_by_age_g,
                )
            else:
                v_ind_u_g = v_ind_u_g + _wake_velocity_at_gpu(
                    P_load_u_g,
                    st["wake_corners_cur"],
                    st["wake_gamma_buf"],
                    step,
                    st["rc_load_u_wake"],
                    ground=ground,
                    precision=prec,
                )

        v_ind_load_g = v_ind_u_g[st["load_inv_idx"]]

        v_surf_load_g = v_trans_g[None, :] + torch.linalg.cross(
            omega_g_g[None, :].expand_as(arm_all_g), arm_all_g, dim=-1,
        )
        if v_fl_load_g is not None:
            v_surf_load_g = v_surf_load_g + v_fl_load_g
        if gust_fn is None:
            v_onset_load_g = st["V_0"][None, :]
        else:
            vg_load_g = engine._f64(_eval_gust(gust_fn, P_load_u_g.cpu().numpy(), t_now))
            v_onset_load_g = st["V_0"][None, :] + vg_load_g[st["load_inv_idx"]]
        V_tot_g = v_onset_load_g - v_surf_load_g + v_ind_load_g
        V_eff_g = V_tot_g - torch.sum(V_tot_g * seg_normal_g, dim=1, keepdim=True) * seg_normal_g

        g_hs_g = _rings_to_horseshoe_gpu(gamma_n_g, rings.n_chord)
        chain_levels = st["chain_levels"]
        if chain_levels:
            g_leg_g = g_hs_g.clone()
            for dst_g, src_g in chain_levels:
                g_leg_g[dst_g] = g_leg_g[dst_g] + g_leg_g[src_g]
        else:
            g_leg_g = g_hs_g
        g_all_g = torch.cat([g_hs_g] + [g_leg_g] * self.n_leg_sets, dim=0)

        F_seg_g = (self.rho * g_all_g * st["seg_scale"])[:, None] * torch.linalg.cross(V_eff_g, seg_vec_g, dim=-1)
        F_tot_g = torch.sum(F_seg_g, dim=0)
        M_tot_g = torch.sum(torch.linalg.cross(arm_all_g, F_seg_g, dim=-1), dim=0)
        s_lift_g = torch.zeros(self.n_strips, dtype=torch.float64, device=dev)
        s_lift_g.index_add_(0, st["eval_strips"], F_seg_g @ st["L_0"])

        if step >= 2:
            dG_dt_g: torch.Tensor | None = (
                3.0 * gamma_n_g - 4.0 * st["prev_gamma"] + st["prev2_gamma"]
            ) / (2.0 * self.dt)
        elif step == 1:
            dG_dt_g = (gamma_n_g - st["prev_gamma"]) / self.dt
        else:
            dG_dt_g = None

        if dG_dt_g is not None:
            coef_am_g = self.rho * dG_dt_g
            F_am_g = coef_am_g[:, None] * area_vec_g
            F_tot_g = F_tot_g + torch.sum(F_am_g, dim=0)
            if area_moment_vec_g is not None:
                M_tot_g = M_tot_g + torch.sum(coef_am_g[:, None] * area_moment_vec_g, dim=0)
            else:
                M_tot_g = M_tot_g + torch.sum(torch.linalg.cross(arm_ring_g, F_am_g, dim=-1), dim=0)
            s_lift_g.index_add_(0, st["panel_strip"], F_am_g @ st["L_0"])

        if self.te_correction and step >= 1:
            V_bound_te_b_g = st["V_bound_te_b"]
            if V_bound_te_b_g is not None:
                v_te_real_g = torch.einsum("mni,n->mi", V_bound_te_b_g.to(torch.float64), gamma_n_g)
                v_ind_te_g = _rotate_vecs_gpu(v_te_real_g, R_g)
                if ground is not None:
                    if self.is_stationary:
                        if st["V_bound_img_te_stat"] is None:
                            st["V_bound_img_te_stat"] = RK.ring_velocity_gpu(
                                p_te_g,
                                corners_g,
                                st["rc_te_ring"],
                                side_nodes=side_nodes_g,
                                side_counts=sc_g,
                                ground=ground,
                                img_mode=2,
                                precision=prec,
                                out_tensor=True,
                            )
                        v_ind_te_g = v_ind_te_g + torch.einsum(
                            "mni,n->mi", st["V_bound_img_te_stat"].to(torch.float64), gamma_n_g,
                        )
                    else:
                        v_ind_te_g = v_ind_te_g + _bound_velocity_at_gpu(
                            p_te_g,
                            corners_g,
                            side_nodes_g,
                            sc_g,
                            gamma_n_g,
                            st["rc_te_ring"],
                            ground=ground,
                            precision=prec,
                            image_only=True,
                        )
            else:
                v_ind_te_g = _bound_velocity_at_gpu(
                    p_te_g,
                    corners_g,
                    side_nodes_g,
                    sc_g,
                    gamma_n_g,
                    st["rc_te_ring"],
                    ground=ground,
                    precision=prec,
                    image_only=False,
                )

            V_wake_te_stat_g = st["V_wake_te_stat"]
            if V_wake_te_stat_g is not None:
                age = step - 1
                wc_age_g = st["wake_corners_cur"][0]
                v_row_te = RK.ring_velocity_gpu(
                    p_te_g,
                    wc_age_g,
                    st["rc_te_wake"],
                    ground=ground,
                    precision=prec,
                    out_tensor=True,
                )
                V_wake_te_stat_g[age] = v_row_te
                gamma_by_age_g = torch.flip(st["wake_gamma_buf"][:step], dims=(0,))
                v_ind_te_g = v_ind_te_g + torch.einsum(
                    "rmsi,rs->mi", V_wake_te_stat_g[:step].to(torch.float64), gamma_by_age_g,
                )
            else:
                v_ind_te_g = v_ind_te_g + _wake_velocity_at_gpu(
                    p_te_g,
                    st["wake_corners_cur"],
                    st["wake_gamma_buf"],
                    step,
                    st["rc_te_wake"],
                    ground=ground,
                    precision=prec,
                )

            v_surf_te_g = v_trans_g[None, :] + torch.linalg.cross(
                omega_g_g[None, :].expand_as(arm_te_g), arm_te_g, dim=-1,
            )
            if v_fl_te_g is not None:
                v_surf_te_g = v_surf_te_g + v_fl_te_g
            if gust_fn is None:
                v_onset_te_g = st["V_0"][None, :]
            else:
                v_onset_te_g = st["V_0"][None, :] + engine._f64(_eval_gust(gust_fn, p_te_g.cpu().numpy(), t_now))
            V_te_g = v_onset_te_g - v_surf_te_g + v_ind_te_g
            dg_te_g = st["f_te"] * (st["wake_gamma_buf"][step - 1] - gamma_n_g[st["te_idx"]])
            F_te_g = (self.rho * dg_te_g)[:, None] * torch.linalg.cross(V_te_g, l_te_g, dim=-1)
            F_tot_g = F_tot_g + torch.sum(F_te_g, dim=0)
            M_tot_g = M_tot_g + torch.sum(torch.linalg.cross(arm_te_g, F_te_g, dim=-1), dim=0)
            s_lift_g = s_lift_g + (F_te_g @ st["L_0"])

        # Copy only the step loads back to the host.
        loads_host = torch.cat([F_tot_g, M_tot_g, s_lift_g], dim=0).cpu().numpy()
        F_tot = loads_host[:3].copy()
        M_tot = loads_host[3:6].copy()
        s_lift = loads_host[6:]

        CL_step = float((F_tot @ self.L_0) / self.qS)
        CD_step = float((F_tot @ self.d_0) / self.qS)
        CY_step = float((F_tot @ self.S_0) / self.qS)
        Cl_step = float(-M_tot[0] / (self.qS * self.b_ref))
        Cm_step = float(M_tot[1] / (self.qS * self.c_ref))
        Cn_step = float(-M_tot[2] / (self.qS * self.b_ref))
        strip_cl_step = s_lift / (self.q_inf * self.strip_area)

        st["prev_base_l"] = base_l_g
        st["prev_base_r"] = base_r_g
        st["prev2_gamma"] = st.get("prev_gamma")
        st["prev_gamma"] = gamma_n_g
        self.step = step + 1

        return _StepLoads(
            CL=CL_step,
            CD=CD_step,
            CY=CY_step,
            Cl=Cl_step,
            Cm=Cm_step,
            Cn=Cn_step,
            strip_cl=strip_cl_step,
            gust_ref=gust_ref_step,
            F_tot=F_tot,
            M_tot=M_tot,
        )

    def gamma_bound(self) -> np.ndarray:
        """Return bound ring circulations of the last completed step [m^2/s], shape ``(n_panels,)``."""
        if self._gpu_state is not None:
            pg = self._gpu_state.tensors.get("prev_gamma")
            if pg is None:
                return np.zeros(self.n_panels, dtype=np.float64)
            return pg.cpu().numpy().copy()
        if self.prev_gamma is None:
            return np.zeros(self.n_panels, dtype=np.float64)
        return self.prev_gamma.copy()

    def gamma_wake(self, n_marched: int | None = None) -> np.ndarray:
        """Return wake ring circulations [m^2/s], shape ``(n_marched, n_strips)`` (newest row first)."""
        nw = max(self.step - 1, 0) if n_marched is None else int(n_marched)
        if self._gpu_state is not None:
            if nw == 0:
                return np.empty((0, self.n_strips), dtype=np.float64)
            return self._gpu_state.tensors["wake_gamma_buf"][:nw].cpu().numpy()[::-1].copy()
        assert self.wake_gamma_buf is not None
        return self.wake_gamma_buf[:nw][::-1].copy()

    def wake_corners(self, n_marched: int | None = None) -> np.ndarray:
        """Return wake ring corners [m], shape ``(n_marched, n_strips, 4, 3)`` (newest row first)."""
        nw = max(self.step - 1, 0) if n_marched is None else int(n_marched)
        if self._gpu_state is not None:
            if nw == 0:
                return np.empty((0, self.n_strips, 4, 3), dtype=np.float64)
            return self._gpu_state.tensors["wake_corners_cur"][:nw].cpu().numpy()[::-1].copy()
        assert self.wake_corners_cur is not None
        return self.wake_corners_cur[:nw][::-1].copy()


def _solve_time_rings(
    lattice: VortexLattice,
    condition: FlightCondition,
    *,
    S_ref: float,
    b_ref: float,
    c_ref: float,
    ref_point: np.ndarray | None = None,
    z: Sequence[float] | np.ndarray | Callable[[float], float] | None = None,
    theta: Sequence[float] | np.ndarray | Callable[[float], float] | None = None,
    motion: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
    controls: Mapping[str, Sequence[float] | np.ndarray | Callable[[float], float]] | None = None,
    gust: Callable[[np.ndarray, float], np.ndarray] | None = None,
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
    control_increments: bool = False,
) -> _TimeDomainResponse:
    """March the prescribed-wake time-domain vortex-ring solver for *n_steps* steps.

    In free air with rigid-body motion only (and for stationary geometry in
    ground effect), the ``N x N`` bound influence matrix is LU-factorised once
    (``O(N^3)`` setup, ``O(N^2)`` solve per step). When a control surface moves
    (free air or ground), only the ``M_c`` columns of the moving flap and
    hinge-crossing rings and the ``M_r`` rows of the moved flap collocation
    points are recomputed in the real bound influence matrix
    (``O(N (M_r + M_c))`` kernel work per changing step), followed by
    ``O(N^3)`` refactorisation. In ground effect with wing or flap motion, the
    image bound influence matrix is recomputed (``O(N^2)`` kernel work) and the
    ``N x N`` system is refactorised (``O(N^3)``) at every changing step.

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
        Moment reference and rotation pivot point [m], shape ``(3,)``. None
        uses the origin.
    z : sequence of float, numpy.ndarray, callable or None, optional
        Prescribed heave displacement of ``ref_point`` [m] (positive up along
        ``e_z``), either an array of length ``n_steps + 1`` or a callable
        ``z(t)`` of time ``t`` [s]. Cannot be set together with
        ``motion['z']``.
    theta : sequence of float, numpy.ndarray, callable or None, optional
        Prescribed pitch angle about ``ref_point`` [rad] (positive nose up),
        either an array of length ``n_steps + 1`` or a callable ``theta(t)``
        of time ``t`` [s]. Cannot be set together with ``motion['pitch']``.
    motion : mapping of str to array or callable, or None, optional
        Prescribed six-axis rigid-body displacements of ``ref_point``:
        translations ``'x'``, ``'y'``, ``'z'`` [m] in mean-flight axes and
        rotations ``'roll'``, ``'pitch'``, ``'yaw'`` [rad] about ``ref_point``
        in the 3-2-1 sequence (AVL signs). Missing entries are zero.
    controls : mapping of str to array or callable, or None, optional
        Prescribed control-surface deflection histories [rad], keyed by
        ``ControlSurface.name``. Missing controls stay at their built lattice
        deflection.
    gust : callable or None, optional
        Optional gust velocity field ``gust(points, t) -> velocities`` where
        *points* has shape ``(n, 3)`` [m] in geometry axes, *t* is time [s],
        and the return array has shape ``(n, 3)`` [m/s] (air velocity added to
        the free stream). None applies no gust. On the GPU, a gust callable is
        evaluated on the host at every step and its velocities are copied to
        the device, so it adds a host cost per step.
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
        Time history of six-axis motion, control deflections and load
        coefficients, final bound and wake ring strengths and wake geometry,
        notes and mesh metadata.

    Raises
    ------
    InputError
        If *n_steps*, *dt*, *z*, *theta*, *motion*, *controls*, *lattice* or
        reference scalars are invalid, if the same motion axis is set twice,
        or if any node of the wing or of a deflected flap reaches or crosses
        *ground* at any step.
    """
    stepper = _TimeDomainStepper.from_prescribed(
        lattice,
        condition,
        S_ref=S_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        ref_point=ref_point,
        z=z,
        theta=theta,
        motion=motion,
        controls=controls,
        gust=gust,
        n_steps=n_steps,
        dt=dt,
        ground=ground,
        wake_dir=wake_dir,
        te_extension_fraction=te_extension_fraction,
        rc=rc,
        use_cross_surface_core=use_cross_surface_core,
        te_correction=te_correction,
        wake_alignment=wake_alignment,
        backend=backend,
        control_increments=control_increments,
    )

    n_steps_int = stepper.n_steps
    n_s = stepper.n_strips
    CL = np.empty(n_steps_int + 1, dtype=np.float64)
    CD = np.empty(n_steps_int + 1, dtype=np.float64)
    CY = np.empty(n_steps_int + 1, dtype=np.float64)
    Cl = np.empty(n_steps_int + 1, dtype=np.float64)
    Cm = np.empty(n_steps_int + 1, dtype=np.float64)
    Cn = np.empty(n_steps_int + 1, dtype=np.float64)
    strip_cl = np.empty((n_steps_int + 1, n_s), dtype=np.float64)
    gust_ref = np.zeros((n_steps_int + 1, 3), dtype=np.float64)

    try:
        for step in range(n_steps_int + 1):
            loads = stepper.advance_prescribed(step)
            CL[step] = loads.CL
            CD[step] = loads.CD
            CY[step] = loads.CY
            Cl[step] = loads.Cl
            Cm[step] = loads.Cm
            Cn[step] = loads.Cn
            strip_cl[step] = loads.strip_cl
            gust_ref[step] = loads.gust_ref

        gamma_bound = stepper.gamma_bound()
        gamma_wake = stepper.gamma_wake()
        wake_corners = stepper.wake_corners()
    finally:
        stepper.release_gpu()

    assert stepper.m_axes is not None
    x_arr, x_dot = stepper.m_axes["x"]
    y_arr, y_dot = stepper.m_axes["y"]
    z_arr, z_dot = stepper.m_axes["z"]
    roll_arr, roll_dot = stepper.m_axes["roll"]
    pitch_arr, pitch_dot = stepper.m_axes["pitch"]
    yaw_arr, yaw_dot = stepper.m_axes["yaw"]

    return _TimeDomainResponse(
        time=stepper.time,
        z=z_arr,
        theta=pitch_arr,
        z_dot=z_dot,
        theta_dot=pitch_dot,
        CL=CL,
        CD=CD,
        CY=CY,
        Cl=Cl,
        Cm=Cm,
        Cn=Cn,
        strip_cl=strip_cl,
        gamma_bound=gamma_bound,
        gamma_wake=gamma_wake,
        wake_corners=wake_corners,
        dt=stepper.dt,
        dt_default=stepper.dt_default,
        dx_w=stepper.dx_w,
        dx_te_mean=stepper.dx_te_mean,
        n_steps=n_steps_int,
        n_panels=stepper.n_panels,
        n_strips=n_s,
        n_chord=stepper.n_chord,
        te_correction=stepper.te_correction,
        c_ref=stepper.c_ref,
        b_ref=stepper.b_ref,
        S_ref=stepper.S_ref,
        x=x_arr,
        y=y_arr,
        roll=roll_arr,
        pitch=pitch_arr,
        yaw=yaw_arr,
        x_dot=x_dot,
        y_dot=y_dot,
        roll_dot=roll_dot,
        pitch_dot=pitch_dot,
        yaw_dot=yaw_dot,
        controls=stepper.ctrl_arrs,
        controls_dot=stepper.ctrl_dots,
        gust_velocity=gust_ref,
        notes=stepper.notes,
        device=stepper.device,
        precision=stepper.precision,
    )
