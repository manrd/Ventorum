# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Batched lattice pipelines on the GPU (lifting line and vortex lattice).

The functions here compute, for K flight conditions on one lattice, the
quantities that the CPU solvers compute: the circulation, the induced
velocities, the Trefftz-plane normal wash and all the loads (forces, moments,
section coefficients). The equations are those of
:mod:`ventorum.solvers.core` and :mod:`ventorum.aero.loads`; only the result
objects are made on the host (see :mod:`ventorum.gpu.pipeline`).

Data on the GPU
---------------
:class:`DeviceLattice` holds the lattice arrays on the GPU, once per lattice
and precision (it is kept in ``lattice.geom_cache``, as the other
lattice-only data). The part of the system from the bound vortices and the
chordwise legs does not depend on the flight condition: it is computed once
per lattice and unknown map (the GPU kernel cache), and each case adds its
wake legs and ground images.

Dense solves
------------
Small systems: LU with partial pivoting in float64 (``torch.linalg.solve``),
also for a float32 matrix. Batches of larger systems: the inverse of a
reference matrix as preconditioner and defect correction with the residual
in float64 (:func:`batched_solve`). A single large float32 system: LU in
float32 and two steps of iterative refinement with the residual in float64.
The corrections stop at the round-off level of a float64 matrix, or three
orders below the errors of the entries of a float32 matrix (Wilkinson, J.
H., "Rounding Errors in Algebraic Processes", Prentice-Hall, 1963; Higham,
N. J., "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM,
2002, chapter 12). The Newton steps of the nonlinear lifting line use LU in
float32: with two refinements for a float64 Jacobian, as it is for a float32
Jacobian (see :func:`newton_solve`).

Section polars
--------------
Tabulated polars use the shape-preserving piecewise cubic Hermite
interpolation of the CPU path (Fritsch, F. N. and Carlson, R. E., "Monotone
piecewise cubic interpolation", SIAM Journal on Numerical Analysis 17(2),
1980, pp. 238-246): the GPU evaluates the same cubic pieces in Horner form.
"""

from __future__ import annotations

import threading

import numpy as np
import torch
import warp as wp

from ventorum.aero.system import UnknownMap, panel_targets
from ventorum.core.datatypes import LinearAirfoil, TabulatedAirfoil
from ventorum.gpu.kernels import kernels

DEVICE = "cuda:0"
_MIRROR = (1.0, -1.0, 1.0)
_F64 = torch.float64

# Largest tensor of one kernel call [bytes]; larger batches are split into
# chunks of cases.
CHUNK_BYTES = 512 * 1024 ** 2

_stream_lock = threading.Lock()
_wp_streams: dict[int, object] = {}


def _dev() -> torch.device:
    return torch.device(DEVICE)


def _wp_stream():
    """Return the Warp stream of the current PyTorch CUDA stream (kernels and torch work stay in order)."""
    ts = torch.cuda.current_stream(_dev())
    key = int(ts.cuda_stream)
    s = _wp_streams.get(key)
    if s is None:
        with _stream_lock:
            s = _wp_streams.get(key)
            if s is None:
                s = wp.stream_from_torch(ts)
                _wp_streams[key] = s
    return s


def launch(kernel, dim, inputs) -> None:
    """Launch a Warp kernel on the current PyTorch stream."""
    wp.launch(kernel, dim=dim, inputs=inputs, device=DEVICE, stream=_wp_stream())


_wp_device = None
_wp_types: dict = {}


def view(t: torch.Tensor, dtype):
    """Return a Warp array view of the CUDA tensor *t* (no copy); the view keeps *t* alive.

    *dtype* is a Warp scalar type or a 3-vector type (then the last axis of
    *t* has length 3 and unit stride).
    """
    global _wp_device
    info = _wp_types.get(dtype)
    if info is None:
        table = {wp.float64: (torch.float64, 1), wp.float32: (torch.float32, 1), wp.int32: (torch.int32, 1),
                 wp.int64: (torch.int64, 1), wp.vec3d: (torch.float64, 3), wp.vec3f: (torch.float32, 3)}
        info = _wp_types[dtype] = table[dtype]
        _wp_device = wp.get_device(DEVICE)
    tdt, width = info
    if t.dtype != tdt or not t.is_cuda:
        raise TypeError(f"A Warp array of {dtype.__name__} needs a CUDA tensor of {tdt}, not {t.dtype} on {t.device}.")
    if width == 1:
        shape = tuple(t.shape)
    else:
        if t.shape[-1] != width or t.stride(-1) != 1:
            raise ValueError(f"A Warp array of {dtype.__name__} needs a last axis of length {width} and unit stride.")
        shape = tuple(t.shape[:-1])
    strides = None
    if not t.is_contiguous():
        es = t.element_size()
        strides = tuple(st * es for st in (t.stride() if width == 1 else t.stride()[:-1]))
    a = wp.array(ptr=t.data_ptr(), dtype=dtype, shape=shape, strides=strides, device=_wp_device)
    a._ventorum_tensor = t
    return a


def torch_dtype(precision: str) -> torch.dtype:
    """Return the torch type of *precision*."""
    return torch.float64 if precision == "float64" else torch.float32


def _f64(x) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float64), device=_dev())


def _i32(x) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x, dtype=np.int32), device=_dev())


def split2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the float32 pair (hi, lo) of the float64 tensor *x*: ``x = hi + lo`` to about 1e-15 relative."""
    hi = x.to(torch.float32)
    lo = (x - hi.to(_F64)).to(torch.float32)
    return hi.contiguous(), lo.contiguous()


def cview(store: dict, name, dtype):
    """Return the Warp view of ``store[name]``, made once (for tensors that live as long as *store*)."""
    key = ("_view", name, dtype)
    v = store.get(key)
    if v is None:
        v = store[key] = view(store[name], dtype)
    return v


_consts: dict = {}


def _const(name: str) -> torch.Tensor:
    """Return a small constant tensor on the GPU (made once): placeholders of unused kernel arrays, the mirror factors."""
    t = _consts.get(name)
    if t is None:
        f32 = torch.float32
        make = {
            "z2v": lambda: torch.zeros((1, 1, 3), dtype=f32, device=_dev()),
            "z2s": lambda: torch.zeros((1, 1), dtype=f32, device=_dev()),
            "z3v": lambda: torch.zeros((1, 1, 1, 3), dtype=f32, device=_dev()),
            "z3s": lambda: torch.zeros((1, 1, 1), dtype=f32, device=_dev()),
            "gk1": lambda: torch.zeros((1, 3), dtype=_F64, device=_dev()),
            "go1": lambda: torch.zeros((1,), dtype=_F64, device=_dev()),
            "mirror": lambda: torch.tensor(_MIRROR, dtype=_F64, device=_dev()),
        }
        t = _consts[name] = make[name]()
    return t


class PointSet:
    """Evaluation points on the GPU: float64 coordinates, normals, cross-surface core data and the Warp struct."""

    def __init__(self, dl: DeviceLattice, P: np.ndarray, normals: np.ndarray | None, group: np.ndarray,
                 rc_target: np.ndarray, use_tg: int):
        k = dl.k
        self.m = int(P.shape[0])
        self.P = _f64(P)
        self.Ph, self.Pl = split2(self.P)
        nrm = np.zeros_like(P) if normals is None else normals
        self.N = _f64(nrm).to(dl.T).contiguous()
        self.group = _i32(group)
        self.rc2 = _f64(np.asarray(rc_target, dtype=float) ** 2).to(dl.T).contiguous()
        s = k.Points()
        s.P = view(self.P, wp.vec3d)
        s.Ph = view(self.Ph, wp.vec3f)
        s.Pl = view(self.Pl, wp.vec3f)
        s.N = view(self.N, k.vec)
        s.group = view(self.group, wp.int32)
        s.rc2 = view(self.rc2, k.T)
        s.use_tg = int(use_tg)
        self.struct = s


class DeviceLattice:
    """The lattice arrays on the GPU for one precision.

    Point coordinates are float64. Vectors between points of one source,
    core radii and normals have the type of *precision*.
    """

    def __init__(self, lattice, precision: str):
        from ventorum.geometry.lattice import CROSS_SURFACE_CORE_FRACTION

        self.precision = precision
        self.k = kernels(precision)
        self.T = torch_dtype(precision)
        self.n_panels = int(lattice.n_panels)
        self.n_strips = int(lattice.n_strips)
        g = lattice.strip_core_group
        core_group = lattice.strip_surface if g is None else g
        self.panel_group = core_group[lattice.panel_strip]
        self.panel_trc = CROSS_SURFACE_CORE_FRACTION * lattice.width[lattice.panel_strip]
        # The target core changes the result only between different groups.
        self.use_tg = 1 if np.unique(self.panel_group).size > 1 else 0
        T = self.T
        self.t = {
            "A": _f64(lattice.a), "B": _f64(lattice.b), "ATE": _f64(lattice.a_te), "BTE": _f64(lattice.b_te),
            "r0b": _f64(lattice.b - lattice.a).to(T).contiguous(),
            "r0a": _f64(lattice.a - lattice.a_te).to(T).contiguous(),
            "r0t": _f64(lattice.b_te - lattice.b).to(T).contiguous(),
            "rc2": _f64(np.asarray(lattice.rc, dtype=float) ** 2).to(T).contiguous(),
            "group": _i32(self.panel_group),
        }
        self.t["Ah"], self.t["Al"] = split2(self.t["A"])
        self.TE = torch.stack([self.t["ATE"], self.t["BTE"]], dim=1)   # (n, 2, 3)
        k = self.k
        s = k.Sources()
        for name in ("A", "B", "ATE", "BTE"):
            setattr(s, name, view(self.t[name], wp.vec3d))
        s.Ah = view(self.t["Ah"], wp.vec3f)
        s.Al = view(self.t["Al"], wp.vec3f)
        for name in ("r0b", "r0a", "r0t"):
            setattr(s, name, view(self.t[name], k.vec))
        s.rc2 = view(self.t["rc2"], k.T)
        s.group = view(self.t["group"], wp.int32)
        self.sources = s
        self._maps: dict = {}
        self._fixed: dict = {}
        self._cache: dict = {}
        self.lat = _lattice_tensors(lattice)

    def unknown_map(self, lattice, umap: UnknownMap) -> dict | None:
        """Return the GPU data of an unknown map, or None if its unknowns have different source counts."""
        key = bool(umap.symmetric)
        data = self._maps.get(key)
        if data is None:
            col = np.asarray(umap.panel_column)
            order = np.argsort(col, kind="stable")
            counts = np.bincount(col, minlength=umap.n)
            if counts.size != umap.n or np.any(counts != counts[0]):
                self._maps[key] = False
                return None
            src_of = order.reshape(umap.n, int(counts[0]))   # sources of each unknown, in panel order
            up = np.asarray(umap.unknown_panels)
            tg = panel_targets(lattice, up)
            if lattice.collocation == "llt":
                from ventorum.solvers.core import _llt_strip_data

                sd = _llt_strip_data(lattice, umap)
                normals = sd["normal"]
            else:
                sd = None
                normals = lattice.normal_bc[up]
            pts = PointSet(self, lattice.cp[up], normals, tg.group, tg.rc, self.use_tg)
            data = {
                "n": umap.n, "symmetric": bool(umap.symmetric), "pts": pts,
                "src_of": _i32(src_of),
                "unknown_panels": torch.as_tensor(up.astype(np.int64), device=_dev()),
                "panel_column": torch.as_tensor(col.astype(np.int64), device=_dev()),
            }
            data["w_src_of"] = view(data["src_of"], wp.int32)
            if sd is not None:
                for name in ("normal", "chord_dir", "dl", "area", "a0", "alpha_L0"):
                    data[name] = _f64(sd[name])
                data["strip_np"] = np.asarray(sd["strip"], dtype=np.int64)
                data["strip"] = torch.as_tensor(data["strip_np"], device=_dev())
            if umap.symmetric:
                left = np.setdiff1d(np.arange(lattice.n_panels), up)
                data["left"] = torch.as_tensor(left, device=_dev())
                data["left_mirror"] = torch.as_tensor(lattice.panel_mirror[left], device=_dev())
            self._maps[key] = data
        return data or None

    def fixed(self, md: dict, want_v: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the transposed velocity tensor (n, n, 3) (or a placeholder) and normal part (n, n) of the fixed legs.

        The bound vortices and the chordwise legs do not depend on the flight
        condition, so this part is computed once per unknown map (the GPU
        kernel cache). Entry (i, j) is at [j, i] (see :mod:`ventorum.gpu.kernels`).
        """
        key = (md["symmetric"], want_v)
        out = self._fixed.get(key)
        if out is None:
            n = md["n"]
            k = self.k
            fv = torch.zeros((1, n, n, 3) if want_v else (1, 1, 1, 3), dtype=self.T, device=_dev())
            fn = torch.zeros((1, n, n), dtype=self.T, device=_dev())
            cs = CaseData(self, torch.zeros((1, 3), dtype=_F64, device=_dev()), [None])
            dummy = torch.zeros((1, 1, 3), dtype=self.T, device=_dev())
            if want_v:
                launch(k.tensors[1], (1, n, n), [self.sources, md["pts"].struct, cs.struct(md["pts"]),
                                                 md["w_src_of"], view(dummy, k.vec), 1, view(fv, k.vec), view(fn, k.T)])
            else:
                launch(k.normals[1], (1, n, n), [self.sources, md["pts"].struct, cs.struct(md["pts"]),
                                                 md["w_src_of"], view(fn[0], k.T), view(fn, k.T)])
            out = (fv[0], fn[0])
            self._fixed[key] = out
        return out


def _lattice_tensors(lattice) -> dict:
    """Return the lattice arrays of the loads on the GPU (float64)."""
    t = {name: _f64(getattr(lattice, name)) for name in (
        "chord", "width", "area", "dl", "normal", "chord_dir", "qc_mid", "a0", "alpha_L0",
        "te_left", "te_right", "cp_frac")}
    t["l_vec"] = _f64(lattice.b - lattice.a)
    t["mid"] = _f64(lattice.force_points)
    t["a_te"] = _f64(lattice.a_te)
    t["b_te"] = _f64(lattice.b_te)
    t["a"] = _f64(lattice.a)
    t["b"] = _f64(lattice.b)
    return t


def device_lattice(lattice, precision: str) -> DeviceLattice:
    """Return the GPU data of *lattice* for *precision* (cached on the lattice)."""
    key = ("gpu_lattice", DEVICE, precision)
    dl = lattice.geom_cache.get(key)
    if dl is None:
        dl = DeviceLattice(lattice, precision)
        lattice.geom_cache[key] = dl
    return dl


class CaseData:
    """Per-case GPU data of a batch: wake directions, ground planes and the float64 wake-leg data.

    The wake-leg data (float32 kernels only) are ``d x TE`` and ``d . TE`` of
    the trailing-edge points of every panel (and of their images), per case,
    made by one kernel (``case_te``). The per-point part (``d x P``,
    ``d . P``) is made for each point set by :meth:`struct`. *GK* and *GO*
    are the ground planes of the cases on the GPU (see :func:`batch_data`);
    without them they are made from *grounds*.
    """

    def __init__(self, dl: DeviceLattice, D: torch.Tensor, grounds: list, GK: torch.Tensor | None = None,
                 GO: torch.Tensor | None = None):
        self.dl = dl
        K = D.shape[0]
        self.K = K
        self.D = D
        self.D_T = D if dl.T == _F64 else D.to(dl.T).contiguous()
        has = [g is not None for g in grounds]
        if any(has) and not all(has):
            raise ValueError("A GPU batch needs all cases in ground effect or none.")
        self.n_img = 1 if all(has) and K > 0 else 0
        if GK is None:
            GKh = np.zeros((K, 3))
            GOh = np.zeros(K)
            if self.n_img:
                for k, g in enumerate(grounds):
                    GKh[k] = g.normal
                    GOh[k] = g.offset
            GK, GO = _f64(GKh), _f64(GOh)
        self.GK = GK
        self.GO = GO
        self.pre = dl.precision != "float64"
        self.arr: dict[str, torch.Tensor] = {}
        if self.pre:
            a = self.arr
            n = dl.n_panels
            f32 = torch.float32
            sfx = ("", "_i") if self.n_img else ("",)
            for x in sfx:
                for name in ("dxTh", "dxTl"):
                    a[name + x] = torch.empty((K, n, 2, 3), dtype=f32, device=_dev())
                for name in ("dTh", "dTl"):
                    a[name + x] = torch.empty((K, n, 2), dtype=f32, device=_dev())
            if self.n_img:
                a["Aih"] = torch.empty((K, n, 3), dtype=f32, device=_dev())
                a["Ail"] = torch.empty((K, n, 3), dtype=f32, device=_dev())
            img = "_i" if self.n_img else ""
            z2v = _const("z2v")
        self._views: dict = {}
        self._w = {
            "D": view(self.D_T, dl.k.vec), "GK": view(self.GK, wp.vec3d), "GO": view(self.GO, wp.float64),
        }
        self._keep: list = []
        if self.pre:
            wv = self._view
            launch(dl.k.case_te, (K, n), [
                view(D, wp.vec3d), self._w["GK"], self._w["GO"], cview(dl.t, "ATE", wp.vec3d),
                cview(dl.t, "BTE", wp.vec3d), cview(dl.t, "A", wp.vec3d), self.n_img,
                wv(a["dxTh"], wp.vec3f), wv(a["dxTl"], wp.vec3f), wv(a["dTh"], wp.float32),
                wv(a["dTl"], wp.float32), wv(a["dxTh" + img], wp.vec3f), wv(a["dxTl" + img], wp.vec3f),
                wv(a["dTh" + img], wp.float32), wv(a["dTl" + img], wp.float32),
                wv(a.get("Aih", z2v), wp.vec3f), wv(a.get("Ail", z2v), wp.vec3f)])

    def _view(self, t: torch.Tensor, dtype):
        """Return the Warp view of *t*, made once per batch (*t* lives as long as the batch)."""
        key = (id(t), dtype)
        v = self._views.get(key)
        if v is None:
            v = self._views[key] = view(t, dtype)
        return v

    def struct(self, pts: PointSet):
        """Return the Warp struct of the cases for the point set *pts* (it keeps its arrays alive)."""
        k = self.dl.k
        s = k.Cases()
        s.D = self._w["D"]
        s.GK = self._w["GK"]
        s.GO = self._w["GO"]
        s.n_img = self.n_img
        keep = []
        f32 = torch.float32
        z2v, z2s, z3v, z3s = _const("z2v"), _const("z2s"), _const("z3v"), _const("z3s")
        v2 = {"dxPh": z2v, "dxPl": z2v, "dPh": z2s, "dPl": z2s, "Aih": z2v, "Ail": z2v}
        v3 = {"dxTh": z3v, "dxTl": z3v, "dTh": z3s, "dTl": z3s}
        if self.pre:
            K, m = self.K, pts.m
            sfx = ("", "_i") if self.n_img else ("",)
            for x in sfx:
                for name in ("dxPh", "dxPl"):
                    v2[name + x] = torch.empty((K, m, 3), dtype=f32, device=_dev())
                for name in ("dPh", "dPl"):
                    v2[name + x] = torch.empty((K, m), dtype=f32, device=_dev())
            img = "_i" if self.n_img else ""
            launch(k.case_points, (K, m), [
                view(self.D, wp.vec3d), self._w["GK"], pts.struct.P, self.n_img,
                view(v2["dxPh"], wp.vec3f), view(v2["dxPl"], wp.vec3f), view(v2["dPh"], wp.float32),
                view(v2["dPl"], wp.float32), view(v2["dxPh" + img], wp.vec3f), view(v2["dxPl" + img], wp.vec3f),
                view(v2["dPh" + img], wp.float32), view(v2["dPl" + img], wp.float32)])
            for name in ("dxTh", "dxTl", "dTh", "dTl"):
                v3[name] = self.arr[name]
            if self.n_img:
                for name in ("dxTh_i", "dxTl_i", "dTh_i", "dTl_i"):
                    v3[name] = self.arr[name]
                v2["Aih"], v2["Ail"] = self.arr["Aih"], self.arr["Ail"]
            keep += list(v2.values()) + list(v3.values())
        for name in ("dxPh", "dxPl", "dPh", "dPl"):
            v2.setdefault(name + "_i", v2[name])
        for name in ("dxTh", "dxTl", "dTh", "dTl"):
            v3.setdefault(name + "_i", v3[name])
        wv = self._view
        for name, t in v2.items():
            setattr(s, name, wv(t, wp.vec3f if t.dim() == 3 else wp.float32))
        for name, t in v3.items():
            setattr(s, name, wv(t, wp.vec3f if t.dim() == 4 else wp.float32))
        # The struct holds views: the tensors live as long as this batch.
        self._keep.extend(keep)
        return s


# Iterative solve of a batch. The cases are split into REF groups of
# consecutive cases (one group per REF_CASES cases, at most REF_MAX_GROUPS);
# the middle case of each group is the reference. The inverse of each reference
# matrix (float32) is the preconditioner of its group: each case is corrected
# with its residual in float64 until the residual is at the round-off level.
REF_CASES = 128
REF_MAX_GROUPS = 4
REF_MIN_BATCH = 4          # smaller batches use one factorisation per case
REF_MAX_ITER = 40
REF_TOL = 2.0e-13          # relative residual of a converged case (float64 matrix)
REF_ACCEPT = 1.0e-11       # largest relative residual kept without a direct solve (float64 matrix)
# A float32 matrix has relative errors of about 1e-7 in its entries: a solve to
# a relative residual of 1e-10 adds an error three orders below them.
REF_TOL32 = 1.0e-10
REF_ACCEPT32 = 1.0e-9
# Systems up to this size are solved directly in float64 for any batch (a
# batched LU of small matrices is faster than the iterations).
REF_MIN_N = 128


def residual(MT: torch.Tensor, x: torch.Tensor, b: torch.Tensor, diag64: torch.Tensor | None,
             kern) -> torch.Tensor:
    """Return ``b - A x`` (float64) for ``A = M`` or ``A = diag(diag64) - M`` with a float64 accumulator.

    *MT* is the transposed matrix, contiguous (``MT[k, j, i] = M[k, i, j]``).
    """
    K, n = b.shape
    out = torch.empty((K, n), dtype=_F64, device=b.device)
    dg = diag64 if diag64 is not None else b
    T = wp.float64 if MT.dtype == torch.float64 else wp.float32
    launch(kern.residual, (K, n), [view(MT, T), view(x.contiguous(), wp.float64), view(b.contiguous(), wp.float64),
                                   view(dg.contiguous(), wp.float64), 0 if diag64 is None else 1, view(out, wp.float64)])
    return out


def _system(MT: torch.Tensor, diag64: torch.Tensor | None, rows=None) -> torch.Tensor:
    """Return the system matrix ``M`` or ``diag(diag64) - M`` (cases *rows*) from its transpose *MT*.

    The result is a transposed view (column-major per case), in the type of *MT*.
    """
    Ms = MT if rows is None else MT[rows]
    if diag64 is None:
        return Ms.transpose(1, 2)
    A = -Ms
    A.diagonal(dim1=1, dim2=2).add_((diag64 if rows is None else diag64[rows]).to(A.dtype))
    return A.transpose(1, 2)


# Largest system that a direct solve factorises in float64 also for a float32
# matrix (the float64 LU is faster than LU in float32 with refinement there).
DIRECT64_MAX_N = 512
DIRECT64_MAX_BYTES = 256 * 1024 ** 2   # largest float64 copy of the matrices of a direct solve


def _system64(MT: torch.Tensor, diag64: torch.Tensor | None) -> torch.Tensor:
    """Return the system matrix ``M`` or ``diag(diag64) - M`` in float64 (the diagonal not rounded to the type of *MT*)."""
    A = MT.transpose(1, 2).to(_F64)
    if diag64 is None:
        return A
    A = A.neg_() if A.data_ptr() != MT.data_ptr() else -A
    A.diagonal(dim1=1, dim2=2).add_(diag64)
    return A


def _direct(MT: torch.Tensor, rhs: torch.Tensor, diag64: torch.Tensor | None, kern) -> torch.Tensor:
    """Solve every case with its own factorisation (float64 LU; large float32 systems: LU and two refinements)."""
    if MT.dtype == torch.float64 or (MT.shape[-1] <= DIRECT64_MAX_N
                                     and MT.shape[0] * MT.shape[-1] ** 2 * 8 <= DIRECT64_MAX_BYTES):
        return torch.linalg.solve(_system64(MT, diag64), rhs.unsqueeze(-1)).squeeze(-1)
    A = _system(MT, diag64)
    LU, piv, _ = torch.linalg.lu_factor_ex(A)
    x = torch.linalg.lu_solve(LU, piv, rhs.to(torch.float32).unsqueeze(-1)).squeeze(-1).to(_F64)
    for _ in range(2):
        r = residual(MT, x, rhs, diag64, kern)
        x = x + torch.linalg.lu_solve(LU, piv, r.to(torch.float32).unsqueeze(-1)).squeeze(-1).to(_F64)
    return x


def batched_solve(MT: torch.Tensor, rhs: torch.Tensor, diag64: torch.Tensor | None = None, kern=None) -> torch.Tensor:
    """Solve the K systems ``A x = rhs`` with ``A = M`` or ``A = diag(diag64) - M``; return x in float64.

    *MT* is the transposed matrix M (K, n, n), contiguous, float32 or float64
    (``MT[k, j, i] = M[k, i, j]``, the layout of the kernels); *rhs* (K, n)
    and *diag64* (K, n) are float64; *kern* is the kernel set of the
    precision of *MT*.

    Small systems (``n <= REF_MIN_N``) and small batches get a direct solve
    (:func:`_direct`). For the others: the cases of a sweep have similar
    matrices, so the inverse of one reference matrix per group of
    consecutive cases is the preconditioner of its group, and each case is
    corrected with its residual in float64 (defect correction; Wilkinson
    1963, Higham 2002, chapter 12) until the relative residual is at the
    round-off level (float64 matrix) or below ``REF_TOL32`` (float32 matrix:
    three orders below the errors of its entries). A case that does not
    converge gets a direct solve.
    """
    if kern is None:
        kern = kernels("float64" if MT.dtype == torch.float64 else "float32")
    K, n = rhs.shape
    if K < REF_MIN_BATCH or n <= REF_MIN_N:
        return _direct(MT, rhs, diag64, kern)
    tol, accept = (REF_TOL, REF_ACCEPT) if MT.dtype == torch.float64 else (REF_TOL32, REF_ACCEPT32)
    ng = max(1, min(REF_MAX_GROUPS, round(K / REF_CASES)))
    size = (K + ng - 1) // ng
    refs = [min(K - 1, g * size + size // 2) for g in range(ng)]
    inv = torch.stack([torch.linalg.inv_ex(_system(MT, diag64, [r]).to(torch.float32))[0][0] for r in refs])
    pad = ng * size - K

    def precond(r: torch.Tensor) -> torch.Tensor:
        rp = torch.nn.functional.pad(r, (0, 0, 0, pad)) if pad else r
        R = rp.reshape(ng, size, n).transpose(1, 2).to(torch.float32)      # (ng, n, size)
        return torch.bmm(inv, R).transpose(1, 2).reshape(ng * size, n)[:K].to(_F64)

    bnorm = torch.clamp(torch.linalg.vector_norm(rhs, dim=1), min=1e-300)
    x = precond(rhs)
    hist: list[torch.Tensor] = []
    for _ in range(REF_MAX_ITER):
        r = residual(MT, x, rhs, diag64, kern)
        res = torch.linalg.vector_norm(r, dim=1) / bnorm
        hist.append(res)
        # One transfer: the largest residual and whether a case still converges.
        still = (hist[-1] <= 0.5 * hist[-3]).any() if len(hist) >= 3 else torch.ones((), dtype=torch.bool,
                                                                                       device=res.device)
        top, going = torch.stack([res.max(), still.to(_F64)]).tolist()
        if top < tol:
            break
        if not going:
            break   # no case is still converging (round-off floor or divergence)
        x = x + precond(r)
    bad = torch.nonzero(hist[-1] > accept).flatten()
    if bad.numel():
        x[bad] = _direct(MT[bad], rhs[bad], None if diag64 is None else diag64[bad], kern)
    return x


def _chunks(K: int, per_case_bytes: int):
    step = max(1, min(K, CHUNK_BYTES // max(1, per_case_bytes)))
    for k0 in range(0, K, step):
        yield k0, min(K, k0 + step)


def batch_data(conditions: list, wds, grounds: list) -> dict:
    """Return the per-case inputs of a batch on the GPU, in one transfer.

    ``D`` (K, 3) free-stream direction, ``V``, ``rho``, ``alpha`` (K,) speed
    [m/s], density [kg/m^3] and angle of attack [rad], ``WD`` (K, 3) wake
    direction, ``GK`` (K, 3) and ``GO`` (K,) normal and offset [m] of the
    ground plane (zero for a case without ground).
    """
    from ventorum.aero.system import freestream_direction

    K = len(conditions)
    buf = np.zeros(13 * K)
    D = buf[:3 * K].reshape(K, 3)
    GK = buf[6 * K:9 * K].reshape(K, 3)
    V, rho, alpha, GO = (buf[(9 + j) * K:(10 + j) * K] for j in range(4))
    for k, c in enumerate(conditions):
        D[k] = freestream_direction(c.alpha, c.beta)
        V[k] = float(c.V_inf)
        rho[k] = float(c.rho)
        alpha[k] = float(c.alpha)
    buf[3 * K:6 * K] = np.asarray(wds, dtype=float).reshape(-1)
    for k, g in enumerate(grounds):
        if g is not None:
            GK[k] = g.normal
            GO[k] = g.offset
    dev = _f64(buf)
    out = {"D": dev[:3 * K].view(K, 3), "WD": dev[3 * K:6 * K].view(K, 3), "GK": dev[6 * K:9 * K].view(K, 3)}
    for j, name in enumerate(("V", "rho", "alpha", "GO")):
        out[name] = dev[(9 + j) * K:(10 + j) * K]
    return out


# --------------------------------------------------------------------------- lifting line
def llt_tensors(dl: DeviceLattice, md: dict, cs: CaseData, k0: int, k1: int, want_v: bool = True):
    """Return the transposed velocity tensor (K, n, n, 3) and normal part (K, n, n) for cases k0..k1.

    Entry (i, j) (point i, unknown j) is at [k, j, i].
    """
    k = dl.k
    K, n = k1 - k0, md["n"]
    fv, fn = dl.fixed(md, want_v=True)
    sub = cs if (k0 == 0 and k1 == cs.K) else cs.subset(k0, k1)
    Vt = torch.empty((K, n, n, 3) if want_v else (1, 1, 1, 3), dtype=dl.T, device=_dev())
    An = torch.empty((K, n, n), dtype=dl.T, device=_dev())
    st = sub.struct(md["pts"])
    launch(k.tensors[2], (K, n, n), [dl.sources, md["pts"].struct, st, md["w_src_of"],
                                     view(fv, k.vec), 1 if want_v else 0, view(Vt, k.vec), view(An, k.T)])
    return Vt, An


def _subset(self: CaseData, k0: int, k1: int) -> CaseData:
    """Return the cases k0..k1 of a batch (views, no copy)."""
    new = CaseData.__new__(CaseData)
    new.dl = self.dl
    new.K = k1 - k0
    new.D = self.D[k0:k1]
    new.D_T = self.D_T[k0:k1]
    new.n_img = self.n_img
    new.GK = self.GK[k0:k1]
    new.GO = self.GO[k0:k1]
    new.pre = self.pre
    new.arr = {name: t[k0:k1] for name, t in self.arr.items()}
    new._views = {}
    k = self.dl.k
    new._w = {"D": view(new.D_T, k.vec), "GK": view(new.GK, wp.vec3d), "GO": view(new.GO, wp.float64)}
    new._keep = []
    return new


CaseData.subset = _subset


def llt_terms(dl: DeviceLattice, md: dict, Dfs: torch.Tensor, V: torch.Tensor) -> tuple:
    """Return the diagonal, the right-hand side and the residual scale (K, n), and the free-stream velocity (K, 3).

    The terms of :func:`ventorum.solvers.core.solve_llt_linear` (one kernel):
    ``2 |u x dl_i| / (a0_i dA_i)``, ``V (u . n_i - alpha_L0_i)``,
    ``V^2 dA_i`` and ``V u``.
    """
    K, n = Dfs.shape[0], md["n"]
    diag = torch.empty((K, n), dtype=_F64, device=_dev())
    rhs = torch.empty((K, n), dtype=_F64, device=_dev())
    scale = torch.empty((K, n), dtype=_F64, device=_dev())
    vinf = torch.empty((K, 3), dtype=_F64, device=_dev())
    launch(dl.k.llt_rhs, (K, n), [view(Dfs.contiguous(), wp.vec3d), view(V.contiguous(), wp.float64),
                                  cview(md, "dl", wp.vec3d), cview(md, "normal", wp.vec3d), cview(md, "area", wp.float64),
                                  cview(md, "a0", wp.float64), cview(md, "alpha_L0", wp.float64),
                                  view(diag, wp.float64), view(rhs, wp.float64), view(scale, wp.float64),
                                  view(vinf, wp.vec3d)])
    return diag, rhs, scale, vinf


def llt_linear(dl: DeviceLattice, md: dict, cs: CaseData, Dfs: torch.Tensor, V: torch.Tensor) -> tuple:
    """Solve the linear lifting line of all cases; return the unknown circulation (K, n) and velocity (K, n, 3).

    The system is that of :func:`ventorum.solvers.core.solve_llt_linear`:
    ``[2 |u x dl_i| / (a0_i dA_i)] G_i - sum_j (v_ij . n_i) G_j = V (u . n_i - alpha_L0_i)``.
    """
    K, n = cs.K, md["n"]
    g_all = torch.empty((K, n), dtype=_F64, device=_dev())
    v_all = torch.empty((K, n, 3), dtype=_F64, device=_dev())
    elem = 8 if dl.T == torch.float64 else 4
    diag, rhs, _, _ = llt_terms(dl, md, Dfs, V)
    for k0, k1 in _chunks(K, n * n * elem * 12):
        Vt, An = llt_tensors(dl, md, cs, k0, k1)
        g = batched_solve(An, rhs[k0:k1], diag64=diag[k0:k1], kern=dl.k)
        g_all[k0:k1] = g
        v = v_all[k0:k1]
        launch(dl.k.contract, (k1 - k0, n), [view(Vt, dl.k.vec), view(g, wp.float64), view(v, wp.vec3d)])
    return g_all, v_all


def newton_solve(JT: torch.Tensor, rhs: torch.Tensor, kern, refine: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the K systems ``J x = rhs`` (Newton steps); return x (float64) and a mask of the singular systems.

    *JT* is the transposed Jacobian, contiguous. LU in float32 and *refine*
    refinement steps with the residual in float64 (the matrix in its own
    type). The default is two steps for a float64 Jacobian (the step of the
    given Jacobian to round-off) and none for a float32 Jacobian: its
    entries have float32 errors of the size of the error of the float32 LU,
    so a refinement does not make the step more accurate.
    """
    if refine is None:
        refine = 2 if JT.dtype == torch.float64 else 0
    LU, piv, info = torch.linalg.lu_factor_ex(JT.transpose(1, 2).to(torch.float32))
    x = torch.linalg.lu_solve(LU, piv, rhs.to(torch.float32).unsqueeze(-1)).squeeze(-1).to(_F64)
    for _ in range(refine):
        r = residual(JT, x, rhs, None, kern)
        x = x + torch.linalg.lu_solve(LU, piv, r.to(torch.float32).unsqueeze(-1)).squeeze(-1).to(_F64)
    bad = (info != 0) | ~torch.isfinite(x).all(dim=1)
    return x, bad


def _airfoil_token(airfoils: list) -> tuple:
    """Return a value that changes when the list of airfoils or a value of one of them changes, and the objects it uses.

    A linear airfoil gives its four values; a tabulated airfoil gives the
    identity of its prepared tables (a new object after a change of its
    tables). The second return value keeps these objects alive, so that
    their identities are not given to new objects while the value is kept.
    """
    uniq: dict[int, object] = {}
    for af in airfoils:
        uniq.setdefault(id(af), af)
    state = []
    refs: list = list(uniq.values())
    for i, af in uniq.items():
        if isinstance(af, LinearAirfoil):
            state.append((i, float(af.a0), float(af.alpha_L0), float(af.Cd0), float(af.Cm0)))
        elif isinstance(af, TabulatedAirfoil):
            tables = af.tables()
            refs.append(tables)
            state.append((i, id(tables)))
        else:
            state.append((i, None))
    return (tuple(map(id, airfoils)), tuple(state)), refs


def polar_struct(dl: DeviceLattice, airfoils: list, role=None):
    """Return the Warp struct of the section polars of *airfoils* (one per strip) and its tensors.

    With a *role* (a key of the list, for example ``"all"`` for
    ``lattice.airfoils``) the struct is kept on *dl* and made again only
    when the airfoils or their values change (:func:`_airfoil_token`).
    """
    if role is not None:
        token, refs = _airfoil_token(airfoils)
        hit = dl._cache.get(("polars", role))
        if hit is not None and hit[0] == token:
            return hit[2], hit[3]
        st, t = polar_struct(dl, airfoils)
        dl._cache[("polars", role)] = (token, refs, st, t)
        return st, t
    k = dl.k
    n = len(airfoils)
    kind = np.zeros(n, dtype=np.int32)
    tab = np.zeros(n, dtype=np.int32)
    a0 = np.zeros(n)
    aL0 = np.zeros(n)
    cd0 = np.zeros(n)
    cm0 = np.zeros(n)
    tabs: dict[int, int] = {}
    tab_objs = []
    for i, af in enumerate(airfoils):
        if isinstance(af, LinearAirfoil):
            a0[i], aL0[i] = float(af.a0), float(af.alpha_L0)
            cd0[i], cm0[i] = float(af.Cd0), float(af.Cm0)
        elif isinstance(af, TabulatedAirfoil):
            kind[i] = 1
            if id(af) not in tabs:
                tabs[id(af)] = len(tab_objs)
                tab_objs.append(af)
            tab[i] = tabs[id(af)]
        else:
            raise TypeError(f"The GPU pipelines support linear and tabulated airfoils, not {type(af).__name__}.")
    n_tab = max(1, len(tab_objs))
    n_max = max([af.tables()[0].size for af in tab_objs] + [2])
    x = np.zeros((n_tab, n_max))
    npts = np.full(n_tab, 2, dtype=np.int32)
    has_cm = np.zeros(n_tab, dtype=np.int32)
    tables = {key: np.zeros((n_tab, n_max - 1, 4)) for key in ("Cl", "dCl", "Cd", "Cm")}
    for t, af in enumerate(tab_objs):
        f = af._pchip()
        xa = af.tables()[0]
        x[t, :xa.size] = xa
        npts[t] = xa.size
        for key in tables:
            if key in f:
                c = np.asarray(f[key].c)            # (order, pieces), highest power first
                tables[key][t, :c.shape[1], 4 - c.shape[0]:] = c.T
        has_cm[t] = 1 if "Cm" in f else 0
    t = {"kind": _i32(kind), "tab": _i32(tab), "a0": _f64(a0), "aL0": _f64(aL0), "cd0": _f64(cd0),
         "cm0": _f64(cm0), "x": _f64(x), "npts": _i32(npts), "has_cm": _i32(has_cm),
         "ccl": _f64(tables["Cl"]), "cdcl": _f64(tables["dCl"]), "ccd": _f64(tables["Cd"]), "ccm": _f64(tables["Cm"])}
    st = k.Polars()
    for name in ("kind", "tab", "npts", "has_cm"):
        setattr(st, name, view(t[name], wp.int32))
    for name in ("a0", "aL0", "cd0", "cm0", "x", "ccl", "cdcl", "ccd", "ccm"):
        setattr(st, name, view(t[name], wp.float64))
    return st, t


def llt_nonlinear_prepare(dl: DeviceLattice, md: dict, cs: CaseData, lattice, Dfs: torch.Tensor,
                          V: torch.Tensor) -> dict:
    """Return the data of the Newton solves of a batch: velocity tensor, projections, polars, linear start."""
    K, n = cs.K, md["n"]
    k = dl.k
    T = dl.T
    Vt, An = llt_tensors(dl, md, cs, 0, K)      # transposed: [k, j, i]
    adir = md["chord_dir"]
    Va = torch.einsum("kjic,ic->kji", Vt, adir.to(T)).contiguous()
    strips = md["strip_np"]
    pol, pol_t = polar_struct(dl, [lattice.airfoils[s] for s in strips], role=("unknowns", md["symmetric"]))
    diag, rhs, scale, vinf = llt_terms(dl, md, Dfs, V)
    g_lin = batched_solve(An, rhs, diag64=diag, kern=k)
    return {"dl": dl, "md": md, "K": K, "n": n, "Vt": Vt, "An": An, "Va": Va, "pol": pol, "pol_t": pol_t,
            "Vinf": vinf, "scale": scale, "g_lin": g_lin}


# The GPU hands the cases that are still not converged after this many Newton
# iterations to the CPU solver (see llt_nonlinear, handover).
HANDOVER_ITERATIONS = 25

# Line search of the Newton steps (the values of solve_llt_nonlinear): the step
# lengths 1, 1/2, ... 1/1024, then the damping factors of the fixed-point fallback.
_LAMBDAS = [0.5 ** i for i in range(11)]
_OMEGAS = [0.5, 0.25, 0.1, 0.05, 0.02]


def _trial_steps() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the step lengths of the trials and their direction (0 Newton step, 1 fixed point) on the GPU."""
    t = _consts.get("trials")
    if t is None:
        t = _consts["trials"] = (_f64(np.array(_LAMBDAS + _OMEGAS)),
                                 _i32(np.array([0] * len(_LAMBDAS) + [1] * len(_OMEGAS))))
    return t


def llt_nonlinear(prep: dict, tol: float, max_iter: int, g_start: torch.Tensor | None = None,
                  active0: torch.Tensor | None = None, handover: int = 0, handover_after: int = 8) -> dict:
    """Solve the nonlinear lifting line of all cases with Newton's method.

    The method of :func:`ventorum.solvers.core.solve_llt_nonlinear` for all
    cases together: the residual ``R_i = 2 |W_i x dl_i| G_i - V^2 dA_i
    Cl_i(alpha_i)``, the Jacobian, the backtracking line search with the
    sufficient-decrease test (step lengths 1 to 1/1024), and the damped
    fixed-point fallback. The local velocity is linear in the circulation,
    so all trial points of one iteration come from two products with the
    velocity tensor and are evaluated in one kernel; each case then takes
    the first trial that the CPU rules accept. A case stops when it
    converges, when no trial lowers its residual, or after *max_iter*
    iterations.

    Parameters
    ----------
    g_start : (K, n) tensor or None
        Start circulation of the unknowns [m^2/s]; None starts from the
        linear lifting line (the same velocity tensor).
    active0 : (K,) bool tensor or None
        The cases to iterate; the others are only evaluated at their start.
    handover, handover_after : int
        After *handover_after* iterations, stop when no more than *handover*
        cases are still iterating; they are returned in ``unresolved``.

    Returns
    -------
    dict
        ``g`` (K, n) circulation of the unknowns, ``alpha`` (K, n) section
        angles [rad], ``W`` (K, n, 3) local velocity [m/s], ``converged``
        (K,), ``iterations`` (K,), ``history`` (list of K lists of floats),
        ``unresolved`` (K,).
    """
    dl, md, K, n = prep["dl"], prep["md"], prep["K"], prep["n"]
    k = dl.k
    dev = _dev()
    i32 = torch.int32
    Vt, An, Va, pol = prep["Vt"], prep["An"], prep["Va"], prep["pol"]
    Vinf, scale = prep["Vinf"], prep["scale"]
    w_dl, w_n, w_a = (cview(md, name, wp.vec3d) for name in ("dl", "normal", "chord_dir"))
    w_scale = view(scale, wp.float64)
    w_Vt = view(Vt, k.vec)
    w_An = view(An, k.T)
    w_Va = view(Va, k.T)
    g = (prep["g_lin"] if g_start is None else g_start).clone().contiguous()
    lam, sel = _trial_steps()
    L = lam.shape[0]
    w_lam, w_sel = view(lam, wp.float64), view(sel, wp.int32)
    st = {name: torch.empty((K, n), dtype=_F64, device=dev) for name in ("R", "wxn", "alpha", "c_an", "c_va", "d2")}
    st["wvec"] = torch.empty((K, n, 3), dtype=_F64, device=dev)
    w_st = {name: view(t, wp.vec3d if name == "wvec" else wp.float64) for name, t in st.items()}
    flags = {name: torch.zeros(K, dtype=i32, device=dev) for name in ("accepted", "conv", "active", "has_step", "iters")}
    flags["active0"] = (torch.ones(K, dtype=i32, device=dev) if active0 is None else active0.to(i32).contiguous())
    w_f = {name: view(t, wp.int32) for name, t in flags.items()}
    f0 = torch.empty(K, dtype=_F64, device=dev)
    w_f0 = view(f0, wp.float64)
    step = torch.zeros((K, n), dtype=_F64, device=dev)
    w_step = view(step, wp.float64)
    U1 = torch.empty((K, n, 3), dtype=_F64, device=dev)
    U2 = torch.empty((K, n, 3), dtype=_F64, device=dev)
    w_U1, w_U2 = view(U1, wp.vec3d), view(U2, wp.vec3d)
    mer = torch.empty((K, L), dtype=_F64, device=dev)
    w_mer = view(mer, wp.float64)
    w_rr = view(torch.empty((K, L, n), dtype=_F64, device=dev), wp.float64)

    W = torch.empty((K, n, 3), dtype=_F64, device=dev)
    launch(k.contract, (K, n), [w_Vt, view(g, wp.float64), view(W, wp.vec3d)])
    W.add_(Vinf[:, None, :])
    w_W, w_g = view(W, wp.vec3d), view(g, wp.float64)

    def state() -> None:
        launch(k.nl_state, (K, n), [w_W, w_g, w_dl, w_n, w_a, w_scale, pol, w_st["R"], w_st["wxn"], w_st["alpha"],
                                    w_st["c_an"], w_st["c_va"], w_st["d2"], w_st["wvec"]])

    def reduce(first: int) -> torch.Tensor:
        res = torch.empty(K, dtype=_F64, device=dev)
        launch(k.nl_reduce, (K,), [w_st["R"], w_scale, float(tol), first, w_f["accepted"], w_f["active0"],
                                   w_f["conv"], w_f["active"], w_f0, view(res, wp.float64)])
        return res

    state()
    hist = [reduce(1)]
    hist_on = [torch.ones(K, dtype=i32, device=dev)]
    unresolved = torch.zeros(K, dtype=torch.bool, device=dev)
    for it_count in range(int(max_iter)):
        idx = torch.nonzero(flags["active"]).flatten()
        m = int(idx.shape[0])
        if m == 0:
            break
        if handover and ((it_count >= handover_after and m <= handover) or it_count >= HANDOVER_ITERATIONS):
            # A few slow cases, or cases that Newton's method does not bring
            # to the tolerance in HANDOVER_ITERATIONS iterations (it converges
            # in a few iterations when it converges): the CPU solves them with
            # its restarts (one GPU iteration costs the same for one case as
            # for many).
            unresolved = flags["active"] != 0
            break
        # Jacobian of the active cases (transposed) and the Newton step.
        JT = torch.empty((m, n, n), dtype=dl.T, device=dev)
        launch(k.nl_jacobian, (m, n, n), [view(idx.to(i32), wp.int32), w_Vt, w_An, w_Va, w_g, w_st["wxn"],
                                          w_st["c_an"], w_st["c_va"], w_st["wvec"], view(JT, k.T)])
        step_a, bad = newton_solve(JT, -st["R"][idx], k)
        step.zero_()
        step[idx] = torch.where(bad[:, None], 0.0, step_a)
        flags["has_step"].zero_()
        flags["has_step"][idx] = (~bad).to(i32)
        launch(k.contract2, (K, n), [w_Vt, w_step, w_st["d2"], w_U1, w_U2])
        launch(k.nl_trials, (K, L, n), [w_W, w_U1, w_U2, w_g, w_step, w_st["d2"], w_lam, w_sel, w_dl, w_n, w_a,
                                        w_scale, pol, w_rr])
        launch(k.nl_merit, (K, L), [w_rr, w_mer])
        # The first accepted trial of each case (the order of the CPU rules)
        # and the new state.
        launch(k.nl_update, (K, n), [w_mer, w_f0, w_lam, w_sel, len(_LAMBDAS), w_f["has_step"], w_f["active"],
                                     w_step, w_st["d2"], w_U1, w_U2, w_g, w_W, w_f["accepted"], w_f["iters"]])
        state()
        hist.append(reduce(0))
        hist_on.append(flags["accepted"].clone())
    H = torch.stack(hist).cpu().numpy()
    Hon = torch.stack(hist_on).cpu().numpy() != 0
    history = [[float(H[t, c]) for t in range(H.shape[0]) if Hon[t, c]] for c in range(K)]
    return {"g": g, "alpha": st["alpha"].clone(), "W": W, "converged": flags["conv"] != 0,
            "iterations": flags["iters"].to(torch.int64), "history": history, "unresolved": unresolved}


def to_panels(lattice, md: dict, g: torch.Tensor, v: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the circulation (K, n_panels) and the control-point velocity (K, n_panels, 3) of every panel.

    On a symmetric map a left panel gets the circulation of its mirror panel,
    and the mirror image of its velocity.
    """
    G = g[:, md["panel_column"]]
    if v is None:
        return G, None
    K = g.shape[0]
    vp = torch.empty((K, lattice.n_panels, 3), dtype=_F64, device=g.device)
    vp[:, md["unknown_panels"]] = v
    if md["symmetric"]:
        vp[:, md["left"]] = vp[:, md["left_mirror"]] * _const("mirror")
    return G, vp


# --------------------------------------------------------------------------- vortex lattice
def vlm_system(dl: DeviceLattice, md: dict, cs: CaseData, k0: int, k1: int, lattice=None) -> torch.Tensor:
    """Return the transposed normal-velocity influence matrices (K, n, n) of the vortex lattice for cases k0..k1.

    The bound vortices and chordwise legs come from the GPU kernel cache;
    each case adds its wake legs and ground images. With more chordwise
    panels than one, the wake legs come from the wake table (one evaluation
    per wake group and point, :func:`_wake_groups`).
    """
    k = dl.k
    K, n = k1 - k0, md["n"]
    _, fn = dl.fixed(md, want_v=False)
    sub = cs if (k0 == 0 and k1 == cs.K) else cs.subset(k0, k1)
    st = sub.struct(md["pts"])
    An = torch.empty((K, n, n), dtype=dl.T, device=_dev())
    wg = _wake_groups(dl, lattice) if lattice is not None else None
    if wg is None or wg["n"] * 2 > lattice.n_panels:
        launch(k.normals[2], (K, n, n), [dl.sources, md["pts"].struct, st, md["w_src_of"], view(fn, k.T), view(An, k.T)])
        return An
    nw = wg["n"]
    table = torch.empty((K, nw, n), dtype=dl.T, device=_dev())
    launch(k.wake_table, (K, nw, n), [dl.sources, md["pts"].struct, st, view(wg["rep"], wp.int32), 0, view(table, k.T)])
    table_img = table
    if cs.n_img:
        table_img = torch.empty((K, nw, n), dtype=dl.T, device=_dev())
        launch(k.wake_table, (K, nw, n), [dl.sources, md["pts"].struct, st, view(wg["rep"], wp.int32), 1,
                                          view(table_img, k.T)])
    wgroup = wg.get("panel_group32")
    if wgroup is None:
        wgroup = wg["panel_group32"] = wg["inverse"].to(torch.int32).contiguous()
    launch(k.normal_from_table, (K, n, n), [dl.sources, md["pts"].struct, st, md["w_src_of"], view(wgroup, wp.int32),
                                            view(fn, k.T), view(table, k.T), view(table_img, k.T), view(An, k.T)])
    return An


def vlm_solve(dl: DeviceLattice, md: dict, cs: CaseData, lattice, Dfs: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Solve the vortex lattice of all cases; return the circulation of the unknowns (K, n) [m^2/s].

    The system is that of :func:`ventorum.solvers.core.solve_vlm`: flow
    tangency ``(V_inf + v) . n_bc = 0`` at the control points.
    """
    K, n = cs.K, md["n"]
    nbc = md.get("nbc64")
    if nbc is None:
        nbc = md["nbc64"] = _f64(lattice.normal_bc[md["unknown_panels"].cpu().numpy()])
    rhs = -V[:, None] * (Dfs @ nbc.T)
    g_all = torch.empty((K, n), dtype=_F64, device=_dev())
    elem = 8 if dl.T == torch.float64 else 4
    for k0, k1 in _chunks(K, n * n * elem * 3):
        An = vlm_system(dl, md, cs, k0, k1, lattice)
        g_all[k0:k1] = batched_solve(An, rhs[k0:k1], kern=dl.k)
    return g_all


def load_points(dl: DeviceLattice, lattice, fold: bool) -> dict:
    """Return the evaluation points of the vortex-lattice loads on the GPU (cached per fold flag).

    The point sets are those of :func:`ventorum.aero.loads.compute_loads_batch`:
    the force points and the two leg mid-points of every panel. With *fold*
    only the panels of the right half are evaluated; a left panel gets the
    mirror image of the point set ``swap[k]`` of its mirror panel.
    """
    from ventorum.aero.loads import _symmetric_fold

    key = ("load_points", fold)
    data = dl._cache.get(key)
    if data is None:
        mid = lattice.force_points
        leg_a = 0.5 * (lattice.a_te + lattice.a)
        leg_b = 0.5 * (lattice.b + lattice.b_te)
        points = [mid, leg_a, leg_b]
        swap = [0, 2, 1]
        geo = _symmetric_fold(lattice, np.ones(lattice.n_panels), points, swap, leg_forces=True) if fold else None
        eval_panels = np.arange(lattice.n_panels) if geo is None else geo[0]
        P = np.vstack([pk[eval_panels] for pk in points])
        tg = panel_targets(lattice, np.tile(eval_panels, len(points)))
        data = {"pts": PointSet(dl, P, None, tg.group, tg.rc, dl.use_tg), "n_e": int(eval_panels.size),
                "eval": torch.as_tensor(eval_panels, device=_dev()), "swap": swap, "folded": geo is not None}
        if geo is not None:
            left = geo[1]
            data["left"] = torch.as_tensor(left, device=_dev())
            data["left_mirror"] = torch.as_tensor(lattice.panel_mirror[left], device=_dev())
        dl._cache[key] = data
    return data


# Largest cached tensor of the fixed legs at the load points [bytes] (float64).
FIXED_POINTS_BYTES = 768 * 1024 ** 2


def _wake_groups(dl: DeviceLattice, lattice) -> dict:
    """Return the wake groups of the panels: one per strip and core radius (the panels that share their wake legs)."""
    data = dl._cache.get("wake_groups")
    if data is None:
        key = np.stack([lattice.panel_strip.astype(float), np.asarray(lattice.rc, dtype=float)], axis=1)
        uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        data = {"rep": _i32(first), "inverse": torch.as_tensor(inverse.ravel().astype(np.int64), device=_dev()),
                "n": int(uniq.shape[0])}
        dl._cache["wake_groups"] = data
    return data


def _fixed_points(dl: DeviceLattice, lattice, lp: dict, fold: bool) -> torch.Tensor | None:
    """Return the cached fixed-leg velocity matrix (m * 3, n_cols) at the load points, or None if it is too large.

    The columns are the panels, or with *fold* the right panels with their
    mirror panels added (the circulation of a symmetric case is the same on
    both). Computed once per lattice (the GPU kernel cache of the loads).
    """
    key = ("fixed_points", fold)
    if key in dl._cache:
        return dl._cache[key]
    pts = lp["pts"]
    n_p = lattice.n_panels
    if fold:
        right = lp["eval"].cpu().numpy()
        cols = np.stack([right, lattice.panel_mirror[right]], axis=1)
    else:
        cols = np.arange(n_p)[:, None]
    n_cols = cols.shape[0]
    if pts.m * 3 * n_cols * 8 > FIXED_POINTS_BYTES:
        dl._cache[key] = None
        return None
    out = torch.empty((pts.m, 3, n_cols), dtype=_F64, device=_dev())
    cs = CaseData(dl, torch.zeros((1, 3), dtype=_F64, device=_dev()), [None])
    launch(dl.k.fixed_points, (pts.m, n_cols), [dl.sources, pts.struct, cs.struct(pts), view(_i32(cols), wp.int32),
                                                view(out, wp.float64)])
    T = out.reshape(pts.m * 3, n_cols)
    dl._cache[key] = (T, torch.as_tensor(cols[:, 0], device=_dev()))
    return dl._cache[key]


def vlm_load_velocities(dl: DeviceLattice, lattice, cs: CaseData, G: torch.Tensor, fold: bool) -> list:
    """Return the induced velocity (K, n_panels, 3) [m/s] at the force points and at the two leg mid-points.

    Three parts, the same sum as :func:`ventorum.aero.vortex.induced_velocity`:

    * the bound vortices and chordwise legs: a product of the cached
      fixed-leg matrix (:func:`_fixed_points`) with the circulation, in
      float64;
    * the wake legs: per wake group (:func:`_wake_groups`) with the summed
      circulation of its panels;
    * the ground images (bound vortices and chordwise legs) per panel.

    Without the cache (too large), one kernel computes all parts per panel.
    """
    k = dl.k
    lp = load_points(dl, lattice, fold)
    pts = lp["pts"]
    K = G.shape[0]
    out = torch.empty((K, pts.m, 3), dtype=_F64, device=_dev())
    fixed = _fixed_points(dl, lattice, lp, lp["folded"])
    elem = 8 if dl.T == torch.float64 else 4
    if fixed is not None:
        T, col0 = fixed
        out.copy_((T @ G[:, col0].T).T.reshape(K, pts.m, 3))
        wg = _wake_groups(dl, lattice)
        Gw = torch.zeros((K, wg["n"]), dtype=_F64, device=_dev())
        Gw.index_add_(1, wg["inverse"], G)
        for k0, k1 in _chunks(K, pts.m * elem * 64):
            sub = cs if (k0 == 0 and k1 == cs.K) else cs.subset(k0, k1)
            st = sub.struct(pts)
            o = out[k0:k1]
            launch(k.induced_wakes, (k1 - k0, pts.m), [dl.sources, pts.struct, st, view(wg["rep"], wp.int32),
                                                       view(Gw[k0:k1].contiguous(), wp.float64), view(o, wp.vec3d)])
            if cs.n_img:
                launch(k.induced_img_fixed, (k1 - k0, pts.m), [dl.sources, pts.struct, st,
                                                              view(G[k0:k1].contiguous(), wp.float64), view(o, wp.vec3d)])
    else:
        for k0, k1 in _chunks(K, pts.m * elem * 64):
            sub = cs if (k0 == 0 and k1 == cs.K) else cs.subset(k0, k1)
            o = out[k0:k1]
            launch(k.induced, (k1 - k0, pts.m), [dl.sources, pts.struct, sub.struct(pts),
                                                view(G[k0:k1].contiguous(), wp.float64), view(o, wp.vec3d)])
    n_e, n_p = lp["n_e"], lattice.n_panels
    sets = []
    for j in range(3):
        v = torch.empty((K, n_p, 3), dtype=_F64, device=_dev())
        v[:, lp["eval"]] = out[:, j * n_e:(j + 1) * n_e]
        sets.append(v)
    if lp["folded"]:
        mirror = _const("mirror")
        for j, sw in enumerate(lp["swap"]):
            sets[j][:, lp["left"]] = sets[sw][:, lp["left_mirror"]] * mirror
    return sets


# --------------------------------------------------------------------------- Trefftz plane
def trefftz(dl: DeviceLattice, lattice, strip_gamma: torch.Tensor, WD: torch.Tensor, grounds: list,
            symmetric: bool = False, GK: torch.Tensor | None = None, GO: torch.Tensor | None = None) -> tuple:
    """Return the Trefftz-plane normal wash (K, n_strips) [m/s] and the projected strip lengths (K, n_strips) [m].

    The geometry is that of :func:`ventorum.aero.loads.trefftz_induced_drag_batch`:
    the trailing-edge points projected on the plane normal to the wake
    direction, the evaluation points at the span fraction of the control
    points, and the ground images of the trailing-edge points (one kernel,
    ``trefftz_prep``). With *symmetric* (all cases symmetric about the x-z
    plane, on a lattice that can fold), the normal wash is evaluated on the
    right strips only; a left strip gets the value of its mirror strip.
    *GK* and *GO* are the ground planes of the cases on the GPU (made from
    *grounds* when not given).
    """
    from ventorum.aero.loads import _trefftz_core_data

    k = dl.k
    K, n_s = strip_gamma.shape
    with_img = all(g is not None for g in grounds) and K > 0
    key = ("trefftz", with_img)
    tc = dl._cache.get(key)
    if tc is None:
        rc2_arr, src_group, tg_group, tg_rc2, use_tg = _trefftz_core_data(lattice, with_img)
        tc = {
            "rc2": _f64(np.asarray(rc2_arr, dtype=float)).to(dl.T).contiguous(),
            "sgroup": _i32(src_group), "tgroup": _i32(tg_group),
            "trc2": _f64(np.asarray(tg_rc2, dtype=float)).to(dl.T).contiguous(),
            "use_tg": 1 if use_tg else 0,
            "eval_all": _i32(np.arange(n_s)),
        }
        right = np.flatnonzero(lattice.strip_is_right)
        left = np.flatnonzero(~lattice.strip_is_right)
        if right.size and lattice.can_fold_symmetry():
            pos = np.full(n_s, -1, dtype=np.int64)
            pos[right] = np.arange(right.size)
            index = pos.copy()
            index[left] = pos[lattice.strip_mirror[left]]
            tc["n_right"] = int(right.size)
            tc["eval_right"] = _i32(pos)
            tc["wn_index"] = torch.as_tensor(index, device=_dev())
            tc["tgroup_r"] = _i32(np.asarray(tg_group)[right])
            tc["trc2_r"] = _f64(np.asarray(tg_rc2, dtype=float)[right]).to(dl.T).contiguous()
        dl._cache[key] = tc
    lt = dl.lat
    half = symmetric and "eval_right" in tc
    m = tc["n_right"] if half else n_s
    if with_img and GK is None:
        GK = _f64(np.array([g.normal for g in grounds]).reshape(K, 3))
        GO = _f64(np.array([g.offset for g in grounds]).reshape(K))
    if GK is None:
        GK, GO = _const("gk1"), _const("go1")
    nv = (4 if with_img else 2) * n_s
    dev = _dev()
    f32 = torch.float32
    pre = dl.T != _F64
    Q = torch.empty((K, m, 3), dtype=_F64, device=dev)
    QN = torch.empty((K, m, 3), dtype=dl.T, device=dev)
    seg_len = torch.empty((K, n_s), dtype=_F64, device=dev)
    PV = torch.empty((K, nv, 3), dtype=_F64, device=dev)
    GV = torch.empty((K, nv), dtype=_F64, device=dev)
    Dn = torch.empty((K, 3), dtype=dl.T, device=dev)
    if pre:
        Qh, Ql = (torch.empty((K, m, 3), dtype=f32, device=dev) for _ in range(2))
        PVh, PVl = (torch.empty((K, nv, 3), dtype=f32, device=dev) for _ in range(2))
    else:
        Qh = Ql = PVh = PVl = _const("z2v")
    launch(k.trefftz_prep, (K, n_s), [
        view(WD.contiguous(), wp.vec3d), cview(lt, "te_left", wp.vec3d), cview(lt, "te_right", wp.vec3d),
        cview(lt, "cp_frac", wp.float64), view(GK, wp.vec3d), view(GO, wp.float64), 1 if with_img else 0,
        view(strip_gamma.contiguous(), wp.float64), cview(tc, "eval_right" if half else "eval_all", wp.int32),
        view(Q, wp.vec3d), view(Qh, wp.vec3f), view(Ql, wp.vec3f), view(QN, k.vec), view(seg_len, wp.float64),
        view(PV, wp.vec3d), view(PVh, wp.vec3f), view(PVl, wp.vec3f), view(GV, wp.float64), view(Dn, k.vec)])
    tgroup, trc2 = ("tgroup_r", "trc2_r") if half else ("tgroup", "trc2")
    out = torch.empty((K, m), dtype=_F64, device=dev)
    launch(k.trefftz, (K, m), [view(Qh, wp.vec3f), view(Ql, wp.vec3f), view(Q, wp.vec3d), view(QN, k.vec),
                               view(PVh, wp.vec3f), view(PVl, wp.vec3f), view(PV, wp.vec3d),
                               view(GV, wp.float64), view(Dn, k.vec),
                               cview(tc, "rc2", k.T), cview(tc, "sgroup", wp.int32), cview(tc, tgroup, wp.int32),
                               cview(tc, trc2, k.T), tc["use_tg"], view(out, wp.float64)])
    if half:
        out = out[:, tc["wn_index"]]
    return out, seg_len


# --------------------------------------------------------------------------- section polars
def check_polars(dl: DeviceLattice, lattice) -> None:
    """Raise TypeError if an airfoil of *lattice* is not supported by the GPU pipelines (linear or tabulated)."""
    polar_struct(dl, lattice.airfoils, role="all")


# --------------------------------------------------------------------------- loads
def loads(dl: DeviceLattice, lattice, G: torch.Tensor, v_points: list[torch.Tensor], Dfs: torch.Tensor,
          V: torch.Tensor, rho: torch.Tensor, alpha: torch.Tensor, WD: torch.Tensor, grounds: list,
          S_ref: float, b_ref: float, c_ref: float, ref_point, alpha_eff: torch.Tensor | None,
          use_leg: bool, extra: dict | None = None, symmetric: bool = False, GK: torch.Tensor | None = None,
          GO: torch.Tensor | None = None) -> dict:
    """Return all loads of the batch as host arrays (the equations of :func:`ventorum.aero.loads.compute_loads_batch`).

    Two kernels compute them: one per strip (forces, moments, section
    values, profile drag and section moment, Trefftz-plane drag) and one per
    case (the sums and the coefficients, and the statistics of the trust
    score). One transfer brings everything to the host.

    Parameters
    ----------
    G : (K, n_panels) tensor
        Circulation of every panel [m^2/s].
    v_points : list of (K, n_panels, 3) tensors
        Induced velocity [m/s] at the force points (and at the two leg
        mid-points with leg forces).
    Dfs, V, rho, alpha : tensors
        Free-stream direction (K, 3), speed [m/s], density [kg/m^3] and
        angle of attack [rad] of each case.
    WD : (K, 3) tensor
        Wake direction of each case.
    alpha_eff : (K, n_strips) tensor or None
        Effective section angle from the solver [rad], or None.
    use_leg : bool
        Forces on the chordwise legs (vortex lattice).
    extra : dict of (K, ...) tensors, optional
        More arrays to bring to the host in the same transfer.
    symmetric : bool
        All cases are symmetric about the x-z plane (see :func:`trefftz`).
    GK, GO : tensors, optional
        Ground planes of the cases on the GPU (see :func:`batch_data`).

    Returns
    -------
    dict
        Host arrays: ``totals`` (K, 12), ``F_total`` and ``M`` (K, 3),
        ``span`` (K, 8, n_strips), ``strip_force`` (K, n_strips, 3),
        ``w_n`` (K, n_strips), ``stats`` (K, 4) (finite, max |Cl|,
        max |alpha_eff| [deg], extrema), and the arrays of *extra* (2-D, (K, -1)).
    """
    k = dl.k
    lt = dl.lat
    K = G.shape[0]
    n_s, n_c = lattice.n_strips, lattice.n_chord
    dev = _dev()
    geo = dl._cache.get("loads_geo")
    if geo is None:
        geo = {
            "lega": _f64(lattice.a - lattice.a_te), "legb": _f64(lattice.b_te - lattice.b),
            "lmida": _f64(0.5 * (lattice.a_te + lattice.a)), "lmidb": _f64(0.5 * (lattice.b + lattice.b_te)),
            "surf0": _i32([sf.strips.start for sf in lattice.surfaces]),
            "surf1": _i32([sf.strips.stop for sf in lattice.surfaces]),
        }
        dl._cache["loads_geo"] = geo
    pol, _ = polar_struct(dl, lattice.airfoils, role="all")
    rp = np.zeros(3) if ref_point is None else np.asarray(ref_point, dtype=float)
    strip_gamma = G.reshape(K, n_s, n_c).sum(dim=2) if n_c > 1 else G
    w_n, seg_len = trefftz(dl, lattice, strip_gamma, WD, grounds, symmetric=symmetric, GK=GK, GO=GO)
    v3 = wp.vec3d
    vm = v_points[0].contiguous()
    vla = v_points[1].contiguous() if use_leg else vm
    vlb = v_points[2].contiguous() if use_leg else vm
    ae = alpha_eff.contiguous() if alpha_eff is not None else w_n
    span = torch.empty((K, 8, n_s), dtype=_F64, device=dev)
    sforce = torch.empty((K, n_s, 3), dtype=_F64, device=dev)
    part = torch.empty((K, n_s, 12), dtype=_F64, device=dev)
    w_D = view(Dfs.contiguous(), v3)
    w_V = view(V.contiguous(), wp.float64)
    w_rho = view(rho.contiguous(), wp.float64)
    w_span = view(span, wp.float64)
    w_part = view(part, wp.float64)
    launch(k.loads_strip, (K, n_s), [
        view(G.contiguous(), wp.float64), view(vm, v3), view(vla, v3), view(vlb, v3), 1 if use_leg else 0,
        w_D, w_V, w_rho, wp.vec3d(*rp),
        cview(lt, "l_vec", v3), cview(lt, "mid", v3), cview(geo, "lega", v3), cview(geo, "legb", v3),
        cview(geo, "lmida", v3), cview(geo, "lmidb", v3), int(n_c),
        cview(lt, "chord", wp.float64), cview(lt, "width", wp.float64), cview(lt, "normal", v3),
        cview(lt, "chord_dir", v3), cview(lt, "dl", v3), cview(lt, "qc_mid", v3), cview(lt, "a0", wp.float64),
        cview(lt, "alpha_L0", wp.float64), view(ae, wp.float64), 1 if alpha_eff is not None else 0, pol,
        view(w_n.contiguous(), wp.float64), view(seg_len.contiguous(), wp.float64),
        w_span, view(sforce, v3), w_part])
    tot = torch.empty((K, 22), dtype=_F64, device=dev)
    launch(k.loads_totals, (K,), [w_part, w_D, w_V, w_rho, view(alpha.contiguous(), wp.float64), float(S_ref),
                                  float(b_ref), float(c_ref), w_span, cview(geo, "surf0", wp.int32),
                                  cview(geo, "surf1", wp.int32), view(tot, wp.float64)])
    parts = [tot, span.reshape(K, -1), sforce.reshape(K, -1), w_n]
    names = [("tot", 22), ("span", 8 * n_s), ("strip_force", 3 * n_s), ("w_n", n_s)]
    for name, t in (extra or {}).items():
        parts.append(t.reshape(K, -1).to(_F64))
        names.append((name, int(t.reshape(K, -1).shape[1])))
    host = torch.cat(parts, dim=1).cpu().numpy()
    o = 0
    out = {}
    for name, size in names:
        out[name] = host[:, o:o + size]
        o += size
    tot_h = out.pop("tot")
    out["totals"] = tot_h[:, :12]
    out["F_total"] = tot_h[:, 12:15]
    out["M"] = tot_h[:, 15:18]
    out["stats"] = tot_h[:, 18:22]
    out["span"] = out["span"].reshape(K, 8, n_s)
    out["strip_force"] = out["strip_force"].reshape(K, n_s, 3)
    return out
