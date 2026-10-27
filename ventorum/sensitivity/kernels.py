# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compiled vector-Jacobian-product kernels of the adjoint gradients.

The O(n^2) parts of the differentiable reference copy
(:mod:`ventorum.sensitivity.reference`) are the system rows, the induced
velocity at the near-field load points and the normal wash at the Trefftz
points. This module gives them as PyTorch autograd functions whose forward
pass and backward pass (the vector-Jacobian product, VJP) run in a compiled
kernel backend:

* ``"numba"``, ``"cython"`` and ``"numpy"`` on the CPU
  (:mod:`ventorum.aero.vortex_vjp_numba`, :mod:`ventorum.aero.vortex_vjp_cython`
  and the numpy reference :mod:`ventorum.aero.vortex_vjp`);
* ``"warp"`` on a CUDA GPU (:mod:`ventorum.gpu.vjp_kernels`, float32 and
  float64).

The PyTorch reference copy (backend ``"torch"``, path ``"reference"``) stays
the correctness reference: every backend agrees with it to round-off.

The backend of a call comes from the ``method`` keyword of the gradient
functions: ``"reference"`` (the PyTorch copy), ``"kernels"`` (the default
compiled backend of the device) or ``"auto"`` (the ``grad_vjp`` entry of the
machine profile for the device and the size class; without a profile the
default compiled backend: Numba on the CPU, Warp on the GPU).
:func:`set_vjp_backend` forces a backend for ``"auto"`` and ``"kernels"``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ventorum.aero.vortex_vjp import row_block_edges, row_blocks

try:
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    _HAVE_TORCH = False

CPU_BACKENDS = ("numba", "cython", "numpy")
GPU_BACKENDS = ("warp",)
BACKENDS = CPU_BACKENDS + GPU_BACKENDS + ("torch",)
METHODS = ("auto", "kernels", "reference")

_forced: str | None = None


def set_vjp_backend(name: str | None) -> None:
    """Force the VJP backend of ``method="auto"`` and ``"kernels"``; None removes the force.

    Parameters
    ----------
    name : str or None
        One of ``"numba"``, ``"cython"``, ``"numpy"``, ``"warp"``, ``"torch"``,
        or None.
    """
    global _forced
    if name is None:
        _forced = None
        return
    key = str(name).lower()
    if key not in BACKENDS:
        raise ValueError(f"Unknown VJP backend {name!r}; use one of {BACKENDS} or None.")
    _forced = key


def get_forced_backend() -> str | None:
    """Return the forced VJP backend, or None."""
    return _forced


def _device_key(device: Any) -> str:
    return "gpu" if str(device).startswith("cuda") else "cpu"


def available(backend: str, device: Any = "cpu") -> bool:
    """Return True when *backend* can run on *device* (``"cpu"`` or ``"cuda"``)."""
    dev = _device_key(device)
    if backend == "torch":
        return _HAVE_TORCH
    if dev == "cpu":
        if backend == "numpy":
            return True
        if backend == "numba":
            from ventorum.aero import vortex_vjp_numba as nb

            return bool(nb.HAVE_NUMBA)
        if backend == "cython":
            return _cython() is not None
        return False
    if backend == "warp":
        return _warp_ok()
    return False


def _cython():  # type: ignore[no-untyped-def]
    """Return the compiled Cython VJP module, or None when it is not built."""
    try:
        from ventorum.aero import vortex_vjp_cython as cy
    except ImportError:
        return None
    return cy


_warp_state: bool | None = None


def _warp_ok() -> bool:
    """Return True when the Warp VJP kernels can run (CUDA, warp-lang)."""
    global _warp_state
    if _warp_state is None:
        ok = False
        if _HAVE_TORCH and torch.cuda.is_available():
            try:
                import warp  # noqa: F401

                ok = True
            except ImportError:
                ok = False
        _warp_state = ok
    return _warp_state


def default_backend(device: Any) -> str:
    """Return the built-in default VJP backend of *device*.

    Numba on the CPU (else Cython, else numpy); Warp on the GPU when it is
    available, else the PyTorch reference copy.
    """
    if _device_key(device) == "gpu":
        return "warp" if _warp_ok() else "torch"
    for b in CPU_BACKENDS:
        if available(b, "cpu"):
            return b
    return "numpy"


def tuned_backend(device: Any, n_panels: int | None) -> str | None:
    """Return the ``grad_vjp`` backend of the machine profile, or None.

    The profile entry is ``settings["grad_vjp"][size_class][device]`` with
    the device ``"cpu"`` or ``"gpu"``. A profile without the entry (an
    older profile) gives None.
    """
    from ventorum.hardware.profile import size_class, tuned_setting

    value = tuned_setting("grad_vjp", size_class(n_panels), _device_key(device), default=None)
    return value if isinstance(value, str) else None


def resolve(method: str, device: Any, n_panels: int | None = None) -> tuple[str, str]:
    """Return (path, backend) of a gradient call.

    Parameters
    ----------
    method : str
        ``"auto"``, ``"kernels"`` or ``"reference"``.
    device : str
        Torch device of the gradient computation.
    n_panels : int or None
        Panel count of the lattice (selects the size class of the profile).

    Returns
    -------
    tuple of str
        ``("kernels", backend)`` or ``("reference", "torch")``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown gradient method {method!r}; use one of {list(METHODS)}.")
    if method == "reference":
        return "reference", "torch"
    forced = _forced if (_forced is not None and available(_forced, device)) else None
    if method == "kernels":
        if forced is not None and forced != "torch":
            return "kernels", forced
        backend = default_backend(device)
        if backend == "torch":
            raise ValueError(f"No compiled VJP backend is available on the device {device!r}.")
        return "kernels", backend
    backend = forced
    if backend is None:
        tuned = tuned_backend(device, n_panels)
        if tuned is not None and available(tuned, device):
            backend = tuned
    if backend is None:
        backend = default_backend(device)
    if backend == "torch":
        return "reference", "torch"
    return "kernels", backend


# --------------------------------------------------------------------------
# Backend adapters: torch tensors in, torch tensors out.
# --------------------------------------------------------------------------

def _np64(t):  # type: ignore[no-untyped-def]
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)


def _npi(t):  # type: ignore[no-untyped-def]
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.int64)


def _to(x: np.ndarray, like):  # type: ignore[no-untyped-def]
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype=like.dtype, device=like.device)


class Topology:
    """Fixed integer data of one horseshoe product (torch int64 tensors).

    Attributes
    ----------
    src_group, tg_group : torch.Tensor
        Core groups of the sources (n,) and of the targets (m,).
    own, edge : torch.Tensor
        Per target the panel of its bound segment and its edge line, or -1.
    src_panel, src_edge_a, src_edge_b : torch.Tensor
        Topology ids of the sources (-2 for images).
    col : torch.Tensor
        Unknown of each source (n,).
    n_u : int
        Number of unknowns.
    use_tg : bool
        Apply the cross-group target core.
    kink, src_edge_ka, src_edge_kb : torch.Tensor or None
        Flags (n,) of the sources whose legs have a node (see
        :class:`ventorum.aero.vortex.HorseshoeSet`), and the edge ids of
        their aft segments (n,). None when no source has a node.
    leg_next : torch.Tensor or None
        Chain flags (n,) of the leg chains (see ``leg_next`` in
        :class:`ventorum.aero.vortex.HorseshoeSet`), or None without chains.
    """

    def __init__(self, src_group, tg_group, own, edge, src_panel, src_edge_a, src_edge_b, col,
                 n_u: int, use_tg: bool, m: int, device, kink=None, src_edge_ka=None,
                 src_edge_kb=None, leg_next=None) -> None:  # type: ignore[no-untyped-def]
        def _full(x):  # type: ignore[no-untyped-def]
            if x is None:
                return torch.full((m,), -1, dtype=torch.int64, device=device)
            return x.to(dtype=torch.int64)

        n = int(src_panel.shape[0])
        self.src_group = src_group.to(torch.int64) if src_group is not None else torch.zeros(
            n, dtype=torch.int64, device=device)
        self.tg_group = tg_group.to(torch.int64) if tg_group is not None else torch.zeros(
            m, dtype=torch.int64, device=device)
        self.own = _full(own)
        self.edge = _full(edge)
        self.src_panel = src_panel.to(torch.int64)
        self.src_edge_a = src_edge_a.to(torch.int64)
        self.src_edge_b = src_edge_b.to(torch.int64)
        self.col = col.to(torch.int64) if col is not None else torch.arange(
            n, dtype=torch.int64, device=device)
        self.n_u = int(n_u)
        self.use_tg = bool(use_tg)
        self.kink = None if kink is None else kink.to(torch.int64)
        self.src_edge_ka = None if src_edge_ka is None else src_edge_ka.to(torch.int64)
        self.src_edge_kb = None if src_edge_kb is None else src_edge_kb.to(torch.int64)
        self.leg_next = None if leg_next is None or not bool(leg_next.any()) else leg_next.to(torch.int64)
        self._np: tuple | None = None
        self._np_kink: tuple | None = None

    def numpy(self) -> tuple:
        """Return (src_group, tg_group, own, src_panel, edge, src_edge_a, src_edge_b, col) as int64 arrays."""
        if self._np is None:
            self._np = tuple(_npi(x) for x in (self.src_group, self.tg_group, self.own, self.src_panel,
                                               self.edge, self.src_edge_a, self.src_edge_b, self.col))
        return self._np

    def numpy_kink(self) -> tuple | None:
        """Return (flags as int8, src_edge_ka, src_edge_kb) as arrays, or None without leg nodes."""
        if self.kink is None:
            return None
        if self._np_kink is None:
            self._np_kink = (np.ascontiguousarray(_npi(self.kink), dtype=np.int8),
                             _npi(self.src_edge_ka), _npi(self.src_edge_kb))
        return self._np_kink

    def numpy_chain(self) -> np.ndarray | None:
        """Return the chain flags as an int8 array, or None without leg chains."""
        if self.leg_next is None:
            return None
        return np.ascontiguousarray(_npi(self.leg_next), dtype=np.int8)


def _cpu_impl(backend: str):  # type: ignore[no-untyped-def]
    """Return (forward, vjp, trefftz_forward, trefftz_vjp) of a CPU backend on float64 arrays."""
    if backend == "numpy":
        from ventorum.aero import vortex_vjp as m

        def _kink(kink):  # type: ignore[no-untyped-def]
            # (a_k, b_k, src_edge_ka, src_edge_kb, flags[, a_e, b_e, leg_next]) of vortex_vjp.
            if kink is None:
                return None
            return (kink[0], kink[1], kink[3], kink[4], kink[2]) + tuple(kink[5:])

        def fwd(mode, P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, n_u, use_tg,
                kink=None):
            return m.horseshoe_forward(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb,
                                       s, col, n_u, mode, use_tg, _kink(kink))

        def vjp(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W, use_tg,
                kink=None):
            return m.horseshoe_vjp(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb,
                                   s, col, W, use_tg, _kink(kink))

        return fwd, vjp, m.trefftz_forward, m.trefftz_vjp
    if backend == "numba":
        from ventorum.aero import vortex_vjp_numba as m

        def fwd(mode, P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, n_u, use_tg,
                kink=None):
            if kink is None:
                return m.horseshoe_forward_kernel(mode, P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge,
                                                  sea, seb, s, col, int(n_u), bool(use_tg))
            return m.horseshoe_forward_kink_kernel(mode, P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp,
                                                   edge, sea, seb, s, col, int(n_u), bool(use_tg), *kink)

        def vjp(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W, use_tg,
                kink=None):
            use_w = W is not None
            Wc = W if use_w else np.zeros((1, 1))
            if kink is None:
                return m.horseshoe_vjp_kernel(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb,
                                              s, col, Wc, use_w, bool(use_tg), m.blocks(P.shape[0]))
            return m.horseshoe_vjp_kink_kernel(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea,
                                               seb, s, col, Wc, use_w, bool(use_tg), m.blocks(P.shape[0]), *kink)

        def tfwd(q, qn, p, gamma, d, rc2, tgrc2, sg, tg, use_tg):
            return m.trefftz_forward_kernel(q, qn, p, gamma, d, rc2, tgrc2, sg, tg, bool(use_tg))

        def tvjp(q, qn, p, gamma, d, rc2, tgrc2, sg, tg, use_tg, gw):
            return m.trefftz_vjp_kernel(q, qn, p, gamma, d, rc2, tgrc2, sg, tg, bool(use_tg), gw,
                                        m.blocks(q.shape[0]))

        return fwd, vjp, tfwd, tvjp
    if backend == "cython":
        return _cython_impl()
    raise ValueError(f"Unknown CPU VJP backend {backend!r}.")


_CY_MIN_ROWS = 8  # smallest block of rows for one Python thread
_cy_pool: ThreadPoolExecutor | None = None
_cy_pool_lock = threading.Lock()


def _cython_pool() -> ThreadPoolExecutor:
    """Return the shared thread pool for the Cython row blocks without OpenMP (created on first use)."""
    global _cy_pool
    if _cy_pool is None:
        with _cy_pool_lock:
            if _cy_pool is None:
                from ventorum.utils.parallel import cpu_cores

                _cy_pool = ThreadPoolExecutor(max_workers=cpu_cores(), thread_name_prefix="ventorum-vjp")
    return _cy_pool


def _row_edges(m: int, threads: int) -> np.ndarray | None:
    """Return the row-block edges of a call on Python threads, or None for one block."""
    blocks = min(int(threads), int(m) // _CY_MIN_ROWS)
    if blocks <= 1:
        return None
    return np.linspace(0, m, blocks + 1).astype(int)


def _cython_impl():  # type: ignore[no-untyped-def]
    """Return (forward, vjp, trefftz_forward, trefftz_vjp) of the Cython backend.

    With OpenMP each kernel splits its rows itself (the kernel threads of
    the calling thread). Without OpenMP (the default macOS build, see
    setup.py) the rows are split into blocks that run on Python threads;
    the kernels release the GIL. The source derivatives of the blocks are
    added in block order. The VJP blocks depend only on the row count, so
    the result is bitwise the same for every thread count.
    """
    from ventorum.utils.parallel import kernel_threads

    cy = _cython()
    omp = bool(cy.openmp_enabled())

    def _nblocks(m: int, threads: int) -> int:
        # The partition depends only on the row count: bitwise the same
        # result for every thread count (*threads* is not used).
        return row_blocks(m)

    def _vjp_edges(m: int) -> np.ndarray | None:
        # Python-thread blocks of the VJP: the same partition as the OpenMP
        # kernel, so that both builds add the same block sums in the same order.
        edges = row_block_edges(m)
        return None if len(edges) <= 2 else edges

    def _kw(kink):  # type: ignore[no-untyped-def]
        if kink is None:
            return {}
        kw = {"ak": kink[0], "bk": kink[1], "kfl": kink[2], "seka": kink[3], "sekb": kink[4]}
        if len(kink) > 5:
            kw.update(ae=kink[5], be=kink[6], nxt=kink[7])
        return kw

    def fwd(mode, P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, n_u, use_tg,
            kink=None):
        threads = kernel_threads()
        args = (a, b, at, bt, d, rc2s)
        kw = _kw(kink)

        def call(i0, i1, nt):  # type: ignore[no-untyped-def]
            return cy.horseshoe_forward(int(mode), P[i0:i1], X[i0:i1], *args, tgrc2[i0:i1], sg, tg[i0:i1],
                                        own[i0:i1], sp, edge[i0:i1], sea, seb, s, col, int(n_u),
                                        bool(use_tg), nt, **kw)

        m = P.shape[0]
        edges = None if omp else _row_edges(m, threads)
        if edges is None:
            return call(0, m, threads if omp else 1)
        parts = list(_cython_pool().map(lambda k: call(edges[k], edges[k + 1], 1), range(len(edges) - 1)))
        return np.concatenate(parts, axis=0)

    def vjp(P, X, a, b, at, bt, d, rc2s, tgrc2, sg, tg, own, sp, edge, sea, seb, s, col, W, use_tg,
            kink=None):
        threads = kernel_threads()
        use_w = W is not None
        Wc = W if use_w else np.zeros((1, 1))
        args = (a, b, at, bt, d, rc2s)
        kw = _kw(kink)
        n_out = 10 if kink is None else (12 if len(kink) <= 5 else 14)

        def call(i0, i1, nb, nt):  # type: ignore[no-untyped-def]
            Wr = Wc[i0:i1] if use_w else Wc
            return cy.horseshoe_vjp(P[i0:i1], X[i0:i1], *args, tgrc2[i0:i1], sg, tg[i0:i1], own[i0:i1], sp,
                                    edge[i0:i1], sea, seb, s, col, Wr, use_w, bool(use_tg), nb, nt, **kw)

        m = P.shape[0]
        edges = None if omp else _vjp_edges(m)
        if edges is None:
            return call(0, m, _nblocks(m, threads), threads if omp else 1)
        parts = list(_cython_pool().map(lambda k: call(edges[k], edges[k + 1], 1, 1), range(len(edges) - 1)))
        rows = [np.concatenate([p[k] for p in parts], axis=0) for k in range(3)]
        srcs = []
        for k in range(3, n_out):
            acc = parts[0][k].copy()
            for p in parts[1:]:
                acc += p[k]
            srcs.append(acc)
        return tuple(rows + srcs)

    def tfwd(q, qn, p, gamma, d, rc2, tgrc2, sg, tg, use_tg):
        threads = kernel_threads()
        m = q.shape[0]

        def call(i0, i1, nt):  # type: ignore[no-untyped-def]
            return cy.trefftz_forward(q[i0:i1], qn[i0:i1], p, gamma, d, rc2, tgrc2[i0:i1], sg, tg[i0:i1],
                                      bool(use_tg), nt)

        edges = None if omp else _row_edges(m, threads)
        if edges is None:
            return call(0, m, threads if omp else 1)
        return np.concatenate(list(_cython_pool().map(lambda k: call(edges[k], edges[k + 1], 1),
                                                      range(len(edges) - 1))))

    def tvjp(q, qn, p, gamma, d, rc2, tgrc2, sg, tg, use_tg, gw):
        threads = kernel_threads()
        m = q.shape[0]

        def call(i0, i1, nb, nt):  # type: ignore[no-untyped-def]
            return cy.trefftz_vjp(q[i0:i1], qn[i0:i1], p, gamma, d, rc2, tgrc2[i0:i1], sg, tg[i0:i1],
                                  bool(use_tg), gw[i0:i1], nb, nt)

        edges = None if omp else _vjp_edges(m)
        if edges is None:
            return call(0, m, _nblocks(m, threads), threads if omp else 1)
        parts = list(_cython_pool().map(lambda k: call(edges[k], edges[k + 1], 1, 1), range(len(edges) - 1)))
        rows = [np.concatenate([p[k] for p in parts], axis=0) for k in range(3)]
        srcs = []
        for k in range(3, 7):
            acc = parts[0][k].copy()
            for p in parts[1:]:
                acc += p[k]
            srcs.append(acc)
        return tuple(rows + srcs)

    return fwd, vjp, tfwd, tvjp


def _gpu_impl(backend: str):  # type: ignore[no-untyped-def]
    if backend != "warp":
        raise ValueError(f"Unknown GPU VJP backend {backend!r}.")
    from ventorum.gpu import vjp_kernels as m

    return m


class _HorseshoeFn(torch.autograd.Function if _HAVE_TORCH else object):  # type: ignore[misc]
    """Horseshoe product (system rows or induced velocity) with a compiled VJP.

    Mode 0: ``A[i, col[j]] += s[j] X[i] . v_ij`` (output (m, n_u)).
    Mode 1: ``V[i] += s[j] v_ij`` (output (m, 3)); X is not used.
    """

    @staticmethod
    def forward(ctx, backend, topo, mode, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s,
                a_k=None, b_k=None, a_e=None, b_e=None):  # type: ignore[no-untyped-def]
        """Evaluate the product with the compiled forward kernel.

        *a_k*, *b_k* are the leg nodes, or None; *a_e*, *b_e* the ends of
        the own leg segments of a set with leg chains (flags in *topo*), or
        None.
        """
        ctx.backend = backend
        ctx.topo = topo
        ctx.mode = mode
        ctx.save_for_backward(P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, a_k, b_k, a_e, b_e)
        return _hs_forward(backend, topo, mode, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, a_k, b_k,
                           a_e, b_e)

    @staticmethod
    def backward(ctx, gout):  # type: ignore[no-untyped-def]
        """Return the input gradients from one compiled VJP call."""
        P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, a_k, b_k, a_e, b_e = ctx.saved_tensors
        if ctx.mode == 0:
            g = _hs_vjp(ctx.backend, ctx.topo, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s,
                        gout.contiguous(), a_k, b_k, a_e, b_e)
            gX = g[1]
        else:
            g = _hs_vjp(ctx.backend, ctx.topo, P, gout.contiguous(), a, b, a_te, b_te, d, rc2_src,
                        tg_rc2, s, None, a_k, b_k, a_e, b_e)
            gX = None
        gP, _, gtg, ga, gb, gat, gbt, gd, grc, gs = g[:10]
        gak, gbk = (g[10], g[11]) if len(g) > 10 else (None, None)
        gae, gbe = (g[12], g[13]) if len(g) > 12 else (None, None)
        need = ctx.needs_input_grad
        out = [gP, gX, ga, gb, gat, gbt, gd, grc, gtg if tg_rc2 is not None else None, gs, gak, gbk, gae, gbe]
        out = [o if (o is not None and need[3 + k]) else None for k, o in enumerate(out)]
        return (None, None, None, *out)


def _np_kink(topo, a_k, b_k, a_e=None, b_e=None):  # type: ignore[no-untyped-def]
    """Return the leg nodes of a CPU call, or None.

    The tuple is ``(a_k, b_k, flags, src_edge_ka, src_edge_kb)``, with
    ``(a_e, b_e, leg_next)`` added for a set with leg chains.
    """
    kt = topo.numpy_kink()
    if kt is None or a_k is None:
        return None
    nxt = topo.numpy_chain()
    if nxt is None or a_e is None:
        return (_np64(a_k), _np64(b_k)) + kt
    return (_np64(a_k), _np64(b_k)) + kt + (_np64(a_e), _np64(b_e), nxt)


def _hs_forward(backend, topo, mode, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s,
                a_k=None, b_k=None, a_e=None, b_e=None):  # type: ignore[no-untyped-def]
    m = P.shape[0]
    if backend in GPU_BACKENDS:
        impl = _gpu_impl(backend)
        return impl.horseshoe_forward(mode, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, topo, a_k, b_k,
                                      a_e, b_e)
    fwd, _, _, _ = _cpu_impl(backend)
    sg, tg, own, sp, edge, sea, seb, col = topo.numpy()
    Xn = _np64(X) if X is not None else np.zeros((m, 3))
    tg2 = _np64(tg_rc2) if tg_rc2 is not None else np.zeros(m)
    out = fwd(mode, _np64(P), Xn, _np64(a), _np64(b), _np64(a_te), _np64(b_te), _np64(d), _np64(rc2_src),
              tg2, sg, tg, own, sp, edge, sea, seb, _np64(s), col, topo.n_u,
              bool(topo.use_tg and tg_rc2 is not None), kink=_np_kink(topo, a_k, b_k, a_e, b_e))
    return _to(out, P)


def _hs_vjp(backend, topo, P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, W,
            a_k=None, b_k=None, a_e=None, b_e=None):  # type: ignore[no-untyped-def]
    m = P.shape[0]
    if backend in GPU_BACKENDS:
        impl = _gpu_impl(backend)
        return impl.horseshoe_vjp(P, X, a, b, a_te, b_te, d, rc2_src, tg_rc2, s, W, topo, a_k, b_k, a_e, b_e)
    _, vjp, _, _ = _cpu_impl(backend)
    sg, tg, own, sp, edge, sea, seb, col = topo.numpy()
    tg2 = _np64(tg_rc2) if tg_rc2 is not None else np.zeros(m)
    g = vjp(_np64(P), _np64(X), _np64(a), _np64(b), _np64(a_te), _np64(b_te), _np64(d), _np64(rc2_src),
            tg2, sg, tg, own, sp, edge, sea, seb, _np64(s), col, None if W is None else _np64(W),
            bool(topo.use_tg and tg_rc2 is not None), kink=_np_kink(topo, a_k, b_k, a_e, b_e))
    return tuple(_to(x, P) for x in g)


class _TrefftzFn(torch.autograd.Function if _HAVE_TORCH else object):  # type: ignore[misc]
    """Normal wash of 2-D point vortices at Trefftz-plane points with a compiled VJP."""

    @staticmethod
    def forward(ctx, backend, groups, q, qn, p, gamma, d, rc2, tg_rc2):  # type: ignore[no-untyped-def]
        """Evaluate the normal wash with the compiled forward kernel."""
        ctx.backend = backend
        ctx.groups = groups
        ctx.save_for_backward(q, qn, p, gamma, d, rc2, tg_rc2)
        if backend in GPU_BACKENDS:
            return _gpu_impl(backend).trefftz_forward(q, qn, p, gamma, d, rc2, tg_rc2, groups)
        _, _, tfwd, _ = _cpu_impl(backend)
        sg, tg, use_tg = groups
        out = tfwd(_np64(q), _np64(qn), _np64(p), _np64(gamma), _np64(d), _np64(rc2), _np64(tg_rc2),
                   _npi(sg), _npi(tg), use_tg)
        return _to(out, q)

    @staticmethod
    def backward(ctx, gw):  # type: ignore[no-untyped-def]
        """Return the input gradients from one compiled VJP call."""
        q, qn, p, gamma, d, rc2, tg_rc2 = ctx.saved_tensors
        if ctx.backend in GPU_BACKENDS:
            g = _gpu_impl(ctx.backend).trefftz_vjp(q, qn, p, gamma, d, rc2, tg_rc2, ctx.groups,
                                                   gw.contiguous())
        else:
            _, _, _, tvjp = _cpu_impl(ctx.backend)
            sg, tg, use_tg = ctx.groups
            r = tvjp(_np64(q), _np64(qn), _np64(p), _np64(gamma), _np64(d), _np64(rc2), _np64(tg_rc2),
                     _npi(sg), _npi(tg), use_tg, _np64(gw))
            g = tuple(_to(x, q) for x in r)
        gq, gqn, gtg, gp, gg, grc, gd = g
        need = ctx.needs_input_grad
        out = [gq, gqn, gp, gg, gd, grc, gtg]
        out = [o if need[2 + k] else None for k, o in enumerate(out)]
        return (None, None, *out)


def horseshoe_rows(backend: str, topo: Topology, P, N, a, b, a_te, b_te, d, rc2_src, tg_rc2, sign,
                   a_k=None, b_k=None, a_e=None, b_e=None):  # type: ignore[no-untyped-def]
    """Return the system rows ``A[i, col[j]] += sign[j] dot(N[i], v_ij)`` (differentiable), shape (m, n_u).

    *a_k*, *b_k* are the leg nodes (n, 3) of the sources with the flags of
    *topo*, or None. *a_e*, *b_e* are the ends of the own leg segments
    (n, 3) of a set with leg chains (the chain flags of *topo*), or None.
    """
    return _HorseshoeFn.apply(backend, topo, 0, P, N, a, b, a_te, b_te, d, rc2_src, tg_rc2, sign, a_k, b_k,
                              a_e, b_e)


def horseshoe_induced(backend: str, topo: Topology, P, a, b, a_te, b_te, d, rc2_src, tg_rc2, g_src,
                      a_k=None, b_k=None, a_e=None, b_e=None):  # type: ignore[no-untyped-def]
    """Return the induced velocity ``V[i] = sum_j g_src[j] v_ij`` (differentiable), shape (m, 3).

    *a_k*, *b_k*, *a_e*, *b_e* are the leg nodes and chain ends, as in
    :func:`horseshoe_rows`.
    """
    return _HorseshoeFn.apply(backend, topo, 1, P, None, a, b, a_te, b_te, d, rc2_src, tg_rc2, g_src, a_k, b_k,
                              a_e, b_e)


def trefftz_wash(backend: str, groups: tuple, q, qn, p, gamma, d, rc2, tg_rc2):  # type: ignore[no-untyped-def]
    """Return the Trefftz normal wash (differentiable), shape (m,).

    *groups* is ``(src_group, tg_group, use_tg)`` with torch int64 tensors.
    """
    return _TrefftzFn.apply(backend, groups, q, qn, p, gamma, d, rc2, tg_rc2)
