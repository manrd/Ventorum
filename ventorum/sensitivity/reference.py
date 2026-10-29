# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Differentiable reference copy of the lattice solvers for adjoint gradients.

The model rebuilds the lattice from the edge nodes of each surface, the
system rows (or the residual and the Newton Jacobian of the nonlinear
lifting line) and the outputs in PyTorch, with the same formulas as the
production code: control deflections, body rates and the interpolation of
tabulated polars included. The mesh topology is fixed. The O(n^2) work (system rows,
load points, Trefftz points) runs in row chunks with a memory budget, or,
when ``build["vjp_backend"]`` names a backend, in the compiled forward and
vector-Jacobian-product kernels of :mod:`ventorum.sensitivity.kernels`.

References
----------
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001: the Biot-Savart law of a straight vortex segment
  and of a semi-infinite vortex, and the horseshoe vortex.
* W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic
  lifting-line theory", Journal of Aircraft 37(4), 2000, pp. 662-670.
* M. Drela, "Flight Vehicle Aerodynamics", MIT Press, 2014: the
  Trefftz-plane induced drag and the near-field forces.
* J. R. R. A. Martins and A. Ning, "Engineering Design Optimization",
  Cambridge University Press, 2021, ch. 6: the discrete adjoint method.
"""

from __future__ import annotations

import math
import os

import numpy as np

try:
    import torch

    _HAVE_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    _HAVE_TORCH = False

# Default chunk budget [values]: the budget of ventorum.aero.vortex_torch.
# Here the budget counts all values that one chunk holds in its forward and
# backward pass, so it bounds the memory of the gradient computation.
_DEFAULT_CHUNK_VALUES = 2.0e7

# Values per (target, source) pair that one chunk of the horseshoe kernel
# holds in its forward and backward pass (measured: about 200 with the
# PyTorch CPU backend; the largest single temporary has 3).
_PAIR_VALUES = 200

_MIRROR = (1.0, -1.0, 1.0)


class _TorchOps:
    """Array operations of ``geometry.lattice.camber_span_field`` for torch tensors."""

    @staticmethod
    def cross(u, v):  # type: ignore[no-untyped-def]
        u, v = torch.broadcast_tensors(u, v)
        return torch.linalg.cross(u, v, dim=-1)

    @staticmethod
    def norm(v):  # type: ignore[no-untyped-def]
        return torch.linalg.norm(v, dim=-1)

    @staticmethod
    def yz(v):  # type: ignore[no-untyped-def]
        return torch.cat((torch.zeros_like(v[..., :1]), v[..., 1:]), dim=-1)

    @staticmethod
    def central(q):  # type: ignore[no-untyped-def]
        # numpy.gradient with unit spacing: one-sided at the ends.
        n_pts = q.shape[0]
        if n_pts == 1:
            return torch.zeros_like(q)
        if n_pts == 2:
            g = q[1:2] - q[0:1]
            return torch.cat((g, g), dim=0)
        return torch.cat((q[1:2] - q[0:1], 0.5 * (q[2:] - q[:-2]), q[-1:] - q[-2:-1]), dim=0)

    @staticmethod
    def cumsum0(w):  # type: ignore[no-untyped-def]
        return torch.cat((torch.zeros_like(w[:1]), torch.cumsum(w, dim=0)))

    @staticmethod
    def detach(x):  # type: ignore[no-untyped-def]
        return x.detach() if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    @staticmethod
    def atan2(y, x):  # type: ignore[no-untyped-def]
        return torch.atan2(y, x)

    @staticmethod
    def sin(x):  # type: ignore[no-untyped-def]
        return torch.sin(x)

    @staticmethod
    def where(c, a, b):  # type: ignore[no-untyped-def]
        return torch.where(c, a, b)

    @staticmethod
    def clip(x, lo, hi):  # type: ignore[no-untyped-def]
        return torch.clamp(x, lo, hi)

    @staticmethod
    def abs(x):  # type: ignore[no-untyped-def]
        return torch.abs(x)

    @staticmethod
    def minimum(a, b):  # type: ignore[no-untyped-def]
        return torch.minimum(a, b)

    @staticmethod
    def maximum(a, b):  # type: ignore[no-untyped-def]
        return torch.maximum(a, b)


def require_torch() -> None:
    """Raise when PyTorch is not available."""
    if not _HAVE_TORCH:
        raise RuntimeError("The gradient module needs the torch package.")


def chunk_budget() -> float:
    """Return the chunk budget [values].

    The environment variable ``VENTORUM_GRAD_CHUNK`` wins when it holds a
    positive number. Else the tuned value of the machine profile applies
    (key ``"chunk_mib"``, ``"gradient"``, in MiB; one value takes 8 bytes).
    Else the default ``_DEFAULT_CHUNK_VALUES`` applies.
    """
    raw = os.environ.get("VENTORUM_GRAD_CHUNK", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            pass
        else:
            if value > 0.0:
                return value
    try:
        from ventorum.hardware.profile import tuned_setting

        mib = float(tuned_setting("chunk_mib", "gradient", default=None))
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_CHUNK_VALUES
    if mib > 0.0:
        return mib * 1024 * 1024 / 8.0
    return _DEFAULT_CHUNK_VALUES


def chunk_rows(m: int, n: int) -> int:
    """Return the row count of one chunk for *m* rows and *n* sources.

    One chunk holds about ``_PAIR_VALUES`` values per (row, source) pair in
    its forward and backward pass; the chunk keeps that total below the
    budget of :func:`chunk_budget`.
    """
    per_row = _PAIR_VALUES * max(1, int(n))
    return max(1, min(int(m), int(chunk_budget() // per_row)))


def rod_rot(v: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate rows of *v* about unit *axis* rows by *angle* [rad] (Rodrigues)."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    if c.dim() == 0:
        c = c.expand(v.shape[0])
        s = s.expand(v.shape[0])
    kdv = (axis * v).sum(-1)
    cr = torch.stack(
        (
            axis[:, 1] * v[:, 2] - axis[:, 2] * v[:, 1],
            axis[:, 2] * v[:, 0] - axis[:, 0] * v[:, 2],
            axis[:, 0] * v[:, 1] - axis[:, 1] * v[:, 0],
        ),
        dim=1,
    )
    return v * c[:, None] + cr * s[:, None] + axis * (kdv * (1.0 - c))[:, None]


def _length_floor(dtype: torch.dtype) -> float:
    """Return the floor of the squared distances in the kernels [m^2].

    The floor is 1e-300, or the smallest normal number of *dtype* when that
    is larger. In float32 the value 1e-300 rounds to zero, and a target on
    a segment end point then gives 0/0 in the unit vectors.
    """
    return max(1e-300, float(torch.finfo(dtype).tiny))


def seg_vel(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rc2: torch.Tensor) -> torch.Tensor:
    """Velocity of unit finite segments A -> B at P (differentiable).

    The core acts on the distance from P to the finite segment, as in
    :func:`ventorum.aero.vortex.segment_velocity`. Over the segment the
    denominator is ``|r1 x r2|^2 + rc2 |r0|^2``. Before A or after B the
    exact Biot-Savart law (Katz and Plotkin, 2001) is multiplied by the
    Scully factor ``D^2 / (D^2 + rc2)`` (Scully, 1975; Vatistas, Kozel and
    Mih, 1991, n = 1), with ``D`` the distance to the nearest end point.
    The unused branch of each ``torch.where`` gets safe denominators, so
    that autograd gives no NaN gradient. Lengths in m, rc2 in m^2.
    """
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = B[None, :, :] - A[None, :, :]
    cx = r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]
    cy = r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]
    cz = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz
    r0sq = r0[..., 0] ** 2 + r0[..., 1] ** 2 + r0[..., 2] ** 2
    # The floor keeps the unit vectors finite when P is on an end point.
    floor = _length_floor(P.dtype)
    d1 = (r1 * r1).sum(-1) + floor
    d2 = (r2 * r2).sum(-1) + floor
    n1 = torch.sqrt(d1)
    n2 = torch.sqrt(d2)
    t1 = r1 / n1[..., None]
    t2 = r2 / n2[..., None]
    proj_in = (
        r0[..., 0] * (t1[..., 0] - t2[..., 0])
        + r0[..., 1] * (t1[..., 1] - t2[..., 1])
        + r0[..., 2] * (t1[..., 2] - t2[..., 2])
    )
    # The same projection without cancellation where r1 . r2 > 0:
    # r0 . (t1 - t2) = |r1 x r2|^2 (n1 + n2) / (n1 n2 (n1 n2 + r1 . r2)).
    # The direct form loses digits there (t1 and t2 are almost equal).
    dot12 = (r1 * r2).sum(-1)
    n12 = n1 * n2
    e = n12 + dot12
    out_side = dot12 > 0.0
    den_os = torch.where(out_side, n12 * e, torch.ones_like(n12))
    proj_os = cr2 * (n1 + n2) / den_os
    proj = torch.where(out_side, proj_os, proj_in)
    denom = cr2 + rc2 * r0sq
    ok = denom > floor
    safe = torch.where(ok, denom, torch.ones_like(denom))
    k_in = torch.where(ok, (1.0 / (4.0 * math.pi)) * proj / safe, torch.zeros_like(proj))
    # Outside the segment: before A (r0 . r1 < 0) or after B (r0 . r2 > 0).
    # Exact law (no cancellation) times D^2 / (D^2 + rc2).
    before = (r0 * r1).sum(-1) < 0.0
    after = (r0 * r2).sum(-1) > 0.0
    outside = before | after
    D2 = torch.where(before, d1, d2)
    ok_out = outside & (e > floor)
    den_out = torch.where(ok_out, n12 * e, torch.ones_like(e))
    k_out = torch.where(
        ok_out,
        (1.0 / (4.0 * math.pi)) * (n1 + n2) / den_out * (D2 / (D2 + rc2)),
        torch.zeros_like(e),
    )
    k = torch.where(outside, k_out, k_in)
    return torch.stack((cx * k, cy * k, cz * k), dim=-1)


def ray_vel(P: torch.Tensor, A: torch.Tensor, d: torch.Tensor, rc2: torch.Tensor) -> torch.Tensor:
    """Velocity of unit semi-infinite lines from A along d at P (differentiable).

    The core acts on the distance from P to the line, as in
    :func:`ventorum.aero.vortex.ray_velocity`. In front of A (``d . r >= 0``)
    the denominator is ``|d x r|^2 + rc2``. Behind A the exact law is
    multiplied by the Scully factor ``D^2 / (D^2 + rc2)`` with ``D = |r|``.
    The unused branch of each ``torch.where`` gets safe denominators, so
    that autograd gives no NaN gradient. Lengths in m, rc2 in m^2.
    """
    r = P[:, None, :] - A[None, :, :]
    dd = d[None, :, :]
    cx = dd[..., 1] * r[..., 2] - dd[..., 2] * r[..., 1]
    cy = dd[..., 2] * r[..., 0] - dd[..., 0] * r[..., 2]
    cz = dd[..., 0] * r[..., 1] - dd[..., 1] * r[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz
    floor = _length_floor(P.dtype)
    D2 = (r * r).sum(-1) + floor
    rn = torch.sqrt(D2)
    dr = (dd * r).sum(-1)
    denom = cr2 + rc2
    ok = denom > floor
    safe = torch.where(ok, denom, torch.ones_like(denom))
    k_in = torch.where(ok, (1.0 / (4.0 * math.pi)) * (1.0 + dr / rn) / safe, torch.zeros_like(denom))
    # Behind the start point: exact law (no cancellation) times D^2 / (D^2 + rc2).
    behind = dr < 0.0
    e = rn * (rn - dr)
    ok_out = behind & (e > floor)
    den_out = torch.where(ok_out, e, torch.ones_like(e))
    k_out = torch.where(ok_out, (1.0 / (4.0 * math.pi)) / den_out * (D2 / (D2 + rc2)), torch.zeros_like(e))
    k = torch.where(behind, k_out, k_in)
    return torch.stack((cx * k, cy * k, cz * k), dim=-1)


def horseshoe_vel(
    P: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_te: torch.Tensor,
    b_te: torch.Tensor,
    d: torch.Tensor,
    rc2: torch.Tensor,
) -> torch.Tensor:
    """Unit-circulation velocity of bent horseshoes at P (differentiable)."""
    fixed = seg_vel(P, a, b, rc2) + seg_vel(P, a_te, a, rc2) + seg_vel(P, b, b_te, rc2)
    return fixed + ray_vel(P, b_te, d, rc2) - ray_vel(P, a_te, d, rc2)


def pairwise_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum *x* along *dim* by recursive pairwise addition (differentiable).

    Pairwise addition grows the rounding error with log2(n) instead of n.
    """
    d = dim if dim >= 0 else x.dim() + dim
    n = x.shape[d]
    if n <= 2:
        return x.sum(d)
    half = n // 2
    a = pairwise_sum(x.narrow(d, 0, half).contiguous(), d)
    b = pairwise_sum(x.narrow(d, half, n - half).contiguous(), d)
    return a + b


def kahan_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum *x* along *dim* with compensated (Kahan) summation (differentiable)."""
    d = dim if dim >= 0 else x.dim() + dim
    moved = x.movedim(d, 0)
    rest = moved.shape[1:]
    flat = moved.reshape(moved.shape[0], -1)
    s = torch.zeros(flat.shape[1], dtype=x.dtype, device=x.device)
    c = torch.zeros_like(s)
    for j in range(flat.shape[0]):
        y = flat[j] - c
        t = s + y
        c = (t - s) - y
        s = t
    return s.reshape(rest)


def pp_from_scipy(poly: object, device: object, dtype: object) -> dict:
    """Return the breakpoints and the cubic coefficients of a scipy piecewise polynomial as tensors.

    Parameters
    ----------
    poly : scipy.interpolate.PPoly
        Piecewise cubic (for example the ``PchipInterpolator`` of a
        tabulated polar), with ``x`` (m,) and ``c`` (4, m - 1) in the local
        power basis: ``f(x) = sum_k c[k, i] (x - x_i)^(3 - k)``.
    device, dtype
        Torch device and dtype of the tensors.

    Returns
    -------
    dict
        ``"x"`` (m,), ``"c"`` (4, m - 1), and the table limits ``"lo"``,
        ``"hi"`` [rad] as floats.
    """
    import numpy as np

    xb = np.asarray(poly.x, dtype=float)  # type: ignore[attr-defined]
    c = np.asarray(poly.c, dtype=float)  # type: ignore[attr-defined]
    if c.shape[0] != 4:
        raise ValueError(f"Expected a piecewise cubic, got order {c.shape[0] - 1}.")
    return {"x": torch.as_tensor(xb.copy(), dtype=dtype, device=device),
            "c": torch.as_tensor(c.copy(), dtype=dtype, device=device),
            "lo": float(xb[0]), "hi": float(xb[-1])}


def _pp_interval(pp: dict, xc: torch.Tensor) -> torch.Tensor:
    """Return the interval index of the clamped points *xc* (no graph)."""
    xb = pp["x"]
    i = torch.searchsorted(xb, xc.detach().contiguous(), right=True) - 1
    return torch.clamp(i, 0, xb.shape[0] - 2)


def _clamp_closed(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clamp *x* to the closed interval [lo, hi] with a fixed derivative convention.

    The derivative is 1 inside the closed interval (the bounds included)
    and 0 outside it. ``torch.clamp`` does not document its derivative at
    a bound, and it changed between torch versions (1 in torch 2.6, 0 in
    torch 2.14 for float bounds). So the clamp is made from
    ``torch.where``, which passes the gradient to the selected value only.
    Both bounds are dimensionless here (lift-slope factor, table angle).
    """
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return torch.where(x < lo_t, lo_t, torch.where(x > hi_t, hi_t, x))


def _pp_clamp(pp: dict, x: torch.Tensor) -> torch.Tensor:
    """Clamp *x* to the table limits with a fixed derivative convention.

    The derivative is 1 inside the closed interval ``[lo, hi]`` (the table
    ends included) and 0 outside it. The clamp is made from
    ``torch.where`` (see :func:`_clamp_closed`), because ``torch.clamp``
    does not document its derivative at a bound, and it changed between
    torch versions (1 in torch 2.6, 0 in torch 2.14 for float bounds).
    """
    return _clamp_closed(x, pp["lo"], pp["hi"])


def pp_eval(pp: dict, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a piecewise cubic of :func:`pp_from_scipy` at *x* (differentiable).

    Outside the table the end values are kept (the angle is clamped), as in
    ``TabulatedAirfoil``. A point on a breakpoint takes the interval to its
    right (the last interval at the upper end), so its derivative is the
    one-sided derivative of that interval. The shape-preserving cubic has a
    continuous first derivative, so both sides agree to round-off. At the
    table ends the derivative is the one-sided derivative inside the table,
    the same as :func:`pp_slope`, for all torch versions.
    """
    xc = _pp_clamp(pp, x)
    i = _pp_interval(pp, xc)
    c = pp["c"]
    dx = xc - pp["x"][i]
    return ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]


def pp_slope(pp: dict, x: torch.Tensor) -> torch.Tensor:
    """Return the derivative of a piecewise cubic at *x* (no graph); zero outside the table."""
    with torch.no_grad():
        xc = _pp_clamp(pp, x)
        i = _pp_interval(pp, xc)
        c = pp["c"]
        dx = xc - pp["x"][i]
        val = (3.0 * c[0, i] * dx + 2.0 * c[1, i]) * dx + c[2, i]
        inside = (x >= pp["lo"]) & (x <= pp["hi"])
        return torch.where(inside, val, torch.zeros_like(val))


def _chain_steps(leg_next: torch.Tensor, reverse: bool) -> list[torch.Tensor]:
    """Return per chain step the indices of the sources at that step (from the chain end, or from its start)."""
    from ventorum.aero.vortex import chain_depth, chain_position

    flags = leg_next.detach().cpu().numpy() != 0
    pos = chain_depth(flags) if reverse else chain_position(flags)
    return [torch.as_tensor(np.flatnonzero(pos == t), device=leg_next.device)
            for t in range(1, int(pos.max(initial=0)) + 1)]


def _chain_suffix(x: torch.Tensor, leg_next: torch.Tensor) -> torch.Tensor:
    """Return per source (axis 1 of *x*) the sum over the sources after it on its leg chain.

    The differentiable form of :func:`ventorum.aero.vortex.chain_suffix`.
    """
    out = torch.zeros_like(x)
    for j in _chain_steps(leg_next, True):
        out = out.index_copy(1, j, x[:, j + 1] + out[:, j + 1])
    return out


def _chain_prefix(x: torch.Tensor, leg_next: torch.Tensor) -> torch.Tensor:
    """Return per panel (axis 0 of *x*) the sum over it and the panels before it on its leg chain.

    The differentiable form of :func:`ventorum.aero.vortex.chain_prefix`.
    """
    out = x
    for j in _chain_steps(leg_next, False):
        out = out.index_copy(0, j, x[j] + out[j - 1])
    return out


def _rows(x: torch.Tensor | None, i0: int, i1: int) -> torch.Tensor | None:
    """Return rows i0:i1 of *x*, or None."""
    return None if x is None else x[i0:i1]


class _ChunkedRows(torch.autograd.Function if _HAVE_TORCH else object):  # type: ignore[misc]
    """Row-chunked evaluation that keeps only its inputs for the backward pass.

    The forward pass evaluates the chunks without a graph. The backward
    pass evaluates each chunk again with a graph, takes its
    vector-Jacobian product with the output gradient of the chunk, adds it
    to the input gradients and frees the chunk graph before the next
    chunk. So the memory of the O(n^2) kernel work stays at one chunk in
    the forward and in the backward pass, also for repeated backward
    passes on the same graph.
    """

    @staticmethod
    def forward(ctx, fn, m, step, *tensors):  # type: ignore[no-untyped-def]
        """Evaluate fn(i0, i1, *tensors) for the row chunks and join the rows."""
        ctx.fn = fn
        ctx.m = m
        ctx.step = step
        ctx.save_for_backward(*tensors)
        return torch.cat([fn(i0, min(m, i0 + step), *tensors) for i0 in range(0, m, step)], dim=0)

    @staticmethod
    def backward(ctx, gout):  # type: ignore[no-untyped-def]
        """Return the input gradients, one chunk at a time."""
        tensors = ctx.saved_tensors
        need = ctx.needs_input_grad[3:]
        grads = [torch.zeros_like(t) if nd else None for t, nd in zip(tensors, need)]
        for i0 in range(0, ctx.m, ctx.step):
            i1 = min(ctx.m, i0 + ctx.step)
            with torch.enable_grad():
                leaves = [t.detach().requires_grad_(True) if nd else t.detach()
                          for t, nd in zip(tensors, need)]
                y = ctx.fn(i0, i1, *leaves)
                wanted = [leaf for leaf, nd in zip(leaves, need) if nd]
                parts = torch.autograd.grad(y, wanted, gout[i0:i1], allow_unused=True)
            it = iter(parts)
            for k, nd in enumerate(need):
                if nd:
                    g = next(it)
                    if g is not None:
                        grads[k] = grads[k] + g
            del y, leaves, wanted, parts
        return (None, None, None, *grads)


def _chunked(fn, m: int, step: int, *tensors: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Evaluate fn(i0, i1, *tensors) in row chunks of *step* rows (see _ChunkedRows)."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return _ChunkedRows.apply(fn, m, step, *tensors)
    return torch.cat([fn(i0, min(m, i0 + step), *tensors) for i0 in range(0, m, step)], dim=0)


class TorchReference:
    """Differentiable copy of one lattice case in PyTorch.

    The topology (strip counts, panel maps, chordwise fractions, mirror
    maps, core groups) is fixed from the production lattice. The inputs
    are torch tensors: the edge nodes of the defining half of each
    surface, the flow scalars, the section properties per strip and the
    reference values. The forward pass gives the system matrix, the
    right-hand side and the outputs for a fixed circulation.
    """

    def __init__(self, build: dict):
        self.b = build

    @property
    def device(self):  # type: ignore[no-untyped-def]
        """Return the torch device of the model."""
        return self.b["device"]

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        """Return the torch dtype of the model."""
        return self.b["dtype"]

    def forward(
        self,
        t: dict,
        g_unknown: torch.Tensor,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Return (outputs, A, rhs) for the inputs *t* and circulation *g_unknown*.

        Parameters
        ----------
        t : dict
            Torch input tensors (nodes per surface, scalars, sections, reference).
        g_unknown : torch.Tensor
            Circulation of each unknown [m^2/s], shape (n,).

        Returns
        -------
        outputs : dict
            Torch scalars for each scalar output, plus ``strip_force``
            (n_strips, 3) and ``section_cl`` (n_strips,).
        A : torch.Tensor
            System matrix, shape (n, n). For the nonlinear lifting line it
            is the Newton Jacobian dR/dG at *g_unknown* (no graph).
        rhs : torch.Tensor
            Right-hand side, shape (n,). For the nonlinear lifting line it
            is ``A G - R(G, x)``. So in both cases ``A G - rhs`` is the
            residual and ``solve(A, rhs)`` is the next Newton iterate (the
            solution for the linear solvers).
        """
        b = self.b
        n_s = b["n_strips"]
        n_p = b["n_panels"]
        n_u = b["n_unknowns"]
        alpha = t["alpha"]
        beta = t["beta"]
        V = t["V_inf"]
        rho = t["rho"]
        rp = t["ref_point"]
        origin = b.get("origin")
        if origin is not None:
            rp = rp - origin
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        cb = torch.cos(beta)
        sb = torch.sin(beta)
        d = torch.stack((ca * cb, -sb, sa * cb))
        if b["wake_body"]:
            wd = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        else:
            wd = d

        # --- body rates: quasi-steady rotation about the reference point ------
        rated = bool(b.get("rates", False))
        if rated:
            # Rates p, q, r in the flight-dynamics body axes; the geometry axes
            # are x aft, y right, z up, so the angular velocity is (-p, q, -r).
            omega = torch.stack((-t["p"], t["q"], -t["r"]))

            def _w_rot(pts: torch.Tensor) -> torch.Tensor:
                return -torch.linalg.cross(omega.expand_as(pts), pts - rp)

            def _onset(pts: torch.Tensor) -> torch.Tensor:
                return V * d + _w_rot(pts)

        # --- lattice from the edge nodes ------------------------------------
        le_l, le_r, te_l, te_r, V_all, N_cp = self._strips(t)
        cd = t.get("camber_data", b.get("camber_data"))
        chord_le_l, chord_le_r = le_l, le_r
        chord_te_l, chord_te_r = te_l, te_r
        chord_l = torch.linalg.norm(te_l - le_l, dim=1)
        chord_r = torch.linalg.norm(te_r - le_r, dim=1)
        chord = 0.5 * (chord_l + chord_r)
        qc_l = le_l + 0.25 * (te_l - le_l)
        qc_r = le_r + 0.25 * (te_r - le_r)
        dl = qc_r - qc_l
        qc_mid = 0.5 * (qc_l + qc_r)
        width = torch.sqrt(dl[:, 1] ** 2 + dl[:, 2] ** 2)
        width = torch.clamp_min(width, 1e-12)
        area = chord * width

        def _norm_rows(x: torch.Tensor) -> torch.Tensor:
            return x / torch.clamp_min(torch.linalg.norm(x, dim=1, keepdim=True), 1e-300)

        tl = _norm_rows(te_l - le_l)
        tr = _norm_rows(te_r - le_r)
        cdir = _norm_rows(tl + tr)
        s_hat = _norm_rows(dl)
        normal = _norm_rows(torch.linalg.cross(cdir, s_hat))
        span_yz = torch.stack(
            (torch.zeros_like(dl[:, 0]), dl[:, 1], dl[:, 2]), dim=1
        )
        span_yz = _norm_rows(span_yz)
        aL0 = t["alpha_L0"]
        cm0 = t["Cm0"]
        # Control deflections. The tensors "controls:<name>" are the changes of
        # the deflection of every control with that name [rad] (zero at the
        # base point). Section mode (lifting lines and one chordwise panel):
        # the thin-airfoil deflected section, alpha_L0 - tau delta and
        # Cm0 + dCm/ddelta delta, and the tabulated polars at alpha + tau delta.
        inc = self._control_increments(t)
        shift_strip = None
        cm_shift_strip = None
        cs = b.get("ctrl_section")
        if cs is not None and inc is not None:
            inc_s = inc[cs["ctrl"]]
            aL0 = aL0.index_add(0, cs["strips"], -cs["tau_sign"] * inc_s)
            cm0 = cm0.index_add(0, cs["strips"], cs["dcm_sign"] * inc_s)
            zero = torch.zeros(n_s, dtype=self.dtype, device=self.device)
            shift_strip = zero.index_add(0, cs["strips"], cs["tau_sign"] * inc_s)
            cm_shift_strip = zero.index_add(0, cs["strips"], cs["dcm_sign"] * inc_s)
        normal_bc_strip = rod_rot(normal, span_yz, -aL0)
        if cd is not None:
            normal_bc_strip = torch.where(cd["strip_camber"][:, None], normal, normal_bc_strip)

        a0 = t["a0"]
        claf = _clamp_closed(a0 / (2.0 * math.pi), 0.2, 2.0)
        xi = b["xi"]
        frac = b["frac"]
        n_chord = b["n_chord"]
        pa_list, pb_list, pcp_list = [], [], []
        # Planform points of the leg nodes between two bound points (the
        # middle chord fraction; on the last panel of a strip half a panel
        # length aft of its bound point, production: build_lattice).
        nb_mid = None
        if n_chord > 1 and b.get("leg_next") is not None:
            xb_all = xi[:, :-1] + 0.25 * (xi[:, 1:] - xi[:, :-1])
            xm_last = xb_all[:, -1:] + 0.5 * (xi[:, -1:] - xi[:, -2:-1])
            xm_all = torch.cat((0.5 * (xb_all[:, :-1] + xb_all[:, 1:]), xm_last), dim=1)
            nb_mid = ((le_l[:, None] + xm_all[..., None] * (te_l - le_l)[:, None]).reshape(-1, 3),
                      (le_r[:, None] + xm_all[..., None] * (te_r - le_r)[:, None]).reshape(-1, 3), xm_all)
        for k in range(n_chord):
            x0 = xi[:, k]
            x1 = xi[:, k + 1]
            dx = x1 - x0
            f_bound = x0 + 0.25 * dx
            pa = le_l + f_bound[:, None] * (te_l - le_l)
            pb = le_r + f_bound[:, None] * (te_r - le_r)
            if b["collocation"] == "llt":
                pcp = pa + frac[:, None] * (pb - pa)
            else:
                f_cp = x0 + 0.25 * dx + 0.5 * dx * claf
                p_l = le_l + f_cp[:, None] * (te_l - le_l)
                p_r = le_r + f_cp[:, None] * (te_r - le_r)
                pcp = p_l + frac[:, None] * (p_r - p_l)
            pa_list.append(pa)
            pb_list.append(pb)
            pcp_list.append(pcp)
        pa_s = torch.stack(pa_list, dim=1)
        pb_s = torch.stack(pb_list, dim=1)
        pcp_s = torch.stack(pcp_list, dim=1)
        a = pa_s.reshape(-1, 3)
        bv = pb_s.reshape(-1, 3)
        cp = pcp_s.reshape(-1, 3)
        panel_strip = b["panel_strip"]
        if cd is not None:
            a = a + cd["W_a"] @ V_all
            bv = bv + cd["W_b"] @ V_all
            if nb_mid is not None:
                nb_mid = (nb_mid[0] + cd["W_ma"] @ V_all, nb_mid[1] + cd["W_mb"] @ V_all, nb_mid[2])
            edge_vec_l = chord_te_l - chord_le_l
            edge_vec_r = chord_te_r - chord_le_r
            cp_chord = torch.linalg.norm(
                (1.0 - frac[:, None]) * edge_vec_l + frac[:, None] * edge_vec_r, dim=1
            )
            off_cp = (cp_chord[:, None] * cd["z_c"]).reshape(-1, 1) * N_cp[panel_strip]
            cp = cp + off_cp
            te_l = te_l + cd["W_te_l"] @ V_all
            te_r = te_r + cd["W_te_r"] @ V_all
        a_te = te_l[panel_strip]
        b_te = te_r[panel_strip]
        d_xi = xi[:, 1:] - xi[:, :-1]
        panel_length = (d_xi * chord[:, None]).reshape(-1)
        rc_panel = 0.01 * torch.minimum(width[panel_strip], panel_length)

        n_bc = normal_bc_strip[panel_strip].clone()
        if cd is not None:
            tangent = cdir[:, None, :] + cd["slope"][..., None] * N_cp[:, None, :]
            nb = torch.linalg.cross(
                tangent.reshape(-1, 3), torch.repeat_interleave(s_hat, n_chord, dim=0)
            )
            nb = _norm_rows(nb)
            n_bc = torch.where(cd["camber_panel_mask"][:, None], nb, n_bc)
        cc = b.get("ctrl_chord")
        kink_geom = None
        leg_turn = None
        if cc is not None:
            # Chordwise mode (vortex lattice, two or more chordwise panels):
            # the points aft of the hinge turn about the hinge axis of their
            # strip edge (the geometry of build_lattice). The axis of an edge
            # node is the central difference of the hinge points of its
            # neighbour nodes; the angle of an edge node is its base angle
            # plus the coefficients of the control increments (the joins at
            # the side edges of the controls). The node chain carries the
            # motion of the hinge points and of the axes.
            s_c = cc["strip"]
            hf = cc["hinge"][:, None]

            def _hp(le, te, idx):
                return le[idx] + hf * (te[idx] - le[idx])

            def _hp_node(node):
                # Hinge point at the node (strip, side) with the hinge of each flap strip.
                idx, side = node[:, 0], (node[:, 1] == 1)[:, None]
                return torch.where(side, _hp(chord_le_r, chord_te_r, idx), _hp(chord_le_l, chord_te_l, idx))

            h_l = _hp(chord_le_l, chord_te_l, s_c)
            h_r = _hp(chord_le_r, chord_te_r, s_c)
            hp_far0 = _hp_node(cc["far"][:, 0])
            hp_far1 = _hp_node(cc["far"][:, 1])
            if cd is not None and cd["W_hl"] is not None:
                h_l = h_l + cd["W_hl"] @ V_all
                h_r = h_r + cd["W_hr"] @ V_all
                hp_far0 = hp_far0 + cd["W_hfar0"] @ V_all
                hp_far1 = hp_far1 + cd["W_hfar1"] @ V_all
            ax_l = _norm_rows(h_r - hp_far0)
            ax_r = _norm_rows(hp_far1 - h_l)
            ang = cc["angle0"]
            if inc is not None:
                ang = ang + torch.einsum("fsc,c->fs", cc["coef"], inc)
            d_l, d_r = ang[:, 0], ang[:, 1]
            f_cp = frac[s_c][:, None]
            h_f = h_l + f_cp * (h_r - h_l)
            ax_f = _norm_rows(ax_l + f_cp * (ax_r - ax_l))
            d_f = d_l + f_cp[:, 0] * (d_r - d_l)
            x0_s = xi[s_c, :-1]
            dx_s = xi[s_c, 1:] - x0_s
            xb_s = x0_s + 0.25 * dx_s
            xc_s = xb_s + 0.5 * dx_s * claf[s_c, None]
            hcol = cc["hinge"][:, None]

            def _turn(points, mask, H, ax, da):
                row, col = torch.where(mask)
                idx = s_c[row] * n_chord + col
                return points.index_copy(0, idx, H[row] + rod_rot(points[idx] - H[row], ax[row], da[row]))

            a = _turn(a, xb_s > hcol, h_l, ax_l, d_l)
            bv = _turn(bv, xb_s > hcol, h_r, ax_r, d_r)
            if nb_mid is not None:
                # The mean-surface leg nodes aft of the hinge turn with the flap.
                xm_s = nb_mid[2][s_c]
                nb_mid = (_turn(nb_mid[0], xm_s > hcol, h_l, ax_l, d_l),
                          _turn(nb_mid[1], xm_s > hcol, h_r, ax_r, d_r), nb_mid[2])
            # A high CLAF can put the control point of a panel ahead of the
            # hinge aft of it: the point moves, its normal does not turn.
            cp = _turn(cp, xc_s > hcol, h_f, ax_f, d_f)
            panels = cc["panels"]
            row_n = torch.searchsorted(s_c, cc["panel_strip"])
            n_bc = n_bc.index_copy(0, panels, rod_rot(n_bc[panels], ax_f[row_n], d_f[row_n]))
            # The wake starts at the deflected trailing edge.
            te_l = te_l.index_copy(0, s_c, h_l + rod_rot(te_l[s_c] - h_l, ax_l, d_l))
            te_r = te_r.index_copy(0, s_c, h_r + rod_rot(te_r[s_c] - h_r, ax_r, d_r))
            a_te, b_te = te_l[panel_strip], te_r[panel_strip]
            # The legs of the panels ahead of the hinge run bound point ->
            # hinge point of the strip edge -> deflected trailing edge, as
            # in the production lattice (also at zero deflection).
            row_f, col_f = torch.where(xb_s <= hcol)
            p_f = s_c[row_f] * n_chord + col_f
            a_k = a_te.index_copy(0, p_f, h_l[row_f])
            b_k = b_te.index_copy(0, p_f, h_r[row_f])
            kink_panel = torch.zeros(n_p, dtype=torch.bool, device=self.device)
            kink_panel[p_f] = True
            kink_geom = (a_k, b_k, kink_panel)
            # The rule normal of a leg on the flap turns with its edge node
            # (production: build_lattice). Aft panel: the whole legs (sets 0,
            # 1); panel ahead of the hinge: the aft segments (sets 2, 3).
            leg_turn = (xb_s > hcol, s_c, ax_l, d_l, ax_r, d_r)
        chain_geom = None
        nxt_p = b.get("leg_next")
        if nxt_p is not None:
            # Leg chains on the strips with geometric camber (production:
            # geometry.lattice._leg_nodes): the legs of a panel end at the
            # bound points of the next panel and continue on its legs. The
            # first node is the mean-surface node between the two bound
            # points, or the hinge point where the hinge lies between them.
            a_nx = torch.cat((a[1:], a[-1:]), dim=0)
            b_nx = torch.cat((bv[1:], bv[-1:]), dim=0)
            a_md, b_md = nb_mid[0], nb_mid[1]
            if kink_geom is None:
                a_k0, b_k0 = a_te, b_te
                hinge = torch.zeros(n_p, dtype=torch.bool, device=self.device)
            else:
                a_k0, b_k0, hinge = kink_geom
            no = torch.zeros(1, dtype=torch.bool, device=self.device)
            hinge_next = torch.cat((hinge[1:], no))
            on_chain = nxt_p | torch.cat((no, nxt_p[:-1]))
            # The last panel of a chain strip has a mean-surface node between
            # its bound point and the trailing edge (or the hinge point there).
            last_p = on_chain & ~nxt_p
            at_hinge = hinge & (~hinge_next | last_p)
            # At an edge shared with a strip of a different hinge (no hinge,
            # another hinge fraction, or a joined surface end) the hinge point
            # is not a node: the mean-surface node is (production: the
            # hinge_off edges of _leg_nodes).
            off = b.get("hinge_off_panel")
            if off is None:
                take_a = take_b = (on_chain & ~at_hinge)[:, None]
            else:
                take_a = (on_chain & ~(at_hinge & ~off[:, 0]))[:, None]
                take_b = (on_chain & ~(at_hinge & ~off[:, 1]))[:, None]
            kink_geom = (torch.where(take_a, a_md, a_k0), torch.where(take_b, b_md, b_k0), hinge | on_chain)
            chain_geom = (torch.where(nxt_p[:, None], a_nx, a_te), torch.where(nxt_p[:, None], b_nx, b_te),
                          on_chain)
        cp_u = cp[b["unknown_panels"]]
        n_bc_u = n_bc[b["unknown_panels"]]
        # Target cross-core radii follow the strip widths (they are not fixed:
        # a node move changes the widths). Groups are fixed mesh topology.
        tgt_rc_panel = 0.5 * width[panel_strip]
        tgt_rc_unk = tgt_rc_panel[b["unknown_panels"]]
        tgt_group_panel = b["tgt_group_panels"]
        tgt_group_unk = b["tgt_group_unknown"]

        # --- ground plane ----------------------------------------------------
        h = t.get("h")
        if h is None:
            normal_g = None
            offset = None
        else:
            phi = t["phi"]
            k0 = torch.stack((-sa, torch.zeros_like(sa), ca))
            v = d
            k = k0 * torch.cos(phi) + torch.linalg.cross(v, k0) * torch.sin(phi)
            normal_g = k / torch.clamp_min(torch.linalg.norm(k), 1e-300)
            offset = (rp * normal_g).sum() - h

        src_group = b["src_group"]
        col = b["column"]
        wd_e = wd.expand(n_p, 3)
        s_kink = None
        mach_active = bool(b.get("mach_active", False))
        if mach_active:
            # Prandtl-Glauert stretch about the moment reference point, in
            # free-stream axes (Goethert 1946). The map runs inside the
            # differentiable reference, so the chain rule includes the change
            # of the stretched geometry with alpha, beta and mach.
            M_mach = t.get("mach")
            beta_pg = torch.sqrt(1.0 - M_mach * M_mach)
            inv_beta = 1.0 / beta_pg
            one_pg = torch.ones_like(beta_pg)
            lift_pg = torch.stack((-sa, torch.zeros_like(sa), ca))
            side_pg = torch.linalg.cross(lift_pg, d)
            R_pg = torch.stack((d, side_pg, lift_pg))

            def _to_s(pts: torch.Tensor) -> torch.Tensor:
                p_fs = (pts - rp) @ R_pg.T
                return p_fs * torch.stack((inv_beta, one_pg, one_pg))

            def _n_eff(n: torch.Tensor) -> torch.Tensor:
                n_fs = n @ R_pg.T
                return n_fs * torch.stack((inv_beta, one_pg, one_pg))

            def _v_back(v_s: torch.Tensor) -> torch.Tensor:
                v_fs = v_s * torch.stack((inv_beta, one_pg, one_pg))
                return v_fs @ R_pg

            w_fs_pg = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype,
                                   device=self.device).expand(n_p, 3)
        else:
            _to_s = None  # type: ignore[assignment]
            _n_eff = None  # type: ignore[assignment]
            _v_back = None  # type: ignore[assignment]
        if h is None:
            if kink_geom is not None:
                pts_k = (kink_geom[0], kink_geom[1]) + (() if chain_geom is None else chain_geom[:2])
                s_kink = tuple(_to_s(x) for x in pts_k) if mach_active else pts_k
            if mach_active:
                s_a, s_b, s_ate, s_bte = _to_s(a), _to_s(bv), _to_s(a_te), _to_s(b_te)
                s_wd = w_fs_pg
                s_rc = rc_panel
                s_sign = torch.ones_like(rc_panel)
                s_group = src_group
                s_col = col
            else:
                s_a, s_b, s_ate, s_bte = a, bv, a_te, b_te
                s_wd = wd_e
                s_rc = rc_panel
                s_sign = torch.ones_like(rc_panel)
                s_group = src_group
                s_col = col
        else:
            def _refl_pts(p: torch.Tensor) -> torch.Tensor:
                hh = p @ normal_g - offset
                return p - 2.0 * hh[:, None] * normal_g

            def _refl_dir(vv: torch.Tensor) -> torch.Tensor:
                return vv - 2.0 * (vv @ normal_g)[:, None] * normal_g

            if kink_geom is not None:
                pts_k = (kink_geom[0], kink_geom[1]) + (() if chain_geom is None else chain_geom[:2])
                s_kink = tuple(torch.cat((x, _refl_pts(x)), dim=0) for x in pts_k)
            s_a = torch.cat((a, _refl_pts(a)), dim=0)
            s_b = torch.cat((bv, _refl_pts(bv)), dim=0)
            s_ate = torch.cat((a_te, _refl_pts(a_te)), dim=0)
            s_bte = torch.cat((b_te, _refl_pts(b_te)), dim=0)
            s_wd = torch.cat((wd_e, _refl_dir(wd_e)), dim=0)
            s_rc = torch.cat((rc_panel, rc_panel), dim=0)
            s_sign = torch.cat((torch.ones_like(rc_panel), -torch.ones_like(rc_panel)), dim=0)
            s_group = torch.cat((src_group, src_group), dim=0)
            s_col = torch.cat((col, col), dim=0)

        # Topology ids of the sources for the on-segment masks: the panel of
        # each bound segment and the lattice edge of each leg. Images get -2:
        # no target lies on an image segment by construction.
        edges = b["strip_edges"]
        pid = torch.arange(n_p, dtype=torch.int64, device=self.device)
        ea = edges[panel_strip, 0]
        eb = edges[panel_strip, 1]
        if cc is not None:
            # Uncontrolled edges keep their common line for all node moves.
            # On a flap strip the forward segments of the legs stay on the
            # edge line (they end at the hinge node). The aft points of an
            # edge and its trailing edge turn about the line of the edge
            # node, so the aft legs and the aft segments of the edge keep a
            # common line; two flap strips with the same hinge share it
            # (see adjoint._aft_line_ids).
            row_a = torch.searchsorted(s_c, cc["panel_strip"])
            ea = ea.index_copy(0, cc["panels"], cc["aft_edge"][row_a, 0])
            eb = eb.index_copy(0, cc["panels"], cc["aft_edge"][row_a, 1])
            kink_panel = kink_geom[2]
            none = torch.full((n_p,), -1, dtype=torch.int64, device=self.device)
            row_k = torch.searchsorted(s_c, panel_strip.clamp_max(int(s_c.max())))
            row_k = torch.where(kink_panel, row_k, torch.zeros_like(row_k))
            eka = torch.where(kink_panel, cc["aft_edge"][row_k, 0], none)
            ekb = torch.where(kink_panel, cc["aft_edge"][row_k, 1], none)
        if cd is not None:
            # On a strip with mean-line camber offsets, the chordwise horseshoe
            # legs curve along chord with the non-linear mean line. The legs of
            # different panels on that edge do not share a common line, and only
            # each panel's own leg contains its leg mid-point. Assign unique
            # negative line ids to cambered panels so only each panel's own leg
            # is masked on itself, matching the production near-field velocity.
            c_mask = cd["camber_panel_mask"]
            ea = torch.where(c_mask, -10 - 2 * pid, ea)
            eb = torch.where(c_mask, -11 - 2 * pid, eb)
        if chain_geom is not None:
            # The aft segments of the chain legs (node to the next bound
            # point) are curved like the forward ones: unique ids too.
            if cc is None:
                eka = torch.full((n_p,), -1, dtype=torch.int64, device=self.device)
                ekb = eka.clone()
            on_chain = chain_geom[2]
            eka = torch.where(on_chain, -(10 + 2 * n_p) - 2 * pid, eka)
            ekb = torch.where(on_chain, -(11 + 2 * n_p) - 2 * pid, ekb)
        edge_l_panel, edge_r_panel = ea, eb
        src_kink = None
        if kink_geom is not None:
            kfl = kink_geom[2].to(torch.int64)
            edge_ka_panel, edge_kb_panel = eka, ekb
            src_kink = {"kink": kfl, "edge_ka": eka, "edge_kb": ekb}
            if chain_geom is not None:
                src_kink["leg_next"] = nxt_p.to(torch.int64)
        if h is not None:
            img = torch.full((n_p,), -2, dtype=torch.int64, device=self.device)
            pid = torch.cat((pid, img))
            ea = torch.cat((ea, img))
            eb = torch.cat((eb, img))
            if src_kink is not None:
                chain_ids = src_kink.get("leg_next")
                src_kink = {"kink": torch.cat((src_kink["kink"], src_kink["kink"])),
                            "edge_ka": torch.cat((src_kink["edge_ka"], img)),
                            "edge_kb": torch.cat((src_kink["edge_kb"], img))}
                if chain_ids is not None:
                    src_kink["leg_next"] = torch.cat((chain_ids, chain_ids))
        b["_src_ids"] = {"panel": pid, "edge_a": ea, "edge_b": eb}
        if src_kink is not None:
            b["_src_ids"].update(src_kink)
        b["_src_kinks"] = s_kink
        panel_ids = torch.arange(n_p, dtype=torch.int64, device=self.device)

        rc2_src = s_rc * s_rc
        tg_group = tgt_group_unk
        tg_rc = tgt_rc_unk

        # --- system matrix and right-hand side -------------------------------
        alpha_u_nl = None
        if b["collocation"] == "vlm":
            if mach_active:
                P_s = _to_s(cp_u)
                n_eff = _n_eff(n_bc_u)
                A = self._normal_matrix(P_s, n_eff, s_a, s_b, s_ate, s_bte, s_wd,
                                        rc2_src, s_group, s_sign, s_col, n_u,
                                        tg_group, tg_rc)
            else:
                A = self._normal_matrix(cp_u, n_bc_u, s_a, s_b, s_ate, s_bte, s_wd,
                                        rc2_src, s_group, s_sign, s_col, n_u,
                                        tg_group, tg_rc)
            rhs = -V * (n_bc_u * d).sum(-1)
            if rated:
                rhs = rhs - (n_bc_u * _w_rot(cp_u)).sum(-1)
        elif b.get("solver") == "nonlinear":
            A, rhs, alpha_u_nl = self._nonlinear_system(
                t, g_unknown, d, V, cp_u, dl, area, normal, cdir, chord, a0, aL0, shift_strip,
                s_a, s_b, s_ate, s_bte, s_wd, rc2_src, s_group, s_sign, s_col, n_u,
                tg_group, tg_rc, h, _onset if rated else None)
        else:
            dl_u = dl[b["strip_of_unknown"]]
            area_u = area[b["strip_of_unknown"]]
            a0_u = a0[b["strip_of_unknown"]]
            aL0_u = aL0[b["strip_of_unknown"]]
            normal_u = normal[b["strip_of_unknown"]]
            # The collocation point of a lifting-line strip lies on its own
            # bound segment.
            if self._vjp_backend() is not None:
                # Compiled kernels: the normal component per source, added
                # per unknown (equal to the tensor path to round-off).
                a_ind = self._normal_matrix(cp_u, normal_u, s_a, s_b, s_ate, s_bte, s_wd,
                                            rc2_src, s_group, s_sign, s_col, n_u,
                                            tg_group, tg_rc, own=b["unknown_panels"])
            else:
                Vt = self._tensor_unknowns(cp_u, s_a, s_b, s_ate, s_bte, s_wd,
                                           rc2_src, s_group, s_sign, s_col, n_u,
                                           tg_group, tg_rc, own=b["unknown_panels"])
                a_ind = torch.einsum("ijk,ik->ij", Vt, normal_u)
            if rated:
                # Local onset at the three-quarter chord of each strip, at
                # the span fraction of the control point (as llt_onset).
                sou = b["strip_of_unknown"]
                p34 = cp_u + 0.5 * chord[sou][:, None] * cdir[sou]
                onset_u = _onset(p34)
                speed = torch.linalg.norm(onset_u, dim=1)
                unit = onset_u / speed[:, None]
                diag = 2.0 * torch.linalg.norm(torch.linalg.cross(unit, dl_u), dim=1) / (a0_u * area_u)
                A = torch.diag(diag) - a_ind
                rhs = (normal_u * onset_u).sum(-1) - speed * aL0_u
            else:
                cross_dl = torch.linalg.cross(d.expand_as(dl_u), dl_u)
                diag = 2.0 * torch.linalg.norm(cross_dl, dim=1) / (a0_u * area_u)
                A = torch.diag(diag) - a_ind
                rhs = V * ((normal_u * d).sum(-1) - aL0_u)

        # --- near-field loads ------------------------------------------------
        f_frac = b["cp_frac_panel"]
        mid = a + f_frac[:, None] * (bv - a)
        l_vec = bv - a
        use_leg = b["collocation"] == "vlm"
        g_panel = g_unknown[b["panel_column_full"]]
        mir = torch.tensor([1.0, -1.0, 1.0], dtype=self.dtype, device=self.device)
        fold = bool(b["fold_loads"])
        if fold:
            right = b["right_panels"]
            left = b["left_panels"]
            pm = b["panel_mirror"]
            mid_e = mid[right]
            tgt_ge = tgt_group_panel[right]
            tgt_re = tgt_rc_panel[right]
            own_e = panel_ids[right]
        else:
            mid_e = mid
            tgt_ge = tgt_group_panel
            tgt_re = tgt_rc_panel
            own_e = panel_ids
        # The load point of a panel lies on its own bound segment.
        if h is None:
            g_e = g_panel * s_sign
        else:
            g_e = torch.cat((g_panel, g_panel), dim=0) * s_sign
        if mach_active:
            # Stretched evaluation with the physical normal map back: the
            # streamwise component scales with 1/beta (Goethert 1946). The
            # force points stay physical; their move enters the outputs.
            v_e_s = self._induced(_to_s(mid_e), s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                  s_group, g_e, tgt_ge, tgt_re, own=own_e)
            v_e = _v_back(v_e_s)
        else:
            v_e = self._induced(mid_e, s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                s_group, g_e, tgt_ge, tgt_re, own=own_e)
        if fold:
            right = b["right_panels"]
            left = b["left_panels"]
            pm = b["panel_mirror"]
            pos_all = torch.empty((n_p,), dtype=torch.int64, device=self.device)
            pos_all[right] = torch.arange(right.shape[0], dtype=torch.int64, device=self.device)
            v_ind = torch.empty_like(mid)
            v_ind[right] = v_e
            v_ind[left] = v_e[pos_all[pm[left]]] * mir
        else:
            v_ind = v_e
        U = _onset(mid) if rated else (V * d).expand_as(v_ind)
        f_panel = rho * g_panel[:, None] * torch.linalg.cross(U + v_ind, l_vec)
        m_geo = torch.linalg.cross(mid - rp, f_panel).sum(0)
        if use_leg:
            # The leg segment sets of the production loads (leg_segments):
            # without hinge nodes the legs a_te -> a and b -> b_te; with them
            # the forward segments to the nodes and the aft segments.
            if kink_geom is None:
                seg_sets = [(a_te, a, edge_l_panel), (bv, b_te, edge_r_panel)]
            else:
                a_k, b_k = kink_geom[0], kink_geom[1]
                # The aft segments end at the trailing edge, or on a chain
                # at the bound points of the next panel.
                a_e, b_e = (a_te, b_te) if chain_geom is None else chain_geom[:2]
                seg_sets = [(a_k, a, edge_l_panel), (bv, b_k, edge_r_panel),
                            (a_e, a_k, edge_ka_panel), (b_k, b_e, edge_kb_panel)]
            n_sets = len(seg_sets)
            set_mid = [0.5 * (s0 + s1) for s0, s1, _ in seg_sets]
            leg_start = torch.cat([s0 for s0, _, _ in seg_sets], dim=0)
            leg_end = torch.cat([s1 for _, s1, _ in seg_sets], dim=0)
            leg_mid = torch.cat(set_mid, dim=0)
            leg_vec = leg_end - leg_start
            # Leg rule of the production loads (leg_force_rules): the strip
            # normal is removed on a shared edge, nothing on a free edge (the
            # full velocity), and a kink leg has no force (scale 0).
            strip_normal = normal[panel_strip]
            leg_normal = torch.cat([strip_normal] * n_sets, dim=0) * b["leg_keep_normal"][:, None]
            if leg_turn is not None and n_sets == 4:
                aft_m, s_t, axl, dl_, axr, dr_ = leg_turn
                # Production (build_lattice): the forward segments of the
                # aft panels turn; the aft segments turn where the first
                # node is the hinge point or aft of it (aft panels on a
                # chain, and panels ahead of the hinge except those on a
                # chain whose next panel is ahead of the hinge too).
                fwd_m, aft_s = aft_m, ~aft_m
                if nxt_p is not None:
                    # Every panel of a chain strip has a mean-surface node.
                    chain_m = chain_geom[2].reshape(-1, n_chord)[s_t]
                    ahead_next = torch.cat((~aft_m[:, 1:], torch.zeros_like(aft_m[:, :1])), dim=1)
                    aft_s = (aft_m & chain_m) | (~aft_m & ~(chain_m & ahead_next))
                for mask, sets in ((fwd_m, (0, 1)), (aft_s, (2, 3))):
                    row, col = torch.where(mask)
                    idx = s_t[row] * n_chord + col
                    for k, ax, da in ((sets[0], axl, dl_), (sets[1], axr, dr_)):
                        rows = k * n_p + idx
                        leg_normal = leg_normal.index_copy(0, rows, rod_rot(leg_normal[rows], ax[row], da[row]))
            leg_tgt_group = torch.cat([tgt_group_panel] * n_sets, dim=0)
            leg_tgt_rc = torch.cat([tgt_rc_panel] * n_sets, dim=0)
            if h is None:
                g_leg_src = g_panel * s_sign
            else:
                g_leg_src = torch.cat((g_panel, g_panel), dim=0) * s_sign
            if fold:
                # Right panels only; left panels mirror with the leg swap:
                # the a-leg of a left panel mirrors the b-leg of its mirror.
                leg_e = torch.cat([m[right] for m in set_mid], dim=0)
                ltgt_g = torch.cat([tgt_ge] * n_sets, dim=0)
                ltgt_r = torch.cat([tgt_re] * n_sets, dim=0)
                # A leg mid-point lies on the line of its edge, so also on
                # the legs of the other panels that use the same edge.
                ledge = torch.cat([e[right] for _, _, e in seg_sets], dim=0)
                if mach_active:
                    v_leg_e_s = self._induced(_to_s(leg_e), s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                              s_group, g_leg_src, ltgt_g, ltgt_r, edge=ledge)
                    v_leg_e = _v_back(v_leg_e_s)
                else:
                    v_leg_e = self._induced(leg_e, s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                            s_group, g_leg_src, ltgt_g, ltgt_r, edge=ledge)
                n_r = right.shape[0]
                pos = pos_all
                mir_right = pos[pm[left]]
                v_leg = torch.empty_like(leg_mid)
                for k, sw in enumerate((1, 0, 3, 2)[:n_sets]):
                    v_leg[k * n_p + right] = v_leg_e[k * n_r:(k + 1) * n_r]
                    v_leg[k * n_p + left] = v_leg_e[sw * n_r + mir_right] * mir
            else:
                ledge = torch.cat([e for _, _, e in seg_sets], dim=0)
                if mach_active:
                    v_leg_s = self._induced(_to_s(leg_mid), s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                            s_group, g_leg_src, leg_tgt_group, leg_tgt_rc,
                                            edge=ledge)
                    v_leg = _v_back(v_leg_s)
                else:
                    v_leg = self._induced(leg_mid, s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                          s_group, g_leg_src, leg_tgt_group, leg_tgt_rc,
                                          edge=ledge)
            U_leg = _onset(leg_mid) if rated else (V * d).expand_as(v_leg)
            V_leg = U_leg + v_leg
            Vn = (V_leg * leg_normal).sum(-1)
            V_leg = V_leg - Vn[:, None] * leg_normal
            # On a chain the own leg segments of a panel carry the circulation
            # of every panel before them (production: loads.leg_strength).
            g_own = g_panel if chain_geom is None else _chain_prefix(g_panel, nxt_p)
            g_leg_only = torch.cat([g_own] * n_sets, dim=0) * b["leg_scale"]
            f_leg = rho * g_leg_only[:, None] * torch.linalg.cross(V_leg, leg_vec)
            m_geo = m_geo + torch.linalg.cross(leg_mid - rp, f_leg).sum(0)
            for k in range(n_sets):
                f_panel = f_panel + f_leg[k * n_p:(k + 1) * n_p]
        f_strip = f_panel.reshape(n_s, n_chord, 3).sum(1)
        strip_gamma = g_panel.reshape(n_s, n_chord).sum(1)

        cl_strip = 2.0 * strip_gamma / (V * chord)
        nd = (normal * d).sum(-1)
        cd_ = (cdir * d).sum(-1)
        alpha_geom = torch.atan2(nd, cd_)
        if alpha_u_nl is not None:
            # Nonlinear lifting line: the section angle of the solve (one
            # panel per strip; a left strip of the fold takes its mirror).
            alpha_eff = alpha_u_nl[b["panel_column_full"]]
        elif mach_active:
            # Compressible section slope a0 / beta (Glauert 1928), as in
            # production (loads.compute_loads_batch).
            alpha_eff = aL0 + cl_strip / (torch.clamp_min(a0, 1e-9) * inv_beta)
        else:
            alpha_eff = aL0 + cl_strip / torch.clamp_min(a0, 1e-9)
        # Prandtl-Glauert: the Cm0 of a linear airfoil scales by 1 / beta
        # (production: loads.compute_loads_batch). Tabulated strips take
        # their table moment (cm0 * lin is zero on them).
        cm0_sec = cm0 * inv_beta if mach_active else cm0
        cd_p, cm_s = self._section_drag_moment(t["Cd0"], cm0_sec, alpha_eff, shift_strip, cm_shift_strip)
        if b.get("strip_camber") is not None:
            cm_s = torch.where(b["strip_camber"], torch.zeros_like(cm_s), cm_s)
        q_inf = 0.5 * rho * V * V
        S_ref = t["S_ref"]
        b_ref = t["b_ref"]
        c_ref = t["c_ref"]
        qS = q_inf * S_ref
        if rated:
            if n_chord == 1:
                # One chordwise panel: the thin-airfoil moment of the
                # chordwise change of the rotation velocity, at the
                # three-quarter chord of the strip mid-point.
                p34s = qc_mid + 0.5 * chord[:, None] * cdir
                slope = -(normal * torch.linalg.cross(omega.expand_as(cdir), cdir)).sum(-1)
                cm_s = cm_s - (math.pi / 8.0) * slope * chord / torch.linalg.norm(_onset(p34s), dim=1)
            u_qc = _onset(qc_mid)
            speed_qc = torch.linalg.norm(u_qc, dim=1)
            d_loc = u_qc / speed_qc[:, None]
            q_loc = 0.5 * rho * speed_qc * speed_qc
        else:
            d_loc = d.expand(n_s, 3)
            q_loc = q_inf
        d_p = q_loc * chord * width * cd_p
        f_prof = d_p[:, None] * d_loc
        m_geo = m_geo + torch.linalg.cross(qc_mid - rp, f_prof).sum(0)
        n_lift = torch.linalg.cross(d_loc, dl)
        n_lift = n_lift / torch.clamp_min(torch.linalg.norm(n_lift, dim=1, keepdim=True), 1e-300)
        axis = torch.linalg.cross(n_lift, d_loc)
        axis = axis / torch.clamp_min(torch.linalg.norm(axis, dim=1, keepdim=True), 1e-300)
        m_geo = m_geo + ((q_loc * chord * chord * width * cm_s)[:, None] * axis).sum(0)

        # --- Trefftz induced drag --------------------------------------------
        di, w_n, _seg = self._trefftz(te_l, te_r, strip_gamma, wd, rho, frac,
                                      normal_g, offset, n_s, width)

        f_near = f_panel.sum(0)
        f_prof_sum = f_prof.sum(0)
        f_total = f_near + f_prof_sum
        lift_dir = torch.stack((-sa, torch.zeros_like(sa), ca))
        side_dir = torch.linalg.cross(lift_dir, d)
        CL = (f_total * lift_dir).sum() / qS
        CY = (f_total * side_dir).sum() / qS
        CDi = di / qS
        CDp = d_p.sum() / qS
        CD = CDi + CDp
        Cl_b = -m_geo[0] / (qS * b_ref)
        Cm_b = m_geo[1] / (qS * c_ref)
        Cn_b = -m_geo[2] / (qS * b_ref)
        coef = self._axes_moments(Cl_b, Cm_b, Cn_b, alpha, beta, b_ref, c_ref, b["axes"])
        outputs = {"CL": CL, "CY": CY, "CDi": CDi, "CDp": CDp, "CD": CD,
                   "Cl": coef[0], "Cm": coef[1], "Cn": coef[2],
                   "strip_force": f_strip, "section_cl": cl_strip}
        _ = alpha_geom, alpha_eff, w_n
        return outputs, A, rhs

    def _strips(self, t: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                        torch.Tensor | None, torch.Tensor | None]:
        """Return (le_l, le_r, te_l, te_r, V_all, N_cp) of all strips from the node inputs.

        With mean lines, ``V_all`` holds the chord times the unit offset
        normal of each edge node [m], shape (n_edges, 3), and ``N_cp`` the
        unit offset normal at the control point of each strip, shape
        (n_strips, 3); both are None without mean lines.
        """
        b = self.b
        le_parts, te_parts = [], []
        for s in b["surfaces"]:
            le = t[f"nodes:{s['name']}:le"]
            te = t[f"nodes:{s['name']}:te"]
            mir = torch.tensor(_MIRROR, dtype=self.dtype, device=self.device)
            if s["kind"] == "symmetric":
                left_le = torch.flip(le * mir, dims=(0,))
                left_te = torch.flip(te * mir, dims=(0,))
                full_le = torch.cat((left_le, le[1:]), dim=0)
                full_te = torch.cat((left_te, te[1:]), dim=0)
            elif s["kind"] == "mirror":
                full_le = torch.flip(le * mir, dims=(0,))
                full_te = torch.flip(te * mir, dims=(0,))
            else:
                full_le, full_te = le, te
            le_parts.append(full_le)
            te_parts.append(full_te)
        edge_le = torch.cat(le_parts, dim=0)
        edge_te = torch.cat(te_parts, dim=0)
        origin = b.get("origin")
        if origin is not None:
            # Detached local origin: the physics is unchanged (all points and
            # the moment reference move together), but the coordinates are
            # smaller, so float32 keeps more digits.
            edge_le = edge_le - origin
            edge_te = edge_te - origin
        idx = b["strip_edges"]
        V_all = None
        N_cp = None
        if b.get("camber_data") is not None:
            # Mean-line offset directions of the edge nodes and of the
            # control points from one span field with a blend at the folds
            # (production: geometry.lattice.camber_span_field).
            from ventorum.geometry.lattice import camber_span_field

            frac_parts, start = [], 0
            for fle in le_parts:
                frac_parts.append(b["frac"][start:start + fle.shape[0] - 1])
                start += fle.shape[0] - 1
            s_node, s_cp, _ = camber_span_field(_TorchOps, le_parts, te_parts, frac_parts,
                                                b.get("camber_folds") or [])
            V_parts, N_parts = [], []
            for fle, fte, sn, sc, fr in zip(le_parts, te_parts, s_node, s_cp, frac_parts):
                edge_vec = fte - fle
                edge_c = torch.linalg.norm(edge_vec, dim=1)
                edge_dir = edge_vec / torch.clamp_min(edge_c[:, None], 1e-300)
                normal = torch.linalg.cross(edge_dir, sn)
                normal = normal / torch.clamp_min(torch.linalg.norm(normal, dim=1, keepdim=True), 1e-300)
                V_parts.append(edge_c[:, None] * normal)
                cp_vec = (1.0 - fr[:, None]) * edge_vec[:-1] + fr[:, None] * edge_vec[1:]
                cp_dir = cp_vec / torch.clamp_min(torch.linalg.norm(cp_vec, dim=1, keepdim=True), 1e-300)
                n_cp = torch.linalg.cross(cp_dir, sc)
                N_parts.append(n_cp / torch.clamp_min(torch.linalg.norm(n_cp, dim=1, keepdim=True), 1e-300))
            V_all = torch.cat(V_parts, dim=0)
            N_cp = torch.cat(N_parts, dim=0)
        return edge_le[idx[:, 0]], edge_le[idx[:, 1]], edge_te[idx[:, 0]], edge_te[idx[:, 1]], V_all, N_cp

    def _control_increments(self, t: dict) -> torch.Tensor | None:
        """Return the deflection changes [rad] per control name, or None without controls."""
        names = self.b.get("control_names") or []
        if not names:
            return None
        return torch.stack([t[f"controls:{n}"] for n in names])

    def _section_drag_moment(self, cd0: torch.Tensor, cm0: torch.Tensor, alpha_eff: torch.Tensor,
                             shift: torch.Tensor | None,
                             cm_shift: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the section profile drag and moment coefficients per strip.

        Linear airfoils give Cd0 and Cm0. A tabulated airfoil gives its
        polar (the shape-preserving cubic of the production code) at the
        effective angle *alpha_eff* [rad], plus the deflection change of the
        section mode (*shift* [rad] on the angle, *cm_shift* on Cm).
        """
        tables = self.b.get("tables") or []
        if not tables:
            return cd0, cm0
        lin = self.b["lin_strip_mask"]
        cd_p = cd0 * lin
        cm_s = cm0 * lin
        record = []
        for tab in tables:
            idx = tab["strips"]
            a_t = alpha_eff[idx]
            if shift is not None:
                a_t = a_t + shift[idx]
            record.append(a_t.detach())
            cd_p = cd_p.index_add(0, idx, pp_eval(tab["Cd"], a_t))
            cm_t = pp_eval(tab["Cm"], a_t) if tab["Cm"] is not None else torch.zeros_like(a_t)
            if cm_shift is not None:
                cm_t = cm_t + cm_shift[idx]
            cm_s = cm_s.index_add(0, idx, cm_t)
        self.b["_table_alpha"] = record
        return cd_p, cm_s

    def _section_cl_unknowns(self, alpha_u: torch.Tensor, a0: torch.Tensor, aL0: torch.Tensor,
                             shift: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the section lift coefficient and its slope [1/rad] per unknown.

        Linear airfoils give ``a0 (alpha - alpha_L0)``. A tabulated airfoil
        gives its polar at ``alpha`` plus the deflection change of the
        section mode. The slope has no graph (it enters the Newton
        Jacobian only); it is zero outside the table, as in the production
        ``Cl_slope``.
        """
        b = self.b
        sou = b["strip_of_unknown"]
        lin = b["lin_strip_mask"][sou] if b.get("tables") else None
        if lin is None:
            cl = a0[sou] * (alpha_u - aL0[sou])
            return cl, a0[sou].detach()
        cl = a0[sou] * (alpha_u - aL0[sou]) * lin
        slope = (a0[sou] * lin).detach()
        rec = []
        for tab in b["tables"]:
            pos = tab["unknowns"]
            if pos.numel() == 0:
                rec.append(alpha_u.new_zeros(0))
                continue
            a_t = alpha_u[pos]
            if shift is not None:
                a_t = a_t + shift[sou[pos]]
            rec.append(a_t.detach())
            cl = cl.index_add(0, pos, pp_eval(tab["Cl"], a_t))
            slope = slope.index_add(0, pos, pp_slope(tab["Cl"], a_t.detach()))
        b["_table_alpha_unknowns"] = rec
        return cl, slope

    def _nonlinear_system(self, t: dict, g_unknown: torch.Tensor, d: torch.Tensor, V: torch.Tensor,
                          cp_u: torch.Tensor, dl: torch.Tensor, area: torch.Tensor,
                          normal: torch.Tensor, cdir: torch.Tensor, chord: torch.Tensor,
                          a0: torch.Tensor, aL0: torch.Tensor, shift: torch.Tensor | None,
                          s_a: torch.Tensor, s_b: torch.Tensor, s_ate: torch.Tensor,
                          s_bte: torch.Tensor, s_wd: torch.Tensor, rc2_src: torch.Tensor,
                          s_group: torch.Tensor, s_sign: torch.Tensor, s_col: torch.Tensor, n_u: int,
                          tg_group: torch.Tensor, tg_rc: torch.Tensor, h: torch.Tensor | None,
                          onset: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (J, J G - R, alpha) of the nonlinear lifting line at the circulation *g_unknown*.

        The residual of strip i (Phillips and Snyder 2000, eq. 15 in
        dimensional form) is ``R_i = 2 |W_i x dl_i| G_i - s_i Cl_i(alpha_i)``
        with ``W_i = U_i + sum_j v_ij G_j`` at the control point,
        ``alpha_i = atan2(W_i . n_i, W_i . a_i)`` and ``s_i = |U_i|^2 dA_i``
        (``U_i`` the free stream, or the local onset at the three-quarter
        chord with body rates). ``J = dR/dG`` is the Newton Jacobian of the
        production solve, without a graph. The second value is
        ``J G - R``, so that ``J G - (J G - R) = R`` carries the graph of the
        residual in the inputs, and ``solve(J, J G - R)`` is the Newton step.
        *alpha* [rad] is the section angle per unknown (with a graph).
        """
        b = self.b
        sou = b["strip_of_unknown"]
        dl_u = dl[sou]
        area_u = area[sou]
        normal_u = normal[sou]
        cdir_u = cdir[sou]
        if onset is not None:
            p34 = cp_u + 0.5 * chord[sou][:, None] * cdir_u
            u_on = onset(p34)  # type: ignore[operator]
            scale = (u_on * u_on).sum(-1) * area_u
        else:
            u_on = (V * d).expand(n_u, 3)
            scale = V * V * area_u
        g_panel = g_unknown[b["panel_column_full"]]
        g_src = g_panel * s_sign if h is None else torch.cat((g_panel, g_panel), dim=0) * s_sign
        v_u = self._induced(cp_u, s_a, s_b, s_ate, s_bte, s_wd, rc2_src, s_group, g_src,
                            tg_group, tg_rc, own=b["unknown_panels"])
        W = u_on + v_u
        wx = torch.linalg.cross(W, dl_u)
        wxn = torch.linalg.norm(wx, dim=1)
        wn = (W * normal_u).sum(-1)
        wa = (W * cdir_u).sum(-1)
        alpha_u = torch.atan2(wn, wa)
        cl, dcl = self._section_cl_unknowns(alpha_u, a0, aL0, shift)
        R = 2.0 * wxn * g_unknown - scale * cl
        with torch.no_grad():
            Vt = self._tensor_unknowns(cp_u.detach(), s_a.detach(), s_b.detach(), s_ate.detach(),
                                       s_bte.detach(), s_wd.detach(), rc2_src.detach(), s_group,
                                       s_sign, s_col, n_u, tg_group,
                                       None if tg_rc is None else tg_rc.detach(),
                                       own=b["unknown_panels"])
            g_d = g_unknown.detach()
            wxd, wxnd, wnd, wad = wx.detach(), wxn.detach(), wn.detach(), wa.detach()
            u = wxd / torch.clamp_min(wxnd, 1e-300)[:, None]
            dwxn = torch.einsum("ijk,ik->ij", Vt, torch.linalg.cross(dl_u.detach(), u))
            vn = torch.einsum("ijk,ik->ij", Vt, normal_u.detach())
            va = torch.einsum("ijk,ik->ij", Vt, cdir_u.detach())
            denom = torch.clamp_min(wad * wad + wnd * wnd, 1e-300)
            dalpha = (wad[:, None] * vn - wnd[:, None] * va) / denom[:, None]
            J = 2.0 * g_d[:, None] * dwxn - (scale.detach() * dcl)[:, None] * dalpha
            J = J + torch.diag(2.0 * wxnd)
            b["_nl_slope"] = dcl.detach()
            b["_nl_residual"] = (R.detach() / scale.detach())
        rhs = J @ g_d - R
        return J, rhs, alpha_u

    def _src_sum(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum *x* along *dim* with the accumulation rule of the model."""
        mode = self.b.get("accum", "sequential")
        if mode == "pairwise":
            return pairwise_sum(x, dim)
        if mode == "kahan":
            return kahan_sum(x, dim)
        return x.sum(dim)

    def _vjp_backend(self) -> str | None:
        """Return the compiled VJP backend of the model, or None for the PyTorch path.

        The adjoint functions set ``build["vjp_backend"]``. The compiled
        kernels add the source terms in their own order, so a model with an
        accumulation rule other than ``"sequential"`` uses the PyTorch path.
        """
        backend = self.b.get("vjp_backend")
        if backend is None or self.b.get("accum", "sequential") != "sequential":
            return None
        return str(backend)

    def _topology(self, m: int, grp: torch.Tensor, col: torch.Tensor | None, n_u: int,
                  tg_group: torch.Tensor | None, tg_rc: torch.Tensor | None,
                  own: torch.Tensor | None, edge: torch.Tensor | None):  # type: ignore[no-untyped-def]
        """Return the fixed integer data of one kernel product."""
        from ventorum.sensitivity.kernels import Topology

        src = self.b["_src_ids"]
        use_tg = tg_group is not None and tg_rc is not None
        return Topology(grp, tg_group if use_tg else None, own, edge, src["panel"], src["edge_a"],
                        src["edge_b"], col, n_u, use_tg, m, self.device,
                        src.get("kink"), src.get("edge_ka"), src.get("edge_kb"), src.get("leg_next"))

    def _horseshoe(
        self,
        P: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_te: torch.Tensor,
        b_te: torch.Tensor,
        d: torch.Tensor,
        rc2: torch.Tensor,
        own: torch.Tensor | None = None,
        edge: torch.Tensor | None = None,
        src: dict | None = None,
        kink: tuple | None = None,
    ) -> torch.Tensor:
        """Unit-circulation velocity of bent horseshoes at P (differentiable).

        Parameters
        ----------
        P : torch.Tensor
            Target points [m], shape (rows, 3).
        a, b, a_te, b_te : torch.Tensor
            Source horseshoe corners [m], shape (n_src, 3).
        d : torch.Tensor
            Trailing directions of the sources, shape (n_src, 3).
        rc2 : torch.Tensor
            Squared core radii [m^2], shape (rows, n_src).
        own : torch.Tensor or None
            Per target the panel whose bound segment holds the target, or -1.
        edge : torch.Tensor or None
            Per target the lattice edge whose line holds the target, or -1.
        src : dict or None
            Topology ids of the sources (panel, edge_a, edge_b, and kink,
            edge_ka, edge_kb with leg nodes); None takes the ids of the last
            forward pass.
        kink : tuple or None
            Leg nodes ``(a_k, b_k)`` [m], shape (n_src, 3), of the sources
            flagged in ``src["kink"]`` (see
            :class:`ventorum.aero.vortex.HorseshoeSet`), or None. With leg
            chains (``src["leg_next"]``) also the ends ``(a_e, b_e)`` of the
            own leg segments.

        Returns
        -------
        torch.Tensor
            Velocity per unit circulation [1/m], shape (rows, n_src, 3).

        Notes
        -----
        A target that lies on a source segment by construction (a load
        point on its own bound segment, a leg mid-point on the legs of the
        same edge line) stays on that segment for every node motion, so the
        induced velocity of that pair is identically zero. The kernel value
        there is zero, but its partial derivatives are O(1/rc^2) terms that
        cancel only in exact arithmetic. These pairs are set to zero by the
        mesh topology (index masks), not by a distance threshold: a
        threshold would also remove real collinear terms of other strips.
        """
        if src is None:
            src = self.b.get("_src_ids")
        a_k, b_k = (a_te, b_te) if kink is None else kink[:2]
        a_e, b_e = (a_te, b_te) if kink is None or len(kink) < 4 else kink[2:4]
        nxt = None if src is None or kink is None or len(kink) < 4 else src.get("leg_next")
        bound = seg_vel(P, a, b, rc2)
        leg_a = seg_vel(P, a_k, a, rc2)
        leg_b = seg_vel(P, b, b_k, rc2)
        if src is not None and own is not None:
            keep = own[:, None] != src["panel"][None, :]
            bound = torch.where(keep[..., None], bound, torch.zeros_like(bound))
        if src is not None and edge is not None:
            keep_a = edge[:, None] != src["edge_a"][None, :]
            keep_b = edge[:, None] != src["edge_b"][None, :]
            leg_a = torch.where(keep_a[..., None], leg_a, torch.zeros_like(leg_a))
            leg_b = torch.where(keep_b[..., None], leg_b, torch.zeros_like(leg_b))
        if kink is not None:
            # Aft segments of the legs with a node (zero on the other sources).
            idx = torch.nonzero(src["kink"] != 0).reshape(-1)
            rck = rc2[:, idx]
            aft_a = seg_vel(P, a_e[idx], a_k[idx], rck)
            aft_b = seg_vel(P, b_k[idx], b_e[idx], rck)
            if edge is not None:
                ka = edge[:, None] != src["edge_ka"][None, idx]
                kb = edge[:, None] != src["edge_kb"][None, idx]
                aft_a = torch.where(ka[..., None], aft_a, torch.zeros_like(aft_a))
                aft_b = torch.where(kb[..., None], aft_b, torch.zeros_like(aft_b))
            leg_a = leg_a.index_add(1, idx, aft_a)
            leg_b = leg_b.index_add(1, idx, aft_b)
        ray_b = ray_vel(P, b_te, d, rc2)
        ray_a = ray_vel(P, a_te, d, rc2)
        mode = self.b.get("accum", "sequential")
        tail = None if nxt is None else _chain_suffix(leg_a + leg_b, nxt)
        if mode == "sequential":
            # Same order of addition as horseshoe_vel.
            v = (bound + leg_a + leg_b) + ray_b - ray_a
            return v if tail is None else v + tail
        terms = (bound, leg_a, leg_b, ray_b, -ray_a) + (() if tail is None else (tail,))
        terms = torch.stack(terms, dim=0)
        if mode == "pairwise":
            return pairwise_sum(terms, 0)
        return kahan_sum(terms, 0)

    def _rc2_chunk(
        self,
        rc2_src: torch.Tensor,
        src_group: torch.Tensor,
        tg_group: torch.Tensor | None,
        tg_rc: torch.Tensor | None,
        i0: int,
        i1: int,
    ) -> torch.Tensor:
        """Return the (rows, sources) squared core radii of rows i0:i1."""
        if tg_group is None or tg_rc is None:
            return rc2_src[None, :].expand(i1 - i0, -1)
        other = tg_group[i0:i1, None] != src_group[None, :]
        return rc2_src[None, :].expand(i1 - i0, -1) + torch.where(
            other, tg_rc[i0:i1, None] ** 2, torch.zeros_like(tg_rc[i0:i1, None]))

    def _normal_matrix(self, P: torch.Tensor, N: torch.Tensor, a: torch.Tensor,
                       bv: torch.Tensor, a_te: torch.Tensor, b_te: torch.Tensor,
                       wd: torch.Tensor, rc2_src: torch.Tensor, grp: torch.Tensor,
                       sign: torch.Tensor, col: torch.Tensor, n_u: int,
                       tg_group: torch.Tensor | None, tg_rc: torch.Tensor | None,
                       own: torch.Tensor | None = None,
                       edge: torch.Tensor | None = None) -> torch.Tensor:
        """Normal-velocity influence matrix with folded columns (chunked)."""
        m = P.shape[0]
        backend = self._vjp_backend()
        if backend is not None:
            from ventorum.sensitivity.kernels import horseshoe_rows

            topo = self._topology(m, grp, col, n_u, tg_group, tg_rc, own, edge)
            tg_rc2 = tg_rc * tg_rc if topo.use_tg else None
            kink = self.b.get("_src_kinks")
            return horseshoe_rows(backend, topo, P, N, a, bv, a_te, b_te, wd, rc2_src, tg_rc2, sign,
                                  *(kink if kink is not None else (None, None)))
        src = self.b.get("_src_ids")
        tgr = tg_rc if tg_group is not None else None
        kink = self.b.get("_src_kinks")
        nk = 0 if kink is None else len(kink)

        def _fn(i0: int, i1: int, P: torch.Tensor, N: torch.Tensor, a: torch.Tensor,
                bv: torch.Tensor, a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                rc2_src: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            kn = tuple(rest[:nk]) if nk else None
            rest = rest[nk:]
            tg = rest[0] if rest else None
            rc2 = self._rc2_chunk(rc2_src, grp, tg_group, tg, i0, i1)
            v = self._horseshoe(P[i0:i1], a, bv, a_te, b_te, wd, rc2,
                                _rows(own, i0, i1), _rows(edge, i0, i1), src=src, kink=kn)
            a_src = (v * N[i0:i1, None, :]).sum(-1) * sign[None, :]
            acc = torch.zeros((i1 - i0, n_u), dtype=self.dtype, device=self.device)
            return acc.index_add(1, col, a_src)

        args = ((P, N, a, bv, a_te, b_te, wd, rc2_src) + (tuple(kink) if nk else ())
                + ((tgr,) if tgr is not None else ()))
        return _chunked(_fn, m, chunk_rows(m, a.shape[0]), *args)

    def _tensor_unknowns(self, P: torch.Tensor, a: torch.Tensor, bv: torch.Tensor,
                         a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                         rc2_src: torch.Tensor, grp: torch.Tensor, sign: torch.Tensor,
                         col: torch.Tensor, n_u: int, tg_group: torch.Tensor | None,
                         tg_rc: torch.Tensor | None, own: torch.Tensor | None = None,
                         edge: torch.Tensor | None = None) -> torch.Tensor:
        """Velocity per unknown at P (chunked)."""
        m = P.shape[0]
        src = self.b.get("_src_ids")
        tgr = tg_rc if tg_group is not None else None
        kink = self.b.get("_src_kinks")
        nk = 0 if kink is None else len(kink)

        def _fn(i0: int, i1: int, P: torch.Tensor, a: torch.Tensor, bv: torch.Tensor,
                a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                rc2_src: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            kn = tuple(rest[:nk]) if nk else None
            rest = rest[nk:]
            tg = rest[0] if rest else None
            rc2 = self._rc2_chunk(rc2_src, grp, tg_group, tg, i0, i1)
            v = self._horseshoe(P[i0:i1], a, bv, a_te, b_te, wd, rc2,
                                _rows(own, i0, i1), _rows(edge, i0, i1), src=src, kink=kn)
            v = v * sign[None, :, None]
            acc = torch.zeros((i1 - i0, n_u, 3), dtype=self.dtype, device=self.device)
            return acc.index_add(1, col, v)

        args = ((P, a, bv, a_te, b_te, wd, rc2_src) + (tuple(kink) if nk else ())
                + ((tgr,) if tgr is not None else ()))
        return _chunked(_fn, m, chunk_rows(m, a.shape[0]), *args)

    def _induced(self, P: torch.Tensor, a: torch.Tensor, bv: torch.Tensor,
                 a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                 rc2_src: torch.Tensor, grp: torch.Tensor, g_src: torch.Tensor,
                 tg_group: torch.Tensor | None, tg_rc: torch.Tensor | None,
                 own: torch.Tensor | None = None,
                 edge: torch.Tensor | None = None) -> torch.Tensor:
        """Total induced velocity at P for source strengths *g_src* (chunked)."""
        m = P.shape[0]
        backend = self._vjp_backend()
        if backend is not None:
            from ventorum.sensitivity.kernels import horseshoe_induced

            topo = self._topology(m, grp, None, a.shape[0], tg_group, tg_rc, own, edge)
            tg_rc2 = tg_rc * tg_rc if topo.use_tg else None
            kink = self.b.get("_src_kinks")
            return horseshoe_induced(backend, topo, P, a, bv, a_te, b_te, wd, rc2_src, tg_rc2, g_src,
                                     *(kink if kink is not None else (None, None)))
        src = self.b.get("_src_ids")
        tgr = tg_rc if tg_group is not None else None
        kink = self.b.get("_src_kinks")
        nk = 0 if kink is None else len(kink)

        def _fn(i0: int, i1: int, P: torch.Tensor, a: torch.Tensor, bv: torch.Tensor,
                a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                rc2_src: torch.Tensor, g_src: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            kn = tuple(rest[:nk]) if nk else None
            rest = rest[nk:]
            tg = rest[0] if rest else None
            rc2 = self._rc2_chunk(rc2_src, grp, tg_group, tg, i0, i1)
            v = self._horseshoe(P[i0:i1], a, bv, a_te, b_te, wd, rc2,
                                _rows(own, i0, i1), _rows(edge, i0, i1), src=src, kink=kn)
            return self._src_sum(v * g_src[None, :, None], 1)

        args = ((P, a, bv, a_te, b_te, wd, rc2_src, g_src) + (tuple(kink) if nk else ())
                + ((tgr,) if tgr is not None else ()))
        return _chunked(_fn, m, chunk_rows(m, a.shape[0]), *args)

    def _trefftz(self, te_l: torch.Tensor, te_r: torch.Tensor, sg: torch.Tensor,
                 wd: torch.Tensor, rho: torch.Tensor, frac: torch.Tensor,
                 normal_g: torch.Tensor | None, offset: torch.Tensor | None,
                 n_s: int, width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trefftz induced drag, normal wash and segment lengths (differentiable)."""
        d = wd / torch.clamp_min(torch.linalg.norm(wd), 1e-300)
        dot_l = te_l @ d
        dot_r = te_r @ d
        pl = te_l - dot_l[:, None] * d
        pr = te_r - dot_r[:, None] * d
        seg = pr - pl
        seg_len = torch.linalg.norm(seg, dim=1)
        q = pl + frac[:, None] * seg
        qn = torch.linalg.cross(d.expand(n_s, 3), seg)
        qn = qn / torch.clamp_min(torch.linalg.norm(qn, dim=1, keepdim=True), 1e-300)
        p_vort = torch.cat((pr, pl), dim=0)
        g_vort = torch.cat((sg, -sg), dim=0)
        groups = self.b["strip_groups"]
        src_group = torch.cat((groups, groups), dim=0)
        rc = 0.01 * width
        rc_vort = torch.cat((rc, rc), dim=0)
        if normal_g is not None:
            def _refl(p: torch.Tensor) -> torch.Tensor:
                hh = p @ normal_g - offset
                return p - 2.0 * hh[:, None] * normal_g

            pts = _refl(torch.cat((te_r, te_l), dim=0))
            p_img = pts - (pts @ d)[:, None] * d
            p_vort = torch.cat((p_vort, p_img), dim=0)
            g_vort = torch.cat((g_vort, -g_vort), dim=0)
            src_group = torch.cat((src_group, src_group), dim=0)
            rc_vort = torch.cat((rc_vort, rc_vort), dim=0)
        w_n = self._trefftz_wash(q, qn, p_vort, g_vort, d, rc_vort, src_group, width)
        di = -0.5 * rho * (sg * w_n * seg_len).sum()
        return di, w_n, seg_len

    def _trefftz_wash(self, q: torch.Tensor, qn: torch.Tensor, p: torch.Tensor,
                      gamma: torch.Tensor, d: torch.Tensor, rc: torch.Tensor,
                      src_group: torch.Tensor, width: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the normal wash of 2-D point vortices (chunked, differentiable)."""
        m = q.shape[0]
        n = p.shape[0]
        tg_group = self.b["strip_groups"]
        tg_rc = 0.5 * width
        rc2 = rc * rc
        tg_rc2 = tg_rc * tg_rc
        backend = self._vjp_backend()
        if backend is not None:
            from ventorum.sensitivity.kernels import trefftz_wash

            return trefftz_wash(backend, (src_group, tg_group, True), q, qn, p, gamma, d, rc2, tg_rc2)

        def _fn(i0: int, i1: int, q: torch.Tensor, qn: torch.Tensor, p: torch.Tensor,
                gamma: torch.Tensor, d: torch.Tensor, rc2: torch.Tensor,
                tg_rc2: torch.Tensor) -> torch.Tensor:
            r = q[i0:i1, None, :] - p[None, :, :]
            vx = d[1] * r[..., 2] - d[2] * r[..., 1]
            vy = d[2] * r[..., 0] - d[0] * r[..., 2]
            vz = d[0] * r[..., 1] - d[1] * r[..., 0]
            other = tg_group[i0:i1, None] != src_group[None, :]
            c2 = rc2[None, :] + torch.where(other, tg_rc2[i0:i1, None],
                                            torch.zeros_like(tg_rc2[i0:i1, None]))
            k = (1.0 / (2.0 * math.pi)) * gamma[None, :] / ((r * r).sum(-1) + c2)
            qq = qn[i0:i1, None, :]
            wn = (vx * qq[..., 0] + vy * qq[..., 1] + vz * qq[..., 2]) * k
            return self._src_sum(wn, 1)

        return _chunked(_fn, m, max(1, chunk_rows(m, n)), q, qn, p, gamma, d, rc2, tg_rc2)

    def _axes_moments(self, Cl: torch.Tensor, Cm: torch.Tensor, Cn: torch.Tensor,
                      alpha: torch.Tensor, beta: torch.Tensor,
                      b_ref: torch.Tensor, c_ref: torch.Tensor,
                      axes: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moment coefficients in *axes* (differentiable in alpha and beta)."""
        if axes == "body":
            return Cl, Cm, Cn
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        r_bs = torch.stack((
            torch.stack((ca, torch.zeros_like(ca), sa)),
            torch.stack((torch.zeros_like(ca), torch.ones_like(ca), torch.zeros_like(ca))),
            torch.stack((-sa, torch.zeros_like(ca), ca)),
        ))
        if axes == "stability":
            rel = r_bs
        else:
            cb = torch.cos(beta)
            sb = torch.sin(beta)
            r_sw = torch.stack((
                torch.stack((cb, sb, torch.zeros_like(cb))),
                torch.stack((-sb, cb, torch.zeros_like(cb))),
                torch.stack((torch.zeros_like(cb), torch.zeros_like(cb), torch.ones_like(cb))),
            ))
            rel = r_sw @ r_bs
        vec = torch.stack((Cl * b_ref, Cm * c_ref, Cn * b_ref))
        out = rel @ vec
        return out[0] / b_ref, out[1] / c_ref, out[2] / b_ref

