# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Differentiable reference copy of the lattice solvers for adjoint gradients.

The model rebuilds the lattice from the edge nodes of each surface, the
system rows and the outputs in PyTorch, with the same formulas as the
production code. The mesh topology is fixed. The O(n^2) work (system rows,
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


def require_torch() -> None:
    """Raise when PyTorch is not available."""
    if not _HAVE_TORCH:
        raise RuntimeError("The gradient module needs the torch package.")


def chunk_budget() -> float:
    """Return the chunk budget [values] from VENTORUM_GRAD_CHUNK or the default."""
    raw = os.environ.get("VENTORUM_GRAD_CHUNK", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            return _DEFAULT_CHUNK_VALUES
        if value > 0.0:
            return value
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


def seg_vel(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rc2: torch.Tensor) -> torch.Tensor:
    """Velocity of unit finite segments A -> B at P (differentiable)."""
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = B[None, :, :] - A[None, :, :]
    cx = r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]
    cy = r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]
    cz = r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz
    r0sq = r0[..., 0] ** 2 + r0[..., 1] ** 2 + r0[..., 2] ** 2
    n1 = torch.sqrt((r1 * r1).sum(-1) + 1e-300)
    n2 = torch.sqrt((r2 * r2).sum(-1) + 1e-300)
    t1 = r1 / n1[..., None]
    t2 = r2 / n2[..., None]
    proj_in = (
        r0[..., 0] * (t1[..., 0] - t2[..., 0])
        + r0[..., 1] * (t1[..., 1] - t2[..., 1])
        + r0[..., 2] * (t1[..., 2] - t2[..., 2])
    )
    # The same projection without cancellation where r1 . r2 > 0 (target
    # not over the segment): r0 . (t1 - t2) = |r1 x r2|^2 (n1 + n2) /
    # (n1 n2 (n1 n2 + r1 . r2)). The direct form loses digits there (t1 and
    # t2 are almost equal), and its partial derivatives carry O(1/rc^2)
    # round-off noise at points near the extension of the segment.
    dot12 = (r1 * r2).sum(-1)
    n12 = n1 * n2
    out_side = dot12 > 0.0
    den_out = torch.where(out_side, n12 * (n12 + dot12), torch.ones_like(n12))
    proj_out = cr2 * (n1 + n2) / den_out
    proj = torch.where(out_side, proj_out, proj_in)
    denom = cr2 + rc2 * r0sq
    ok = denom > 1e-300
    safe = torch.where(ok, denom, torch.ones_like(denom))
    k = torch.where(ok, (1.0 / (4.0 * math.pi)) * proj / safe, torch.zeros_like(proj))
    return torch.stack((cx * k, cy * k, cz * k), dim=-1)


def ray_vel(P: torch.Tensor, A: torch.Tensor, d: torch.Tensor, rc2: torch.Tensor) -> torch.Tensor:
    """Velocity of unit semi-infinite lines from A along d at P (differentiable)."""
    r = P[:, None, :] - A[None, :, :]
    dd = d[None, :, :]
    cx = dd[..., 1] * r[..., 2] - dd[..., 2] * r[..., 1]
    cy = dd[..., 2] * r[..., 0] - dd[..., 0] * r[..., 2]
    cz = dd[..., 0] * r[..., 1] - dd[..., 1] * r[..., 0]
    cr2 = cx * cx + cy * cy + cz * cz
    rn = torch.sqrt((r * r).sum(-1) + 1e-300)
    dr = (dd * r).sum(-1)
    # 1 + cos(t) for a unit d. Behind the start point (d . r < 0) the same
    # value without cancellation: |d x r|^2 / (rn (rn - d . r)).
    behind = dr < 0.0
    den_b = torch.where(behind, rn * (rn - dr), torch.ones_like(rn))
    one_cos = torch.where(behind, cr2 / den_b, 1.0 + dr / rn)
    denom = cr2 + rc2
    ok = denom > 1e-300
    safe = torch.where(ok, denom, torch.ones_like(denom))
    k = torch.where(ok, (1.0 / (4.0 * math.pi)) * one_cos / safe, torch.zeros_like(one_cos))
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
            System matrix, shape (n, n).
        rhs : torch.Tensor
            Right-hand side, shape (n,).
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

        # --- lattice from the edge nodes ------------------------------------
        le_l, le_r, te_l, te_r = self._strips(t)
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
        normal_bc_strip = rod_rot(normal, span_yz, -aL0)

        a0 = t["a0"]
        claf = torch.clamp(a0 / (2.0 * math.pi), 0.2, 2.0)
        xi = b["xi"]
        frac = b["frac"]
        n_chord = b["n_chord"]
        pa_list, pb_list, pcp_list = [], [], []
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
        a_te = te_l[panel_strip]
        b_te = te_r[panel_strip]
        d_xi = xi[:, 1:] - xi[:, :-1]
        panel_length = (d_xi * chord[:, None]).reshape(-1)
        rc_panel = 0.01 * torch.minimum(width[panel_strip], panel_length)

        n_bc = normal_bc_strip[panel_strip]
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
        if h is None:
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
        if h is not None:
            img = torch.full((n_p,), -2, dtype=torch.int64, device=self.device)
            pid = torch.cat((pid, img))
            ea = torch.cat((ea, img))
            eb = torch.cat((eb, img))
        b["_src_ids"] = {"panel": pid, "edge_a": ea, "edge_b": eb}
        panel_ids = torch.arange(n_p, dtype=torch.int64, device=self.device)
        edge_l_panel = edges[panel_strip, 0]
        edge_r_panel = edges[panel_strip, 1]

        rc2_src = s_rc * s_rc
        tg_group = tgt_group_unk
        tg_rc = tgt_rc_unk

        # --- system matrix and right-hand side -------------------------------
        if b["collocation"] == "vlm":
            A = self._normal_matrix(cp_u, n_bc_u, s_a, s_b, s_ate, s_bte, s_wd,
                                    rc2_src, s_group, s_sign, s_col, n_u,
                                    tg_group, tg_rc)
            rhs = -V * (n_bc_u * d).sum(-1)
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
        U = (V * d).expand_as(v_ind)
        f_panel = rho * g_panel[:, None] * torch.linalg.cross(U + v_ind, l_vec)
        m_geo = torch.linalg.cross(mid - rp, f_panel).sum(0)
        if use_leg:
            leg_a_mid = 0.5 * (a_te + a)
            leg_b_mid = 0.5 * (bv + b_te)
            leg_start = torch.cat((a_te, bv), dim=0)
            leg_end = torch.cat((a, b_te), dim=0)
            leg_mid = torch.cat((leg_a_mid, leg_b_mid), dim=0)
            leg_vec = leg_end - leg_start
            strip_normal = normal[panel_strip]
            leg_normal = torch.cat((strip_normal, strip_normal), dim=0)
            leg_tgt_group = torch.cat((tgt_group_panel, tgt_group_panel), dim=0)
            leg_tgt_rc = torch.cat((tgt_rc_panel, tgt_rc_panel), dim=0)
            if h is None:
                g_leg_src = g_panel * s_sign
            else:
                g_leg_src = torch.cat((g_panel, g_panel), dim=0) * s_sign
            if fold:
                # Right panels only; left panels mirror with the leg swap:
                # the a-leg of a left panel mirrors the b-leg of its mirror.
                la_e = leg_a_mid[right]
                lb_e = leg_b_mid[right]
                leg_e = torch.cat((la_e, lb_e), dim=0)
                ltgt_g = torch.cat((tgt_ge, tgt_ge), dim=0)
                ltgt_r = torch.cat((tgt_re, tgt_re), dim=0)
                # A leg mid-point lies on the line of its edge, so also on
                # the legs of the other panels that use the same edge.
                ledge = torch.cat((edge_l_panel[right], edge_r_panel[right]), dim=0)
                v_leg_e = self._induced(leg_e, s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                        s_group, g_leg_src, ltgt_g, ltgt_r, edge=ledge)
                n_r = right.shape[0]
                pos = pos_all
                mir_right = pos[pm[left]]
                v_leg = torch.empty_like(leg_mid)
                v_leg[right] = v_leg_e[:n_r]
                v_leg[n_p + right] = v_leg_e[n_r:]
                v_leg[left] = v_leg_e[n_r + mir_right] * mir
                v_leg[n_p + left] = v_leg_e[mir_right] * mir
            else:
                ledge = torch.cat((edge_l_panel, edge_r_panel), dim=0)
                v_leg = self._induced(leg_mid, s_a, s_b, s_ate, s_bte, s_wd, rc2_src,
                                      s_group, g_leg_src, leg_tgt_group, leg_tgt_rc,
                                      edge=ledge)
            U_leg = (V * d).expand_as(v_leg)
            V_leg = U_leg + v_leg
            Vn = (V_leg * leg_normal).sum(-1)
            V_leg = V_leg - Vn[:, None] * leg_normal
            g_leg_only = torch.cat((g_panel, g_panel), dim=0)
            f_leg = rho * g_leg_only[:, None] * torch.linalg.cross(V_leg, leg_vec)
            m_geo = m_geo + torch.linalg.cross(leg_mid - rp, f_leg).sum(0)
            f_panel = f_panel + f_leg[:n_p] + f_leg[n_p:]
        f_strip = f_panel.reshape(n_s, n_chord, 3).sum(1)
        strip_gamma = g_panel.reshape(n_s, n_chord).sum(1)

        cl_strip = 2.0 * strip_gamma / (V * chord)
        nd = (normal * d).sum(-1)
        cd_ = (cdir * d).sum(-1)
        alpha_geom = torch.atan2(nd, cd_)
        alpha_eff = aL0 + cl_strip / torch.clamp_min(a0, 1e-9)
        cd0 = t["Cd0"]
        cm0 = t["Cm0"]
        q_inf = 0.5 * rho * V * V
        S_ref = t["S_ref"]
        b_ref = t["b_ref"]
        c_ref = t["c_ref"]
        qS = q_inf * S_ref
        d_p = q_inf * chord * width * cd0
        f_prof = d_p[:, None] * d.expand(n_s, 3)
        m_geo = m_geo + torch.linalg.cross(qc_mid - rp, f_prof).sum(0)
        n_lift = torch.linalg.cross(d.expand(n_s, 3), dl)
        n_lift = n_lift / torch.clamp_min(torch.linalg.norm(n_lift, dim=1, keepdim=True), 1e-300)
        axis = torch.linalg.cross(n_lift, d.expand(n_s, 3))
        axis = axis / torch.clamp_min(torch.linalg.norm(axis, dim=1, keepdim=True), 1e-300)
        m_geo = m_geo + ((q_inf * chord * chord * width * cm0)[:, None] * axis).sum(0)

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

    def _strips(self, t: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (le_l, le_r, te_l, te_r) of all strips from the node inputs."""
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
        return edge_le[idx[:, 0]], edge_le[idx[:, 1]], edge_te[idx[:, 0]], edge_te[idx[:, 1]]

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
                        src["edge_b"], col, n_u, use_tg, m, self.device)

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
            Topology ids of the sources (panel, edge_a, edge_b); None takes
            the ids of the last forward pass.

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
        bound = seg_vel(P, a, b, rc2)
        leg_a = seg_vel(P, a_te, a, rc2)
        leg_b = seg_vel(P, b, b_te, rc2)
        if src is None:
            src = self.b.get("_src_ids")
        if src is not None and own is not None:
            keep = own[:, None] != src["panel"][None, :]
            bound = torch.where(keep[..., None], bound, torch.zeros_like(bound))
        if src is not None and edge is not None:
            keep_a = edge[:, None] != src["edge_a"][None, :]
            keep_b = edge[:, None] != src["edge_b"][None, :]
            leg_a = torch.where(keep_a[..., None], leg_a, torch.zeros_like(leg_a))
            leg_b = torch.where(keep_b[..., None], leg_b, torch.zeros_like(leg_b))
        ray_b = ray_vel(P, b_te, d, rc2)
        ray_a = ray_vel(P, a_te, d, rc2)
        mode = self.b.get("accum", "sequential")
        if mode == "sequential":
            # Same order of addition as horseshoe_vel.
            return (bound + leg_a + leg_b) + ray_b - ray_a
        terms = torch.stack((bound, leg_a, leg_b, ray_b, -ray_a), dim=0)
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
            return horseshoe_rows(backend, topo, P, N, a, bv, a_te, b_te, wd, rc2_src, tg_rc2, sign)
        src = self.b.get("_src_ids")
        tgr = tg_rc if tg_group is not None else None

        def _fn(i0: int, i1: int, P: torch.Tensor, N: torch.Tensor, a: torch.Tensor,
                bv: torch.Tensor, a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                rc2_src: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            tg = rest[0] if rest else None
            rc2 = self._rc2_chunk(rc2_src, grp, tg_group, tg, i0, i1)
            v = self._horseshoe(P[i0:i1], a, bv, a_te, b_te, wd, rc2,
                                _rows(own, i0, i1), _rows(edge, i0, i1), src=src)
            a_src = (v * N[i0:i1, None, :]).sum(-1) * sign[None, :]
            acc = torch.zeros((i1 - i0, n_u), dtype=self.dtype, device=self.device)
            return acc.index_add(1, col, a_src)

        args = (P, N, a, bv, a_te, b_te, wd, rc2_src) + ((tgr,) if tgr is not None else ())
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

        def _fn(i0: int, i1: int, P: torch.Tensor, a: torch.Tensor, bv: torch.Tensor,
                a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                rc2_src: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            tg = rest[0] if rest else None
            rc2 = self._rc2_chunk(rc2_src, grp, tg_group, tg, i0, i1)
            v = self._horseshoe(P[i0:i1], a, bv, a_te, b_te, wd, rc2,
                                _rows(own, i0, i1), _rows(edge, i0, i1), src=src)
            v = v * sign[None, :, None]
            acc = torch.zeros((i1 - i0, n_u, 3), dtype=self.dtype, device=self.device)
            return acc.index_add(1, col, v)

        args = (P, a, bv, a_te, b_te, wd, rc2_src) + ((tgr,) if tgr is not None else ())
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
            return horseshoe_induced(backend, topo, P, a, bv, a_te, b_te, wd, rc2_src, tg_rc2, g_src)
        src = self.b.get("_src_ids")
        tgr = tg_rc if tg_group is not None else None

        def _fn(i0: int, i1: int, P: torch.Tensor, a: torch.Tensor, bv: torch.Tensor,
                a_te: torch.Tensor, b_te: torch.Tensor, wd: torch.Tensor,
                rc2_src: torch.Tensor, g_src: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            tg = rest[0] if rest else None
            rc2 = self._rc2_chunk(rc2_src, grp, tg_group, tg, i0, i1)
            v = self._horseshoe(P[i0:i1], a, bv, a_te, b_te, wd, rc2,
                                _rows(own, i0, i1), _rows(edge, i0, i1), src=src)
            return self._src_sum(v * g_src[None, :, None], 1)

        args = (P, a, bv, a_te, b_te, wd, rc2_src, g_src) + ((tgr,) if tgr is not None else ())
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

