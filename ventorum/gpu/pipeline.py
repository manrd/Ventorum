# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Dispatch of the lattice solves to the GPU pipelines.

:func:`solve_batch` is called by
:meth:`ventorum.solvers.lattice_base.LatticeSolver.solve_batch` and
``solve_lattice`` before the CPU path. It returns None when the GPU is not
selected for the solve or when the GPU pipelines do not support it; the
caller then uses the CPU path. The setup of each case (ground plane,
validity checks, notes) and the result objects are the CPU code, so a GPU
solve gives the same result objects as a CPU solve.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from ventorum import gpu

# A Newton solve of a continuation pass gives the same root as the pass
# before it when the section lift coefficient of no strip changes more than
# this value (2 |dG| / (V c)). Two converged solves of one root differ by
# about the solver tolerance (in section lift, default 1e-6); two roots past
# the maximum lift differ by much more.
CONTINUATION_SAME_ROOT = 1.0e-3


def work_estimate(lattice, n_cases: int, in_ground: bool) -> float:
    """Return the horseshoe evaluations of the system of a batch (the measure of the ``"auto"`` device).

    The system of one case has one evaluation per pair of control point and
    panel (twice in ground effect, with the images). A symmetric case
    evaluates half of the control points, but the loads add about as much
    work again, so the full count is used.
    """
    n_p = int(lattice.n_panels)
    return float(n_cases) * n_p * n_p * (2.0 if in_ground else 1.0)


def solve_batch(solver, lattice, conditions: list, settings, S_ref: float, b_ref: float, c_ref: float,
                ref_point, main_surface: int, continuation: bool, t0: float, gamma0=None, grounds=None,
                return_reason: bool = False):
    """Solve the cases on the GPU, or return None (the caller then solves them on the CPU).

    The arguments are those of
    :meth:`ventorum.solvers.lattice_base.LatticeSolver.solve_batch`;
    *gamma0* is the start circulation of every panel of a single nonlinear
    solve (see ``solve_lattice``), and *grounds* the explicit ground plane
    of each case (None: from ``condition.h``).

    Parameters
    ----------
    return_reason : bool, optional
        If False (default), return only the results (or None). If True,
        return the tuple ``(results, reason)``.

    Returns
    -------
    results : list of SolverResult or None
        One result per condition, in order, or None when the cases go to
        the CPU path. This is the only return value when *return_reason* is
        False.
    reason : str
        One plain sentence that gives the reason of the GPU decision or of
        the fallback to the CPU path, for example
        ``"called inside a CPU worker thread"``. Returned only when
        *return_reason* is True.
    """
    if gpu.get_device() == "cpu":
        return (None, gpu._REASON_DEVICE_CPU) if return_reason else None
    if solver.name not in ("linear", "nonlinear", "vlm"):
        return (None, f"solver '{solver.name}' is not supported on GPU") if return_reason else None
    from ventorum.utils.parallel import _local as _par_local

    if gpu.get_device() == "auto" and getattr(_par_local, "worker_threads", None) is not None:
        # A worker of a CPU pool (cases in parallel): the pool was chosen for
        # the CPU, and many small GPU solves from several threads are slow.
        return (None, "called inside a CPU worker thread") if return_reason else None
    in_ground = any(c.h is not None for c in conditions) or (grounds is not None and any(g is not None for g in grounds))
    work = work_estimate(lattice, len(conditions), in_ground)
    use_gpu_dec, use_gpu_reason = gpu.use_gpu(work, solver.name, len(conditions),
                                              lattice.n_panels, return_reason=True)
    if not use_gpu_dec:
        return (None, use_gpu_reason) if return_reason else None
    from ventorum.aero.system import is_symmetric_condition
    from ventorum.solvers.core import _unknown_map

    precision = gpu.get_precision()
    K = len(conditions)
    use_sym = getattr(settings, "use_symmetry", True)
    explicit = [None] * K if grounds is None else list(grounds)
    setups = [solver._case_setup(lattice, c, settings, c_ref, g, ref_point) for c, g in zip(conditions, explicit)]
    # Groups of cases with the same unknown map, ground presence and symmetry:
    # each group is one GPU batch.
    groups: dict[tuple, list[int]] = {}
    for k, (c, st) in enumerate(zip(conditions, setups)):
        umap = _unknown_map(lattice, c, st[0], use_sym)
        key = (id(umap), st[0] is not None, is_symmetric_condition(c, st[0]))
        groups.setdefault(key, []).append(k)
    if len(groups) > 1 and continuation and solver.name == "nonlinear":
        return (None, "continuation needs the cases in order: the CPU path") if return_reason else None
    solved = []
    for idx in groups.values():
        sub, group_reason = _solve_group(solver, lattice, [conditions[k] for k in idx], [setups[k] for k in idx], settings,
                                         S_ref, b_ref, c_ref, ref_point, continuation, gamma0, precision, use_sym)
        if sub is None:
            return (None, group_reason) if return_reason else None
        solved.append((idx, sub))
    share = (time.perf_counter() - t0) / K
    out: list = [None] * K
    reason = use_gpu_reason if gpu.get_device() == "auto" else "device setting is gpu"
    for idx, (arrays, infos, umap) in solved:
        res = _results(solver, lattice, [conditions[k] for k in idx], arrays, [setups[k] for k in idx], umap, b_ref,
                       c_ref, main_surface, t0, share, precision, infos=infos, ref_point=ref_point, S_ref=S_ref,
                       dispatch_reason=reason)
        for k, r in zip(idx, res):
            out[k] = r
    if return_reason:
        return out, reason
    return out


def _solve_group(solver, lattice, conditions: list, setups: list, settings, S_ref: float, b_ref: float,
                 c_ref: float, ref_point, continuation: bool, gamma0, precision: str, use_sym: bool):
    """Solve one group of cases (one unknown map, ground presence and symmetry) on the GPU.

    Returns
    -------
    solved : tuple or None
        ``(arrays, infos, umap)`` for :func:`_results`, or None when the GPU
        pipelines do not support the group.
    reason : str or None
        The reason when *solved* is None (an unknown map or airfoils that
        the GPU pipelines do not support), else None.
    """
    from ventorum.aero.system import is_symmetric_condition
    from ventorum.gpu import engine
    from ventorum.solvers.core import _unknown_map

    K = len(conditions)
    grounds = [st[0] for st in setups]
    wds = np.array([st[3] for st in setups], dtype=float).reshape(K, 3)
    umap = _unknown_map(lattice, conditions[0], grounds[0], use_sym)
    dl = engine.device_lattice(lattice, precision)
    md = dl.unknown_map(lattice, umap)
    if md is None:
        return None, "unknown map is not supported on GPU"
    try:
        engine.check_polars(dl, lattice)
    except TypeError:
        return None, "airfoils are not supported on GPU"
    bd = engine.batch_data(conditions, wds, grounds)
    Dfs, V, rho, alpha, WD = bd["D"], bd["V"], bd["rho"], bd["alpha"], bd["WD"]
    # The host flag avoids a device-to-host sync per solve in rate-free cases.
    omega = bd["omega"] if bd["has_rates"] else None
    cs = engine.CaseData(dl, WD, grounds, GK=bd["GK"], GO=bd["GO"])
    infos = None
    alpha_eff = None
    symmetric = umap.symmetric
    if solver.name == "linear":
        g, v = engine.llt_linear(dl, md, cs, Dfs, V, omega=omega, ref_point=ref_point)
        G, vp = engine.to_panels(lattice, md, g, v)
        v_points = [vp]
    elif solver.name == "nonlinear":
        G, vp, alpha_eff, infos = _nonlinear(solver, engine, dl, md, cs, lattice, conditions, settings, grounds,
                                             wds, Dfs, V, umap, continuation, gamma0, omega=omega,
                                             ref_point=ref_point)
        v_points = [vp]
    else:
        from ventorum.solvers.horseshoe import _has_tabulated

        if _has_tabulated(lattice):
            warnings.warn(
                "The VLM uses the linear part of the tabulated polars (lift slope and zero-lift "
                "angle from a fit). Stall is not modelled; for the start of stall on an unswept wing use solver='nonlinear'.",
                RuntimeWarning, stacklevel=6)
        g = engine.vlm_solve(dl, md, cs, lattice, Dfs, V, omega=omega, ref_point=ref_point)
        G = g[:, md["panel_column"]]
        symmetric = all(is_symmetric_condition(c, gr) for c, gr in zip(conditions, grounds))
        v_points = engine.vlm_load_velocities(dl, lattice, cs, G, symmetric)
    arrays = engine.loads(dl, lattice, G, v_points, Dfs, V, rho, alpha, WD, grounds, S_ref, b_ref, c_ref,
                          ref_point, alpha_eff, use_leg=solver.name == "vlm", extra={"gamma": G}, symmetric=symmetric,
                          GK=bd["GK"], GO=bd["GO"], omega=omega)
    return (arrays, infos, umap), None


# Continuation passes of the GPU nonlinear solves (see _nonlinear).
MAX_CONTINUATION_PASSES = 3


def _nonlinear(solver, engine, dl, md, cs, lattice, conditions, settings, grounds, wds, Dfs, V, umap,
               continuation: bool, gamma0, omega=None, ref_point=None):
    """Newton solves of the nonlinear lifting line on the GPU; return panel circulation, velocity, alpha and infos.

    With *continuation* (a sweep out of ground effect), the CPU solver starts
    each case from the converged circulation of the case before it. The GPU
    solves all cases together, so it repeats the solves: the first pass
    starts every case from the linear lifting line; each further pass starts
    case k from the solution of case k - 1 of the pass before (when that case
    converged), until no case changes its root (at most
    ``MAX_CONTINUATION_PASSES`` passes). The cases that are not settled
    then (a different root, no convergence, or the last few slow cases that
    the GPU hands over) are solved on the CPU, in order, each from the
    solution of the case before it, with the restarts of
    :meth:`ventorum.solvers.nonlinear.NonlinearLLTSolver.solve_circulation`.
    """
    import torch

    from ventorum.solvers.core import SolveInfo

    K = cs.K
    tol = float(settings.tolerance)
    max_it = int(settings.max_iterations)
    handover = max(1, K // 16)
    prep = engine.llt_nonlinear_prepare(dl, md, cs, lattice, Dfs, V, omega=omega, ref_point=ref_point)
    g_start = None
    if gamma0 is not None and K == 1:
        up = md["unknown_panels"]
        g_start = engine._f64(np.asarray(gamma0, dtype=float))[up][None, :]
    out = engine.llt_nonlinear(prep, tol, max_it, g_start, handover=handover)
    ok = out["converged"] & ~out["unresolved"]
    unsettled = torch.zeros(K, dtype=torch.bool, device=ok.device)
    if continuation and K > 1:
        for p in range(MAX_CONTINUATION_PASSES + 1):
            cand = torch.zeros(K, dtype=torch.bool, device=ok.device)
            cand[1:] = ok[:-1]
            if p == MAX_CONTINUATION_PASSES:
                # Not settled after the passes: the CPU solves from the first such case on.
                break
            g1 = out["g"]
            start = torch.cat([g1[:1], g1[:-1]], dim=0)
            if p == 0:
                strips = md["strip"].cpu().numpy()
                cl_per_g = 2.0 / (V[:, None] * engine._f64(lattice.chord[strips])[None, :])
            out2 = engine.llt_nonlinear(prep, tol, max_it, start, active0=cand, handover=handover)
            ok2 = out2["converged"] & ~out2["unresolved"]
            same_root = ((out2["g"] - g1).abs() * cl_per_g).max(dim=1).values <= CONTINUATION_SAME_ROOT
            # Changed: converged in one pass only, or converged in both to different roots.
            changed = cand & ((ok != ok2) | (ok & ok2 & ~same_root))
            take = cand
            for key in ("g", "alpha", "W", "converged", "iterations", "unresolved"):
                m = take.view(-1, *([1] * (out[key].dim() - 1)))
                out[key] = torch.where(m, out2[key], out[key])
            out["history"] = [h2 if bool(t) else h1 for h1, h2, t in zip(out["history"], out2["history"], take.tolist())]
            ok = out["converged"] & ~out["unresolved"]
            if not bool(changed.any()):
                break
            if p + 1 == MAX_CONTINUATION_PASSES:
                first = int(torch.nonzero(changed).flatten()[0])
                unsettled[first:] = True
    g, alpha_u, W = out["g"], out["alpha"], out["W"]
    vinf = prep["Vinf"]
    vp_sub = W - (vinf if vinf.dim() == 3 else vinf[:, None, :])
    G, vp = engine.to_panels(lattice, md, g, vp_sub)
    alpha_full = torch.empty((K, lattice.n_strips), dtype=torch.float64, device=g.device)
    alpha_full[:, md["strip"]] = alpha_u
    if umap.symmetric:
        left = np.flatnonzero(~lattice.strip_is_right)
        alpha_full[:, torch.as_tensor(np.array(left), device=g.device)] = alpha_full[
            :, torch.as_tensor(np.array(lattice.strip_mirror[left]), device=g.device)]
    conv_h = out["converged"].cpu().numpy() & ~out["unresolved"].cpu().numpy()
    iters = out["iterations"].cpu().numpy()
    infos = [SolveInfo(converged=bool(conv_h[k]), iterations=int(iters[k]), residual_history=out["history"][k],
                       symmetric=umap.symmetric) for k in range(K)]
    # The cases that are not settled: the CPU solver, in order, with its restarts.
    # With continuation, every case after the first such case starts from a
    # solution that the CPU can change, so the CPU solves all of them.
    redo = np.flatnonzero(~conv_h | unsettled.cpu().numpy())
    if continuation and redo.size:
        redo = np.arange(int(redo[0]), K)
    if redo.size:
        from ventorum.utils.parallel import solve_threads

        # The thread policy of the CPU solves (BLAS on one thread, kernel threads).
        with solve_threads(lattice.n_panels):
            for k in redo:
                g_prev = None
                if gamma0 is not None and K == 1:
                    g_prev = np.asarray(gamma0, dtype=float)
                elif continuation and k > 0 and infos[k - 1].converged:
                    g_prev = G[k - 1].cpu().numpy()
                gk, ak, info = solver.solve_circulation(lattice, conditions[k], settings, grounds[k], wds[k],
                                                        gamma0=g_prev, ref_point=ref_point)
                G[k] = engine._f64(gk)
                alpha_full[k] = engine._f64(ak)
                vp[k] = engine._f64(info.v_control)
                info.v_control = None
                infos[k] = info
    return G, vp, alpha_full, infos


def _results(solver, lattice, conditions, arrays: dict, setups: list, umap, b_ref: float, c_ref: float,
             main_surface: int, t0: float, share: float, precision: str, infos: list | None = None,
             ref_point=None, S_ref: float | None = None, dispatch_reason: str = "device setting is gpu") -> list:
    """Return the result objects of a GPU batch (the loads objects and the CPU result builder).

    The far-field lateral diagnostics (``CY_farfield``, ``Cn_trefftz``) are
    computed on the host from the strip circulation and the Trefftz-plane
    normal wash of the GPU (:func:`ventorum.aero.loads.trefftz_lateral_batch`,
    the CPU equations). *S_ref* [m^2] is the reference area; None gives no
    far-field values. *dispatch_reason* is the reason of the GPU decision;
    it goes into the ``reason`` of the dispatch record
    (``details["dispatch"]``), which is made once for the batch.
    """
    from ventorum.aero.loads import loads_result, trefftz_lateral_batch
    from ventorum.aero.vortex import KERNELS
    from ventorum.hardware.profile import active_profile
    from ventorum.solvers.core import SolveInfo

    prof = active_profile()
    dispatch = {
        "device": "gpu",
        "precision": precision,
        "backends": {kernel: "warp" for kernel in KERNELS},
        "threads": 1,
        "workers": 1,
        "plan_source": "default",
        "profile": prof.get("fingerprint") if isinstance(prof, dict) else None,
        "reason": dispatch_reason,
    }
    totals = arrays["totals"].tolist()
    span = arrays["span"]
    gammas = arrays["gamma"]
    K = len(conditions)
    if S_ref is not None:
        wds = np.array([st[3] for st in setups], dtype=float).reshape(K, 3)
        cy_ff, cn_tr = trefftz_lateral_batch(
            lattice, np.asarray(span[:, 0], dtype=float), np.asarray(arrays["w_n"], dtype=float), wds, conditions,
            [st[0] for st in setups], S_ref, b_ref, ref_point)
        cy_ff, cn_tr = cy_ff.tolist(), cn_tr.tolist()
    else:
        cy_ff = cn_tr = [None] * K
    # Statistics of the trust score from the GPU (None for a case with a
    # non-finite value: the trust score then names the array).
    stats = [{"max_local_cl": r[1], "max_alpha_eff_deg": r[2], "n_extrema": int(r[3])} if r[0] != 0.0 else None
             for r in arrays["stats"].tolist()]
    out = []
    for k, (cond, st) in enumerate(zip(conditions, setups)):
        t = totals[k]
        hp = t[10] != 0.0
        ld = loads_result(
            lattice, CL=t[0], CDi=t[1], CDp=t[2] if hp else None, CD_total=t[3] if hp else None, e=t[4],
            AR=t[11], Cl=t[5], Cm=t[6], Cn=t[7], CY=t[8], CDi_nearfield=t[9],
            strip_gamma=span[k, 0], Cl_strip=span[k, 1], Cd_i=span[k, 2], Cd_profile=span[k, 3] if hp else None,
            alpha_eff=span[k, 4], alpha_i=span[k, 5], local_lift_factor=float(cond.rho) * float(cond.V_inf),
            Cm_section=span[k, 7], strip_force=arrays["strip_force"][k], trefftz_normalwash=arrays["w_n"][k],
            force_total=arrays["F_total"][k], moment_total=arrays["M"][k],
            CY_farfield=cy_ff[k], Cn_trefftz=cn_tr[k],
        )
        info = infos[k] if infos is not None else SolveInfo(symmetric=umap.symmetric)
        res = solver._case_result(lattice, cond, ld, info, gammas[k], st[0], st[1], st[2], st[3], b_ref,
                                  c_ref, main_surface, t0, execution_time=share, spanwise_stats=stats[k],
                                  ref_point=ref_point, dispatch=dispatch)
        res.details["device"] = "gpu"
        res.details["precision"] = precision
        out.append(res)
    return out
