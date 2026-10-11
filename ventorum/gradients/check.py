# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Richardson check of the adjoint gradients against finite differences.

The reference is Richardson-extrapolated central differences of the
production solve with steps h and h/2, with the error estimate
|FD(h) - FD(h/2)| / 3. A component is compared only where the estimate is
below 1/10 of the tolerance.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Absolute fallback steps of the section properties when the base value is
# zero (or very small): a0 [1/rad], alpha_L0 [rad], Cd0 [-], Cm0 [-]. They
# are 1e-4 of a typical magnitude of each property.
_SECTION_STEP_FLOOR = {"a0": 1e-4 * 2.0 * np.pi, "alpha_L0": 1e-4, "Cd0": 1e-6, "Cm0": 1e-5}

# Factors of the step ladders. The node step is 1e-3 of the smallest core
# radius of the lattice (one step); the other steps are relative to the
# base step of the input. Each component uses the step with the smallest
# error bound (the larger of the Richardson estimate and the round-off).
_NODE_STEP_FACTORS = (1.0,)
_STEP_FACTORS = (1.0, 0.3, 0.1)

# Zero-derivative check: a reference that is zero within this factor times
# its error bound cannot be compared relatively (see check_gradients).
_ZERO_FACTOR = 10.0


def _production_scalars(geometry: Any, condition: Any, settings: Any, axes: str) -> dict[str, float]:
    """Return the scalar outputs of one production solve in *axes*."""
    import ventorum as vt

    res = vt.analyze(geometry.clone(), condition, settings.clone())
    out = {
        "CL": float(res.totals.CL),
        "CY": float(res.totals.CY),
        "CDi": float(res.totals.CDi),
        "CDp": float(res.totals.CDp if res.totals.CDp is not None else 0.0),
        "CD": float(res.totals.CD_total if res.totals.CD_total is not None else res.totals.CDi),
    }
    m = res.moments(axes)
    out["Cl"] = float(m["Cl"])
    out["Cm"] = float(m["Cm"])
    out["Cn"] = float(m["Cn"])
    return out


def _richardson(get: Any, x0: float, h: float, outs: tuple[str, ...],
                eps: float) -> dict[str, tuple[float, float, float]]:
    """Return per output (reference, error estimate, round-off level).

    Central differences with steps h and h/2 of the vector function *get*
    (a dict of outputs). The reference is (4 FD(h/2) - FD(h)) / 3, the
    estimate |FD(h) - FD(h/2)| / 3, and the round-off level of the
    differences eps * max|f| / h (*eps* is the machine epsilon of the
    production solve).
    """
    f_p_h = get(x0 + h)
    f_m_h = get(x0 - h)
    f_p_h2 = get(x0 + h / 2.0)
    f_m_h2 = get(x0 - h / 2.0)
    out = {}
    for o in outs:
        fd_h = (f_p_h[o] - f_m_h[o]) / (2.0 * h)
        fd_h2 = (f_p_h2[o] - f_m_h2[o]) / h
        est = abs(fd_h - fd_h2) / 3.0
        ref = (4.0 * fd_h2 - fd_h) / 3.0
        size = max(abs(f_p_h[o]), abs(f_m_h[o]), abs(f_p_h2[o]), abs(f_m_h2[o]))
        out[o] = (float(ref), float(est), float(eps * size / (0.5 * h)))
    return out


def _ladder(get: Any, x0: float, h: float, factors: tuple[float, ...], outs: tuple[str, ...],
            eps: float) -> dict[str, list[tuple[float, float, float, float]]]:
    """Return per output the Richardson candidates (step, ref, est, round-off) of the steps h * factors."""
    out: dict[str, list[tuple[float, float, float, float]]] = {o: [] for o in outs}
    fds = [(h * f, _richardson(get, x0, h * f, outs, eps)) for f in factors]
    for o in outs:
        for k, (step, fd) in enumerate(fds):
            ref, est, rnd = fd[o]
            if len(fds) > 1:
                # Convergence: the Richardson value of a neighbour step must
                # agree; the spread to the nearer neighbour joins the bound.
                spread = min(abs(ref - fds[j][1][o][0]) for j in (k - 1, k + 1) if 0 <= j < len(fds))
                rnd = max(rnd, spread)
            out[o].append((step, ref, est, rnd))
    return out


def check_gradients(
    geometry: Any,
    condition: Any,
    settings: Any = None,
    *,
    outputs: tuple[str, ...] = ("CL", "CD", "Cm"),
    wrt: tuple[str, ...] = ("alpha", "nodes"),
    n_directions: int = 3,
    rtol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Check adjoint gradients against Richardson finite differences.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. It is not changed.
    condition : FlightCondition
        Flight condition.
    settings : SolverSettings or None
        Solver settings.
    outputs : tuple of str
        Scalar outputs to check.
    wrt : tuple of str
        Inputs to check. ``"nodes"`` is checked along *n_directions*
        random unit directions over all node coordinates (seeded).
    n_directions : int
        Number of random node directions.
    rtol : float
        Relative tolerance: 1e-6 in float64, 1e-3 in float32.
    seed : int
        Seed of the random node directions.

    Returns
    -------
    dict
        ``"pass"`` (bool), ``"compared"``, ``"skipped"``, ``"groups"``
        (per input group the compared and skipped counts),
        ``"reasons"`` (why the check failed, empty when it passed) and
        ``"details"`` per component with the reference value, the error
        estimate, the step [unit of the input], the relative error, the
        mode and the verdict.

    Notes
    -----
    The reference of each component is the Richardson value of decision 10
    of the gradient method: central differences with steps h and h/2, with
    the error estimate |FD(h) - FD(h/2)| / 3. The error bound of a
    reference is the larger of the estimate and the round-off of the
    differences (machine epsilon of the production solve times the output
    magnitude over the step); with a step ladder, the spread to the
    Richardson value of the neighbour step joins the bound.

    * A component with an error bound below rtol/10 of its reference is
      compared relatively: it passes when |grad - ref| <= rtol |ref|.
    * The scale S of an input component is the largest resolved reference
      of the outputs for that input (else the largest output magnitude per
      unit input). A resolved component below rtol S has no relative scale
      of its own in the precision of the gradient (it is the difference of
      larger terms); it passes when |grad - ref| <= rtol S (mode
      ``"column"``).
    * A component whose reference is zero within 10 times its error bound
      is a zero derivative (for example dCL/drho). It passes when |grad|
      is below the larger of that bound and rtol S (mode ``"zero"``).
    * Every other component is skipped.

    The node step is 1e-3 of the smallest core radius of the lattice. A
    larger step crosses the core-radius length scale, where the
    differences do not converge. The other inputs use the steps 1, 0.3 and
    0.1 times the base step (1e-4 of the base value, angles 1e-4 rad, with
    an absolute floor per section property for a zero base value); each
    component uses the step with the smallest error bound.

    The check fails when a component fails, and also when an input group
    (each scalar input, ``"sections"``, ``"reference"``, ``"nodes"``) has
    no compared component: such a group is not verified. With a float32
    production solve (the GPU in float32) the differences do not resolve
    the node directions, and the check reports that group.
    """
    import ventorum as vt
    from ventorum.core.datatypes import Aircraft, LiftingSurface, SolverSettings
    from ventorum.gradients.adjoint import _check_names, gradients

    outs, ins = _check_names(outputs, wrt)
    st = settings.clone() if settings is not None else SolverSettings()
    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    axes = "body"
    grad_res = gradients(ac, condition, st, outputs=outs, wrt=ins, axes=axes)
    jac = grad_res.jacobian
    base_res = vt.analyze(ac.clone(), condition, st.clone())
    prec = str(base_res.details.get("precision", "float64"))
    eps = float(np.finfo(np.float32 if prec == "float32" else np.float64).eps)
    rng = np.random.default_rng(seed)
    details: dict[str, Any] = {}
    groups: dict[str, dict[str, int]] = {}
    ok_all = True

    f0 = _production_scalars(ac, condition, st, axes)

    def _record(group: str, suffix: str, grads: dict[str, float],
                fd: dict[str, list[tuple[float, float, float, float]]], x_scale: float) -> None:
        """Record the components of all outputs for one input component.

        *fd* holds per output the candidates (step, ref, est, round-off);
        *x_scale* is the magnitude of the input [unit of the input].
        """
        nonlocal ok_all
        cnt = groups.setdefault(group, {"compared": 0, "skipped": 0})
        pick = {}
        for o in outs:
            h, ref, est, rnd = min(fd[o], key=lambda c: max(c[2], c[3]) / max(abs(c[1]), 1e-300))
            err = max(est, rnd)
            resolved = bool(np.isfinite(ref) and np.isfinite(err) and ref != 0.0
                            and err <= 0.1 * rtol * abs(ref))
            pick[o] = (h, ref, est, err, resolved)
        # Scale of the column: the largest resolved reference of the outputs,
        # else the output magnitude per unit input.
        scale = max((abs(v[1]) for v in pick.values() if v[4]), default=0.0)
        if scale == 0.0:
            scale = max(abs(f0[o]) for o in outs) / max(abs(x_scale), 1e-300)
        for o in outs:
            h, ref, est, err, resolved = pick[o]
            grad = float(grads[o])
            info = {"grad": grad, "ref": float(ref), "est": float(est), "step": float(h)}
            if resolved:
                near_zero = abs(ref) < rtol * scale
                denom = scale if near_zero else abs(ref)
                rel = abs(grad - ref) / denom
                info.update(rel_err=float(rel), mode="column" if near_zero else "relative",
                            compared=True, passed=bool(rel <= rtol))
            elif np.isfinite(grad) and all(abs(c[1]) <= _ZERO_FACTOR * max(c[2], c[3]) for c in fd[o]):
                bound = max(min(_ZERO_FACTOR * max(c[2], c[3]) for c in fd[o]), rtol * scale)
                info.update(rel_err=None, mode="zero", bound=float(bound),
                            compared=True, passed=bool(abs(grad) <= bound))
            else:
                info.update(rel_err=None, mode="skipped", compared=False, passed=None)
            details[f"{o}:{suffix}"] = info
            if info["compared"]:
                cnt["compared"] += 1
                if not info["passed"]:
                    ok_all = False
            else:
                cnt["skipped"] += 1

    base_h = {
        "alpha": 1e-4, "beta": 1e-4, "phi": 1e-4,
        "V_inf": 1e-4 * abs(float(condition.V_inf)),
        "rho": 1e-4 * abs(float(condition.rho)),
        "h": 1e-4 * abs(float(condition.h)) if condition.h is not None else 0.0,
    }
    for key in ("alpha", "beta", "V_inf", "rho", "h", "phi"):
        if key not in ins:
            continue
        if key == "h" and condition.h is None:
            continue
        h = base_h[key]

        def _get(v: float, _k: str = key) -> dict[str, float]:
            c2 = dataclasses.replace(condition, **{_k: v})
            return _production_scalars(ac, c2, st, axes)

        fd = _ladder(_get, float(getattr(condition, key)), h, _STEP_FACTORS, outs, eps)
        _record(key, key, {o: float(jac[o][key]) for o in outs}, fd, h / 1e-4)
    if "sections" in ins:
        from ventorum.solvers.factory import make_solver, resolve_solver_type

        ac2 = ac.clone()
        ac2.compute_reference_values()
        solver = make_solver(resolve_solver_type(st.solver_type))
        lat0 = solver.build(ac2, st, condition)
        base_props = {
            "a0": np.asarray(lat0.a0, dtype=float),
            "alpha_L0": np.asarray(lat0.alpha_L0, dtype=float),
            "Cd0": np.asarray(lat0.Cd0, dtype=float),
            "Cm0": np.asarray(lat0.Cm0, dtype=float),
        }
        for surf in ac2.surfaces:
            for j, sec in enumerate(surf.sections):
                weights = _section_weights(ac, solver, st, condition, surf.name, j, base_props)
                for prop in ("a0", "alpha_L0", "Cd0", "Cm0"):
                    x0 = float(getattr(sec.airfoil, prop))
                    h = _section_step(prop, x0)
                    w = weights[prop]

                    def _get(v: float, _n: str = surf.name, _j: int = j,
                             _p: str = prop) -> dict[str, float]:
                        ac3 = ac.clone()
                        _set_section_prop(ac3, _n, _j, _p, v)
                        return _production_scalars(ac3, condition, st, axes)

                    fd = _ladder(_get, x0, h, _STEP_FACTORS, outs, eps)
                    grads = {o: float(np.asarray(jac[o]["sections"][prop], dtype=float).reshape(-1) @ w)
                             for o in outs}
                    _record("sections", f"sections.{surf.name}[{j}].{prop}", grads, fd, h / 1e-4)
    if "reference" in ins:
        ac2 = ac.clone()
        ac2.compute_reference_values()
        ref_base = {"S_ref": float(ac2.S_ref), "b_ref": float(ac2.b_ref),
                    "c_ref": float(ac2.c_ref),
                    "ref_point": np.asarray(ac2.moment_reference(), dtype=float)}
        for key in ("S_ref", "b_ref", "c_ref"):
            h = 1e-4 * abs(ref_base[key])

            def _get(v: float, _k: str = key) -> dict[str, float]:
                ac3 = ac.clone()
                setattr(ac3, _k, v)
                ac3._auto_ref.discard(_k)
                return _production_scalars(ac3, condition, st, axes)

            fd = _ladder(_get, ref_base[key], h, _STEP_FACTORS, outs, eps)
            _record("reference", f"reference.{key}",
                    {o: float(jac[o]["reference"][key]) for o in outs}, fd, ref_base[key])
        rp0 = ref_base["ref_point"]
        h = 1e-4 * abs(float(ac2.c_ref))
        for k in range(3):
            def _get(v: float, _kk: int = k) -> dict[str, float]:
                ac3 = ac.clone()
                rp = rp0.copy()
                rp[_kk] = v
                ac3.ref_point = rp
                return _production_scalars(ac3, condition, st, axes)

            fd = _ladder(_get, float(rp0[k]), h, _STEP_FACTORS, outs, eps)
            grads = {o: float(np.asarray(jac[o]["reference"]["ref_point"], dtype=float).reshape(3)[k])
                     for o in outs}
            _record("reference", f"reference.ref_point[{k}]", grads, fd, float(ac2.c_ref))
    if "nodes" in ins:
        from ventorum.geometry.deformation import undeformed_nodes
        from ventorum.solvers.factory import resolve_solver_type

        ac2 = ac.clone()
        canonical = resolve_solver_type(st.solver_type)
        solver_name = "vlm" if canonical == "vlm" else "linear"
        base_nodes = undeformed_nodes(ac2, st, solver=solver_name)
        disp0 = {s.name: (s.node_displacements.clone() if s.node_displacements is not None else None)
                 for s in ac2.surfaces}
        flat: list[tuple[str, str, int, int]] = []
        for s in ac2.surfaces:
            bn = base_nodes[s.name]
            n_e = int(bn["eta"].shape[0])
            for tag in ("le", "te"):
                for i in range(n_e):
                    for c in range(3):
                        flat.append((s.name, tag, i, c))
        dim = len(flat)
        rc_min = float(np.min(np.asarray(base_res.details["lattice"].rc, dtype=float)))
        h0 = 1e-3 * rc_min
        span = max(float(s.semi_span) for s in ac2.surfaces)
        # A symmetric surface is mirrored about y = 0 and must start there:
        # freeze the root y-displacements of symmetric surfaces, else the
        # perturbed geometry is refused.
        sym_names = {s.name for s in ac2.surfaces if s.is_symmetric}
        frozen = [j for j, (name, tag, i, c) in enumerate(flat)
                  if name in sym_names and i == 0 and c == 1]
        for dj in range(int(n_directions)):
            u = rng.standard_normal(dim)
            u[frozen] = 0.0
            u = u / max(float(np.linalg.norm(u)), 1e-300)

            def _get(v: float, _u: Any = u) -> dict[str, float]:
                ac3 = ac.clone()
                _apply_node_step(ac3, base_nodes, disp0, flat, _u, v)
                return _production_scalars(ac3, condition, st, axes)

            fd = _ladder(_get, 0.0, h0, _NODE_STEP_FACTORS, outs, eps)
            grads = {o: float(_gather_node_grad(jac[o], flat) @ u) for o in outs}
            _record("nodes", f"nodes.dir{dj}", grads, fd, span)
    reasons = [f"input group {g!r} has no compared component" for g, c in groups.items()
               if c["compared"] == 0]
    failed = [k for k, v in details.items() if v["compared"] and not v["passed"]]
    if failed:
        reasons.insert(0, f"{len(failed)} component(s) above rtol {rtol:g}: {', '.join(failed)}")
    n_compared = sum(c["compared"] for c in groups.values())
    n_skipped = sum(c["skipped"] for c in groups.values())
    return {"pass": bool(ok_all and not reasons), "compared": n_compared, "skipped": n_skipped,
            "rtol": float(rtol), "groups": groups, "reasons": reasons, "details": details}


def _section_step(prop: str, x0: float) -> float:
    """Return the finite-difference step of section property *prop* at *x0*.

    The step is 1e-4 of the base value, with an absolute floor for a zero
    or small base value (``_SECTION_STEP_FLOOR``, unit of the property).
    """
    return max(1e-4 * abs(float(x0)), _SECTION_STEP_FLOOR[prop])


def _gather_node_grad(jac_o: dict, flat: list) -> np.ndarray:
    """Gather the node gradient of one output along the flat index."""
    out = np.zeros(len(flat))
    nodes = jac_o.get("nodes", {})
    for j, (name, tag, i, c) in enumerate(flat):
        arr = nodes.get(name, {}).get(tag)
        if arr is not None:
            out[j] = float(np.asarray(arr)[i, c])
    return out


def _apply_node_step(ac: Any, base_nodes: Any, disp0: Any, flat: list,
                     u: np.ndarray, v: float) -> None:
    """Add v * u to the node displacements of *ac* (absolute node step)."""
    from ventorum.core.datatypes import NodeDisplacements

    step = (np.asarray(u, dtype=float) * float(v))
    per: dict = {}
    for j, (name, tag, i, c) in enumerate(flat):
        per.setdefault(name, {}).setdefault(tag, []).append((i, c, step[j]))
    for name, tags in per.items():
        surf = next(s for s in ac.surfaces if s.name == name)
        bn = base_nodes[name]
        n_e = int(bn["eta"].shape[0])
        d0 = disp0.get(name)
        if d0 is None:
            le = np.zeros((n_e, 3))
            te = np.zeros((n_e, 3))
        else:
            le = np.asarray(d0.le, dtype=float).copy()
            te = np.asarray(d0.te, dtype=float).copy()
        for tag, items in tags.items():
            arr = le if tag == "le" else te
            for i, c, val in items:
                arr[i, c] += val
        eta = np.asarray(bn["eta"], dtype=float).copy()
        surf.node_displacements = NodeDisplacements(le=le, te=te, eta=eta)


def _set_section_prop(ac: Any, surf_name: str, j: int, prop: str, v: float) -> None:
    """Set property *prop* of section *j* of surface *surf_name* to *v*."""
    from ventorum.core.datatypes import LinearAirfoil, TabulatedAirfoil

    surf = next(s for s in ac.surfaces if s.name == surf_name)
    af = surf.sections[j].airfoil
    if isinstance(af, TabulatedAirfoil):
        raise ValueError("Section checks need linear airfoils.")
    surf.sections[j].airfoil = LinearAirfoil(
        name=af.name,
        a0=float(v) if prop == "a0" else float(af.a0),
        alpha_L0=float(v) if prop == "alpha_L0" else float(af.alpha_L0),
        Cd0=float(v) if prop == "Cd0" else float(af.Cd0),
        Cm0=float(v) if prop == "Cm0" else float(af.Cm0),
    )


def _section_weights(ac: Any, solver: Any, st: Any, condition: Any,
                     surf_name: str, j: int, base_props: dict) -> dict[str, np.ndarray]:
    """Return the strip-prop weights per unit section-prop change.

    Perturbs each property of the section by a fixed step, rebuilds the
    lattice, and returns ``(strip_props - base) / step`` per property. The
    spanwise interpolation is linear, so the weights are exact.
    """
    weights = {}
    for prop in ("a0", "alpha_L0", "Cd0", "Cm0"):
        ac3 = ac.clone()
        surf = next(s for s in ac3.surfaces if s.name == surf_name)
        x0 = float(getattr(surf.sections[j].airfoil, prop))
        dv = _section_step(prop, x0)
        _set_section_prop(ac3, surf_name, j, prop, x0 + dv)
        lat = solver.build(ac3, st, condition)
        cur = np.asarray(getattr(lat, prop), dtype=float)
        weights[prop] = (cur - base_props[prop]) / dv
    return weights
