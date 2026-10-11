# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Richardson check of the adjoint gradients against finite differences.

The reference is Richardson-extrapolated central differences of the
production solve with steps h and h/2, with the error estimate
|FD(h) - FD(h/2)| / 3. A component is compared only where the estimate is
below 1/10 of the tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ventorum.core.datatypes import FlightCondition  # noqa: F401 - re-exported name check


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


def _richardson(
    get: Any,
    x0: float,
    h: float,
    rtol: float,
) -> tuple[float, float, bool]:
    """Return (FD(h/2) extrapolated value, error estimate, usable).

    Central differences with steps h and h/2; the estimate is
    |FD(h) - FD(h/2)| / 3. Usable when the estimate is below rtol/10 of
    the reference size.
    """
    f_p_h = get(x0 + h)
    f_m_h = get(x0 - h)
    f_p_h2 = get(x0 + h / 2.0)
    f_m_h2 = get(x0 - h / 2.0)
    fd_h = (f_p_h - f_m_h) / (2.0 * h)
    fd_h2 = (f_p_h2 - f_m_h2) / h
    est = abs(fd_h - fd_h2) / 3.0
    ref = (4.0 * fd_h2 - fd_h) / 3.0
    scale = max(abs(ref), 1e-300)
    usable = bool(np.isfinite(ref) and np.isfinite(est) and est <= 0.1 * rtol * scale)
    return float(ref), float(est), usable


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
        ``"pass"`` (bool), ``"compared"``, ``"skipped"`` and per-component
        details with the reference value, the error estimate, the relative
        error and the verdict.
    """
    from ventorum.core.datatypes import Aircraft, LiftingSurface, SolverSettings
    from ventorum.gradients.adjoint import _check_names, gradients

    outs, ins = _check_names(outputs, wrt)
    st = settings.clone() if settings is not None else SolverSettings()
    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    axes = "body"
    grad_res = gradients(ac, condition, st, outputs=outs, wrt=ins, axes=axes)
    jac = grad_res.jacobian
    rng = np.random.default_rng(seed)
    details: dict[str, Any] = {}
    n_compared = 0
    n_skipped = 0
    ok_all = True

    def _record(key: str, grad: float, ref: float, est: float, usable: bool) -> None:
        nonlocal n_compared, n_skipped, ok_all
        scale = max(abs(ref), 1e-300)
        if not usable:
            n_skipped += 1
            details[key] = {"grad": float(grad), "ref": float(ref), "est": float(est),
                            "rel_err": None, "pass": None, "compared": False}
            return
        rel = abs(grad - ref) / scale
        # Relative gate per D-13, with an absolute floor at 1e-3 of the
        # tolerance: an exactly-zero derivative (for example dCL/drho, or a
        # tiny derivative like dCD/dbeta) cannot be resolved relatively; the
        # finite differences only amplify production round-off there, and the
        # float32 partials carry absolute noise around 1e-7.
        passed = bool(rel <= rtol or abs(grad - ref) <= 1e-3 * rtol)
        n_compared += 1
        if not passed:
            ok_all = False
        details[key] = {"grad": float(grad), "ref": float(ref), "est": float(est),
                        "rel_err": float(rel), "pass": passed, "compared": True}

    base_h = {
        "alpha": 1e-4, "beta": 1e-4, "phi": 1e-4,
        "V_inf": 1e-4 * max(float(condition.V_inf), 1e-300),
        "rho": 1e-4 * max(float(condition.rho), 1e-300),
        "h": 1e-4 * max(float(condition.h) if condition.h is not None else 1.0, 1e-300),
    }
    for key in ("alpha", "beta", "V_inf", "rho", "h", "phi"):
        if key not in ins:
            continue
        if key == "h" and condition.h is None:
            continue
        h = base_h[key]
        for o in outs:
            def _get(v: float, _k: str = key, _o: str = o) -> float:
                import dataclasses

                c2 = dataclasses.replace(condition, **{_k: v})
                return _production_scalars(ac, c2, st, axes)[_o]

            x0 = float(getattr(condition, key))
            ref, est, usable = _richardson(_get, x0, h, rtol)
            grad = float(jac[o][key]) if isinstance(jac[o], dict) else 0.0
            _record(f"{o}:{key}", grad, ref, est, usable)
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
                    h = 1e-4 * max(abs(x0), 1e-300)
                    w = weights[prop]
                    for o in outs:
                        gvec = np.asarray(jac[o]["sections"][prop], dtype=float).reshape(-1)
                        pred = float(gvec @ w)

                        def _get(v: float, _n: str = surf.name, _j: int = j,
                                 _p: str = prop, _o: str = o) -> float:
                            ac3 = ac.clone()
                            _set_section_prop(ac3, _n, _j, _p, v)
                            return _production_scalars(ac3, condition, st, axes)[_o]

                        ref, est, usable = _richardson(_get, x0, h, rtol)
                        _record(f"{o}:sections.{surf.name}[{j}].{prop}", pred, ref, est, usable)
    if "reference" in ins:
        ac2 = ac.clone()
        ac2.compute_reference_values()
        ref_base = {"S_ref": float(ac2.S_ref), "b_ref": float(ac2.b_ref),
                    "c_ref": float(ac2.c_ref),
                    "ref_point": np.asarray(ac2.moment_reference(), dtype=float)}
        for key in ("S_ref", "b_ref", "c_ref"):
            h = 1e-4 * max(abs(ref_base[key]), 1e-300)
            for o in outs:
                def _get(v: float, _k: str = key, _o: str = o) -> float:
                    ac3 = ac.clone()
                    setattr(ac3, _k, v)
                    ac3._auto_ref.discard(_k)
                    return _production_scalars(ac3, condition, st, axes)[_o]

                ref, est, usable = _richardson(_get, ref_base[key], h, rtol)
                grad = float(jac[o]["reference"][key])
                _record(f"{o}:reference.{key}", grad, ref, est, usable)
        rp0 = ref_base["ref_point"]
        h = 1e-4 * max(float(ac2.c_ref), 1e-300)
        for o in outs:
            gvec = np.asarray(jac[o]["reference"]["ref_point"], dtype=float).reshape(3)
            for k in range(3):
                def _get(v: float, _kk: int = k, _o: str = o) -> float:
                    ac3 = ac.clone()
                    rp = rp0.copy()
                    rp[_kk] = v
                    ac3.ref_point = rp
                    return _production_scalars(ac3, condition, st, axes)[_o]

                ref, est, usable = _richardson(_get, float(rp0[k]), h, rtol)
                _record(f"{o}:reference.ref_point[{k}]", float(gvec[k]), ref, est, usable)
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
        # Small step: the node response is smooth, and 1e-5 of the span keeps
        # both the truncation and the round-off of the differences small.
        h = 1e-5 * max(float(s.semi_span) for s in ac2.surfaces)
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
            for o in outs:
                g_full = _gather_node_grad(jac[o], flat)
                grad = float(g_full @ u)

                def _get(v: float, _u: Any = u, _o: str = o) -> float:
                    ac3 = ac.clone()
                    _apply_node_step(ac3, base_nodes, disp0, flat, _u, v)
                    return _production_scalars(ac3, condition, st, axes)[_o]

                ref, est, usable = _richardson(_get, 0.0, h, rtol)
                _record(f"{o}:nodes.dir{dj}", grad, ref, est, usable)
    return {"pass": bool(ok_all), "compared": n_compared, "skipped": n_skipped,
            "rtol": float(rtol), "details": details}


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
    import numpy as _np

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
            le = _np.zeros((n_e, 3))
            te = _np.zeros((n_e, 3))
        else:
            le = _np.asarray(d0.le, dtype=float).copy()
            te = _np.asarray(d0.te, dtype=float).copy()
        for tag, items in tags.items():
            arr = le if tag == "le" else te
            for i, c, val in items:
                arr[i, c] += val
        from ventorum.core.datatypes import NodeDisplacements

        eta = _np.asarray(bn["eta"], dtype=float).copy()
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
        dv = 1e-4 * max(abs(x0), 1e-300)
        _set_section_prop(ac3, surf_name, j, prop, x0 + dv)
        lat = solver.build(ac3, st, condition)
        cur = np.asarray(getattr(lat, prop), dtype=float)
        weights[prop] = (cur - base_props[prop]) / dv
    return weights
