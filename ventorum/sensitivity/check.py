# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Richardson check of the adjoint gradients against finite differences.

The reference is Richardson-extrapolated central differences of the
production solve with steps h and h/2, with the error estimate
abs(FD(h) - FD(h/2)) / 3. A component is compared only where the estimate is
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

# Noise probe of the node directions: the number of points, their spacing
# as a fraction of the node step, and the factor of the measured noise
# level in the round-off bound (see _noise_level and _richardson).
_NOISE_POINTS = 10
_NOISE_SPACING = 1e-3
_NOISE_FACTOR = 6.0


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
                eps: float, noise: dict[str, float] | None = None, side: int = 0
                ) -> dict[str, tuple[float, float, float]]:
    """Return per output (reference, error estimate, round-off level).

    Central differences with steps h and h/2 of the vector function *get*
    (a dict of outputs). The reference is (4 FD(h/2) - FD(h)) / 3, the
    estimate abs(FD(h) - FD(h/2)) / 3, and the round-off level of the
    differences 2 eps max(abs(f)) / h (*eps* is the machine epsilon of the
    production solve). When *noise* gives the measured noise level sigma
    of an output (see :func:`_noise_level`), the round-off level is the
    larger of that value and 6 sigma / h: the noise of the Richardson
    value is 1.9 sigma / h, and 6 sigma / h is about three times that.

    With *side* +1 or -1 the differences are one-sided, to the side of
    *side* only (for an input at a limit of its range). The second-order
    one-sided difference
    ``D(h) = (-3 f(x0) + 4 f(x0 + s h) - f(x0 + 2 s h)) / (2 s h)``
    replaces the central difference, with the same Richardson combination
    and estimate (both differences have an error of order h^2). The
    round-off level is then 12 eps max(abs(f)) / h (the sum of the
    magnitudes of the weights of the Richardson value).
    """
    if side:
        sgn = 1.0 if side > 0 else -1.0
        f0 = get(x0)
        f_1 = get(x0 + sgn * h / 2.0)
        f_2 = get(x0 + sgn * h)
        f_4 = get(x0 + sgn * 2.0 * h)
        out1 = {}
        for o in outs:
            d_h = (-3.0 * f0[o] + 4.0 * f_2[o] - f_4[o]) / (2.0 * sgn * h)
            d_h2 = (-3.0 * f0[o] + 4.0 * f_1[o] - f_2[o]) / (sgn * h)
            est = abs(d_h - d_h2) / 3.0
            ref = (4.0 * d_h2 - d_h) / 3.0
            size = max(abs(f0[o]), abs(f_1[o]), abs(f_2[o]), abs(f_4[o]))
            rnd = 12.0 * eps * size / h
            if noise is not None:
                rnd = max(rnd, _NOISE_FACTOR * noise[o] / h)
            out1[o] = (float(ref), float(est), float(rnd))
        return out1
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
        rnd = eps * size / (0.5 * h)
        if noise is not None:
            rnd = max(rnd, _NOISE_FACTOR * noise[o] / h)
        out[o] = (float(ref), float(est), float(rnd))
    return out


def _noise_level(get: Any, x0: float, delta: float, outs: tuple[str, ...]) -> dict[str, float]:
    """Return per output the measured noise level sigma of *get* near *x0*.

    The function is evaluated at _NOISE_POINTS points with the spacing
    *delta* around *x0*. A quadratic least-squares fit removes the smooth
    part; sigma is the standard deviation of the residuals (unit of the
    output). The spacing is much smaller than the finite-difference step,
    so the residuals are the round-off of the production solve. This
    round-off can be much larger than eps max(abs(f)): a node motion
    changes the rounding of all lattice quantities, and the solve
    amplifies it.
    """
    n = _NOISE_POINTS
    s = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * float(delta)
    vals = [get(x0 + v) for v in s]
    out = {}
    for o in outs:
        y = np.array([float(v[o]) for v in vals])
        if not np.all(np.isfinite(y)):
            out[o] = float("inf")
            continue
        coef = np.polyfit(s, y, 2)
        res = y - np.polyval(coef, s)
        out[o] = float(np.sqrt(np.sum(res * res) / (n - 3)))
    return out


def _ladder(get: Any, x0: float, h: float, factors: tuple[float, ...], outs: tuple[str, ...],
            eps: float, noise: dict[str, float] | None = None, side: int = 0
            ) -> dict[str, list[tuple[float, float, float, float]]]:
    """Return per output the Richardson candidates (step, ref, est, round-off) of the steps h * factors.

    *side* selects central (0) or one-sided (+1, -1) differences (see
    :func:`_richardson`).
    """
    out: dict[str, list[tuple[float, float, float, float]]] = {o: [] for o in outs}
    fds = [(h * f, _richardson(get, x0, h * f, outs, eps, noise, side)) for f in factors]
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
    method: str = "auto",
    node_step: float | None = None,
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
        ``"controls"`` is checked per control name (all controls with one
        name move together; base step 1e-4 rad). Near the deflection limit
        of 30 deg the differences are one-sided, away from the limit
        (``"one_sided"`` in the details gives the side). ``"p"``, ``"q"``, ``"r"``
        use the base step 1e-4 of the nondimensional rate (1e-4 times
        2 V / b_ref for p and r, 2 V / c_ref for q) [rad/s].
        ``"planform"`` is checked parameter by parameter (surface
        semi-span, sweep, dihedral, incidence and position; per-section
        chord, twist, x_le, z_le, span fraction and airfoil incidence).
        Components are skipped with a reason when the mesh stations jump
        within the step, and for the copies of a mirror pair that gives
        automatic reference values (a change of one copy breaks the pair).
    n_directions : int
        Number of random node directions.
    rtol : float
        Relative tolerance: 1e-6 in float64, 1e-3 in float32.
    seed : int
        Seed of the random node directions.
    method : str
        Path of the partial derivatives of the adjoint gradients (see
        :func:`ventorum.sensitivity.gradients`).
    node_step : float or None, optional
        Finite-difference step for node directions [m]. If None (default),
        1e-3 of the smallest core radius of the lattice is used.

    Returns
    -------
    dict
        ``"pass"`` (bool), ``"compared"``, ``"skipped"``, ``"groups"``
        (per input group the compared and skipped counts),
        ``"reasons"`` (why the check failed, empty when it passed) and
        ``"details"`` per component with the reference value, the error
        estimate, the step [unit of the input], the relative error, the
        mode and the verdict, and ``"partials"`` (the path of the partial
        derivatives of the adjoint gradients).

    Notes
    -----
    The reference of each component is a Richardson value: central
    differences with steps h and h/2, with the error estimate abs(FD(h) - FD(h/2)) / 3. The error bound of a
    reference is the larger of the estimate and the round-off of the
    differences (two times the machine epsilon of the production solve times
    the output magnitude over the step); with a step ladder, the spread to the
    Richardson value of the neighbour step joins the bound. The node
    directions have one step and no spread: their round-off is also at
    least 6 sigma / h, where sigma is the noise level of the production
    solve, measured per output with 10 solves at a spacing of 1e-3 of the
    step along the same direction (the residual of a quadratic fit). A
    node motion changes the rounding of the whole lattice, and this noise
    can be much larger than the machine epsilon times the output.

    * A component with an error bound below rtol/10 of its reference is
      compared relatively: it passes when abs(grad - ref) <= rtol abs(ref).
    * The scale S of an input component is the largest resolved reference
      of the outputs for that input (else the largest output magnitude per
      unit input). A resolved component below rtol S has no relative scale
      of its own in the precision of the gradient (it is the difference of
      larger terms); it passes when abs(grad - ref) <= rtol S (mode
      ``"column"``).
    * A component whose reference is zero within 10 times its error bound
      is a zero derivative (for example dCL/drho). When 10 times its error
      bound (the noise bound) is below rtol S, it passes when abs(grad) is
      below rtol S (mode ``"zero"``). A larger noise bound would accept
      almost any gradient, so the component is skipped, with a reason.
    * Every other component is skipped.

    The node step is 1e-3 of the smallest core radius of the lattice (or
    *node_step* when specified). A larger step crosses the core-radius length
    scale, where the differences do not converge. The other inputs use the steps 1, 0.3 and
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
    from ventorum.sensitivity.adjoint import _check_names, gradients
    from ventorum.sensitivity.errors import GradientScopeError
    from ventorum.solvers.factory import resolve_solver_type

    outs, ins = _check_names(outputs, wrt)
    st0 = settings.clone() if settings is not None else SolverSettings()
    mach = getattr(condition, "mach", None)
    mach_on = mach is not None and float(mach) > 0.0
    if mach_on:
        canonical = resolve_solver_type(st0.solver_type)
        if canonical != "vlm":
            raise GradientScopeError(
                f"check_gradients is not supported with compressibility correction "
                f"(mach > 0) for solver '{canonical}'. Use solver_type 'vlm'.")
        if getattr(condition, "h", None) is not None:
            raise GradientScopeError(
                "check_gradients is not supported with compressibility correction "
                "(mach > 0) in ground effect.")
    st = settings.clone() if settings is not None else SolverSettings()
    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    axes = "body"
    grad_res = gradients(ac, condition, st, outputs=outs, wrt=ins, axes=axes, method=method)
    jac = grad_res.jacobian
    base_res = vt.analyze(ac.clone(), condition, st.clone())
    prec = str(base_res.details.get("precision", "float64"))
    eps = float(np.finfo(np.float32 if prec == "float32" else np.float64).eps)
    rng = np.random.default_rng(seed)
    details: dict[str, Any] = {}
    groups: dict[str, dict[str, int]] = {}
    ok_all = True

    f0 = _production_scalars(ac, condition, st, axes)

    from ventorum.sensitivity.topology import shared_hinge_topology
    from ventorum.solvers.factory import make_solver

    solver = make_solver(resolve_solver_type(st.solver_type))
    base_lat = base_res.details.get("lattice")
    base_topo = shared_hinge_topology(base_lat) if base_lat is not None else None

    def _evaluate(geometry_pert: Any, condition_pert: Any, input_name: str, step: float) -> dict[str, float]:
        if base_topo is not None:
            pert_lat = solver.build(geometry_pert, st, condition_pert)
            pert_topo = shared_hinge_topology(pert_lat)
            if pert_topo != base_topo:
                changed = base_topo.diff_surfaces(pert_topo)
                surfs = ", ".join(repr(s) for s in sorted(changed))
                step_val = abs(float(step))
                raise GradientScopeError(
                    f"Finite-difference step {step_val:g} for input {input_name!r} changed the "
                    f"shared-hinge topology of surface(s) {surfs}. Adjoint gradients are "
                    "defined at a fixed shared-hinge topology. Use a smaller finite-difference step."
                )
        return _production_scalars(geometry_pert, condition_pert, st, axes)

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
                noise_bound = min(_ZERO_FACTOR * max(c[2], c[3]) for c in fd[o])
                if noise_bound < rtol * scale:
                    bound = rtol * scale
                    info.update(rel_err=None, mode="zero", bound=float(bound),
                                compared=True, passed=bool(abs(grad) <= bound))
                else:
                    # A large noise bound would accept almost any gradient:
                    # the component verifies nothing.
                    info.update(rel_err=None, mode="skipped", compared=False, passed=None,
                                bound=float(noise_bound),
                                reason=(f"zero reference, but its noise bound {noise_bound:.3e} is not "
                                        f"below rtol S = {rtol * scale:.3e}"))
            else:
                info.update(rel_err=None, mode="skipped", compared=False, passed=None,
                            bound=float(err),
                            reason=(f"error bound {err:.3e} of the reference is above rtol/10 |ref| = "
                                    f"{0.1 * rtol * abs(ref):.3e}"))
            details[f"{o}:{suffix}"] = info
            if info["compared"]:
                cnt["compared"] += 1
                if not info["passed"]:
                    ok_all = False
            else:
                cnt["skipped"] += 1

    base_h = {
        "alpha": 1e-4, "beta": 1e-4, "phi": 1e-4, "mach": 1e-4,
        "V_inf": 1e-4 * abs(float(condition.V_inf)),
        "rho": 1e-4 * abs(float(condition.rho)),
        "h": 1e-4 * abs(float(condition.h)) if condition.h is not None else 0.0,
    }
    for key in ("alpha", "beta", "V_inf", "rho", "h", "phi", "mach"):
        if key not in ins:
            continue
        if key == "h" and condition.h is None:
            continue
        if key == "mach" and getattr(condition, "mach", None) is None:
            continue
        h = base_h[key]

        def _get(v: float, _k: str = key) -> dict[str, float]:
            c2 = dataclasses.replace(condition, **{_k: v})
            return _evaluate(ac, c2, _k, v - float(getattr(condition, _k)))

        fd = _ladder(_get, float(getattr(condition, key)), h, _STEP_FACTORS, outs, eps,
                     side=(1 if key == "mach" and float(getattr(condition, key)) == 0.0 else 0))
        _record(key, key, {o: float(jac[o][key]) for o in outs}, fd, h / 1e-4)
        if key == "mach" and float(getattr(condition, key)) == 0.0:
            for o in outs:
                details[f"{o}:{key}"]["one_sided"] = 1
    if any(k in ins for k in ("p", "q", "r")):
        ac2 = ac.clone()
        ac2.compute_reference_values()
        V = abs(float(condition.V_inf))
        # The base step is 1e-4 of the nondimensional rates p b/(2V),
        # q c/(2V) and r b/(2V) [rad/s].
        to_rate = {"p": 2.0 * V / float(ac2.b_ref), "q": 2.0 * V / float(ac2.c_ref),
                   "r": 2.0 * V / float(ac2.b_ref)}
        for key in ("p", "q", "r"):
            if key not in ins:
                continue
            h = 1e-4 * to_rate[key]

            def _get(v: float, _k: str = key) -> dict[str, float]:
                c2 = dataclasses.replace(condition, **{_k: v})
                return _evaluate(ac, c2, _k, v - float(getattr(condition, _k, 0.0)))

            fd = _ladder(_get, float(getattr(condition, key, 0.0)), h, _STEP_FACTORS, outs, eps)
            _record(key, key, {o: float(jac[o][key]) for o in outs}, fd, to_rate[key])
    if "controls" in ins:
        from ventorum.sensitivity.adjoint import control_names

        for name in control_names(ac):
            h = 1e-4
            defl = [float(c.deflection) for s in ac.surfaces for c in getattr(s, "controls", [])
                    if c.name == name]
            side = _control_side(name, defl, h * max(_STEP_FACTORS))

            def _get(v: float, _n: str = name) -> dict[str, float]:
                ac3 = ac.clone()
                for s in ac3.surfaces:
                    for c in getattr(s, "controls", []):
                        if c.name == _n:
                            c.deflection = float(c.deflection) + float(v)
                return _evaluate(ac3, condition, f"controls.{_n}", v)

            fd = _ladder(_get, 0.0, h, _STEP_FACTORS, outs, eps, side=side)
            _record("controls", f"controls.{name}",
                    {o: float(jac[o]["controls"][name]) for o in outs}, fd, 1.0)
            if side:
                for o in outs:
                    details[f"{o}:controls.{name}"]["one_sided"] = side
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
                             _p: str = prop, _x0: float = x0) -> dict[str, float]:
                        ac3 = ac.clone()
                        _set_section_prop(ac3, _n, _j, _p, v)
                        return _evaluate(ac3, condition, f"sections.{_n}[{_j}].{_p}", v - _x0)

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
                return _evaluate(ac3, condition, f"reference.{_k}", v - ref_base[_k])

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
                return _evaluate(ac3, condition, f"reference.ref_point[{_kk}]", v - float(rp0[_kk]))

            fd = _ladder(_get, float(rp0[k]), h, _STEP_FACTORS, outs, eps)
            grads = {o: float(np.asarray(jac[o]["reference"]["ref_point"], dtype=float).reshape(3)[k])
                     for o in outs}
            _record("reference", f"reference.ref_point[{k}]", grads, fd, float(ac2.c_ref))
    if "planform" in ins:
        from ventorum.sensitivity import planform as PF
        from ventorum.solvers.factory import resolve_solver_type

        ac2 = ac.clone()
        ac2.compute_reference_values()
        c_ref = float(ac2.c_ref)
        canonical = resolve_solver_type(st.solver_type)
        collocation = "vlm" if canonical == "vlm" else "llt"
        base_geom = PF.build_planform(ac, st, collocation)
        mirror_auto = set(base_geom.mirror_ref_auto)
        base_snap = {n: (list(tr.snap_sec), tr.n_sp, tr.spacing)
                     for n, tr in base_geom.surfaces.items()}
        for tgt in _planform_targets(ac2, c_ref):
            if tgt.get("skip"):
                details[tgt["suffix"]] = {}
                details[tgt["suffix"]].update(rel_err=None, mode="skipped", compared=False,
                                              passed=None, reason=tgt["skip"])
                groups.setdefault("planform", {"compared": 0, "skipped": 0})["skipped"] += 1
                continue
            if tgt["surf"] in mirror_auto:
                details[tgt["suffix"]] = {}
                details[tgt["suffix"]].update(
                    rel_err=None, mode="skipped", compared=False, passed=None,
                    reason=("the surface is one copy of a mirror pair that gives the automatic "
                            "reference values: a change of one copy breaks the pair and the "
                            "production reference values jump, so the per-copy gradient has "
                            "no finite-difference reference (set explicit reference values)"))
                groups.setdefault("planform", {"compared": 0, "skipped": 0})["skipped"] += 1
                continue
            if _planform_topology_jumps(ac, st, collocation, base_geom, base_snap, tgt):
                details[tgt["suffix"]] = {}
                details[tgt["suffix"]].update(
                    rel_err=None, mode="skipped", compared=False, passed=None,
                    reason=("the mesh stations jump within the difference step "
                            "(a spanwise snap appears or moves): the fixed-topology "
                            "gradient has no finite-difference reference there"))
                groups.setdefault("planform", {"compared": 0, "skipped": 0})["skipped"] += 1
                continue

            def _get(v: float, _t: dict = tgt) -> dict[str, float]:
                ac3 = ac.clone()
                _set_planform_param(ac3, _t["surf"], _t["sec"], _t["key"], _t.get("sub"), v)
                return _evaluate(ac3, condition, f"planform.{_t['suffix']}", v - _t["x0"])

            fd = _ladder(_get, tgt["x0"], tgt["h"], _STEP_FACTORS, outs, eps)
            grads = {}
            for o in outs:
                jp = jac[o]["planform"][tgt["surf"]]
                if tgt["sec"] is None:
                    if tgt["key"] == "position":
                        g = float(np.asarray(jp["position"], dtype=float).reshape(3)[tgt["sub"]])
                    else:
                        g = float(jp[tgt["key"]])
                else:
                    g = float(jp["sections"][tgt["sec"]][tgt["key"]])
                grads[o] = g
            _record("planform", tgt["suffix"], grads, fd, tgt["scale"])
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
        h0 = 1e-3 * rc_min if node_step is None else float(node_step)
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
                return _evaluate(ac3, condition, "nodes", v)

            noise = _noise_level(_get, 0.0, _NOISE_SPACING * h0, outs)
            fd = _ladder(_get, 0.0, h0, _NODE_STEP_FACTORS, outs, eps, noise)
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
            "rtol": float(rtol), "groups": groups, "reasons": reasons, "details": details,
            "partials": grad_res.partials}


# Largest deflection magnitude of a control [rad] (the limit of
# ventorum.geometry.controls.validate_controls).
_DEFLECTION_LIMIT = float(np.radians(30.0))


def _control_side(name: str, deflections: list[float], h_max: float) -> int:
    """Return the side of the differences of one control name: 0 central, +1 or -1 one-sided.

    The central differences move every control with the name by up to
    +/- *h_max* [rad]. When this step goes past the deflection limit of
    30 deg for one of them, the differences are one-sided, away from the
    limit; the one-sided points reach 2 *h_max*.

    Parameters
    ----------
    name : str
        Control name.
    deflections : list of float
        Deflections of the controls with this name [rad].
    h_max : float
        Largest step of the step ladder [rad].

    Returns
    -------
    int
        0 (central), +1 or -1 (one-sided to that side).

    Raises
    ------
    ValueError
        When neither side stays inside the limit (controls with one name
        at both limits).
    """
    lim = _DEFLECTION_LIMIT
    if all(abs(d) + h_max <= lim for d in deflections):
        return 0
    for side in ((-1, 1) if max(deflections, key=abs) > 0.0 else (1, -1)):
        if all(abs(d + side * 2.0 * h_max) <= lim for d in deflections):
            return side
    raise ValueError(
        f"Control {name!r}: the differences of check_gradients cannot stay inside the "
        f"deflection limit of 30 deg on either side (deflections "
        f"{[round(float(np.degrees(d)), 6) for d in deflections]} deg). Check the controls of "
        "this name at a smaller deflection.")


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


def _planform_targets(ac: Any, c_ref: float) -> list:
    """List the finite-difference targets of the planform input.

    Each target holds the surface, the section (None for surface parameters),
    the key, the position component (for ``"position"``), the base value, the
    step, the suffix of the details entry, the input scale and, for an invalid
    direction, the skip reason. Parameters that do not exist (unset ``x_le``
    or ``z_le``, pinned root and tip span fractions) are omitted.
    """
    out = []
    for surf in ac.surfaces:
        has_z = bool(any(s.z_le is not None for s in surf.sections))
        for key in ("semi_span", "sweep_le", "dihedral", "incidence"):
            if key == "dihedral" and has_z:
                continue
            x0 = float(getattr(surf, key))
            h = 1e-4 if key != "semi_span" else 1e-4 * abs(x0)
            out.append({"surf": surf.name, "sec": None, "key": key, "sub": None,
                        "x0": x0, "h": h, "suffix": f"planform.{surf.name}.{key}",
                        "scale": 1.0 if key != "semi_span" else h / 1e-4})
        pos = np.asarray(surf.position, dtype=float).reshape(3)
        for k in range(3):
            if k == 1 and bool(surf.is_symmetric):
                out.append({"surf": surf.name, "sec": None, "key": "position", "sub": k,
                            "suffix": f"planform.{surf.name}.position[{k}]",
                            "skip": "position y of a symmetric surface must stay 0"})
                continue
            out.append({"surf": surf.name, "sec": None, "key": "position", "sub": k,
                        "x0": float(pos[k]), "h": 1e-4 * c_ref,
                        "suffix": f"planform.{surf.name}.position[{k}]",
                        "scale": c_ref})
        n = len(surf.sections)
        for j, sec in enumerate(surf.sections):
            base = {"chord": (float(sec.chord), 1e-4 * abs(float(sec.chord))),
                    "twist": (float(sec.twist), 1e-4),
                    "alpha_L0": (float(sec.airfoil.alpha_L0),
                                 max(1e-4 * abs(float(sec.airfoil.alpha_L0)), 1e-4))}
            for key, (x0, h) in base.items():
                out.append({"surf": surf.name, "sec": j, "key": key, "sub": None,
                            "x0": x0, "h": h,
                            "suffix": f"planform.{surf.name}.sec[{j}].{key}",
                            "scale": 1.0 if key != "chord" else h / 1e-4})
            for key in ("x_le", "z_le"):
                v = getattr(sec, key)
                if v is None:
                    continue
                x0 = float(v)
                chord = float(sec.chord)
                out.append({"surf": surf.name, "sec": j, "key": key, "sub": None,
                            "x0": x0, "h": max(1e-4 * abs(x0), 1e-4 * chord),
                            "suffix": f"planform.{surf.name}.sec[{j}].{key}",
                            "scale": chord})
            if 0 < j < n - 1:
                lo, hi = float(surf.sections[j - 1].y_frac), float(surf.sections[j + 1].y_frac)
                gap = min(float(sec.y_frac) - lo, hi - float(sec.y_frac))
                h = min(1e-4, 0.25 * gap) if gap > 0.0 else 0.0
                if h > 0.0:
                    out.append({"surf": surf.name, "sec": j, "key": "y_frac", "sub": None,
                                "x0": float(sec.y_frac), "h": h,
                                "suffix": f"planform.{surf.name}.sec[{j}].y_frac",
                                "scale": 1.0})
    return out


def _planform_topology_jumps(ac: Any, st: Any, collocation: str, base_geom: Any,
                             base_snap: dict, tgt: dict) -> bool:
    """Return True when the mesh stations jump within the difference step.

    A spanwise snap that appears, disappears or moves to another edge (or a
    spacing flip) changes the mesh stations by much more than the smooth
    motion with the parameter. The fixed-topology gradient then has no
    finite-difference reference there.

    An edge that lies exactly on its break is not a jump here: both
    difference steps move the edge with the break and use the edge means,
    and the y_frac gradient of that section is the derivative of that
    branch (see :func:`ventorum.sensitivity.adjoint._adjoint_solve`).
    """
    from ventorum.sensitivity import planform as PF

    for sgn in (+1.0, -1.0):
        ac3 = ac.clone()
        _set_planform_param(ac3, tgt["surf"], tgt["sec"], tgt["key"], tgt.get("sub"),
                            tgt["x0"] + sgn * tgt["h"])
        try:
            geom = PF.build_planform(ac3, st, collocation)
        except (RuntimeError, ValueError):
            return True
        for name, (snap, n_sp, spacing) in base_snap.items():
            tr = geom.surfaces[name]
            if (list(tr.snap_sec) != list(snap) or tr.n_sp != n_sp
                    or tr.spacing != spacing):
                return True
    return False


def _set_planform_param(ac: Any, surf_name: str, sec: Any, key: str,
                        sub: Any, v: float) -> None:
    """Set one planform parameter of a cloned aircraft to *v*."""
    surf = next(s for s in ac.surfaces if s.name == surf_name)
    if sec is None:
        if key == "position":
            pos = np.asarray(surf.position, dtype=float).reshape(3).copy()
            pos[int(sub)] = float(v)
            surf.position = pos
        else:
            setattr(surf, key, float(v))
        return
    s = surf.sections[int(sec)]
    if key == "alpha_L0":
        from ventorum.core.datatypes import LinearAirfoil

        af = s.airfoil
        s.airfoil = LinearAirfoil(name=af.name, a0=float(af.a0), alpha_L0=float(v),
                                  Cd0=float(af.Cd0), Cm0=float(af.Cm0))
    else:
        setattr(s, key, float(v))
