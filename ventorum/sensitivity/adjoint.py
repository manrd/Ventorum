# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Adjoint gradients of the vortex lattice and the linear lifting line.

Discrete adjoint: residual R(G, x) = A(x) G - b(x). The circulation G comes
from the normal production solve. One LU factor of A gives G and every
adjoint vector: A^T lambda_j = df_j/dG. The total derivative is the gradient
of the Lagrangian f_j(G*, x) - lambda_j . (A(x) G* - b(x)) at the fixed
converged G* (Martins and Hwang 2013; Martins and Ning 2021, ch. 6). No
automatic differentiation runs through the linear solve.

References
----------
* J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods
  for computing derivatives of multidisciplinary computational models",
  AIAA Journal 51(11), 2013, pp. 2582-2599.
* J. R. R. A. Martins and A. Ning, "Engineering Design Optimization",
  Cambridge University Press, 2021, ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.sensitivity.reference import TorchReference, require_torch
from ventorum.core.datatypes import SolverSettings as _SolverSettings

SCALAR_OUTPUTS = ("CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn")
VECTOR_OUTPUTS = ("strip_force", "section_cl")
ALL_OUTPUTS = SCALAR_OUTPUTS + VECTOR_OUTPUTS
SCALAR_INPUTS = ("alpha", "beta", "V_inf", "rho", "h", "phi")
ALL_WRT = ("nodes", "alpha", "beta", "V_inf", "rho", "h", "phi", "sections", "reference")

# Inputs that keep a symmetric state symmetric (no beta, phi, node, section
# or reference-point change that breaks the y = 0 symmetry).
_SYMMETRIC_WRT = ("alpha", "V_inf", "rho", "h")
# Outputs that keep a symmetric state symmetric (longitudinal only).
_SYMMETRIC_OUTPUTS = ("CL", "CDi", "CDp", "CD", "Cm")

# Phase-2 card for refused configurations (nonlinear lifting line, tabulated
# airfoils, controls, body rates, planform parameters).
PHASE2_CARD = "T-0071"


def _gradient_device(result: Any) -> str:
    """Return the torch device of the gradient computation.

    The gradients run on the device of the production solve: the device
    selection of :mod:`ventorum.gpu` (setting ``"auto"``, ``"cpu"`` or
    ``"gpu"``, and the size rule of ``"auto"``) decides it for the solve,
    and the gradients follow it.
    """
    require_torch()
    import torch

    if str(result.details.get("device", "cpu")) == "gpu" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gradient_precision() -> str:
    """Return the precision of the gradient partials and the adjoint solve."""
    from ventorum import gpu

    precision = gpu.get_precision()
    return precision if precision in ("float32", "float64") else "float32"


def _torch_dtype(precision: str):  # type: ignore[no-untyped-def]
    """Return the torch dtype of *precision*."""
    import torch

    return torch.float64 if precision == "float64" else torch.float32


def _adjoint_use_symmetry(production_symmetric: bool, wrt: tuple[str, ...],
                          outputs: tuple[str, ...]) -> bool:
    """Return True when the folded symmetric system gives the exact adjoint.

    The fold constrains the circulation to the symmetric subspace, so it is
    exact only when the case is symmetric and every requested input and
    output is symmetric (no beta, phi, node, section or reference change,
    no lateral output). Else the full system is used.
    """
    if not production_symmetric:
        return False
    if any(w not in _SYMMETRIC_WRT for w in wrt):
        return False
    return all(o in _SYMMETRIC_OUTPUTS for o in outputs)


def _check_names(outputs: Any, wrt: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Check output and input names; return them as tuples."""
    outs = tuple(outputs)
    ins = tuple(wrt)
    if not outs:
        raise ValueError(f"No output requested; use one or more of {list(ALL_OUTPUTS)}.")
    if not ins:
        raise ValueError(f"No input requested; use one or more of {list(ALL_WRT)}.")
    for o in outs:
        if o not in ALL_OUTPUTS:
            raise ValueError(f"Unknown output {o!r}; use one of {list(ALL_OUTPUTS)}.")
    for w in ins:
        if w not in ALL_WRT:
            raise ValueError(f"Unknown input {w!r}; use one of {list(ALL_WRT)}.")
    return outs, ins


def _refuse_configuration(ac: Any, condition: Any, settings: Any, canonical: str) -> None:
    """Raise ValueError for configurations outside phase 1 (naming the phase-2 card)."""
    if canonical == "fourier":
        raise ValueError(
            f"The Fourier solver has no adjoint gradients in phase 1 (see {PHASE2_CARD}).")
    if canonical == "nonlinear":
        raise ValueError(
            f"The nonlinear lifting line has no adjoint gradients in phase 1 (see {PHASE2_CARD}).")
    for surf in ac.surfaces:
        for sec in surf.sections:
            from ventorum.core.datatypes import TabulatedAirfoil

            if isinstance(sec.airfoil, TabulatedAirfoil):
                raise ValueError(
                    "Tabulated airfoils have no adjoint gradients in phase 1 "
                    f"(see {PHASE2_CARD}); use linear airfoils.")
        for c in getattr(surf, "controls", []):
            if float(c.deflection) != 0.0:
                raise ValueError(
                    f"Control deflection {c.name!r} = {float(c.deflection):g} rad is not zero: "
                    f"controls have no adjoint gradients in phase 1 (see {PHASE2_CARD}).")
    if (float(getattr(condition, "p", 0.0)) != 0.0
            or float(getattr(condition, "q", 0.0)) != 0.0
            or float(getattr(condition, "r", 0.0)) != 0.0):
        raise ValueError(
            f"Body rates have no adjoint gradients in phase 1 (see {PHASE2_CARD}).")


def _recover_xi(lattice: Any) -> np.ndarray:
    """Return the chordwise fractions (n_strips, n_chord + 1) of the lattice.

    The fractions come from the bound-vortex positions, so hinge-shifted
    panels of undeflected controls are reproduced exactly.
    """
    n_s = int(lattice.n_strips)
    n_c = int(lattice.n_chord)
    xi = np.zeros((n_s, n_c + 1))
    a = np.asarray(lattice.a, dtype=float).reshape(n_s, n_c, 3)
    le_l = np.asarray(lattice.le_left, dtype=float)
    te_l = np.asarray(lattice.te_left, dtype=float)
    for s in range(n_s):
        edge = te_l[s] - le_l[s]
        denom = float(edge @ edge)
        x = 0.0
        xi[s, 0] = 0.0
        for k in range(n_c):
            f = float((a[s, k] - le_l[s]) @ edge / denom) if denom > 0.0 else x + 0.25 / n_c
            xn = x + (f - x) / 0.25
            xi[s, k + 1] = xn
            x = xn
        xi[s, -1] = 1.0
    return xi


def build_reference(
    aircraft: Any,
    condition: Any,
    settings: Any,
    lattice: Any,
    axes: str,
    device: str,
    precision: str,
    symmetric: bool,
    accum: str = "sequential",
    local_origin: bool = False,
) -> tuple[TorchReference, dict]:
    """Build the torch reference model and the base input values.

    Returns (model, base) with the numpy base values of every input.
    *accum* selects the source summation (``"sequential"``, ``"pairwise"``
    or ``"kahan"``); *local_origin* measures the coordinates from the
    detached centroid of the lattice. The gradient functions use the
    defaults (sequential, body axes): the other variants gave the same
    float32 accuracy within the round-off and are slower; they stay here
    for comparisons only.
    """
    import torch

    if accum not in ("sequential", "pairwise", "kahan"):
        raise ValueError(f"Unknown accumulation {accum!r}.")

    dtype = _torch_dtype(precision)
    n_s = int(lattice.n_strips)
    n_p = int(lattice.n_panels)
    panel_strip = np.asarray(lattice.panel_strip, dtype=np.int64)
    n_chord = int(lattice.n_chord)
    collocation = str(lattice.collocation)
    use_sym = bool(symmetric)
    if use_sym:
        from ventorum.aero.system import make_unknown_map

        umap = make_unknown_map(lattice, True)
    else:
        from ventorum.aero.system import make_unknown_map

        umap = make_unknown_map(lattice, False)
    unknown_panels = np.asarray(umap.unknown_panels, dtype=np.int64)
    panel_column = np.asarray(umap.panel_column, dtype=np.int64)
    col_real = panel_column.copy()
    n_u = int(umap.n)
    fold_loads = bool(use_sym and np.all(np.asarray(lattice.panel_mirror) >= 0))
    panel_mirror = np.asarray(lattice.panel_mirror, dtype=np.int64)
    right_panels = np.flatnonzero(np.asarray(lattice.strip_is_right)[panel_strip])
    left_panels = np.flatnonzero(~np.asarray(lattice.strip_is_right)[panel_strip])
    strip_of_unknown = panel_strip[unknown_panels]
    g = lattice.strip_core_group
    strip_groups = np.asarray(lattice.strip_surface if g is None else g, dtype=np.int64)
    src_group = strip_groups[panel_strip]
    from ventorum.aero.system import panel_targets

    tg_u = panel_targets(lattice, unknown_panels)
    tg_p = panel_targets(lattice, np.arange(n_p))
    xi = _recover_xi(lattice)
    frac = np.asarray(lattice.cp_frac, dtype=float)
    cp_frac_panel = frac[panel_strip]

    # Per-surface defining edges and global strip edge indices.
    surfaces: list[dict] = []
    strip_edges = np.zeros((n_s, 2), dtype=np.int64)
    offset = 0
    for sl in lattice.surfaces:
        full_le = np.asarray(sl.edge_le, dtype=float)
        full_te = np.asarray(sl.edge_te, dtype=float)
        name = sl.name
        # Find the aircraft surface for the mirror flag.
        ac_s = next((s for s in aircraft.surfaces if s.name == name), None)
        mirror_y = bool(getattr(ac_s, "mirror_y", False)) if ac_s is not None else False
        if sl.is_symmetric:
            e = (full_le.shape[0] + 1) // 2
            base_le, base_te = full_le[e - 1:].copy(), full_te[e - 1:].copy()
            kind = "symmetric"
        elif mirror_y:
            kind = "mirror"
            mir = np.array([1.0, -1.0, 1.0])
            base_le = (full_le * mir)[::-1].copy()
            base_te = (full_te * mir)[::-1].copy()
        else:
            kind = "plain"
            base_le, base_te = full_le.copy(), full_te.copy()
        surfaces.append({"name": name, "kind": kind, "n_edges": int(base_le.shape[0]),
                         "base_le": base_le, "base_te": base_te})
        n_full = int(full_le.shape[0])
        for k in range(sl.strips.start, sl.strips.stop):
            strip_edges[k, 0] = offset + (k - sl.strips.start)
            strip_edges[k, 1] = offset + (k - sl.strips.start) + 1
        offset += n_full

    def _t(name: str, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(arr).copy(), dtype=dtype, device=device)

    def _l(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(arr).copy(), dtype=torch.int64, device=device)

    build = {
        "device": torch.device(device),
        "dtype": dtype,
        "collocation": collocation,
        "axes": axes,
        "n_strips": n_s,
        "n_panels": n_p,
        "n_chord": n_chord,
        "n_unknowns": n_u,
        "n_sources": n_p,
        "wake_body": str(getattr(settings, "wake_alignment", "freestream")) == "body",
        "surfaces": surfaces,
        "strip_edges": _l(strip_edges),
        "xi": _t("xi", xi),
        "frac": _t("frac", frac),
        "panel_strip": _l(panel_strip),
        "unknown_panels": _l(unknown_panels),
        "column": _l(col_real),
        "panel_column_full": _l(panel_column),
        "strip_of_unknown": _l(strip_of_unknown),
        "src_group": _l(src_group),
        "strip_groups": _l(strip_groups),
        "tgt_group_unknown": _l(np.asarray(tg_u.group, dtype=np.int64)),
        "tgt_group_panels": _l(np.asarray(tg_p.group, dtype=np.int64)),
        "cp_frac_panel": _t("cpf", cp_frac_panel),
        "fold_loads": bool(fold_loads),
        "panel_mirror": _l(panel_mirror),
        "right_panels": _l(right_panels),
        "left_panels": _l(left_panels),
        "accum": accum,
        "origin": None,
    }
    if local_origin:
        pts = np.vstack(
            [np.asarray(s["base_le"]) for s in surfaces]
            + [np.asarray(s["base_te"]) for s in surfaces])
        origin = torch.as_tensor(np.ascontiguousarray(pts.mean(axis=0)).copy(),
                                 dtype=dtype, device=device)
        build["origin"] = origin.detach()
    base = {
        "alpha": float(condition.alpha),
        "beta": float(condition.beta),
        "V_inf": float(condition.V_inf),
        "rho": float(condition.rho),
        "h": None if condition.h is None else float(condition.h),
        "phi": float(getattr(condition, "phi", 0.0)),
        "a0": np.asarray(lattice.a0, dtype=float).copy(),
        "alpha_L0": np.asarray(lattice.alpha_L0, dtype=float).copy(),
        "Cd0": np.asarray(lattice.Cd0, dtype=float).copy(),
        "Cm0": np.asarray(lattice.Cm0, dtype=float).copy(),
        "S_ref": float(aircraft.S_ref),
        "b_ref": float(aircraft.b_ref),
        "c_ref": float(aircraft.c_ref),
        "ref_point": np.asarray(aircraft.moment_reference(), dtype=float).copy(),
    }
    return TorchReference(build), base


def make_input_tensors(
    model: TorchReference,
    base: dict,
    wrt: tuple[str, ...],
    device: str,
    precision: str,
) -> dict:
    """Return the torch input tensors; requested inputs require grad."""
    import torch

    dtype = _torch_dtype(precision)
    dev = torch.device(device)

    def _s(name: str, value: float) -> torch.Tensor:
        t = torch.tensor(float(value), dtype=dtype, device=dev)
        if name in wrt:
            t.requires_grad_(True)
        return t

    def _a(arr: np.ndarray, flag: bool) -> torch.Tensor:
        out = torch.as_tensor(np.ascontiguousarray(arr).copy(), dtype=dtype, device=dev)
        if flag:
            out.requires_grad_(True)
        return out

    t: dict = {}
    t["alpha"] = _s("alpha", base["alpha"])
    t["beta"] = _s("beta", base["beta"])
    t["V_inf"] = _s("V_inf", base["V_inf"])
    t["rho"] = _s("rho", base["rho"])
    t["phi"] = _s("phi", base["phi"])
    if base["h"] is None:
        t["h"] = None
    else:
        t["h"] = _s("h", base["h"])
    for s in model.b["surfaces"]:
        for tag, key in (("le", "base_le"), ("te", "base_te")):
            t[f"nodes:{s['name']}:{tag}"] = _a(s[key], "nodes" in wrt)
    for key in ("a0", "alpha_L0", "Cd0", "Cm0"):
        t[key] = _a(base[key], "sections" in wrt)
    for key in ("S_ref", "b_ref", "c_ref"):
        t[key] = _s("reference", base[key])
    rp = _a(base["ref_point"], "reference" in wrt)
    t["ref_point"] = rp
    return t


@dataclass
class GradientResult:
    """Adjoint gradients of one flight state.

    Attributes
    ----------
    values : dict[str, float]
        Production output values of the scalar outputs.
    jacobian : dict[str, Any]
        Gradient of each output per input. ``"nodes"`` maps each surface
        name to ``{"le": array, "te": array}`` with the shape of
        ``NodeDisplacements.le`` and ``.te``. ``"sections"`` maps each
        property name to an array per strip. ``"reference"`` maps each
        reference value to a float, with ``"ref_point"`` an array of 3.
    device : str
        Torch device of the gradient computation (``"cpu"`` or ``"cuda"``).
    precision_primal : str
        Precision of the production solve (``"float64"`` on the CPU; the
        GPU precision setting on the GPU).
    precision_gradient : str
        Precision of the gradient partials and the adjoint solve.
    method : str
        Always ``"adjoint"``.
    forward_mismatch : float
        Largest relative difference between the differentiated reference
        copy and the production result (see :func:`gradients`).
    n_solves : int
        Number of production solves (always 1).
    n_adjoint : int
        Number of adjoint solves (one per output).
    symmetric : bool
        True when the folded symmetric system was used.
    notes : list[str]
        Notes on the method and its limits.
    """

    values: dict[str, float]
    jacobian: dict[str, Any]
    device: str
    precision_primal: str
    precision_gradient: str
    method: str = "adjoint"
    forward_mismatch: float = 0.0
    n_solves: int = 1
    n_adjoint: int = 0
    symmetric: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary (arrays as lists)."""

        def _conv(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: _conv(u) for k, u in v.items()}
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return float(v) if isinstance(v, (int, float)) else v

        return {
            "values": {k: float(v) for k, v in self.values.items()},
            "jacobian": {k: _conv(v) for k, v in self.jacobian.items()},
            "device": self.device,
            "precision_primal": self.precision_primal,
            "precision_gradient": self.precision_gradient,
            "method": self.method,
            "forward_mismatch": float(self.forward_mismatch),
            "n_solves": int(self.n_solves),
            "n_adjoint": int(self.n_adjoint),
            "symmetric": bool(self.symmetric),
            "notes": list(self.notes),
        }


def _prepare_case(geometry: Any, condition: Any, settings: Any, axes: str,
                  wrt: tuple[str, ...], outputs: tuple[str, ...]) -> dict:
    """Run the production solve and build the reference model.

    Returns a dict with the aircraft, lattice, result, model, base values,
    device, precision and flags. Raises ValueError for refused input.
    The adjoint uses the folded symmetric system only when the case is
    symmetric and every requested input and output is symmetric.
    """
    from ventorum.core.datatypes import Aircraft, LiftingSurface
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import (
        validate_aircraft,
        validate_flight_condition,
        validate_solver_settings,
    )

    if axes not in ("body", "stability", "wind"):
        raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}.")
    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    st = canonical_settings(settings.clone() if settings is not None else _SolverSettings())
    validate_flight_condition(condition)
    validate_solver_settings(st)
    canonical = resolve_solver_type(st.solver_type)
    _refuse_configuration(ac, condition, st, canonical)
    for v in (condition.alpha, condition.beta, condition.V_inf, condition.rho,
              getattr(condition, "phi", 0.0)):
        if not math.isfinite(float(v)):
            raise ValueError(f"Flight condition has a non-finite value {v!r}.")
    if condition.h is not None and not math.isfinite(float(condition.h)):
        raise ValueError(f"Flight condition has a non-finite height {condition.h!r}.")
    validate_aircraft(ac)
    solver = make_solver(canonical)
    result = solver.solve(ac, condition, st)
    lattice = result.details["lattice"]
    production_symmetric = bool(result.symmetry_used)
    symmetric = _adjoint_use_symmetry(production_symmetric, wrt, outputs)
    device = _gradient_device(result)
    precision = _gradient_precision()
    precision_primal = str(result.details.get("precision", "float64"))
    model, base = build_reference(ac, condition, st, lattice, axes, device, precision, symmetric)
    gamma_panel = np.asarray(result.details["gamma"], dtype=float)
    col = np.asarray(model.b["panel_column_full"].cpu())
    # Circulation per unknown from the production panels (mean of the mapped panels).
    cnt = np.bincount(col, minlength=int(model.b["n_unknowns"]))
    acc = np.zeros(int(model.b["n_unknowns"]))
    np.add.at(acc, col, gamma_panel)
    g_unknown = acc / np.maximum(cnt, 1)
    return {
        "aircraft": ac, "settings": st, "condition": condition, "canonical": canonical,
        "result": result, "lattice": lattice, "model": model, "base": base,
        "device": device, "precision": precision, "precision_primal": precision_primal,
        "symmetric": symmetric, "production_symmetric": production_symmetric,
        "g_unknown": g_unknown, "axes": axes,
    }


def _adjoint_solve(
    prep: dict,
    wrt: tuple[str, ...],
    make_targets: Any,
    vectors: tuple[str, ...] = (),
) -> tuple[dict, dict, float, int]:
    """Run the adjoint computation for the targets of *make_targets*.

    *make_targets* maps the torch outputs dict to a list of (name, scalar).
    One forward pass of the differentiated model gives the outputs, the
    system and the values of the forward check; *vectors* names the vector
    outputs that the forward check also compares.
    Returns (values, jacobian, forward_mismatch, n_adjoint).
    """
    import torch

    model: TorchReference = prep["model"]
    base = prep["base"]
    device, precision = prep["device"], prep["precision"]
    t = make_input_tensors(model, base, wrt, device, precision)
    g0 = torch.as_tensor(np.ascontiguousarray(prep["g_unknown"]),
                         dtype=model.b["dtype"], device=model.b["device"])
    g_var = g0.detach().clone().requires_grad_(True)
    outputs, A, rhs = model.forward(t, g_var)
    targets = make_targets(outputs)
    A_det = A.detach()
    g_star = g0.detach()
    prod = _production_values(prep)
    worst = _forward_check(prep, outputs, prod, vectors)
    # One backward pass per target gives df/dG (the adjoint right-hand
    # side) and the explicit partials df/dx together.
    names = [name for name, _ in targets]
    leaves = [v for v in t.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    grads_g = []
    grads_fx = []
    for _, f in targets:
        # An output without a dependence on the circulation or the inputs
        # (for example CDp with respect to alpha only) has no graph.
        if f.requires_grad:
            parts = torch.autograd.grad(f, [g_var] + leaves, retain_graph=True, allow_unused=True)
        else:
            parts = (None,) * (1 + len(leaves))
        gg = parts[0] if parts[0] is not None else torch.zeros_like(g_var)
        grads_g.append(gg.detach())
        grads_fx.append(parts[1:])
    B = torch.stack(grads_g, dim=1)
    Lambda = torch.linalg.solve(A_det.T, B)
    residual = A @ g_star - rhs
    jacobian: dict = {}
    values = {k: prod[k] for k in names if k in prod}
    for j, name in enumerate(names):
        # Total derivative: df/dx - lambda^T dR/dx at the fixed G*.
        lam = Lambda[:, j].detach()
        r_term = (lam * residual).sum()
        if r_term.requires_grad:
            grads_r = torch.autograd.grad(r_term, leaves, retain_graph=True, allow_unused=True)
        else:
            grads_r = (None,) * len(leaves)
        gmap = {}
        for v, gf, gr in zip(leaves, grads_fx[j], grads_r):
            if gf is None and gr is None:
                gmap[id(v)] = None
                continue
            tot = (gf if gf is not None else 0.0) - (gr if gr is not None else 0.0)
            gmap[id(v)] = tot.detach().cpu().numpy()
        jacobian[name] = _pack_jacobian(model, t, gmap, wrt, base)
    return values, jacobian, worst, len(targets)


# Scalar outputs that are components of one force vector and of one
# moment vector (relative scale of the forward check).
_FORCE_VECTOR = ("CL", "CY")
_MOMENT_VECTOR = ("Cl", "Cm", "Cn")


def _forward_check(prep: dict, outputs: dict, prod: dict[str, float],
                   vectors: tuple[str, ...]) -> float:
    """Compare the differentiated model with the production result.

    Returns the largest relative difference (``forward_mismatch``). Raises
    RuntimeError with the output names when it is above 1e-10 (float64) or
    1e-4 (float32).

    The difference of CDi, CDp and CD is relative to the production value
    of each. CL and CY are components of one force vector, and Cl, Cm and
    Cn are components of one moment vector: their differences are relative
    to the largest production magnitude of the components of the same
    vector (the normwise relative error of the vector). The round-off of a
    vector component is relative to the vector, so a small component (CY
    in a small sideslip, or CY, Cl and Cn of a symmetric state, which are
    zero by symmetry) has no relative scale of its own. An output with an
    exact zero production value (for example CDp with Cd0 = 0) uses the
    force scale.

    A requested vector output (``strip_force`` [N], ``section_cl``) is
    compared with its production array: the largest difference relative
    to the largest production magnitude of the array, because single
    components go through zero along the span.
    """
    precision = prep["precision"]
    both64 = precision == "float64" and prep["precision_primal"] == "float64"
    tol = 1e-10 if both64 else 1e-4
    force_scale = max(abs(prod[k]) for k in _FORCE_VECTOR)
    moment_scale = max(abs(prod[k]) for k in _MOMENT_VECTOR)
    worst = 0.0
    bad = []
    for k, p in prod.items():
        r = float(outputs[k].detach().cpu())
        if not (math.isfinite(r) and math.isfinite(p)):
            worst = float("inf")
            bad.append(k)
            continue
        if k in _FORCE_VECTOR:
            denom = force_scale
        elif k in _MOMENT_VECTOR:
            denom = moment_scale
        else:
            denom = abs(p) if p != 0.0 else force_scale
        rel = abs(r - p) / denom if denom > 0.0 else (0.0 if r == p else float("inf"))
        worst = max(worst, rel)
        if rel > tol:
            bad.append(k)
    for k, arr in _production_vectors(prep, vectors).items():
        r = outputs[k].detach().cpu().numpy().reshape(arr.shape)
        scale = float(np.max(np.abs(arr)))
        diff = float(np.max(np.abs(r - arr)))
        rel = diff / scale if scale > 0.0 else (0.0 if diff == 0.0 else float("inf"))
        if not np.isfinite(rel):
            rel = float("inf")
        worst = max(worst, rel)
        if rel > tol:
            bad.append(k)
    if bad:
        raise RuntimeError(
            f"The reference copy differs from the production result above {tol:g} "
            f"(outputs: {', '.join(bad)}; worst {worst:.3g}). "
            "The gradient would not be the gradient of the reported result.")
    return worst


def _production_vectors(prep: dict, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return the production arrays of the vector outputs in *names*."""
    loads = prep["result"].details["loads"]
    out = {}
    if "strip_force" in names:
        out["strip_force"] = np.asarray(loads.strip_force, dtype=float)
    if "section_cl" in names:
        out["section_cl"] = np.concatenate([np.asarray(sp.Cl, dtype=float) for sp in loads.spanwise])
    return out


def _production_values(prep: dict) -> dict[str, float]:
    """Return the production scalar values in the requested axes."""
    prod = {
        "CL": float(prep["result"].totals.CL),
        "CY": float(prep["result"].totals.CY),
        "CDi": float(prep["result"].totals.CDi),
        "CDp": float(prep["result"].totals.CDp if prep["result"].totals.CDp is not None else 0.0),
        "CD": float(prep["result"].totals.CD_total
                    if prep["result"].totals.CD_total is not None
                    else prep["result"].totals.CDi),
    }
    m = prep["result"].moments(prep["axes"])
    prod["Cl"] = float(m["Cl"])
    prod["Cm"] = float(m["Cm"])
    prod["Cn"] = float(m["Cn"])
    return prod


def _pack_jacobian(model: TorchReference, t: dict, gmap: dict, wrt: tuple[str, ...],
                   base: dict) -> dict:
    """Pack the flat autograd gradients into the jacobian structure."""
    j: dict = {}
    if "nodes" in wrt:
        nodes: dict = {}
        for s in model.b["surfaces"]:
            gle = gmap.get(id(t[f"nodes:{s['name']}:le"]))
            gte = gmap.get(id(t[f"nodes:{s['name']}:te"]))
            n_e = int(s["n_edges"])
            nodes[s["name"]] = {
                "le": np.zeros((n_e, 3)) if gle is None else np.asarray(gle, dtype=float),
                "te": np.zeros((n_e, 3)) if gte is None else np.asarray(gte, dtype=float),
            }
        j["nodes"] = nodes
    for key in SCALAR_INPUTS:
        if key in wrt:
            if key == "h" and base["h"] is None:
                continue
            gr = gmap.get(id(t[key])) if t.get(key) is not None else None
            j[key] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
    if "sections" in wrt:
        sec = {}
        for key in ("a0", "alpha_L0", "Cd0", "Cm0"):
            gr = gmap.get(id(t[key]))
            sec[key] = np.zeros_like(base[key]) if gr is None else np.asarray(gr, dtype=float)
        j["sections"] = sec
    if "reference" in wrt:
        ref = {}
        for key in ("S_ref", "b_ref", "c_ref"):
            gr = gmap.get(id(t[key]))
            ref[key] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
        gr = gmap.get(id(t["ref_point"]))
        ref["ref_point"] = np.zeros(3) if gr is None else np.asarray(gr, dtype=float).reshape(3)
        j["reference"] = ref
    return j


def gradients(
    geometry: Any,
    condition: Any,
    settings: Any = None,
    *,
    outputs: tuple[str, ...] = ("CL", "CD", "Cm"),
    wrt: tuple[str, ...] = ("alpha", "nodes"),
    axes: str = "body",
) -> GradientResult:
    """Compute exact gradients of scalar solver outputs by the discrete adjoint.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. It is not changed. The mesh topology is fixed.
    condition : FlightCondition
        Flight condition. Angles in rad, height in m.
    settings : SolverSettings or None
        Solver settings. Only ``"vlm"`` and ``"linear"`` with linear
        airfoils, zero control deflections and zero body rates are accepted.
    outputs : tuple of str
        Scalar outputs from ``"CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn"``.
        Vector outputs go through :func:`gradient_vjp`.
    wrt : tuple of str
        Inputs from ``"nodes", "alpha", "beta", "V_inf", "rho", "h", "phi",
        "sections", "reference"``. Lengths in m, angles in rad.
    axes : str
        Axes of the moments: ``"body"``, ``"stability"`` or ``"wind"``.

    Returns
    -------
    GradientResult
        Values, jacobian, device, precisions, method, forward mismatch,
        solve counts, symmetry flag and notes.

    Raises
    ------
    ValueError
        For an unknown output or input name, a refused configuration, or a
        non-finite input.
    RuntimeError
        When the reference copy differs from the production result above
        the tolerance: the gradient would not match the reported result.
    """
    outs, ins = _check_names(outputs, wrt)
    for o in outs:
        if o in VECTOR_OUTPUTS:
            raise ValueError(
                f"Output {o!r} is a vector output; use gradient_vjp with seeds.")
    prep = _prepare_case(geometry, condition, settings, axes, ins, outs)

    def _make(outputs: dict) -> list:
        return [(o, outputs[o]) for o in outs]

    values, jacobian, worst, n_adj = _adjoint_solve(prep, ins, _make)
    notes = [
        "Discrete adjoint of the production solve (Martins and Hwang 2013; Martins and Ning 2021, ch. 6).",
        "The mesh topology is fixed (spanwise stations, chordwise count and fractions).",
    ]
    if prep["precision"] == "float32":
        notes.append(
            "Float32 gradients: the D-13 gate is 1e-3 relative; check the result with "
            "check_gradients (rtol=1e-3).")
    return GradientResult(
        values=values, jacobian=jacobian, device=prep["device"],
        precision_primal=prep["precision_primal"], precision_gradient=prep["precision"],
        forward_mismatch=worst, n_adjoint=n_adj, symmetric=prep["symmetric"], notes=notes)


def gradient_vjp(
    geometry: Any,
    condition: Any,
    settings: Any = None,
    *,
    seeds: dict,
    wrt: tuple[str, ...] = ("alpha", "nodes"),
    axes: str = "body",
) -> dict:
    """Compute the gradient of a weighted sum of outputs (vector-Jacobian product).

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. It is not changed.
    condition : FlightCondition
        Flight condition.
    settings : SolverSettings or None
        Solver settings (same limits as :func:`gradients`).
    seeds : dict
        Output name to weight: a float for scalar outputs, an array of
        shape (n_strips, 3) for ``"strip_force"`` and (n_strips,) for
        ``"section_cl"``.
    wrt : tuple of str
        Inputs, as in :func:`gradients`.
    axes : str
        Axes of the moments.

    Returns
    -------
    dict
        Gradient of the weighted sum, same input structure as the jacobian
        of :func:`gradients`.
    """
    if not seeds:
        raise ValueError(f"seeds must name at least one output; use one of {list(ALL_OUTPUTS)}.")
    for o in seeds:
        if o not in ALL_OUTPUTS:
            raise ValueError(f"Unknown output {o!r}; use one of {list(ALL_OUTPUTS)}.")
    _, ins = _check_names(tuple(seeds), wrt)
    prep = _prepare_case(geometry, condition, settings, axes, ins, tuple(seeds))
    import torch

    model: TorchReference = prep["model"]

    def _make(outputs: dict) -> list:
        total = None
        for o, w in seeds.items():
            f = outputs[o]
            if o in SCALAR_OUTPUTS:
                term = float(w) * f
            else:
                wt = torch.as_tensor(np.ascontiguousarray(np.asarray(w)).copy(),
                                     dtype=model.b["dtype"], device=model.b["device"])
                term = (wt * f).sum()
            total = term if total is None else total + term
        assert total is not None
        return [("__vjp__", total)]

    vectors = tuple(o for o in seeds if o in VECTOR_OUTPUTS)
    _, jacobian, _, _ = _adjoint_solve(prep, ins, _make, vectors=vectors)
    return jacobian["__vjp__"]

