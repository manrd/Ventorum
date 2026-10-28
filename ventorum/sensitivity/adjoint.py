# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Adjoint gradients of the vortex lattice and the linear and nonlinear lifting lines.

Discrete adjoint: residual R(G, x) = A(x) G - b(x) for the linear solvers.
The circulation G comes from the normal production solve. One LU factor of
A gives G and every adjoint vector: A^T lambda_j = df_j/dG. The total
derivative is the gradient of the Lagrangian f_j(G*, x) - lambda_j . R(G*, x)
at the fixed converged G* (Martins and Hwang 2013; Martins and Ning 2021,
ch. 6). No automatic differentiation runs through the solve. The nonlinear
lifting line has a residual that is nonlinear in G; its adjoint uses the
Newton Jacobian dR/dG at the converged state (Phillips and Snyder 2000).

References
----------
* J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods
  for computing derivatives of multidisciplinary computational models",
  AIAA Journal 51(11), 2013, pp. 2582-2599.
* J. R. R. A. Martins and A. Ning, "Engineering Design Optimization",
  Cambridge University Press, 2021, ch. 6.
* W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic
  lifting-line theory", Journal of Aircraft 37(4), 2000, pp. 662-670.
* H. Glauert, "The Elements of Aerofoil and Airscrew Theory", Cambridge
  University Press, 1926: the thin-airfoil flap effectiveness.
* W. Oettli and W. Prager, "Compatibility of approximate solution of linear
  equations with given error bounds for coefficients and right-hand sides",
  Numerische Mathematik 6, 1964, pp. 405-409: the componentwise backward
  error of the residual check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.sensitivity.errors import GradientScopeError  # noqa: F401 (re-export)
from ventorum.sensitivity.reference import TorchReference, require_torch
from ventorum.core.datatypes import SolverSettings as _SolverSettings

SCALAR_OUTPUTS = ("CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn")
VECTOR_OUTPUTS = ("strip_force", "section_cl")
ALL_OUTPUTS = SCALAR_OUTPUTS + VECTOR_OUTPUTS
SCALAR_INPUTS = ("alpha", "beta", "V_inf", "rho", "h", "phi", "p", "q", "r", "mach")
RATE_INPUTS = ("p", "q", "r")
ALL_WRT = ("nodes", "alpha", "beta", "V_inf", "rho", "h", "phi", "sections", "reference",
           "planform", "controls", "p", "q", "r", "mach")

# Inputs that keep a symmetric state symmetric (no beta, phi, roll or yaw
# rate, section or reference-point change that breaks the y = 0 symmetry).
# The pitch rate q keeps the symmetry. The node inputs of a symmetric
# surface are the nodes of its defining half, mirrored to the other half,
# so they keep the symmetry too when every surface is symmetric; the control
# inputs keep it when every control is symmetric (see _adjoint_use_symmetry).
_SYMMETRIC_WRT = ("alpha", "V_inf", "rho", "h", "q", "mach")
_SYMMETRIC_NODE_WRT = _SYMMETRIC_WRT + ("nodes",)
# Outputs that keep a symmetric state symmetric (longitudinal only).
_SYMMETRIC_OUTPUTS = ("CL", "CDi", "CDp", "CD", "Cm")


CPU_KERNELS_FLOAT64_NOTE = (
    "The CPU VJP kernels evaluate the O(n^2) partials in float64; the O(n) parts and the adjoint "
    "solve use float32.")


def _cpu_kernels_in_float64(partials: str) -> bool:
    """Return True when *partials* names a CPU VJP kernel (it evaluates each pair in float64).

    Parameters
    ----------
    partials : str
        Path and backend of the partials, ``"<path>:<backend>"``.
    """
    path, _, backend = partials.partition(":")
    return path == "kernels" and backend in ("numba", "cython", "numpy")


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
                          outputs: tuple[str, ...], surfaces_symmetric: bool = False,
                          controls_symmetric: bool = False) -> bool:
    """Return True when the folded symmetric system gives the exact adjoint.

    The fold constrains the circulation to the symmetric subspace, so it is
    exact only when the case is symmetric and every requested input and
    output is symmetric (no beta, phi, roll rate p, yaw rate r, section or
    reference change, no lateral output). The node inputs are symmetric
    when every surface is symmetric (*surfaces_symmetric*): the inputs are
    the nodes of the defining half, and the other half is their mirror
    image. The control inputs are symmetric when every control is symmetric
    (*controls_symmetric*); an antisymmetric control (an aileron) moves the
    two halves in opposite directions. Else the full system is used.
    """
    if not production_symmetric:
        return False
    allowed = _SYMMETRIC_NODE_WRT if surfaces_symmetric else _SYMMETRIC_WRT
    if controls_symmetric:
        allowed = allowed + ("controls",)
    if any(w not in allowed for w in wrt):
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


def _refuse_configuration(ac: Any, condition: Any, settings: Any, canonical: str,
                          wrt: tuple[str, ...] = ()) -> None:
    """Raise GradientScopeError for configurations that the adjoint gradients do not cover.

    The Fourier solver is refused (closed form: central differences are the
    method there). The planform input needs linear airfoils: the planform
    chain interpolates the linear section properties of the sections.
    The section input (``"sections"``: a0, alpha_L0, Cd0 and Cm0 of every
    strip) is refused when a surface has a tabulated airfoil: on the
    strips of a tabulated polar some of these values do not enter the
    result, so their derivatives would be zeros without a meaning. The
    input names all four values of every surface, so one tabulated
    surface refuses it also on an aircraft with other, linear surfaces;
    the other inputs stay available. A nonlinear lifting-line solve that
    did not converge is refused after the solve (see :func:`_prepare_case`).
    The compressibility correction (mach > 0) is supported only for the
    vortex lattice in free air; ground effect and the lifting lines keep
    refusing it. The ``"mach"`` input needs a Mach number in the condition.
    The planform input is refused when a chordwise control of the vortex
    lattice is deflected: the width of the control joins (see
    :class:`ventorum.geometry.controls.ControlJoinField`) follows the
    planform, and the gradient chain keeps it fixed. At zero deflection the
    join does not move a point, so the planform input stays available.
    """
    from ventorum.core.datatypes import TabulatedAirfoil

    mach = getattr(condition, "mach", None)
    mach_on = mach is not None and float(mach) > 0.0
    if "mach" in wrt and mach is None:
        raise GradientScopeError(
            "Input 'mach' needs a Mach number in the flight condition "
            "(FlightCondition mach); the condition has mach=None.")
    if mach_on:
        if canonical != "vlm":
            raise GradientScopeError(
                f"Adjoint gradients are not supported with compressibility correction "
                f"(mach > 0) for solver '{canonical}'. Use solver_type 'vlm'.")
        if getattr(condition, "h", None) is not None:
            raise GradientScopeError(
                "Adjoint gradients are not supported with compressibility correction "
                "(mach > 0) in ground effect.")
    if canonical == "fourier":
        raise GradientScopeError(
            "The Fourier solver has no adjoint gradients (closed form): use central differences "
            "of the production solve, or solver_type 'vlm', 'linear' or 'nonlinear'.")
    lattice_mean_line = canonical == "vlm" and getattr(settings, "n_chord", None) != 1
    shapes_given = lattice_mean_line and any(
        surf.sections and all(sec.shape is not None for sec in surf.sections) for surf in ac.surfaces
    )
    if "sections" in wrt:
        if shapes_given:
            raise GradientScopeError(
                "Input 'sections' includes alpha_L0. A given mean line sets the VLM boundary condition, "
                "so its alpha_L0 is not an independent section input. "
                "Remove 'sections' from wrt.")
        tab = [surf.name for surf in ac.surfaces
               if any(isinstance(sec.airfoil, TabulatedAirfoil) for sec in surf.sections)]
        if tab:
            if canonical == "nonlinear":
                why = ("the nonlinear lifting line takes the lift, the drag and the moment of these "
                       "strips from the tables, so a0, alpha_L0, Cd0 and Cm0 do not enter the result")
            else:
                why = ("the linear solvers take the profile drag and the moment of these strips "
                       "from the tables, so Cd0 and Cm0 do not enter the result, and a0 and "
                       "alpha_L0 are a fit of the table, not inputs of the airfoil")
            raise GradientScopeError(
                f"Input 'sections' is not available for the tabulated airfoils of surface(s) "
                f"{tab}: {why}. Their derivatives would be zeros without a meaning. Remove "
                "'sections' from wrt, use linear airfoils for section gradients, or change the "
                "table data and use differences of the production solve.")
    if "planform" in wrt and lattice_mean_line:
        deflected = [(surf.name, c.name) for surf in ac.surfaces for c in getattr(surf, "controls", [])
                     if c.deflection != 0.0]
        if deflected:
            raise GradientScopeError(
                f"Input 'planform' is not available with the deflected chordwise controls {deflected}. "
                "The join at each side edge of a chordwise control (0.2 local chords) has a width that "
                "changes with the planform, and the gradient chain keeps that width fixed, so the "
                "planform derivative would miss a term. No partial gradient is returned. Use "
                "differences of the production solve, or the planform gradient at zero deflection. "
                "The other inputs (nodes, sections, controls) stay available.")
    if "planform" in wrt:
        for surf in ac.surfaces:
            for sec in surf.sections:
                if isinstance(sec.airfoil, TabulatedAirfoil):
                    raise GradientScopeError(
                        f"Surface {surf.name!r} has a tabulated airfoil: the planform input needs "
                        "linear airfoils (the planform chain interpolates their linear section "
                        "properties). Use wrt='nodes' or 'sections' for tabulated airfoils.")



def _has_camber_offset(surf) -> bool:
    """Return True if a fully shaped surface needs the mean-line offset model.

    As in the lattice: when every section gives a shape, the mean line sets
    the boundary condition and the polar alpha_L0 and Cm0 do not enter. The
    chord-line model equals this model only if every mean line is flat and
    every polar has alpha_L0 = 0 rad and Cm0 = 0 [-].
    """
    from ventorum.core.datatypes import LinearAirfoil

    if not surf.sections or any(sec.shape is None for sec in surf.sections):
        return False
    from ventorum.geometry.lattice import _MeanLine

    for sec in surf.sections:
        if bool(np.any(_MeanLine(sec.shape).z != 0.0)):
            return True
        af = sec.airfoil
        if not isinstance(af, LinearAirfoil) or float(af.alpha_L0) != 0.0 or float(af.Cm0) != 0.0:
            return True
    return False

# Relative tolerance (to the smallest chordwise panel fraction of the strip)
# within which a recovered chordwise fraction is the same node as a
# production fraction. The recovery from the panel lengths has an error of
# a few units of round-off (about 1e-15); two different nodes are at least
# half a panel apart (the hinge rule of the mesh builder).
_XI_NODE_RTOL = 1e-6


def _recover_xi(lattice: Any, aircraft: Any = None) -> np.ndarray:
    """Return the chordwise fractions (n_strips, n_chord + 1) of the lattice.

    The fractions come from the panel lengths of the lattice
    (``panel_length / chord``, summed from the leading edge). This sum adds
    the round-off of each panel; it does not amplify it. Each strip is then
    compared with the fractions of the mesh builder (uniform and cosine
    spacing, with the hinge node of a chordwise control moved to its
    effective hinge fraction, also on the strips of a control join): when
    one of them agrees within
    ``_XI_NODE_RTOL`` of the smallest panel fraction, the exact production
    fractions are used. Without *aircraft* the hinge nodes are not known,
    and a strip with a moved hinge node keeps the summed fractions.

    Parameters
    ----------
    lattice : VortexLattice
        Production lattice.
    aircraft : Aircraft or None
        Aircraft of the lattice (for the hinge nodes of the controls).

    Returns
    -------
    numpy.ndarray
        Chordwise fractions [-] of the panel edges, 0 at the leading edge
        and 1 at the trailing edge.
    """
    from ventorum.geometry.lattice import _chordwise_fractions

    n_s = int(lattice.n_strips)
    n_c = int(lattice.n_chord)
    chord = np.asarray(lattice.chord, dtype=float)
    length = np.asarray(lattice.panel_length, dtype=float).reshape(n_s, n_c)
    d = length / np.where(chord > 0.0, chord, 1.0)[:, None]
    xi = np.zeros((n_s, n_c + 1))
    xi[:, 1:] = np.cumsum(d, axis=1)
    xi[:, -1] = 1.0
    hinge: list[float | None] = [None] * n_s
    if aircraft is not None and str(lattice.collocation) == "vlm" and n_c >= 2:
        for k, v in enumerate(_strip_controls(aircraft, lattice)):
            if v is not None and v[1] is not None:
                hinge[k] = float(v[1])
        # The strips of a join outside its control have the hinge node of
        # that control too (see ventorum.geometry.controls.ControlJoinField).
        field = lattice.geom_cache.get("flap_field")
        if field is not None:
            for k in np.flatnonzero(np.isfinite(field["hinge"])):
                hinge[int(k)] = float(field["hinge"][k])
    bases = [_chordwise_fractions(n_c, "uniform"), _chordwise_fractions(n_c, "cosine")]
    for s in range(n_s):
        for base in bases:
            cand = base.copy()
            if hinge[s] is not None and n_c >= 2:
                kh = 1 + int(np.argmin(np.abs(base[1:n_c] - hinge[s])))
                cand[kh] = hinge[s]
            tol = _XI_NODE_RTOL * float(np.min(np.diff(cand)))
            if tol > 0.0 and float(np.max(np.abs(cand - xi[s]))) <= tol:
                xi[s] = cand
                break
    return xi


def control_names(aircraft: Any) -> list[str]:
    """Return the sorted unique control names of *aircraft* (the keys of the ``"controls"`` input)."""
    return sorted({c.name for s in aircraft.surfaces for c in getattr(s, "controls", [])})


def _strip_controls(aircraft: Any, lattice: Any) -> list[tuple[Any, float | None] | None]:
    """Return per strip the control of the strip and its effective hinge fraction, or None.

    The rule is the rule of the mesh builder
    (:func:`ventorum.geometry.lattice.build_lattice`): a control covers the
    strips whose span fraction of the control point is inside its limits
    (1e-9 tolerance), a later control of the same surface replaces an
    earlier one, and the effective hinge fraction of the chordwise mode
    comes from ``lattice.control_info`` (same order as the controls).
    """
    out: list[tuple[Any, float | None] | None] = [None] * int(lattice.n_strips)
    info = list(getattr(lattice, "control_info", []))
    pos = 0
    eta = np.abs(np.asarray(lattice.eta, dtype=float))
    for sl in lattice.surfaces:
        surf = aircraft.surfaces[int(sl.index)]
        for c in getattr(surf, "controls", []):
            entry = info[pos] if pos < len(info) else None
            pos += 1
            if entry is None or entry["name"] != c.name or entry["surface"] != surf.name:
                raise RuntimeError("The control information of the lattice does not match the aircraft.")
            for k in range(sl.strips.start, sl.strips.stop):
                if c.eta_start - 1e-9 <= eta[k] <= c.eta_end + 1e-9:
                    out[k] = (c, entry["hinge_x_c_eff"])
    return out


def _control_maps(aircraft: Any, lattice: Any, xi: np.ndarray, names: list[str], device: str,
                  dtype: Any, strip_edges: np.ndarray | None = None) -> tuple[dict | None, dict | None]:
    """Return the section-mode and the chordwise-mode control maps of the reference model.

    Section mode (lifting lines, or the vortex lattice with one chordwise
    panel): per control strip the strip index, the control index, the
    signed flap effectiveness ``sign tau`` and the signed moment derivative
    ``sign dCm/ddelta`` (Glauert 1926). Chordwise mode (vortex lattice with
    two or more chordwise panels): the flap strips of the lattice (controls
    and their joins), their effective hinge fractions, the base angle of
    their edge nodes [rad] and its coefficient per control name, the
    neighbour nodes of the hinge axes, and the flap panels (the panels
    whose leading fraction is aft of the effective hinge) with their strips.
    ``aft_edge`` gives per flap strip edge the line id of its aft legs
    (``strip_edges`` gives the node ids, see :func:`_aft_line_ids`).
    """
    import torch

    from ventorum.geometry.controls import flap_effectiveness, flap_moment_derivative

    per_strip = _strip_controls(aircraft, lattice)
    if not any(v is not None for v in per_strip):
        return None, None
    index = {n: i for i, n in enumerate(names)}
    is_right = np.asarray(lattice.strip_is_right, dtype=bool)
    n_chord = int(lattice.n_chord)
    chordwise = str(lattice.collocation) == "vlm" and n_chord >= 2

    def _t(a: list, integer: bool = False) -> Any:
        return torch.as_tensor(np.asarray(a), dtype=torch.int64 if integer else dtype,
                               device=device)

    if not chordwise:
        strips, ctrl, tau_s, dcm_s, delta0 = [], [], [], [], []
        for k, v in enumerate(per_strip):
            if v is None:
                continue
            c, _ = v
            sgn = 1.0 if (is_right[k] or c.symmetric) else -1.0
            strips.append(k)
            ctrl.append(index[c.name])
            tau_s.append(sgn * flap_effectiveness(c.hinge_x_c))
            dcm_s.append(sgn * flap_moment_derivative(c.hinge_x_c))
            delta0.append(float(c.deflection))
        sec = {"strips": _t(strips, True), "ctrl": _t(ctrl, True), "tau_sign": _t(tau_s),
               "dcm_sign": _t(dcm_s),
               "np": {"strips": np.asarray(strips, dtype=np.int64),
                      "tau_sign": np.asarray(tau_s), "dcm_sign": np.asarray(dcm_s),
                      "delta0": np.asarray(delta0)}}
        return sec, None
    field = lattice.geom_cache.get("flap_field")
    if field is None:
        return None, None
    # Chordwise mode: the deflected geometry of the lattice (controls and
    # their joins, see ventorum.geometry.controls.ControlJoinField). Each
    # flap strip has its effective hinge fraction; each of its two edge
    # nodes has the base angle [rad] and its derivative per unit deflection
    # of each control name (the join factor and the side sign).
    strips = np.flatnonzero(np.isfinite(field["hinge"]))
    hinge = np.asarray(field["hinge"], dtype=float)[strips]
    coef = np.zeros((strips.size, 2, len(names)))
    for name, arr in field["coef"].items():
        coef[:, :, index[name]] = np.asarray(arr)[strips]
    panels, pstrip = [], []
    for row, k in enumerate(strips):
        # The mesh builder turns the panels whose leading node is the hinge
        # node or aft of it. The hinge node is the node at the effective
        # hinge fraction; the recovered fraction of that node agrees with
        # it within the round-off, far below the panel size.
        kh = 1 + int(np.argmin(np.abs(xi[k, 1:n_chord] - float(hinge[row]))))
        tol = _XI_NODE_RTOL * float(np.min(np.diff(xi[k])))
        if abs(float(xi[k, kh]) - float(hinge[row])) > tol:
            raise RuntimeError(
                f"Strip {k}: no chordwise node of the lattice is at the effective hinge fraction "
                f"{float(hinge[row]):.15g} (nearest {float(xi[k, kh]):.15g}).")
        for j in range(kh, n_chord):
            panels.append(k * n_chord + j)
            pstrip.append(k)
    aft_edge = _aft_line_ids(field, strips, strip_edges)
    chord = {"strip": _t(list(strips), True), "hinge": _t(list(hinge)), "aft_edge": _t(aft_edge, True),
             "angle0": _t(np.asarray(field["angle"], dtype=float)[strips]),
             "coef": torch.as_tensor(coef, dtype=dtype, device=device),
             "far": _t(_far_nodes(field, strips), True),
             "panels": _t(panels, True), "panel_strip": _t(pstrip, True)}
    return None, chord


def _far_nodes(field: dict, strips: np.ndarray) -> np.ndarray:
    """Return per flap strip edge the (strip, side) of the node beyond it for the hinge axis, shape (n_f, 2, 2).

    An edge without a neighbour node gives its own node (left edge: the
    strip itself, side 0; right edge: side 1), so the axis is one-sided.
    """
    far = np.asarray(field["far"], dtype=np.int64)[strips].copy()
    for side in (0, 1):
        none = far[:, side, 0] < 0
        far[none, side, 0] = strips[none]
        far[none, side, 1] = side
    return far


def _aft_line_ids(field: dict, strips: np.ndarray, strip_edges: np.ndarray) -> np.ndarray:
    """Return the line id of the aft legs at the two edge nodes of each flap strip, shape (n_f, 2).

    The masks of the on-line pairs of the gradient kernels need one id per
    line that holds its points for every input. A node that never turns
    (zero base angle and zero coefficients: the outer end of a join) keeps
    its aft legs on the edge line: the id of that edge line. A node that two
    flap strips with the same hinge fraction turn about the same line gets
    one id for both strips (also across two joined surfaces). Any other node
    gets an id of its own strip side.
    """
    edges = np.asarray(strip_edges, dtype=np.int64)
    n_nodes = int(edges.max()) + 1
    hinge = np.asarray(field["hinge"], dtype=float)
    angle = np.asarray(field["angle"], dtype=float)
    coef = list(field["coef"].values())
    out = np.empty((strips.size, 2), dtype=np.int64)
    partner = np.asarray(field["partner"], dtype=np.int64)
    for row, s in enumerate(strips):
        for side in (0, 1):
            node = int(edges[s, side])
            q, jq = int(partner[s, side, 0]), int(partner[s, side, 1])
            if angle[s, side] == 0.0 and all(c[s, side] == 0.0 for c in coef):
                out[row, side] = node
            elif q < 0:
                out[row, side] = n_nodes + node
            elif hinge[q] == hinge[s]:
                out[row, side] = n_nodes + min(node, int(edges[q, jq]))
            else:
                out[row, side] = 2 * n_nodes + 2 * int(s) + side
    return out


def _airfoil_tables(lattice: Any, unknown_panels: np.ndarray, device: str,
                    dtype: Any) -> tuple[list, Any]:
    """Return the polar tables of the tabulated strips and the mask of the linear strips.

    Each table entry holds the strips of one airfoil object (the strips
    that the production loads group together), the positions of their
    unknowns, and the cubic pieces of the production interpolation of Cl,
    Cd and Cm (Cm None without moment data).
    """
    import torch

    from ventorum.aero.loads import group_strips_by_airfoil
    from ventorum.core.datatypes import TabulatedAirfoil
    from ventorum.sensitivity.reference import pp_from_scipy

    groups = group_strips_by_airfoil(list(lattice.airfoils))
    unknown_strip = np.asarray(lattice.panel_strip, dtype=np.int64)[np.asarray(unknown_panels)]
    lin = np.ones(int(lattice.n_strips))
    tables = []
    for af, idx in groups:
        if not isinstance(af, TabulatedAirfoil):
            continue
        lin[idx] = 0.0
        f = af._pchip()
        pos = np.flatnonzero(np.isin(unknown_strip, idx))
        tables.append({
            "name": af.name,
            "strips": torch.as_tensor(np.asarray(idx, dtype=np.int64), device=device),
            "unknowns": torch.as_tensor(pos.astype(np.int64), device=device),
            "Cl": pp_from_scipy(f["Cl"], device, dtype),
            "Cd": pp_from_scipy(f["Cd"], device, dtype),
            "Cm": pp_from_scipy(f["Cm"], device, dtype) if "Cm" in f else None,
            "table_alpha": np.asarray(af.tables()[0], dtype=float),
        })
    if not tables:
        return [], None
    return tables, torch.as_tensor(lin, dtype=dtype, device=device)


def _camber_matrices(
    lattice: Any,
    xi: np.ndarray,
    chord_map: dict | None,
    device: str,
    dtype: Any,
    planform: tuple | None = None,
) -> dict | None:
    """Build mean-line offsets [chord units] and slopes [-] for the reference.

    Section mean-line values at each chord fraction are constants. With
    *planform*, the spanwise weights follow the section and mesh stations
    of ``(PlanformGeometry, PlanformTensors)`` through the torch chain.
    """
    camber_frames = getattr(lattice, "camber_frames", None)
    if camber_frames is None:
        camber_frames = lattice.geom_cache.get("camber_frames")
    if camber_frames is None:
        return None
    mean_surfaces = lattice.geom_cache.get("mean_surfaces")
    if not getattr(lattice, "has_camber_offset", False):
        return None
    if not any(all(line is not None for line in lines) for _, lines, _ in mean_surfaces):
        return None

    import torch

    n_s = int(lattice.n_strips)
    n_p = int(lattice.n_panels)
    n_chord = int(lattice.n_chord)

    edge_start = [0]
    for sf in lattice.surfaces:
        edge_start.append(edge_start[-1] + int(sf.edge_le.shape[0]))
    E_total = edge_start[-1]

    def _tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=dtype, device=device)

    def _zeros(shape: tuple) -> Any:
        return torch.zeros(shape, dtype=dtype, device=device)

    section_fr, edge_eta = [], []
    strip_eta = _tensor(np.abs(np.asarray(lattice.eta, dtype=float)))
    for si, sf in enumerate(lattice.surfaces):
        if planform is None:
            frame = camber_frames[si]
            section_fr.append(_tensor([sec.y_frac for sec in frame["sections"]]))
            edge_eta.append(_tensor(np.abs(frame["eta"])))
        else:
            geom, pt = planform
            name = sf.name
            tr = geom.surfaces[name]
            section_fr.append(torch.stack([sec["y_frac"] for sec in pt.params[name]["sec"]]))
            eta = pt.nodes[name]["eta"]
            if tr.is_symmetric:
                eta = torch.cat((eta.flip(0), eta[1:]))
            elif tr.mirror_y:
                eta = eta.flip(0)
            edge_eta.append(eta)
            strip_eta[sf.strips] = pt.strips[name]["eta"][tr.full_perm]

    def _mean_at(si: int, eta: Any, xi_val: Any, *, derivative: bool = False) -> Any:
        xi_arr = np.asarray(xi_val, dtype=float)
        lines = camber_frames[si]["lines"]
        values = _tensor(np.array([
            np.zeros_like(xi_arr) if line is None else line.at(xi_arr, derivative=derivative)
            for line in lines
        ]))
        fr = section_fr[si]
        eta = torch.broadcast_to(eta, values.shape[1:]).contiguous()
        j = torch.clamp(torch.searchsorted(fr.detach(), eta.detach(), right=True) - 1,
                        0, len(lines) - 2)
        w = torch.clamp((eta - fr[j]) / (fr[j + 1] - fr[j]), 0.0, 1.0)
        lo = torch.gather(values, 0, j[None])[0]
        hi = torch.gather(values, 0, (j + 1)[None])[0]
        return (1.0 - w) * lo + w * hi

    def _joined_offset_weights(si: int, e: int, xi_val: np.ndarray | float) -> Any:
        xi_arr = np.asarray(xi_val, dtype=float)
        single = xi_arr.ndim == 0
        if single:
            xi_arr = xi_arr[None]
        K = xi_arr.shape[0]
        w = _zeros((K, E_total))
        members = camber_frames[si]["joined"].get(e, [(si, e)])
        scale = 1.0 / len(members)
        for s_m, e_m in members:
            z_m = _mean_at(s_m, edge_eta[s_m][e_m], xi_arr)
            w[:, edge_start[s_m] + e_m] += scale * z_m
        return w[0] if single else w

    W_a = _zeros((n_p, E_total))
    W_b = _zeros((n_p, E_total))
    # Mean-line offsets of the leg nodes between two bound points (the
    # middle chord fraction) and between the last bound point and the
    # trailing edge (geometry.lattice._leg_nodes).
    W_ma = _zeros((n_p, E_total))
    W_mb = _zeros((n_p, E_total))
    W_le_l = _zeros((n_s, E_total))
    W_le_r = _zeros((n_s, E_total))
    W_te_l = _zeros((n_s, E_total))
    W_te_r = _zeros((n_s, E_total))
    z_c = _zeros((n_s, n_chord))
    slope = _zeros((n_s, n_chord))
    strip_camber = np.zeros(n_s, dtype=bool)

    claf = np.clip(np.asarray(lattice.a0, dtype=float) / (2.0 * np.pi), 0.2, 2.0)
    for si, (sf, (_secs, lines, _eta_e)) in enumerate(zip(lattice.surfaces, mean_surfaces)):
        active = all(line is not None for line in lines)
        frame = camber_frames[si]
        if not active and not frame["joined"]:
            continue
        idx = np.arange(sf.strips.start, sf.strips.stop)
        known = np.full(len(idx), active)
        strip_camber[idx] = known

        x0 = xi[idx, :-1]
        dx = np.diff(xi[idx], axis=1)
        xb = x0 + 0.25 * dx
        xc = xb + 0.5 * dx * claf[idx, None]

        z_c[idx] = _mean_at(si, strip_eta[idx, None], xc)
        slope[idx] = _mean_at(si, strip_eta[idx, None], xc, derivative=True)

        for k_idx, s in enumerate(idx):
            e = k_idx
            for j in range(n_chord):
                p = s * n_chord + j
                W_a[p] = _joined_offset_weights(si, e, xb[k_idx, j])
                W_b[p] = _joined_offset_weights(si, e + 1, xb[k_idx, j])
                if n_chord > 1:
                    # The last panel has its node half a panel length aft of
                    # its bound point (geometry.lattice.build_lattice).
                    xm = (0.5 * (xb[k_idx, j] + xb[k_idx, j + 1]) if j < n_chord - 1
                          else xb[k_idx, j] + 0.5 * dx[k_idx, j])
                    W_ma[p] = _joined_offset_weights(si, e, xm)
                    W_mb[p] = _joined_offset_weights(si, e + 1, xm)
            W_le_l[s] = _joined_offset_weights(si, e, 0.0)
            W_le_r[s] = _joined_offset_weights(si, e + 1, 0.0)
            W_te_l[s] = _joined_offset_weights(si, e, 1.0)
            W_te_r[s] = _joined_offset_weights(si, e + 1, 1.0)

    W_hl, W_hr, W_hfar0, W_hfar1 = None, None, None, None
    if chord_map is not None:
        strips = chord_map["strip"].cpu().numpy()
        hinge = chord_map["hinge"].cpu().numpy()
        far = chord_map["far"].cpu().numpy()
        strip_surf = np.asarray(lattice.strip_surface, dtype=np.int64)
        n_f = len(strips)
        W_hl_arr = _zeros((n_f, E_total))
        W_hr_arr = _zeros((n_f, E_total))
        W_hfar0_arr = _zeros((n_f, E_total))
        W_hfar1_arr = _zeros((n_f, E_total))
        for row, s in enumerate(strips):
            h_val = float(hinge[row])
            si_s = int(strip_surf[s])
            e_l = s - lattice.surfaces[si_s].strips.start
            e_r = e_l + 1
            W_hl_arr[row] = _joined_offset_weights(si_s, e_l, h_val)
            W_hr_arr[row] = _joined_offset_weights(si_s, e_r, h_val)

            s_f0, side_f0 = int(far[row, 0, 0]), int(far[row, 0, 1])
            si_f0 = int(strip_surf[s_f0])
            e_f0 = s_f0 - lattice.surfaces[si_f0].strips.start + side_f0
            W_hfar0_arr[row] = _joined_offset_weights(si_f0, e_f0, h_val)

            s_f1, side_f1 = int(far[row, 1, 0]), int(far[row, 1, 1])
            si_f1 = int(strip_surf[s_f1])
            e_f1 = s_f1 - lattice.surfaces[si_f1].strips.start + side_f1
            W_hfar1_arr[row] = _joined_offset_weights(si_f1, e_f1, h_val)

        W_hl = torch.as_tensor(W_hl_arr, dtype=dtype, device=device)
        W_hr = torch.as_tensor(W_hr_arr, dtype=dtype, device=device)
        W_hfar0 = torch.as_tensor(W_hfar0_arr, dtype=dtype, device=device)
        W_hfar1 = torch.as_tensor(W_hfar1_arr, dtype=dtype, device=device)

    camber_panel_mask = np.repeat(strip_camber, n_chord)

    return {
        "W_a": torch.as_tensor(W_a, dtype=dtype, device=device),
        "W_b": torch.as_tensor(W_b, dtype=dtype, device=device),
        "W_ma": torch.as_tensor(W_ma, dtype=dtype, device=device),
        "W_mb": torch.as_tensor(W_mb, dtype=dtype, device=device),
        "W_le_l": torch.as_tensor(W_le_l, dtype=dtype, device=device),
        "W_le_r": torch.as_tensor(W_le_r, dtype=dtype, device=device),
        "W_te_l": torch.as_tensor(W_te_l, dtype=dtype, device=device),
        "W_te_r": torch.as_tensor(W_te_r, dtype=dtype, device=device),
        "W_hl": W_hl,
        "W_hr": W_hr,
        "W_hfar0": W_hfar0,
        "W_hfar1": W_hfar1,
        "z_c": torch.as_tensor(z_c, dtype=dtype, device=device),
        "slope": torch.as_tensor(slope, dtype=dtype, device=device),
        "strip_camber": torch.as_tensor(strip_camber, dtype=torch.bool, device=device),
        "camber_panel_mask": torch.as_tensor(camber_panel_mask, dtype=torch.bool, device=device),
    }


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
    xi = _recover_xi(lattice, aircraft)
    frac = np.asarray(lattice.cp_frac, dtype=float)
    cp_frac_panel = frac[panel_strip]

    # Per-surface defining edges and global strip edge indices.
    surfaces: list[dict] = []
    strip_edges = np.zeros((n_s, 2), dtype=np.int64)
    offset = 0
    for sl in lattice.surfaces:
        full_le = np.asarray(sl.edge_le_chord if sl.edge_le_chord is not None else sl.edge_le, dtype=float)
        full_te = np.asarray(sl.edge_te_chord if sl.edge_te_chord is not None else sl.edge_te, dtype=float)
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

    # Leg-force rule of each chordwise leg (a-legs, then b-legs), as in the
    # production loads (ventorum.aero.loads.leg_force_rules). The rule is a
    # property of the mesh topology; it does not change with the geometry
    # inputs of the gradient.
    from ventorum.aero.loads import LEG_FREE, leg_force_rules

    _, leg_scale, leg_kind = leg_force_rules(lattice)
    leg_keep_normal = (leg_kind != LEG_FREE).astype(float)

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
        "leg_keep_normal": _t("lkn", leg_keep_normal),
        "leg_scale": _t("lsc", np.asarray(leg_scale, dtype=float)),
        # Leg chains of the strips with geometric camber (VortexLattice.leg_next).
        "leg_next": (None if getattr(lattice, "leg_next", None) is None or not np.any(lattice.leg_next)
                     else torch.as_tensor(np.asarray(lattice.leg_next, dtype=bool), device=device)),
        "accum": accum,
        "origin": None,
    }
    # Strip edges of the flap strips whose hinge point is not a leg node
    # (geometry.lattice._leg_nodes), per panel, shape (n_panels, 2).
    off = (lattice.geom_cache.get("flap_field") or {}).get("hinge_off")
    build["hinge_off_panel"] = (None if off is None else
                                torch.as_tensor(np.asarray(off, dtype=bool)[np.asarray(lattice.panel_strip)],
                                                device=device))
    from ventorum.solvers.factory import resolve_solver_type

    build["solver"] = resolve_solver_type(settings.solver_type)
    mach_val = getattr(condition, "mach", None)
    build["mach_active"] = bool(mach_val is not None and float(mach_val) > 0.0)
    rates = {k: float(getattr(condition, k, 0.0)) for k in RATE_INPUTS}
    # The onset with the rotation is used when a rate is not zero; the
    # gradient functions also set it when a rate is a requested input.
    build["rates"] = any(v != 0.0 for v in rates.values())
    names = control_names(aircraft)
    build["control_names"] = names
    sec_map, chord_map = _control_maps(aircraft, lattice, xi, names, device, dtype, strip_edges)
    build["ctrl_section"] = sec_map
    build["ctrl_chord"] = chord_map
    camber_data = _camber_matrices(lattice, xi, chord_map, device, dtype)
    build["camber_data"] = camber_data
    # Folds of the reference surfaces for the blend of the mean-line offset
    # (geometry.lattice.camber_span_field); a property of the mesh.
    frames = getattr(lattice, "camber_frames", None) or lattice.geom_cache.get("camber_frames") or []
    build["camber_folds"] = [f for frame in frames for f in frame.get("folds", [])]
    if camber_data is not None:
        build["strip_camber"] = camber_data["strip_camber"]
    tables, lin_mask = _airfoil_tables(lattice, unknown_panels, device, dtype)
    build["tables"] = tables
    build["lin_strip_mask"] = lin_mask
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
        "p": rates["p"],
        "q": rates["q"],
        "r": rates["r"],
        "mach": None if getattr(condition, "mach", None) is None else float(condition.mach),
        "control_names": list(names),
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
    prep: dict | None = None,
    planform_means_at: tuple | None = None,
) -> dict:
    """Return the torch input tensors; requested inputs require grad.

    With ``"planform"`` in *wrt*, the node, strip-property and automatic
    reference tensors are differentiable functions of the planform parameters
    (see :mod:`ventorum.sensitivity.planform`), and *prep* (the dict of
    :func:`_prepare_case`) is required. *planform_means_at* selects the
    branch of a coincident section break (see
    :func:`ventorum.sensitivity.planform.planform_tensors`); None gives the
    production branch at the base point.
    """
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
    for key in RATE_INPUTS:
        t[key] = _s(key, base.get(key, 0.0))
    if base.get("mach") is None:
        t["mach"] = None
    else:
        t["mach"] = _s("mach", base["mach"])
    for name in base.get("control_names", []):
        # The change of the deflection of every control with this name [rad].
        t[f"controls:{name}"] = _s("controls", 0.0)
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
    if "planform" in wrt:
        if prep is None:
            raise ValueError("The planform input needs the case dict of _prepare_case.")
        _attach_planform(model, t, prep, wrt, device, precision, planform_means_at)
    return t


def _attach_planform(model: TorchReference, t: dict, prep: dict, wrt: tuple[str, ...],
                     device: str, precision: str, means_at: tuple | None = None) -> None:
    """Replace the geometry tensors of *t* by the differentiable planform port.

    The node tensors become the port nodes plus the fixed node displacements,
    the strip tensors become the interpolated section properties, the automatic
    reference values become functions of the planform, and the control-point
    fractions follow the spanwise stations. The planform leaves are stored in
    *t* under ``"planform:..."`` keys. Intermediate input tensors whose own
    jacobian is requested (strips for ``"sections"``, automatic references for
    ``"reference"``) are registered for the extra gradients of
    :func:`_adjoint_solve`. *means_at* selects the branch of a coincident
    section break (None: the production branch at the base point).
    """
    import torch

    from ventorum.sensitivity import planform as PF

    dtype = _torch_dtype(precision)
    dev = torch.device(device)
    ac = prep["aircraft"]
    canonical = prep["canonical"]
    collocation = "vlm" if canonical == "vlm" else "llt"
    geom = PF.build_planform(ac, prep["settings"], collocation)
    pt = PF.planform_tensors(geom, device, dtype, differentiable=True, means_at=means_at)
    base_surf = {s["name"]: s for s in model.b["surfaces"]}
    nsec = {}
    for name in geom.order:
        tr = geom.surfaces[name]
        nsec[name] = tr.n_sections
        p = pt.params[name]
        t[f"planform:{name}:semi_span"] = p["semi_span"]
        t[f"planform:{name}:sweep_le"] = p["sweep_le"]
        t[f"planform:{name}:dihedral"] = p["dihedral"]
        t[f"planform:{name}:incidence"] = p["incidence"]
        t[f"planform:{name}:position"] = p["position"]
        for j, d in enumerate(p["sec"]):
            for key in ("chord", "twist", "y_frac", "alpha_L0", "x_le", "z_le"):
                if d[key] is not None:
                    t[f"planform:{name}:sec{j}:{key}"] = d[key]
        base_le = np.asarray(base_surf[name]["base_le"], dtype=float)
        base_te = np.asarray(base_surf[name]["base_te"], dtype=float)
        nn = pt.nodes[name]

        def _atm(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=dev)

        n_e = int(base_le.shape[0])
        if nn["le"].shape[0] != n_e:
            raise RuntimeError(
                f"[{name}] the planform port gives {nn['le'].shape[0]} edges, "
                f"the lattice has {n_e}.")
        d_le = _atm(base_le - tr.undef["le"])
        d_te = _atm(base_te - tr.undef["te"])
        t[f"nodes:{name}:le"] = nn["le"] + d_le
        t[f"nodes:{name}:te"] = nn["te"] + d_te
    per = {}
    for prop in ("a0", "alpha_L0", "Cd0", "Cm0"):
        per[prop] = PF.full_strip_tensor(geom, {n: pt.strips[n][prop] for n in geom.order})
        if per[prop].shape != t[prop].shape:
            raise RuntimeError(
                f"The planform port gives {per[prop].shape[0]} strips, "
                f"the lattice has {t[prop].shape[0]}.")
        t[prop] = per[prop]
    cs = model.b.get("ctrl_section")
    if cs is not None:
        # The port interpolates the sections without the deflection: add the
        # thin-airfoil change of the base deflection of the control strips.
        npm = cs["np"]
        d_al0 = np.zeros(t["alpha_L0"].shape[0])
        d_cm0 = np.zeros(t["Cm0"].shape[0])
        np.add.at(d_al0, npm["strips"], -npm["tau_sign"] * npm["delta0"])
        np.add.at(d_cm0, npm["strips"], npm["dcm_sign"] * npm["delta0"])
        t["alpha_L0"] = t["alpha_L0"] + torch.as_tensor(d_al0, dtype=dtype, device=dev)
        t["Cm0"] = t["Cm0"] + torch.as_tensor(d_cm0, dtype=dtype, device=dev)
        per["alpha_L0"] = t["alpha_L0"]
        per["Cm0"] = t["Cm0"]
    auto = set(getattr(ac, "_auto_ref", set())) & {"S_ref", "b_ref", "c_ref"}
    for key in ("S_ref", "b_ref", "c_ref"):
        if key in auto:
            t[key] = pt.refs[key]
    frac_full = PF.full_frac_tensor(geom, {n: pt.strips[n]["frac"] for n in geom.order})
    if frac_full.shape != model.b["frac"].shape:
        raise RuntimeError("The planform port gives a fraction array of a wrong shape.")
    model.b["frac"] = frac_full
    model.b["cp_frac_panel"] = frac_full[model.b["panel_strip"]]
    if model.b.get("camber_data") is not None:
        t["camber_data"] = _camber_matrices(
            prep["lattice"], model.b["xi"].detach().cpu().numpy(), model.b.get("ctrl_chord"),
            device, dtype, planform=(geom, pt),
        )
    extra = []
    if "sections" in wrt:
        extra += [t[prop] for prop in ("a0", "alpha_L0", "Cd0", "Cm0")]
    if "reference" in wrt:
        extra += [t[key] for key in ("S_ref", "b_ref", "c_ref") if key in auto]
    model.b["_planform"] = {"nsec": nsec, "extra": extra,
                            "coincident": list(geom.coincident), "notes": list(geom.notes)}


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
        ``"planform"`` maps each surface name to its parameters:
        ``"semi_span"``, ``"sweep_le"``, ``"dihedral"``, ``"incidence"``,
        ``"position"`` (array of 3) and ``"sections"`` (a list per section
        with ``"chord"``, ``"twist"``, ``"x_le"``, ``"z_le"``,
        ``"y_frac"`` and ``"alpha_L0"``).
    device : str
        Torch device of the gradient computation (``"cpu"`` or ``"cuda"``).
    precision_primal : str
        Precision of the production solve (``"float64"`` on the CPU; the
        GPU precision setting on the GPU).
    precision_gradient : str
        Precision of the gradient partials and the adjoint solve.
    method : str
        Always ``"adjoint"``.
    partials : str
        Path and backend of the partial derivatives: ``"kernels:<backend>"``
        (compiled VJP kernels: ``numba``, ``cython``, ``numpy`` or ``warp``)
        or ``"reference:torch"`` (the PyTorch reference copy).
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
    partials: str = "reference:torch"
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
            "partials": self.partials,
            "forward_mismatch": float(self.forward_mismatch),
            "n_solves": int(self.n_solves),
            "n_adjoint": int(self.n_adjoint),
            "symmetric": bool(self.symmetric),
            "notes": list(self.notes),
        }


def _prepare_case(geometry: Any, condition: Any, settings: Any, axes: str,
                  wrt: tuple[str, ...], outputs: tuple[str, ...], method: str = "auto") -> dict:
    """Run the production solve and build the reference model.

    Returns a dict with the aircraft, lattice, result, model, base values,
    device, precision, flags and the path of the partial derivatives
    (*method*, see :func:`ventorum.sensitivity.kernels.resolve`). Raises
    ValueError for refused input. The adjoint uses the folded symmetric
    system only when the case is symmetric and every requested input and
    output is symmetric.
    """
    from ventorum.core.datatypes import Aircraft, LiftingSurface
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import (
        validate_aircraft,
        validate_flight_condition,
        validate_solver_settings,
    )

    from ventorum.sensitivity.kernels import METHODS, resolve

    if "controls" in wrt and not control_names(geometry if not isinstance(geometry, LiftingSurface)
                                               else Aircraft(name="SingleWing", surfaces=[geometry])):
        raise ValueError('Input "controls" needs an aircraft with one or more control surfaces.')
    if "nodes" in wrt and "planform" in wrt:
        raise ValueError(
            'Inputs "nodes" and "planform" describe the same freedom: use "planform" '
            "for design parameters (the chain rule gives the node effect), or "
            '"nodes" for free node moves.')
    if axes not in ("body", "stability", "wind"):
        raise ValueError(f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}.")
    if method not in METHODS:
        raise ValueError(f"Unknown gradient method {method!r}; use one of {list(METHODS)}.")
    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    st = canonical_settings(settings.clone() if settings is not None else _SolverSettings())
    validate_flight_condition(condition)
    validate_solver_settings(st)
    canonical = resolve_solver_type(st.solver_type)
    _refuse_configuration(ac, condition, st, canonical, wrt)
    for v in (condition.alpha, condition.beta, condition.V_inf, condition.rho,
              getattr(condition, "phi", 0.0), getattr(condition, "p", 0.0),
              getattr(condition, "q", 0.0), getattr(condition, "r", 0.0)):
        if not math.isfinite(float(v)):
            raise ValueError(f"Flight condition has a non-finite value {v!r}.")
    if condition.h is not None and not math.isfinite(float(condition.h)):
        raise ValueError(f"Flight condition has a non-finite height {condition.h!r}.")
    validate_aircraft(ac)
    solver = make_solver(canonical)
    result = solver.solve(ac, condition, st)
    if canonical == "nonlinear" and not bool(result.converged):
        last = result.residual_history[-1] if result.residual_history else float("nan")
        raise GradientScopeError(
            f"The nonlinear lifting-line solve did not converge in {result.iterations} Newton "
            f"iterations (section Cl error {last:.2e}, tolerance {float(st.tolerance):.1e}). The adjoint "
            "needs the converged state: the gradient of a state that does not solve the equations "
            "is not defined. Change the case (for example a smaller angle of attack) or the "
            "settings (max_iterations).")
    lattice = result.details["lattice"]
    production_symmetric = bool(result.symmetry_used)
    surfaces_symmetric = all(bool(sl.is_symmetric) for sl in lattice.surfaces)
    controls_symmetric = all(bool(c.symmetric) for s in ac.surfaces for c in getattr(s, "controls", []))
    symmetric = _adjoint_use_symmetry(production_symmetric, wrt, outputs, surfaces_symmetric,
                                      controls_symmetric)
    device = _gradient_device(result)
    precision = _gradient_precision()
    precision_primal = str(result.details.get("precision", "float64"))
    model, base = build_reference(ac, condition, st, lattice, axes, device, precision, symmetric)
    if any(w in RATE_INPUTS for w in wrt):
        model.b["rates"] = True
    path, backend = resolve(method, device, int(lattice.n_panels))
    model.b["vjp_backend"] = backend if path == "kernels" else None
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
        "g_unknown": g_unknown, "axes": axes, "partials": f"{path}:{backend}",
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

    With ``"planform"`` in *wrt* and a section break that lies exactly on a
    mesh edge (a coincident break), any nonzero change of the span fraction
    of that section moves the edge and changes the neighbour control
    stations to the edge means. One more adjoint pass per coincident section
    then runs on that branch (its own solved state, no forward check against
    the base-point production result), and gives the y_frac gradient of
    that section.
    """
    values, jacobian, worst, n_adj = _adjoint_pass(prep, wrt, make_targets, vectors)
    if "planform" in wrt:
        model: TorchReference = prep["model"]
        for surf_name, sec in list(model.b.get("_planform", {}).get("coincident", [])):
            _, jac_m, _, n_m = _adjoint_pass(prep, wrt, make_targets, (),
                                             means_at=(surf_name, int(sec)))
            n_adj += n_m
            for name in jacobian:
                jacobian[name]["planform"][surf_name]["sections"][int(sec)]["y_frac"] = (
                    jac_m[name]["planform"][surf_name]["sections"][int(sec)]["y_frac"])
    return values, jacobian, worst, n_adj


def _adjoint_pass(
    prep: dict,
    wrt: tuple[str, ...],
    make_targets: Any,
    vectors: tuple[str, ...] = (),
    means_at: tuple | None = None,
) -> tuple[dict, dict, float, int]:
    """Run one adjoint pass (see :func:`_adjoint_solve`).

    With *means_at* None, the state is the production circulation and the
    forward check runs. With *means_at* a coincident section, the planform
    port takes the branch of a nonzero change of that span fraction, the
    state is the solution of that linear system, and no forward check runs
    (the production result at the base point is on the other branch).
    """
    import torch

    model: TorchReference = prep["model"]
    base = prep["base"]
    device, precision = prep["device"], prep["precision"]
    t = make_input_tensors(model, base, wrt, device, precision, prep,
                           planform_means_at=means_at)
    g0 = torch.as_tensor(np.ascontiguousarray(prep["g_unknown"]),
                         dtype=model.b["dtype"], device=model.b["device"])
    prod = _production_values(prep)
    if means_at is not None:
        with torch.no_grad():
            # Newton iterations from the production state: one step solves
            # the linear systems; the nonlinear lifting line iterates.
            n_it = 1 if model.b.get("solver") != "nonlinear" else _BRANCH_NEWTON_STEPS
            for _ in range(n_it):
                _, A0, rhs0 = model.forward(t, g0)
                g_new = torch.linalg.solve(A0, rhs0)
                step = float(torch.max(torch.abs(g_new - g0)))
                g0 = g_new
                if step <= 1e-14 * max(float(torch.max(torch.abs(g0))), 1e-300):
                    break
    g_var = g0.detach().clone().requires_grad_(True)
    outputs, A, rhs = model.forward(t, g_var)
    targets = make_targets(outputs)
    A_det = A.detach()
    g_star = g0.detach()
    worst = 0.0 if means_at is not None else _forward_check(prep, outputs, prod, vectors,
                                                            A_det, rhs.detach(), g_star)
    # One backward pass per target gives df/dG (the adjoint right-hand
    # side) and the explicit partials df/dx together.
    names = [name for name, _ in targets]
    leaves = [v for v in t.values()
              if isinstance(v, torch.Tensor) and v.is_leaf and v.requires_grad]
    extra = list(model.b.get("_planform", {}).get("extra", []))
    grads_g = []
    grads_fx = []
    grads_ex = []
    for _, f in targets:
        # An output without a dependence on the circulation or the inputs
        # (for example CDp with respect to alpha only) has no graph.
        if f.requires_grad:
            parts = torch.autograd.grad(f, [g_var] + leaves + extra, retain_graph=True,
                                        allow_unused=True)
        else:
            parts = (None,) * (1 + len(leaves) + len(extra))
        gg = parts[0] if parts[0] is not None else torch.zeros_like(g_var)
        grads_g.append(gg.detach())
        grads_fx.append(parts[1:1 + len(leaves)])
        grads_ex.append(parts[1 + len(leaves):])
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
            grads_r = torch.autograd.grad(r_term, leaves + extra, retain_graph=True,
                                          allow_unused=True)
        else:
            grads_r = (None,) * (len(leaves) + len(extra))
        gmap = {}
        for v, gf, gr in zip(leaves, grads_fx[j], grads_r[:len(leaves)]):
            if gf is None and gr is None:
                gmap[id(v)] = None
                continue
            tot = (gf if gf is not None else 0.0) - (gr if gr is not None else 0.0)
            gmap[id(v)] = tot.detach().cpu().numpy()
        for v, gf, gr in zip(extra, grads_ex[j], grads_r[len(leaves):]):
            if gf is None and gr is None:
                gmap[id(v)] = None
                continue
            tot = (gf if gf is not None else 0.0) - (gr if gr is not None else 0.0)
            gmap[id(v)] = tot.detach().cpu().numpy()
        jacobian[name] = _pack_jacobian(model, t, gmap, wrt, base, bool(prep["symmetric"]))
    return values, jacobian, worst, len(targets)


# Newton steps of the nonlinear lifting line on the edge-mean branch of a
# coincident section break (see _adjoint_pass); the iteration stops earlier
# when the step is at the round-off of the circulation.
_BRANCH_NEWTON_STEPS = 30


# Scalar outputs that are components of one force vector and of one
# moment vector (relative scale of the forward check).
_FORCE_VECTOR = ("CL", "CY")
_MOMENT_VECTOR = ("Cl", "Cm", "Cn")


# Limits of the componentwise backward error of the production state in the
# system of the reference copy (see _residual_check). Measured round-off
# (CPU and GPU; vortex lattice up to 2560 unknowns, linear lifting line up
# to 1000 unknowns, nonlinear lifting line up to 600 unknowns): float64
# at most 3.6e-9 (linear lifting line, 1000 unknowns; the vortex lattice
# at most 2e-15), float32 at most 2.8e-3 (linear lifting line, 1000
# unknowns; the vortex lattice at most 2.5e-4). An angle error of 0.01 deg
# in the boundary condition gave 2.8e-5 or more in the same cases.
_RESIDUAL_TOL_FLOAT64 = 1e-6
_RESIDUAL_TOL_FLOAT32 = 3e-2
# The nonlinear lifting line stops at its section Cl tolerance: its limit is
# also at least this factor times the tolerance of the settings.
_RESIDUAL_TOL_NONLINEAR_FACTOR = 100.0


def _residual_check(prep: dict, A: Any, rhs: Any, g: Any) -> float:
    """Check that the production circulation solves the system of the reference copy.

    The check computes the componentwise backward error (Oettli and
    Prager 1964) of the production state *g* in the system ``A g = rhs``
    of the reference copy::

        omega = max_i |A g - rhs|_i / (|A| |g| + |rhs|)_i

    The output comparison of :func:`_forward_check` cannot see a wrong
    term of the boundary condition (a normal, a section alpha_L0, the
    onset of a body rate) when the outputs do not depend strongly on it.
    This check sees it: the production state then does not solve the
    equations of the reference copy. The componentwise form has the scale
    of the round-off also when the lifting line has large vortex-core
    terms in A (the plain ratio ``|A g - rhs| / |rhs|`` does not: it gave
    2.5 in float32 at 1000 unknowns).

    Parameters
    ----------
    prep : dict
        Case of :func:`_prepare_case`.
    A : torch.Tensor
        System matrix (the Newton Jacobian for the nonlinear lifting line).
    rhs : torch.Tensor
        Right-hand side of the system.
    g : torch.Tensor
        Production circulation per unknown [m^2/s].

    Returns
    -------
    float
        The backward error omega [-].

    Raises
    ------
    RuntimeError
        When omega is above ``_RESIDUAL_TOL_FLOAT64`` (1e-6; the primal and
        the gradient precision are both float64) or ``_RESIDUAL_TOL_FLOAT32``
        (3e-2; one of them is float32). For the nonlinear lifting line the
        limit is at least 100 times the solver tolerance.
    """
    import torch

    both64 = prep["precision"] == "float64" and prep["precision_primal"] == "float64"
    tol = _RESIDUAL_TOL_FLOAT64 if both64 else _RESIDUAL_TOL_FLOAT32
    if prep.get("canonical") == "nonlinear":
        tol = max(tol, _RESIDUAL_TOL_NONLINEAR_FACTOR * float(prep["settings"].tolerance))
    with torch.no_grad():
        r = torch.abs(A @ g - rhs)
        denom = torch.abs(A) @ torch.abs(g) + torch.abs(rhs)
        tiny = torch.finfo(r.dtype).tiny
        ratio = torch.where(denom > tiny, r / torch.clamp(denom, min=tiny),
                            torch.where(r > 0.0, torch.full_like(r, float("inf")), torch.zeros_like(r)))
        omega = float(torch.max(ratio)) if ratio.numel() else 0.0
    if not math.isfinite(omega) or omega > tol:
        raise RuntimeError(
            f"The production circulation does not solve the system of the reference copy: "
            f"the backward error is {omega:.3g} (limit {tol:g}). A term of the boundary "
            "condition of the reference copy differs from the production solve. "
            "The gradient would not be the gradient of the reported result.")
    return omega


def _forward_check(prep: dict, outputs: dict, prod: dict[str, float],
                   vectors: tuple[str, ...], A: Any = None, rhs: Any = None,
                   g: Any = None) -> float:
    """Compare the differentiated model with the production result.

    With *A*, *rhs* and *g* given, :func:`_residual_check` also checks that
    the production circulation solves the system of the reference copy.

    Returns the largest relative difference (``forward_mismatch``). Raises
    RuntimeError with the output names when it is above 1e-10 (the primal
    and the gradient precision are both float64) or 1e-4 (one of them is
    float32).

    The difference of CDi, CDp and CD is relative to the production value
    of each. CL and CY are components of one force vector, and Cl, Cm and
    Cn are components of one moment vector: their differences are relative
    to the largest production magnitude of the components of the same
    vector (the normwise relative error of the vector). The round-off of a
    vector component is relative to the vector, so a small component (CY
    in a small sideslip, or CY, Cl and Cn of a symmetric state, which are
    zero by symmetry) has no relative scale of its own. The moment scale
    is not smaller than the force scale, so a moment vector that is zero
    as a whole (all its components at the round-off) does not give a
    false mismatch. An output with an exact zero production value (for
    example CDp with Cd0 = 0) uses the force scale.

    A requested vector output (``strip_force`` [N], ``section_cl``) is
    compared with its production array: the largest difference relative
    to the largest production magnitude of the array, because single
    components go through zero along the span.
    """
    precision = prep["precision"]
    both64 = precision == "float64" and prep["precision_primal"] == "float64"
    tol = 1e-10 if both64 else 1e-4
    force_scale = max(abs(prod[k]) for k in _FORCE_VECTOR)
    # The force scale is the lower limit of the moment scale: a moment is a
    # sum of forces times arms, so its round-off is at most of the order of
    # the round-off of the forces. A state whose whole moment vector is zero
    # by geometry (reference point on the quarter-chord line of a straight
    # wing) has moments at the round-off only, with no relative scale.
    moment_scale = max(max(abs(prod[k]) for k in _MOMENT_VECTOR), force_scale)
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
    if A is not None:
        prep["bc_backward_error"] = _residual_check(prep, A, rhs, g)
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


# Distance [rad] within which a section angle is on a table point of a
# tabulated polar (the result notes name these strips).
_TABLE_POINT_TOL = 1e-12

LLT_NODE_NOTE = (
    "Lifting-line node gradients: the vortex core acts on the distance to the finite filament, so "
    "the bound segments of the strips on one line do not change the node sensitivity; the node "
    "gradients of the linear lifting line agree with an exact-law reference within 2 % in the tests. It keeps "
    "the limits of its model (one panel per strip; not grid convergent with sweep): for the "
    "chordwise shape use the vortex lattice.")


def _unknown_strips(b: dict, unknowns: Any) -> list[int]:
    """Return the sorted strips of the panels of the unknowns *unknowns*.

    With the folded symmetric system one unknown holds a right-half panel
    and its mirror panel, so the strips of both halves are returned. The
    strip numbers are the strip numbers of the full lattice.
    """
    col = np.asarray(b["panel_column_full"].cpu())
    strip = np.asarray(b["panel_strip"].cpu())
    sel = np.isin(col, np.asarray(unknowns, dtype=np.int64))
    return sorted({int(k) for k in strip[sel]})


def _table_notes(model: TorchReference) -> list[str]:
    """Return the notes on the strips of tabulated polars at a table point or outside the table."""
    b = model.b
    tables = b.get("tables") or []
    if not tables:
        return []
    at_point: set[int] = set()
    outside: set[int] = set()
    pairs = []
    rec_s = b.get("_table_alpha") or []
    for tab, a in zip(tables, rec_s):
        pairs.append((tab, np.asarray(tab["strips"].cpu()), np.asarray(a.cpu(), dtype=float)))
    rec_u = b.get("_table_alpha_unknowns") or []
    for tab, a in zip(tables, rec_u):
        pos = np.asarray(tab["unknowns"].cpu())
        # One unknown can hold the strips of both halves (folded system).
        pairs.append((tab, [_unknown_strips(b, [u]) for u in pos], np.asarray(a.cpu(), dtype=float)))
    for tab, strips, a in pairs:
        ta = tab["table_alpha"]
        for s, v in zip(strips, a):
            group = s if isinstance(s, list) else [int(s)]
            if v < ta[0] or v > ta[-1]:
                outside.update(group)
            elif float(np.min(np.abs(ta - v))) <= _TABLE_POINT_TOL * max(1.0, abs(v)):
                at_point.update(group)
    notes = []
    if at_point:
        notes.append(
            "Tabulated polars: the section angle of strip(s) "
            f"{sorted(at_point)} is on a table point; the gradient uses the one-sided derivative "
            "of the interpolation (the interval to the right of the point).")
    if outside:
        notes.append(
            "Tabulated polars: the section angle of strip(s) "
            f"{sorted(outside)} is outside the table; the polar keeps its end value there, so its "
            "derivative with the angle is zero.")
    return notes


def _nonlinear_notes(model: TorchReference) -> list[str]:
    """Return the notes of the nonlinear lifting line and warn past the maximum lift."""
    import warnings

    b = model.b
    if b.get("solver") != "nonlinear":
        return []
    notes = ["Nonlinear lifting line: the adjoint uses the Newton Jacobian dR/dG of the section "
             "lift residual at the converged state (Phillips and Snyder 2000)."]
    slope = b.get("_nl_slope")
    if slope is not None:
        dcl = np.asarray(slope.cpu(), dtype=float)
        # A section angle outside a table has a zero slope (the polar keeps
        # its end value): the outside-table note of _table_notes covers it,
        # and it is not a state past the maximum lift.
        outside = np.zeros(dcl.shape, dtype=bool)
        for tab, a in zip(b.get("tables") or [], b.get("_table_alpha_unknowns") or []):
            pos = np.asarray(tab["unknowns"].cpu())
            v = np.asarray(a.cpu(), dtype=float)
            ta = tab["table_alpha"]
            outside[pos] |= (v < ta[0]) | (v > ta[-1])
        bad = _unknown_strips(b, np.flatnonzero((dcl <= 0.0) & ~outside))
        if bad:
            msg = (f"Past the maximum lift: the section lift slope is zero or negative on strip(s) {bad}. "
                   "The gradient is local, and the solution may not be unique.")
            notes.append(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    res = b.get("_nl_residual")
    if res is not None:
        worst = float(np.max(np.abs(res.cpu().numpy())))
        notes.append(f"Section Cl residual of the converged state: {worst:.2e}.")
    return notes


def _case_notes(prep: dict, ins: tuple[str, ...]) -> list[str]:
    """Return the notes of the configuration: controls, nonlinear lifting line, tables, node caveat."""
    model: TorchReference = prep["model"]
    notes: list[str] = []
    if "controls" in ins:
        if model.b.get("ctrl_chord") is not None:
            notes.append(
                "Controls (chordwise mode): the derivative per unit deflection [1/rad] turns the "
                "points and normals of the panels aft of the hinge about the hinge line, including "
                "the trailing edges and the node chain. All controls with one "
                "name move together; the left half of an antisymmetric control moves the opposite way.")
        else:
            notes.append(
                "Controls (section mode): the derivative per unit deflection [1/rad] of the "
                "thin-airfoil deflected section (alpha_L0 - tau delta, Cm0 + dCm/ddelta delta; "
                "Glauert 1926). All controls with one name move together; the left half of an "
                "antisymmetric control moves the opposite way.")
    notes.extend(_nonlinear_notes(model))
    notes.extend(_table_notes(model))
    if prep["canonical"] in ("linear", "nonlinear") and ("nodes" in ins or "planform" in ins):
        notes.append(LLT_NODE_NOTE)
    return notes


def _rate_notes(prep: dict, jacobian: dict, outs: tuple[str, ...], ins: tuple[str, ...]) -> list[str]:
    """Return the nondimensional rate derivatives of the outputs as notes.

    The nondimensional rates are p b/(2V), q c/(2V) and r b/(2V), with the
    reference values of the case (as :func:`ventorum.rate_derivatives`).
    """
    rates = [w for w in RATE_INPUTS if w in ins]
    if not rates:
        return []
    base = prep["base"]
    V = float(base["V_inf"])
    to_hat = {"p": 2.0 * V / float(base["b_ref"]), "q": 2.0 * V / float(base["c_ref"]),
              "r": 2.0 * V / float(base["b_ref"])}
    parts = []
    for o in outs:
        for w in rates:
            parts.append(f"{o}_{w} = {float(jacobian[o][w]) * to_hat[w]:.6g}")
    return [
        "Body rates: quasi-steady rotation about the moment reference point; the jacobian is per "
        "rad/s. Nondimensional derivatives per unit p b/(2V), q c/(2V), r b/(2V) "
        f"(b_ref {float(base['b_ref']):.6g} m, c_ref {float(base['c_ref']):.6g} m; rates in body "
        f"axes, moments in the '{prep['axes']}' axes): " + ", ".join(parts) + "."]


def _pack_jacobian(model: TorchReference, t: dict, gmap: dict, wrt: tuple[str, ...],
                   base: dict, symmetric: bool = False) -> dict:
    """Pack the flat autograd gradients into the jacobian structure.

    With the folded system (*symmetric*), the y-component of the root node
    of each symmetric surface is set to zero. A y-change of the root node
    is the only node change that is not mirrored (the root lies on y = 0),
    so the fold does not cover it. Its exact derivative is zero: the
    geometry with the root at y = +e is the mirror image of the geometry
    with the root at y = -e, and a symmetric output has the same value for
    both, so it is an even function of e. The full system gives this zero
    to round-off.
    """
    j: dict = {}
    if "nodes" in wrt:
        nodes: dict = {}
        for s in model.b["surfaces"]:
            gle = gmap.get(id(t[f"nodes:{s['name']}:le"]))
            gte = gmap.get(id(t[f"nodes:{s['name']}:te"]))
            n_e = int(s["n_edges"])
            nodes[s["name"]] = {
                "le": np.zeros((n_e, 3)) if gle is None else np.asarray(gle, dtype=float).copy(),
                "te": np.zeros((n_e, 3)) if gte is None else np.asarray(gte, dtype=float).copy(),
            }
            if symmetric and s["kind"] == "symmetric":
                nodes[s["name"]]["le"][0, 1] = 0.0
                nodes[s["name"]]["te"][0, 1] = 0.0
        j["nodes"] = nodes
    for key in SCALAR_INPUTS:
        if key in wrt:
            if key == "h" and base["h"] is None:
                continue
            gr = gmap.get(id(t[key])) if t.get(key) is not None else None
            j[key] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
    if "controls" in wrt:
        ctrl = {}
        for name in base.get("control_names", []):
            gr = gmap.get(id(t[f"controls:{name}"]))
            ctrl[name] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
        j["controls"] = ctrl
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
    if "planform" in wrt:
        nsec = model.b.get("_planform", {}).get("nsec", {})
        pf: dict = {}
        for name, n_s in nsec.items():
            entry: dict = {}
            for key in ("semi_span", "sweep_le", "dihedral", "incidence"):
                gr = gmap.get(id(t[f"planform:{name}:{key}"]))
                entry[key] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
            gr = gmap.get(id(t[f"planform:{name}:position"]))
            entry["position"] = (np.zeros(3) if gr is None
                                 else np.asarray(gr, dtype=float).reshape(3).copy())
            secs = []
            for jj in range(int(n_s)):
                d: dict = {}
                for key in ("chord", "twist", "y_frac", "alpha_L0"):
                    tkey = f"planform:{name}:sec{jj}:{key}"
                    gr = gmap.get(id(t[tkey])) if tkey in t else None
                    d[key] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
                for key in ("x_le", "z_le"):
                    tkey = f"planform:{name}:sec{jj}:{key}"
                    if tkey not in t:
                        d[key] = None
                    else:
                        gr = gmap.get(id(t[tkey]))
                        d[key] = 0.0 if gr is None else float(np.asarray(gr, dtype=float))
                secs.append(d)
            entry["sections"] = secs
            pf[name] = entry
        j["planform"] = pf
    return j


def gradients(
    geometry: Any,
    condition: Any,
    settings: Any = None,
    *,
    outputs: tuple[str, ...] = ("CL", "CD", "Cm"),
    wrt: tuple[str, ...] = ("alpha", "nodes"),
    axes: str = "body",
    method: str = "auto",
) -> GradientResult:
    """Compute exact gradients of scalar solver outputs by the discrete adjoint.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry. It is not changed. The mesh topology is fixed.
    condition : FlightCondition
        Flight condition. Angles in rad, height in m.
    settings : SolverSettings or None
        Solver settings: ``"vlm"``, ``"linear"`` or ``"nonlinear"``, with
        linear or tabulated airfoils, control deflections and body rates.
        The Fourier solver is refused (``GradientScopeError``), and so is a
        nonlinear solve that did not converge.
    outputs : tuple of str
        Scalar outputs from ``"CL", "CY", "CDi", "CDp", "CD", "Cl", "Cm", "Cn"``.
        Vector outputs go through :func:`gradient_vjp`.
    wrt : tuple of str
        Inputs from ``"nodes", "alpha", "beta", "V_inf", "rho", "h", "phi",
        "sections", "reference", "planform", "controls", "p", "q", "r", "mach"``.
        Derivatives with node inputs are taken at a fixed shared-hinge topology.
        Lengths in m, angles in rad, rates in rad/s, ``y_frac``
        dimensionless, Mach number in [-] per unit Mach. ``"planform"`` gives per surface the semi-span [m],
        sweep_le, dihedral and incidence [rad], position [m] and per-section
        lists (chord [m], twist [rad], x_le [m] or None, z_le [m] or None,
        y_frac [-], alpha_L0 [rad]); it cannot be combined with ``"nodes"``
        and needs linear airfoils. ``"controls"`` gives per control name the
        derivative per unit deflection [1/rad] (all controls with one name
        move together). ``"p"``, ``"q"``, ``"r"`` are the body rates
        [rad/s] (flight-dynamics body axes); the notes give the
        nondimensional derivatives. ``"mach"`` gives the derivative per unit
        Mach number [-]; it needs mach set in the condition (at mach 0 the
        derivative is 0, at mach None it is refused).
    axes : str
        Axes of the moments: ``"body"``, ``"stability"`` or ``"wind"``.
    method : str
        Path of the partial derivatives: ``"auto"`` (the ``grad_vjp``
        choice of the machine profile; without a profile the compiled
        kernels: Numba on the CPU, NVIDIA Warp on the GPU), ``"kernels"``
        (the compiled vector-Jacobian-product kernels) or ``"reference"``
        (the PyTorch reference copy). All paths give the same gradients to
        round-off.

    Returns
    -------
    GradientResult
        Values, jacobian, device, precisions, method, path of the partials,
        forward mismatch, solve counts, symmetry flag and notes.

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
    prep = _prepare_case(geometry, condition, settings, axes, ins, outs, method)

    def _make(outputs: dict) -> list:
        return [(o, outputs[o]) for o in outs]

    values, jacobian, worst, n_adj = _adjoint_solve(prep, ins, _make)
    notes = [
        "Discrete adjoint of the production solve (Martins and Hwang 2013; Martins and Ning 2021, ch. 6).",
        "The mesh topology is fixed (spanwise stations, chordwise count and fractions).",
    ]
    if "nodes" in ins:
        notes.append("Derivatives with node inputs are taken at a fixed shared-hinge topology.")
    if "planform" in ins:
        notes.append(
            "The planform chain runs through a differentiable port of the geometry builder: "
            "the spanwise stations move with the parameters, their count does not change. "
            "Root and tip span fractions are pinned (0 and 1); symmetric surfaces must "
            "keep position y at 0.")
        if prep["production_symmetric"]:
            notes.append("Planform inputs use the full system, also for symmetric cases.")
        notes.extend(prep["model"].b.get("_planform", {}).get("notes", []))
    notes.extend(_case_notes(prep, ins))
    notes.extend(_rate_notes(prep, jacobian, outs, ins))
    if bool(prep["model"].b.get("mach_active", False)):
        notes.append(
            "Compressibility correction (mach > 0): the Prandtl-Glauert stretch "
            "about the moment reference point is in the differentiable reference; "
            "the chain rule includes the change of the stretched geometry with "
            "alpha, beta and mach. The Trefftz plane is unchanged.")
    if prep["precision"] == "float32":
        notes.append(
            "Float32 gradients: the gradient gate of float32 is 1e-3 relative; check the result with "
            "check_gradients (rtol=1e-3).")
        if _cpu_kernels_in_float64(prep["partials"]):
            notes.append(CPU_KERNELS_FLOAT64_NOTE)
    return GradientResult(
        values=values, jacobian=jacobian, device=prep["device"],
        precision_primal=prep["precision_primal"], precision_gradient=prep["precision"],
        partials=prep["partials"], forward_mismatch=worst, n_adjoint=n_adj,
        symmetric=prep["symmetric"], notes=notes)


def gradient_vjp(
    geometry: Any,
    condition: Any,
    settings: Any = None,
    *,
    seeds: dict,
    wrt: tuple[str, ...] = ("alpha", "nodes"),
    axes: str = "body",
    method: str = "auto",
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
    method : str
        Path of the partial derivatives, as in :func:`gradients`.

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
    prep = _prepare_case(geometry, condition, settings, axes, ins, tuple(seeds), method)
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
    # The warning past the maximum lift of the nonlinear lifting line.
    _nonlinear_notes(model)
    return jacobian["__vjp__"]

