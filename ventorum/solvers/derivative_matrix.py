# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Full aerodynamic derivative matrix with axis labels and units.

:func:`derivative_matrix` computes the sensitivity of the six force and
moment coefficients (CL, CD, CY, Cl, Cm, Cn) to static variables (alpha,
beta, and in ground effect phi and h/c), control deflections (per control
surface name), and body rotation rates (p_hat, q_hat, r_hat).

References
----------
* B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control,
  3rd ed., Wiley, 1996.
* B. L. Stevens, F. L. Lewis and E. N. Johnson, Aircraft Control and
  Simulation, 3rd ed., Wiley, 2016.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ventorum.core.axes import moment_axes_matrix, transform_moments
from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverResult, SolverSettings
from ventorum.core.trust import near_far_lateral_warnings
from ventorum.solvers.control_derivatives import control_derivatives
from ventorum.solvers.height_derivatives import height_derivatives
from ventorum.solvers.rate_derivatives import rate_derivatives
from ventorum.solvers.static_derivatives import static_derivatives

__all__ = ["DerivativeMatrixResult", "DerivativeMatrixAllDict", "derivative_matrix"]

_ROWS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
_VALID_GROUPS = ("static", "control", "rate", "height")
_VALID_AXES = ("body", "stability", "wind", "all")


def _json_float(value: float | None) -> float | None:
    """Return ``float(value)``, or None when the value is None or not finite."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class DerivativeMatrixResult:
    """Result of :func:`derivative_matrix` for a single axis system.

    Attributes
    ----------
    matrix : numpy.ndarray
        The (6, n) matrix of derivatives.
    rows : tuple of str
        Fixed row labels: ``("CL", "CD", "CY", "Cl", "Cm", "Cn")``.
    columns : tuple of str
        Column labels in group order.
    units : tuple of str
        Units per column: ``"1/rad"`` for angles and deflections,
        ``"-"`` for h/c and nondimensional body rates.
    methods : tuple of str
        Finite-difference scheme and step per column.
    axes : str
        Moment axis system: ``"body"``, ``"stability"`` or ``"wind"``.
    drag_basis : str
        ``"CD_total"`` if sections have profile drag, else ``"CDi"``.
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray
        Moment reference point [m], shape (3,).
    device : str
        Device of the solves (``"cpu"``, ``"gpu"`` or ``"mixed"``).
    precision : str
        Precision of the solves (``"float64"``, ``"float32"`` or ``"mixed"``).
    diagnostics : dict[str, Any]
        Diagnostic quantities: static margin and neutral point, far-field
        diagnostics per control with lateral trust warnings, and Irodov
        margin in ground effect.
    notes : list of str
        Notes on the methods and conventions.
    base : SolverResult or None
        Result of the base flight condition.
    """

    matrix: np.ndarray
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    units: tuple[str, ...]
    methods: tuple[str, ...]
    axes: str
    drag_basis: str
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    device: str
    precision: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    base: SolverResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary of the result."""
        matrix_list = []
        for row in self.matrix:
            matrix_list.append([_json_float(v) for v in row])
        from ventorum.utils.jsonsafe import json_safe

        return {
            "matrix": matrix_list,
            "rows": list(self.rows),
            "columns": list(self.columns),
            "units": list(self.units),
            "methods": list(self.methods),
            "axes": self.axes,
            "drag_basis": self.drag_basis,
            "b_ref": _json_float(self.b_ref),
            "c_ref": _json_float(self.c_ref),
            "ref_point": [_json_float(v) for v in self.ref_point],
            "device": self.device,
            "precision": self.precision,
            "diagnostics": json_safe(self.diagnostics),
            "notes": list(self.notes),
        }

    def as_table(self) -> str:
        """Return a Markdown table of the matrix with row and column labels and axes in the caption."""
        header = ["Coefficient"] + list(self.columns)
        sep = ["---"] * len(header)
        lines = [
            f"Table: Derivative matrix in {self.axes} axes.",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for i, row_name in enumerate(self.rows):
            vals = [
                f"{self.matrix[i, j]:.6g}" if math.isfinite(self.matrix[i, j]) else "nan"
                for j in range(len(self.columns))
            ]
            lines.append("| " + " | ".join([row_name] + vals) + " |")
        return "\n".join(lines)


class DerivativeMatrixAllDict(dict):
    """Dictionary holding the derivative matrices for body, stability and wind axes.

    Parameters
    ----------
    matrices : dict of str to numpy.ndarray
        Derivative matrices keyed by axis system: ``"body"``, ``"stability"``, ``"wind"``.
    results : dict of str to DerivativeMatrixResult
        Full result objects for each axis system.
    rows : tuple of str
        Fixed row labels: ``("CL", "CD", "CY", "Cl", "Cm", "Cn")``.
    columns : tuple of str
        Column labels in group order.
    units : tuple of str
        Units per column: ``"1/rad"`` for angles and deflections, ``"-"`` for h/c and rates.
    methods : tuple of str
        Finite-difference scheme and step per column.
    drag_basis : str
        ``"CD_total"`` if sections have profile drag, else ``"CDi"``.
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray
        Moment reference point [m], shape (3,).
    device : str
        Device of the solves (``"cpu"``, ``"gpu"`` or ``"mixed"``).
    precision : str
        Precision of the solves (``"float64"``, ``"float32"`` or ``"mixed"``).
    diagnostics : dict of str to Any
        Diagnostic quantities.
    notes : list of str
        Notes on the methods and conventions.
    base : SolverResult or None, optional
        Result of the base flight condition.
    """

    def __init__(
        self,
        matrices: dict[str, np.ndarray],
        *,
        results: dict[str, DerivativeMatrixResult],
        rows: tuple[str, ...],
        columns: tuple[str, ...],
        units: tuple[str, ...],
        methods: tuple[str, ...],
        drag_basis: str,
        b_ref: float,
        c_ref: float,
        ref_point: np.ndarray,
        device: str,
        precision: str,
        diagnostics: dict[str, Any],
        notes: list[str],
        base: SolverResult | None = None,
    ) -> None:
        super().__init__(matrices)
        self.results = results
        self.rows = rows
        self.columns = columns
        self.units = units
        self.methods = methods
        self.axes = "all"
        self.drag_basis = drag_basis
        self.b_ref = b_ref
        self.c_ref = c_ref
        self.ref_point = ref_point
        self.device = device
        self.precision = precision
        self.diagnostics = diagnostics
        self.notes = notes
        self.base = base

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        from ventorum.utils.jsonsafe import json_safe

        out: dict[str, Any] = {
            "matrices": {
                name: [[_json_float(v) for v in row] for row in mat]
                for name, mat in self.items()
            },
            "rows": list(self.rows),
            "columns": list(self.columns),
            "units": list(self.units),
            "methods": list(self.methods),
            "axes": "all",
            "drag_basis": self.drag_basis,
            "b_ref": _json_float(self.b_ref),
            "c_ref": _json_float(self.c_ref),
            "ref_point": [_json_float(v) for v in self.ref_point],
            "device": self.device,
            "precision": self.precision,
            "diagnostics": json_safe(self.diagnostics),
            "notes": list(self.notes),
        }
        return out

    def as_table(self) -> str:
        """Return Markdown tables for body, stability and wind axes."""
        parts = []
        for name in ("body", "stability", "wind"):
            res = self.results[name]
            parts.append(res.as_table())
        return "\n\n".join(parts)


def derivative_matrix(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    groups: tuple[str, ...] | list[str] = ("static", "control", "rate"),
    controls: list[str] | None = None,
    axes: str = "body",
) -> DerivativeMatrixResult | DerivativeMatrixAllDict:
    """Compute the aerodynamic derivative matrix of one flight state.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Case geometry. A single surface is wrapped into an aircraft.
    condition : FlightCondition
        Base flight condition.
    settings : SolverSettings or None
        Solver settings (default: :class:`SolverSettings`).
    groups : sequence of str
        Derivative groups to include. Default ``("static", "control", "rate")``.
        In ground effect (``condition.h`` set), ``"height"`` is added by default.
    controls : list of str or None
        Control surface names to differentiate. None selects all controls
        of the aircraft.
    axes : str
        Moment axes: ``"body"`` (default), ``"stability"``, ``"wind"``,
        or ``"all"`` (returns three matrices in a dict).

    Returns
    -------
    DerivativeMatrixResult or DerivativeMatrixAllDict
        The derivative matrix, row and column labels, units, methods,
        diagnostics, and solver metadata.

    Raises
    ------
    ValueError
        For an unknown group, an unknown control name, an unknown axis system,
        or requesting ``"height"`` in free air.
    """
    from ventorum.geometry import lattice_cache
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import validate_aircraft, validate_flight_condition, validate_solver_settings

    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    settings = SolverSettings() if settings is None else settings
    validate_flight_condition(condition)
    settings = canonical_settings(settings)
    validate_solver_settings(settings)
    canonical = resolve_solver_type(settings.solver_type)
    if canonical == "fourier":
        raise ValueError("The Fourier solver has no derivative matrix. Use solver_type 'vlm', 'linear' or 'nonlinear'.")

    if axes not in _VALID_AXES:
        raise ValueError(f"'axes' must be one of {list(_VALID_AXES)}, got {axes!r}.")

    requested_groups = list(groups)
    for g in requested_groups:
        if g not in _VALID_GROUPS:
            raise ValueError(f"Unknown group {g!r}; allowed groups: {list(_VALID_GROUPS)}.")

    in_ground = condition.h is not None
    if not in_ground and "height" in requested_groups:
        raise ValueError("Requesting 'height' in free air is not allowed (condition.h is not set).")

    # In ground effect, 'height' is added by default when default groups were used
    effective_groups = list(requested_groups)
    if in_ground and "height" not in effective_groups and set(requested_groups) == {"static", "control", "rate"}:
        effective_groups.append("height")

    known_controls = list(ac.control_names())
    ctrl_names: list[str] = []
    if "control" in effective_groups:
        if controls is not None:
            for n in controls:
                if n not in known_controls:
                    raise ValueError(f"Unknown control surface {n!r}; known control names: {known_controls}.")
            ctrl_names = list(controls)
        else:
            ctrl_names = list(known_controls)

    validate_aircraft(ac)
    solver = make_solver(canonical)
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=float)
    b_ref, c_ref = float(ac.b_ref), float(ac.c_ref)

    # Cost rule: resolve n_chord once so that all groups share the exact same mesh.
    if settings.n_chord is None and solver.collocation == "vlm":
        settings = settings.clone()
        if in_ground:
            # Match the resolved mesh of height_derivatives
            from ventorum.aero.system import ground_plane_from_condition
            from ventorum.geometry.lattice import build_lattice
            chord_spacing = getattr(settings, "chord_spacing", "uniform")
            key = lattice_cache.lattice_key(ac, settings, "vlm", 1, chord_spacing, fingerprint=fp)
            probe = lattice_cache.get_or_build(key, lambda: build_lattice(
                ac, settings, collocation="vlm", n_chord=1, chord_spacing=chord_spacing))
            h_min = float(np.min(ground_plane_from_condition(probe, condition, rp).height(probe.all_points())))
            d_h = 0.05 * h_min
            d_alpha = math.radians(0.5)
            d_beta = math.radians(1.0)
            d_phi = math.radians(0.5)
            conds_probe = [
                condition,
                dataclasses.replace(condition, alpha=float(condition.alpha) + d_alpha),
                dataclasses.replace(condition, alpha=float(condition.alpha) - d_alpha),
                dataclasses.replace(condition, beta=float(condition.beta) + d_beta),
                dataclasses.replace(condition, beta=float(condition.beta) - d_beta),
                dataclasses.replace(condition, phi=float(condition.phi) + d_phi),
                dataclasses.replace(condition, phi=float(condition.phi) - d_phi),
                dataclasses.replace(condition, h=float(condition.h) + d_h),
                dataclasses.replace(condition, h=float(condition.h) - d_h),
            ]
            settings.n_chord = max(solver.resolve_n_chord(ac, settings, c, None, rp, fingerprint=fp) for c in conds_probe)
        else:
            settings.n_chord = solver.resolve_n_chord(ac, settings, condition, None, rp, fingerprint=fp)

    # Solve the constituent groups in body axes
    body_columns: list[np.ndarray] = []
    col_names: list[str] = []
    col_units: list[str] = []
    col_methods: list[str] = []
    diagnostics: dict[str, Any] = {}
    notes: list[str] = []
    devices: set[str] = set()
    precisions: set[str] = set()
    base_result: SolverResult | None = None
    drag_basis = "CDi"

    # 1. Static and height groups
    if in_ground:
        has_static = "static" in effective_groups
        has_height = "height" in effective_groups
        if has_static or has_height:
            h_res = height_derivatives(ac, condition, settings, axes="body")
            devices.add(h_res.device)
            precisions.add(h_res.precision)
            base_result = h_res.base
            drag_basis = h_res.drag_basis
            notes.extend(h_res.notes)
            diagnostics["irodov_margin"] = h_res.irodov.get("margin")
            diagnostics["irodov"] = dict(h_res.irodov)
            cl_a = float(h_res.matrix[0, 0])
            cm_a = float(h_res.matrix[4, 0])
            if abs(cl_a) > 1e-9:
                x_np = float(rp[0] - c_ref * cm_a / cl_a)
                diagnostics["neutral_point_x"] = x_np
                diagnostics["static_margin"] = float((x_np - rp[0]) / c_ref)
            else:
                diagnostics["neutral_point_x"] = None
                diagnostics["static_margin"] = None

            if has_static:
                # alpha
                body_columns.append(h_res.matrix[:, 0])
                col_names.append("alpha")
                col_units.append("1/rad")
                col_methods.append(f"central, {h_res.steps['alpha_deg']:g} deg")
                # beta
                body_columns.append(h_res.matrix[:, 1])
                col_names.append("beta")
                col_units.append("1/rad")
                col_methods.append(f"central, {h_res.steps['beta_deg']:g} deg")

            if has_height:
                # phi
                body_columns.append(h_res.matrix[:, 2])
                col_names.append("phi")
                col_units.append("1/rad")
                col_methods.append(f"central, {h_res.steps['phi_deg']:g} deg")
                # h_over_c
                body_columns.append(h_res.matrix[:, 3])
                col_names.append("h_over_c")
                col_units.append("-")
                col_methods.append(f"central, {h_res.steps['h_m'] / c_ref:.4g}")
    else:
        if "static" in effective_groups:
            s_res = static_derivatives(ac, condition, settings, axes="body")
            devices.add(s_res.device)
            precisions.add(s_res.precision)
            base_result = s_res.base
            drag_basis = s_res.drag_basis
            notes.extend(s_res.notes)
            diagnostics["static_margin"] = s_res.static_margin
            diagnostics["neutral_point_x"] = s_res.neutral_point_x

            # alpha
            body_columns.append(s_res.matrix[:, 0])
            col_names.append("alpha")
            col_units.append("1/rad")
            col_methods.append(f"central, {s_res.steps['alpha_deg']:g} deg")
            # beta
            body_columns.append(s_res.matrix[:, 1])
            col_names.append("beta")
            col_units.append("1/rad")
            col_methods.append(f"central, {s_res.steps['beta_deg']:g} deg")

    # 2. Control group
    if "control" in effective_groups and ctrl_names:
        c_res = control_derivatives(ac, condition, settings, controls=ctrl_names, axes="body", hinge_alpha=False)
        devices.add(c_res.device)
        precisions.add(c_res.precision)
        if base_result is None:
            base_result = c_res.base
        drag_basis = c_res.drag_basis
        notes.extend(c_res.notes)

        ctrl_diags: dict[str, Any] = {}
        for name in ctrl_names:
            cd = c_res.derivatives[name]
            col_vec = np.array([cd.CL, cd.CD, cd.CY, cd.Cl, cd.Cm, cd.Cn], dtype=float)
            body_columns.append(col_vec)
            col_names.append(f"delta:{name}")
            col_units.append("1/rad")
            col_methods.append(f"{cd.scheme}, {math.degrees(cd.step):g} deg")

            # Near/far lateral diagnostics
            cn_stab = transform_moments(
                cd.Cl, cd.Cm, cd.Cn,
                float(condition.alpha), float(condition.beta),
                to="stability", from_axes="body",
                b_ref=b_ref, c_ref=c_ref,
            )["Cn"]
            warns = near_far_lateral_warnings(
                CY=cd.CY,
                CY_farfield=cd.CY_farfield,
                Cn=cn_stab,
                Cn_trefftz=cd.Cn_trefftz,
            )
            ctrl_diag = {
                "CY_farfield": _json_float(cd.CY_farfield),
                "Cn_trefftz": _json_float(cd.Cn_trefftz),
                "warnings": list(warns),
            }
            ctrl_diags[name] = ctrl_diag
            diagnostics[name] = ctrl_diag
        diagnostics["controls"] = ctrl_diags

    # 3. Rate group
    if "rate" in effective_groups:
        r_res = rate_derivatives(ac, condition, settings, axes="body")
        devices.add(r_res.device)
        precisions.add(r_res.precision)
        if base_result is None:
            base_result = r_res.base
        drag_basis = r_res.drag_basis
        notes.extend(r_res.notes)

        for j, rate_name in enumerate(("p_hat", "q_hat", "r_hat")):
            body_columns.append(r_res.matrix[:, j])
            col_names.append(rate_name)
            col_units.append("-")
            col_methods.append(f"central, {r_res.step:g}")

    n_cols = len(body_columns)
    if n_cols > 0:
        matrix_body = np.column_stack(body_columns)
    else:
        matrix_body = np.empty((6, 0), dtype=float)

    dev_str = devices.pop() if len(devices) == 1 else "mixed"
    prec_str = precisions.pop() if len(precisions) == 1 else "mixed"

    # Moment transformation matrices
    rot_stab = moment_axes_matrix(
        float(condition.alpha), float(condition.beta),
        to="stability", from_axes="body",
        b_ref=b_ref, c_ref=c_ref,
    )
    rot_wind = moment_axes_matrix(
        float(condition.alpha), float(condition.beta),
        to="wind", from_axes="body",
        b_ref=b_ref, c_ref=c_ref,
    )

    def _matrix_in(ax_name: str) -> np.ndarray:
        if ax_name == "body" or n_cols == 0:
            return matrix_body.copy()
        rot = rot_stab if ax_name == "stability" else rot_wind
        m = matrix_body.copy()
        m[3:] = rot @ matrix_body[3:]
        return m

    if axes == "all":
        matrices = {name: _matrix_in(name) for name in ("body", "stability", "wind")}
        results = {
            name: DerivativeMatrixResult(
                matrix=matrices[name],
                rows=_ROWS,
                columns=tuple(col_names),
                units=tuple(col_units),
                methods=tuple(col_methods),
                axes=name,
                drag_basis=drag_basis,
                b_ref=b_ref,
                c_ref=c_ref,
                ref_point=rp,
                device=dev_str,
                precision=prec_str,
                diagnostics=dict(diagnostics),
                notes=list(dict.fromkeys(notes)),
                base=base_result,
            )
            for name in ("body", "stability", "wind")
        }
        return DerivativeMatrixAllDict(
            matrices,
            results=results,
            rows=_ROWS,
            columns=tuple(col_names),
            units=tuple(col_units),
            methods=tuple(col_methods),
            drag_basis=drag_basis,
            b_ref=b_ref,
            c_ref=c_ref,
            ref_point=rp,
            device=dev_str,
            precision=prec_str,
            diagnostics=diagnostics,
            notes=list(dict.fromkeys(notes)),
            base=base_result,
        )

    mat_out = _matrix_in(axes)
    return DerivativeMatrixResult(
        matrix=mat_out,
        rows=_ROWS,
        columns=tuple(col_names),
        units=tuple(col_units),
        methods=tuple(col_methods),
        axes=axes,
        drag_basis=drag_basis,
        b_ref=b_ref,
        c_ref=c_ref,
        ref_point=rp,
        device=dev_str,
        precision=prec_str,
        diagnostics=diagnostics,
        notes=list(dict.fromkeys(notes)),
        base=base_result,
    )
