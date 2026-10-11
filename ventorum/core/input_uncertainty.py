# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Compute input-uncertainty error bars (layer 2) by linear propagation.

First-order (linear) propagation of uncertainty, the law of propagation
of uncertainty of the GUM (JCGM 100:2008, section 5.1, equation 10 for
independent inputs and equation 13 with correlations): ``u_c(f)^2 = J U J^T``
with ``J`` the gradient of the coefficient with respect to the inputs and
``U`` the covariance matrix of the inputs. The expanded half-width is
``k u_c`` with the coverage factor ``k`` (default 2, GUM section 6).
The combination of the numerical layer and the input layer by root sum
of squares of independent contributions follows Oberkampf and Roy (2010).

The gradients come from ``ventorum.gradients`` (discrete adjoint) when
the configuration is in its scope. Outside its scope the flow-input
gradients come from central differences of the production solve; section
inputs are then refused. The device and the precision of the production
solve are used; no CPU or float64 forcing applies.

References
----------
JCGM 100:2008 (GUM), Evaluation of measurement data: Guide to the
expression of uncertainty in measurement, Joint Committee for Guides in
Metrology, 2008, sections 5.1 and 6.
W. L. Oberkampf and C. J. Roy, Verification and Validation in
Scientific Computing, Cambridge University Press, 2010.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverSettings,
)
from ventorum.core.error_bars import (
    COEFFICIENT_NAMES,
    INPUT_LAYER_NOTE,
    ErrorBar,
    ErrorBarResult,
    LayerContribution,
)

#: Flow inputs with an uncertainty, in output order of the shares.
FLOW_INPUTS = ("alpha", "beta", "phi", "V_inf", "rho", "h")

#: Section properties with an uncertainty.
SECTION_PROPS = ("a0", "alpha_L0", "Cd0", "Cm0")

#: Central-difference steps of the fallback outside the adjoint scope.
#: Angles in rad, V_inf and rho relative, h as a fraction of the smallest
#: ground clearance (as in height_derivatives).
_FALLBACK_STEPS = {
    "alpha": math.radians(0.5),
    "beta": math.radians(1.0),
    "phi": math.radians(0.5),
    "V_inf": 0.01,
    "rho": 0.01,
    "h": 0.05,
}


def _valid_uncertainty_keys() -> list[str]:
    """Return the valid top-level keys of the uncertainties dict."""
    return list(FLOW_INPUTS) + ["sections"]


def _check_sigma(name: str, value: Any, unit: str) -> float:
    """Check one standard uncertainty and return it as a float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Uncertainty {name!r} must be a finite number [{unit}], got {value!r}."
        ) from None
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"Uncertainty {name!r} must be finite and never negative [{unit}], got {value!r}."
        )
    return number


def _parse_uncertainties(
    uncertainties: dict,
    correlations: dict | None,
    condition: FlightCondition,
    surface_names: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    """Check the input dicts and return (flow, sections, correlations).

    *flow* maps each uncertain flow input to its standard uncertainty
    [unit of the input]. *sections* maps each surface name to a dict of
    property to standard uncertainty [unit of the property]. The
    correlation keys are ordered pairs of flow-input names.
    """
    if not isinstance(uncertainties, dict):
        raise ValueError("uncertainties must be a dict with optional keys "
                         f"{_valid_uncertainty_keys()}.")
    units = {"alpha": "rad", "beta": "rad", "phi": "rad",
             "V_inf": "m/s", "rho": "kg/m^3", "h": "m"}
    section_units = {"a0": "1/rad", "alpha_L0": "rad", "Cd0": "-", "Cm0": "-"}
    for key in uncertainties:
        if key not in _valid_uncertainty_keys():
            raise ValueError(
                f"Unknown uncertainty key {key!r}. "
                f"Use one of {_valid_uncertainty_keys()}."
            )
    flow: dict[str, float] = {}
    for key in FLOW_INPUTS:
        if key in uncertainties and uncertainties[key] is not None:
            flow[key] = _check_sigma(key, uncertainties[key], units[key])
    if "h" in flow and condition.h is None:
        raise ValueError(
            "Uncertainty 'h' needs ground effect: set condition.h [m], "
            "the height of the moment reference point above the ground."
        )
    sections: dict[str, dict[str, float]] = {}
    raw_sections = uncertainties.get("sections", None)
    if raw_sections is not None:
        if not isinstance(raw_sections, dict):
            raise ValueError("'sections' must be a dict of surface name to dict.")
        for surf, props in raw_sections.items():
            if surf not in surface_names:
                raise ValueError(
                    f"Unknown surface {surf!r} in 'sections'. "
                    f"Known surfaces: {surface_names}."
                )
            if not isinstance(props, dict):
                raise ValueError(
                    f"Sections of surface {surf!r} must be a dict with optional keys "
                    f"{list(SECTION_PROPS)}."
                )
            for prop in props:
                if prop not in SECTION_PROPS:
                    raise ValueError(
                        f"Unknown section property {prop!r} of surface {surf!r}. "
                        f"Use one of {list(SECTION_PROPS)}."
                    )
            parsed = {p: _check_sigma(f"sections.{surf}.{p}", props[p], section_units[p])
                      for p in SECTION_PROPS if p in props and props[p] is not None}
            if parsed:
                sections[surf] = parsed
    corr: dict[tuple[str, str], float] = {}
    if correlations is not None:
        if not isinstance(correlations, dict):
            raise ValueError("correlations must be a dict of (input, input) pairs to numbers in [-1, 1].")
        for key, value in correlations.items():
            pair = tuple(key) if isinstance(key, (tuple, list)) else None
            if pair is None or len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(
                    f"Correlation key {key!r} must be a pair of two different flow inputs "
                    f"from {list(FLOW_INPUTS)}."
                )
            for name in pair:
                if name not in FLOW_INPUTS:
                    raise ValueError(
                        f"Unknown correlation input {name!r} in {key!r}. "
                        f"Use flow inputs from {list(FLOW_INPUTS)}."
                    )
            try:
                rho = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Correlation {key!r} must be a finite number in [-1, 1], got {value!r}."
                ) from None
            if not math.isfinite(rho) or rho < -1.0 or rho > 1.0:
                raise ValueError(
                    f"Correlation {key!r} must be in [-1, 1], got {value!r}."
                )
            corr[(pair[0], pair[1])] = rho
    if not flow and not sections:
        raise ValueError(
            "uncertainties has no uncertain input: give at least one of "
            f"{_valid_uncertainty_keys()}."
        )
    return flow, sections, corr


def _production_coeffs(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    axes: str,
) -> tuple[dict[str, float], Any]:
    """Run one production solve and return the six coefficients and the result."""
    from ventorum import analyze

    res = analyze(aircraft.clone(), condition=condition.clone(), settings=settings.clone())
    moments = res.moments(axes)
    coeffs = {
        "CL": float(res.totals.CL),
        "CD": float(res.totals.CD_total if res.totals.CD_total is not None else res.totals.CDi),
        "CY": float(res.totals.CY),
        "Cl": float(moments["Cl"]),
        "Cm": float(moments["Cm"]),
        "Cn": float(moments["Cn"]),
    }
    return coeffs, res


def _surface_strip_ranges(lattice: Any, surface_names: list[str]) -> dict[str, range]:
    """Return the strip index range of each surface name in the lattice."""
    ranges: dict[str, range] = {}
    for surf in lattice.surfaces:
        ranges[str(surf.name)] = range(int(surf.strips.start), int(surf.strips.stop))
    missing = [name for name in surface_names if name not in ranges]
    if missing:
        raise ValueError(f"Lattice has no strips of surfaces {missing}.")
    return ranges


def _adjoint_gradients(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    axes: str,
    flow: dict[str, float],
    sections: dict[str, dict[str, float]],
    lattice: Any,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (values, jacobian) from the adjoint gradients.

    *jacobian* maps each coefficient to a dict of input name to
    derivative [unit of the coefficient per unit of the input]. Section
    inputs use one name ``"sections.<surface>.<prop>"`` with the sum of
    the strip gradients of the surface (fully correlated along the span).
    """
    from ventorum import gradients

    wrt = [key for key in FLOW_INPUTS if key in flow]
    if sections:
        wrt.append("sections")
    res = gradients(
        aircraft.clone(), condition.clone(), settings.clone(),
        outputs=tuple(COEFFICIENT_NAMES), wrt=tuple(wrt), axes=axes,
    )
    values = {name: float(res.values[name]) for name in COEFFICIENT_NAMES}
    ranges = _surface_strip_ranges(lattice, list(sections)) if sections else {}
    jacobian: dict[str, dict[str, float]] = {}
    for name in COEFFICIENT_NAMES:
        jac = res.jacobian[name]
        row: dict[str, float] = {}
        for key in FLOW_INPUTS:
            if key in flow:
                row[key] = float(jac.get(key, 0.0))
        for surf, props in sections.items():
            strip = ranges[surf]
            idx = list(strip)
            for prop in props:
                arr = np.asarray(jac["sections"][prop], dtype=float).reshape(-1)
                row[f"sections.{surf}.{prop}"] = float(np.sum(arr[idx]))
        jacobian[name] = row
    return values, jacobian


def _smallest_clearance(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
) -> float:
    """Return the smallest clearance of the geometry above the ground [m]."""
    from ventorum.aero.system import ground_plane_from_condition
    from ventorum.geometry import lattice_cache
    from ventorum.geometry.lattice import build_lattice

    ac = aircraft.clone()
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=float)
    chord_spacing = getattr(settings, "chord_spacing", "uniform")
    key = lattice_cache.lattice_key(ac, settings, "vlm", 1, chord_spacing, fingerprint=fp)
    probe = lattice_cache.get_or_build(key, lambda: build_lattice(
        ac, settings, collocation="vlm", n_chord=1, chord_spacing=chord_spacing))
    plane = ground_plane_from_condition(probe, condition, rp)
    assert plane is not None
    return float(np.min(plane.height(probe.all_points())))


def _central_gradients(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    axes: str,
    flow: dict[str, float],
    base_coeffs: dict[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (values, jacobian) from central differences of the production solve.

    Steps: alpha 0.5 deg, beta 1 deg, phi 0.5 deg, V_inf 1 %, rho 1 %,
    h 5 % of the smallest ground clearance.
    """
    h_min = _smallest_clearance(aircraft, condition, settings) if "h" in flow else 0.0
    steps: dict[str, float] = {}
    for key in flow:
        if key == "alpha":
            steps[key] = _FALLBACK_STEPS["alpha"]
        elif key == "beta":
            steps[key] = _FALLBACK_STEPS["beta"]
        elif key == "phi":
            steps[key] = _FALLBACK_STEPS["phi"]
        elif key == "V_inf":
            steps[key] = _FALLBACK_STEPS["V_inf"] * abs(float(condition.V_inf))
        elif key == "rho":
            steps[key] = _FALLBACK_STEPS["rho"] * abs(float(condition.rho))
        elif key == "h":
            steps[key] = _FALLBACK_STEPS["h"] * h_min
    jacobian: dict[str, dict[str, float]] = {name: {} for name in COEFFICIENT_NAMES}
    for key, step in steps.items():
        base_val = float(getattr(condition, key))
        plus = dataclasses.replace(condition, **{key: base_val + step})
        minus = dataclasses.replace(condition, **{key: base_val - step})
        c_plus, _ = _production_coeffs(aircraft, plus, settings, axes)
        c_minus, _ = _production_coeffs(aircraft, minus, settings, axes)
        for name in COEFFICIENT_NAMES:
            jacobian[name][key] = float((c_plus[name] - c_minus[name]) / (2.0 * step))
    return dict(base_coeffs), jacobian


def _propagate(
    jacobian: dict[str, dict[str, float]],
    order: list[str],
    sigmas: dict[str, float],
    corr: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Propagate the uncertainties: return (u_c, shares) per coefficient.

    ``u_c^2 = J U J^T`` with ``U[i, i] = sigma_i^2`` and
    ``U[i, j] = rho_ij sigma_i sigma_j``. The share of input ``i`` is
    ``J_i (U J)_i / u_c^2``; it equals ``(J_i sigma_i)^2 / u_c^2`` for
    independent inputs, and the shares sum to 1 when ``u_c > 0``.
    """
    sigma = np.array([sigmas[key] for key in order], dtype=float)
    cov = np.diag(sigma ** 2)
    index = {key: k for k, key in enumerate(order)}
    for (first, second), rho in corr.items():
        if first in index and second in index:
            i, j = index[first], index[second]
            cov[i, j] += rho * sigma[i] * sigma[j]
            cov[j, i] += rho * sigma[i] * sigma[j]
    u_c: dict[str, float] = {}
    shares: dict[str, dict[str, float]] = {}
    for name in COEFFICIENT_NAMES:
        grad = np.array([float(jacobian[name].get(key, 0.0)) for key in order], dtype=float)
        var = float(grad @ cov @ grad)
        var = max(var, 0.0)
        u_c[name] = math.sqrt(var)
        if var <= 0.0:
            shares[name] = {key: 0.0 for key in order}
        else:
            contrib = grad * (cov @ grad) / var
            shares[name] = {key: float(contrib[k]) for k, key in enumerate(order)}
    return u_c, shares


def input_error_bars(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition | None = None,
    settings: SolverSettings | None = None,
    *,
    uncertainties: dict,
    correlations: dict | None = None,
    axes: str = "body",
    coverage_factor: float = 2.0,
) -> ErrorBarResult:
    """Compute layer-2 input-uncertainty error bars by linear propagation.

    First-order propagation of the standard uncertainties of the flow
    inputs and of the linear section properties through the gradients of
    the six force and moment coefficients (CL, CD, CY, Cl, Cm, Cn):
    ``u_c(f)^2 = J U J^T`` (GUM JCGM 100:2008, section 5.1, equation 10
    for independent inputs and equation 13 with correlations). CD is
    CDi + CDp as in ``gradients``. The expanded half-width is
    ``k u_c`` with the coverage factor ``k`` (default 2, GUM section 6).

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Case geometry. A single surface is wrapped into an aircraft.
    condition : FlightCondition or None
        Flight condition. When None, the default condition is used
        (alpha 5 deg, V_inf 50 m/s [m/s], free air).
    settings : SolverSettings or None
        Solver settings. When None, the defaults are used (the VLM).
    uncertainties : dict
        Standard uncertainties (one standard deviation), never negative,
        finite. Optional flow keys: ``alpha`` [rad], ``beta`` [rad],
        ``phi`` [rad], ``V_inf`` [m/s], ``rho`` [kg/m^3], ``h`` [m]
        (ground effect only). Optional ``sections``: dict of surface name
        to dict with optional ``a0`` [1/rad], ``alpha_L0`` [rad],
        ``Cd0`` [-], ``Cm0`` [-]. A section uncertainty of a surface
        applies to every strip of that surface and is fully correlated
        along the span.
    correlations : dict or None
        Correlation coefficients of pairs of flow inputs, for example
        ``{("alpha", "V_inf"): 0.5}``, each in [-1, 1]. Default
        independent inputs.
    axes : str
        Moment axis system of the Cl, Cm and Cn bars: ``"body"``,
        ``"stability"`` or ``"wind"``. ``"all"`` is refused.
    coverage_factor : float
        Coverage factor ``k`` [-], finite and positive. Default 2.

    Returns
    -------
    ErrorBarResult
        Bars with status ``"input_only"`` and coverage None. Each bar
        gives the input shares of ``u_c^2`` and the gradient method per
        input (``"adjoint"`` or ``"central differences"``).

    Raises
    ------
    ValueError
        For an unknown axis system, a non-positive coverage factor, a
        negative, non-finite or unknown uncertainty key, ``h`` in free
        air, an unknown surface name, a correlation outside [-1, 1], or
        a section input outside the adjoint scope (nonlinear lifting
        line, tabulated airfoils, deflected controls, body rates,
        Fourier solver), which names the limit.
    """
    if axes not in ("body", "stability", "wind"):
        raise ValueError(
            f"'axes' must be one of ['body', 'stability', 'wind'], got {axes!r}."
        )
    try:
        cover = float(coverage_factor)
    except (TypeError, ValueError):
        raise ValueError(
            f"coverage_factor must be a finite number above 0, got {coverage_factor!r}."
        ) from None
    if not math.isfinite(cover) or cover <= 0.0:
        raise ValueError(
            f"coverage_factor must be a finite number above 0, got {coverage_factor!r}."
        )
    aircraft = (
        Aircraft(name="SingleWing", surfaces=[geometry])
        if isinstance(geometry, LiftingSurface)
        else geometry.clone()
    )
    cond = condition.clone() if condition is not None else FlightCondition()
    base = settings.clone() if settings is not None else SolverSettings()
    surface_names = [surf.name for surf in aircraft.surfaces]
    flow, section_unc, corr = _parse_uncertainties(
        uncertainties, correlations, cond, surface_names)

    base_coeffs, base_res = _production_coeffs(aircraft, cond, base, axes)
    lattice = base_res.details.get("lattice")

    method_per_input: dict[str, str] = {}
    try:
        if lattice is None:
            raise ValueError("no lattice of the production solve (see T-0071)")
        values, jacobian = _adjoint_gradients(
            aircraft, cond, base, axes, flow, section_unc, lattice)
        uniform_method = "adjoint"
    except ValueError as exc:
        if "T-0071" not in str(exc) and "no lattice" not in str(exc):
            raise
        if section_unc:
            raise ValueError(
                "Section uncertainties need adjoint gradients, which the present "
                "configuration has no scope for (nonlinear lifting line, tabulated "
                "airfoils, deflected controls, body rates or Fourier solver, see "
                "T-0071): flow inputs use central differences of the production solve."
            ) from exc
        values, jacobian = _central_gradients(aircraft, cond, base, axes, flow, base_coeffs)
        uniform_method = "central differences"

    order = [key for key in FLOW_INPUTS if key in flow]
    order += [f"sections.{surf}.{prop}" for surf, props in section_unc.items() for prop in props]
    sigmas: dict[str, float] = dict(flow)
    for surf, props in section_unc.items():
        for prop, sigma in props.items():
            sigmas[f"sections.{surf}.{prop}"] = sigma
    for key in order:
        method_per_input[key] = uniform_method
    u_c, shares = _propagate(jacobian, order, sigmas, corr)

    if uniform_method == "adjoint":
        method_text = (
            "GUM law of propagation of uncertainty "
            "(JCGM 100:2008, eq. 10 for independent inputs, eq. 13 with correlations; "
            "Oberkampf and Roy 2010) with adjoint gradients"
        )
    else:
        method_text = (
            "GUM law of propagation of uncertainty "
            "(JCGM 100:2008, eq. 10 for independent inputs, eq. 13 with correlations; "
            "Oberkampf and Roy 2010) with central differences of the production solve "
            "(alpha 0.5 deg, beta 1 deg, phi 0.5 deg, V_inf 1 %, rho 1 %, "
            "h 5 % of the smallest ground clearance)"
        )
    use_total = base_res.totals.CDp is not None
    drag_basis = ("CD_total (induced + profile)" if use_total
                  else "CDi only: the airfoils have no profile drag")
    bars: dict[str, ErrorBar] = {}
    for name in COEFFICIENT_NAMES:
        half = cover * u_c[name]
        value = float(values[name])
        layer = LayerContribution(
            layer="input",
            half_width=half,
            method=method_text,
            state="propagated",
            levels=[],
            values=[],
        )
        bar_notes = [
            f"k = {cover:g} assumes a normal distribution of the output "
            f"(expanded half-width k u_c with u_c = {u_c[name]:.4g})."
        ]
        bars[name] = ErrorBar(
            name=name,
            value=value,
            interval_low=value - half,
            interval_high=value + half,
            status="input_only",
            coverage=None,
            layers={"input": layer},
            notes=bar_notes,
            input_shares=dict(shares[name]),
            gradient_methods={key: uniform_method for key in order},
        )
    notes = [
        INPUT_LAYER_NOTE,
        f"First-order propagation u_c^2 = J U J^T with {uniform_method} gradients; "
        f"coverage factor k = {cover:g}.",
    ]
    return ErrorBarResult(
        bars=bars,
        axes=axes,
        drag_basis=drag_basis,
        solver=str(base_res.solver_type),
        levels=[],
        results=[base_res],
        notes=notes,
    )
