# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parasite-drag build-up from skin friction and form factors.

This module estimates the zero-lift (parasite) drag of aircraft parts that
the lifting models do not cover (fuselages, hulls, nacelles, struts) and
gives a cross-check of the polar-based profile drag of lifting surfaces.
It changes no solver result and it couples to no solver.

Skin friction follows flat-plate theory: the laminar Blasius relation, the
turbulent Prandtl-Schlichting relation, the Schlichting composite rule for
mixed laminar-turbulent flow at a given transition position, and the cutoff
Reynolds number for rough surfaces. Form factors follow the component
build-up of Raymer, with the wing option of Torenbeek and the body option
of Hoerner as alternatives.

Scope limit: the package covers incompressible flow (Mach below 0.3). The
Mach factor ``1.34 M^0.18`` of the Raymer wing form factor is set to 1 and
no compressibility correction of the skin friction is applied. This is a
stated limit of the module, not a result of a low-Mach evaluation of
Raymer's factor.

References
----------
* H. Blasius, "Grenzschichten in Fluessigkeiten mit kleiner Reibung",
  Zeitschrift fuer Mathematik und Physik 56 (1908) 1-37: the laminar
  flat-plate skin friction ``1.328 / sqrt(Re)``.
* L. Prandtl (1927) and H. Schlichting, "Boundary-Layer Theory",
  Springer, 9th ed. 2017, chapter on the smooth flat plate at zero
  incidence: the turbulent skin friction ``0.455 / (log10 Re)^2.58`` and
  the composite rule for mixed laminar-turbulent flow.
* D. P. Raymer, "Aircraft Design: A Conceptual Approach", AIAA, 6th ed.
  2018, section 12.5: the parasite-drag build-up, the wing form factor,
  the fuselage form factor, the interference factors and the cutoff
  Reynolds number ``38.21 (L/k)^1.053``; chapter 7 (wetted area): a
  thickness-dependent factor between the exposed planform area and the
  wetted area of a lifting surface. This module uses the approximation
  ``2 (1 + 0.2 t/c)`` of that thickness factor (see Raymer).
* E. Torenbeek, "Synthesis of Subsonic Airplane Design", Delft University
  Press, 1982, appendix F: the alternative wing form factor
  ``1 + 2.7 t/c + 100 (t/c)^4``.
* S. F. Hoerner, "Fluid-Dynamic Drag", published by the author, 1965,
  chapter 6: the alternative streamlined-body form factor.
* ISO 2533:1975 (ISA sea level): the air density and dynamic viscosity
  used for the Reynolds number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ventorum.core.datatypes import FlightCondition, LiftingSurface
from ventorum.geometry.lattice import surface_reference_line
from ventorum.viscous.alpha_correction import MU_AIR_15C

#: Dynamic viscosity of air at ISA sea level [Pa s] (ISO 2533:1975). The
#: value is the package constant of the viscous coupling, so the parasite
#: drag and the strip Reynolds numbers use the same air.
MU_SL = MU_AIR_15C

#: Largest thickness ratio that the lifting form factors accept [-].
TC_MAX = 0.4


def _cf_laminar(re: float) -> float:
    """Blasius mean skin friction ``1.328 / sqrt(Re)`` [-]."""
    return float(1.328 / np.sqrt(re))


def _cf_turbulent(re: float) -> float:
    """Prandtl-Schlichting mean skin friction ``0.455 / (log10 Re)^2.58`` [-]."""
    return float(0.455 / np.log10(re) ** 2.58)


def _equal_friction_reynolds() -> float:
    """Return the Reynolds number where the laminar and turbulent laws agree.

    Bisection in log10(Re) on [3, 6]. Below this Reynolds number the
    Blasius value is larger than the Prandtl-Schlichting value.
    """
    lo, hi = 3.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        re = 10.0**mid
        if _cf_turbulent(re) > _cf_laminar(re):
            hi = mid
        else:
            lo = mid
    return float(10.0 ** (0.5 * (lo + hi)))


#: Smallest transition Reynolds number with a laminar run [-]: the Reynolds
#: number where the Blasius and the Prandtl-Schlichting mean skin friction
#: agree (about 1.2e4). Below it the composite rule would add drag for a
#: laminar run (and for a run near Re = 1 it gives a negative value), so the
#: run counts as turbulent. At this value the composite rule equals the
#: fully turbulent value, so the skin friction stays continuous.
RE_TRANSITION_MIN = _equal_friction_reynolds()


def flat_plate_cf(
    re: float,
    x_transition: float = 0.0,
    roughness: float | None = None,
    length: float | None = None,
) -> float:
    """Return the mean skin-friction coefficient of a smooth flat plate.

    Laminar flow uses the Blasius relation ``1.328 / sqrt(Re)`` (Blasius
    1908). Fully turbulent flow uses the Prandtl-Schlichting relation
    ``0.455 / (log10 Re)^2.58`` (Schlichting, "Boundary-Layer Theory").
    Mixed flow uses the Schlichting composite rule: with transition at
    the Reynolds number ``Re_tr = x_transition * Re``,

    ``Cf = Cf_turb(Re) - x_transition * (Cf_turb(Re_tr) - Cf_lam(Re_tr))``,

    so ``x_transition = 0`` is fully turbulent and ``x_transition = 1``
    is fully laminar. At a transition Reynolds number of 5e5 this gives
    a correction term of 1612/Re, against the value 1700/Re quoted in
    some textbooks; the difference is a small fraction of the correction.

    Guard of the composite rule: a laminar run counts only when
    ``Re_tr >= RE_TRANSITION_MIN`` (about 1.2e4, where the laminar and
    the turbulent law give the same mean skin friction). A shorter run
    counts as turbulent. With this guard and ``Re >= RE_TRANSITION_MIN``,
    the skin friction is continuous in ``x_transition``, does not rise
    when the laminar run grows, and stays between the laminar and the
    fully turbulent value. Below ``Re = RE_TRANSITION_MIN`` the turbulent
    law is outside its range: the function returns the turbulent value
    for ``x_transition < 1`` and the laminar value for 1.

    Roughness follows the cutoff Reynolds number of Raymer (2018),
    section 12.5: ``Re_co = 38.21 (L/k)^1.053`` with the reference
    length ``L`` [m] and the equivalent sand roughness ``k`` [m]. Raymer
    evaluates the turbulent skin friction at ``min(Re, Re_co)``. The
    cutoff acts on the turbulent part only; the laminar part keeps its
    value. For mixed flow this module adds the roughness increment of the
    turbulent law, ``Cf_turb(min(Re, Re_co)) - Cf_turb(Re)``, weighted by
    the turbulent fraction ``1 - x_transition`` of the length (an
    approximation of this module). The increment is zero below the
    cutoff, so the skin friction is continuous across the cutoff.

    Parameters
    ----------
    re : float
        Reynolds number based on the reference length [-].
    x_transition : float
        Fraction of the reference length with laminar run [-], in
        [0, 1]. Default 0 (fully turbulent). A run with a Reynolds number
        below ``RE_TRANSITION_MIN`` counts as turbulent.
    roughness : float or None
        Equivalent sand roughness height [m]. None (default) means a
        hydraulically smooth surface.
    length : float or None
        Reference length [m]. Required when *roughness* is given.

    Returns
    -------
    float
        Mean skin-friction coefficient [-].

    Raises
    ------
    ValueError
        If *re* is not positive, *x_transition* is outside [0, 1], or
        *roughness* is given without a positive *length*.
    """
    re = float(re)
    x_transition = float(x_transition)
    if not np.isfinite(re) or re <= 0.0:
        raise ValueError(f"Reynolds number must be positive, got {re}.")
    if not np.isfinite(x_transition) or not 0.0 <= x_transition <= 1.0:
        raise ValueError(f"x_transition must be in [0, 1], got {x_transition}.")
    if roughness is not None:
        roughness = float(roughness)
        if not np.isfinite(roughness) or roughness <= 0.0:
            raise ValueError(f"roughness must be positive, got {roughness}.")
        if length is None or not np.isfinite(float(length)) or float(length) <= 0.0:
            raise ValueError("length [m] must be given and positive when roughness is given.")
    cf = _cf_smooth(re, x_transition)
    if roughness is not None:
        re_cutoff = 38.21 * (float(length) / roughness) ** 1.053
        if re > re_cutoff:
            increment = _cf_turbulent(re_cutoff) - _cf_turbulent(re)
            cf += (1.0 - x_transition) * increment
    return float(cf)


def _cf_smooth(re: float, x_transition: float) -> float:
    """Mean skin friction [-] of a smooth plate (composite rule with guard)."""
    if x_transition == 1.0:
        return _cf_laminar(re)
    cf_turb = _cf_turbulent(re)
    re_transition = x_transition * re
    if x_transition == 0.0 or re_transition < RE_TRANSITION_MIN:
        return cf_turb
    return cf_turb - x_transition * (_cf_turbulent(re_transition) - _cf_laminar(re_transition))


def form_factor_lifting(
    t_c: float,
    x_t: float = 0.3,
    sweep_max_t: float = 0.0,
    method: str = "raymer",
) -> float:
    """Return the form factor of a lifting surface section.

    The ``"raymer"`` method (default) uses the wing form factor of Raymer
    (2018), section 12.5, in the incompressible scope of this package:
    the Mach factor ``1.34 M^0.18`` of Raymer is set to 1 (scope limit:
    Mach below 0.3, no compressibility correction), and the sweep factor
    is kept, so with the thickness ratio ``t/c`` [-], the position of
    maximum thickness ``x_t`` [-] and the sweep of the maximum-thickness
    line ``sweep_max_t`` [rad],

    ``FF = (1 + 0.6 / x_t * t/c + 100 (t/c)^4) (cos sweep_max_t)^0.28``.

    The ``"torenbeek"`` method uses the wing form factor of Torenbeek
    (1982), appendix F, on the streamwise thickness ratio:

    ``FF = 1 + 2.7 t/c + 100 (t/c)^4``.

    This form has no sweep term, so ``sweep_max_t`` does not change it.

    Parameters
    ----------
    t_c : float
        Streamwise thickness ratio [-], in (0, 0.4].
    x_t : float
        Chordwise position of maximum thickness [-], in (0, 1].
        Used only by ``method="raymer"``.
    sweep_max_t : float
        Sweep angle of the maximum-thickness line [rad], in [0, pi/2).
        Used only by ``method="raymer"``.
    method : str
        ``"raymer"`` (default) or ``"torenbeek"``.

    Returns
    -------
    float
        Form factor [-], 1 for a flat plate, larger for thick sections.

    Raises
    ------
    ValueError
        If an input is out of range or the method is unknown.
    """
    t_c = float(t_c)
    x_t = float(x_t)
    sweep_max_t = float(sweep_max_t)
    if not np.isfinite(t_c) or not 0.0 < t_c <= TC_MAX:
        raise ValueError(f"t_c must be in (0, {TC_MAX}], got {t_c}.")
    if not np.isfinite(x_t) or not 0.0 < x_t <= 1.0:
        raise ValueError(f"x_t must be in (0, 1], got {x_t}.")
    if not np.isfinite(sweep_max_t) or not 0.0 <= sweep_max_t < np.pi / 2.0:
        raise ValueError(f"sweep_max_t [rad] must be in [0, pi/2), got {sweep_max_t}.")
    if method == "raymer":
        return float(
            (1.0 + 0.6 / x_t * t_c + 100.0 * t_c**4) * np.cos(sweep_max_t) ** 0.28
        )
    if method == "torenbeek":
        return float(1.0 + 2.7 * t_c + 100.0 * t_c**4)
    raise ValueError(f"Unknown lifting form-factor method {method!r}; use 'raymer' or 'torenbeek'.")


def form_factor_body(fineness: float, method: str = "raymer") -> float:
    """Return the form factor of a streamlined body.

    The ``"raymer"`` method (default) uses the fuselage form factor of
    Raymer (2018), section 12.5, with the fineness ``f = l/d`` [-]:

    ``FF = 1 + 60 / f^3 + f / 400``.

    The ``"hoerner"`` method uses the streamlined-body form factor of
    Hoerner (1965), chapter 6, with ``d/l = 1/f``:

    ``FF = 1 + 1.5 (d/l)^1.5 + 7 (d/l)^3``.

    Both factors fall with rising fineness.

    Parameters
    ----------
    fineness : float
        Length-to-diameter ratio [-]. Must be positive.
    method : str
        ``"raymer"`` (default) or ``"hoerner"``.

    Returns
    -------
    float
        Form factor [-].

    Raises
    ------
    ValueError
        If *fineness* is not positive or the method is unknown.
    """
    fineness = float(fineness)
    if not np.isfinite(fineness) or fineness <= 0.0:
        raise ValueError(f"fineness must be positive, got {fineness}.")
    if method == "raymer":
        return float(1.0 + 60.0 / fineness**3 + fineness / 400.0)
    if method == "hoerner":
        d_l = 1.0 / fineness
        return float(1.0 + 1.5 * d_l**1.5 + 7.0 * d_l**3)
    raise ValueError(f"Unknown body form-factor method {method!r}; use 'raymer' or 'hoerner'.")


@dataclass(frozen=True, slots=True)
class DragComponent:
    """One part of the parasite-drag build-up.

    Parameters
    ----------
    name : str
        Component label (for example ``"wing"``).
    wetted_area : float
        Wetted area [m^2]. Must be positive.
    reference_length : float
        Length for the Reynolds number [m] (mean chord of a lifting
        part, body length of a fuselage). Must be positive.
    form_factor : float
        Form factor [-] (see :func:`form_factor_lifting` and
        :func:`form_factor_body`). Must be at least 1.
    interference : float
        Interference factor [-] (1 for no interference; typical values
        1.0 to 1.3 for a wing-body junction, from Raymer (2018),
        section 12.5). Passed as an input, never a hidden constant.
    laminar_fraction : float
        Fraction of the reference length with laminar run [-], in
        [0, 1]. Maps to *x_transition* of :func:`flat_plate_cf`.
    roughness_k : float or None
        Equivalent sand roughness height [m]. None (default) means
        hydraulically smooth.
    uncertainty : float
        Relative uncertainty of this component [-] (default 0.10, the
        typical scatter of form-factor methods). Passed as an input.
    """

    name: str
    wetted_area: float
    reference_length: float
    form_factor: float
    interference: float = 1.0
    laminar_fraction: float = 0.0
    roughness_k: float | None = None
    uncertainty: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string.")
        if not np.isfinite(float(self.wetted_area)) or float(self.wetted_area) <= 0.0:
            raise ValueError(f"wetted_area [m^2] must be positive, got {self.wetted_area}.")
        if not np.isfinite(float(self.reference_length)) or float(self.reference_length) <= 0.0:
            raise ValueError(
                f"reference_length [m] must be positive, got {self.reference_length}."
            )
        if not np.isfinite(float(self.form_factor)) or float(self.form_factor) < 1.0:
            raise ValueError(f"form_factor [-] must be at least 1, got {self.form_factor}.")
        if not np.isfinite(float(self.interference)) or float(self.interference) <= 0.0:
            raise ValueError(f"interference [-] must be positive, got {self.interference}.")
        if not np.isfinite(float(self.laminar_fraction)) or not (
            0.0 <= float(self.laminar_fraction) <= 1.0
        ):
            raise ValueError(
                f"laminar_fraction [-] must be in [0, 1], got {self.laminar_fraction}."
            )
        if self.roughness_k is not None and (
            not np.isfinite(float(self.roughness_k)) or float(self.roughness_k) <= 0.0
        ):
            raise ValueError(f"roughness_k [m] must be positive, got {self.roughness_k}.")
        if not np.isfinite(float(self.uncertainty)) or float(self.uncertainty) < 0.0:
            raise ValueError(f"uncertainty [-] must be non-negative, got {self.uncertainty}.")


@dataclass(frozen=True, slots=True)
class ParasiteDragResult:
    """Zero-lift drag of a component build-up.

    Parameters
    ----------
    cd0_by_component : dict[str, float]
        Zero-lift drag coefficient of each component [-], counted on
        *s_ref*.
    cd0_total : float
        Sum of the component coefficients [-].
    relative_uncertainty_by_component : dict[str, float]
        Input relative uncertainty of each component [-].
    cd0_total_uncertainty : float
        Absolute uncertainty of the total [-] (root sum square of the
        absolute component uncertainties, treated as independent).
    s_ref : float
        Reference area of the coefficients [m^2].
    """

    cd0_by_component: dict[str, float]
    cd0_total: float
    relative_uncertainty_by_component: dict[str, float]
    cd0_total_uncertainty: float
    s_ref: float


def wetted_area_lifting_surface(
    surface: LiftingSurface,
    t_c: float,
    covered_root_span: float = 0.0,
) -> float:
    """Return the wetted area of the exposed part of a lifting surface.

    The planform area comes from the section chords by the trapezium rule
    along the span (both halves when the surface is symmetric). The span
    distance between two sections is the true distance between their
    leading edges in the y-z plane: ``semi_span`` is measured along the
    dihedral line (or, when a section sets ``z_le``, the y extent with the
    z offsets of the sections), as :class:`LiftingSurface` defines it. The
    chords are streamwise, and two parallel streamwise chords at this
    distance bound a trapezium of area ``(c1 + c2) / 2`` times the
    distance, whatever the sweep. So sweep and dihedral need no cosine
    factor; section twist is neglected.

    The exposed area leaves out the span length ``covered_root_span``
    [m] next to the root section on each half (the part inside a
    fuselage). With the default 0 the result is the gross area.

    The wetted area is the exposed area times ``2 (1 + 0.2 t/c)``: the
    factor 2 counts the upper and the lower surface, and ``0.2 t/c`` is an
    approximation of the thickness factor between the planform and the
    wetted area of a section (see Raymer (2018), chapter 7, wetted area).

    Parameters
    ----------
    surface : LiftingSurface
        Lifting surface with at least two sections.
    t_c : float
        Thickness ratio [-], in (0, 0.4].
    covered_root_span : float
        Span length [m] from the root section, along the surface, that a
        fuselage covers on each half. In [0, span length of one half).
        Default 0 (gross area). For a wing with small dihedral this is
        about the fuselage half-width at the wing.

    Returns
    -------
    float
        Wetted area [m^2] of the exposed part.

    Raises
    ------
    ValueError
        If *t_c* or *covered_root_span* is out of range or the surface has
        fewer than two sections.
    """
    t_c = float(t_c)
    if not np.isfinite(t_c) or not 0.0 < t_c <= TC_MAX:
        raise ValueError(f"t_c must be in (0, {TC_MAX}], got {t_c}.")
    sections = sorted(surface.sections, key=lambda s: s.y_frac)
    if len(sections) < 2:
        raise ValueError("The surface needs at least two sections.")
    eta = np.array([s.y_frac for s in sections], dtype=float)
    chords = np.array([s.chord for s in sections], dtype=float)
    if np.any(chords <= 0.0):
        raise ValueError("All section chords [m] must be positive.")
    _, y, z = surface_reference_line(surface, eta)
    span_step = np.hypot(np.diff(y), np.diff(z))
    station = np.concatenate([[0.0], np.cumsum(span_step)])
    covered = float(covered_root_span)
    if not np.isfinite(covered) or not 0.0 <= covered < float(station[-1]):
        raise ValueError(
            f"covered_root_span [m] must be in [0, {float(station[-1])}), got {covered}."
        )
    keep = station > covered
    s_exposed = np.concatenate([[covered], station[keep]])
    c_exposed = np.concatenate([[float(np.interp(covered, station, chords))], chords[keep]])
    area_half = float(np.sum(0.5 * (c_exposed[:-1] + c_exposed[1:]) * np.diff(s_exposed)))
    area = 2.0 * area_half if surface.is_symmetric else area_half
    return float(2.0 * area * (1.0 + 0.2 * t_c))


def parasite_drag(
    components: Sequence[DragComponent],
    condition: FlightCondition,
    s_ref: float,
    mu: float = MU_SL,
) -> ParasiteDragResult:
    """Return the parasite drag of a component build-up.

    Each component gives, with its Reynolds number
    ``Re = rho V L / mu`` (density [kg/m^3] and velocity [m/s] from the
    flight condition, dynamic viscosity ``mu`` [Pa s]),

    ``CD0_i = Cf_i FF_i Q_i S_wet_i / S_ref``,

    with the skin-friction coefficient ``Cf_i`` [-] from
    :func:`flat_plate_cf` (transition at ``laminar_fraction`` of the
    reference length, roughness ``roughness_k``), the form factor
    ``FF_i`` [-], the interference factor ``Q_i`` [-] and the wetted
    area [m^2]. The total is the sum of the components (Raymer (2018),
    section 12.5).

    Parameters
    ----------
    components : Sequence[DragComponent]
        Non-empty list of components with unique names.
    condition : FlightCondition
        Flight condition (uses ``V_inf`` [m/s] and ``rho`` [kg/m^3]).
    s_ref : float
        Reference area of the coefficients [m^2]. Must be positive.
    mu : float
        Dynamic viscosity of the air [Pa s]. Default ``MU_SL``, the ISA
        sea-level value. Give the value of the flight altitude or of the
        test air.

    Returns
    -------
    ParasiteDragResult
        Per-component and total zero-lift drag coefficients with
        uncertainties.

    Raises
    ------
    ValueError
        If the component list is empty, two names match, or *s_ref* or
        *mu* is not positive.
    """
    comps = list(components)
    if not comps:
        raise ValueError("components must hold at least one DragComponent.")
    names = [c.name for c in comps]
    if len(set(names)) != len(names):
        raise ValueError(f"Component names must be unique, got {names}.")
    s_ref = float(s_ref)
    if not np.isfinite(s_ref) or s_ref <= 0.0:
        raise ValueError(f"s_ref [m^2] must be positive, got {s_ref}.")
    rho = float(condition.rho)
    v_inf = float(condition.V_inf)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"condition.rho [kg/m^3] must be positive, got {rho}.")
    if not np.isfinite(v_inf) or v_inf <= 0.0:
        raise ValueError(f"condition.V_inf [m/s] must be positive, got {v_inf}.")
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"mu [Pa s] must be positive, got {mu}.")
    cd0_by_component: dict[str, float] = {}
    rel_unc: dict[str, float] = {}
    for comp in comps:
        re = rho * v_inf * float(comp.reference_length) / mu
        cf = flat_plate_cf(
            re,
            x_transition=float(comp.laminar_fraction),
            roughness=None if comp.roughness_k is None else float(comp.roughness_k),
            length=float(comp.reference_length),
        )
        cd0 = cf * float(comp.form_factor) * float(comp.interference) * float(
            comp.wetted_area
        ) / s_ref
        cd0_by_component[comp.name] = float(cd0)
        rel_unc[comp.name] = float(comp.uncertainty)
    cd0_total = float(sum(cd0_by_component.values()))
    cd0_total_uncertainty = float(
        np.sqrt(sum((rel_unc[c.name] * cd0_by_component[c.name]) ** 2 for c in comps))
    )
    return ParasiteDragResult(
        cd0_by_component=cd0_by_component,
        cd0_total=cd0_total,
        relative_uncertainty_by_component=rel_unc,
        cd0_total_uncertainty=cd0_total_uncertainty,
        s_ref=s_ref,
    )
