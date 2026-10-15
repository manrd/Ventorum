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
  Reynolds number ``38.21 (L/k)^1.053``; section 7.9: the wetted-area
  approximation of a lifting surface.
* E. Torenbeek, "Synthesis of Subsonic Airplane Design", Delft University
  Press, 1982, appendix F: the alternative wing form factor.
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

#: Dynamic viscosity of air at ISA sea level [kg/(m s)] (ISO 2533:1975).
MU_SL = 1.789e-5

#: Largest thickness ratio that the lifting form factors accept [-].
TC_MAX = 0.4


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

    Roughness follows the cutoff Reynolds number of Raymer (2018),
    section 12.5: ``Re_co = 38.21 (L/k)^1.053`` with the reference
    length ``L`` [m] and the equivalent sand roughness ``k`` [m]. The
    skin friction is evaluated at ``min(Re, Re_co)``; when the flight
    Reynolds number exceeds the cutoff, the layer is roughness-limited
    and treated as fully turbulent at the cutoff Reynolds number.

    Parameters
    ----------
    re : float
        Reynolds number based on the reference length [-].
    x_transition : float
        Fraction of the reference length with laminar run [-], in
        [0, 1]. Default 0 (fully turbulent).
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
        re_cutoff = 38.21 * (float(length) / roughness) ** 1.053
        if re > re_cutoff:
            return float(0.455 / np.log10(re_cutoff) ** 2.58)
    if x_transition == 0.0:
        return float(0.455 / np.log10(re) ** 2.58)
    if x_transition == 1.0:
        return float(1.328 / np.sqrt(re))
    re_transition = x_transition * re
    if re_transition < 1.0:
        return float(0.455 / np.log10(re) ** 2.58)
    cf_turb = float(0.455 / np.log10(re) ** 2.58)
    cf_turb_tr = float(0.455 / np.log10(re_transition) ** 2.58)
    cf_lam_tr = float(1.328 / np.sqrt(re_transition))
    return cf_turb - x_transition * (cf_turb_tr - cf_lam_tr)


def form_factor_lifting(
    t_c: float,
    x_t: float = 0.3,
    sweep_max_t: float = 0.0,
    method: str = "raymer",
) -> float:
    """Return the form factor of a lifting surface section.

    The ``"raymer"`` method (default) uses the wing form factor of Raymer
    (2018), section 12.5, adapted to incompressible flow: the Mach factor
    ``1.34 M^0.18`` is set to 1 (this package covers Mach below 0.3 and
    applies no compressibility correction here) and the sweep factor is
    kept, so with the thickness ratio ``t/c`` [-], the position of maximum
    thickness ``x_t`` [-] and the sweep of the maximum-thickness line
    ``sweep_max_t`` [rad],

    ``FF = (1 + 0.6 / x_t * t/c + 100 (t/c)^4) (cos sweep_max_t)^0.28``.

    The ``"torenbeek"`` method uses the wing form factor of Torenbeek
    (1982), appendix F, evaluated on the thickness ratio normal to the
    maximum-thickness line ``(t/c)_n = (t/c) / cos(sweep_max_t)``:

    ``FF = 1 + 2.7 (t/c)_n + 100 (t/c)_n^4``.

    Parameters
    ----------
    t_c : float
        Streamwise thickness ratio [-], in (0, 0.4].
    x_t : float
        Chordwise position of maximum thickness [-], in (0, 1].
        Used only by ``method="raymer"``.
    sweep_max_t : float
        Sweep angle of the maximum-thickness line [rad], in [0, pi/2).
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
        t_cn = t_c / float(np.cos(sweep_max_t))
        if t_cn > TC_MAX:
            raise ValueError(
                f"normal thickness ratio (t/c)/cos(sweep) = {t_cn} exceeds {TC_MAX}."
            )
        return float(1.0 + 2.7 * t_cn + 100.0 * t_cn**4)
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


def wetted_area_lifting_surface(surface: LiftingSurface, t_c: float) -> float:
    """Return the wetted area of a lifting surface.

    The planform area comes from the section chords by the trapezium
    rule along the semi-span (both halves when the surface is
    symmetric). The wetted area is twice the planform area times
    ``(1 + 0.2 t/c)``, the approximation of Raymer (2018), section 7.9,
    which adds 20 percent of the thickness ratio for the section
    curvature. The span uses ``y = y_frac * semi_span`` (flat
    projection); dihedral and sweep change the true area by a cosine
    factor that this approximation omits. For a wing-body combination,
    subtract the fuselage carry-through or pass the exposed area through
    :class:`DragComponent` directly.

    Parameters
    ----------
    surface : LiftingSurface
        Lifting surface with at least two sections.
    t_c : float
        Thickness ratio [-], in (0, 0.4].

    Returns
    -------
    float
        Wetted area [m^2].

    Raises
    ------
    ValueError
        If *t_c* is out of range or the surface has fewer than two
        sections.
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
    y = eta * float(surface.semi_span)
    area_half = float(np.sum(0.5 * (chords[:-1] + chords[1:]) * np.abs(np.diff(y))))
    area = 2.0 * area_half if surface.is_symmetric else area_half
    return float(2.0 * area * (1.0 + 0.2 * t_c))


def parasite_drag(
    components: Sequence[DragComponent],
    condition: FlightCondition,
    s_ref: float,
) -> ParasiteDragResult:
    """Return the parasite drag of a component build-up.

    Each component gives, with its Reynolds number
    ``Re = rho V L / mu`` (density [kg/m^3] and velocity [m/s] from the
    flight condition, viscosity at ISA sea level),

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

    Returns
    -------
    ParasiteDragResult
        Per-component and total zero-lift drag coefficients with
        uncertainties.

    Raises
    ------
    ValueError
        If the component list is empty, two names match, or *s_ref* is
        not positive.
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
    cd0_by_component: dict[str, float] = {}
    rel_unc: dict[str, float] = {}
    for comp in comps:
        re = rho * v_inf * float(comp.reference_length) / MU_SL
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
