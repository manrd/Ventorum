# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Viscous-inviscid coupling by angle-of-attack correction (V1, decambering with 2D polars).

This module couples the linear potential-flow solvers (vortex-lattice
method and linear lifting line) with two-dimensional section polars. One
correction angle ``delta_j`` [rad] per spanwise strip changes the
right-hand side of that strip (a uniform change of its angle of attack).
The system matrix does not change, so its LU factorization is reused in
every iteration, for the response matrix of the corrections, for the
second branch of a post-stall case and for every condition of a sweep
whose matrix is the same.

Effective angle and flaps. Each strip of the lattice behaves like a 2D
section of the lattice model: ``cl = a_j (alpha_e - alpha0_j)``. For the
vortex-lattice method, ``a_j`` [1/rad] and ``alpha0_j`` [rad] come from
the lattice itself: the chordwise vortices, control points and boundary
normals of the strip (flap panels included) form a 2D discrete-vortex
section, and two right-hand sides (a uniform angle and the normal angles
of the panels) with one factorization give its lift slope and its
zero-lift angle (thin-airfoil theory; Katz and Plotkin 2001, discrete
vortex method). For the lifting line the section model is the strip
airfoil of the lattice (its ``a0`` and ``alpha_L0``, with the Glauert flap
shift on a deflected strip). The effective angle of strip j is then
``alpha_e,j = alpha0_j + cl_inv,j / a_j``, measured from the chord of the
clean section, and the viscous polar of the strip (flapped when the strip
has a deflected control) is read at ``alpha_v,j = alpha_e,j + delta_j``.
The flap is thus counted once: the lattice gives the inviscid flap effect
in ``cl_inv``, and ``alpha0_j`` takes it out of the effective angle.

Reference of a flapped lattice strip. The coarse lattice has a smaller
flap effect than thin-airfoil theory (the error falls as 1 / n_chord),
while a flapped polar carries the full flap effect. The coupling
therefore uses the departure of the polar from its own thin-airfoil
value, added to the lift of the lattice section (the decambering idea of
Mukherjee and Gopalarathnam 2006: only the departure from the linear
model is a correction). The thin-airfoil model of strip j has the slope
``a_j`` and the zero-lift angle
``alpha0_TA,j = alpha0_clean,j - tau(x_h) phi_j kappa_j``: the clean
section of the lattice (its camber tilt), the Glauert flap shift of the
polar flap (hinge ``x_h`` [-], deflection ``phi_j`` [rad]) and, when the
polar holds the ground height, the ground factor ``kappa_j`` of the flap
shift from a converged 2D discrete-vortex section with images (1 in free
air). The target lift is
``cl_target = cl_visc(alpha_v) - a_j (alpha0_j - alpha0_TA,j)``. A polar
that equals thin-airfoil theory then gives zero correction at any
chordwise panel count, and the coupled result equals the inviscid lattice
at the same panel count (the reference that the owner chose). Strips
without a deflected control and the lifting line (whose strip airfoil
already has the Glauert shift) have ``alpha0_TA,j = alpha0_j``: no change.

Ground effect. When the polar of a strip has the ground height (a table
with a height axis, or a source with ground support), the 2D section of
the vortex-lattice strip also has its images in the ground, so the 2D
ground effect is not counted twice. A free-air polar is read with the
free-air section relation.

Residual and Newton iteration::

    cl_inv(delta) = cl_inv(0) - D delta
    alpha_v(delta) = alpha0 + cl_inv(delta) / a + delta
    r(delta) = cl_inv(delta) - cl_visc(alpha_v(delta)) + a (alpha0 - alpha0_TA)
    J = -(I - diag(m / a)) D - diag(m),   m = dcl_visc / dalpha at alpha_v

with a backtracking line search, an optional under-relaxation factor and
a damped fixed-point step when the Newton step fails.

References
----------
* J. B. Tseng and C. E. Lan, NASA TP-2828, 1988 (alpha-reduction coupling
  of a vortex lattice with section data).
* C. P. van Dam, J. C. Vander Kam and P. C. Paris, "Design-Oriented
  High-Lift Methodology for General Aviation and Civil Transport Aircraft",
  Journal of Aircraft 38(6), 2001, pp. 1076-1084.
* S. Gallay and E. Laurendeau, "Nonlinear Generalized Lifting-Line Coupling
  Algorithms for Pre/Poststall Flows", AIAA Journal 53(7), 2015,
  pp. 1784-1792.
* R. Mukherjee and A. Gopalarathnam, "Poststall Prediction of
  Multiple-Lifting-Surface Configurations Using a Decambering Approach",
  Journal of Aircraft 43(3), 2006, pp. 660-668.
* H. Glauert, "Theoretical Relationships for an Aerofoil with Hinged
  Flap", ARC R&M 1095, 1927 (thin-airfoil flap shift).
* H. Glauert, "The Elements of Aerofoil and Airscrew Theory", Cambridge
  University Press, 1926 (thin-airfoil theory, flap effectiveness).
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001 (discrete vortex method, method of images).
* ISO 2533:1975 (ISA sea-level dynamic viscosity of air).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import os
import re
import time
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import scipy.linalg

from ventorum.aero.loads import compute_loads
from ventorum.core.constants import MU_SL
from ventorum.aero.polar_table import PolarTable, source_supports_ground
from ventorum.aero.polars import blend_tabulated
from ventorum.aero.system import GroundPlane, UnknownMap, freestream_direction, has_rates
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    LinearAirfoil,
    SolverResult,
    SolverSettings,
    TabulatedAirfoil,
)
from ventorum.core.trust import (
    evaluate_aerodynamic_trust,
    polar_limits,
    polar_stats,
)
from ventorum.geometry.controls import deflected_airfoil, flap_effectiveness
from ventorum.geometry.lattice import (
    VortexLattice,
    _AirfoilBlender,
    _sorted_sections,
    section_cl,
    section_cl_slope,
    section_cd,
    section_cm,
)
from ventorum.solvers.core import (
    _assemble_llt_linear,
    _llt_strip_data,
    assemble_vlm,
)
from ventorum.solvers.factory import make_solver
from ventorum.solvers.lattice_base import (
    as_aircraft,
    main_surface_strip_count,
    quarter_chord_sweep_deg,
)
from ventorum.utils.parallel import blas_single_thread, solve_threads
from ventorum.utils.validation import validate_aircraft

__all__ = [
    "BRANCH_CL_DIFFERENCE",
    "SOURCE_ALPHAS_DEG",
    "STALL_START_MARGIN_DEG",
    "solve_viscous",
    "strip_reynolds_number",
]

#: Angles of attack [deg] that a section polar source computes for each
#: strip polar.
SOURCE_ALPHAS_DEG = np.arange(-15.0, 25.0 + 1e-9, 0.5)

#: Two converged solutions are different branches when their CL differ by
#: more than this value [-].
BRANCH_CL_DIFFERENCE = 1e-3

#: The branch search starts once with every strip this angle [deg] above
#: the angle of maximum Cl of its polar (stalled start) and once with every
#: strip this angle below it (attached start).
STALL_START_MARGIN_DEG = 5.0

#: Hinge height of a control surface for a polar source, as a fraction of
#: the local thickness [-] (ControlSurface has no hinge height).
SOURCE_Y_HINGE = 0.5

#: Chordwise panels of the converged 2D discrete-vortex section that gives
#: the ground factor of the thin-airfoil flap shift [-].
GROUND_FLAP_REFERENCE_PANELS = 256

#: Trust and runtime warning when a thick section polar runs with a flap in ground effect.
_THICK_GROUND_FLAP_NOTE = (
    "The ground factor of the flap increment is a thin-airfoil value; "
    "for thick sections below h/c 0.30 it can differ by more than 2 % "
    "(about 0.1 to 0.2 deg of flap shift)."
)

_NACA_NAME = re.compile(r"^\s*(?:naca)?\s*(\d{4,5})\s*$", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# Strip data
# ═══════════════════════════════════════════════════════════════════════════════

def strip_reynolds_number(
    condition: FlightCondition,
    chord: float,
    mu: float = MU_SL,
) -> float:
    """Return the Reynolds number of a strip.

    Parameters
    ----------
    condition : FlightCondition
        Flight condition: speed ``V_inf`` [m/s] and density ``rho``
        [kg/m^3]. The condition has no viscosity field.
    chord : float
        Local chord [m].
    mu : float
        Dynamic viscosity [Pa s]. Default: the single ISA sea-level
        constant of the package, :data:`ventorum.core.constants.MU_SL`.

    Returns
    -------
    float
        ``rho V c / mu`` [-].
    """
    return float(condition.rho) * float(condition.V_inf) * float(chord) / float(mu)


def _strip_control(lattice: VortexLattice, aircraft: Aircraft, s: int) -> tuple[float, float | None]:
    """Return the flap deflection [deg] and the hinge [-] of strip *s* (0 and None without a control).

    On a vortex lattice with chordwise controls the deflection is the one of
    the lattice strip at the span fraction of its control point, so a strip
    in the join at a control side edge has the partial deflection of the
    join (see :class:`ventorum.geometry.controls.ControlJoinField`).
    """
    field = lattice.geom_cache.get("flap_field")
    if field is not None:
        if not np.isfinite(field["hinge"][s]) or field["strip_angle"][s] == 0.0:
            return 0.0, None
        return float(np.degrees(field["strip_angle"][s])), float(field["hinge_x_c"][s])
    surf = aircraft.surfaces[int(lattice.strip_surface[s])]
    eta = abs(float(lattice.eta[s]))
    is_right = bool(lattice.strip_is_right[s])
    for c in getattr(surf, "controls", []) or []:
        if c.eta_start - 1e-9 <= eta <= c.eta_end + 1e-9:
            if getattr(surf, "mirror_y", False):
                delta = c.deflection if c.symmetric else -c.deflection
            else:
                delta = c.deflection if (c.symmetric or is_right) else -c.deflection
            if delta == 0.0:
                return 0.0, None
            return float(np.degrees(delta)), float(c.hinge_x_c)
    return 0.0, None


def _base_airfoil(aircraft: Aircraft, lattice: VortexLattice, s: int) -> Any:
    """Return the section airfoil of strip *s* without a flap deflection."""
    surf = aircraft.surfaces[int(lattice.strip_surface[s])]
    return _AirfoilBlender(_sorted_sections(surf)).at(abs(float(lattice.eta[s])))


def _section_spec(section: Any) -> Any:
    """Return the shape of a wing section for a polar source, or None."""
    if section.shape is not None:
        return section.shape
    name = str(getattr(section.airfoil, "name", "") or "")
    if _NACA_NAME.match(name):
        return name
    return None


def _source_section(spec: Any) -> tuple[Any, str]:
    """Return the section argument of a polar source and its memo key."""
    from ventorum.airfoil import airfoil

    if isinstance(spec, (str, int, np.integer)) and _NACA_NAME.match(str(spec)):
        digits = _NACA_NAME.match(str(spec)).group(1)  # type: ignore[union-attr]
        return f"naca {digits}", f"naca {digits}"
    coords = np.asarray(airfoil(spec).coords, dtype=float)
    key = hashlib.sha256(np.round(coords, 9).tobytes()).hexdigest()
    return coords, key


def _strip_sections(aircraft: Aircraft, lattice: VortexLattice, s: int) -> list[tuple[Any, float]]:
    """Return the bracketing section shapes of strip *s* and their weights.

    Raises
    ------
    ValueError
        If a needed section has no shape: no silent default section.
    """
    surf = aircraft.surfaces[int(lattice.strip_surface[s])]
    secs = _sorted_sections(surf)
    eta = abs(float(lattice.eta[s]))
    fr = [float(sec.y_frac) for sec in secs]
    if eta <= fr[0]:
        picks = [(secs[0], 1.0)]
    elif eta >= fr[-1]:
        picks = [(secs[-1], 1.0)]
    else:
        j = int(np.clip(np.searchsorted(fr, eta, side="right") - 1, 0, len(fr) - 2))
        span = fr[j + 1] - fr[j]
        w = 0.0 if span <= 0.0 else (eta - fr[j]) / span
        picks = [(secs[j], 1.0 - w), (secs[j + 1], w)]
    out = []
    for sec, w in picks:
        if w <= 1e-12:
            continue
        spec = _section_spec(sec)
        if spec is None:
            raise ValueError(
                f"Surface {surf.name!r}: the section at y_frac={sec.y_frac:g} has no shape for the "
                "polar source. Set WingSection.shape (a NACA designation or coordinates), name its "
                "airfoil with a NACA designation, or give a PolarTable or a TabulatedAirfoil."
            )
        out.append((spec, float(w)))
    return out


def _strip_thickness_ratio(aircraft: Aircraft, lattice: VortexLattice, s: int, cand: Any = None) -> float:
    """Return the thickness-to-chord ratio t/c [-] of strip *s*, or 0.0 if not found."""
    surf = aircraft.surfaces[int(lattice.strip_surface[s])]
    secs = _sorted_sections(surf)
    eta = abs(float(lattice.eta[s]))
    fr = [float(sec.y_frac) for sec in secs]
    if eta <= fr[0]:
        picks = [(secs[0], 1.0)]
    elif eta >= fr[-1]:
        picks = [(secs[-1], 1.0)]
    else:
        j = int(np.clip(np.searchsorted(fr, eta, side="right") - 1, 0, len(fr) - 2))
        span = fr[j + 1] - fr[j]
        w = 0.0 if span <= 0.0 else (eta - fr[j]) / span
        picks = [(secs[j], 1.0 - w), (secs[j + 1], w)]

    t_vals = []
    w_sum = 0.0
    for sec, w in picks:
        if w <= 1e-12:
            continue
        spec = _section_spec(sec)
        if spec is not None:
            try:
                from ventorum.airfoil import airfoil

                af = airfoil(spec)
            except (KeyError, ValueError):
                continue
            t_vals.append(w * float(af.max_thickness()[0]))
            w_sum += w
    if w_sum > 0.0:
        return float(sum(t_vals) / w_sum)

    if isinstance(cand, PolarTable) and isinstance(cand.source_info.get("section"), str):
        try:
            from ventorum.airfoil import airfoil

            return float(airfoil(cand.source_info["section"]).max_thickness()[0])
        except (KeyError, ValueError):
            pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section polars of the strips
# ═══════════════════════════════════════════════════════════════════════════════

class _StripPolars:
    """Resolve the section polar of each strip; memo over the conditions of one call.

    A dict of polars maps a strip index, a surface name, an airfoil name or
    ``"default"`` to a polar. A key that matches none of these raises a
    ``ValueError``. A strip without an entry uses the section airfoil of the
    geometry; ``last_fallback`` flags these strips of the last
    :meth:`resolve` call, and :meth:`fallback_note` gives the trust note.
    """

    def __init__(self, polars: Any, aircraft: Aircraft) -> None:
        if polars is None:
            raise ValueError("polars must be a PolarTable, a SectionPolarSource, an airfoil or a dict.")
        self.polars = polars
        self.aircraft = aircraft
        self._memo: dict[Any, TabulatedAirfoil] = {}
        self._keys_checked = False
        self.last_fallback = np.zeros(0, dtype=bool)

    def check_keys(self, lattice: VortexLattice) -> None:
        """Refuse dict keys that match no strip, surface or airfoil name.

        Raises
        ------
        ValueError
            If a key of the polar dict is not a strip index of *lattice*, a
            surface name, an airfoil name of a section or ``"default"``. The
            message lists the valid keys.
        """
        if self._keys_checked or not isinstance(self.polars, dict):
            return
        surfaces = {str(surf.name) for surf in self.aircraft.surfaces if surf.name}
        airfoils = {str(getattr(sec.airfoil, "name", "") or "")
                    for surf in self.aircraft.surfaces for sec in surf.sections}
        airfoils |= {str(getattr(_base_airfoil(self.aircraft, lattice, s), "name", "") or "")
                     for s in range(lattice.n_strips)}
        airfoils.discard("")
        n = int(lattice.n_strips)
        unknown = []
        for key in self.polars:
            if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
                if 0 <= int(key) < n:
                    continue
            elif isinstance(key, str) and (key == "default" or key in surfaces or key in airfoils):
                continue
            unknown.append(key)
        if unknown:
            raise ValueError(
                f"The polar dict keys {unknown!r} match no strip, surface or airfoil. Valid keys: "
                f"strip indices 0 to {n - 1}, surface names {sorted(surfaces)}, airfoil names "
                f"{sorted(airfoils)} and 'default'."
            )
        self._keys_checked = True

    def _entry(self, lattice: VortexLattice, s: int) -> tuple[Any, bool]:
        """Return the polar of strip *s* and True when it falls back to the section airfoil."""
        if not isinstance(self.polars, dict):
            return self.polars, False
        surf = self.aircraft.surfaces[int(lattice.strip_surface[s])]
        base = _base_airfoil(self.aircraft, lattice, s)
        for key in (s, surf.name, getattr(base, "name", None), "default"):
            if key is not None and key in self.polars:
                return self.polars[key], False
        return base, True

    def _candidate(self, lattice: VortexLattice, s: int) -> Any:
        return self._entry(lattice, s)[0]

    def fallback_note(self, lattice: VortexLattice, strips: np.ndarray) -> str | None:
        """Return the trust note of the strips of the last resolve without a dict entry, or None."""
        flags = self.last_fallback
        if flags.size == 0 or not np.any(flags):
            return None
        names = sorted({str(self.aircraft.surfaces[int(lattice.strip_surface[s])].name)
                        for s in np.asarray(strips)[flags]})
        return (
            f"{int(np.sum(flags))} of {len(flags)} strips (surfaces {', '.join(names)}) have no entry "
            "in the polar dict; they use the section airfoil of the geometry."
        )

    def resolve(
        self,
        lattice: VortexLattice,
        strips: np.ndarray,
        condition: FlightCondition,
        ground: GroundPlane | None,
    ) -> tuple[list[Any], np.ndarray, np.ndarray]:
        """Return the polar of each strip, the flags "polar uses the height" and the strip h/c [-]."""
        self.check_keys(lattice)
        n = len(strips)
        self.last_fallback = np.zeros(n, dtype=bool)
        out: list[Any] = [None] * n
        uses_height = np.zeros(n, dtype=bool)
        h_over_c = np.full(n, np.nan)
        jobs: dict[Any, tuple[Any, Any, float, tuple | None, float | None]] = {}
        pending: list[tuple[int, list[tuple[Any, float]]]] = []
        for i, s in enumerate(strips):
            chord = float(lattice.chord[s])
            re_s = strip_reynolds_number(condition, chord)
            flap_deg, hinge = _strip_control(lattice, self.aircraft, s)
            if ground is not None:
                h_over_c[i] = float(ground.height(lattice.qc_mid[s][None, :])[0]) / chord
            cand, self.last_fallback[i] = self._entry(lattice, s)
            if isinstance(cand, PolarTable):
                out[i], uses_height[i] = self._from_table(cand, re_s, flap_deg, hinge, h_over_c[i], ground)
            elif isinstance(cand, (TabulatedAirfoil, LinearAirfoil)):
                out[i] = cand if flap_deg == 0.0 else deflected_airfoil(cand, np.radians(flap_deg), hinge)
            elif callable(getattr(cand, "polar", None)):
                with_height = ground is not None and source_supports_ground(cand)
                uses_height[i] = with_height
                flap = None if flap_deg == 0.0 else (hinge, SOURCE_Y_HINGE, flap_deg)
                h_arg = round(float(h_over_c[i]), 6) if with_height else None
                keys = []
                for spec, w in _strip_sections(self.aircraft, lattice, s):
                    section, skey = _source_section(spec)
                    key = (id(cand), skey, round(re_s, 3), flap, h_arg)
                    if key not in self._memo:
                        jobs[key] = (cand, section, re_s, flap, h_arg)
                    keys.append((key, w))
                pending.append((i, keys))
            else:
                raise TypeError(
                    "Each strip polar must be a PolarTable, a SectionPolarSource, a TabulatedAirfoil "
                    f"or a LinearAirfoil; got {type(cand).__name__}."
                )
        self._run_jobs(jobs)
        for i, keys in pending:
            if len(keys) == 1:
                out[i] = self._memo[keys[0][0]]
            else:
                (k1, _), (k2, w2) = keys
                out[i] = blend_tabulated(self._memo[k1], self._memo[k2], w2)
        return out, uses_height, h_over_c

    @staticmethod
    def _from_table(
        table: PolarTable,
        re_s: float,
        flap_deg: float,
        hinge: float | None,
        h_over_c: float,
        ground: GroundPlane | None,
    ) -> tuple[TabulatedAirfoil, bool]:
        if flap_deg != 0.0:
            x_table = (table.source_info.get("options") or {}).get("x_hinge")
            if x_table is not None and abs(float(x_table) - float(hinge)) > 1e-9:
                raise ValueError(
                    f"The polar table has its flap hinge at x/c = {float(x_table):g}, but the control "
                    f"surface has its hinge at x/c = {float(hinge):g}. Build the table with the hinge "
                    "of the control."
                )
        if table.h_over_c is None:
            return table.at(Re=re_s, flap_deg=flap_deg), False
        if ground is None:
            raise ValueError(
                "The polar table has a height axis, but the condition is in free air "
                "(condition.h is None). Use a free-air table."
            )
        return table.at(Re=re_s, flap_deg=flap_deg, h_over_c=float(h_over_c)), True

    def _run_jobs(self, jobs: dict) -> None:
        """Run the source calls that are not in the memo (in parallel threads)."""
        if not jobs:
            return

        def run(item):
            key, (src, section, re_s, flap, h_arg) = item
            kwargs: dict[str, Any] = {"flap": flap}
            if h_arg is not None:
                kwargs["height"] = h_arg
            return key, src.polar(section=section, Re=re_s, alphas_deg=SOURCE_ALPHAS_DEG, **kwargs)

        items = list(jobs.items())
        workers = max(1, min(len(items), (os.cpu_count() or 2) - 1))
        if workers == 1:
            results = [run(item) for item in items]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, items))
        for key, polar in results:
            if len(np.asarray(polar.alpha)) < 2:
                raise ValueError(
                    f"The polar source gave fewer than 2 converged angles for a strip ({polar.name})."
                )
            self._memo[key] = polar


# ═══════════════════════════════════════════════════════════════════════════════
# Section relation of the lattice strips
# ═══════════════════════════════════════════════════════════════════════════════

def _section_frame(
    pts: np.ndarray, origin: np.ndarray, cdir: np.ndarray, nrm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the chordwise and normal coordinates [m] of points in the strip section plane."""
    d = pts - origin
    return d @ cdir, d @ nrm


def _vlm_section_lines(
    lattice: VortexLattice,
    strips: np.ndarray,
    ground: GroundPlane | None,
    with_images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lift slope [1/rad] and zero-lift angle [rad] of each VLM strip section.

    The chordwise bound vortices and control points of the strip form a 2D
    discrete-vortex section (thin-airfoil theory, small angles). The
    boundary normals give the panel angles ``theta_k`` (camber and flap).
    Two right-hand sides with one factorization give ``Gamma_alpha`` (unit
    angle) and ``Gamma_theta`` (the panel angles): the slope is
    ``a = 2 sum(Gamma_alpha) / c`` and the zero-lift angle is
    ``alpha0 = -sum(Gamma_theta) / sum(Gamma_alpha)``. With
    ``with_images`` the section has its image vortices in the ground plane.

    Returns
    -------
    tuple of numpy.ndarray
        Lift slope [1/rad], zero-lift angle [rad] and the zero-lift angle
        of the clean strip [rad]. The points of the section are the lattice
        points before the turn of a chordwise flap: the section is the
        linear model, with the flap in the boundary normals (the lattice
        itself has the turned points). The clean strip has the panel normals
        before the flap turn. Without a mean line, all its panels have the
        strip normal: a uniform panel angle, so this angle is its zero-lift
        angle for any panel count, also with images. On a mean line, the
        panel angles follow the camber, and a third right-hand side gives
        the zero-lift angle of the clean section.
    """
    a_out = np.empty(len(strips))
    a0_out = np.empty(len(strips))
    clean_out = np.empty(len(strips))
    clean = lattice.geom_cache.get("flap_points_clean")
    for i, s in enumerate(strips):
        pidx = np.flatnonzero(lattice.panel_strip == s)
        cdir = lattice.chord_dir[s]
        nrm = lattice.normal[s]
        origin = 0.5 * (lattice.le_left[s] + lattice.le_right[s])
        # The section is linear (thin-airfoil theory): the points before the
        # turn of a chordwise flap, the flap in the boundary normals.
        pa, pb, pc = clean if clean is not None else (lattice.a, lattice.b, lattice.cp)
        q = 0.5 * (pa[pidx] + pb[pidx])
        p = pc[pidx]

        xq, zq = _section_frame(q, origin, cdir, nrm)
        xp, zp = _section_frame(p, origin, cdir, nrm)
        rx = xp[:, None] - xq[None, :]
        rz = zp[:, None] - zq[None, :]
        M = -rx / (2.0 * np.pi * (rx * rx + rz * rz))
        if ground is not None and with_images[i]:
            xi, zi = _section_frame(ground.reflect_points(q), origin, cdir, nrm)
            rxi = xp[:, None] - xi[None, :]
            rzi = zp[:, None] - zi[None, :]
            M = M + rxi / (2.0 * np.pi * (rxi * rxi + rzi * rzi))
        nb = lattice.normal_bc[pidx]
        theta = np.arctan2(-(nb @ cdir), nb @ nrm)
        rhs = [-np.ones(len(pidx)), theta]
        on_mean_line = (
            lattice.normal_bc_clean is not None
            and lattice.geometric_camber is not None
            and bool(lattice.geometric_camber[s])
        )
        if on_mean_line:
            nb_c = lattice.normal_bc_clean[pidx]
            rhs.append(np.arctan2(-(nb_c @ cdir), nb_c @ nrm))
        # One factorization for all right-hand sides.
        g = np.linalg.solve(M, np.column_stack(rhs))
        g_alpha, g_theta = g[:, 0].sum(), g[:, 1].sum()
        a_out[i] = 2.0 * g_alpha / float(lattice.chord[s])
        a0_out[i] = -g_theta / g_alpha
        if on_mean_line:
            clean_out[i] = -g[:, 2].sum() / g_alpha
        else:
            nb_clean = lattice.normal_bc_strip[s]
            clean_out[i] = float(np.arctan2(-(nb_clean @ cdir), nb_clean @ nrm))
    return a_out, a0_out, clean_out


def _section_flap_shift(
    lattice: VortexLattice,
    s: int,
    x_hinge: float,
    ground: GroundPlane | None,
    n_panels: int = GROUND_FLAP_REFERENCE_PANELS,
) -> float:
    """Return the zero-lift shift per unit flap angle of a converged 2D strip section.

    The section is the mean chord line of strip *s* with *n_panels* lumped
    vortices (vortex at the quarter point and control point at the
    three-quarter point of each panel; thin-airfoil theory, Katz and
    Plotkin 2001), a panel edge at the hinge and a unit panel angle behind
    it. With *ground* the vortices have their images in the ground plane.
    The panel angle has the sign of the angles ``theta_k`` of
    :func:`_vlm_section_lines` (positive for a trailing-edge-up turn), so in
    free air the shift tends to ``+tau(x_hinge)`` of Glauert as the panel
    count grows. The coupling uses only the ratio of two shifts.

    Parameters
    ----------
    lattice : VortexLattice
        The lattice.
    s : int
        Strip index.
    x_hinge : float
        Hinge position as a fraction of the chord [-], in (0, 1).
    ground : GroundPlane or None
        Ground plane for the images, or None for free air.
    n_panels : int
        Chordwise panel count of the section [-].

    Returns
    -------
    float
        Change of the zero-lift angle per unit flap angle [rad/rad].

    Raises
    ------
    ValueError
        If *ground* is given and the trailing edge of strip *s* is at or
        below the ground plane (height <= 0).
    """
    n1 = int(np.clip(round(n_panels * x_hinge), 1, n_panels - 1))
    edges = np.concatenate([np.linspace(0.0, x_hinge, n1 + 1), np.linspace(x_hinge, 1.0, n_panels - n1 + 1)[1:]])
    x0, dx = edges[:-1], np.diff(edges)
    le = 0.5 * (lattice.le_left[s] + lattice.le_right[s])
    te = 0.5 * (lattice.te_left[s] + lattice.te_right[s])
    if ground is not None:
        h_te = float(ground.height(te))
        if h_te <= 0.0:
            raise ValueError(
                f"Strip {s} trailing edge is at or below the ground (height {h_te:.4g} m <= 0); "
                "increase the height or reduce the angle."
            )
    cdir = lattice.chord_dir[s]
    nrm = lattice.normal[s]
    q = le + (x0 + 0.25 * dx)[:, None] * (te - le)
    p = le + (x0 + 0.75 * dx)[:, None] * (te - le)
    xq, zq = _section_frame(q, le, cdir, nrm)
    xp, zp = _section_frame(p, le, cdir, nrm)
    rx = xp[:, None] - xq[None, :]
    rz = zp[:, None] - zq[None, :]
    M = -rx / (2.0 * np.pi * (rx * rx + rz * rz))
    if ground is not None:
        xi, zi = _section_frame(ground.reflect_points(q), le, cdir, nrm)
        rxi = xp[:, None] - xi[None, :]
        rzi = zp[:, None] - zi[None, :]
        M = M + rxi / (2.0 * np.pi * (rxi * rxi + rzi * rzi))
    theta = (x0 >= x_hinge - 1e-12).astype(float)
    g = np.linalg.solve(M, np.column_stack([-np.ones(n_panels), theta]))
    return float(-g[:, 1].sum() / g[:, 0].sum())


def _thin_airfoil_zero_lift(
    lattice: VortexLattice,
    aircraft: Aircraft,
    strips: np.ndarray,
    alpha0: np.ndarray,
    alpha0_clean: np.ndarray,
    ground: GroundPlane | None,
    with_images: np.ndarray,
) -> np.ndarray:
    """Return the zero-lift angle [rad] of the thin-airfoil model of each VLM strip.

    A strip without a deflected control keeps the zero-lift angle of the
    lattice section, so its lift offset is zero. A strip with a control has
    ``alpha0_clean - tau(x_h) phi kappa``: the clean lattice section, the
    Glauert flap shift of the polar flap (hinge ``x_h`` [-], deflection
    ``phi`` [rad]) and the ground factor ``kappa`` of the flap shift. In
    free air (or with a free-air polar) ``kappa`` is 1. With a polar that
    holds the ground height, ``kappa`` is the ratio of the flap shifts of
    the converged 2D section with and without images
    (:func:`_section_flap_shift`).

    Parameters
    ----------
    lattice : VortexLattice
        The lattice.
    aircraft : Aircraft
        The geometry (controls of the strips).
    strips : numpy.ndarray
        Independent strip indices.
    alpha0 : numpy.ndarray
        Zero-lift angles of the lattice sections [rad].
    alpha0_clean : numpy.ndarray
        Zero-lift angles of the clean lattice sections [rad].
    ground : GroundPlane or None
        Ground plane of the case.
    with_images : numpy.ndarray
        True for the strips whose polar holds the ground height.

    Returns
    -------
    numpy.ndarray
        Zero-lift angles of the thin-airfoil models [rad].
    """
    out = np.asarray(alpha0, dtype=float).copy()
    free_shift: dict[float, float] = {}
    for i, s in enumerate(strips):
        flap_deg, hinge = _strip_control(lattice, aircraft, s)
        if flap_deg == 0.0 or hinge is None:
            continue
        kappa = 1.0
        if ground is not None and with_images[i] and 0.0 < float(hinge) < 1.0:
            key = round(float(hinge), 12)
            # BLAS on one thread: a threaded solve of this size is much slower.
            with blas_single_thread():
                if key not in free_shift:
                    free_shift[key] = _section_flap_shift(lattice, s, float(hinge), None)
                kappa = _section_flap_shift(lattice, s, float(hinge), ground) / free_shift[key]
        tau = flap_effectiveness(float(hinge))
        out[i] = float(alpha0_clean[i]) - tau * np.radians(flap_deg) * kappa
    return out


def _strip_span_axes(lattice: VortexLattice, strips: np.ndarray) -> np.ndarray:
    """Return the unit span axis ``n x c`` [-] of each strip, shape (n_strips, 3).

    The axis is normal to the section plane of the strip: the plane of its
    chord direction ``c`` and its normal ``n``. The 2D section relation of
    the strip and the section cut of the panel coupling use this plane. An
    axis within 1e-12 of +y is set to +y exactly, so a planar strip keeps
    the arithmetic of the global angle of attack.
    """
    axes = np.cross(lattice.normal[strips], lattice.chord_dir[strips])
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    planar = (np.abs(axes[:, 0]) < 1e-12) & (np.abs(axes[:, 2]) < 1e-12) & (axes[:, 1] > 0.0)
    axes[planar] = (0.0, 1.0, 0.0)
    return axes


def _section_onset_derivative(direction: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Return the change of the onset direction per unit section angle [1/rad].

    The section angle of a strip is the angle of the onset in its section
    plane, measured from the chord towards the strip normal. A change of
    this angle turns the in-plane part of the onset ``d_p = d - (d . e) e``
    about the span axis ``e``. Thus the onset changes by ``d_p x e`` per
    radian. The spanwise part of the onset does not change. For a planar
    strip (``e`` = +y) this is the derivative of the free-stream direction
    with respect to the angle of attack,
    ``cos(beta) [-sin(alpha), 0, cos(alpha)]``.

    Parameters
    ----------
    direction : numpy.ndarray
        Unit free-stream direction [-], shape (3,).
    axes : numpy.ndarray
        Unit span axes of the strips [-], shape (n_strips, 3).

    Returns
    -------
    numpy.ndarray
        Onset change per unit section angle [1/rad], shape (n_strips, 3).
        Its norm is the in-plane part of the unit onset; it is parallel to
        the section lift direction.
    """
    direction = np.asarray(direction, dtype=float)
    in_plane = direction[None, :] - (axes @ direction)[:, None] * axes
    return np.cross(in_plane, axes)


def _rhs_derivative(
    lattice: VortexLattice,
    model: str,
    umap: UnknownMap,
    V_inf: float,
    u_strip: np.ndarray,
    strips: np.ndarray,
) -> np.ndarray:
    """Return the right-hand-side change per unit correction of each strip.

    A correction ``delta_s`` lowers the section angle of strip s: the
    in-plane onset of its boundary condition turns about the strip span
    axis by ``-delta_s``. The onset changes by ``-delta_s u_s``, with
    ``u_s`` from :func:`_section_onset_derivative` (one row per strip of
    *strips*, [1/rad]; a single vector applies to all strips). The columns
    are ``d rhs / d delta_s``, shape (n_unknowns, n_strips).
    """
    u_strip = np.atleast_2d(np.asarray(u_strip, dtype=float))
    unk_strip = lattice.panel_strip[umap.unknown_panels]
    if model == "vlm":
        n_vec = lattice.normal_bc[umap.unknown_panels]
        sign = 1.0
    else:
        n_vec = _llt_strip_data(lattice, umap)["normal"]
        sign = -1.0
    if np.all(u_strip == u_strip[0]):
        # One onset change for all strips (a planar configuration).
        w = sign * V_inf * (n_vec @ u_strip[0])
    else:
        col_of = np.full(lattice.n_strips, -1)
        col_of[strips] = np.arange(len(strips))
        cols = col_of[unk_strip]
        w = np.zeros(len(unk_strip))
        own = cols >= 0
        w[own] = sign * V_inf * np.sum(n_vec[own] * u_strip[cols[own]], axis=1)
    B = np.zeros((umap.n, len(strips)))
    for col, s in enumerate(strips):
        mask = unk_strip == s
        B[mask, col] = w[mask]
    return B


# ═══════════════════════════════════════════════════════════════════════════════
# One condition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class _Factorization:
    """LU factors of the last system matrix and the number of factorizations."""

    matrix: np.ndarray | None = None
    lu: tuple | None = None
    count: int = 0

    def get(self, A: np.ndarray) -> tuple[tuple, bool]:
        """Return the LU factors of *A*; reuse them when *A* did not change."""
        if self.matrix is not None and self.matrix.shape == A.shape and np.array_equal(self.matrix, A):
            return self.lu, False  # type: ignore[return-value]
        self.lu = scipy.linalg.lu_factor(A)
        self.matrix = A.copy()
        self.count += 1
        return self.lu, True


def _not_converged_note(sol: dict, max_iter: int, tol: float) -> str:
    """Return the warning text of a coupling that did not converge."""
    return (
        f"The alpha-correction coupling did not converge in {max_iter} iterations "
        f"(largest strip Cl residual {sol['hist'][-1]:.2e}, tolerance {tol:.1e}). "
        "The result is not reliable; no branch search was run."
    )


def _alpha_cl_max(polar: Any) -> float | None:
    """Return the angle [rad] of the largest Cl of a tabulated polar, or None."""
    if not isinstance(polar, TabulatedAirfoil):
        return None
    alpha, cl, _, _ = polar.tables()
    return float(np.asarray(alpha)[int(np.argmax(np.asarray(cl)))])


def _solve_condition(
    solver: Any,
    model: str,
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    resolver: _StripPolars,
    fact: _Factorization,
    ref_point: np.ndarray,
    main_surface: int,
    delta0: np.ndarray | None,
    max_iter: int,
    tol: float,
    relaxation: float,
    warned: set[str] | None = None,
) -> SolverResult:
    """Solve one condition with the alpha-correction coupling."""
    t0 = time.perf_counter()
    S_ref, b_ref, c_ref = aircraft.S_ref, aircraft.b_ref, aircraft.c_ref
    V_inf = float(condition.V_inf)
    lattice = solver.build(aircraft, settings, condition, None, ref_point)
    ground, h_min, notes, wd = solver._case_setup(lattice, condition, settings, c_ref, None, ref_point)
    use_sym = getattr(settings, "use_symmetry", True)

    with solve_threads(lattice.n_panels):
        if model == "vlm":
            A, rhs0, umap = assemble_vlm(lattice, condition, ground, use_sym, wd, ref_point)
        else:
            A, rhs0, umap, _ = _assemble_llt_linear(lattice, condition, ground, use_sym, wd, ref_point)
        lu, factorized = fact.get(A)
        strips = np.flatnonzero(lattice.strip_is_right) if umap.symmetric else np.arange(lattice.n_strips)
        # The correction turns the onset in the section plane of each strip.
        u_strip = _section_onset_derivative(
            freestream_direction(condition.alpha, condition.beta), _strip_span_axes(lattice, strips),
        )
        B = _rhs_derivative(lattice, model, umap, V_inf, u_strip, strips)
        G = scipy.linalg.lu_solve(lu, B)
        gamma0 = scipy.linalg.lu_solve(lu, rhs0)

    n_ind = len(strips)
    unk_strip = lattice.panel_strip[umap.unknown_panels]
    scale = 2.0 / (V_inf * np.maximum(lattice.chord[strips], 1e-12))
    cl0 = np.array([np.sum(gamma0[unk_strip == s]) for s in strips]) * scale
    # cl_inv(delta) = cl0 - D delta: a correction lowers the lift of the strips.
    D = -np.array([np.sum(G[unk_strip == s, :], axis=0) for s in strips]) * scale[:, None]

    polars, uses_height, h_over_c = resolver.resolve(lattice, strips, condition, ground)
    fallback = resolver.fallback_note(lattice, strips)
    if fallback is not None:
        notes = [*notes, fallback]
    if ground is not None and model in ("vlm", "llt"):
        h_vals = h_over_c[np.isfinite(h_over_c)]
        if h_vals.size > 0 and float(np.min(h_vals)) < 0.30:
            for s in strips:
                flap_deg, hinge = _strip_control(lattice, aircraft, int(s))
                if flap_deg != 0.0 and hinge is not None:
                    cand = resolver._candidate(lattice, int(s))
                    if _strip_thickness_ratio(aircraft, lattice, int(s), cand) > 0.04:
                        if _THICK_GROUND_FLAP_NOTE not in notes:
                            notes.append(_THICK_GROUND_FLAP_NOTE)
                        break
    if model == "vlm":
        a_sec, alpha0, alpha0_clean = _vlm_section_lines(lattice, strips, ground, uses_height)
        alpha0_ta = _thin_airfoil_zero_lift(lattice, aircraft, strips, alpha0, alpha0_clean, ground, uses_height)
    else:
        a_sec = np.maximum(lattice.a0[strips], 1e-6)
        alpha0 = lattice.alpha_L0[strips].astype(float)
        # The strip airfoil of the lifting line already has the Glauert flap shift.
        alpha0_ta = alpha0.copy()
    # Lift of the lattice section minus the lift of its thin-airfoil model.
    # The polar departs from thin-airfoil theory by cl_visc - cl_TA, and
    # this departure is added to the lift of the lattice section. The
    # offset is zero on a strip without a deflected control.
    cl_offset = a_sec * (alpha0 - alpha0_ta)

    def evaluate(delta: np.ndarray):
        cl_inv = cl0 - D @ delta
        alpha_v = alpha0 + cl_inv / a_sec + delta
        cl_v = np.array([float(section_cl(p, a)) for p, a in zip(polars, alpha_v)])
        m = np.array([float(section_cl_slope(p, a)) for p, a in zip(polars, alpha_v)])
        return cl_inv - (cl_v - cl_offset), cl_inv, alpha_v, m

    def newton(start: np.ndarray):
        delta = start.copy()
        r, cl_inv, alpha_v, m = evaluate(delta)
        hist = [float(np.max(np.abs(r)))]
        conv = hist[-1] < tol
        it = 0
        while not conv and it < max_iter:
            it += 1
            J = -(1.0 - (m / a_sec))[:, None] * D - np.diag(m)
            f0 = 0.5 * float(r @ r)
            accepted = False
            try:
                step = relaxation * np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                step = None
            if step is not None and np.all(np.isfinite(step)):
                lam = 1.0
                while lam >= 1.0 / 1024.0:
                    trial = evaluate(delta + lam * step)
                    if 0.5 * float(trial[0] @ trial[0]) <= (1.0 - 1e-4 * lam) * f0:
                        delta = delta + lam * step
                        r, cl_inv, alpha_v, m = trial
                        accepted = True
                        break
                    lam *= 0.5
            if not accepted:
                # Damped fixed-point step on the inviscid response.
                try:
                    target = delta + np.linalg.solve(D, r)
                except np.linalg.LinAlgError:
                    target = delta + 0.1 * r
                for omega in (0.5, 0.25, 0.1, 0.05, 0.02):
                    trial_delta = delta + omega * relaxation * (target - delta)
                    trial = evaluate(trial_delta)
                    if 0.5 * float(trial[0] @ trial[0]) < f0:
                        delta = trial_delta
                        r, cl_inv, alpha_v, m = trial
                        accepted = True
                        break
            if not accepted:
                break
            hist.append(float(np.max(np.abs(r))))
            conv = hist[-1] < tol
        return {"delta": delta, "alpha_v": alpha_v, "m": m, "hist": hist, "converged": bool(conv)}

    start = np.zeros(n_ind)
    if delta0 is not None and np.asarray(delta0).shape == (n_ind,):
        start = np.asarray(delta0, dtype=float).copy()
    primary = newton(start)

    # The airfoils change, so the cache starts empty, except the entries of
    # the flap geometry: the leg-force rules of a lattice with chordwise
    # controls come from the lattice before the flap turn and cannot be
    # found again from its points.
    keep = ("flap_hinges", "flap_field", "flap_points_clean", "leg_force_rules")
    loads_lattice = dataclasses.replace(
        lattice, airfoils=list(lattice.airfoils),
        geom_cache={k: lattice.geom_cache[k] for k in keep if k in lattice.geom_cache}, kernel_cache=None,
    )
    for i, s in enumerate(strips):
        loads_lattice.airfoils[s] = polars[i]
    left = np.flatnonzero(~lattice.strip_is_right) if umap.symmetric else np.empty(0, dtype=int)
    for s in left:
        loads_lattice.airfoils[s] = loads_lattice.airfoils[lattice.strip_mirror[s]]

    def full(values: np.ndarray, dtype=float) -> np.ndarray:
        out = np.empty(lattice.n_strips, dtype=dtype)
        out[strips] = values
        if umap.symmetric:
            out[left] = out[lattice.strip_mirror[left]]
        return out

    def loads_of(sol: dict):
        gamma_unk = gamma0 + G @ sol["delta"]
        return compute_loads(
            loads_lattice, gamma_unk[umap.panel_column], condition, S_ref, b_ref, c_ref,
            ground=ground, ref_point=ref_point, alpha_eff_strip=full(sol["alpha_v"]), wake_dir=wd,
        ), gamma_unk

    primary_loads = loads_of(primary)
    branches: list[tuple[dict, Any]] = []
    post_stall = primary["m"] <= 0.0
    if primary["converged"] and np.any(post_stall):
        # The stalled start puts every strip past the angle of its Cl maximum,
        # the attached start puts every strip below it.
        margin = np.radians(STALL_START_MARGIN_DEG)
        up_shift = np.zeros(n_ind)
        down_shift = np.zeros(n_ind)
        for i, p in enumerate(polars):
            a_max = _alpha_cl_max(p)
            if a_max is not None:
                a_v = float(primary["alpha_v"][i])
                up_shift[i] = max(0.0, a_max + margin - a_v)
                down_shift[i] = max(0.0, a_v - (a_max - margin))
        seeds = [primary["delta"] + up_shift, primary["delta"] - down_shift]
        for seed in seeds:
            if np.allclose(seed, primary["delta"], atol=1e-9):
                continue
            other = newton(seed)
            if not other["converged"]:
                continue
            other_loads = loads_of(other)
            cls = [primary_loads[0].totals.CL] + [b_l[0].totals.CL for _, b_l in branches]
            if all(abs(other_loads[0].totals.CL - c) > BRANCH_CL_DIFFERENCE for c in cls):
                branches.append((other, other_loads))
    non_unique = bool(branches)

    sweep_deg = quarter_chord_sweep_deg(lattice)
    main_strips = main_surface_strip_count(lattice, main_surface)
    limits = polar_limits(loads_lattice.airfoils)
    solver_name = f"viscous_{model}"

    def make_result(sol: dict, loads_gamma, is_primary: bool) -> SolverResult:
        loads, gamma_unk = loads_gamma
        totals = loads.totals
        res_notes = list(notes)
        if not sol["converged"]:
            res_notes.append(_not_converged_note(sol, max_iter, tol))
        if non_unique:
            res_notes.append(
                f"Non-unique post-stall solution: {len(branches) + 1} converged branches differ in CL by "
                f"more than {BRANCH_CL_DIFFERENCE:g}; see details['branches']."
            )
        polar = None
        if limits is not None:
            cl_strip = np.empty(lattice.n_strips)
            ae_strip = np.empty(lattice.n_strips)
            for surf, sw in zip(lattice.surfaces, loads.spanwise):
                cl_strip[surf.strips] = sw.Cl
                ae_strip[surf.strips] = sw.alpha_eff
            polar = polar_stats(limits, cl_strip, ae_strip)
        totals.trust = evaluate_aerodynamic_trust(
            AR=totals.AR,
            surfaces=None,
            condition=condition,
            spanwise_list=loads.spanwise,
            CL=totals.CL,
            CDi=totals.CDi,
            CD_total=totals.CD_total,
            converged=sol["converged"],
            n_panels=main_strips,
            max_sweep_rad=np.radians(sweep_deg),
            solver_type="vlm" if model == "vlm" else "linear",
            h_over_c=(h_min / c_ref) if (h_min is not None and c_ref) else None,
            n_chord=lattice.n_chord,
            notes=res_notes,
            max_chord_over_c=float(np.max(lattice.chord) / c_ref) if c_ref else 1.0,
            polar=polar,
            CY=totals.CY,
            CY_farfield=totals.CY_farfield,
            Cn_trefftz=totals.Cn_trefftz,
            post_stall_branch=non_unique,
        )
        flags = np.where(sol["m"] <= 0.0, "post-stall, non-unique", "attached")
        res = SolverResult(
            spanwise=loads.spanwise,
            totals=totals,
            solver_type=solver_name,
            converged=sol["converged"],
            iterations=len(sol["hist"]) - 1,
            residual_history=list(sol["hist"]),
            condition=condition,
            execution_time=time.perf_counter() - t0,
            symmetry_used=umap.symmetric,
            b_ref=float(b_ref),
            c_ref=float(c_ref),
        )
        res.details = {
            "lattice": loads_lattice,
            "loads": loads,
            "ground": ground,
            "h_min": h_min,
            "notes": res_notes,
            "gamma": gamma_unk[umap.panel_column],
            "wake_dir": wd,
            "ref_point": np.asarray(ref_point, dtype=float),
            "delta": full(sol["delta"]),
            "alpha_eff": full(sol["alpha_v"]),
            "alpha0_section": full(alpha0),
            "alpha0_thin_airfoil": full(alpha0_ta),
            "lift_slope_section": full(a_sec),
            "strip_h_over_c": full(h_over_c),
            "post_stall": bool(np.any(sol["m"] <= 0.0)),
            "post_stall_strips": full(sol["m"] <= 0.0, dtype=bool),
            "strip_flags": list(full(flags, dtype=object)),
            "non_unique": non_unique,
            "factorized": bool(factorized) if is_primary else False,
            "viscous_iterations": len(sol["hist"]) - 1,
            "viscous_residual_history": list(sol["hist"]),
        }
        return res

    result = make_result(primary, primary_loads, True)
    if not primary["converged"]:
        warnings.warn(_not_converged_note(primary, max_iter, tol), RuntimeWarning, stacklevel=3)
    if non_unique:
        result.details["branches"] = [result] + [make_result(sol, bl, False) for sol, bl in branches]
    mesh = getattr(lattice, "mesh_warnings", ())
    for msg in notes:
        if msg in mesh:
            continue  # Issued by build_lattice or by the lattice cache.
        if warned is not None and msg == _THICK_GROUND_FLAP_NOTE:
            if msg in warned:
                continue
            warned.add(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def solve_viscous(
    aircraft: Aircraft | LiftingSurface,
    conditions: FlightCondition | Sequence[FlightCondition],
    settings: SolverSettings,
    polars: Any,
    *,
    model: Literal["vlm", "llt", "panel3d"] = "vlm",
    max_iter: int = 50,
    tol: float = 1e-6,
    relaxation: float = 1.0,
) -> SolverResult | list[SolverResult]:
    """Solve the viscous-inviscid coupling by angle-of-attack correction.

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
        The geometry.
    conditions : FlightCondition or sequence of FlightCondition
        One condition or a sweep. The density [kg/m^3] and the speed [m/s]
        of each condition set the strip Reynolds numbers (dynamic
        viscosity :data:`ventorum.core.constants.MU_SL` [Pa s]).
    settings : SolverSettings
        Discretisation settings of the lattice.
    polars : PolarTable, SectionPolarSource, TabulatedAirfoil, LinearAirfoil or dict
        Section polars. A table gives the polar at the strip Reynolds
        number, flap deflection [deg] and, with a height axis, the strip
        height h/c [-]. A source runs at the strip Reynolds number with the
        flap ``(hinge_x_c, 0.5, deflection_deg)`` of the strip control and,
        when it supports ground effect, the strip height; the strip section
        comes from ``WingSection.shape`` or a NACA airfoil name (no default
        section). A single airfoil is used on every strip, with the Glauert
        flap shift on a deflected strip. A dict maps a strip index, a
        surface name, an airfoil name or ``"default"`` to one of these.
        A key that matches none of them raises ``ValueError``. Strips
        without an entry use their section airfoil, with a trust note.
    model : {"vlm", "llt", "panel3d"}
        Potential-flow solver. The panel method matches surface-pressure
        strip lift and also closes the pressure Kutta residual when selected.
    max_iter : int
        Largest number of Newton iterations per condition.
    tol : float
        Tolerance on the largest strip Cl residual [-].
    relaxation : float
        Under-relaxation factor of the Newton step, in (0, 1]. Default 1.

    Returns
    -------
    SolverResult or list of SolverResult
        One result per condition, in the input order. ``totals.CDp`` is the
        profile drag from the polars at the solved angles and ``totals.CDi``
        the induced drag. ``details`` holds ``delta`` [rad] and
        ``alpha_eff`` [rad] (the viscous section angle of each strip),
        ``alpha0_section`` [rad] and ``lift_slope_section`` [1/rad] (the
        section relation of the lattice), ``alpha0_thin_airfoil`` [rad]
        (the zero-lift angle of the thin-airfoil model of each strip; it
        differs from ``alpha0_section`` only on a vortex-lattice strip with
        a deflected control), ``strip_h_over_c`` [-],
        ``post_stall_strips``, ``strip_flags``, ``non_unique``,
        ``branches`` (when a post-stall case has more than one converged
        solution), ``factorized`` (True when this condition needed a new
        factorization) and ``factorization_count`` (the number of LU
        factorizations of the system matrix in the whole call).

    Notes
    -----
    A sweep runs from the angle nearest to zero upward and from there
    downward; each condition starts from the corrections of its converged
    neighbour. The factorization is reused when the system matrix does not
    change between conditions (for example ``wake_alignment="body"`` out
    of ground effect). With the default free-stream wake the matrix
    depends on the angle of attack, so a sweep factorizes once per
    condition. The method runs on the CPU.

    For ``panel3d``, free-air conditions share the fixed panel wake and
    one body factorization. Strip lift comes from surface pressure. The
    section relation comes from a 2D Morino solve of the mesh section.
    Ground ``auto`` uses a direct solve per configuration; explicit
    ``krylov`` and GPU requests are refused. ``solve_count`` counts body
    triangular-solve calls; ``solve_rhs_count`` counts their columns.

    On a vortex-lattice strip with a deflected control the polar is read
    relative to its thin-airfoil model: the target lift is the polar lift
    minus ``lift_slope_section * (alpha0_section - alpha0_thin_airfoil)``.
    A polar that follows thin-airfoil theory (slope 2 pi, Glauert flap
    shift) then gives the inviscid vortex-lattice result at the same
    chordwise panel count, not the mesh-converged flap effect.
    """
    model_str = str(model).lower()
    if model_str not in ("vlm", "llt", "panel3d"):
        raise ValueError(f"model must be 'vlm', 'llt' or 'panel3d', got {model!r}.")
    if not 0.0 < float(relaxation) <= 1.0:
        raise ValueError(f"relaxation must lie in (0, 1], got {relaxation}.")
    if int(max_iter) < 0:
        raise ValueError(f"max_iter must not be negative, got {max_iter}.")
    if not float(tol) > 0.0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if model_str == "panel3d":
        return _solve_panel_viscous(aircraft, conditions, settings, polars,
                                    int(max_iter), float(tol), float(relaxation))

    aircraft_obj = as_aircraft(aircraft)
    validate_aircraft(aircraft_obj)
    aircraft_obj.compute_reference_values()
    rp = aircraft_obj.moment_reference()
    main_surface = aircraft_obj.main_surface_index()

    is_seq = isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes))
    cond_list = list(conditions) if is_seq else [conditions]
    if not cond_list:
        return []

    solver = make_solver("vlm" if model_str == "vlm" else "linear")
    resolver = _StripPolars(polars, aircraft_obj)
    fact = _Factorization()
    warned: set[str] = set()
    results: list[SolverResult | None] = [None] * len(cond_list)
    alphas = np.array([float(c.alpha) for c in cond_list])
    up = [int(i) for i in np.argsort(alphas, kind="stable") if alphas[i] >= 0.0]
    down = [int(i) for i in np.argsort(-alphas, kind="stable") if alphas[i] < 0.0]

    def run(order: list[int], start: np.ndarray | None) -> np.ndarray | None:
        delta = start
        for i in order:
            res = _solve_condition(
                solver, model_str, aircraft_obj, cond_list[i], settings, resolver, fact, rp,
                main_surface, delta, int(max_iter), float(tol), float(relaxation), warned,
            )
            results[i] = res
            if res.converged:
                delta = _independent(res)
        return delta

    first_up = run(up[:1], None) if up else None
    run(up[1:], first_up)
    run(down, first_up)
    for res in results:
        res.details["factorization_count"] = fact.count  # type: ignore[union-attr]
    return results if is_seq else results[0]  # type: ignore[return-value]


def _independent(res: SolverResult) -> np.ndarray:
    """Return the corrections of the independent strips of a result."""
    lattice = res.details["lattice"]
    delta = res.details["delta"]
    if res.symmetry_used:
        return np.asarray(delta[np.flatnonzero(lattice.strip_is_right)], dtype=float)
    return np.asarray(delta, dtype=float)


def _panel_residual(x, velocity0, dv, speed, incidence, weights, area,
                    zero, slope, polars, offset, up, lo, pressure, onset_sq=None):
    """Return the coupled residual and its analytic Jacobian.

    Body velocities and their columns are in m/s; speed is in m/s. The
    first unknowns are strip angles [rad], followed by optional wake
    strengths [m^2/s]. Cl and the trailing-edge Cp jump are dimensionless.
    The chain rule uses dCp/dx = -2 V dot (dV/dx) / V_inf^2. With body
    rates, onset_sq holds the squared onset speed |V_k|^2 [m^2/s^2] of each
    panel and Cp = (|V_k|^2 - |V|^2) / V_inf^2 (see
    :func:`ventorum.panel3d.post.compute_surface_velocity_and_cp`); None
    means no rates.
    """
    ns = len(area)
    n = len(weights)
    velocity = velocity0 + np.einsum("ijk,k->ij", dv, x)
    if onset_sq is None:
        cp = 1 - np.sum(velocity**2, axis=1) / speed**2
    else:
        cp = (onset_sq - np.sum(velocity**2, axis=1)) / speed**2
    dcp = -2 * np.einsum("ij,ijk->ik", velocity, dv) / speed**2
    cl = (incidence @ (weights * cp[:n])) / area
    dc = (incidence @ (weights[:, None] * dcp[:n])) / area[:, None]
    alpha_v = zero + cl / slope + x[:ns]
    clv = np.array([float(section_cl(p, a)) for p, a in zip(polars, alpha_v)])
    m = np.array([float(section_cl_slope(p, a)) for p, a in zip(polars, alpha_v)])
    residual = cl - clv + offset
    jac = (1 - m / slope)[:, None] * dc
    jac[np.arange(ns), np.arange(ns)] -= m
    jump = cp[up] - cp[lo]
    if pressure:
        residual = np.concatenate([residual, jump])
        jac = np.vstack([jac, dcp[up] - dcp[lo]])
        dc = np.vstack([dc, dcp[up] - dcp[lo]])
    return dict(r=residual, jac=jac, dc=dc, cl=cl, alpha_v=alpha_v, m=m,
                cp=cp, velocity=velocity, jump=jump)


class _PanelCoupling:
    """Pressure-based strip correction and simultaneous pressure Kutta closure.

    The onset change is the linear transpiration equivalent of an angle
    reduction (Lighthill 1958). Source and doublet responses use the Morino
    system (Katz and Plotkin 2001). Wake shifts enforce the pressure Kutta
    condition of Kinnas and Hsin (1992). All angles are in rad.
    """

    def __init__(self, polars, max_iter, tol, relaxation):
        self.polars = polars
        self.max_iter = max_iter
        self.tol = tol
        self.relaxation = relaxation
        self.states = {}
        self.previous = None
        self.first_positive = None
        self.negative_started = False
        self.last_factorization_count = 0

    def prepare(self, solver, geometry, aircraft, settings, conditions):
        """Prepare strip metadata and 2D Morino references before assembly."""
        import scipy.sparse

        from ventorum.airfoil import airfoil
        from ventorum.solvers.panel3d import _build_geometry, _panel_section_lines

        # The lattice supplies only section/polar metadata, never aerodynamic
        # loads or section slopes. Its strips have the same reporting stations.
        self.lattice = make_solver("vlm").build(
            aircraft, dataclasses.replace(settings, n_chord=1), conditions[0], None, solver.ref_point,
        )
        self.strips = (np.flatnonzero(self.lattice.strip_is_right) if solver.mesh.half
                       else np.arange(self.lattice.n_strips))
        self.maps = []
        areas = []
        for (right, left), meta in zip(geometry.strip_maps, geometry.geom_meta):
            self.maps.extend((left or []) + right)
            areas.extend(np.concatenate([meta["area"][::-1], meta["area"]])
                         if left is not None else meta["area"])
        self.area = np.asarray(areas)
        self.slope, self.zero, self.moment = _panel_section_lines(
            geometry.mesh, self.maps, self.lattice, self.strips,
        )
        self.geometry = geometry
        self.clean_geometry = None
        self.clean_zero = self.zero.copy()
        if any(s.controls for s in aircraft.surfaces):
            clean = dataclasses.replace(aircraft, surfaces=[dataclasses.replace(s, controls=[]) for s in aircraft.surfaces])
            coords = [[airfoil(sec.shape if sec.shape is not None else "0012").coords
                       for sec in s.sections] for s in clean.surfaces]
            geom = _build_geometry(clean, settings, coords, settings.n_chord or 40,
                                   settings.panel_chord_spacing, solver.mesh.half, settings.panel_scale_root_thickness)
            clean_maps = [ids for right, left in geom.strip_maps for ids in (left or []) + right]
            _, self.clean_zero, _ = _panel_section_lines(geom.mesh, clean_maps, self.lattice, self.strips)
            self.clean_geometry = geom, clean_maps
        self.resolver = _StripPolars(self.polars, aircraft)
        self.aircraft = aircraft
        row = np.repeat(np.arange(len(self.maps)), [len(ids) for ids in self.maps])
        col = np.concatenate(self.maps)
        self.incidence = scipy.sparse.csr_matrix(
            (np.ones(len(col)), (row, col)), shape=(len(self.maps), len(solver.centroids)),
        )
        self.owner = np.empty(len(col), dtype=int)
        self.owner[col] = row
        # Span axes of the reporting sections. A strip whose axis is not +y
        # (dihedral, fin) needs the y onset response of its panels.
        self.axes = _strip_span_axes(self.lattice, self.strips)
        self.lateral = bool(np.any(self.axes != np.array([0.0, 1.0, 0.0])))
        solver._set_viscous_strips(self.maps, lateral=self.lateral)
        solver._viscous_evaluator = self.evaluate

    def full(self, values):
        """Expand independent values to public strip order (same units)."""
        values = np.asarray(values)
        result = np.empty((self.lattice.n_strips, *values.shape[1:]), dtype=values.dtype)
        result[self.strips] = values
        if len(self.strips) < self.lattice.n_strips:
            left = np.flatnonzero(~self.lattice.strip_is_right)
            result[left] = result[self.lattice.strip_mirror[left]]
        return result

    def evaluate(self, solver, cfg, condition):
        """Solve one condition with analytic Cp and strip-lift Jacobians."""
        from ventorum.panel3d.post import (
            compute_surface_velocity_and_cp, integrate_forces_and_moments, surface_gradient,
        )
        from ventorum.panel3d.solver import PanelResult
        from ventorum.solvers.panel3d import _panel_section_lines

        ns = len(self.maps)
        n = len(solver.centroids)
        factorized = solver.factorization_count != self.last_factorization_count
        self.last_factorization_count = solver.factorization_count
        speed = float(condition.V_inf)
        coeffs = solver._condition_coeffs(condition)
        mu0 = cfg.unit_mu @ coeffs
        mu_full0 = solver._reconstruct_full_mu(cfg.unit_mu, coeffs)
        if self.lateral:
            # Section onset change and section lift direction of each strip.
            u_strip = _section_onset_derivative(freestream_direction(condition.alpha, condition.beta), self.axes)
            norm = np.linalg.norm(u_strip, axis=1)
            lift_strip = u_strip / np.where(norm > 1e-12, norm, np.inf)[:, None]
        else:
            u_strip = lift_strip = None
        g = solver._viscous_response(cfg, condition, u_strip)
        gf = np.vstack([g, g]) if solver.mesh.half else g
        if solver.kutta == "pressure":
            z, zf, up, lo, expand = solver._viscous_pressure_response(cfg, coeffs)
        else:
            z, zf = np.zeros((n, 0)), np.zeros((len(mu_full0), 0))
            up, lo = cfg.wake.wake_up, cfg.wake.wake_lo
            def expand(values):
                return values
        response = np.column_stack([g, z])
        response_full = np.column_stack([gf, zf])
        dv = -surface_gradient(solver._grad_op, response_full)
        normal = solver._full_geom.normal
        owners = np.tile(self.owner, 2) if solver.mesh.half else self.owner
        if self.lateral:
            onset_delta = -speed * u_strip[owners]
            if solver.mesh.half:
                # The mirrored panels have the mirrored onset change.
                onset_delta[n:, 1] *= -1.0
            tangential = onset_delta - np.sum(normal * onset_delta, axis=1)[:, None] * normal
            # Strip lift along the section lift direction: in the section
            # plane, normal to the in-plane onset (the 2D section reference).
            weights = -solver.areas * np.sum(solver.normals * lift_strip[self.owner], axis=1)
            source_rate = speed * np.sum(solver.normals * u_strip[self.owner], axis=1)
        else:
            ua = np.cos(condition.beta) * np.array([-np.sin(condition.alpha), 0, np.cos(condition.alpha)])
            onset_delta = -speed * ua
            tangential = onset_delta - (normal @ onset_delta)[:, None] * normal
            lift_axis = np.array([-np.sin(condition.alpha), 0, np.cos(condition.alpha)])
            weights = -solver.areas * (solver.normals @ lift_axis)
            source_rate = speed * (solver.normals @ ua)
        dv[np.arange(len(owners)), :, owners] += tangential
        onset = solver._condition_velocity(condition, solver._full_geom.centroid)
        velocity0, _ = compute_surface_velocity_and_cp(
            onset, normal, surface_gradient(solver._grad_op, mu_full0), speed,
        )
        onset_sq = np.sum(onset**2, axis=1) if has_rates(condition) else None
        ground = None if cfg.ground is None else GroundPlane(*cfg.ground)
        polars, uses_height, h_over_c = self.resolver.resolve(self.lattice, self.strips, condition, ground)
        fallback = self.resolver.fallback_note(self.lattice, self.strips)
        slope, zero, moment = self.slope, self.zero, self.moment
        clean_zero = self.clean_zero
        if np.any(uses_height):
            slope, zero, moment = _panel_section_lines(
                self.geometry.mesh, self.maps, self.lattice, self.strips, ground, uses_height,
            )
            clean_zero = zero
            if self.clean_geometry is not None:
                clean_geom, clean_maps = self.clean_geometry
                _, clean_zero, _ = _panel_section_lines(
                    clean_geom.mesh, clean_maps, self.lattice, self.strips, ground, uses_height,
                )
        zero_target = _thin_airfoil_zero_lift(
            self.lattice, self.aircraft, self.strips, zero, clean_zero, ground, uses_height,
        )
        offset = slope * (zero - zero_target)

        def evaluate(x):
            return _panel_residual(x, velocity0, dv, speed, self.incidence, weights, self.area,
                                   zero, slope, polars, offset, up, lo, solver.kutta == "pressure",
                                   onset_sq)

        def newton(start, budget=None):
            x = start.copy()
            state = evaluate(x)
            hist = [float(np.max(np.abs(state["r"])))]
            target_tol = min(self.tol, 1e-9) if solver.kutta == "pressure" else self.tol
            for _ in range(self.max_iter if budget is None else budget):
                if hist[-1] < target_tol:
                    break
                accepted = False
                for matrix in (state["jac"], state["dc"]):
                    try:
                        step = self.relaxation * np.linalg.solve(matrix, -state["r"])
                    except np.linalg.LinAlgError:
                        continue
                    if not np.all(np.isfinite(step)):
                        continue
                    f0 = float(state["r"] @ state["r"])
                    lam = 1.0
                    while lam >= 1 / 1024:
                        trial = evaluate(x + lam * step)
                        if float(trial["r"] @ trial["r"]) <= (1 - 1e-4 * lam) * f0:
                            x += lam * step
                            state = trial
                            accepted = True
                            break
                        lam *= 0.5
                    if accepted:
                        break
                if not accepted:
                    break
                hist.append(float(np.max(np.abs(state["r"]))))
            state.update(x=x, hist=hist, converged=hist[-1] < self.tol and (
                solver.kutta != "pressure" or np.max(np.abs(state["jump"])) < 1e-8))
            state.update(section_slope=slope, section_zero=zero, section_moment=moment)
            return state

        def panel_result(state):
            mu = mu0 + response @ state["x"]
            muf = mu_full0 + response_full @ state["x"]
            force = integrate_forces_and_moments(
                state["cp"], solver._full_geom.area, normal, solver._full_geom.centroid,
                condition, solver.S_ref, solver.b_ref, solver.c_ref, solver.ref_point,
            )
            wake = muf[cfg.full_wake.wake_up] - muf[cfg.full_wake.wake_lo]
            if solver.kutta == "pressure":
                wake += expand(state["x"][ns:])
            sigma = solver._unit_sigma @ coeffs + source_rate * state["x"][:ns][self.owner]
            return PanelResult(
                CL=force["CL"], CD_pressure=force["CD_pressure"], CY=force["CY"],
                Cl=force["Cl"], Cm=force["Cm"], Cn=force["Cn"],
                CDi_trefftz=solver._trefftz(cfg, muf, condition, wake_gamma=wake),
                cp=state["cp"][:n], mu=mu, sigma=sigma, velocity=state["velocity"][:n],
                condition=condition, kutta=solver.kutta,
                kutta_iters=state.get("primary_iterations", len(state["hist"]) - 1),
                kutta_residual=float(np.max(np.abs(state["jump"]))), kutta_converged=state["converged"],
                mu_w=wake[:len(cfg.wake.wake_up)],
            )

        def branch_seed(state, sign):
            seed = state["x"].copy()
            for j, polar in enumerate(polars):
                maximum = _alpha_cl_max(polar)
                if maximum is not None:
                    shift = maximum + sign * np.radians(STALL_START_MARGIN_DEG) - state["alpha_v"][j]
                    seed[j] += max(0, shift) if sign == 1 else min(0, shift)
            return seed

        start = np.zeros(response.shape[1])
        if condition.alpha < 0 and not self.negative_started:
            self.previous = self.first_positive
            self.negative_started = True
        if self.previous is not None:
            start[:ns] = self.previous
        primary = newton(start)
        # A sharp polar turnover can stop a mixed attached/stalled start
        # at a local residual minimum. Reuse the V1 branch seeds within
        # the remaining primary iteration budget. Failed attempts are
        # never reported as branches.
        used = len(primary["hist"]) - 1
        recovery_runs = 0
        recovery_histories = []
        if not primary["converged"] and np.any(primary["m"] <= 0):
            for sign in (1, -1):
                if used >= self.max_iter:
                    break
                recovered = newton(branch_seed(primary, sign), self.max_iter - used)
                recovery_histories.append(recovered["hist"].copy())
                recovery_runs += 1
                used += len(recovered["hist"]) - 1
                if recovered["converged"]:
                    primary = recovered
                    break
        primary["recovery_runs"] = recovery_runs
        primary["recovery_residual_histories"] = recovery_histories
        primary["primary_iterations"] = used
        primary["factorized"] = factorized
        primary["panel"] = panel_result(primary)
        branches = []
        branch_runs = 0
        if primary["converged"]:
            self.previous = primary["x"][:ns].copy()
            if condition.alpha >= 0 and self.first_positive is None:
                self.first_positive = self.previous.copy()
            if np.any(primary["m"] <= 0):
                for sign in (1, -1):
                    seed = branch_seed(primary, sign)
                    branch_runs += 1
                    other = newton(seed)
                    if other["converged"]:
                        other["panel"] = panel_result(other)
                        if all(abs(other["panel"].CL - s["panel"].CL) > BRANCH_CL_DIFFERENCE
                               for s in [primary, *branches]):
                            branches.append(other)
        self.states[id(condition)] = (primary, branches, polars, offset, h_over_c, branch_runs, fallback)
        for state in [primary, *branches]:
            state.pop("jac")
            state.pop("dc")
        return primary["panel"]

    def finish_result(self, result, solver, geometry, aircraft):
        """Add polar drag, the lumped moment correction, and branch reports."""
        from ventorum.solvers.panel3d import _spanwise_results_batch

        primary, branches, polars, offset, heights, branch_runs, fallback = self.states[id(result.condition)]
        if fallback is not None and fallback not in result.details["notes"]:
            result.details["notes"].append(fallback)
        area = self.full(self.area)
        chord = self.lattice.chord
        axis = np.cross(self.lattice.normal, self.lattice.chord_dir)

        def finish(res, state):
            slope, zero = state["section_slope"], state["section_zero"]
            alpha_e = zero + state["cl"] / slope
            co, si = np.cos(alpha_e), np.sin(alpha_e)
            cm_inv = np.sum(state["section_moment"] * np.column_stack([np.ones_like(co), co**2, co * si, si**2]), axis=1)
            cm_visc = np.array([float(section_cm(p, a)) for p, a in zip(polars, state["alpha_v"])])
            cd = np.array([float(section_cd(p, a)) for p, a in zip(polars, state["alpha_v"])])
            delta_cm = self.full(cm_visc - cm_inv)
            pressure_cm = np.concatenate([sw.Cm_section.copy() for sw in res.spanwise])
            cd_full = self.full(cd)
            av = self.full(state["alpha_v"])
            for surf, sw in zip(self.lattice.surfaces, res.spanwise):
                ids = surf.strips
                sw.Cm_section = sw.Cm_section + delta_cm[ids] * axis[ids, 1]
                sw.alpha_eff = av[ids]
                sw.alpha_i = res.condition.alpha - sw.alpha_eff
            moment = np.sum((area * chord * delta_cm)[:, None] * axis, axis=0) / aircraft.S_ref
            res.totals.Cl -= moment[0] / aircraft.b_ref
            res.totals.Cm += moment[1] / aircraft.c_ref
            res.totals.Cn -= moment[2] / aircraft.b_ref
            res.totals.CDp = float(area @ cd_full / aircraft.S_ref)
            res.totals.CD_total = res.totals.CDi + res.totals.CDp
            res.totals.e = (res.totals.CL**2 / (np.pi * res.totals.AR * res.totals.CDi)
                            if res.totals.CDi > 1e-12 else float("nan"))
            res.solver_type = "viscous_panel3d"
            res.converged = state["converged"]
            res.iterations = state.get("primary_iterations", len(state["hist"]) - 1)
            res.residual_history = state["hist"].copy()
            res.details.update(
                delta=self.full(state["x"][:len(self.maps)]), alpha_eff=av,
                alpha0_section=self.full(zero), lift_slope_section=self.full(slope),
                alpha0_thin_airfoil=self.full(zero - offset / slope),
                cl_inviscid=self.full(state["cl"]), cl_viscous=self.full(state["cl"] - state["r"][:len(self.maps)] + offset),
                flap_offset=self.full(offset), strip_area=area, strip_h_over_c=self.full(heights),
                cd_profile=cd_full,
                cm_section_inviscid=self.full(cm_inv), cm_section_pressure=pressure_cm,
                trailing_edge_pressure_jump=state["jump"].copy(),
                post_stall=bool(np.any(state["m"] <= 0)), post_stall_strips=self.full(state["m"] <= 0),
                strip_flags=list(self.full(np.where(state["m"] <= 0, "post-stall, non-unique", "attached"))),
                non_unique=bool(branches), branch_search_runs=branch_runs,
                recovery_runs=state.get("recovery_runs", 0),
                recovery_residual_histories=state.get("recovery_residual_histories", []),
                factorized=state.get("factorized", False),
                factorization_count=solver.factorization_count, solve_count=solver._viscous_solve_count,
                solve_rhs_count=solver._viscous_rhs_count,
                viscous_iterations=res.iterations, viscous_residual_history=res.residual_history.copy(),
            )
            if not res.converged:
                res.details["notes"].append(_not_converged_note(state, self.max_iter, self.tol))
            if branches:
                res.details["notes"].append("Non-unique post-stall solution; see details['branches'].")
            limits = polar_limits(list(self.full(np.asarray(polars, dtype=object))))
            res.totals.trust = evaluate_aerodynamic_trust(
                AR=res.totals.AR, condition=res.condition, spanwise_list=res.spanwise,
                CL=res.totals.CL, CDi=res.totals.CDi, CD_total=res.totals.CD_total,
                converged=res.converged, n_panels=len(self.maps), solver_type="panel3d",
                n_chord=res.details["panel"]["n_chord"], mesh_stats=geometry.mesh_stats,
                h_over_c=None if res.details["h_min"] is None else res.details["h_min"] / aircraft.c_ref,
                notes=res.details["notes"], CD_nearfield=res.totals.CDi_nearfield,
                polar=None if limits is None else polar_stats(limits, self.full(state["cl"]), av),
                post_stall_branch=bool(branches),
            )
            return res

        branch_results = []
        for branch in branches:
            panel = branch["panel"]
            totals = dataclasses.replace(result.totals, CL=panel.CL, CDi=panel.CDi_trefftz,
                                          Cm=panel.Cm, Cl=panel.Cl, Cn=panel.Cn, CY=panel.CY,
                                          CDi_nearfield=panel.CD_pressure)
            sw = _spanwise_results_batch(solver, [panel], [result.condition], geometry.strip_maps, geometry.geom_meta)[0]
            details = {**result.details, "panel_result": panel, "cp": panel.cp, "panel_cp": panel.cp,
                       "notes": list(result.details["notes"])}
            branch_results.append(finish(dataclasses.replace(result, totals=totals, spanwise=sw, details=details), branch))
        finish(result, primary)
        if branch_results:
            result.details["branches"] = [result, *branch_results]


def _solve_panel_viscous(aircraft, conditions, settings, polars, max_iter, tol, relaxation):
    """Dispatch the CPU panel coupling with one LU per configuration."""
    from ventorum import gpu
    from ventorum.solvers.panel3d import _auto_cpu_backend

    if settings.panel_estimate_mesh_error:
        raise ValueError(
            "panel_estimate_mesh_error is not supported with the viscous panel coupling; "
            "run the mesh study inviscid"
        )
    backend = str(settings.panel_backend).lower()
    if backend == "gpu" or gpu.get_device() == "gpu":
        raise ValueError("Viscous panel coupling requires a CPU backend; GPU is not supported.")
    if settings.panel_ground_solver == "krylov":
        raise ValueError("Viscous panel coupling needs one direct LU per configuration; Krylov is not supported.")
    if backend == "auto":
        backend = _auto_cpu_backend()[0]
    settings = dataclasses.replace(settings, panel_backend=backend)
    single = isinstance(conditions, FlightCondition)
    conds = [conditions] if single else list(conditions)
    if not conds:
        return []
    coupling = _PanelCoupling(polars, max_iter, tol, relaxation)
    solver = make_solver("panel3d")
    solver._viscous_coupling = coupling
    up = sorted((i for i, c in enumerate(conds) if c.alpha >= 0), key=lambda i: conds[i].alpha)
    down = sorted((i for i, c in enumerate(conds) if c.alpha < 0), key=lambda i: -conds[i].alpha)
    order = up + down
    solved = solver.solve(aircraft, [conds[i] for i in order], settings)
    results = [None] * len(conds)
    for i, result in zip(order, solved):
        results[i] = result
    return results[0] if single else results
