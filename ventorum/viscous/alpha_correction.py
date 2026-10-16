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

Ground effect. When the polar of a strip has the ground height (a table
with a height axis, or a source with ground support), the 2D section of
the vortex-lattice strip also has its images in the ground, so the 2D
ground effect is not counted twice. A free-air polar is read with the
free-air section relation.

Residual and Newton iteration::

    cl_inv(delta) = cl_inv(0) - D delta
    alpha_v(delta) = alpha0 + cl_inv(delta) / a + delta
    r(delta) = cl_inv(delta) - cl_visc(alpha_v(delta))
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
from ventorum.aero.system import GroundPlane, UnknownMap
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
from ventorum.geometry.controls import deflected_airfoil
from ventorum.geometry.lattice import (
    VortexLattice,
    _AirfoilBlender,
    _sorted_sections,
    section_cl,
    section_cl_slope,
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
from ventorum.utils.parallel import solve_threads
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
    """Return the flap deflection [deg] and the hinge [-] of strip *s* (0 and None without a control)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Section polars of the strips
# ═══════════════════════════════════════════════════════════════════════════════

class _StripPolars:
    """Resolve the section polar of each strip; memo over the conditions of one call."""

    def __init__(self, polars: Any, aircraft: Aircraft) -> None:
        if polars is None:
            raise ValueError("polars must be a PolarTable, a SectionPolarSource, an airfoil or a dict.")
        self.polars = polars
        self.aircraft = aircraft
        self._memo: dict[Any, TabulatedAirfoil] = {}

    def _candidate(self, lattice: VortexLattice, s: int) -> Any:
        if not isinstance(self.polars, dict):
            return self.polars
        surf = self.aircraft.surfaces[int(lattice.strip_surface[s])]
        base = _base_airfoil(self.aircraft, lattice, s)
        for key in (s, surf.name, getattr(base, "name", None), "default"):
            if key is not None and key in self.polars:
                return self.polars[key]
        return base

    def resolve(
        self,
        lattice: VortexLattice,
        strips: np.ndarray,
        condition: FlightCondition,
        ground: GroundPlane | None,
    ) -> tuple[list[Any], np.ndarray, np.ndarray]:
        """Return the polar of each strip, the flags "polar uses the height" and the strip h/c [-]."""
        n = len(strips)
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
            cand = self._candidate(lattice, s)
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
    """
    a_out = np.empty(len(strips))
    a0_out = np.empty(len(strips))
    for i, s in enumerate(strips):
        pidx = np.flatnonzero(lattice.panel_strip == s)
        cdir = lattice.chord_dir[s]
        nrm = lattice.normal[s]
        origin = 0.5 * (lattice.le_left[s] + lattice.le_right[s])
        q = 0.5 * (lattice.a[pidx] + lattice.b[pidx])
        p = lattice.cp[pidx]

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
        # One factorization for both right-hand sides.
        g = np.linalg.solve(M, np.column_stack([-np.ones(len(pidx)), theta]))
        g_alpha, g_theta = g[:, 0].sum(), g[:, 1].sum()
        a_out[i] = 2.0 * g_alpha / float(lattice.chord[s])
        a0_out[i] = -g_theta / g_alpha
    return a_out, a0_out


def _rhs_derivative(
    lattice: VortexLattice,
    model: str,
    umap: UnknownMap,
    V_inf: float,
    u_alpha: np.ndarray,
    strips: np.ndarray,
) -> np.ndarray:
    """Return the right-hand-side change per unit correction of each strip.

    A correction ``delta_s`` lowers the angle of strip s: the onset direction
    of its boundary condition becomes ``d(alpha - delta_s)``. The columns are
    ``d rhs / d delta_s``, shape (n_unknowns, n_strips).
    """
    if model == "vlm":
        n_vec = lattice.normal_bc[umap.unknown_panels]
        w = V_inf * (n_vec @ u_alpha)
    else:
        sd = _llt_strip_data(lattice, umap)
        w = -V_inf * (sd["normal"] @ u_alpha)
    unk_strip = lattice.panel_strip[umap.unknown_panels]
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
        ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
        cb = np.cos(condition.beta)
        u_alpha = np.array([-sa * cb, 0.0, ca * cb])
        strips = np.flatnonzero(lattice.strip_is_right) if umap.symmetric else np.arange(lattice.n_strips)
        B = _rhs_derivative(lattice, model, umap, V_inf, u_alpha, strips)
        G = scipy.linalg.lu_solve(lu, B)
        gamma0 = scipy.linalg.lu_solve(lu, rhs0)

    n_ind = len(strips)
    unk_strip = lattice.panel_strip[umap.unknown_panels]
    scale = 2.0 / (V_inf * np.maximum(lattice.chord[strips], 1e-12))
    cl0 = np.array([np.sum(gamma0[unk_strip == s]) for s in strips]) * scale
    # cl_inv(delta) = cl0 - D delta: a correction lowers the lift of the strips.
    D = -np.array([np.sum(G[unk_strip == s, :], axis=0) for s in strips]) * scale[:, None]

    polars, uses_height, h_over_c = resolver.resolve(lattice, strips, condition, ground)
    if model == "vlm":
        a_sec, alpha0 = _vlm_section_lines(lattice, strips, ground, uses_height)
    else:
        a_sec = np.maximum(lattice.a0[strips], 1e-6)
        alpha0 = lattice.alpha_L0[strips].astype(float)

    def evaluate(delta: np.ndarray):
        cl_inv = cl0 - D @ delta
        alpha_v = alpha0 + cl_inv / a_sec + delta
        cl_v = np.array([float(section_cl(p, a)) for p, a in zip(polars, alpha_v)])
        m = np.array([float(section_cl_slope(p, a)) for p, a in zip(polars, alpha_v)])
        return cl_inv - cl_v, cl_inv, alpha_v, m

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

    loads_lattice = dataclasses.replace(
        lattice, airfoils=list(lattice.airfoils), geom_cache={}, kernel_cache=None,
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
    for msg in notes:
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
    model: Literal["vlm", "llt"] = "vlm",
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
        surface name, an airfoil name or ``"default"`` to one of these;
        strips without an entry use their section airfoil.
    model : {"vlm", "llt"}
        Linear solver: the vortex-lattice method or the linear lifting
        line.
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
        section relation of the lattice), ``strip_h_over_c`` [-],
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
    """
    model_str = str(model).lower()
    if model_str not in ("vlm", "llt"):
        raise ValueError(f"model must be 'vlm' or 'llt', got {model!r}.")
    if not 0.0 < float(relaxation) <= 1.0:
        raise ValueError(f"relaxation must lie in (0, 1], got {relaxation}.")
    if int(max_iter) < 0:
        raise ValueError(f"max_iter must not be negative, got {max_iter}.")
    if not float(tol) > 0.0:
        raise ValueError(f"tol must be positive, got {tol}.")

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
    results: list[SolverResult | None] = [None] * len(cond_list)
    alphas = np.array([float(c.alpha) for c in cond_list])
    up = [int(i) for i in np.argsort(alphas, kind="stable") if alphas[i] >= 0.0]
    down = [int(i) for i in np.argsort(-alphas, kind="stable") if alphas[i] < 0.0]

    def run(order: list[int], start: np.ndarray | None) -> np.ndarray | None:
        delta = start
        for i in order:
            res = _solve_condition(
                solver, model_str, aircraft_obj, cond_list[i], settings, resolver, fact, rp,
                main_surface, delta, int(max_iter), float(tol), float(relaxation),
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
