# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Shared driver for the lattice-based solvers (vortex lattice and lifting line).

A solve has four steps:

1. Build the vortex lattice of the aircraft (body axes).
2. Place the ground plane, if any (``condition.h`` or an explicit plane).
3. Solve for the circulation (done by the subclass).
4. Integrate loads and evaluate the trust score.
"""

from __future__ import annotations

import time
import warnings
from typing import Literal

import numpy as np

from ventorum.aero.loads import compute_loads, compute_loads_batch
from ventorum.aero.system import (
    GroundPlane,
    check_ground_clearance,
    ground_plane_from_condition,
    wake_direction,
)
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
)
from ventorum.core.errors import ValidityError
from ventorum.core.trust import evaluate_aerodynamic_trust
from ventorum.geometry import lattice_cache
from ventorum.geometry.lattice import VortexLattice, build_lattice
from ventorum.solvers.base import BaseSolver
from ventorum.utils.parallel import solve_threads
from ventorum.utils.validation import validate_aircraft

# Out of ground effect, 4 chordwise panels put the aerodynamic centre within
# about 0.01 c of the converged value, also for aspect ratio 1 (with 1 panel
# the centre is always at c/4, about 0.08 c wrong at aspect ratio 1).
DEFAULT_N_CHORD = 4
MAX_AUTO_N_CHORD = 32

# Phillips & Snyder lifting line is not grid convergent when the quarter-chord
# line has a kink (sweep): the lift keeps falling as the mesh is refined
# (about -1.3 % from 20 to 160 panels at 5 deg sweep, -10 % at 30 deg). A
# tapered wing with a straight leading edge has a forward-swept quarter-chord
# line; at 2.4 deg (AR 8, taper 0.5) the lift slope is 0.5 % low with
# half-cosine spacing.
LLT_SWEEP_WARNING_DEG = 2.5

# Lifting line in ground effect: against the vortex lattice it under-predicts
# the lift increment by 3 to 5 % of CL at h_min/c = 1 and by 13 to 18 % at
# h_min/c = 0.5 (rectangular wings, AR 4 to 8, alpha 2 to 6 deg). Refused
# below the first value, warned below the second.
LLT_GE_MIN_H_OVER_C = 1.0
LLT_GE_WARN_H_OVER_C = 2.0


def as_aircraft(geometry: Aircraft | LiftingSurface) -> Aircraft:
    """Return *geometry* as an aircraft; wrap a single surface in a new aircraft."""
    if isinstance(geometry, LiftingSurface):
        return Aircraft(name="SingleWing", surfaces=[geometry])
    return geometry


def main_surface_strip_count(lattice: VortexLattice, index: int = 0) -> int:
    """Spanwise strips per semi-span on surface *index* (for the trust score).

    On a symmetric surface only the right half is counted. A surface that is
    not symmetric (also a mirror copy, which has no right half) counts all
    its strips.
    """
    on_surf = lattice.strip_surface == index
    n_right = int(np.sum(on_surf & lattice.strip_is_right))
    return n_right if n_right > 0 else int(np.sum(on_surf))


def quarter_chord_sweep_deg(lattice: VortexLattice) -> float:
    """Return the area-weighted mean of abs(sweep) of the bound-vortex line [deg]."""
    dl = lattice.dl
    lam = np.degrees(np.arctan2(np.abs(dl[:, 0]), np.sqrt(dl[:, 1] ** 2 + dl[:, 2] ** 2)))
    w = lattice.area
    return float(np.sum(lam * w) / max(np.sum(w), 1e-300))


class LatticeSolver(BaseSolver):
    """Base class of the vortex-lattice and lifting-line solvers."""

    collocation: Literal["vlm", "llt"] = "vlm"
    name: str = "lattice"

    # ------------------------------------------------------------------ setup
    def resolve_n_chord(
        self,
        aircraft: Aircraft,
        settings: SolverSettings,
        condition: FlightCondition,
        ground: GroundPlane | None,
        ref_point: np.ndarray | None = None,
    ) -> int:
        """Return the number of chordwise panels.

        The lifting-line solvers use 1. A value in *settings* is used as
        given. Out of ground effect the default is used. In ground effect a
        chordwise panel is not longer than the smallest gap to the ground,
        within the default and the maximum automatic count.

        Parameters
        ----------
        aircraft : Aircraft
            The aircraft.
        settings : SolverSettings
            Solver settings.
        condition : FlightCondition
            Flight condition. ``condition.h`` [m] sets the ground if
            *ground* is None.
        ground : GroundPlane or None
            Ground plane, or None.
        ref_point : numpy.ndarray or None, optional
            Moment reference point [m], shape (3,). If None, the aircraft
            reference point is used.

        Returns
        -------
        int
            Number of chordwise panels.
        """
        if self.collocation == "llt":
            return 1
        if settings.n_chord is not None:
            return max(1, int(settings.n_chord))
        if ground is None and condition.h is None:
            return DEFAULT_N_CHORD
        # In ground effect a chordwise panel should not be longer than the gap.
        probe = build_lattice(aircraft, settings, collocation="vlm", n_chord=1)
        rp = aircraft.moment_reference() if ref_point is None else ref_point
        gp = ground if ground is not None else ground_plane_from_condition(probe, condition, rp)
        h_min = float(np.min(gp.height(probe.all_points())))
        if h_min <= 0.0:
            return DEFAULT_N_CHORD  # the strike is reported later
        n = int(np.ceil(float(np.max(probe.chord)) / h_min))
        return int(np.clip(n, DEFAULT_N_CHORD, MAX_AUTO_N_CHORD))

    def build(
        self,
        aircraft: Aircraft,
        settings: SolverSettings,
        condition: FlightCondition,
        ground: GroundPlane | None = None,
        ref_point: np.ndarray | None = None,
        fingerprint: object = None,
    ) -> VortexLattice:
        """Build the vortex lattice of the aircraft for this solver.

        The arguments are the same as for :meth:`resolve_n_chord`;
        *fingerprint* is the surfaces fingerprint if the caller computed it.
        """
        n_chord = self.resolve_n_chord(aircraft, settings, condition, ground, ref_point)
        chord_spacing = getattr(settings, "chord_spacing", "uniform")
        # The lattice depends only on the geometry and the mesh settings: reuse
        # it for repeated solves (see ventorum.geometry.lattice_cache).
        key = lattice_cache.lattice_key(aircraft, settings, self.collocation, n_chord, chord_spacing,
                                        fingerprint=fingerprint)
        return lattice_cache.get_or_build(key, lambda: build_lattice(
            aircraft, settings, collocation=self.collocation,
            n_chord=n_chord, chord_spacing=chord_spacing,
        ))

    # ------------------------------------------------------------------ solve
    def solve_circulation(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        settings: SolverSettings,
        ground: GroundPlane | None,
        wake_dir: np.ndarray | None = None,
        gamma0: np.ndarray | None = None,
    ):
        """Return ``(gamma_panel, alpha_eff_strip or None, SolveInfo)``.

        *gamma0* is a start value (used by the nonlinear solver in sweeps).
        """
        raise NotImplementedError

    def solve_lattice(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        settings: SolverSettings,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        ground: GroundPlane | None = None,
        ref_point: np.ndarray | None = None,
        gamma0: np.ndarray | None = None,
        main_surface: int = 0,
    ) -> SolverResult:
        """Solve a lattice that is already built.

        *ref_point* is the moment reference point and, for ``condition.h``,
        the point whose height is ``h`` (the origin if None). *main_surface*
        is the index of the main surface (see
        :meth:`Aircraft.main_surface_index`), used for the panel count of
        the trust score.

        In the result, ``totals.CDi`` is the induced drag from the Trefftz
        plane (the reference value). ``totals.CDi_nearfield`` comes from the
        surface forces; it is a diagnostic and is not reliable on swept wings.
        """
        with solve_threads(lattice.n_panels):
            return self._solve_lattice(
                lattice, condition, settings, S_ref, b_ref, c_ref,
                ground=ground, ref_point=ref_point, gamma0=gamma0, main_surface=main_surface,
            )

    def _solve_lattice(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        settings: SolverSettings,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        ground: GroundPlane | None = None,
        ref_point: np.ndarray | None = None,
        gamma0: np.ndarray | None = None,
        main_surface: int = 0,
    ) -> SolverResult:
        t0 = time.perf_counter()
        ground, h_min, notes, wd = self._case_setup(lattice, condition, settings, c_ref, ground, ref_point)
        gamma, alpha_eff, info = self.solve_circulation(lattice, condition, settings, ground, wd, gamma0=gamma0)
        loads = compute_loads(
            lattice, gamma, condition, S_ref, b_ref, c_ref,
            ground=ground, ref_point=ref_point, alpha_eff_strip=alpha_eff, wake_dir=wd,
            v_control=getattr(info, "v_control", None),
        )
        return self._case_result(lattice, condition, loads, info, gamma, ground, h_min, notes, wd,
                                 c_ref, main_surface, t0)

    def _case_setup(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        settings: SolverSettings,
        c_ref: float,
        ground: GroundPlane | None,
        ref_point: np.ndarray | None,
    ) -> tuple[GroundPlane | None, float | None, list[str], np.ndarray]:
        """Return the ground plane, the smallest ground clearance [m], the notes and the wake direction of one case.

        Raises the errors of the validity checks (ground strike, lifting line
        too near the ground).
        """
        if ground is None and condition.h is not None:
            ground = ground_plane_from_condition(lattice, condition, ref_point)
        h_min = check_ground_clearance(lattice, ground)
        if self.collocation == "llt" and h_min is not None and c_ref:
            hc = h_min / c_ref
            if hc < LLT_GE_MIN_H_OVER_C:
                raise ValidityError(
                    f"Lifting-line solver '{self.name}' is not valid in ground effect below "
                    f"h_min/c = {LLT_GE_MIN_H_OVER_C} (here {hc:.3f}). Use solver='vlm'."
                )
        notes: list[str] = list(lattice.join_warnings)
        if self.collocation == "llt" and h_min is not None and c_ref and h_min / c_ref < LLT_GE_WARN_H_OVER_C:
            notes.append(
                f"Lifting line in ground effect at h_min/c = {h_min / c_ref:.2f}: it under-predicts the "
                "ground-effect lift increment compared with the VLM (about 1 to 5 % of CL at "
                "h_min/c 1 to 2)."
            )
        wd = wake_direction(condition, ground, getattr(settings, "wake_alignment", "freestream"))
        return ground, h_min, notes, wd

    @staticmethod
    def _cached(lattice: VortexLattice, key, compute):
        """Return the lattice-only value *key* from ``lattice.geom_cache``; compute it on the first use."""
        value = lattice.geom_cache.get(key)
        if value is None:
            value = compute()
            lattice.geom_cache[key] = value
        return value

    def _case_result(
        self,
        lattice: VortexLattice,
        condition: FlightCondition,
        loads,
        info,
        gamma: np.ndarray,
        ground: GroundPlane | None,
        h_min: float | None,
        notes: list[str],
        wd: np.ndarray,
        c_ref: float,
        main_surface: int,
        t0: float,
        execution_time: float | None = None,
    ) -> SolverResult:
        """Return the result object of one case: trust score, details and warnings.

        *execution_time* [s] replaces the time since *t0* when given (a
        batch gives each case its share of the batch time).
        """
        sweep_deg = self._cached(lattice, "quarter_chord_sweep_deg", lambda: quarter_chord_sweep_deg(lattice))
        if self.collocation == "llt" and sweep_deg > LLT_SWEEP_WARNING_DEG:
            notes.append(
                f"Lifting line with {sweep_deg:.1f} deg mean quarter-chord sweep (a tapered wing with a "
                "straight leading edge also has it): the method is not grid convergent with sweep, and "
                "the lift falls as panels are added (lift-slope error measured against classical theory: "
                "about -0.5 % at 2.4 deg and -4 % at 8.5 deg with 40 half-cosine panels). For linear "
                "analysis use the VLM solver."
            )
        h_over_c = (h_min / c_ref) if (h_min is not None and c_ref) else None
        totals = loads.totals
        totals.trust = evaluate_aerodynamic_trust(
            AR=totals.AR,
            surfaces=None,
            condition=condition,
            spanwise_list=loads.spanwise,
            CL=totals.CL,
            CDi=totals.CDi,
            CD_total=totals.CD_total,
            converged=info.converged,
            n_panels=self._cached(lattice, ("main_strip_count", main_surface),
                                  lambda: main_surface_strip_count(lattice, main_surface)),
            max_sweep_rad=np.radians(sweep_deg),
            solver_type=self.name,
            h_over_c=h_over_c,
            n_chord=lattice.n_chord,
            notes=notes,
            max_chord_over_c=(float(self._cached(lattice, "max_chord", lambda: np.max(lattice.chord)) / c_ref)
                              if c_ref else 1.0),
        )
        res = SolverResult(
            spanwise=loads.spanwise,
            totals=totals,
            solver_type=self.name,
            converged=info.converged,
            iterations=info.iterations,
            residual_history=list(info.residual_history),
            condition=condition,
            execution_time=(time.perf_counter() - t0) if execution_time is None else execution_time,
            symmetry_used=info.symmetric,
        )
        # Kept for advanced use (plots, diagnostics, ground-effect post-processing).
        res.details = {"lattice": lattice, "loads": loads, "ground": ground, "h_min": h_min,
                       "sweep_deg": sweep_deg, "notes": notes, "gamma": gamma, "wake_dir": wd}
        for msg in notes:
            warnings.warn(msg, RuntimeWarning, stacklevel=4)
        return res

    def solve(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
        *,
        ground: GroundPlane | None = None,
        ref_point: np.ndarray | None = None,
    ) -> SolverResult:
        """Solve one flight condition.

        Parameters
        ----------
        aircraft : Aircraft or LiftingSurface
            Geometry definition.
        condition : FlightCondition
            Free-stream conditions.
        settings : SolverSettings
            Discretisation and convergence parameters.
        ground : GroundPlane or None, optional
            Ground plane. If None, ``condition.h`` sets the ground.
        ref_point : numpy.ndarray or None, optional
            Moment reference point [m], shape (3,). If None, the aircraft
            reference point is used.

        Returns
        -------
        SolverResult
            Spanwise and integrated results.
        """
        aircraft = as_aircraft(aircraft)
        validate_aircraft(aircraft)
        # The main surface and the automatic reference values depend only on
        # the surfaces: computed once per geometry (same fingerprint as the
        # lattice cache).
        fp = lattice_cache.surfaces_fingerprint(aircraft)
        geo = lattice_cache.geometry_info(fp, aircraft)
        aircraft.compute_reference_values(auto=geo["auto_ref"])
        rp = aircraft.moment_reference() if ref_point is None else np.asarray(ref_point, dtype=float)
        lattice = self.build(aircraft, settings, condition, ground, rp, fingerprint=fp)
        return self.solve_lattice(
            lattice, condition, settings,
            aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
            ground=ground, ref_point=rp, main_surface=geo["main"],
        )

    def solve_sweep(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
        ref_point: np.ndarray | None = None,
    ) -> list[SolverResult]:
        """Angle-of-attack sweep.

        The lifting-line solvers solve the angles as one batch (see
        :meth:`solve_batch`): each angle gets the same result as a single
        solve. The vortex lattice builds its lattice once (out of ground
        effect) and keeps the influence of the bound vortices and the
        chordwise legs between the angles. Each angle starts from the
        circulation of the previous one (continuation), which helps the
        nonlinear solver near the maximum lift.
        """
        aircraft = as_aircraft(aircraft)
        validate_aircraft(aircraft)
        aircraft.compute_reference_values()
        rp = aircraft.moment_reference() if ref_point is None else np.asarray(ref_point, dtype=float)
        conds = [
            FlightCondition(
                V_inf=condition.V_inf, alpha=float(a), beta=condition.beta,
                rho=condition.rho, h=condition.h, phi=getattr(condition, "phi", 0.0),
            )
            for a in np.asarray(alpha_range, dtype=float)
        ]
        if self.collocation == "llt":
            # The lifting-line lattice does not depend on the attitude (one
            # chordwise panel), also in ground effect.
            lattice = self.build(aircraft, settings, condition, None, rp)
            return self.solve_batch(lattice, conds, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
                                    ref_point=rp, main_surface=aircraft.main_surface_index(),
                                    continuation=condition.h is None)
        lattice = None if condition.h is not None else self.build(aircraft, settings, condition, None, rp)
        if lattice is not None:
            # The bound vortices and the chordwise legs do not change with
            # the angle of attack: compute their influence once.
            lattice.kernel_cache = {}
        main = aircraft.main_surface_index()
        out: list[SolverResult] = []
        g_prev = None
        for cond in conds:
            # In ground effect the automatic chordwise count depends on the attitude.
            lat = lattice if lattice is not None else self.build(aircraft, settings, cond, None, rp)
            g0 = g_prev if (g_prev is not None and lat is lattice) else None
            res = self.solve_lattice(
                lat, cond, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
                ref_point=rp, gamma0=g0, main_surface=main,
            )
            g_prev = res.details["gamma"] if res.converged else None
            out.append(res)
        return out

    def solve_batch(
        self,
        lattice: VortexLattice,
        conditions: list[FlightCondition],
        settings: SolverSettings,
        S_ref: float,
        b_ref: float,
        c_ref: float,
        ref_point: np.ndarray | None = None,
        main_surface: int = 0,
        continuation: bool = True,
    ) -> list[SolverResult]:
        """Solve several flight conditions on one lattice that is already built.

        The cases are solved together where the solver allows it
        (:meth:`solve_circulation_batch`), and the loads of all cases are
        computed together (:func:`ventorum.aero.loads.compute_loads_batch`).
        Each case gets the same result as :meth:`solve_lattice` for that
        case; with *continuation*, a case starts from the converged
        circulation of the case before it (used by the nonlinear solver).
        The ground plane of each case comes from ``condition.h``.

        Returns
        -------
        list of SolverResult
            One result per condition, in order. ``execution_time`` is the
            share of each case in the batch time.
        """
        t0 = time.perf_counter()
        if not conditions:
            return []
        with solve_threads(lattice.n_panels, batch=len(conditions)):
            setups = [self._case_setup(lattice, c, settings, c_ref, None, ref_point) for c in conditions]
            grounds = [st[0] for st in setups]
            wds = np.array([st[3] for st in setups], dtype=float).reshape(len(conditions), 3)
            sols = self.solve_circulation_batch(lattice, conditions, settings, grounds, wds, continuation)
            gammas = np.array([g for g, _, _ in sols], dtype=float)
            alpha_effs = None if any(a is None for _, a, _ in sols) else np.array([a for _, a, _ in sols])
            vcs = [getattr(info, "v_control", None) for _, _, info in sols]
            loads = compute_loads_batch(
                lattice, gammas, conditions, S_ref, b_ref, c_ref, grounds=grounds, ref_point=ref_point,
                alpha_eff_strips=alpha_effs, wake_dirs=wds,
                v_controls=None if any(v is None for v in vcs) else np.array(vcs),
            )
        share = (time.perf_counter() - t0) / len(conditions)
        return [
            self._case_result(lattice, cond, ld, info, g, st[0], st[1], st[2], st[3], c_ref, main_surface,
                              t0, execution_time=share)
            for cond, ld, (g, _, info), st in zip(conditions, loads, sols, setups)
        ]

    def solve_circulation_batch(
        self,
        lattice: VortexLattice,
        conditions: list[FlightCondition],
        settings: SolverSettings,
        grounds: list[GroundPlane | None],
        wake_dirs: np.ndarray,
        continuation: bool = True,
    ) -> list[tuple]:
        """Return ``(gamma_panel, alpha_eff_strip or None, SolveInfo)`` for each case.

        The default solves the cases one by one with :meth:`solve_circulation`.
        With *continuation*, a case starts from the converged circulation of
        the case before it.
        """
        out = []
        g_prev = None
        for cond, ground, wd in zip(conditions, grounds, wake_dirs):
            sol = self.solve_circulation(lattice, cond, settings, ground, wd,
                                         gamma0=g_prev if continuation else None)
            g_prev = sol[0] if sol[2].converged else None
            out.append(sol)
        return out


def assemble_system_matrix(
    aircraft: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings,
) -> np.ndarray:
    """Return the linear-system matrix that the selected solver uses (for conditioning checks).

    The Fourier solver and the nonlinear lifting line report the matrix of the
    linear lifting line (the start of the Newton iteration).
    """
    from ventorum.solvers.core import assemble_llt_linear, assemble_vlm
    from ventorum.solvers.factory import resolve_solver_type

    aircraft = as_aircraft(aircraft)
    aircraft.compute_reference_values()
    canonical = resolve_solver_type(settings.solver_type)
    collocation = "vlm" if canonical == "vlm" else "llt"
    if collocation == "vlm":
        from ventorum.solvers.horseshoe import HorseshoeSolver
        lattice = HorseshoeSolver().build(aircraft, settings, condition)
    else:
        lattice = build_lattice(aircraft, settings, collocation="llt")
    ground = ground_plane_from_condition(lattice, condition, aircraft.moment_reference()) if condition.h is not None else None
    wd = wake_direction(condition, ground, getattr(settings, "wake_alignment", "freestream"))
    use_sym = getattr(settings, "use_symmetry", True)
    if collocation == "vlm":
        A, _, _ = assemble_vlm(lattice, condition, ground, use_sym, wd)
    else:
        A, _, _ = assemble_llt_linear(lattice, condition, ground, use_sym, wd)
    return A
