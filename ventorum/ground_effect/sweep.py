# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run parameter sweeps in ground effect and compute the static stability.

The module gives the static-stability quantities of a wing in ground
effect.

Static stability in ground effect (Irodov criterion)
----------------------------------------------------
Near the ground the lift depends on the angle of attack and on the height.
Each has its own aerodynamic centre (positions measured aft of the
reference point, in metres)::

    x_alpha = -c * (dCm/dalpha) / (dCL/dalpha)   (pitch aerodynamic centre)
    x_h     = -c * (dCm/dh)     / (dCL/dh)       (height aerodynamic centre)

Longitudinal static stability at constant speed needs the height centre
upstream of (ahead of) the pitch centre::

    x_h < x_alpha,   so   irodov_margin = (x_alpha - x_h) / c > 0.

The criterion applies to the derivatives about the centre of gravity: the
pitch derivative is a rotation about the moment reference point, and the
margin changes when that point moves. Give the centre of gravity as the
moment reference point.

(R. D. Irodov, "Criteria of longitudinal stability of ekranoplan", Uchenye
Zapiski TsAGI 1(4), 1970, pp. 63-74, in Russian; English machine
translation, Foreign Technology Division, 1974, DTIC AD-A002918. See also
K. V. Rozhdestvensky, "Wing-in-ground effect vehicles", Progress in
Aerospace Sciences 42, 2006, pp. 211-283.)

Both derivatives are about the moment reference point, with the height of
that point held fixed, whatever ``height_ref`` the sweep uses: the margin of
one flight state then has one value. ``height_ref`` only sets where the
given heights are measured. A change of height at constant alpha moves every
point by the same distance, so the height derivative is the same in every
height convention. The alpha derivative is not: with ``height_ref`` other
than ``'ref'``, the alpha columns of the grid are at constant height of
another point, and the reference point moves. The chain rule corrects it::

    dF/dalpha (h_ref const) = dF/dalpha (h const) - dF/dh * dh_ref/dalpha (h const)

where ``h_ref`` is the height of the reference point of each case.

Sign conventions are the standard ones: a bank-restoring rolling moment has
``Cl < 0`` for ``phi > 0``, so ground-effect roll stiffness is ``Cl_phi < 0``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal
from collections.abc import Sequence

import numpy as np

from ventorum.core.datatypes import Aircraft, FlightCondition, LiftingSurface, SolverSettings
from ventorum.core.errors import GroundStrikeError, ValidityError
from ventorum.utils.jsonsafe import json_safe
from ventorum.utils.parallel import estimate_panels, run_cases
from ventorum.aero.system import make_ground_plane
from ventorum.geometry.lattice import build_lattice
from ventorum.core.constants import RHO_SL
from ventorum.ground_effect.solver import (
    MAX_BANK_DEG,
    _as_aircraft,
    analyze_ground_effect,
    check_height,
    find_bank_strike_limit,
    plane_reference,
)
from ventorum.solvers.factory import make_solver, resolve_solver_type
from ventorum.ground_effect.state import GroundEffectResult


def _nan_slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if np.sum(ok) < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


@dataclass
class GroundEffectSweepResult:
    """Grids of results over heights, angles of attack and bank angles.

    Arrays are indexed ``[h_idx, alpha_idx, phi_idx]``. Cases that strike the
    ground (or that the solver refuses) are NaN with ``is_strike`` or
    ``refused`` set, and the reason is in ``errors``. ``results`` has one
    entry per grid point in C order (None for a failed case); use
    :meth:`result_at` to get one.

    ``phi_strike_limit`` is the bank angle [deg] at the first ground contact
    (NaN if not computed). If no point touches up to the search limit
    (60 deg), the value is that limit and ``strike_limit_found`` is False:
    the true limit is larger. ``strike_limit_found`` is False also where the
    limit was not computed.

    ``n_chord`` is the number of chordwise panels of all cases. If the
    settings do not give it, the sweep sets it once from the case with the
    smallest gap to the ground (the lowest height), so that all heights use
    the same mesh and the height derivatives do not include a mesh change.

    Attributes
    ----------
    heights : numpy.ndarray
        Heights of the grid [m], in the convention of ``height_ref``.
    alphas_deg : numpy.ndarray
        Angles of attack of the grid [deg].
    phis_deg : numpy.ndarray
        Bank angles of the grid [deg].
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray or None
        Moment reference point [m], shape (3,).
    """

    heights: np.ndarray
    alphas_deg: np.ndarray
    phis_deg: np.ndarray
    results: list[GroundEffectResult | None] = field(default_factory=list)
    grid_shape: tuple[int, int, int] = (0, 0, 0)
    errors: dict[tuple[int, int, int], str] = field(default_factory=dict)

    CL: np.ndarray = field(default_factory=lambda: np.empty(0))
    CDi: np.ndarray = field(default_factory=lambda: np.empty(0))
    CD: np.ndarray = field(default_factory=lambda: np.empty(0))
    CY: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cl: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cm: np.ndarray = field(default_factory=lambda: np.empty(0))
    Cn: np.ndarray = field(default_factory=lambda: np.empty(0))
    L_over_D: np.ndarray = field(default_factory=lambda: np.empty(0))
    e: np.ndarray = field(default_factory=lambda: np.empty(0))
    h_min: np.ndarray = field(default_factory=lambda: np.empty(0))
    is_strike: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    refused: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    phi_strike_limit: np.ndarray = field(default_factory=lambda: np.empty(0))
    strike_limit_found: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    n_chord: int | None = None

    S_ref: float = 0.0
    b_ref: float = 0.0
    c_ref: float = 0.0
    ref_point: np.ndarray | None = None
    height_ref: str = "ref"
    execution_time: float = 0.0
    # Height [m] of the moment reference point in each case (NaN for a failed
    # case). It is needed for the Irodov derivatives when height_ref is not 'ref'.
    h_ref_grid: np.ndarray | None = None

    def result_at(self, h_idx: int, alpha_idx: int, phi_idx: int) -> GroundEffectResult | None:
        """Return the result of one grid point (None if it failed)."""
        _, na, npf = self.grid_shape
        return self.results[(h_idx * na + alpha_idx) * npf + phi_idx]

    def get_slice_1d(
        self,
        var_name: str,
        h_val: float | None = None,
        alpha_val: float | None = None,
        phi_val: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """1-D slice of a quantity. Give exactly two of (h_val, alpha_val, phi_val)."""
        arr = getattr(self, var_name)
        if h_val is None and alpha_val is not None and phi_val is not None:
            a_idx = int(np.argmin(np.abs(self.alphas_deg - alpha_val)))
            p_idx = int(np.argmin(np.abs(self.phis_deg - phi_val)))
            return self.heights, arr[:, a_idx, p_idx]
        if alpha_val is None and h_val is not None and phi_val is not None:
            h_idx = int(np.argmin(np.abs(self.heights - h_val)))
            p_idx = int(np.argmin(np.abs(self.phis_deg - phi_val)))
            return self.alphas_deg, arr[h_idx, :, p_idx]
        if phi_val is None and h_val is not None and alpha_val is not None:
            h_idx = int(np.argmin(np.abs(self.heights - h_val)))
            a_idx = int(np.argmin(np.abs(self.alphas_deg - alpha_val)))
            return self.phis_deg, arr[h_idx, a_idx, :]
        raise ValueError("Specify exactly two of (h_val, alpha_val, phi_val) to obtain a 1D slice.")

    def compute_stability_derivatives(self) -> dict[str, Any]:
        """Compute the derivatives, the aerodynamic centres and the Irodov margin.

        Returns
        -------
        dict
            The positions are in metres aft of the reference point. The
            keys include:

            * ``CL_alpha``, ``Cm_alpha`` [1/rad] and ``x_ac`` [m] vs height
              (linear fit over abs(alpha) <= 8 deg, phi = 0).
            * ``CL_h``, ``Cm_h``: derivatives with respect to h/c, shape
              (n_h, n_alpha).
            * ``x_alpha``, ``x_h``: local pitch and height aerodynamic
              centres [m], shape (n_h, n_alpha).
            * ``irodov_margin``: (x_alpha - x_h)/c. It must be > 0 for
              static stability in ground effect (Irodov criterion). Both
              centres are about the moment reference point
              (``pivot_point_m``) with its height held fixed, for every
              ``height_ref``. The criterion applies only when that point is
              the centre of gravity.
            * ``Cl_phi``, ``Cn_phi``, ``CY_phi`` [1/rad] vs height (bank fit
              over abs(phi) <= 5 deg at the alpha nearest 4 deg). A
              bank-restoring moment has Cl_phi < 0.
        """
        out: dict[str, Any] = {}
        h_vals = self.heights
        c = self.c_ref if self.c_ref > 0 else 1.0
        h_over_c = h_vals / c
        out.update(heights=h_vals, h_over_c=h_over_c, alphas_deg=self.alphas_deg, phis_deg=self.phis_deg)
        out["pivot_point_m"] = None if self.ref_point is None else np.asarray(self.ref_point, dtype=float)
        out["note"] = ("Pitch and height derivatives are about the moment reference point, with the height "
                       "of that point held fixed (also when height_ref is not 'ref'). Height derivatives "
                       "are at constant alpha. End points of the height grid use one-sided (first-order) "
                       "differences. The Irodov criterion applies when the moment reference point is the "
                       "centre of gravity.")
        p0 = int(np.argmin(np.abs(self.phis_deg)))
        a_rad = np.radians(self.alphas_deg)

        if len(self.alphas_deg) >= 2:
            mask = np.abs(self.alphas_deg) <= 8.0
            if np.sum(mask) < 2:
                mask = np.ones_like(self.alphas_deg, dtype=bool)
            CL_a = np.array([_nan_slope(a_rad[mask], self.CL[hi, mask, p0]) for hi in range(len(h_vals))])
            Cm_a = np.array([_nan_slope(a_rad[mask], self.Cm[hi, mask, p0]) for hi in range(len(h_vals))])
            with np.errstate(divide="ignore", invalid="ignore"):
                x_ac = np.where(np.abs(CL_a) > 1e-9, -Cm_a / CL_a * c, np.nan)
            out.update(CL_alpha=CL_a, CL_alpha_deg=CL_a * np.pi / 180.0,
                       Cm_alpha=Cm_a, Cm_alpha_deg=Cm_a * np.pi / 180.0, x_ac=x_ac)

        if len(self.phis_deg) >= 2:
            phi_rad = np.radians(self.phis_deg)
            a_nom = int(np.argmin(np.abs(self.alphas_deg - 4.0))) if len(self.alphas_deg) else 0
            mask = np.abs(self.phis_deg) <= 5.0
            if np.sum(mask) < 2:
                mask = np.ones_like(self.phis_deg, dtype=bool)
            Cl_phi = np.array([_nan_slope(phi_rad[mask], self.Cl[hi, a_nom, mask]) for hi in range(len(h_vals))])
            Cn_phi = np.array([_nan_slope(phi_rad[mask], self.Cn[hi, a_nom, mask]) for hi in range(len(h_vals))])
            CY_phi = np.array([_nan_slope(phi_rad[mask], self.CY[hi, a_nom, mask]) for hi in range(len(h_vals))])
            out.update(Cl_phi=Cl_phi, Cl_phi_deg=Cl_phi * np.pi / 180.0,
                       Cn_phi=Cn_phi, Cn_phi_deg=Cn_phi * np.pi / 180.0, CY_phi=CY_phi)

        if len(h_over_c) >= 2:
            CL_h = np.gradient(self.CL[:, :, p0], h_over_c, axis=0)
            Cm_h = np.gradient(self.Cm[:, :, p0], h_over_c, axis=0)
            out.update(CL_h=CL_h, Cm_h=Cm_h)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_h = np.where(np.abs(CL_h) > 1e-12, -Cm_h / CL_h * c, np.nan)
            out["x_h"] = x_h
            if len(self.alphas_deg) >= 2:
                CL_al = np.gradient(self.CL[:, :, p0], a_rad, axis=1)
                Cm_al = np.gradient(self.Cm[:, :, p0], a_rad, axis=1)
                if self.height_ref != "ref":
                    # Chain rule to constant height of the reference point (see the module text).
                    if self.h_ref_grid is None:
                        nan = np.full_like(CL_al, np.nan)
                        CL_al, Cm_al = nan, nan
                        out["note"] += (" The heights of the moment reference point are not known for "
                                        f"height_ref='{self.height_ref}', so x_alpha and the Irodov "
                                        "margin are not available.")
                    else:
                        dhr = np.gradient(self.h_ref_grid[:, :, p0] / c, a_rad, axis=1)
                        CL_al = CL_al - CL_h * dhr
                        Cm_al = Cm_al - Cm_h * dhr
                with np.errstate(divide="ignore", invalid="ignore"):
                    x_alpha = np.where(np.abs(CL_al) > 1e-12, -Cm_al / CL_al * c, np.nan)
                out["x_alpha"] = x_alpha
                out["irodov_margin"] = (x_alpha - x_h) / c
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary (NaN and infinity become None)."""
        d = {
            "heights": self.heights, "alphas_deg": self.alphas_deg, "phis_deg": self.phis_deg,
            "CL": self.CL, "CDi": self.CDi, "CD": self.CD, "CY": self.CY, "Cl": self.Cl, "Cm": self.Cm,
            "Cn": self.Cn, "L_over_D": self.L_over_D, "e": self.e, "h_min": self.h_min,
            "is_strike": self.is_strike, "refused": self.refused, "phi_strike_limit": self.phi_strike_limit,
            "strike_limit_found": self.strike_limit_found, "n_chord": self.n_chord,
            "errors": {f"{k[0]},{k[1]},{k[2]}": v for k, v in self.errors.items()},
            "S_ref": self.S_ref, "b_ref": self.b_ref, "c_ref": self.c_ref,
            "height_ref": self.height_ref, "execution_time": self.execution_time,
            "derivatives": self.compute_stability_derivatives(),
        }
        return json_safe(d)


class GroundEffectSweep:
    """Sweep over heights, angles of attack and bank angles."""

    def __init__(
        self,
        geometry: Aircraft | LiftingSurface,
        n_workers: int | str = "auto",
        backend: str = "auto",
        solver: str | None = None,
        settings: SolverSettings | None = None,
        n_panels: int = 80,
        spacing: str = "auto",
    ):
        self.aircraft = _as_aircraft(geometry)
        self.n_workers = n_workers
        self.backend = (backend or "auto").lower()
        self.solver = solver
        self.settings = settings
        self.n_panels = n_panels
        self.spacing = spacing

    def run_sweep(
        self,
        heights: Sequence[float] | np.ndarray,
        alphas_deg: Sequence[float] | np.ndarray = (4.0,),
        phis_deg: Sequence[float] | np.ndarray = (0.0,),
        *,
        V_inf: float = 50.0,
        rho: float = RHO_SL,
        ref_point: np.ndarray | Sequence[float] | None = None,
        height_ref: Literal["ref", "min", "qc", "te"] = "ref",
        compute_strike_limit: bool = True,
        progress: bool = False,
    ) -> GroundEffectSweepResult:
        """Run every combination of height, angle of attack and bank angle.

        Raises ValueError if a height is not a finite number larger than 0,
        or if an angle is not finite.
        """
        t0 = time.perf_counter()
        h_arr = np.atleast_1d(np.asarray(heights, dtype=float))
        a_arr = np.atleast_1d(np.asarray(alphas_deg, dtype=float))
        p_arr = np.atleast_1d(np.asarray(phis_deg, dtype=float))
        for hv in h_arr:
            check_height(hv)
        for name, arr in (("alphas_deg", a_arr), ("phis_deg", p_arr)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite numbers (got {arr.tolist()}).")
        shape = (len(h_arr), len(a_arr), len(p_arr))

        ac = self.aircraft
        ac.compute_reference_values()
        sett = self.settings.clone() if self.settings is not None else SolverSettings(
            solver_type=self.solver or "auto", n_panels=self.n_panels, spacing=self.spacing,
        )
        if self.solver is not None:
            sett.solver_type = self.solver
        rp = ac.moment_reference() if ref_point is None else np.asarray(ref_point, dtype=float)
        if sett.n_chord is None:
            sett.n_chord = _fixed_n_chord(ac, sett, h_arr, a_arr, p_arr, rp, height_ref)

        grids = {k: np.full(shape, np.nan) for k in
                 ("CL", "CDi", "CD", "CY", "Cl", "Cm", "Cn", "L_over_D", "e", "h_min", "h_ref", "phi_strike_limit")}
        strike = np.zeros(shape, dtype=bool)
        refused = np.zeros(shape, dtype=bool)
        results: list[GroundEffectResult | None] = [None] * int(np.prod(shape))

        found = np.zeros(shape, dtype=bool)
        limits: dict[tuple[int, int], float] = {}
        if compute_strike_limit:
            for hi, hv in enumerate(h_arr):
                for ai, av in enumerate(a_arr):
                    limits[(hi, ai)] = find_bank_strike_limit(
                        ac, sett, float(hv), float(av), 0.0, rp, height_ref, max_bank_deg=MAX_BANK_DEG)

        cases = [(hi, ai, pi) for hi in range(shape[0]) for ai in range(shape[1]) for pi in range(shape[2])]
        batch = _gpu_batch(ac, sett, cases, h_arr, a_arr, p_arr, V_inf, rho, rp, height_ref)

        def run(case: tuple[int, int, int]):
            hi, ai, pi = case
            try:
                res = analyze_ground_effect(
                    ac, h=float(h_arr[hi]), alpha_deg=float(a_arr[ai]), phi_deg=float(p_arr[pi]),
                    V_inf=V_inf, rho=rho, ref_point=rp, height_ref=height_ref,
                    settings=sett, compute_strike_limit=False,
                )
                return case, res, None
            except (GroundStrikeError, ValidityError) as exc:
                return case, None, exc

        backend = "thread" if self.backend == "auto" else self.backend
        if backend not in ("thread", "serial"):
            raise ValueError(f"backend={self.backend!r}: use 'auto', 'thread' or 'serial'.")
        if batch is None:
            batch = _cpu_batch(ac, sett, cases, h_arr, a_arr, p_arr, V_inf, rho, rp, height_ref, backend=backend)
        if batch is not None:
            outputs = batch
        elif backend == "serial":
            outputs = [run(c) for c in cases]
        else:
            outputs = run_cases(run, cases, estimate_panels(ac, sett), self.n_workers)
        errors: dict[tuple[int, int, int], str] = {}

        for count, (case, res, err) in enumerate(outputs):
            hi, ai, pi = case
            if (hi, ai) in limits:
                grids["phi_strike_limit"][hi, ai, pi] = limits[(hi, ai)]
                found[hi, ai, pi] = limits[(hi, ai)] < MAX_BANK_DEG
            if res is None:
                if isinstance(err, GroundStrikeError):
                    strike[hi, ai, pi] = True
                else:
                    refused[hi, ai, pi] = True
                errors[case] = str(err)
                continue
            for k in ("CL", "CDi", "CD", "CY", "Cl", "Cm", "Cn", "L_over_D", "e", "h_min", "h_ref"):
                grids[k][hi, ai, pi] = getattr(res, k)
            res.phi_strike_limit = limits.get((hi, ai))
            res.strike_limit_found = bool(found[hi, ai, pi]) if (hi, ai) in limits else None
            results[hi * shape[1] * shape[2] + ai * shape[2] + pi] = res
            if progress and (count + 1) % max(1, len(cases) // 10) == 0:
                print(f"Ground effect sweep: {count + 1}/{len(cases)}")

        return GroundEffectSweepResult(
            heights=h_arr, alphas_deg=a_arr, phis_deg=p_arr,
            results=results, grid_shape=shape, errors=errors,
            CL=grids["CL"], CDi=grids["CDi"], CD=grids["CD"], CY=grids["CY"],
            Cl=grids["Cl"], Cm=grids["Cm"], Cn=grids["Cn"], L_over_D=grids["L_over_D"],
            e=grids["e"], h_min=grids["h_min"], is_strike=strike, refused=refused,
            phi_strike_limit=grids["phi_strike_limit"], strike_limit_found=found, n_chord=sett.n_chord,
            S_ref=float(ac.S_ref), b_ref=float(ac.b_ref), c_ref=float(ac.c_ref),
            ref_point=rp, height_ref=height_ref, h_ref_grid=grids["h_ref"],
            execution_time=float(time.perf_counter() - t0),
        )


def _prepare_sweep_cases(
    ac: Aircraft,
    sett: SolverSettings,
    cases: list[tuple[int, int, int]],
    h_arr: np.ndarray,
    a_arr: np.ndarray,
    p_arr: np.ndarray,
    V_inf: float,
    rho: float,
    rp: np.ndarray,
    height_ref: str,
) -> tuple[Any, list[tuple[tuple[int, int, int], dict[str, Any]]], dict[tuple[int, int, int], tuple[Any, None, Exception]]]:
    """Check the cases of a sweep and place their ground, with one shared probe lattice.

    The probe lattice comes from :func:`ventorum.ground_effect.solver.place_ground`
    (from the lattice cache). Each case is prepared with
    :func:`ventorum.ground_effect.solver.prepare_ground_case`, as
    :func:`ventorum.ground_effect.analyze_ground_effect` does. This helper
    is shared by the GPU and the CPU batches of the sweep.

    Parameters
    ----------
    ac : Aircraft
        The aircraft.
    sett : SolverSettings
        Solver settings of the sweep.
    cases : list of tuple of int
        Grid indices ``(height, angle of attack, bank angle)`` of the cases.
    h_arr : numpy.ndarray
        Heights of the sweep [m].
    a_arr : numpy.ndarray
        Angles of attack of the sweep [deg].
    p_arr : numpy.ndarray
        Bank angles of the sweep [deg].
    V_inf : float
        Free-stream speed [m/s].
    rho : float
        Air density [kg/m^3].
    rp : numpy.ndarray
        Moment reference point [m], shape (3,).
    height_ref : str
        Convention of the heights (``"ref"``, ``"min"``, ``"qc"`` or ``"te"``).

    Returns
    -------
    probe : VortexLattice
        The probe lattice (one chordwise panel, read-only).
    prepared : list of tuple
        ``(case, data)`` of each valid case, in the order of *cases*. *data*
        is the dictionary of ``prepare_ground_case`` (flight condition,
        ground plane and the other values of the case).
    outputs : dict
        ``case -> (case, None, error)`` for each case that fails a check
        (ground strike or validity error).
    """
    from ventorum.ground_effect.solver import place_ground, prepare_ground_case

    first = cases[0]
    _, probe = place_ground(ac, sett, float(h_arr[first[0]]), 0.0, 0.0, 0.0, rp, height_ref)
    prepared: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    outputs: dict[tuple[int, int, int], tuple[Any, None, Exception]] = {}
    for case in cases:
        hi, ai, pi = case
        try:
            prepared.append((case, prepare_ground_case(
                ac, float(h_arr[hi]), float(a_arr[ai]), float(p_arr[pi]),
                V_inf=V_inf, rho=rho, ref_point=rp,
                height_ref=height_ref, settings=sett, probe=probe,
            )))
        except (GroundStrikeError, ValidityError) as exc:
            outputs[case] = (case, None, exc)
    return probe, prepared, outputs


def _gpu_batch(ac, sett, cases, h_arr, a_arr, p_arr, V_inf, rho, rp, height_ref) -> list | None:
    """Solve all cases of a sweep as one GPU batch; return the outputs ``(case, result, error)`` or None.

    The GPU takes the batch when :func:`ventorum.gpu.use_gpu` selects it
    for its work (all cases on the lattice of the fixed chordwise count).
    Each case is checked and its ground is placed as in
    :func:`ventorum.ground_effect.analyze_ground_effect`; the cases that
    fail a check keep their error, the others are solved together. None
    means the CPU path.
    """
    from ventorum import gpu
    from ventorum.gpu.pipeline import work_estimate
    from ventorum.ground_effect.solver import ground_case_result
    from ventorum.solvers.factory import make_solver, resolve_solver_type

    if gpu.get_device() == "cpu" or not cases:
        return None
    canonical = resolve_solver_type(sett.solver_type)
    if canonical not in ("vlm", "linear", "nonlinear"):
        return None
    solver = make_solver(canonical)
    first = cases[0]
    probe_cond = FlightCondition(V_inf=V_inf, alpha=float(np.radians(a_arr[first[1]])), rho=rho)
    lattice = solver.build(ac, sett, probe_cond, None, rp)
    if not gpu.use_gpu(work_estimate(lattice, len(cases), True), canonical, len(cases), lattice.n_panels):
        return None
    t0 = time.perf_counter()
    _, prepared, outputs = _prepare_sweep_cases(ac, sett, cases, h_arr, a_arr, p_arr, V_inf, rho, rp, height_ref)
    if prepared:
        conds = [p["condition"] for _, p in prepared]
        grounds = [p["ground"] for _, p in prepared]
        results = solver.solve_batch(lattice, conds, sett, ac.S_ref, ac.b_ref, ac.c_ref, ref_point=rp,
                                     main_surface=ac.main_surface_index(), continuation=False, grounds=grounds)
        share = (time.perf_counter() - t0) / len(prepared)
        for (case, p), res in zip(prepared, results):
            outputs[case] = (case, ground_case_result(p, res, False, time.perf_counter() - share), None)
    return [outputs[c] for c in cases]


_MAX_BATCH_BYTES = 256 * 1024 * 1024


def _chunk_size(canonical: str, n_unknowns: int) -> int:
    """Return the largest number of cases K keeping matrices within 256 MiB."""
    factor = 4 if canonical in ("linear", "nonlinear") else 1
    per_case = factor * (n_unknowns ** 2) * 8
    if per_case <= 0:
        return 1
    return max(1, int(_MAX_BATCH_BYTES // per_case))


def _cpu_batch(
    ac: Aircraft,
    sett: SolverSettings,
    cases: list[tuple[int, int, int]],
    h_arr: np.ndarray,
    a_arr: np.ndarray,
    p_arr: np.ndarray,
    V_inf: float,
    rho: float,
    rp: np.ndarray,
    height_ref: str,
    backend: str = "auto",
) -> list | None:
    """Solve the cases of a sweep as batches on the CPU, or return None.

    The batch is used when *backend* is ``"serial"`` or when the lattice is
    of the size class "small" (:func:`ventorum.hardware.profile.size_class`).
    Each case is checked and its ground is placed as in
    :func:`ventorum.ground_effect.analyze_ground_effect`
    (``_prepare_sweep_cases``); the cases that fail a check keep their
    error. The valid cases are put into groups with the same unknown map
    (symmetric and not symmetric cases), because one batch solves only
    cases with the same unknowns. Each group is solved in chunks that keep
    the matrices of one chunk within ``_MAX_BATCH_BYTES``, with the CPU
    branch of the solver (``LatticeSolver.solve_batch`` with ``_cpu_only=True``, no
    continuation). This function does not change the device setting of
    :mod:`ventorum.gpu`.

    Parameters
    ----------
    ac : Aircraft
        The aircraft.
    sett : SolverSettings
        Solver settings of the sweep (with the fixed chordwise count).
    cases : list of tuple of int
        Grid indices ``(height, angle of attack, bank angle)`` of the cases.
    h_arr : numpy.ndarray
        Heights of the sweep [m].
    a_arr : numpy.ndarray
        Angles of attack of the sweep [deg].
    p_arr : numpy.ndarray
        Bank angles of the sweep [deg].
    V_inf : float
        Free-stream speed [m/s].
    rho : float
        Air density [kg/m^3].
    rp : numpy.ndarray
        Moment reference point [m], shape (3,).
    height_ref : str
        Convention of the heights (``"ref"``, ``"min"``, ``"qc"`` or ``"te"``).
    backend : str, optional
        Parallel backend of the sweep (``"thread"`` or ``"serial"``).

    Returns
    -------
    list of tuple or None
        The outputs ``(case, result, error)`` in the order of *cases*, or
        None when the sweep uses the case-by-case path (no cases, a solver
        without a lattice batch, or a lattice that is not small with a
        backend other than ``"serial"``).
    """
    from ventorum.ground_effect.solver import ground_case_result
    from ventorum.hardware.profile import size_class
    from ventorum.solvers.core import _unknown_map
    from ventorum.solvers.factory import make_solver, resolve_solver_type

    if not cases:
        return None
    canonical = resolve_solver_type(sett.solver_type)
    if canonical not in ("vlm", "linear", "nonlinear"):
        return None
    solver = make_solver(canonical)
    first = cases[0]
    probe_cond = FlightCondition(V_inf=V_inf, alpha=float(np.radians(a_arr[first[1]])), rho=rho)
    lattice = solver.build(ac, sett, probe_cond, None, rp)
    if backend != "serial" and size_class(lattice.n_panels) != "small":
        return None

    t0 = time.perf_counter()
    _, prepared, outputs = _prepare_sweep_cases(ac, sett, cases, h_arr, a_arr, p_arr, V_inf, rho, rp, height_ref)
    if prepared:
        # One batch needs one unknown map (the same object, as the batch
        # solvers check): group the cases by their map. The symmetric map
        # has about half the unknowns. The groups keep the order of the cases.
        use_sym = getattr(sett, "use_symmetry", True)
        groups: dict[int, list[int]] = {}
        n_of_group: dict[int, int] = {}
        for idx, (_, p) in enumerate(prepared):
            umap = _unknown_map(lattice, p["condition"], p["ground"], use_sym)
            groups.setdefault(id(umap), []).append(idx)
            n_of_group[id(umap)] = umap.n
        all_results: list[Any] = [None] * len(prepared)
        for key, members in groups.items():
            chunk_sz = _chunk_size(canonical, n_of_group[key])
            for k0 in range(0, len(members), chunk_sz):
                chunk = members[k0: k0 + chunk_sz]
                # The CPU branch of solve_batch: no GPU check, no change of the
                # device setting (safe when sweeps run in parallel threads).
                chunk_res = solver.solve_batch(
                    lattice, [prepared[i][1]["condition"] for i in chunk], sett, ac.S_ref, ac.b_ref, ac.c_ref,
                    ref_point=rp, main_surface=ac.main_surface_index(),
                    continuation=False, grounds=[prepared[i][1]["ground"] for i in chunk], _cpu_only=True,
                )
                for i, res in zip(chunk, chunk_res):
                    all_results[i] = res

        share = (time.perf_counter() - t0) / len(prepared)
        for (case, p), res in zip(prepared, all_results):
            outputs[case] = (case, ground_case_result(p, res, False, time.perf_counter() - share), None)

    return [outputs[c] for c in cases]


def _fixed_n_chord(
    aircraft: Aircraft,
    settings: SolverSettings,
    heights: np.ndarray,
    alphas_deg: np.ndarray,
    phis_deg: np.ndarray,
    ref_point: np.ndarray,
    height_ref: str,
) -> int | None:
    """Chordwise panel count for all cases of a sweep.

    It is the count that the solver selects for the case with the smallest
    positive gap to the ground (the lowest height). Cases that touch the
    ground are ignored. Returns None if the solver has no chordwise panels
    or if all cases touch the ground.
    """
    try:
        solver = make_solver(resolve_solver_type(settings.solver_type))
    except (ValueError, KeyError):
        return None
    if not hasattr(solver, "resolve_n_chord"):
        return None
    from ventorum.geometry import lattice_cache

    chord_spacing = getattr(settings, "chord_spacing", "uniform")
    key = lattice_cache.lattice_key(aircraft, settings, "vlm", 1, chord_spacing)
    probe = lattice_cache.get_or_build(
        key, lambda: build_lattice(aircraft, settings, collocation="vlm", n_chord=1, chord_spacing=chord_spacing)
    )
    pts = probe.all_points()
    p_plane, mode = plane_reference(aircraft, ref_point, height_ref)
    best_gap, best_gp = np.inf, None
    for hv in heights:
        for av in alphas_deg:
            for pv in phis_deg:
                gp = make_ground_plane(probe, float(hv), np.radians(av), 0.0, np.radians(pv),
                                       ref_point=p_plane, height_ref=mode)
                gap = float(np.min(gp.height(pts)))
                if 0.0 < gap < best_gap:
                    best_gap, best_gp = gap, gp
    if best_gp is None:
        return None
    cond = FlightCondition()
    return int(solver.resolve_n_chord(aircraft, settings, cond, best_gp, ref_point))


def _split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    keys = ("solver", "settings", "n_panels", "spacing")
    return ({k: v for k, v in kwargs.items() if k in keys},
            {k: v for k, v in kwargs.items() if k not in keys})


def sweep_height(
    geometry: Aircraft | LiftingSurface,
    heights: Sequence[float] | np.ndarray,
    alpha_deg: float = 4.0,
    phi_deg: float = 0.0,
    n_workers: int | str = "auto",
    backend: str = "auto",
    **kwargs,
) -> GroundEffectSweepResult:
    """Sweep the height at fixed angle of attack and bank.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        The geometry.
    heights : sequence of float
        Heights above the ground [m] (each > 0), in the convention of ``height_ref``.
    alpha_deg : float
        Angle of attack [deg].
    phi_deg : float
        Bank angle [deg].
    n_workers : int or str
        Number of parallel workers, or ``"auto"``.
    backend : str
        Parallel backend, or ``"auto"``.
    **kwargs
        Solver options (``solver``, ``settings``, ``n_panels``, ``spacing``)
        and options of :meth:`GroundEffectSweep.run_sweep` (for example
        ``V_inf`` [m/s], ``rho`` [kg/m^3], ``ref_point`` [m],
        ``height_ref``).

    Returns
    -------
    GroundEffectSweepResult
        The grids of results.
    """
    sk, rk = _split_kwargs(kwargs)
    return GroundEffectSweep(geometry, n_workers=n_workers, backend=backend, **sk).run_sweep(
        heights=heights, alphas_deg=[alpha_deg], phis_deg=[phi_deg], **rk)


def sweep_roll(
    geometry: Aircraft | LiftingSurface,
    phis_deg: Sequence[float] | np.ndarray,
    heights: Sequence[float] | np.ndarray = (0.5, 1.0, 2.0),
    alpha_deg: float = 4.0,
    n_workers: int | str = "auto",
    backend: str = "auto",
    **kwargs,
) -> GroundEffectSweepResult:
    """Sweep the bank angle at several heights.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        The geometry.
    phis_deg : sequence of float
        Bank angles [deg].
    heights : sequence of float
        Heights above the ground [m] (each > 0), in the convention of ``height_ref``.
    alpha_deg : float
        Angle of attack [deg].
    n_workers : int or str
        Number of parallel workers, or ``"auto"``.
    backend : str
        Parallel backend, or ``"auto"``.
    **kwargs
        Solver options (``solver``, ``settings``, ``n_panels``, ``spacing``)
        and options of :meth:`GroundEffectSweep.run_sweep` (for example
        ``V_inf`` [m/s], ``rho`` [kg/m^3], ``ref_point`` [m],
        ``height_ref``).

    Returns
    -------
    GroundEffectSweepResult
        The grids of results.
    """
    sk, rk = _split_kwargs(kwargs)
    return GroundEffectSweep(geometry, n_workers=n_workers, backend=backend, **sk).run_sweep(
        heights=heights, alphas_deg=[alpha_deg], phis_deg=phis_deg, **rk)


def sweep_alpha(
    geometry: Aircraft | LiftingSurface,
    alphas_deg: Sequence[float] | np.ndarray,
    heights: Sequence[float] | np.ndarray = (0.5, 1.0, 2.0),
    phi_deg: float = 0.0,
    n_workers: int | str = "auto",
    backend: str = "auto",
    **kwargs,
) -> GroundEffectSweepResult:
    """Sweep the angle of attack at several heights.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        The geometry.
    alphas_deg : sequence of float
        Angles of attack [deg].
    heights : sequence of float
        Heights above the ground [m] (each > 0), in the convention of ``height_ref``.
    phi_deg : float
        Bank angle [deg].
    n_workers : int or str
        Number of parallel workers, or ``"auto"``.
    backend : str
        Parallel backend, or ``"auto"``.
    **kwargs
        Solver options (``solver``, ``settings``, ``n_panels``, ``spacing``)
        and options of :meth:`GroundEffectSweep.run_sweep` (for example
        ``V_inf`` [m/s], ``rho`` [kg/m^3], ``ref_point`` [m],
        ``height_ref``).

    Returns
    -------
    GroundEffectSweepResult
        The grids of results.
    """
    sk, rk = _split_kwargs(kwargs)
    return GroundEffectSweep(geometry, n_workers=n_workers, backend=backend, **sk).run_sweep(
        heights=heights, alphas_deg=alphas_deg, phis_deg=[phi_deg], **rk)
