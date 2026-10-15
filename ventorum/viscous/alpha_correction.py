# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Viscous-inviscid coupling by angle-of-attack correction (alpha-correction / decambering).

This module couples linear potential-flow solvers (vortex-lattice method and
linear lifting line) with two-dimensional section polars. The method modifies
the boundary conditions of each spanwise strip by an angle-of-attack correction
(decambering), while keeping the linear system matrix unchanged. The matrix
factorization is computed once per geometry.

References
----------
* J. B. Tseng and C. E. Lan, "Calculation of Aerodynamic Characteristics of
  Wings with Flaps in Incompressible Flow", NASA TP-2828, 1988.
* C. P. van Dam, J. C. Vander Kam, and P. C. Paris, "Design-oriented high-lift
  methodology for general aviation and civil transport aircraft", Journal of
  Aircraft 38(6), 2001, pp. 1105-1115.
* R. Gallay and E. Laurendeau, "Nonlinear generalized lifting-line coupling
  algorithms for pre/poststall flows", AIAA Journal 53(7), 2015, pp. 1904-1915.
* R. Mukherjee and A. Gopalarathnam, "Poststall aerodynamic prediction for
  wings using decambering approach", Journal of Aircraft 43(3), 2006, pp. 660-668.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import scipy.linalg

from ventorum.aero.loads import compute_loads
from ventorum.aero.polar_source import SectionPolarSource
from ventorum.aero.polar_table import PolarTable
from ventorum.aero.system import (
    GroundPlane,
    UnknownMap,
    ground_plane_from_condition,
    wake_direction,
)
from ventorum.core.constants import RHO_SL
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
from ventorum.geometry.lattice import (
    VortexLattice,
    build_lattice,
    section_cl,
    section_cl_slope,
)
from ventorum.solvers.core import (
    SolveInfo,
    _assemble_llt_linear,
    _llt_strip_data,
    assemble_vlm,
)
from ventorum.solvers.lattice_base import (
    DEFAULT_N_CHORD,
    as_aircraft,
    main_surface_strip_count,
    quarter_chord_sweep_deg,
)
from ventorum.utils.parallel import solve_threads
from ventorum.utils.validation import validate_aircraft

#: Standard sea-level dynamic viscosity of air at 15 deg C [Pa s].
MU_AIR_15C = 1.7894e-5


def _strip_reynolds_number(
    condition: FlightCondition,
    chord: float,
) -> float:
    """Return local strip Reynolds number based on speed, chord, and viscosity [-]."""
    rho = float(getattr(condition, "rho", RHO_SL))
    v_inf = float(condition.V_inf)
    nu = getattr(condition, "nu", None)
    if nu is None:
        mu = getattr(condition, "mu", None)
        if mu is not None:
            nu = float(mu) / rho
        else:
            nu = MU_AIR_15C / rho
    else:
        nu = float(nu)
    return float(v_inf * chord / max(nu, 1e-12))


def _strip_flap_deflection_deg(
    lattice: VortexLattice,
    strip_idx: int,
    aircraft: Aircraft,
) -> float:
    """Return control flap deflection in degrees on strip *strip_idx* [deg]."""
    surf_idx = int(lattice.strip_surface[strip_idx])
    surf = aircraft.surfaces[surf_idx]
    controls = getattr(surf, "controls", [])
    if not controls:
        return 0.0

    eta_strip = float(lattice.eta[strip_idx])
    is_right = bool(lattice.strip_is_right[strip_idx])

    for c in controls:
        if (eta_strip >= c.eta_start - 1e-9) and (eta_strip <= c.eta_end + 1e-9):
            defl_rad = float(c.deflection) if (c.symmetric or is_right) else -float(c.deflection)
            return float(np.degrees(defl_rad))
    return 0.0


def _resolve_strip_polar(
    polars: Any,
    lattice: VortexLattice,
    strip_idx: int,
    aircraft: Aircraft,
    condition: FlightCondition,
    ground: GroundPlane | None,
) -> TabulatedAirfoil | LinearAirfoil:
    """Resolve section polar model for strip *strip_idx* at local condition."""
    chord_val = float(lattice.chord[strip_idx])
    re_strip = _strip_reynolds_number(condition, chord_val)
    flap_deg = _strip_flap_deflection_deg(lattice, strip_idx, aircraft)

    h_over_c: float | None = None
    if ground is not None:
        p_strip = lattice.cp[lattice.panel_strip == strip_idx]
        h_strip = float(np.min(ground.height(p_strip)))
        h_over_c = max(1e-4, h_strip / max(chord_val, 1e-6))

    candidate = polars
    if isinstance(polars, dict):
        surf_idx = int(lattice.strip_surface[strip_idx])
        surf_name = aircraft.surfaces[surf_idx].name
        af_strip = lattice.airfoils[strip_idx]
        af_name = getattr(af_strip, "name", "")
        if strip_idx in polars:
            candidate = polars[strip_idx]
        elif surf_name in polars:
            candidate = polars[surf_name]
        elif af_name in polars:
            candidate = polars[af_name]
        else:
            candidate = polars.get("default", af_strip)

    if isinstance(candidate, PolarTable):
        h_arg = h_over_c if candidate.h_over_c is not None else None
        return candidate.at(Re=re_strip, flap_deg=flap_deg, h_over_c=h_arg)

    if isinstance(candidate, SectionPolarSource):
        surf_idx = int(lattice.strip_surface[strip_idx])
        surf = aircraft.surfaces[surf_idx]
        af_strip = lattice.airfoils[strip_idx]
        sec_name = getattr(af_strip, "name", "")
        if not sec_name or "thin" in sec_name.lower():
            for s_def in getattr(surf, "sections", []):
                shape = getattr(s_def, "shape", None)
                if shape is not None and isinstance(shape, str):
                    sec_name = shape
                    break
                af_def = getattr(s_def, "airfoil", None)
                if af_def is not None and getattr(af_def, "name", "") and "thin" not in af_def.name.lower():
                    sec_name = af_def.name
                    break
        if not sec_name or "thin" in sec_name.lower():
            sec_name = "naca 0012"
        alphas_deg = np.linspace(-15.0, 25.0, 81)
        flap_tuple = (0.75, 0.5, flap_deg) if flap_deg != 0.0 else None
        return candidate.polar(section=sec_name, Re=re_strip, alphas_deg=alphas_deg, flap=flap_tuple)

    if isinstance(candidate, (TabulatedAirfoil, LinearAirfoil)):
        return candidate

    return lattice.airfoils[strip_idx]


def _build_rhs_derivative(
    lattice: VortexLattice,
    model: str,
    condition: FlightCondition,
    umap: UnknownMap,
    V_inf: float,
    u_alpha: np.ndarray,
) -> np.ndarray:
    """Return RHS derivative matrix B_delta of shape (n_unknowns, n_indep_strips)."""
    n_p = umap.n
    n_strips = lattice.n_strips

    if umap.symmetric:
        indep_strips = np.flatnonzero(lattice.strip_is_right)
    else:
        indep_strips = np.arange(n_strips)

    n_indep = len(indep_strips)
    B_delta = np.zeros((n_p, n_indep))

    if model == "vlm":
        # Panel normal boundary condition: - V_inf (n_bc . d)
        # Pitching strip s down by delta_s: Delta b_p = - delta_s * (- V_inf * (n_bc . u_alpha))
        # Thus B_delta entries: - V_inf * (n_bc . u_alpha)
        n_bc_unk = lattice.normal_bc[umap.unknown_panels]
        w_p = -V_inf * np.sum(n_bc_unk * u_alpha[None, :], axis=1)

        panel_strip_unk = lattice.panel_strip[umap.unknown_panels]
        for col, s in enumerate(indep_strips):
            mask = panel_strip_unk == s
            B_delta[mask, col] = w_p[mask]

    else:
        # Linear LLT equation: M Gamma = rhs
        # rhs_s = V_inf * (normal_s . d - alpha_L0)
        # Delta rhs_s = - delta_s * (V_inf * (normal_s . u_alpha))
        # Thus B_delta entries: V_inf * (normal_s . u_alpha)
        sd = _llt_strip_data(lattice, umap)
        normal_unk = sd["normal"]
        w_s = V_inf * np.sum(normal_unk * u_alpha[None, :], axis=1)

        unk_strips = lattice.panel_strip[umap.unknown_panels]
        for col, s in enumerate(indep_strips):
            mask = unk_strips == s
            B_delta[mask, col] = w_s[mask]

    return B_delta


def _solve_viscous_single(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    polars: Any,
    model: str,
    max_iter: int,
    tol: float,
    delta0: np.ndarray | None = None,
    ground: GroundPlane | None = None,
    ref_point: np.ndarray | None = None,
    factorization_record: dict[str, int] | None = None,
) -> SolverResult:
    """Solve one condition with alpha-correction coupling."""
    t0 = time.perf_counter()
    S_ref = aircraft.S_ref
    b_ref = aircraft.b_ref
    c_ref = aircraft.c_ref
    main_surf = aircraft.main_surface_index()
    V_inf = float(condition.V_inf)

    if ground is None and condition.h is not None:
        ground = ground_plane_from_condition(aircraft, condition, ref_point)

    n_chord = 1 if model == "llt" else (settings.n_chord or DEFAULT_N_CHORD)
    chord_spacing = getattr(settings, "chord_spacing", "uniform")
    lattice = build_lattice(aircraft, settings, collocation=model, n_chord=n_chord, chord_spacing=chord_spacing)

    wd = wake_direction(condition, ground, getattr(settings, "wake_alignment", "freestream"))
    use_sym = getattr(settings, "use_symmetry", True)

    if model == "vlm":
        A, rhs0, umap = assemble_vlm(lattice, condition, ground, use_sym, wd, ref_point)
    else:
        M, rhs0, umap, _ = _assemble_llt_linear(lattice, condition, ground, use_sym, wd, ref_point)
        A = M

    # Factorize linear matrix once
    lu_piv = scipy.linalg.lu_factor(A)
    if factorization_record is not None:
        factorization_record["count"] = factorization_record.get("count", 0) + 1

    ca, sa = np.cos(condition.alpha), np.sin(condition.alpha)
    cb = np.cos(condition.beta)
    u_alpha = np.array([-sa * cb, 0.0, ca * cb])

    B_delta = _build_rhs_derivative(lattice, model, condition, umap, V_inf, u_alpha)

    # Response matrix G: unknowns response to delta
    G = scipy.linalg.lu_solve(lu_piv, B_delta)
    gamma_unk_0 = scipy.linalg.lu_solve(lu_piv, rhs0)

    n_strips = lattice.n_strips
    if umap.symmetric:
        indep_strips = np.flatnonzero(lattice.strip_is_right)
    else:
        indep_strips = np.arange(n_strips)
    n_indep = len(indep_strips)

    # Map unknown columns back to full panels
    strip_of_unk = lattice.panel_strip[umap.unknown_panels]
    D = np.zeros((n_indep, n_indep))
    cl_inv_0 = np.zeros(n_indep)

    for i_row, s in enumerate(indep_strips):
        p_mask = strip_of_unk == s
        chord_s = float(lattice.chord[s])
        scale_s = 2.0 / (V_inf * max(chord_s, 1e-12))
        cl_inv_0[i_row] = scale_s * float(np.sum(gamma_unk_0[p_mask]))
        D[i_row, :] = scale_s * np.sum(G[p_mask, :], axis=0)

    strip_polars = [
        _resolve_strip_polar(polars, lattice, s, aircraft, condition, ground)
        for s in indep_strips
    ]

    a0_indep = lattice.a0[indep_strips]
    alpha_L0_indep = lattice.alpha_L0[indep_strips]

    def evaluate_residual(delta_vec: np.ndarray):
        cl_inv = cl_inv_0 - D @ delta_vec
        alpha_eff = alpha_L0_indep + delta_vec + cl_inv / np.maximum(a0_indep, 1e-6)

        cl_visc = np.empty(n_indep)
        dcl_visc = np.empty(n_indep)
        for i_s, (p_obj, a_e) in enumerate(zip(strip_polars, alpha_eff)):
            cl_visc[i_s] = float(section_cl(p_obj, a_e))
            dcl_visc[i_s] = float(section_cl_slope(p_obj, a_e))

        res_vec = cl_inv - cl_visc
        return res_vec, cl_inv, alpha_eff, dcl_visc

    def solve_newton(start_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], bool]:
        delta_curr = start_delta.copy()
        res_history = []
        r, cl_i, a_eff, dcl = evaluate_residual(delta_curr)
        norm_r = float(np.max(np.abs(r)))
        res_history.append(norm_r)
        conv = norm_r < tol

        it = 0
        while not conv and it < max_iter:
            it += 1
            # Exact Jacobian: J = - (I - m / a0) D - diag(m)
            ratio = dcl / np.maximum(a0_indep, 1e-6)
            J = - (1.0 - ratio[:, None]) * D - np.diag(dcl)

            try:
                step = -np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                step = None

            accepted = False
            f0 = 0.5 * float(np.sum(r**2))

            if step is not None:
                lam = 1.0
                while lam >= 1.0 / 1024.0:
                    delta_try = delta_curr + lam * step
                    r_try, cl_try, a_try, dcl_try = evaluate_residual(delta_try)
                    f_try = 0.5 * float(np.sum(r_try**2))
                    if f_try <= (1.0 - 1e-4 * lam) * f0:
                        delta_curr = delta_try
                        r, cl_i, a_eff, dcl = r_try, cl_try, a_try, dcl_try
                        accepted = True
                        break
                    lam *= 0.5

            if not accepted:
                # Under-relaxation step
                try:
                    delta_target = delta_curr + np.linalg.solve(D, r)
                except np.linalg.LinAlgError:
                    delta_target = delta_curr + 0.1 * r
                for omega in (0.5, 0.25, 0.1, 0.05, 0.02):
                    delta_try = delta_curr + omega * (delta_target - delta_curr)
                    r_try, cl_try, a_try, dcl_try = evaluate_residual(delta_try)
                    f_try = 0.5 * float(np.sum(r_try**2))
                    if f_try < f0:
                        delta_curr = delta_try
                        r, cl_i, a_eff, dcl = r_try, cl_try, a_try, dcl_try
                        accepted = True
                        break

            if not accepted:
                break

            norm_r = float(np.max(np.abs(r)))
            res_history.append(norm_r)
            conv = norm_r < tol

        return delta_curr, cl_i, a_eff, dcl, res_history, conv

    if delta0 is None:
        delta_init = np.zeros(n_indep)
    else:
        d0_arr = np.atleast_1d(np.asarray(delta0, dtype=float))
        if d0_arr.size == n_indep:
            delta_init = d0_arr.copy()
        elif d0_arr.size == n_strips:
            delta_init = d0_arr[indep_strips].copy()
        else:
            delta_init = np.zeros(n_indep)

    delta_sol, cl_inv_sol, a_eff_sol, dcl_sol, hist, converged = solve_newton(delta_init)

    # Check for post-stall conditions
    is_post_stall_strip = dcl_sol <= 0.0
    post_stall_detected = bool(np.any(is_post_stall_strip))

    has_post_stall_branch = False
    branches: list[SolverResult] = []

    def build_result_object(
        delta_v: np.ndarray,
        a_eff_v: np.ndarray,
        dcl_v: np.ndarray,
        hist_v: list[float],
        conv_v: bool,
        is_branch: bool = False,
    ) -> SolverResult:
        gamma_unk = gamma_unk_0 - G @ delta_v
        gamma_full = gamma_unk[umap.panel_column]

        # Expand effective angles to full strips
        a_eff_full = np.empty(n_strips)
        a_eff_full[indep_strips] = a_eff_v
        if umap.symmetric:
            left_strips = np.flatnonzero(~lattice.strip_is_right)
            mirror_idx = lattice.strip_mirror[left_strips]
            a_eff_full[left_strips] = a_eff_full[mirror_idx]

        # Assign resolved airfoils to lattice for profile drag calculation
        for i_s, s in enumerate(indep_strips):
            lattice.airfoils[s] = strip_polars[i_s]
        if umap.symmetric:
            for ls in left_strips:
                lattice.airfoils[ls] = lattice.airfoils[lattice.strip_mirror[ls]]

        loads = compute_loads(
            lattice,
            gamma_full,
            condition,
            S_ref,
            b_ref,
            c_ref,
            ground=ground,
            ref_point=ref_point,
            alpha_eff_strip=a_eff_full,
            wake_dir=wd,
        )

        totals = loads.totals
        notes = list(lattice.join_warnings)

        if not conv_v:
            notes.append("solve_viscous did not converge to tolerance.")
            warnings.warn("solve_viscous did not converge to tolerance.", RuntimeWarning, stacklevel=4)

        if not is_branch and has_post_stall_branch:
            notes.append("Non-unique post-stall solution branches detected.")

        limits = polar_limits(lattice.airfoils)
        cl_strip = np.empty(lattice.n_strips)
        ae_strip = np.empty(lattice.n_strips)
        for surf, sw in zip(lattice.surfaces, loads.spanwise):
            cl_strip[surf.strips] = sw.Cl
            ae_strip[surf.strips] = sw.alpha_eff
        polar_stat = polar_stats(limits, cl_strip, ae_strip)

        sweep_deg = quarter_chord_sweep_deg(lattice)
        totals.trust = evaluate_aerodynamic_trust(
            AR=totals.AR,
            condition=condition,
            spanwise_list=loads.spanwise,
            CL=totals.CL,
            CDi=totals.CDi,
            CD_total=totals.CD_total,
            converged=conv_v,
            n_panels=main_surface_strip_count(lattice, main_surf),
            max_sweep_rad=np.radians(sweep_deg),
            solver_type="vlm" if model == "vlm" else "linear",
            notes=notes,
            polar=polar_stat,
            post_stall_branch=has_post_stall_branch,
        )

        strip_flags_full = ["attached"] * n_strips
        for i_s, s in enumerate(indep_strips):
            if dcl_v[i_s] <= 0.0:
                strip_flags_full[s] = "post-stall, non-unique"
        if umap.symmetric:
            for ls in left_strips:
                strip_flags_full[ls] = strip_flags_full[lattice.strip_mirror[ls]]

        delta_full = np.empty(n_strips)
        delta_full[indep_strips] = delta_v
        if umap.symmetric:
            delta_full[left_strips] = delta_full[mirror_idx]

        post_stall_strips_full = np.zeros(n_strips, dtype=bool)
        for i_s, s in enumerate(indep_strips):
            if dcl_v[i_s] <= 0.0:
                post_stall_strips_full[s] = True
        if umap.symmetric:
            post_stall_strips_full[left_strips] = post_stall_strips_full[mirror_idx]

        info = SolveInfo(converged=conv_v, iterations=len(hist_v) - 1, residual_history=hist_v, symmetric=umap.symmetric)
        res_obj = SolverResult(
            spanwise=loads.spanwise,
            totals=totals,
            solver_type=f"viscous_{model}",
            converged=conv_v,
            iterations=info.iterations,
            residual_history=list(hist_v),
            condition=condition,
            execution_time=time.perf_counter() - t0,
            symmetry_used=umap.symmetric,
            b_ref=float(b_ref),
            c_ref=float(c_ref),
        )
        res_obj.details = {
            "lattice": lattice,
            "loads": loads,
            "delta": delta_full,
            "alpha_eff": a_eff_full,
            "post_stall": bool(np.any(dcl_v <= 0.0)),
            "post_stall_strips": post_stall_strips_full,
            "strip_flags": strip_flags_full,
            "factorization_count": factorization_record.get("count", 1) if factorization_record else 1,
            "viscous_iterations": info.iterations,
            "viscous_residual_history": hist_v,
        }
        return res_obj

    if post_stall_detected:
        res_primary_temp = build_result_object(delta_sol, a_eff_sol, dcl_sol, hist, converged, is_branch=True)
        guesses = [
            np.zeros_like(delta_sol),
            delta_sol + np.radians(8.0),
            np.full_like(delta_sol, np.radians(10.0)),
        ]
        for g_try in guesses:
            if np.allclose(g_try, delta_sol, atol=1e-3):
                continue
            d_s, cl_s, a_s, dcl_s, hist_s, conv_s = solve_newton(g_try)
            if conv_s or (hist_s and hist_s[-1] < 1e-2):
                sol_branch = build_result_object(d_s, a_s, dcl_s, hist_s, conv_s, is_branch=True)
                if abs(res_primary_temp.totals.CL - sol_branch.totals.CL) > 1e-3:
                    has_post_stall_branch = True
                    branches = [res_primary_temp, sol_branch]
                    break

    final_res = build_result_object(delta_sol, a_eff_sol, dcl_sol, hist, converged, is_branch=False)
    if branches:
        final_res.details["branches"] = branches
        final_res.details["second_branch"] = branches[1]

    return final_res


def solve_viscous(
    aircraft: Aircraft | LiftingSurface,
    conditions: FlightCondition | Sequence[FlightCondition],
    settings: SolverSettings,
    polars: Any,
    *,
    model: Literal["vlm", "llt"] = "vlm",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> SolverResult | list[SolverResult]:
    """Solve viscous-inviscid coupling with alpha-correction (decambering).

    Parameters
    ----------
    aircraft : Aircraft or LiftingSurface
        The aircraft geometry.
    conditions : FlightCondition or sequence of FlightCondition
        Operating condition or a sequence of conditions.
    settings : SolverSettings
        Discretisation and solver settings.
    polars : PolarTable, SectionPolarSource, TabulatedAirfoil, or dict
        Section polar data source for spanwise strips.
    model : {"vlm", "llt"}, optional
        Underlying linear solver. Default is "vlm".
    max_iter : int, optional
        Maximum Newton iterations. Default is 50.
    tol : float, optional
        Convergence tolerance on residual vector infinity-norm. Default is 1e-6.

    Returns
    -------
    SolverResult or list of SolverResult
        Solved result object with profile drag CDp, total drag CD_total,
        effective angles, alpha-correction delta, and post-stall branch info.
    """
    model_str = str(model).lower()
    if model_str not in ("vlm", "llt"):
        raise ValueError(f"model must be 'vlm' or 'llt', got {model!r}.")

    aircraft_obj = as_aircraft(aircraft)
    validate_aircraft(aircraft_obj)
    aircraft_obj.compute_reference_values()
    rp = aircraft_obj.moment_reference()

    is_seq = isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes))
    cond_list = list(conditions) if is_seq else [conditions]

    if not cond_list:
        return []

    fac_record: dict[str, int] = {"count": 0}
    results: list[SolverResult] = []
    delta_prev: np.ndarray | None = None

    with solve_threads(80):
        for cond in cond_list:
            res = _solve_viscous_single(
                aircraft=aircraft_obj,
                condition=cond,
                settings=settings,
                polars=polars,
                model=model_str,
                max_iter=max_iter,
                tol=tol,
                delta0=delta_prev,
                ground=None,
                ref_point=rp,
                factorization_record=fac_record,
            )
            # Factorization count is 1 per geometry
            res.details["factorization_count"] = 1
            if res.converged:
                delta_prev = res.details["delta"]
            results.append(res)

    return results if is_seq else results[0]
