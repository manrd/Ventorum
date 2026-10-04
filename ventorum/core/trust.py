# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Validity checks and trust score for a solution.

The score (0 to 1) adds penalties for conditions where the method in use is
known to be weak. The rules depend on the method:

* Lifting-line methods (``linear``, ``nonlinear``, ``fourier``) need a high
  aspect ratio and no sweep: with sweep the numerical lifting line is not
  grid convergent.
* The vortex-lattice method (``vlm``) handles sweep and low aspect ratio,
  but not the leading-edge vortex of slender, highly swept wings.
* All methods are inviscid and incompressible and use thin-wing theory. In
  ground effect, thickness and viscous effects grow when the gap is small
  compared with the chord; the limits below are stated in h/c.

The uncertainty values (``uncertainty_CL`` and the others) are heuristic
bands. They are NOT calibrated against experimental data. Do not use them as
error bars in a report.
"""

from __future__ import annotations

import math

from ventorum.core.constants import A_SL, MACH_LIMIT_INCOMPRESSIBLE

from collections.abc import Sequence

import numpy as np

from ventorum.core.datatypes import (
    DiscretizedSurface,
    FlightCondition,
    LiftingSurface,
    SpanwiseResult,
    TrustScore,
)

LIFTING_LINE_SOLVERS = ("linear", "linear_llt", "llt", "nonlinear", "fourier")


def _is_finite(value: object) -> bool:
    """Return True when *value* is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count_extrema(g: np.ndarray, rel_tol: float = 1e-6) -> int:
    """Return the number of local extrema of a sequence (tiny steps are ignored)."""
    g = np.asarray(g, dtype=float)
    if g.size < 4:
        return 0
    d = g[1:] - g[:-1]
    scale = max(float(np.abs(g).max()), 1e-300)
    d = d[np.abs(d) > rel_tol * scale]
    if d.size < 2:
        return 0
    s = np.sign(d)
    return int(np.count_nonzero(s[1:] != s[:-1]))


def _count_extrema_rows(G: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Return :func:`_count_extrema` of each row of *G* (shape (K, m)), with the same rules."""
    G = np.asarray(G, dtype=float)
    K, m = G.shape
    if m < 4:
        return np.zeros(K, dtype=int)
    d = G[:, 1:] - G[:, :-1]
    scale = np.maximum(np.abs(G).max(axis=1), 1e-300)
    keep = np.abs(d) > rel_tol * scale[:, None]
    s = np.sign(d)
    # The sign of the last kept step before each step (forward fill).
    idx = np.where(keep, np.arange(m - 1)[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    prev = np.concatenate([np.full((K, 1), -1), last[:, :-1]], axis=1)
    prev_sign = np.take_along_axis(s, np.maximum(prev, 0), axis=1)
    change = keep & (prev >= 0) & (s != prev_sign)
    return change.sum(axis=1).astype(int)


def spanwise_stats_batch(surface_strips: Sequence, arrays: dict[str, np.ndarray]) -> list[dict | None]:
    """Return the spanwise statistics of the trust score for K cases at once.

    Parameters
    ----------
    surface_strips : sequence of slices or index arrays
        The strips of each surface (the split of the spanwise results).
    arrays : dict of (K, n_strips) arrays
        ``gamma``, ``Cl``, ``Cd_i``, ``alpha_eff``, ``alpha_i``,
        ``local_lift`` and, if present, ``Cd_profile`` and ``Cm_section``
        (a value of None means "absent" for all cases).

    Returns
    -------
    list of dict or None
        For each case the values that :func:`evaluate_aerodynamic_trust`
        computes from the spanwise results (``max_local_cl``,
        ``max_alpha_eff_deg``, ``n_extrema``), or None when an array of the
        case is not finite (the caller then passes the spanwise results, so
        that the warning names the array).
    """
    names = [n for n in ("gamma", "Cl", "Cd_i", "alpha_eff", "alpha_i", "local_lift", "Cd_profile", "Cm_section")
             if arrays.get(n) is not None]
    K = arrays["gamma"].shape[0]
    finite = np.ones(K, dtype=bool)
    for n in names:
        finite &= np.isfinite(arrays[n]).all(axis=1)
    cl = np.abs(arrays["Cl"])
    ae = np.abs(arrays["alpha_eff"])
    max_cl = np.zeros(K)
    max_ae = np.zeros(K)
    n_ext = np.zeros(K, dtype=int)
    for sl in surface_strips:
        g = arrays["gamma"][:, sl]
        if g.shape[1] == 0:
            continue
        max_cl = np.maximum(max_cl, cl[:, sl].max(axis=1))
        max_ae = np.maximum(max_ae, np.degrees(ae[:, sl].max(axis=1)))
        n_ext = np.maximum(n_ext, _count_extrema_rows(g))
    return [{"max_local_cl": float(max_cl[k]), "max_alpha_eff_deg": float(max_ae[k]), "n_extrema": int(n_ext[k])}
            if finite[k] else None for k in range(K)]


def evaluate_aerodynamic_trust(
    AR: float,
    surfaces: Sequence[DiscretizedSurface | LiftingSurface] | None = None,
    condition: FlightCondition | None = None,
    spanwise_list: Sequence[SpanwiseResult] | None = None,
    CL: float = 0.0,
    CDi: float = 0.0,
    CD_total: float | None = None,
    converged: bool = True,
    n_panels: int = 80,
    max_sweep_rad: float | None = None,
    *,
    solver_type: str | None = None,
    h_over_c: float | None = None,
    n_chord: int | None = None,
    notes: Sequence[str] | None = None,
    max_chord_over_c: float = 1.0,
    spanwise_stats: dict | None = None,
) -> TrustScore:
    """Evaluate the validity of a solution.

    Parameters
    ----------
    AR : float
        Reference aspect ratio ``b^2 / S``.
    surfaces : optional
        Used only to find the sweep when *max_sweep_rad* is not given.
    condition : FlightCondition, optional
        Speed (for the Mach number check).
    spanwise_list : optional
        Spanwise results (section lift and circulation checks).
    CL, CDi, CD_total : float
        Integrated coefficients.
    converged : bool
        Whether the solver met its tolerance.
    n_panels : int
        Spanwise panels per semi-span of the main surface.
    max_sweep_rad : float, optional
        Representative sweep of the lifting line [rad].
    solver_type : str, optional
        ``"vlm"``, ``"linear"``, ``"nonlinear"`` or ``"fourier"``. If not
        given, the lifting-line rules are used.
    h_over_c : float, optional
        Smallest ground clearance divided by the reference chord.
    n_chord : int, optional
        Chordwise panels of the vortex lattice.
    notes : list of str, optional
        Extra warnings from the solver.
    max_chord_over_c : float
        Largest strip chord divided by the reference chord (for the
        chordwise panel check in ground effect).
    spanwise_stats : dict or None
        The statistics of the spanwise results from
        :func:`spanwise_stats_batch` (all arrays finite). A batch computes
        them for all its cases at once; the result is the same as from
        *spanwise_list*.

    A result with a non-finite coefficient never keeps a trust level above
    LOW: it gets the penalty ``non_finite_result`` and the warning
    "non-finite result". This is a second safety line; the input checks
    must stop the known cases first.
    """
    # Non-finite result guard: any total coefficient that is not a finite
    # number caps the trust level at LOW.
    bad_totals: list[str] = []
    for label, value in (("CL", CL), ("CDi", CDi), ("CD_total", CD_total)):
        if value is not None and not _is_finite(value):
            bad_totals.append(f"{label}={value}")
    for sw in (spanwise_list or ()) if spanwise_stats is None else ():
        arrays = [
            ("gamma", sw.gamma), ("Cl", sw.Cl), ("Cd_i", sw.Cd_i),
            ("alpha_eff", sw.alpha_eff), ("alpha_i", sw.alpha_i),
            ("local_lift", sw.local_lift),
        ]
        if sw.Cd_profile is not None:
            arrays.append(("Cd_profile", sw.Cd_profile))
        if sw.Cm_section is not None:
            arrays.append(("Cm_section", sw.Cm_section))
        # One test for all arrays first (the usual case: all finite); the
        # array-by-array test below only finds the label of the first bad one.
        try:
            all_finite = bool(np.isfinite(np.concatenate(
                [np.asarray(arr, dtype=float).ravel() for _, arr in arrays])).all())
        except ValueError:
            all_finite = False
        if all_finite:
            continue
        for label, arr in arrays:
            values = np.asarray(arr, dtype=float)
            if values.size and not bool(np.all(np.isfinite(values))):
                bad_totals.append(f"{sw.surface_name}.{label}")
                break
    non_finite_result = bool(bad_totals)
    penalties: dict[str, float] = {}
    warnings: list[str] = []
    recommendations: list[str] = []
    if non_finite_result:
        penalties["non_finite_result"] = 0.6
        warnings.append(
            "Non-finite result (non-finite result coefficients: "
            + ", ".join(bad_totals)
            + "): the coefficients are not valid numbers; check the inputs."
        )
    solver = (solver_type or "linear").lower()
    is_llt = solver in LIFTING_LINE_SOLVERS

    # 1. Aspect ratio -----------------------------------------------------------
    ar_pen = 0.0
    if AR <= 0:
        ar_pen = 1.0
        warnings.append("Aspect ratio is non-positive or undefined.")
    elif is_llt and AR < 3.0:
        ar_pen = 0.45 + 0.35 * min(1.0, (3.0 - AR) / 2.0)
        warnings.append(
            f"Very low aspect ratio (AR={AR:.2f} < 3) for a lifting-line method: "
            "chordwise effects are large."
        )
        recommendations.append("Use the VLM solver (solver='vlm') for AR < 3.")
    elif is_llt and AR < 5.0:
        ar_pen = 0.20 * ((5.0 - AR) / 2.0)
        warnings.append(f"Moderate aspect ratio (AR={AR:.2f} < 5) for a lifting-line method.")
        recommendations.append("Compare with the VLM solver for AR < 5.")
    penalties["aspect_ratio"] = ar_pen

    # 2. Sweep ------------------------------------------------------------------
    sweep_pen = 0.0
    max_sweep = 0.0
    if max_sweep_rad is not None:
        max_sweep = float(abs(max_sweep_rad))
    elif surfaces:
        for s in surfaces:
            if hasattr(s, "sweep_le"):
                max_sweep = max(max_sweep, abs(float(getattr(s, "sweep_le", 0.0))))
    sweep_deg = float(np.degrees(max_sweep))
    if is_llt and sweep_deg > 2.5:
        sweep_pen = min(0.8, 0.20 + 0.60 * (sweep_deg - 2.5) / 27.5)
        warnings.append(
            f"Lifting line with sweep angle {sweep_deg:.1f} deg: the numerical lifting line is not "
            "grid convergent with sweep (lift falls as panels are added)."
        )
        recommendations.append("Use the VLM solver (solver='vlm') for swept wings.")
    elif not is_llt and sweep_deg > 55.0 and AR < 2.5:
        sweep_pen = 0.35
        warnings.append(
            f"Slender wing (sweep {sweep_deg:.0f} deg, AR={AR:.2f}): leading-edge vortex lift is not modelled."
        )
    penalties["sweep"] = sweep_pen

    # 3. Section lift and stall ---------------------------------------------------
    stall_pen = 0.0
    max_local_cl = 0.0
    max_alpha_eff_deg = 0.0
    if spanwise_stats is not None:
        max_local_cl = spanwise_stats["max_local_cl"]
        max_alpha_eff_deg = spanwise_stats["max_alpha_eff_deg"]
    elif spanwise_list:
        for sw in spanwise_list:
            if len(sw.Cl) > 0:
                max_local_cl = max(max_local_cl, float(np.abs(sw.Cl).max()))
            if len(sw.alpha_eff) > 0:
                # degrees(max |a|) equals max |degrees(a)|: the scaling is monotonic.
                max_alpha_eff_deg = max(max_alpha_eff_deg, float(np.degrees(np.abs(sw.alpha_eff).max())))
    if max_local_cl == 0.0 and abs(CL) > 0:
        max_local_cl = abs(CL) * 1.25
    linear_model = solver != "nonlinear"
    if max_local_cl > 1.55 or max_alpha_eff_deg > 16.0:
        stall_pen = 0.50 + 0.35 * min(1.0, max(0.0, (max_local_cl - 1.55) / 0.5))
        if linear_model:
            warnings.append(
                f"Severe stall risk: peak section Cl={max_local_cl:.2f} (alpha_eff={max_alpha_eff_deg:.1f} deg). "
                "The linear section model has no stall and over-predicts lift here."
            )
            recommendations.append("Use solver='nonlinear' with section polars (e.g. from XFOIL).")
        else:
            warnings.append(
                f"Post-stall sections: peak section Cl={max_local_cl:.2f} (alpha_eff={max_alpha_eff_deg:.1f} deg). "
                "The lifting-line solution past maximum lift is not unique and not validated."
            )
    elif max_local_cl > 1.20 or max_alpha_eff_deg > 12.0:
        stall_pen = 0.25 * max(0.0, (max_local_cl - 1.20) / 0.35)
        warnings.append(
            f"Approaching section stall: peak section Cl={max_local_cl:.2f} (alpha_eff={max_alpha_eff_deg:.1f} deg)."
        )
    penalties["stall_proximity"] = stall_pen

    # 4. Compressibility (outside the valid envelope) ---------------------------------
    mach_pen = 0.0
    if condition is not None:
        mach = float(condition.V_inf) / A_SL
        if mach > MACH_LIMIT_INCOMPRESSIBLE:
            mach_pen = 0.60 + 0.30 * min(1.0, (mach - MACH_LIMIT_INCOMPRESSIBLE) / 0.35)
            warnings.append(
                f"Out of the valid envelope: Mach={mach:.2f} > {MACH_LIMIT_INCOMPRESSIBLE} (sea-level speed "
                "of sound). The methods are incompressible and no compressibility correction is applied."
            )
            recommendations.append("Reduce the speed, or use a compressible method.")
    penalties["compressibility"] = mach_pen

    # 5. Ground proximity (stated in h/c) ------------------------------------------------
    ground_pen = 0.0
    if h_over_c is not None:
        hc = float(h_over_c)
        if hc < 0.1:
            ground_pen = 0.45
            warnings.append(
                f"Extreme ground proximity (h_min/c={hc:.3f} < 0.1): thin-wing theory without thickness "
                "or viscous effects; real thick sections can lose lift here."
            )
        elif hc < 0.3:
            ground_pen = 0.20
            warnings.append(
                f"Strong ground effect (h_min/c={hc:.3f} < 0.3): thickness and viscous effects are not modelled."
            )
        if not is_llt and n_chord is not None and n_chord > 0 and (max_chord_over_c / n_chord) > hc:
            ground_pen += 0.20
            warnings.append(
                f"Chordwise panels ({n_chord} on the largest chord, {max_chord_over_c:.2f} c_ref) are longer "
                f"than the ground gap (h_min/c={hc:.3f})."
            )
            recommendations.append("Increase n_chord so that c/n_chord < h_min.")
    penalties["ground_proximity"] = ground_pen

    # 6. Numerical health -------------------------------------------------------------------
    num_pen = 0.0
    if not converged:
        num_pen += 0.55
        warnings.append("Solver did not converge to the specified tolerance.")
        recommendations.append("Increase max_iterations, or check for post-stall conditions.")
    if n_panels < 8:
        num_pen += 0.15 * ((8 - n_panels) / 4.0)
        warnings.append(f"Coarse spanwise mesh (n_panels={n_panels} < 8).")
    if spanwise_list or spanwise_stats is not None:
        if spanwise_stats is not None:
            n_extrema = spanwise_stats["n_extrema"]
        else:
            n_extrema = max((_count_extrema(sw.gamma) for sw in spanwise_list), default=0)
        if n_extrema > 6:
            num_pen += 0.40
            warnings.append(
                f"Spanwise circulation oscillates ({n_extrema} local extrema): probable numerical problem."
            )
    penalties["numerical"] = num_pen

    for msg in notes or ():
        if msg not in warnings:
            warnings.append(msg)

    # Composite score ------------------------------------------------------------------------
    total_penalty = min(1.0, sum(penalties.values()))
    score = max(0.0, 1.0 - total_penalty)
    if score >= 0.85:
        rating = "HIGH"
    elif score >= 0.65:
        rating = "MODERATE"
    elif score >= 0.40:
        rating = "LOW"
    else:
        rating = "UNRELIABLE"
    if non_finite_result and rating in ("HIGH", "MODERATE"):
        rating = "LOW"  # a cap: an UNRELIABLE rating stays UNRELIABLE

    # Heuristic bands (not calibrated against data).
    fidelity_factor = 1.0 - score
    unc_cl = max(0.015, abs(CL) * (0.03 + 0.20 * fidelity_factor) + 0.01 * fidelity_factor)
    unc_cdi = max(0.0004, CDi * (0.05 + 0.35 * fidelity_factor) + 0.0003 * fidelity_factor)
    cd_tot = CD_total if CD_total is not None and CD_total > 0 else (CDi if CDi > 1e-6 else 0.02)
    ld_ratio = CL / cd_tot if cd_tot > 0 else 0.0
    unc_ld = max(0.2, abs(ld_ratio) * (0.05 + 0.30 * fidelity_factor))

    return TrustScore(
        score=score,
        rating=rating,
        uncertainty_CL=unc_cl,
        uncertainty_CDi=unc_cdi,
        uncertainty_LD=unc_ld,
        factors=penalties,
        warnings=warnings,
        recommendations=recommendations,
    )
