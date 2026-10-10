# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Classical lifting-line solver: Lanchester–Prandtl theory with Glauert's Fourier series.

The circulation is expanded as

    Gamma(theta) = 2 b V  sum_n A_n sin(n theta),   y = (b/2) cos(theta)

and the monoplane equation is collocated at ``theta_k = k pi / (N + 1)``,
``k = 1..N``:

    sum_n A_n sin(n theta_k) (1 + n mu_k / sin(theta_k)) = mu_k (alpha + twist_k - alpha_L0_k),
    mu_k = c_k a0_k / (4 b)

Results: ``CL = pi AR A_1``, ``CDi = pi AR sum n A_n^2``. The pitching moment
about ``Aircraft.ref_point`` adds the section moment ``Cm0`` and the moment
``r x F`` of the section forces, which act on the quarter-chord line. The
force of a section is ``rho V Gamma dy`` normal to the local flow: its lift
part is normal to the free stream and its induced-drag part is
``alpha_i`` times that value along the free stream. A wing above or below
the reference point therefore gets a moment from both parts.

Limitations: one symmetric, unswept, planar wing; linear section data; no
ground effect; no node displacements (the solver refuses a surface with
``node_displacements`` with ``ValueError``). The solver refuses a wing outside that model with
``ValidityError`` (see ``_check_geometry_in_model``): quarter-chord sweep
above 1 deg on a spanwise interval, dihedral, or a vertical offset of the
sections. Reference: H. Glauert, "The Elements of Aerofoil and Airscrew
Theory", Cambridge University Press, 1926, chapter XI.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    IntegratedResult,
    LiftingSurface,
    LinearAirfoil,
    SolverResult,
    SolverSettings,
    SpanwiseResult,
)
from ventorum.core.errors import ValidityError
from ventorum.core.trust import evaluate_aerodynamic_trust
from ventorum.geometry import lattice_cache
from ventorum.geometry.controls import validate_controls
from ventorum.geometry.discretization import fourier_collocation_angles
from ventorum.geometry.lattice import _AirfoilBlender, _sorted_sections, airfoil_linear_properties, surface_reference_line
from ventorum.solvers.base import BaseSolver
from ventorum.utils.parallel import blas_single_thread
from ventorum.utils.validation import validate_aircraft, validate_fourier_applicability

#: Largest quarter-chord sweep on one spanwise interval that the Fourier
#: solver accepts [deg] (owner decision D-08).
SWEEP_LIMIT_DEG = 1.0
#: Smallest vertical offset that counts as out of plane, as a fraction of
#: the semi-span [m] (owner decision D-08).
PLANAR_TOL_FRACTION = 1.0e-9


def _check_geometry_in_model(surf: LiftingSurface) -> None:
    """Refuse a surface outside the model of the classical Fourier solution.

    Glauert's series solution assumes one straight, unswept lifting line in
    one plane (H. Glauert, "The Elements of Aerofoil and Airscrew Theory",
    Cambridge University Press, 1926, chapter XI). The lifting line is the
    quarter-chord line. Sweep or vertical offsets from the section positions
    change the result and the solver cannot model them: with 35 deg of sweep
    from ``x_le`` its CL is 22 % above the vortex lattice (4th review, item
    5). The section positions are the ones the lattice uses (see
    :func:`ventorum.geometry.lattice.surface_reference_line`).

    Parameters
    ----------
    surf : LiftingSurface
        Surface to check.

    Raises
    ------
    ValueError
        If the surface has node displacements set.
    ValidityError
        If the quarter-chord line sweeps more than ``SWEEP_LIMIT_DEG`` on a
        spanwise interval between two sections, or if a section has a
        vertical offset ``z_le``, or if the surface has dihedral, above
        ``PLANAR_TOL_FRACTION`` times the semi-span. The message names the
        vortex lattice (``solver_type="vlm"``) as the solver to use.
    """
    if getattr(surf, "node_displacements", None) is not None:
        raise ValueError(
            f"The Fourier solver cannot model deformed geometry with node displacements "
            f"(surface '{surf.name}' has node_displacements set). "
            "Use the vortex lattice solver (solver_type='vlm') or lifting line solvers "
            "(solver_type='linear' or 'nonlinear')."
        )
    use_vlm = 'Use the vortex lattice solver: solver_type="vlm".'
    b = surf.semi_span
    z_tol = PLANAR_TOL_FRACTION * b
    for k, sec in enumerate(_sorted_sections(surf)):
        if sec.z_le is not None and abs(sec.z_le) > z_tol:
            raise ValidityError(
                "The Fourier solver needs a flat lifting line in one plane (Glauert 1926). "
                f"Section {k} of surface '{surf.name}' has z_le={sec.z_le:.3g} m; the limit is "
                f"{z_tol:.3g} m (1e-9 times the semi-span). {use_vlm}"
            )
    rise = abs(b * np.sin(surf.dihedral))
    if rise > z_tol:
        raise ValidityError(
            "The Fourier solver needs a flat lifting line in one plane (Glauert 1926). "
            f"The dihedral of {np.degrees(surf.dihedral):.1f} deg of surface '{surf.name}' moves "
            f"the line {rise:.3g} m in z over the semi-span; the limit is {z_tol:.3g} m "
            f"(1e-9 times the semi-span). {use_vlm}"
        )
    secs = _sorted_sections(surf)
    fr = np.array([s.y_frac for s in secs])
    x_le, y, _ = surface_reference_line(surf, fr)
    x_qc = x_le + 0.25 * np.array([s.chord for s in secs])
    dx, dy = np.diff(x_qc), np.diff(y)
    sweep = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    if sweep.size and float(np.max(sweep)) > SWEEP_LIMIT_DEG:
        k = int(np.argmax(sweep))
        raise ValidityError(
            "The Fourier solver needs one straight, unswept lifting line (Glauert 1926). "
            f"The quarter-chord line sweeps {sweep[k]:.1f} deg between sections {k} and "
            f"{k + 1} of surface '{surf.name}'; the limit is {SWEEP_LIMIT_DEG:g} deg. {use_vlm}"
        )


#: Cache of Gauss-Legendre nodes and weights by order. The nodes depend
#: only on the order, so one solve per order pays for ``leggauss`` once.
_GAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights of order *n*.

    Parameters
    ----------
    n : int
        Quadrature order (number of points).

    Returns
    -------
    x, w : np.ndarray
        Nodes in (-1, 1) and weights. The cached arrays are read-only:
        callers must not modify them.
    """
    hit = _GAUSS_CACHE.get(int(n))
    if hit is None:
        x, w = np.polynomial.legendre.leggauss(int(n))
        hit = (np.asarray(x, dtype=float), np.asarray(w, dtype=float))
        for arr in hit:
            arr.flags.writeable = False  # shared by every solve of this order
        _GAUSS_CACHE[int(n)] = hit
    return hit


#: Cache of the Fourier basis by number of terms: the collocation angles, the
#: quadrature angles and the sine tables depend only on N, not on the geometry.
_BASIS_CACHE: dict[int, tuple[np.ndarray, ...]] = {}


def _fourier_basis(N: int) -> tuple[np.ndarray, ...]:
    """Return ``(theta, sin_theta, n_idx, sin_n_theta, theta_q, sin_n_theta_q, weights_q)`` for N terms.

    The arrays are the same as a direct evaluation (same operations in the
    same order); they are cached and read-only, so callers must not modify
    them. The quadrature has 8 N Gauss-Legendre points in theta.
    """
    hit = _BASIS_CACHE.get(int(N))
    if hit is None:
        theta = fourier_collocation_angles(N)
        n_idx = np.arange(1, N + 1)
        sin_theta = np.sin(theta)
        sin_n_theta = np.sin(theta[:, None] * n_idx[None, :])
        xg, wg = _gauss_legendre(8 * N)
        theta_q = 0.5 * np.pi * (xg + 1.0)
        sin_n_theta_q = np.sin(theta_q[:, None] * n_idx[None, :])
        weights_q = 0.5 * np.pi * wg
        hit = (theta, sin_theta, n_idx, sin_n_theta, theta_q, sin_n_theta_q, weights_q)
        for arr in hit:
            arr.flags.writeable = False  # shared by every solve with N terms
        _BASIS_CACHE[int(N)] = hit
    return hit


def _section_data(surf: LiftingSurface, y_frac: np.ndarray) -> dict[str, np.ndarray]:
    """Chord, twist (with incidence), a0, alpha_L0, Cd0, Cm0, x_qc and z_qc at span fractions."""
    secs = _sorted_sections(surf)
    fr = np.array([s.y_frac for s in secs])
    eta = np.asarray(y_frac, dtype=float)
    chord = np.interp(eta, fr, np.array([s.chord for s in secs]))
    twist = np.interp(eta, fr, np.array([s.twist for s in secs])) + surf.incidence
    if all(isinstance(s.airfoil, LinearAirfoil) for s in secs):
        # Linear section data blends linearly along the span, so one
        # vectorised interpolation per property gives the same values as
        # blending the airfoils one station at a time.
        a0 = np.interp(eta, fr, np.array([s.airfoil.a0 for s in secs]))
        alpha_l0 = np.interp(eta, fr, np.array([s.airfoil.alpha_L0 for s in secs]))
        cd0 = np.interp(eta, fr, np.array([s.airfoil.Cd0 for s in secs]))
        cm0 = np.interp(eta, fr, np.array([s.airfoil.Cm0 for s in secs]))
        props = np.column_stack([a0, alpha_l0, cd0, cm0])
    else:
        blender = _AirfoilBlender(secs)
        props = np.array([airfoil_linear_properties(blender.at(float(e))) for e in eta]).reshape(-1, 4)
    x_le, _, z_le = surface_reference_line(surf, eta)
    pos = np.asarray(surf.position, dtype=float)
    # The sections twist about the quarter chord, so z_qc does not change with the twist.
    x_qc = x_le + 0.25 * chord + float(pos[0])
    z_qc = np.asarray(z_le, dtype=float) + float(pos[2])
    return {"chord": chord, "twist": twist, "a0": props[:, 0], "alpha_L0": props[:, 1],
            "Cd0": props[:, 2], "Cm0": props[:, 3], "x_qc": x_qc, "z_qc": z_qc}


_SYSTEM_CACHE: OrderedDict = OrderedDict()
_SYSTEM_LOCK = threading.Lock()


def _fourier_system(fingerprint, surf: LiftingSurface, N: int):
    """Return the Fourier system of one surface with N terms, cached by the surfaces fingerprint.

    Returns ``(b, theta, st, mu, n_idx, lhs, theta_q, q_geo, sin_n_theta,
    sin_n_theta_q)``; the arrays are read-only. *q_geo* holds the quadrature
    weights, the section data at the quadrature points and the exact span
    integrals of ``c^2 Cm0`` and ``c Cd0``.
    """
    key = (fingerprint, int(N), _section_data, surface_reference_line)
    with _SYSTEM_LOCK:
        hit = _SYSTEM_CACHE.get(key)
        if hit is not None:
            _SYSTEM_CACHE.move_to_end(key)
            return hit
    b = 2.0 * surf.semi_span
    # The angles and the sine tables depend only on N (cached, read-only);
    # the quadrature for the moment of the lift is Gauss-Legendre in theta.
    theta, sin_theta, n_idx, sin_n_theta, theta_q, sin_n_theta_q, weights_q = _fourier_basis(N)
    st = _section_data(surf, np.abs(np.cos(theta)))
    mu = st["chord"] * st["a0"] / (4.0 * b)
    lhs = sin_n_theta * (1.0 + (mu / sin_theta)[:, None] * n_idx[None, :])
    q_geo = _section_data(surf, np.abs(np.cos(theta_q)))
    q_geo["weights"] = weights_q
    # Exact span integrals of piecewise-linear section data (Simpson on each
    # section interval is exact for the cubic c^2 Cm0 and the quadratic c Cd0).
    secs = _sorted_sections(surf)
    fr = np.array([s_.y_frac for s_ in secs])
    e3 = np.sort(np.concatenate([fr, 0.5 * (fr[:-1] + fr[1:])]))
    sd = _section_data(surf, e3)
    h = np.diff(fr) * surf.semi_span  # interval lengths along the span
    c2cm = sd["chord"] ** 2 * sd["Cm0"]
    ccd = sd["chord"] * sd["Cd0"]

    def simpson(f):
        return float(np.sum(h / 6.0 * (f[0:-1:2] + 4.0 * f[1::2] + f[2::2])))
    q_geo["int_c2_cm0"] = 2.0 * simpson(c2cm)   # both halves
    q_geo["int_c_cd0"] = 2.0 * simpson(ccd)
    for arr in [mu, lhs, *st.values(), *(v for v in q_geo.values() if isinstance(v, np.ndarray))]:
        arr.flags.writeable = False
    hit = (b, theta, st, mu, n_idx, lhs, theta_q, q_geo, sin_n_theta, sin_n_theta_q)
    with _SYSTEM_LOCK:
        _SYSTEM_CACHE[key] = hit
        while len(_SYSTEM_CACHE) > 16:
            _SYSTEM_CACHE.popitem(last=False)
    return hit


class FourierSolver(BaseSolver):
    """Classical Fourier-series lifting-line solver."""

    name = "fourier"

    @staticmethod
    def _n_terms(surf: LiftingSurface, settings: SolverSettings) -> int:
        return int(surf.n_panels if surf.n_panels is not None else settings.n_panels)

    def _prepare(self, aircraft: Aircraft | LiftingSurface, condition: FlightCondition, settings: SolverSettings):
        if isinstance(aircraft, LiftingSurface):
            aircraft = Aircraft(surfaces=[aircraft])
        validate_aircraft(aircraft)
        # The geometry guard runs before the older applicability check: the
        # owner decisions D-08 fix ValidityError and a message that names
        # solver_type="vlm" for geometry outside the model.
        for _surf in aircraft.surfaces:
            _check_geometry_in_model(_surf)
            # Refuse invalid controls first, with the same message as the
            # lattice solvers; then refuse any deflection that is not zero.
            validate_controls(_surf)
            for _ctrl in getattr(_surf, "controls", []):
                if _ctrl.deflection != 0.0:
                    raise ValueError(
                        f"The Fourier solver cannot model deflected control surfaces (control '{_ctrl.name}' "
                        f"on surface '{_surf.name}' has deflection {np.degrees(_ctrl.deflection):.2f} deg). "
                        "Use the vortex lattice solver (solver_type='vlm') or lifting line solvers "
                        "(solver_type='linear' or 'nonlinear')."
                    )
        validate_fourier_applicability(aircraft)
        if condition.h is not None:
            raise ValidityError("The Fourier solver has no ground effect. Use solver='vlm' or 'linear'.")
        if abs(condition.beta) > 1e-12:
            raise ValidityError("The Fourier solver is symmetric and has no sideslip. Use solver='vlm'.")
        # The reference values and the whole Fourier system depend only on the
        # surfaces and N: computed once per geometry (same fingerprint as the
        # lattice cache), read-only. The moment reference point is per call.
        fp = lattice_cache.surfaces_fingerprint(aircraft)
        aircraft.compute_reference_values(auto=lattice_cache.geometry_info(fp, aircraft)["auto_ref"])
        surf = aircraft.surfaces[0]
        N = self._n_terms(surf, settings)
        b, theta, st, mu, n_idx, lhs, theta_q, q_geo, sin_n_theta, sin_n_theta_q = _fourier_system(fp, surf, N)
        q_data = dict(q_geo)
        ref = aircraft.moment_reference()
        q_data["x_ref"] = float(ref[0])
        q_data["z_ref"] = float(ref[2])
        return aircraft, surf, N, b, theta, st, mu, n_idx, lhs, theta_q, q_data, sin_n_theta, sin_n_theta_q

    @staticmethod
    def _stations(b, theta, st, sin_n_theta, theta_q) -> dict:
        """Return the angle-independent spanwise data of :meth:`_result` (computed once per sweep)."""
        order = np.argsort(np.cos(theta))
        th = theta[order]
        Cd0 = st["Cd0"][order]
        return {
            "th": th,
            "sin_th": np.sin(th),
            "y": 0.5 * b * np.cos(th),
            "sin_n": sin_n_theta[order],
            "chord": st["chord"][order],
            "twist": st["twist"][order],
            "Cd0": Cd0,
            "Cm0": st["Cm0"][order],
            "has_profile": bool(np.any(Cd0 != 0.0)),
            "dy_dth": 0.5 * b * np.sin(theta_q),
            "sin_th_q": np.sin(theta_q),
        }

    def _result(self, aircraft, surf, N, b, theta, st, mu, n_idx, A, theta_q, q_data,
                sin_n_theta, sin_n_theta_q, alpha, condition, stations: dict | None = None) -> SolverResult:
        V = condition.V_inf
        rho = condition.rho
        S, c_ref = aircraft.S_ref, aircraft.c_ref
        AR = b * b / S
        CL = float(np.pi * AR * A[0])
        CDi = float(np.pi * AR * np.sum(n_idx * A ** 2))
        e = CL ** 2 / (np.pi * AR * CDi) if CDi > 1e-14 else float("nan")

        # Spanwise output at the collocation stations (left tip to right tip).
        sta = self._stations(b, theta, st, sin_n_theta, theta_q) if stations is None else stations
        sin_n = sta["sin_n"]
        gamma = 2.0 * b * V * (sin_n @ A)
        alpha_i = (sin_n @ (n_idx * A)) / sta["sin_th"]
        chord = sta["chord"].copy()
        Cl = 2.0 * gamma / (V * chord)
        alpha_eff = alpha + sta["twist"] - alpha_i
        has_profile = sta["has_profile"]
        spanwise = SpanwiseResult(
            y=sta["y"].copy(), gamma=gamma, Cl=Cl, Cd_i=Cl * alpha_i,
            Cd_profile=sta["Cd0"].copy() if has_profile else None,
            alpha_eff=alpha_eff, alpha_i=alpha_i,
            local_lift=rho * V * gamma, surface_name=surf.name,
            chord=chord, Cm_section=sta["Cm0"].copy(),
        )

        # Moment of the section forces about the reference point, M_y = r_z F_x - r_x F_z,
        # with y = (b/2) cos(theta): dy = (b/2) sin(theta) d(theta). The force of
        # each section is rho V Gamma dy normal to the local flow, which the
        # downwash turns by alpha_i: a lift part normal to the free stream,
        # (-sin(alpha), 0, cos(alpha)), and an induced-drag part alpha_i along
        # the free stream, (cos(alpha), 0, sin(alpha)) (x aft, z up).
        g_q = 2.0 * b * V * (sin_n_theta_q @ A)
        alpha_i_q = (sin_n_theta_q @ (n_idx * A)) / sta["sin_th_q"]
        dF = q_data["weights"] * rho * V * g_q * sta["dy_dth"]
        ca, sa = np.cos(alpha), np.sin(alpha)
        F_x = dF * (-sa + alpha_i_q * ca)
        F_z = dF * (ca + alpha_i_q * sa)
        q_inf = 0.5 * rho * V * V
        lift_moment = float(np.sum((q_data["z_qc"] - q_data["z_ref"]) * F_x
                                   - (q_data["x_qc"] - q_data["x_ref"]) * F_z))
        section_moment = q_inf * q_data["int_c2_cm0"]
        Cm = float((lift_moment + section_moment) / (q_inf * S * c_ref))
        CDp = float(q_data["int_c_cd0"] / S) if has_profile else None
        CD_total = CDi + (CDp if CDp is not None else 0.0)

        totals = IntegratedResult(
            CL=CL, CDi=CDi, CDp=CDp, CD_total=CD_total if has_profile else None,
            e=e, AR=AR, Cm=Cm,
        )
        totals.trust = evaluate_aerodynamic_trust(
            AR=AR, condition=condition, spanwise_list=[spanwise], CL=CL, CDi=CDi,
            CD_total=totals.CD_total, converged=True, n_panels=N, max_sweep_rad=0.0,
            solver_type="fourier",
        )
        return SolverResult(
            spanwise=[spanwise], totals=totals, solver_type="fourier", converged=True,
            iterations=1, fourier_coefficients=A, condition=condition,
        )

    def solve(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
    ) -> SolverResult:
        """Solve the Fourier lifting line for one flight condition.

        Parameters
        ----------
        aircraft : Aircraft or LiftingSurface
            Geometry definition.
        condition : FlightCondition
            Free-stream conditions.
        settings : SolverSettings
            Discretisation parameters.

        Returns
        -------
        SolverResult
            Spanwise and integrated results, with the Fourier coefficients.
        """
        aircraft, surf, N, b, theta, st, mu, n_idx, lhs, theta_q, q_data, sin_n, sin_n_q = self._prepare(
            aircraft, condition, settings)
        rhs = mu * (condition.alpha + st["twist"] - st["alpha_L0"])
        with blas_single_thread():
            A = np.linalg.solve(lhs, rhs)
        return self._result(aircraft, surf, N, b, theta, st, mu, n_idx, A, theta_q, q_data,
                            sin_n, sin_n_q, condition.alpha, condition)

    def solve_sweep(
        self,
        aircraft: Aircraft | LiftingSurface,
        condition: FlightCondition,
        settings: SolverSettings,
        alpha_range: np.ndarray,
    ) -> list[SolverResult]:
        """Angle-of-attack sweep: one factorisation, many right-hand sides."""
        aircraft, surf, N, b, theta, st, mu, n_idx, lhs, theta_q, q_data, sin_n, sin_n_q = self._prepare(
            aircraft, condition, settings)
        alphas = np.asarray(alpha_range, dtype=float)
        rhs = mu[:, None] * (alphas[None, :] + st["twist"][:, None] - st["alpha_L0"][:, None])
        with blas_single_thread():
            A_all = np.linalg.solve(lhs, rhs)
        stations = self._stations(b, theta, st, sin_n, theta_q)
        out = []
        for i, a in enumerate(alphas):
            cond = FlightCondition(V_inf=condition.V_inf, alpha=float(a), beta=condition.beta, rho=condition.rho)
            out.append(self._result(aircraft, surf, N, b, theta, st, mu, n_idx, A_all[:, i], theta_q, q_data,
                                    sin_n, sin_n_q, float(a), cond, stations=stations))
        return out
