# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Public frequency-domain dynamic derivatives from the harmonic vortex-ring lattice.

:func:`dynamic_derivatives` computes the unsteady (frequency-dependent)
longitudinal aerodynamic derivatives of a wing or aircraft configuration in
free air or in flat-ground effect using the harmonic vortex-ring lattice
solver (:mod:`ventorum.unsteady.harmonic`).

Definitions
-----------
Nondimensional time is ``tau = 2 V_inf t / c_ref`` [-], reduced frequency is
``k = omega c_ref / (2 V_inf)`` [-], and pitch and incidence rates are
normalised by ``c_ref / (2 V_inf)`` [s] as in
:func:`ventorum.solvers.rate_derivatives`.

Two harmonic motions about the steady mean flight state are evaluated at each
reduced frequency ``k``:

* Pitch ``theta`` [rad] (positive nose up) about the moment reference point
  ``ref_point`` [m] at constant height: complex response ``H_theta(k)`` [1/rad].
* Heave ``z`` [m] (positive up along lift direction in free air or ground
  normal in ground effect) at constant attitude: complex response ``H_z(k)``
  [-] per unit ``z / c_ref``.

For each coefficient ``C`` in ``(CL, CD, CY, Cl, Cm, Cn)``, the equivalent
oscillatory derivatives at reduced frequency ``k`` (Rodden and Giesing 1970)
are:

* ``C_alpha(k) = Re H_theta(k)`` [1/rad].
* ``C_q(k) = Im H_pureq(k) / k`` [-] from the pure pitch-rate motion (pitch
  ``theta`` combined with heave ``z = V_inf theta / (i omega)`` so that the
  incidence stays zero; in loaded ground effect, evaluated directly at
  constant height so the static height term ``C_h`` does not introduce a
  ``1 / k^2`` singularity).
* ``C_alpha_dot(k) = Im H_theta(k) / k - C_q(k)`` [-] per unit
  ``(dalpha/dt) c_ref / (2 V_inf)``.
* ``C_h(k) = Re H_z(k)`` [-] per unit ``h / c_ref`` (zero in free air).
* ``C_h_dot(k) = Im H_z(k) / (2 k)`` [-] per unit ``(dz/dt) / V_inf``. In
  free air at zero mean load, ``C_h_dot = -C_alpha`` as ``k -> 0``.
* ``C_h_dot_ground(k) = C_h_dot(k) + C_alpha(k)`` [-], the ground-effect part
  of the heave-rate derivative (zero in free air at ``k -> 0`` for an unloaded
  mean state at ``beta = 0``). The definition is for ``beta = 0``: with
  sideslip the heave axis is ``L_0`` but an incidence change moves the onset
  along ``cos(beta) L_0``, so in free air ``C_h_dot(0) = -C_alpha(0) /
  cos(beta)`` and ``C_h_dot_ground`` is not zero.

The ``k -> 0`` limits are always returned alongside the per-``k`` values:
``C_alpha(0)``, ``C_q(0)`` and ``C_h(0)`` come from the exact steady limits
of the harmonic ring system, while ``C_alpha_dot(0)`` and ``C_h_dot(0)`` use
a small-frequency probe at ``k = K_ZERO_PROBE`` (``1e-3``).

References
----------
* W. P. Rodden and J. P. Giesing, "Application of oscillatory aerodynamic
  theory to estimation of dynamic stability derivatives", Journal of Aircraft
  7(3), 1970, pp. 272-275 (equivalent derivatives at reduced frequency k).
* T. Theodorsen, "General theory of aerodynamic instability and the mechanism
  of flutter", NACA Report 496, 1935 (2D thin-airfoil harmonic lift and
  pitching moment limits).
* J. Katz and A. Plotkin, "Low-Speed Aerodynamics", 2nd ed., Cambridge
  University Press, 2001, chapters 12 and 13 (vortex-ring lattice and
  unsteady vortex-lattice method).
* K. C. Hall, "Eigenanalysis of unsteady flows about airfoils, cascades, and
  wings", AIAA Journal 32(12), 1994, pp. 2426-2432 (harmonic wake phase
  relation; cited through Parenteau 2021).
* M. Parenteau, "A Non-Linear Frequency Domain Potential Flow Model for
  Compressible, Transonic and Viscous Aeroelastic Analyses", PhD thesis,
  Polytechnique Montreal, 2021, chapter 4 (frequency-domain vortex lattice
  and Woodbury wake update; cited through study material).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ventorum.aero.system import ground_plane_from_condition
from ventorum.aero.vortex import kernel_cache_scope
from ventorum.core.axes import rotation_body_to_stability
from ventorum.core.constants import MACH_LIMIT_INCOMPRESSIBLE
from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
    TrustScore,
)
from ventorum.core.errors import InputError
from ventorum.core.trust import isa_speed_of_sound
from ventorum.solvers.rate_derivatives import rate_axes_transform
from ventorum.unsteady.harmonic import _HarmonicResponse, _HarmonicRingSystem, _ey_cross

__all__ = ["DynamicDerivativesResult", "dynamic_derivatives"]

_COEFFS = ("CL", "CD", "CY", "Cl", "Cm", "Cn")
_DERIV_SUFFIXES = ("alpha", "q", "alpha_dot", "h", "h_dot", "h_dot_ground")

#: Small reduced frequency [-] used to evaluate the ``k -> 0`` limits of
#: ``C_alpha_dot`` and ``C_h_dot`` without far-wake quadrature cancellation.
K_ZERO_PROBE = 1.0e-3

#: Minimum chordwise panel count [-] for unsteady pitching-moment trust.
MIN_UNSTEADY_N_CHORD = 8

#: Bounds on the per-strip trailing-edge panel length ratio ``dx_TE,s / dx_w`` [-].
DX_TE_RATIO_MIN = 0.8
DX_TE_RATIO_MAX = 1.25

#: Upper reduced frequency [-] of the verified test range.
MAX_VERIFIED_K = 1.0

#: Minimum discrete wake length in reference chords [-] without a short-wake note.
MIN_TRUST_WAKE_CHORDS = 5.0


def _json_float(value: float | None) -> float | None:
    """Return ``float(value)``, or None when *value* is None or not finite (strict JSON)."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class DynamicDerivativesResult:
    """Result of :func:`dynamic_derivatives`.

    Attributes
    ----------
    k : numpy.ndarray
        Evaluated reduced frequencies ``omega c_ref / (2 V_inf)`` [-],
        shape ``(n_k,)``.
    k0 : dict[str, float]
        ``k -> 0`` limits of all 36 longitudinal dynamic derivatives, keyed
        ``"<C>_<suffix>"`` for ``C`` in ``(CL, CD, CY, Cl, Cm, Cn)`` and
        ``suffix`` in ``(alpha, q, alpha_dot, h, h_dot, h_dot_ground)``
        ([1/rad] for ``alpha``, [-] for all others).
    derivatives : dict[str, numpy.ndarray]
        Equivalent oscillatory derivatives at each requested ``k``, same 36
        keys as ``k0``, each a 1-D float64 array of shape ``(n_k,)``. Entries
        where ``k == 0`` hold the ``k -> 0`` limit.
    H_theta : dict[str, numpy.ndarray]
        Complex pitch response [1/rad] per unit ``theta`` for each coefficient
        in ``(CL, CD, CY, Cl, Cm, Cn)``, shape ``(n_k,)``, dtype ``complex128``.
    H_z : dict[str, numpy.ndarray]
        Complex heave response [-] per unit ``z / c_ref`` for each coefficient,
        shape ``(n_k,)``, dtype ``complex128``.
    H_pureq : dict[str, numpy.ndarray]
        Complex pure pitch-rate response [1/rad] (zero incidence, so
        ``Im H_pureq / k = C_q``) for each coefficient, shape ``(n_k,)``,
        dtype ``complex128``.
    mean_state : dict[str, float]
        Steady mean-state load coefficients ``CL, CD, CY, Cl, Cm, Cn`` [-] in
        the requested ``axes``.
    CL0 : float
        Steady mean lift coefficient [-].
    Cm0 : float
        Steady mean pitching-moment coefficient [-] in the requested ``axes``.
    axes : str
        Moment axis system: ``"body"`` or ``"stability"`` of the mean state.
    in_ground_effect : bool
        True when solved over a flat ground plane.
    h_m : float or None
        Height of ``ref_point`` above the ground [m], or None in free air.
    S_ref : float
        Reference area [m^2].
    b_ref : float
        Reference span [m].
    c_ref : float
        Reference chord [m].
    ref_point : numpy.ndarray
        Moment reference and pitch pivot point [m], shape ``(3,)``.
    n_panels : int
        Number of bound vortex rings [-].
    n_strips : int
        Number of spanwise strips [-].
    n_chord : int
        Number of chordwise panels per strip [-].
    n_wake_rows : int
        Number of discrete wake rows behind each strip [-].
    wake_chords : float
        Requested discrete wake length in reference chords [-].
    wake_length : float
        Actual discrete wake length ``n_wake_rows * dx_w`` [m].
    dx_w : float
        Discrete wake row length [m].
    dt : float
        Wake time step ``dx_w / V_inf`` [s].
    dx_te_ratio : numpy.ndarray
        Per-strip trailing-edge panel length ratio ``dx_TE,s / dx_w`` [-],
        shape ``(n_strips,)``.
    k_zero_probe : float
        Small reduced frequency [-] used for the ``k -> 0`` limits of
        ``C_alpha_dot`` and ``C_h_dot``.
    base : SolverResult
        Steady vortex-lattice result of the mean state (holds the steady
        trust assessment).
    trust : TrustScore or None
        Trust score of the mean state with unsteady warnings added (steady
        score and rating are preserved).
    trust_notes : list[str]
        Unsteady trust warnings and notes triggered for this solve.
    notes : list[str]
        Method notes, device fallback notes and unsteady trust notes.
    device : str
        Execution device of the harmonic solve: ``"cpu"`` or ``"gpu"``.
    precision : str
        Kernel arithmetic precision of the harmonic solve: ``"float64"`` or
        ``"float32"``.
    H_delta : dict[str, dict[str, numpy.ndarray]]
        Complex control-surface response [1/rad] per unit deflection for each
        requested control name and each coefficient in
        ``(CL, CD, CY, Cl, Cm, Cn)``, shape ``(n_k,)``, dtype ``complex128``.
        Empty when no controls are requested.
    """

    k: np.ndarray
    k0: dict[str, float]
    derivatives: dict[str, np.ndarray]
    H_theta: dict[str, np.ndarray]
    H_z: dict[str, np.ndarray]
    H_pureq: dict[str, np.ndarray]
    mean_state: dict[str, float]
    CL0: float
    Cm0: float
    axes: str
    in_ground_effect: bool
    h_m: float | None
    S_ref: float
    b_ref: float
    c_ref: float
    ref_point: np.ndarray
    n_panels: int
    n_strips: int
    n_chord: int
    n_wake_rows: int
    wake_chords: float
    wake_length: float
    dx_w: float
    dt: float
    dx_te_ratio: np.ndarray
    k_zero_probe: float
    base: SolverResult
    trust: TrustScore | None
    trust_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    device: str = "cpu"
    precision: str = "float64"
    H_delta: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible dictionary of the result."""
        def _pack_complex(mapping: dict[str, np.ndarray]) -> dict[str, dict[str, list[float | None]]]:
            return {
                coeff: {
                    "re": [_json_float(float(z.real)) for z in arr],
                    "im": [_json_float(float(z.imag)) for z in arr],
                }
                for coeff, arr in mapping.items()
            }

        trust_dict: dict[str, Any] | None = None
        if self.trust is not None:
            trust_dict = {
                "score": _json_float(self.trust.score),
                "rating": str(self.trust.rating),
                "warnings": list(self.trust.warnings),
                "recommendations": list(self.trust.recommendations),
            }

        responses: dict[str, Any] = {
            "H_theta": _pack_complex(self.H_theta),
            "H_z": _pack_complex(self.H_z),
            "H_pureq": _pack_complex(self.H_pureq),
        }
        if self.H_delta:
            responses["H_delta"] = {
                name: _pack_complex(mapping)
                for name, mapping in self.H_delta.items()
            }

        return {
            "k": [_json_float(float(v)) for v in self.k],
            "k0": {name: _json_float(val) for name, val in self.k0.items()},
            "derivatives": {
                name: [_json_float(float(v)) for v in arr]
                for name, arr in self.derivatives.items()
            },
            "responses": responses,
            "mean_state": {c: _json_float(v) for c, v in self.mean_state.items()},
            "CL0": _json_float(self.CL0),
            "Cm0": _json_float(self.Cm0),
            "axes": self.axes,
            "in_ground_effect": bool(self.in_ground_effect),
            "h_m": None if self.h_m is None else _json_float(self.h_m),
            "S_ref_m2": _json_float(self.S_ref),
            "b_ref_m": _json_float(self.b_ref),
            "c_ref_m": _json_float(self.c_ref),
            "ref_point_m": [_json_float(float(x)) for x in self.ref_point],
            "mesh": {
                "n_panels": int(self.n_panels),
                "n_strips": int(self.n_strips),
                "n_chord": int(self.n_chord),
                "n_wake_rows": int(self.n_wake_rows),
                "wake_chords": _json_float(self.wake_chords),
                "wake_length_m": _json_float(self.wake_length),
                "dx_w_m": _json_float(self.dx_w),
                "dt_s": _json_float(self.dt),
                "dx_te_ratio_min": _json_float(float(np.min(self.dx_te_ratio))),
                "dx_te_ratio_max": _json_float(float(np.max(self.dx_te_ratio))),
            },
            "k_zero_probe": _json_float(self.k_zero_probe),
            "device": str(self.device),
            "precision": str(self.precision),
            "trust": trust_dict,
            "trust_notes": list(self.trust_notes),
            "notes": list(self.notes),
        }


def _validate_k(k: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate reduced frequency sequence *k* [-] and return a 1-D float64 array."""
    if isinstance(k, (str, bytes, bool)):
        raise InputError(f"k must be a non-empty sequence of non-negative finite numbers, got {k!r}.")
    try:
        items = list(k) if not isinstance(k, np.ndarray) else k
    except TypeError as exc:
        raise InputError(f"k must be a non-empty sequence of non-negative finite numbers, got {k!r}.") from exc
    if isinstance(items, list) and any(isinstance(v, bool) for v in items):
        raise InputError(f"k must contain numbers, not booleans; got {k!r}.")
    try:
        arr = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"k must contain finite numbers, got {k!r}.") from exc
    if arr.ndim != 1 or arr.size == 0:
        raise InputError("k must be a non-empty 1-D sequence of reduced frequencies.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise InputError(f"All reduced frequencies k must be finite and >= 0, got {k!r}.")
    return arr


def _validate_wake_chords(wake_chords: float) -> float:
    """Validate *wake_chords* [-] as a finite positive float."""
    if isinstance(wake_chords, bool):
        raise InputError(f"wake_chords must be a positive number, got {wake_chords!r}.")
    try:
        val = float(wake_chords)
    except (TypeError, ValueError) as exc:
        raise InputError(f"wake_chords must be a positive number, got {wake_chords!r}.") from exc
    if not math.isfinite(val) or val <= 0.0:
        raise InputError(f"wake_chords must be finite and > 0, got {wake_chords!r}.")
    return val


def _validate_axes(axes: str) -> str:
    """Validate *axes* as ``'body'`` or ``'stability'``."""
    if not isinstance(axes, str) or axes not in ("body", "stability"):
        raise InputError(
            f"'axes' must be 'body' or 'stability' for longitudinal dynamic derivatives, got {axes!r}."
        )
    return axes


_RATE_HINTS = {
    "dynamic_derivatives": "The rate derivatives come from the harmonic motion about a mean state without rotation.",
    "time_response": "Give a rotation through motion, for example motion={'pitch': lambda t: q * t}.",
    "indicial_response": "The indicial steps start from a mean state without rotation.",
    "free_flight": "Give initial rates through initial_state, for example initial_state={'pitch_dot': q}.",
}


def _check_no_steady_rates(condition: FlightCondition, caller: str) -> None:
    """Refuse nonzero steady body rates ``p``, ``q``, ``r`` [rad/s] with :class:`InputError`.

    The unsteady solvers linearise about, or march from, a mean state that
    does not rotate. The steady base solve would use the rates and the march
    or the harmonic loads would not, so the result would be inconsistent.
    """
    rates = {name: float(getattr(condition, name, 0.0) or 0.0) for name in ("p", "q", "r")}
    nonzero = {name: val for name, val in rates.items() if val != 0.0}
    if nonzero:
        given = ", ".join(f"{name}={val:g} rad/s" for name, val in nonzero.items())
        raise InputError(
            f"{caller} does not use the steady body rates of the flight condition ({given}); "
            f"set p, q and r to 0. {_RATE_HINTS.get(caller, '')}".rstrip()
        )


def _check_incompressible_mach(condition: FlightCondition) -> None:
    """Refuse Mach numbers above the incompressible limit with :class:`InputError`."""
    mach_stated = getattr(condition, "mach", None)
    if mach_stated is not None:
        if isinstance(mach_stated, bool):
            raise InputError(f"condition.mach must be a number, got {mach_stated!r}.")
        try:
            m_val = float(mach_stated)
        except (TypeError, ValueError) as exc:
            raise InputError(f"condition.mach must be a finite number, got {mach_stated!r}.") from exc
        if not math.isfinite(m_val) or m_val < 0.0:
            raise InputError(f"condition.mach must be finite and >= 0, got {mach_stated!r}.")
        if m_val > MACH_LIMIT_INCOMPRESSIBLE:
            raise InputError(
                f"mach={m_val:g} is above the incompressible limit {MACH_LIMIT_INCOMPRESSIBLE}: "
                "the harmonic vortex-ring solver models incompressible flow only."
            )

    if math.isfinite(float(condition.V_inf)) and math.isfinite(float(condition.rho)) and float(condition.rho) > 0.0:
        a_sound = isa_speed_of_sound(float(condition.rho))
        m_flow = float(condition.V_inf) / a_sound
        if m_flow > MACH_LIMIT_INCOMPRESSIBLE:
            raise InputError(
                f"Mach {m_flow:.3f} (V_inf={float(condition.V_inf):g} m/s, ISA speed of sound "
                f"{a_sound:.1f} m/s) is above the incompressible limit {MACH_LIMIT_INCOMPRESSIBLE}: "
                "the harmonic vortex-ring solver models incompressible flow only."
            )


def _solve_pure_pitch_rate_responses(
    system: _HarmonicRingSystem,
    resp: _HarmonicResponse,
    k_solve: np.ndarray,
) -> np.ndarray:
    """Return complex pure pitch-rate responses ``H_pureq(k)`` [1/rad], shape ``(len(k_solve), 6)``.

    For an unloaded mean state (``not system.has_mean_load``) with the heave
    axis ``e_h = cos(beta) L_0``, the exact kinematic combination
    ``H_theta(k) + H_z(k) / (2 i k)`` cancels the incidence term identically
    with zero extra linear solves. With sideslip, or banked in ground effect,
    ``e_h`` is not ``cos(beta) L_0`` and the combination does not cancel the
    incidence, so these states use the direct solve. For a loaded
    mean state (especially in ground effect where ``C_h != 0``), solving the
    pure pitch-rate right-hand side directly on the already-factorised
    Woodbury system keeps the height fixed and avoids the ``-C_h / (2 k^2)``
    singularity while agreeing with ``H_theta + H_z / (2 i k)`` to round-off
    when the mean load is zero.

    The loaded right-hand side holds the kinematic rotation velocity only. It
    does not add the loaded-mean-state geometric terms (bound rings moving
    relative to the steady wake, travelling-wave wake displacement), so the
    wake stays straight in this response. This is the model of
    ``rate_derivatives``; the size of the omitted terms is not measured.
    """
    n_k = k_solve.size
    out = np.zeros((n_k, 6), dtype=np.complex128)

    cos_beta = math.hypot(float(system.d_0[0]), float(system.d_0[2]))
    shortcut_valid = bool(np.allclose(system.e_h, cos_beta * system.L_0, rtol=0.0, atol=1e-12))
    if not system.has_mean_load and shortcut_valid:
        for ik, kf in enumerate(k_solve):
            if kf == 0.0:
                continue
            for ic, name in enumerate(_COEFFS):
                h_th = getattr(resp.pitch, name)[ik]
                h_z = getattr(resp.heave, name)[ik]
                out[ik, ic] = h_th + h_z / (2.0j * kf)
        return out

    zero_v = np.zeros_like(system.P_load)
    ey_arm = _ey_cross(system.arm_all)
    rel_cp = system.rings.cp - system.ref_point[None, :]
    n_bc = system.rings.normal_bc
    ey_rel_cp = _ey_cross(rel_cp)

    for ik, kf in enumerate(k_solve):
        k_val = float(kf)
        if k_val == 0.0:
            continue
        omega = 2.0 * k_val * system.V_inf / system.c_ref
        kappa = omega / system.V_inf
        W = system.wake_matrix(k_val)
        dU_rot_cp = -(1j * omega) * ey_rel_cp
        rhs_q = (-np.sum(n_bc * dU_rot_cp, axis=1)).astype(np.complex128)[:, None]
        G_q = system._solve_linear(W, rhs_q, "woodbury")[:, 0]

        FM_wake_k = SL_wake_k = None
        if system.has_mean_load:
            # The mean circulation times the perturbation velocity of the wake.
            # An unloaded state (sideslip or bank in ground effect) has no such term.
            lags = np.exp(-1j * omega * np.arange(1, system.n_wake_rows + 1, dtype=np.float64) * system.dt)
            FM_wake_k = np.tensordot(system.FM_wake_tensor, lags, axes=([1], [0]))
            SL_wake_k = np.tensordot(system.SL_wake_tensor, lags, axes=([1], [0]))
            if system.use_tail:
                phase0 = complex(np.exp(-1j * omega * (system.n_wake_rows + 0.5) * system.dt))
                fm_t, sl_t = system._tail_load_matrices(kappa, phase0)
                FM_wake_k = FM_wake_k + fm_t
                SL_wake_k = SL_wake_k + sl_t

        dU_q_load = -(1j * omega) * ey_arm
        coeffs_q, _ = system._evaluate_linearised_loads(
            G_q,
            k=k_val,
            dU_load=dU_q_load,
            dv_geo_load=zero_v,
            rotate_mean_geometry=False,
            FM_wake_k=FM_wake_k,
            SL_wake_k=SL_wake_k,
            include_apparent_mass=True,
            include_te_correction=system.te_correction,
        )
        out[ik] = coeffs_q
    return out


def _transform_six_to_stability(
    vec: np.ndarray,
    rot_bs: np.ndarray,
    b_ref: float,
    c_ref: float,
) -> np.ndarray:
    """Transform a 6-component coefficient array ``(..., 6)`` from body to stability axes.

    Uses :func:`ventorum.solvers.rate_derivatives.rate_axes_transform` on the
    pitch-axis (y) column so forces stay in wind axes and moments rotate by
    ``alpha`` about ``y`` (``Cm`` invariant, ``Cl`` and ``Cn`` rotated).
    Works for both real and complex arrays.
    """
    arr = np.asarray(vec)
    flat = arr.reshape(-1, 6)
    out = np.empty_like(flat)
    for idx in range(flat.shape[0]):
        row = flat[idx]
        if np.iscomplexobj(row):
            m_re = np.zeros((6, 3), dtype=np.float64)
            m_im = np.zeros((6, 3), dtype=np.float64)
            m_re[:, 1] = row.real
            m_im[:, 1] = row.imag
            t_re = rate_axes_transform(m_re, rot_bs, b_ref, c_ref)[:, 1]
            t_im = rate_axes_transform(m_im, rot_bs, b_ref, c_ref)[:, 1]
            out[idx] = t_re + 1j * t_im
        else:
            m = np.zeros((6, 3), dtype=np.float64)
            m[:, 1] = row
            out[idx] = rate_axes_transform(m, rot_bs, b_ref, c_ref)[:, 1]
    return out.reshape(arr.shape)


def _build_unsteady_trust(
    base_trust: TrustScore | None,
    *,
    n_chord: int,
    dx_te_ratio: np.ndarray,
    k_values: np.ndarray,
    wake_chords: float,
) -> tuple[TrustScore | None, list[str]]:
    """Evaluate unsteady trust rules while preserving the steady trust score and rating."""
    trust_notes: list[str] = []
    trust_copy = copy.deepcopy(base_trust) if base_trust is not None else None

    if int(n_chord) < MIN_UNSTEADY_N_CHORD:
        warn_msg = (
            f"Warning: chordwise panel count n_chord={int(n_chord)} is below {MIN_UNSTEADY_N_CHORD}. "
            "Unsteady pitching-moment derivatives (Cm_q, Cm_alpha_dot) need at least "
            f"{MIN_UNSTEADY_N_CHORD} chordwise panels for chordwise convergence."
        )
        trust_notes.append(warn_msg)
        if trust_copy is not None:
            trust_copy.warnings.append(warn_msg)
            trust_copy.recommendations.append(
                f"Increase n_chord to at least {MIN_UNSTEADY_N_CHORD} for unsteady pitching-moment derivatives."
            )

    r_min = float(np.min(dx_te_ratio))
    r_max = float(np.max(dx_te_ratio))
    if r_min < DX_TE_RATIO_MIN or r_max > DX_TE_RATIO_MAX:
        trust_notes.append(
            f"Note: trailing-edge panel length ratio dx_TE/dx_w spans [{r_min:.3f}, {r_max:.3f}], "
            f"outside [{DX_TE_RATIO_MIN:g}, {DX_TE_RATIO_MAX:g}]. All strips share one wake time step "
            "dt = dx_w / V_inf; strong taper causes local phase-step mismatch at the trailing edge."
        )

    k_max = float(np.max(k_values))
    if k_max > MAX_VERIFIED_K:
        trust_notes.append(
            f"Note: maximum reduced frequency k={k_max:g} is above {MAX_VERIFIED_K:.1f}, which is "
            "outside the verified range of the tests."
        )

    if float(wake_chords) < MIN_TRUST_WAKE_CHORDS:
        trust_notes.append(
            f"Note: discrete wake length wake_chords={float(wake_chords):g} is below "
            f"{MIN_TRUST_WAKE_CHORDS:g} chords. Although the 3D far-wake tail closes the wake, "
            "a shorter discrete wake increases near-wake discretisation error."
        )

    return trust_copy, trust_notes


def dynamic_derivatives(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition,
    settings: SolverSettings | None = None,
    *,
    k: Sequence[float] | np.ndarray = (0.0,),
    wake_chords: float = 5.0,
    axes: str = "body",
    controls: Sequence[str] = (),
) -> DynamicDerivativesResult:
    """Compute frequency-dependent longitudinal and control dynamic derivatives via harmonic vortex rings.

    Solves small-amplitude harmonic pitch, heave and optional named
    control-surface deflection motions about the steady mean flight state
    *condition* on a vortex-ring lattice, in free air or in flat-ground effect
    (``condition.h`` set), and returns the equivalent oscillatory derivatives
    (Rodden and Giesing 1970) at every requested reduced frequency ``k`` and in
    the ``k -> 0`` limit.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        Geometry to analyse. It is not modified. Pitch rotation is about the
        moment reference point (``Aircraft.ref_point``, default the origin)
        [m].
    condition : FlightCondition
        Steady mean flight state: free-stream speed ``V_inf`` [m/s], air
        density ``rho`` [kg/m^3], angle of attack ``alpha`` [rad], sideslip
        ``beta`` [rad], bank angle ``phi`` [rad], and optional ground height
        ``h`` [m] of ``ref_point``. Must stay within the incompressible limit
        (Mach <= 0.3). The body rates ``p``, ``q``, ``r`` must be 0.
    settings : SolverSettings or None, optional
        Solver settings (default :class:`SolverSettings`). Only the
        vortex-lattice solver (``"vlm"`` or ``"auto"``) is supported; other
        solvers are refused with :class:`InputError`.
    k : sequence of float, optional
        Reduced frequencies ``k = omega c_ref / (2 V_inf)`` [-], all finite
        and ``>= 0``. Default ``(0.0,)`` (the ``k -> 0`` limit only).
    wake_chords : float, optional
        Discrete harmonic wake length in reference chords [-], must be finite
        and ``> 0``. Default ``5.0``. The closed-form 3D far-wake tail closes
        the wake beyond this distance.
    axes : {'body', 'stability'}, optional
        Moment axis system of the mean state, held fixed during the harmonic
        motion: ``"body"`` (default) or ``"stability"``. Force coefficients
        ``CL``, ``CD`` and ``CY`` are relative to the free stream in both sets.
    controls : sequence of str, optional
        Control-surface names to solve as unit harmonic deflection modes
        [1/rad] about the built lattice deflection. For each name, the result
        adds ``H_delta[name]`` and equivalent oscillatory derivatives
        ``"<C>_delta_<name>"`` and ``"<C>_delta_dot_<name>"`` [1/rad]
        (nondimensional rate ``delta_dot c_ref / (2 V_inf)``). Requires a
        chordwise lattice (``n_chord >= 2``). Default ``()``.

    Returns
    -------
    DynamicDerivativesResult
        Reduced frequencies, ``k -> 0`` limit derivatives, per-``k``
        derivatives, complex responses ``H_theta``, ``H_z``, ``H_pureq`` and
        ``H_delta``, steady mean state, trust assessment, trust notes and mesh
        metadata.

    Raises
    ------
    InputError
        If *k* is empty, negative or non-finite; *wake_chords* is non-positive
        or non-finite; *axes* is unknown; *controls* contains an unknown,
        duplicate or section-mode control name; *settings.solver_type* is not
        VLM; Mach is above the incompressible limit (0.3); or the lattice is
        refused by the ring-lattice builder (for example lifting-line
        collocation).
    GroundStrikeError
        If the geometry touches or crosses the ground plane when
        ``condition.h`` is set.
    """
    from ventorum.geometry import lattice_cache
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import (
        validate_aircraft,
        validate_flight_condition,
        validate_solver_settings,
    )

    k_req = _validate_k(k)
    wc = _validate_wake_chords(wake_chords)
    ax = _validate_axes(axes)
    ctrl_names = _HarmonicRingSystem._validate_controls(controls)

    ac = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    ac = ac.clone()
    sett = SolverSettings() if settings is None else settings.clone()
    sett = canonical_settings(sett)
    validate_solver_settings(sett)
    canonical = resolve_solver_type(sett.solver_type)
    if canonical != "vlm":
        raise InputError(
            f"dynamic_derivatives supports only solver_type='vlm' (or 'auto'), got {sett.solver_type!r}."
        )

    _check_incompressible_mach(condition)
    validate_flight_condition(condition)
    _check_no_steady_rates(condition, "dynamic_derivatives")
    validate_aircraft(ac)

    if ctrl_names:
        known_ctrl = set(ac.control_names())
        for cname in ctrl_names:
            if cname not in known_ctrl:
                raise InputError(
                    f"Unknown control surface {cname!r}; available controls: {sorted(known_ctrl)}."
                )

    solver = make_solver("vlm")
    fp = lattice_cache.surfaces_fingerprint(ac)
    geo = lattice_cache.geometry_info(fp, ac)
    ac.compute_reference_values(auto=geo["auto_ref"])
    rp = np.asarray(ac.moment_reference(), dtype=np.float64)
    S_ref, b_ref, c_ref = float(ac.S_ref), float(ac.b_ref), float(ac.c_ref)

    lattice = solver.build(ac, sett, condition, None, rp, fingerprint=fp)
    ground = ground_plane_from_condition(lattice, condition, rp)

    # Assemble harmonic ring system once and solve all requested k plus the k -> 0 probe.
    k_probe = float(K_ZERO_PROBE)
    probe_hits = np.flatnonzero(np.isclose(k_req, k_probe, rtol=0.0, atol=1.0e-15))
    if probe_hits.size > 0:
        k_solve = k_req.copy()
        probe_idx = int(probe_hits[0])
    else:
        k_solve = np.append(k_req, k_probe)
        probe_idx = int(k_req.size)

    system = _HarmonicRingSystem(
        lattice,
        condition,
        S_ref=S_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        ground=ground,
        ref_point=rp,
        wake_chords=wc,
        controls=ctrl_names,
        wake_alignment=sett.wake_alignment,
    )
    resp = system.solve(k_solve)
    pureq_solve = _solve_pure_pitch_rate_responses(system, resp, k_solve)

    # Steady base solve on the identical lattice for the preserved steady trust assessment.
    fixed_sett = sett.clone()
    fixed_sett.n_chord = int(lattice.n_chord)
    with kernel_cache_scope(lattice):
        base = solver.solve_batch(
            lattice,
            [condition],
            fixed_sett,
            S_ref,
            b_ref,
            c_ref,
            ref_point=rp,
            main_surface=geo["main"],
            continuation=False,
        )[0]
    base.condition = condition

    # Stack body-axis complex responses (n_solve, 6) and exact k0 vectors (6,).
    n_req = int(k_req.size)
    H_th_all = np.column_stack([getattr(resp.pitch, c) for c in _COEFFS])
    H_z_all = np.column_stack([getattr(resp.heave, c) for c in _COEFFS])
    H_pq_all = pureq_solve

    k0_pitch_vec = np.array([getattr(resp.k0_pitch, c) for c in _COEFFS], dtype=np.float64)
    k0_q_vec = np.array([getattr(resp.k0_pitch_rate, c) for c in _COEFFS], dtype=np.float64)
    k0_h_vec = np.array([getattr(resp.k0_heave, c) for c in _COEFFS], dtype=np.float64)
    mean_vec = np.array([getattr(resp.mean_state, c) for c in _COEFFS], dtype=np.float64)
    H_ctrl_all = {
        cname: np.column_stack([getattr(resp.controls[cname], c) for c in _COEFFS])
        for cname in ctrl_names
    }
    k0_ctrl_vec = {
        cname: np.array([getattr(resp.k0_controls[cname], c) for c in _COEFFS], dtype=np.float64)
        for cname in ctrl_names
    }

    # Apply stability-axes transform when requested (via rate_axes_transform).
    if ax == "stability":
        rot_bs = rotation_body_to_stability(float(condition.alpha))
        H_th_all = _transform_six_to_stability(H_th_all, rot_bs, b_ref, c_ref)
        H_z_all = _transform_six_to_stability(H_z_all, rot_bs, b_ref, c_ref)
        H_pq_all = _transform_six_to_stability(H_pq_all, rot_bs, b_ref, c_ref)
        k0_pitch_vec = _transform_six_to_stability(k0_pitch_vec, rot_bs, b_ref, c_ref)
        k0_q_vec = _transform_six_to_stability(k0_q_vec, rot_bs, b_ref, c_ref)
        k0_h_vec = _transform_six_to_stability(k0_h_vec, rot_bs, b_ref, c_ref)
        mean_vec = _transform_six_to_stability(mean_vec, rot_bs, b_ref, c_ref)
        for cname in ctrl_names:
            H_ctrl_all[cname] = _transform_six_to_stability(H_ctrl_all[cname], rot_bs, b_ref, c_ref)
            k0_ctrl_vec[cname] = _transform_six_to_stability(k0_ctrl_vec[cname], rot_bs, b_ref, c_ref)

    # Exact k -> 0 limits for alpha, q, h; small-k probe at k_probe for alpha_dot and h_dot.
    k0_alpha_dot_vec = H_th_all[probe_idx].imag / k_probe - k0_q_vec
    k0_h_dot_vec = H_z_all[probe_idx].imag / (2.0 * k_probe)
    k0_h_dot_ground_vec = k0_h_dot_vec + k0_pitch_vec

    k0_map: dict[str, float] = {}
    for ic, coeff in enumerate(_COEFFS):
        k0_map[f"{coeff}_alpha"] = float(k0_pitch_vec[ic])
        k0_map[f"{coeff}_q"] = float(k0_q_vec[ic])
        k0_map[f"{coeff}_alpha_dot"] = float(k0_alpha_dot_vec[ic])
        k0_map[f"{coeff}_h"] = float(k0_h_vec[ic])
        k0_map[f"{coeff}_h_dot"] = float(k0_h_dot_vec[ic])
        k0_map[f"{coeff}_h_dot_ground"] = float(k0_h_dot_ground_vec[ic])

    # Per-k equivalent derivatives on the user-requested k slice [0:n_req].
    H_th_req = H_th_all[:n_req]
    H_z_req = H_z_all[:n_req]
    H_pq_req = H_pq_all[:n_req]

    c_alpha_arr = np.empty((n_req, 6), dtype=np.float64)
    c_q_arr = np.empty((n_req, 6), dtype=np.float64)
    c_adot_arr = np.empty((n_req, 6), dtype=np.float64)
    c_h_arr = np.empty((n_req, 6), dtype=np.float64)
    c_hdot_arr = np.empty((n_req, 6), dtype=np.float64)
    c_hdot_g_arr = np.empty((n_req, 6), dtype=np.float64)

    for ik in range(n_req):
        kf = float(k_req[ik])
        if kf == 0.0:
            c_alpha_arr[ik] = k0_pitch_vec
            c_q_arr[ik] = k0_q_vec
            c_adot_arr[ik] = k0_alpha_dot_vec
            c_h_arr[ik] = k0_h_vec
            c_hdot_arr[ik] = k0_h_dot_vec
            c_hdot_g_arr[ik] = k0_h_dot_ground_vec
        else:
            c_alpha_arr[ik] = H_th_req[ik].real
            c_q_arr[ik] = H_pq_req[ik].imag / kf
            c_adot_arr[ik] = H_th_req[ik].imag / kf - c_q_arr[ik]
            c_h_arr[ik] = H_z_req[ik].real
            c_hdot_arr[ik] = H_z_req[ik].imag / (2.0 * kf)
            c_hdot_g_arr[ik] = c_hdot_arr[ik] + c_alpha_arr[ik]

    derivs: dict[str, np.ndarray] = {}
    H_theta_dict: dict[str, np.ndarray] = {}
    H_z_dict: dict[str, np.ndarray] = {}
    H_pureq_dict: dict[str, np.ndarray] = {}
    for ic, coeff in enumerate(_COEFFS):
        derivs[f"{coeff}_alpha"] = c_alpha_arr[:, ic].copy()
        derivs[f"{coeff}_q"] = c_q_arr[:, ic].copy()
        derivs[f"{coeff}_alpha_dot"] = c_adot_arr[:, ic].copy()
        derivs[f"{coeff}_h"] = c_h_arr[:, ic].copy()
        derivs[f"{coeff}_h_dot"] = c_hdot_arr[:, ic].copy()
        derivs[f"{coeff}_h_dot_ground"] = c_hdot_g_arr[:, ic].copy()
        H_theta_dict[coeff] = H_th_req[:, ic].copy()
        H_z_dict[coeff] = H_z_req[:, ic].copy()
        H_pureq_dict[coeff] = H_pq_req[:, ic].copy()

    H_delta_dict: dict[str, dict[str, np.ndarray]] = {}
    for cname in ctrl_names:
        H_d_all = H_ctrl_all[cname]
        k0_d_vec = k0_ctrl_vec[cname]
        k0_ddot_vec = H_d_all[probe_idx].imag / k_probe
        H_d_req = H_d_all[:n_req]

        c_d_arr = np.empty((n_req, 6), dtype=np.float64)
        c_ddot_arr = np.empty((n_req, 6), dtype=np.float64)
        for ik in range(n_req):
            kf = float(k_req[ik])
            if kf == 0.0:
                c_d_arr[ik] = k0_d_vec
                c_ddot_arr[ik] = k0_ddot_vec
            else:
                c_d_arr[ik] = H_d_req[ik].real
                c_ddot_arr[ik] = H_d_req[ik].imag / kf

        H_delta_dict[cname] = {}
        for ic, coeff in enumerate(_COEFFS):
            k0_map[f"{coeff}_delta_{cname}"] = float(k0_d_vec[ic])
            k0_map[f"{coeff}_delta_dot_{cname}"] = float(k0_ddot_vec[ic])
            derivs[f"{coeff}_delta_{cname}"] = c_d_arr[:, ic].copy()
            derivs[f"{coeff}_delta_dot_{cname}"] = c_ddot_arr[:, ic].copy()
            H_delta_dict[cname][coeff] = H_d_req[:, ic].copy()

    mean_state = {coeff: float(mean_vec[ic]) for ic, coeff in enumerate(_COEFFS)}

    trust, trust_notes = _build_unsteady_trust(
        base.totals.trust,
        n_chord=resp.n_chord,
        dx_te_ratio=resp.dx_te_ratio,
        k_values=k_req,
        wake_chords=wc,
    )

    notes = [
        "Equivalent oscillatory longitudinal derivatives at reduced frequency k = omega c_ref / (2 V_inf) "
        "(Rodden and Giesing 1970). Rates are normalised by c_ref / (2 V_inf); heave is per unit h / c_ref "
        "and heave rate is per unit (dz/dt) / V_inf.",
        f"k -> 0 rule: C_alpha, C_q and C_h come from the exact steady limits of the harmonic ring system; "
        f"C_alpha_dot and C_h_dot are evaluated at the small-frequency probe k = {k_probe:g}.",
        f"Moments in '{ax}' axes of the mean state, held fixed; CL, CD and CY are relative to the free stream.",
    ]
    if ctrl_names:
        notes.append(
            f"Harmonic control-surface modes for {list(ctrl_names)}: <C>_delta_<name> = Re(H_delta) [1/rad] "
            f"(exact steady limit at k = 0) and <C>_delta_dot_<name> = Im(H_delta) / k [1/rad] per unit "
            f"delta_dot c_ref / (2 V_inf) (k -> 0 limit evaluated at k = {k_probe:g})."
        )
    if ground is not None:
        notes.append(
            f"Ground effect at h={float(condition.h):g} m (h/c_ref={float(condition.h) / c_ref:.4g}): "
            "pitch is about ref_point at constant height, heave moves along the ground normal at constant attitude, "
            "and C_q is evaluated from the pure pitch-rate motion at constant height."
        )
    notes.extend(trust_notes)
    for note in resp.notes:
        if note not in notes:
            notes.append(note)
        if trust is not None and note not in trust.warnings:
            trust.warnings.append(note)

    return DynamicDerivativesResult(
        k=k_req.copy(),
        k0=k0_map,
        derivatives=derivs,
        H_theta=H_theta_dict,
        H_z=H_z_dict,
        H_pureq=H_pureq_dict,
        mean_state=mean_state,
        CL0=float(mean_state["CL"]),
        Cm0=float(mean_state["Cm"]),
        axes=ax,
        in_ground_effect=ground is not None,
        h_m=None if condition.h is None else float(condition.h),
        S_ref=S_ref,
        b_ref=b_ref,
        c_ref=c_ref,
        ref_point=rp.copy(),
        n_panels=int(resp.n_panels),
        n_strips=int(resp.n_strips),
        n_chord=int(resp.n_chord),
        n_wake_rows=int(resp.n_wake_rows),
        wake_chords=wc,
        wake_length=float(resp.wake_length),
        dx_w=float(resp.dx_w),
        dt=float(resp.dt),
        dx_te_ratio=resp.dx_te_ratio.copy(),
        k_zero_probe=k_probe,
        base=base,
        trust=trust,
        trust_notes=trust_notes,
        notes=notes,
        device=str(resp.device),
        precision=str(resp.precision),
        H_delta=H_delta_dict,
    )
