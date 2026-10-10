# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Hinge moments of the control surfaces of a solved case.

Two models, the same as the two control modes of the lattice
(:mod:`ventorum.geometry.controls`):

* Chordwise mode (vortex lattice with ``n_chord >= 2``): the hinge moment is
  the moment about the hinge line of the Kutta-Joukowski forces on the bound
  vortices of the flap panels, ``F = rho * Gamma * (V_inf + v) x l``.
* Section mode (lifting lines, vortex lattice with ``n_chord = 1``): each
  strip gets the thin-airfoil section hinge moment
  ``Ch = Ch_alpha * alpha_fp + Ch_delta * delta``
  (:func:`ventorum.geometry.controls.hinge_moment_derivatives`), with
  ``alpha_fp`` the flat-plate angle of attack ahead of the hinge.

The hinge moment of a copy of a control is positive when it tends to
increase the local deflection of that copy: a right-hand moment about the
hinge axis of the lattice (trailing edge down on a horizontal surface, on
both halves). The coefficient is ``Ch = H / (q S_f c_f)``, with ``S_f`` the
flap area aft of the hinge and ``c_f`` the root-mean-square flap chord
``sum(c_f,i^2 w_i) / sum(c_f,i w_i)``. Thus ``Ch`` is the
``c_f^2 w``-weighted mean of the strip values, and a two-dimensional
section gives the section coefficient.

References
----------
* J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
  University Press, 2001: the Kutta-Joukowski force on the bound vortex of
  a lattice panel.
* H. Glauert, The Elements of Aerofoil and Airscrew Theory, Cambridge
  University Press, 1926: thin-airfoil theory of a hinged flap.
* B. Etkin and L. D. Reid, Dynamics of Flight: Stability and Control,
  3rd ed., Wiley, 1996: the hinge-moment coefficient referred to the area
  and chord of the control aft of the hinge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ventorum.aero.system import UnknownMap, build_sources, freestream_direction, panel_targets
from ventorum.aero.vortex import induced_velocity
from ventorum.core.datatypes import SolverResult
from ventorum.geometry.controls import flap_effectiveness, hinge_moment_derivatives
from ventorum.geometry.lattice import VortexLattice
from ventorum.utils.vec import cross3

__all__ = ["HingeMoment", "hinge_moments"]


@dataclass
class HingeMoment:
    """Hinge moment of one copy of one control surface.

    Attributes
    ----------
    surface : str
        Name of the lifting surface.
    control : str
        Name of the control surface.
    side : str
        ``"right"`` or ``"left"`` for the two halves of a symmetric
        surface, ``"single"`` for a surface without a mirror half.
    mode : str
        ``"chordwise"`` (panel loads aft of the hinge) or ``"section"``
        (thin-airfoil section theory).
    deflection : float
        Local deflection of this copy [rad]: the deflection of the control,
        with the opposite sign on the left half of an antisymmetric control.
    hinge_x_c : float
        Hinge position used [-]: the effective hinge of the chordwise mode,
        the given hinge of the section mode.
    H : float
        Hinge moment [N m], positive when it tends to increase the local
        deflection.
    Ch : float
        Hinge-moment coefficient ``H / (q S_f c_f)`` [-].
    S_f : float
        Flap area aft of the hinge [m^2].
    c_f : float
        Root-mean-square flap chord aft of the hinge [m].
    n_strips : int
        Number of strips of this copy.
    """

    surface: str
    control: str
    side: str
    mode: str
    deflection: float
    hinge_x_c: float
    H: float
    Ch: float
    S_f: float
    c_f: float
    n_strips: int

    def to_dict(self) -> dict[str, Any]:
        """Return the values as a dictionary of plain Python types."""
        return {
            "surface": self.surface, "control": self.control, "side": self.side, "mode": self.mode,
            "deflection_rad": float(self.deflection), "hinge_x_c": float(self.hinge_x_c),
            "H_Nm": float(self.H), "Ch": float(self.Ch), "S_f_m2": float(self.S_f),
            "c_f_m": float(self.c_f), "n_strips": int(self.n_strips),
        }


def _copies(lattice: VortexLattice, info: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Return ``(side, strips)`` for each copy of the control *info* on the lattice."""
    surf = next((s for s in lattice.surfaces if s.name == info["surface"]), None)
    if surf is None:
        raise ValueError(f"Surface {info['surface']!r} of control {info['name']!r} is not in the lattice.")
    strips = np.arange(lattice.n_strips)[surf.strips]
    eta = np.abs(lattice.eta[strips])
    # The strip mid-points of a control are strictly inside its snapped limits.
    tol = 1e-9
    inside = (eta > info["eta_start_eff"] - tol) & (eta < info["eta_end_eff"] + tol)
    sel = strips[inside]
    if surf.is_symmetric:
        right = sel[lattice.strip_is_right[sel]]
        left = sel[~lattice.strip_is_right[sel]]
        copies = [("right", right), ("left", left)]
    else:
        copies = [("single", sel)]
    for side, idx in copies:
        if idx.size != info["n_strips"]:
            raise RuntimeError(
                f"Control {info['name']!r} on {info['surface']!r} ({side}): found {idx.size} strips, "
                f"the lattice has {info['n_strips']}."
            )
    return copies


def _local_deflection(lattice: VortexLattice, info: dict[str, Any], strip: int) -> float:
    """Return the local deflection [rad] of a strip (the same rule as the lattice builder)."""
    delta = float(info["deflection_rad"])
    if bool(lattice.strip_is_right[strip]) or bool(info["symmetric"]):
        return delta
    return -delta


def _bound_forces(result: SolverResult, panels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the force points [m] and the bound-vortex forces [N] of the given panels."""
    d = result.details
    lattice: VortexLattice = d["lattice"]
    gamma = np.asarray(d["gamma"], dtype=float)
    cond = result.condition
    n = lattice.n_panels
    umap = UnknownMap(unknown_panels=np.arange(n), panel_column=np.arange(n), symmetric=False)
    src = build_sources(lattice, np.asarray(d["wake_dir"], dtype=float), umap, d.get("ground"))
    P = lattice.force_points[panels]
    v = induced_velocity(P, src, gamma, panel_targets(lattice, panels))
    V = float(cond.V_inf) * freestream_direction(cond.alpha, cond.beta)
    l_vec = lattice.b[panels] - lattice.a[panels]
    F = float(cond.rho) * gamma[panels, None] * cross3(V[None, :] + v, l_vec)
    return P, F


def _hinge_line(lattice: VortexLattice, s: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the mid-point [m] and the unit axis of the hinge line of strip *s* (lattice order)."""
    H_l = lattice.le_left[s] + h * (lattice.te_left[s] - lattice.le_left[s])
    H_r = lattice.le_right[s] + h * (lattice.te_right[s] - lattice.le_right[s])
    axis = H_r - H_l
    return 0.5 * (H_l + H_r), axis / max(float(np.linalg.norm(axis)), 1e-300)


def hinge_moments(result: SolverResult) -> list[HingeMoment]:
    """Return the hinge moment of each copy of each control surface of a solved case.

    Parameters
    ----------
    result : SolverResult
        Result of a vortex-lattice or lifting-line solve (CPU or GPU), with
        ``result.details["lattice"]``. The Fourier solver has no controls.

    Returns
    -------
    list of HingeMoment
        One item per copy (right and left half of a symmetric surface), in
        the order of the controls of the lattice.

    Raises
    ------
    ValueError
        If the result has no lattice (Fourier solver) or no control surface.

    Notes
    -----
    Chordwise mode: the force on the bound vortex of each flap panel is
    ``rho Gamma (V_inf + v) x l``, with the induced velocity ``v`` from a new
    evaluation of the kernel at the force points (as the loads, with the
    ground images and the wake of the case). The forces on the chordwise
    legs are left out: the two legs of one horseshoe carry opposite
    vorticity over the same chordwise interval, so their moments about the
    hinge cancel except for the difference of the induced velocity at the
    two legs. The hinge moment converges to first order in the number of
    chordwise panels (the thin-airfoil load is logarithmic at the hinge).

    Section mode: ``alpha_fp = alpha_eff - alpha_L0 - tau delta`` per strip,
    with ``alpha_L0`` the zero-lift angle of the deflected strip section and
    ``tau`` the flap effectiveness. In the linear range this is
    ``Cl / a0 - tau delta``. The camber of the base section adds no hinge
    moment in this model, and thin-airfoil theory does not hold past stall.
    """
    d = getattr(result, "details", None) or {}
    lattice = d.get("lattice")
    if lattice is None:
        raise ValueError("The result has no lattice (the Fourier solver has no control surfaces). "
                         "Use solver 'vlm', 'linear' or 'nonlinear'.")
    infos = list(getattr(lattice, "control_info", []) or [])
    if not infos:
        raise ValueError("The lattice of the result has no control surface.")
    cond = result.condition
    q = 0.5 * float(cond.rho) * float(cond.V_inf) ** 2
    n_c = lattice.n_chord

    plan: list[tuple[dict, str, np.ndarray]] = []
    for info in infos:
        for side, strips in _copies(lattice, info):
            plan.append((info, side, strips))

    # One kernel evaluation for the flap panels of all chordwise copies.
    flap_of: dict[int, np.ndarray] = {}
    for info, _, strips in plan:
        if info["mode"] != "chordwise":
            continue
        h = float(info["hinge_x_c_eff"])
        for s in strips:
            panels = s * n_c + np.arange(n_c)
            x0 = np.concatenate([[0.0], np.cumsum(lattice.panel_length[panels])[:-1]]) / lattice.chord[s]
            flap_of[int(s)] = panels[x0 >= h - 1e-9]
    if flap_of:
        all_flap = np.concatenate(list(flap_of.values()))
        P_all, F_all = _bound_forces(result, all_flap)
        row = {int(p): i for i, p in enumerate(all_flap)}

    alpha_eff = None
    if any(info["mode"] == "section" for info, _, _ in plan):
        alpha_eff = np.asarray(d["loads"].alpha_eff, dtype=float)

    out: list[HingeMoment] = []
    for info, side, strips in plan:
        chordwise = info["mode"] == "chordwise"
        h = float(info["hinge_x_c_eff"] if chordwise else info["hinge_x_c"])
        cf = (1.0 - h) * lattice.chord[strips]
        w = lattice.width[strips]
        H = 0.0
        if chordwise:
            for s in strips:
                rows = [row[int(p)] for p in flap_of[int(s)]]
                mid, axis = _hinge_line(lattice, int(s), h)
                H += float(np.sum(cross3(P_all[rows] - mid, F_all[rows]) @ axis))
        else:
            ch_a, ch_d = hinge_moment_derivatives(h)
            tau = flap_effectiveness(h)
            for i, s in enumerate(strips):
                dl = _local_deflection(lattice, info, int(s))
                a_fp = alpha_eff[s] - lattice.alpha_L0[s] - tau * dl
                H += q * cf[i] ** 2 * w[i] * (ch_a * a_fp + ch_d * dl)
        denom = q * float(np.sum(cf ** 2 * w))
        S_f = float(np.sum(cf * w))
        out.append(HingeMoment(
            surface=info["surface"], control=info["name"], side=side, mode=info["mode"],
            deflection=_local_deflection(lattice, info, int(strips[0])), hinge_x_c=h,
            H=H, Ch=H / denom if denom > 0.0 else float("nan"),
            S_f=S_f, c_f=float(np.sum(cf ** 2 * w)) / S_f if S_f > 0.0 else float("nan"),
            n_strips=int(strips.size),
        ))
    return out
