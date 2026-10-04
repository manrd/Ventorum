# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Spanwise discretisation schemes.

Every scheme maps a uniform parameter ``t`` in [0, 1] to the span fraction.
Panel edges are at ``t_k = k / n`` and the control points at the mid
parameter ``t = (k + 1/2) / n``. For cosine spacing this is the classical
"semicircle" (Multhopp) rule. With control points and Trefftz-plane points
at these locations, the elliptic wing gives ``e = 1`` already with 10 to 20
panels per semi-span; with control points at the mean of the edges the error
in ``e`` falls only as ``1/n``.

Root clustering (``"cosine"`` per semi-span) on a planar symmetric wing
leaves a small spurious dip of the circulation next to the root (0.1 % to
0.3 % of the root value, growing slowly with refinement; the integrated
coefficients still converge). ``"half-cosine"`` (cosine over the whole span)
has no such artifact, and ``"auto"`` uses it for planar wings.

The rules are ``"auto"`` (the default, see
:func:`ventorum.geometry.lattice.resolve_spacing`), ``"cosine"``,
``"half-cosine"``, ``"root-cosine"``, ``"uniform"`` and ``"power"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _param_mids(n_panels: int) -> np.ndarray:
    """Parameter values half way between the panel edges, t = (k + 1/2) / n."""
    return (np.arange(n_panels) + 0.5) / n_panels


def cosine_spacing(n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-cosine spaced panel edges for one semi-span (clustered at both root and tip).

    Parameters
    ----------
    n_panels : int
        Number of panels in the semi-span.

    Returns
    -------
    y_frac_edges : np.ndarray, shape (n_panels + 1,)
        Panel-edge locations as fractions of the semi-span, from 0 (root)
        to 1 (tip).
    y_frac_mids : np.ndarray, shape (n_panels,)
        Control-point locations at the mid parameter of each panel
        (the "theta" midpoints, not the mean of the edges).
    """
    y_frac_edges = (1.0 - np.cos(np.linspace(0.0, np.pi, n_panels + 1))) / 2.0
    y_frac_mids = (1.0 - np.cos(np.pi * _param_mids(n_panels))) / 2.0
    return y_frac_edges, y_frac_mids


def half_cosine_spacing(n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-cosine spaced panel edges (clustered specifically at the wing tip).

    Parameters
    ----------
    n_panels : int
        Number of panels in the semi-span.

    Returns
    -------
    y_frac_edges : np.ndarray, shape (n_panels + 1,)
    y_frac_mids : np.ndarray, shape (n_panels,)
    """
    y_frac_edges = np.sin(np.linspace(0.0, np.pi / 2.0, n_panels + 1))
    y_frac_mids = np.sin(0.5 * np.pi * _param_mids(n_panels))
    return y_frac_edges, y_frac_mids


def root_cosine_spacing(n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine spaced panel edges clustered specifically at the wing root (coarse at tip).

    Parameters
    ----------
    n_panels : int
        Number of panels in the semi-span.

    Returns
    -------
    y_frac_edges : np.ndarray, shape (n_panels + 1,)
    y_frac_mids : np.ndarray, shape (n_panels,)
    """
    y_frac_edges = 1.0 - np.cos(np.linspace(0.0, np.pi / 2.0, n_panels + 1))
    y_frac_mids = 1.0 - np.cos(0.5 * np.pi * _param_mids(n_panels))
    return y_frac_edges, y_frac_mids


def uniform_spacing(n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly spaced panel edges for one semi-span.

    Parameters
    ----------
    n_panels : int
        Number of panels in the semi-span.

    Returns
    -------
    y_frac_edges : np.ndarray, shape (n_panels + 1,)
    y_frac_mids : np.ndarray, shape (n_panels,)
    """
    y_frac_edges = np.linspace(0.0, 1.0, n_panels + 1)
    y_frac_mids = 0.5 * (y_frac_edges[:-1] + y_frac_edges[1:])
    return y_frac_edges, y_frac_mids


def power_spacing(
    n_panels: int,
    power: float = 1.4,
    clustering: str = "tip",
) -> tuple[np.ndarray, np.ndarray]:
    """Power-law stretched panel edges with controlled clustering exponent.

    Provides gentle, non-singular clustering without the aggressive O(1/N^2)
    sub-millimeter collapse of full cosine spacing.

    Parameters
    ----------
    n_panels : int
        Number of panels in the semi-span.
    power : float
        Stretching exponent beta >= 1.0 (1.0 = uniform, 1.3-1.6 = gentle clustering).
    clustering : str
        ``"tip"`` (default), ``"root"``, or ``"both"``.

    Returns
    -------
    y_frac_edges, y_frac_mids
    """
    p = max(1.0, float(power))
    c_mode = clustering.lower()

    def mapping(t: np.ndarray) -> np.ndarray:
        if c_mode in ("tip", "half-cosine"):
            return 1.0 - (1.0 - t) ** p
        if c_mode in ("root", "root-cosine"):
            return t ** p
        if c_mode in ("both", "symmetric", "cosine"):
            u = 2.0 * t - 1.0
            return 0.5 * (1.0 + np.sign(u) * (np.abs(u) ** p))
        return t

    y_frac_edges = mapping(np.linspace(0.0, 1.0, n_panels + 1))
    y_frac_mids = mapping(_param_mids(n_panels))
    return y_frac_edges, y_frac_mids


def determine_optimal_spacing(
    surf: any = None,
    dihedral_threshold_deg: float = 5.0,
    sweep_threshold_deg: float = 15.0,
) -> str:
    """Return the best spanwise spacing scheme for the geometry of a surface.

    Parameters
    ----------
    surf : LiftingSurface or None, optional
        The surface. If None, the function returns ``"half-cosine"``.
    dihedral_threshold_deg : float, optional
        Dihedral limit [deg] of a near-planar wing. The default is 5.
    sweep_threshold_deg : float, optional
        Sweep limit [deg] of a low-sweep wing. The default is 15.

    Returns
    -------
    str
        ``"half-cosine"`` (panels clustered at the tip) or ``"cosine"``
        (panels clustered at the root and at the tip).

    Notes
    -----
    The angles are those of the quarter-chord line next to the root. The
    criteria with the default limits are:

    1. Planar or near-planar symmetric wing (abs(dihedral) <= 5 deg and
       abs(sweep) <= 15 deg). By mirror symmetry, dGamma/dy = 0 at the root
       (y = 0). More panels at the root do not help. The panels go to the
       tip, where dGamma/dy becomes infinite. The result is
       ``"half-cosine"``.
    2. Symmetric wing with a dihedral kink (abs(dihedral) > 5 deg, for
       example the apex of a V-tail). The two halves meet at the symmetry
       plane at an angle of 2 * abs(dihedral). The bound and trailing
       vortices of each half cause curvature and cross-flow gradients near
       the apex. The result is ``"cosine"``.
    3. Symmetric wing with high sweep (abs(sweep) > 15 deg). The Kuchemann
       centre-section effect causes downwash gradients at the root. The
       result is ``"cosine"``.
    4. Surface that is not symmetric (for example a vertical tail). The root
       is attached and the tip is free. The result is ``"half-cosine"``.
    """
    if surf is None:
        return "half-cosine"

    dih_thresh = np.radians(dihedral_threshold_deg)
    sweep_thresh = np.radians(sweep_threshold_deg)

    is_sym = getattr(surf, "is_symmetric", True)
    dihedral, sweep_le = _effective_angles(surf)

    if is_sym:
        if dihedral > dih_thresh + 1e-9:
            # Significant dihedral kink / non-planar apex (e.g. inverted V-tail)
            return "cosine"
        if sweep_le > sweep_thresh + 1e-9:
            # Significant sweep root effect
            return "cosine"
        # Planar or near-planar wing: dΓ/dy = 0 at root, refine tip only
        return "half-cosine"
    else:
        # Asymmetric surface (e.g., vertical tail): refine tip
        return "half-cosine"


def _effective_angles(surf: Any) -> tuple[float, float]:
    """Return abs(dihedral) and abs(sweep) [rad] of the quarter-chord line next to the root.

    The root kink of the quarter-chord line is what needs root clustering.
    The angles come from the actual geometry over the inner 10 % of the span,
    so explicit section ``x_le`` and ``z_le`` count the same as the
    ``sweep_le`` and ``dihedral`` fields, and a smooth curved planform (for
    example an elliptic or circular wing) is not taken as swept.
    """
    try:
        from ventorum.geometry.lattice import _sorted_sections, surface_reference_line

        eta = np.array([0.0, 0.1])
        x, y, z = surface_reference_line(surf, eta)
        secs = _sorted_sections(surf)
        fr = np.array([sec.y_frac for sec in secs])
        chord = np.interp(eta, fr, np.array([sec.chord for sec in secs]))
        xq = x + 0.25 * chord
        dx, dy, dz = float(xq[1] - xq[0]), float(y[1] - y[0]), float(z[1] - z[0])
        span_yz = max(np.hypot(dy, dz), 1e-12)
        return abs(float(np.arctan2(dz, abs(dy) + 1e-300))), abs(float(np.arctan2(dx, span_yz)))
    except (AttributeError, TypeError, ValueError):
        return abs(getattr(surf, "dihedral", 0.0)), abs(getattr(surf, "sweep_le", 0.0))


def compute_surface_n_panels(
    surf: any,
    base_n_panels: int = 40,
    reference_semi_span: float | None = None,
    min_panels: int = 8,
    max_panels: int | None = None,
) -> int:
    """Compute the appropriate panel count for a surface to prevent over/under-refinement.

    Scales panel count proportionally to relative semi-span so that secondary surfaces
    (e.g., small tailplanes) do not suffer from sub-millimeter aspect-ratio collapse.

    Parameters
    ----------
    surf : LiftingSurface
    base_n_panels : int
        Nominal panels per semi-span for the primary reference wing.
    reference_semi_span : float, optional
        Reference wing semi-span. If None, uses surf.semi_span.
    min_panels : int
        Minimum allowed panels per semi-span.
    max_panels : int or None, optional
        Maximum allowed panels per semi-span. If None, defaults to max(500, base_n_panels * 2).
    """
    # 1. Explicit user override on surface takes precedence
    user_n = getattr(surf, "n_panels", None)
    if user_n is not None:
        return max(min_panels, int(user_n))

    b_surf = getattr(surf, "semi_span", 1.0)
    b_ref = reference_semi_span if reference_semi_span is not None and reference_semi_span > 0 else b_surf

    # Square-root span scaling balances aspect ratio without starving small surfaces
    scale = np.sqrt(max(0.01, b_surf / b_ref))
    n_computed = int(round(base_n_panels * scale))

    # Safeguard secondary surfaces (tails, canards) against sub-millimeter aspect-ratio
    # collapse and discrete vortex dipole singularities when base_n_panels is very large:
    if b_surf < b_ref * (1.0 - 1e-9):
        # Strips on a smaller surface are not narrower than half the strips of
        # the reference surface (independent of the length unit).
        max_sec_panels = max(min_panels, int(round(2.0 * base_n_panels * b_surf / b_ref)))
        n_computed = min(n_computed, max_sec_panels)

    eff_max = max_panels if max_panels is not None else max(500, int(base_n_panels * 2))
    return int(np.clip(n_computed, min_panels, eff_max))


def get_spacing(
    method: str,
    n_panels: int,
    surf: any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the requested spacing scheme.

    Parameters
    ----------
    method : str
        ``"cosine"``, ``"half-cosine"``, ``"root-cosine"``, ``"uniform"``,
        ``"power"``, or ``"auto"``.
    n_panels : int
        Number of panels per semi-span.
    surf : LiftingSurface, optional
        Surface instance, inspected when ``method == "auto"``.

    Returns
    -------
    y_frac_edges, y_frac_mids
    """
    m = method.lower().replace("_", "-")
    if m == "auto":
        m = determine_optimal_spacing(surf).replace("_", "-")

    if m in ("cosine", "full-cosine", "both"):
        return cosine_spacing(n_panels)
    if m in ("half-cosine", "tip", "tip-clustered"):
        return half_cosine_spacing(n_panels)
    if m in ("root-cosine", "root", "root-clustered"):
        return root_cosine_spacing(n_panels)
    if m == "uniform":
        return uniform_spacing(n_panels)
    if m in ("power", "stretched"):
        return power_spacing(n_panels, power=1.4, clustering="tip")
    raise ValueError(
        f"Unknown spacing method: {method!r}. Options: 'auto', 'cosine', 'half-cosine', 'root-cosine', 'uniform', 'power'."
    )


def fourier_collocation_angles(n_terms: int) -> np.ndarray:
    """Collocation angles θ_k for the Fourier solver.

    Uses the standard placement ``θ_k = k·π / (N+1)`` for *k = 1 … N* to
    avoid the wing-tip singularities at θ = 0 and θ = π.

    Parameters
    ----------
    n_terms : int
        Number of Fourier terms / collocation stations.

    Returns
    -------
    theta : np.ndarray, shape (n_terms,)
        Angles in ``(0, π)`` rad.
    """
    return np.arange(1, n_terms + 1, dtype=float) * (np.pi / (n_terms + 1))
