# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Spanwise discretisation schemes.

Two strategies are provided:

* **cosine** (default) - nodes are clustered near the wing tips where
  circulation gradients are steepest, giving superior convergence.
* **uniform** - evenly spaced nodes along the span.
"""

from __future__ import annotations

import numpy as np


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
        Panel-centre locations (arithmetic mean of edges).
    """
    y_frac_edges = (1.0 - np.cos(np.linspace(0.0, np.pi, n_panels + 1))) / 2.0
    y_frac_mids = 0.5 * (y_frac_edges[:-1] + y_frac_edges[1:])
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
    y_frac_mids = 0.5 * (y_frac_edges[:-1] + y_frac_edges[1:])
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
    y_frac_mids = 0.5 * (y_frac_edges[:-1] + y_frac_edges[1:])
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
    xi = np.linspace(0.0, 1.0, n_panels + 1)
    c_mode = clustering.lower()
    if c_mode in ("tip", "half-cosine"):
        y_frac_edges = 1.0 - (1.0 - xi) ** p
    elif c_mode in ("root", "root-cosine"):
        y_frac_edges = xi ** p
    elif c_mode in ("both", "symmetric", "cosine"):
        xi_sym = np.linspace(-1.0, 1.0, n_panels + 1)
        y_frac_edges = 0.5 * (1.0 + np.sign(xi_sym) * (np.abs(xi_sym) ** p))
    else:
        y_frac_edges = xi

    y_frac_mids = 0.5 * (y_frac_edges[:-1] + y_frac_edges[1:])
    return y_frac_edges, y_frac_mids


def determine_optimal_spacing(
    surf: any = None,
    dihedral_threshold_deg: float = 5.0,
    sweep_threshold_deg: float = 15.0,
) -> str:
    """Decide the optimal spanwise discretisation scheme based on surface geometry.

    Criteria:
    ---------
    1. Planar / Near-Planar Wing (|dihedral| <= 5° and |sweep| <= 15°):
       By mirror symmetry, dΓ/dy = 0 at the root (y = 0). Refinement at the root
       is redundant and wastes DOF. Concentrates panels at the wingtip where
       dΓ/dy -> ∞.
       -> Returns ``"half-cosine"`` (tip-clustered).

    2. Angled Junction / Dihedral Kink (|dihedral| > 5°, e.g. V-tail apex):
       Adjacent panels meet at the symmetry plane with a non-planar angle
       (2 * |dihedral|). Mutual 3-D bound and trailing vortex induction produces
       curvature and crossflow gradients near the apex.
       -> Returns ``"cosine"`` (clustered at both root and tip).

    3. High-Sweep Wing (|sweep_le| > 15°):
       Kuchemann center-section effect creates root downwash gradients.
       -> Returns ``"cosine"`` (clustered at both root and tip).

    4. Asymmetric Surface (e.g. vertical tail):
       Attached root with free tip.
       -> Returns ``"half-cosine"``.
    """
    if surf is None:
        return "half-cosine"

    dih_thresh = np.radians(dihedral_threshold_deg)
    sweep_thresh = np.radians(sweep_threshold_deg)

    is_sym = getattr(surf, "is_symmetric", True)
    dihedral = abs(getattr(surf, "dihedral", 0.0))
    sweep_le = abs(getattr(surf, "sweep_le", 0.0))

    if is_sym:
        if dihedral > dih_thresh:
            # Significant dihedral kink / non-planar apex (e.g. inverted V-tail)
            return "cosine"
        if sweep_le > sweep_thresh:
            # Significant sweep root effect
            return "cosine"
        # Planar or near-planar wing: dΓ/dy = 0 at root, refine tip only
        return "half-cosine"
    else:
        # Asymmetric surface (e.g., vertical tail): refine tip
        return "half-cosine"


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
    if b_surf < (b_ref - 1e-4):
        max_sec_panels = max(min_panels, int(round(b_surf * 80.0)))
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
