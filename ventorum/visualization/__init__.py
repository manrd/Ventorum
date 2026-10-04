# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Provide the plot functions of Ventorum."""

from __future__ import annotations

from ventorum.visualization.distributions import (
    plot_lift_distribution,
    plot_cl_distribution,
    plot_alpha_distribution,
    plot_induced_drag_distribution,
    plot_all_distributions,
    plot_fourier_spectrum,
    plot_component_breakdown,
)
from ventorum.visualization.drag_polar import (
    alpha_sweep,
    plot_drag_polar,
    plot_cl_vs_alpha,
    plot_sweep_summary,
)
from ventorum.visualization.geometry_plot import (
    plot_geometry,
    plot_surface_results,
    plot_panel_results,
)
from ventorum.visualization.convergence import plot_convergence
from ventorum.visualization.wake_plot import (
    plot_vortex_wake,
    plot_trefftz_plane,
)
from ventorum.visualization.planform_plot import (
    plot_planform_2d,
    plot_span_loading,
)
from ventorum.visualization.performance_plot import (
    plot_efficiency_curves,
    plot_pitching_moment,
    plot_airfoil_polar,
)
from ventorum.visualization.dashboard import plot_aircraft_dashboard

__all__ = [
    # Distributions
    "plot_lift_distribution",
    "plot_cl_distribution",
    "plot_alpha_distribution",
    "plot_induced_drag_distribution",
    "plot_all_distributions",
    "plot_fourier_spectrum",
    "plot_component_breakdown",
    # Polars & Sweeps
    "alpha_sweep",
    "plot_drag_polar",
    "plot_cl_vs_alpha",
    "plot_sweep_summary",
    # Geometry & 3D Surface Panels
    "plot_geometry",
    "plot_surface_results",
    "plot_panel_results",
    # Convergence
    "plot_convergence",
    # Wake & Trefftz Plane
    "plot_vortex_wake",
    "plot_trefftz_plane",
    # Planform & Span Loading
    "plot_planform_2d",
    "plot_span_loading",
    # Performance & Stability
    "plot_efficiency_curves",
    "plot_pitching_moment",
    "plot_airfoil_polar",
    # Dashboard
    "plot_aircraft_dashboard",
]
