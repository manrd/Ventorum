# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Wing geometry: spanwise spacing, vortex lattice and strip discretisation."""

from ventorum.geometry.discretization import (
    get_spacing,
    fourier_collocation_angles,
    cosine_spacing,
    half_cosine_spacing,
    root_cosine_spacing,
    uniform_spacing,
    power_spacing,
    determine_optimal_spacing,
    compute_surface_n_panels,
)
from ventorum.geometry.processing import (
    discretize_surface,
    discretize_aircraft_surfaces,
)
from ventorum.geometry.transform import (
    rotation_matrix,
    transform_lattice,
)
from ventorum.geometry.controls import (
    flap_effectiveness,
    flap_moment_derivative,
    deflected_airfoil,
    validate_controls,
)

__all__ = [
    "get_spacing",
    "fourier_collocation_angles",
    "cosine_spacing",
    "half_cosine_spacing",
    "root_cosine_spacing",
    "uniform_spacing",
    "power_spacing",
    "determine_optimal_spacing",
    "compute_surface_n_panels",
    "discretize_surface",
    "discretize_aircraft_surfaces",
    "rotation_matrix",
    "transform_lattice",
    "flap_effectiveness",
    "flap_moment_derivative",
    "deflected_airfoil",
    "validate_controls",
]

