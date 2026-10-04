# Author: Manuel Alejandro Rodriguez Diaz, PhD
# Wing geometry definition, discretization, processing, and transformation for Ventorum.

from ventorum.legacy.geometry.discretization import (
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
from ventorum.legacy.geometry.processing import (
    discretize_surface,
    discretize_aircraft_surfaces,
    build_fourier_stations,
)
from ventorum.legacy.geometry.transform import (
    rotation_matrix_body,
    transform_discretized_surface,
    compute_ground_clearance,
    find_roll_strike_limit,
    prepare_ground_effect_geometry,
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
    "build_fourier_stations",
    "rotation_matrix_body",
    "transform_discretized_surface",
    "compute_ground_clearance",
    "find_roll_strike_limit",
    "prepare_ground_effect_geometry",
]
