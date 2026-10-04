# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Physical and numerical constants used throughout Ventorum.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Atmospheric constants (ISA sea-level)
# ---------------------------------------------------------------------------
RHO_SL = 1.225          # Air density at sea level [kg/m³]
T_SL = 288.15            # Temperature at sea level [K]
P_SL = 101325.0          # Pressure at sea level [Pa]
MU_SL = 1.7894e-5        # Dynamic viscosity at sea level [Pa·s]
A_SL = 340.294            # Speed of sound at sea level [m/s]

# ---------------------------------------------------------------------------
# Mathematical / Aerodynamic constants
# ---------------------------------------------------------------------------
TWO_PI = 2.0 * np.pi     # Thin-airfoil-theory lift-curve slope [1/rad]

# ---------------------------------------------------------------------------
# Numerical defaults
# ---------------------------------------------------------------------------
VORTEX_CORE_RADIUS = 1.0e-10   # Regularisation radius for Biot-Savart [m]
DEFAULT_N_PANELS = 80           # Default number of panels per semi-span
DEFAULT_TOLERANCE = 1.0e-6      # Default convergence tolerance
DEFAULT_MAX_ITER = 200          # Default maximum iterations (nonlinear)
DEFAULT_RELAXATION = 0.3        # Default relaxation factor (nonlinear)
