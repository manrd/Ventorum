# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Physical constants (ISA sea level) and the limits of the valid envelope."""

RHO_SL = 1.225       # Air density at sea level [kg/m^3]
A_SL = 340.294       # Speed of sound at sea level [m/s]

# The methods are incompressible. Above this Mach number the results are out
# of the valid envelope (no compressibility correction is applied).
MACH_LIMIT_INCOMPRESSIBLE = 0.3
