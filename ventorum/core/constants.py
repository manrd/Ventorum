# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Physical constants (ISA sea level) and the limits of the valid envelope."""

RHO_SL = 1.225       # Air density at sea level [kg/m^3]
A_SL = 340.294       # Speed of sound at sea level [m/s]
MU_SL = 1.7894e-5    # Dynamic viscosity of air at sea level [Pa s] (ISO 2533:1975)

# The methods are incompressible. Above this Mach number the results are out
# of the valid envelope (no compressibility correction is applied).
MACH_LIMIT_INCOMPRESSIBLE = 0.3

# A lift or moment slope below this value [per rad or per unit h/c] gives no
# aerodynamic centre (pitch or height) and no Irodov margin.
MIN_CENTRE_SLOPE = 1e-9
