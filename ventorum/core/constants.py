# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Physical constants (ISA sea level) and the limits of the valid envelope."""

RHO_SL = 1.225       # Air density at sea level [kg/m^3]
A_SL = 340.294       # Speed of sound at sea level [m/s]
MU_SL = 1.7894e-5    # Dynamic viscosity of air at sea level [Pa s] (ISO 2533:1975)

# The methods are incompressible. Above this Mach number the results are out
# of the valid envelope. The Prandtl-Glauert (Goethert) correction of the
# vortex lattice is then an estimate with the trust rating UNRELIABLE.
MACH_LIMIT_INCOMPRESSIBLE = 0.3

# A stated Mach number at or above this value is refused (ValidityError):
# near the critical Mach number the local flow can become supersonic, and the
# linear Prandtl-Glauert rule has no meaning there.
MACH_LIMIT_REFUSED = 0.7

# Largest difference [-] between the stated Mach number and V_inf / a before
# a warning and a trust note say that the two do not agree.
MACH_SPEED_TOLERANCE = 0.05

# A lift or moment slope below this value [per rad or per unit h/c] gives no
# aerodynamic centre (pitch or height) and no Irodov margin.
MIN_CENTRE_SLOPE = 1e-9
