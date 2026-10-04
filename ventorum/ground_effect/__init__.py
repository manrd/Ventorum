# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Analyse wings and aircraft in ground effect.

The module does these tasks:

1. Rotate the geometry in 3-D about a reference point (pitch alpha, roll
   phi, yaw beta).
2. Keep the free stream parallel to the ground plane (V_z = 0).
3. Compute the 3-D forces and moments. These include the rolling moment
   that the ground makes when the aircraft banks (Cl_phi).
4. Track the clearances, the tip heights, the ground strikes and the
   critical bank limits.
5. Run parameter sweeps and compute the stability derivatives. Make plots
   of the results.
"""

from ventorum.ground_effect.state import (
    GroundEffectCondition,
    GroundEffectResult,
)
from ventorum.ground_effect.solver import (
    analyze_ground_effect,
)
from ventorum.ground_effect.sweep import (
    GroundEffectSweep,
    GroundEffectSweepResult,
    sweep_height,
    sweep_roll,
    sweep_alpha,
)
from ventorum.ground_effect.plotting import (
    plot_height_sweep,
    plot_roll_effect,
    plot_pitch_stability,
    plot_asymmetric_distributions,
    plot_ground_effect_matrix,
    plot_clearance_envelope,
)

__all__ = [
    "GroundEffectCondition",
    "GroundEffectResult",
    "analyze_ground_effect",
    "GroundEffectSweep",
    "GroundEffectSweepResult",
    "sweep_height",
    "sweep_roll",
    "sweep_alpha",
    "plot_height_sweep",
    "plot_roll_effect",
    "plot_pitch_stability",
    "plot_asymmetric_distributions",
    "plot_ground_effect_matrix",
    "plot_clearance_envelope",
]
