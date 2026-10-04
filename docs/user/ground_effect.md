# Ground effect

```python
from ventorum.ground_effect import analyze_ground_effect

res = analyze_ground_effect(wing, h=0.6, alpha_deg=4.0, phi_deg=2.0, height_ref="te")
print(res.CL, res.CDi, res.Cl, res.h_min_over_c)
```

- The ground is flat and rigid. The method of images makes the velocity normal to the ground zero. The wake leaves the trailing edge parallel to the ground.
- `height_ref` sets the definition of h: `"ref"` (the moment reference point), `"min"` (the lowest edge point), `"qc"` or `"te"` (the root quarter-chord or trailing-edge point of the main surface).
- Contact with the ground raises `GroundStrikeError`. The lifting-line solvers raise `ValidityError` below h_min/c = 1. Both are subclasses of `ValueError`.
- `GroundEffectSweep` runs grids of heights, angles of attack and bank angles. It uses one chordwise mesh for all cases. `compute_stability_derivatives()` gives the pitch and height aerodynamic centres and the Irodov margin `(x_alpha - x_h)/c` (stable if larger than 0).
- `vt.analyze(..., condition=vt.FlightCondition(..., h=...))` uses the same convention: h is the height of the moment reference point.
