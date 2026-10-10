# Run a ground-effect study

The ground is flat and rigid. The method of images makes the velocity normal to the ground zero. The wake leaves the trailing edge parallel to the ground.

## One case

```python
from ventorum.ground_effect import analyze_ground_effect

res = analyze_ground_effect(wing, h=0.6, alpha_deg=4.0, phi_deg=2.0, height_ref="te")
print(res.CL, res.CDi, res.Cl, res.h_min_over_c)
```

`height_ref` sets the definition of `h`: `"ref"` (the moment reference point), `"min"` (the lowest edge point), `"qc"` or `"te"` (the root quarter-chord or trailing-edge point of the main surface). Contact with the ground raises `GroundStrikeError`. The lifting-line solvers raise `ValidityError` below `h_min/c = 1`. Both are subclasses of `ValueError`.

`vt.analyze` with `condition=vt.FlightCondition(..., h=...)` uses the same physics; there `h` is the height of the moment reference point.

## A grid of cases

`GroundEffectSweep` runs every combination of heights, angles of attack and bank angles with one chordwise mesh for all cases. Its cases run in parallel; `n_jobs` works like `n_jobs` in [Run a sweep in parallel](how_to_sweeps):

```python
from ventorum.ground_effect import GroundEffectSweep

sweep = GroundEffectSweep(wing, n_jobs="auto", solver="vlm", n_panels=40)
out = sweep.run_sweep(heights=[0.4, 0.6, 1.0], alphas_deg=[2.0, 4.0, 6.0], phis_deg=[0.0])
deriv = out.compute_stability_derivatives()
```

`run_sweep` accepts `V_inf`, `rho`, `ref_point`, `height_ref`, `compute_strike_limit` and `progress`. `compute_stability_derivatives()` gives the pitch and height aerodynamic centres and the Irodov margin `(x_alpha - x_h)/c` (stable if larger than 0; R. D. Irodov 1970). The criterion applies when the moment reference point is the centre of gravity: give it as `ref_point`. Plot helpers (`plot_height_sweep`, `plot_roll_effect`, `plot_pitch_stability`, `plot_ground_effect_matrix`) draw the grids.

## Background

For the solver limits in ground effect, see [Solvers](solvers.md). For the equations, see the [theory manual](../theory/index). For the verification study, see the ground-effect report under [Verification and validation](../vv/index).
