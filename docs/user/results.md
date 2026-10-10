# Results and trust

`analyze` returns a `SolverResult`.

| Attribute | Content |
| --- | --- |
| `totals` | Integrated coefficients: `CL`, `CDi` (Trefftz plane), `CDp`, `CD_total`, `e`, `CY`, `Cl`, `Cm`, `Cn`, and the trust score |
| `spanwise` | One `SpanwiseResult` per surface: circulation, section lift, induced drag, effective angle of attack |
| `converged` | False if the solver did not meet its tolerance |
| `details` | Solver data (lattice, circulation, wake direction, ground plane) for plots and checks |

## Trust score

Each result has a `TrustScore` with a rating (`HIGH`, `MODERATE`, `LOW`, `UNRELIABLE`), warnings and recommendations. It checks the aspect ratio, sweep, stall, Mach number, ground proximity and mesh.

- Stall: with tabulated section polars, the limits are the maximum Cl and the stall angle of each polar. With linear sections, the limits are a section Cl of 1.55 and an effective angle of 16 deg.
- Polar range: a section angle outside its polar table lowers the score. The end values of the table are used there.
- Low aspect ratio: the vortex lattice does not model the vortex lift of the side edges of a wing with aspect ratio below 2 at more than 10 deg (Polhamus 1966; Lamar 1974).
- Mach number: the speed of sound is the ISA value at the air density of the flight condition.

:::{warning}
The uncertainty bands of the trust score are heuristic. They are not calibrated against experiments. Do not use them as error bars.
:::

## Numerical error bars

`ventorum.numerical_error_bars` solves the case on three spanwise meshes (the user mesh is the fine level) and gives each of the six force and moment coefficients a discretisation error bar. This is layer 1 of the planned error bars (see [Trust score and error bars](../theory/trust_score)).

```python
bars = vt.numerical_error_bars(wing, settings=vt.SolverSettings(n_panels=40))
print(bars.summary())
```

Each `bars.bars[name]` holds the fine-mesh value, the interval bounds (`None` when the coefficient diverges with mesh refinement), the status (always `"numerical_only"` in layer 1), the coverage (always `None` until a calibrated layer exists) and the numerical layer contribution (state, observed order, safety factor, levels and values). The notes in `bars.notes` state the scope: spanwise discretisation error only, not an error against experiment. The moments use the selected axes (`axes="body"`, `"stability"` or `"wind"`). With proportional panels, a note names a surface that `min_panels` holds at the same panel count on two mesh levels. When the fine level has profile drag but a coarser level has no `CD_total`, the result has no `"CD"` bar and a note says why.
