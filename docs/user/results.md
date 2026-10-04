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
