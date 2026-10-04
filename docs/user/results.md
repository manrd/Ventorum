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

:::{warning}
The uncertainty bands of the trust score are heuristic. They are not calibrated against experiments. Do not use them as error bars.
:::
