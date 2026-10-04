# Trust score and error bars

:::{warning}
**Experimental.** The trust score is a set of heuristic rules. Its uncertainty bands (`uncertainty_CL`, `uncertainty_CDi`, `uncertainty_LD`) are **not** calibrated against experimental data. Do not use them as error bars in a report.
:::

## Model

Every result carries a trust score from 0 to 1 and a rating (HIGH, MODERATE, LOW, UNRELIABLE), with the warnings and recommendations that caused it. The score starts at 1. Each rule adds a penalty when the case is near or outside a limit of the method in use. The rules come from the envelopes in the chapters of this manual. The score is a statement of how far the case is inside the envelope of the method; it is not an error estimate.

## Rules

`ventorum.core.trust.evaluate_aerodynamic_trust` applies these rules. The penalty values and the thresholds are in the code; they are design settings, not calibrated values.

| Rule (`factors` key) | Applies to | Reason | Chapter |
| --- | --- | --- | --- |
| `aspect_ratio` | lifting-line solvers | Lifting-line theory needs a high aspect ratio; chordwise effects grow at low aspect ratio | [Classical lifting line](classical_lifting_line.md), [Numerical lifting line](numerical_lifting_line.md) |
| `sweep` | lifting-line solvers; slender wings for the vortex lattice | The numerical lifting line is not grid convergent with sweep. Slender, highly swept wings have leading-edge vortex lift (Polhamus 1966) | [Numerical lifting line](numerical_lifting_line.md), [Vortex-lattice method](vortex_lattice.md) |
| `stall_proximity` | all | Linear sections have no stall. With tabulated polars, each polar gives its own maximum lift and stall angle | [Numerical lifting line](numerical_lifting_line.md) |
| `polar_range` | tabulated polars | A section angle outside the table uses the end values of the table | [Numerical lifting line](numerical_lifting_line.md) |
| `side_edge_vortex` | vortex lattice | A wing of low aspect ratio at high angle of attack gets side-edge vortex lift, which potential flow does not model (Lamar 1974) | [Vortex-lattice method](vortex_lattice.md) |
| `compressibility` | all | The methods are incompressible. The Mach number uses the speed of sound of the ISA atmosphere at the given air density | [Conventions](conventions.md) |
| `ground_proximity` | all in ground effect | Thickness and viscous effects grow at small $h_\text{min}/c$; vortex-lattice panels longer than the gap | [Ground effect](ground_effect.md) |
| `numerical` | all | No convergence; a coarse spanwise mesh; a spanwise circulation with many local extrema | all |
| `non_finite_result` | all | A coefficient that is not a finite number caps the rating at LOW | |

The score is $1 - \min(1, \sum \text{penalties})$, and the rating follows from fixed bands of the score. The uncertainty bands grow with $1 - \text{score}$, with fixed coefficients; this is why they are not error bars. A non-positive aspect ratio gives the full penalty for every solver. The notes of the solver (for example a lifting line near the ground) are added to the warnings.

## Planned error-bar layers

A calibrated error bar is planned for the force and moment coefficients. It is built in layers, so that each result gets the layers that the available data support, and each layer states its status:

1. **Numerical uncertainty.** The discretisation error from mesh refinement, with the Grid Convergence Index of P. J. Roache (1998). The mesh-convergence study (`ventorum.geometry.mesh_convergence.run_mesh_convergence_study`) already computes the observed order and the GCI on three meshes with a safety factor of 1.25. The refinement ratio is the ratio of the panel counts; for unequal ratios the observed order uses their mean, which is an approximation. Triplets that are not monotonic, or that have a ratio below a fixed limit, are skipped.
2. **Input uncertainty.** The effect of the uncertainty of the inputs (geometry, flight condition, section data) when the user gives it.
3. **Model-form uncertainty.** The difference between the model and reality, from validation comparisons that include the uncertainty of the experiment, with the method of ASME V&V 20 (2009) and Oberkampf and Roy (2010). Only graded reference data are used.
4. **Statistical coverage.** A stated probability of coverage, only where enough good reference cases exist. The calibration method is not chosen yet.

Until these layers exist, a result has only the trust score of this chapter.

## Implementation

- `ventorum.core.trust`: `evaluate_aerodynamic_trust`, `polar_limits`, `polar_stats`, `isa_speed_of_sound`, `spanwise_stats_batch`.
- `ventorum.core.datatypes.TrustScore`.
- `ventorum.geometry.mesh_convergence`: the GCI.

## Verification

The rules are tested in the test suite (for example the penalties at the limits of the polars and of the flow envelope). The score itself is not validated.

## References

- W. L. Oberkampf and C. J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, American Society of Mechanical Engineers, 2009.
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998.
- E. C. Polhamus, "A concept of the vortex lift of sharp-edge delta wings based on a leading-edge-suction analogy", NASA TN D-3767, 1966.
- J. E. Lamar, "Extension of leading-edge-suction analogy to wings with separated flow around side edges at subsonic speeds", NASA TR R-428, 1974.
