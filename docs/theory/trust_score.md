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

### Layer 1: numerical error (implemented)

Each of the six force and moment coefficients (CL, CD, CY, Cl, Cm, Cn) of a case gets a numerical error bar: the spanwise discretisation error from three mesh levels, with the Grid Convergence Index (GCI) of P. J. Roache (1998). `ventorum.core.error_bars.numerical_error_bars` solves the case on the three levels and builds one bar per coefficient.

Mesh levels. The user mesh is the fine level: `N1 = settings.n_panels`. The coarser levels are `N2 = round(N1 / sqrt(2))` and `N3 = round(N2 / sqrt(2))`. Only the spanwise panel count changes; the chordwise count, the spacing rule and every other setting stay as the user gave them. An automatic chordwise count is resolved one time on the fine level and then used for all three levels, because in ground effect it can depend on the mesh. A surface with its own `n_panels` is scaled in the same way. The representative cell size is `h_i = 1 / N_i` (spanwise only). The refinement ratios are `r21 = N1 / N2` and `r32 = N2 / N3` from the real integers, not the nominal sqrt(2).

Observed order. For non-constant ratios the observed order comes from the fixed-point iteration of Celik et al. (2008), with index 1 = fine: `eps21 = f2 - f1`, `eps32 = f3 - f2`, `s = sign(eps32 / eps21)`, `p = | ln|eps32 / eps21| + q(p) | / ln(r21)`, `q(p) = ln((r21^p - s) / (r32^p - s))`. The iteration starts from `p` without `q`, runs at most 100 times and stops when the change of `p` is below 1e-10.

Convergence states. With `R = eps21 / eps32`: `"roundoff"` (the changes are at round-off level; half-width `3 * max(|eps21|, |eps32|)`), `"monotonic"` (`0 < R < 1`; GCI with safety factor 1.25 for an observed order in [0.5, 4.0], else 3.0 with the order clipped to that range), `"oscillatory"` (`R < 0`; half-width `3.0 * 0.5 * (max - min)` of the three values), `"divergent"` (`R >= 1`; the change grows with refinement, so there is no interval and a note says so), and `"not_converged_order"` (monotonic values whose order iteration fails; half-width `3.0 * |eps21|`). The exact limit `R = 0` (fine equals medium) takes the monotonic path; the order iteration then finds no order, so the conservative rule gives a zero half-width. A bar with an interval is centred on the fine-level value `f1` (`[f1 - U, f1 + U]`); the Richardson-extrapolated value is stored as information only.

The oscillatory rule, the order limits 0.5 and 4.0, and the `"roundoff"` and `"not_converged_order"` rules are project decisions (conservative choices), not results of these sources.

Output structure. An `ErrorBar` holds the coefficient name, the fine-level value, the interval bounds (`None` without an interval), a status, a coverage (always `None` here) and one `LayerContribution` per layer (here only `"numerical"`, with the half-width, method, state, observed order, safety factor, extrapolated value, levels and values). The status is one of `"statistically_calibrated"`, `"model_form_estimated"`, `"numerical_only"` or `"outside_envelope"`; layer 1 sets only `"numerical_only"` (also for a divergent coefficient, whose interval is `None`). A coverage exists only on a calibrated bar.

Limits. The bar covers the spanwise discretisation error only; the chordwise error is not in the bar. The bar is about the mesh, not about the model: it is not an error against experiment.

The later layers are planned:

2. **Input uncertainty.** The effect of the uncertainty of the inputs (geometry, flight condition, section data) when the user gives it.
3. **Model-form uncertainty.** The difference between the model and reality, from validation comparisons that include the uncertainty of the experiment, with the method of ASME V&V 20 (2009) and Oberkampf and Roy (2010). Only graded reference data are used.
4. **Statistical coverage.** A stated probability of coverage, only where enough good reference cases exist. The calibration method is not chosen yet.

Until these layers exist, a result has only the trust score of this chapter.

## Implementation

- `ventorum.core.trust`: `evaluate_aerodynamic_trust`, `polar_limits`, `polar_stats`, `isa_speed_of_sound`, `spanwise_stats_batch`.
- `ventorum.core.datatypes.TrustScore`.
- `ventorum.geometry.mesh_convergence`: the GCI.
- `ventorum.core.error_bars`: layer 1 (numerical error bars from three mesh levels).

## Verification

The rules are tested in the test suite (for example the penalties at the limits of the polars and of the flow envelope). The score itself is not validated.

## References

- W. L. Oberkampf and C. J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, American Society of Mechanical Engineers, 2009.
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998.
- I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and P. E. Raad, "Procedure for estimation and reporting of uncertainty due to discretization in CFD applications", *Journal of Fluids Engineering* 130(7), 078001, 2008.
- E. C. Polhamus, "A concept of the vortex lift of sharp-edge delta wings based on a leading-edge-suction analogy", NASA TN D-3767, 1966.
- J. E. Lamar, "Extension of leading-edge-suction analogy to wings with separated flow around side edges at subsonic speeds", NASA TR R-428, 1974.
