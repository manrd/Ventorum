# Results and trust

`analyze` returns a `SolverResult`.

| Attribute | Content |
| --- | --- |
| `totals` | Integrated coefficients: `CL`, `CDi` (Trefftz plane), `CDp`, `CD_total`, `e`, `CY`, `Cl`, `Cm`, `Cn`, the far-field diagnostics `CY_farfield` (wind axes) and `Cn_trefftz` (stability axes; both None for the Fourier solver), and the trust score |
| `spanwise` | One `SpanwiseResult` per surface: circulation, section lift, induced drag, effective angle of attack |
| `converged` | False if the solver did not meet its tolerance |
| `details` | Solver data (lattice, circulation, wake direction, ground plane, dispatch) for plots and checks |

## Dispatch record

Every result of a lattice solver records how it was computed and why in `result.details["dispatch"]`.

| Field | Type | Description |
| --- | --- | --- |
| `device` | str | `"cpu"` or `"gpu"` |
| `precision` | str | `"float64"` on CPU; `"float32"` or `"float64"` on GPU |
| `backends` | dict | Kernel name (`"tensor"`, `"influence"`, `"induced"`, `"trefftz"`) mapped to the backend used |
| `threads` | int | Kernel threads of the thread that solved the case |
| `workers` | int | Cases that ran at the same time: the size of the worker pool that solved the case, or 1 when the case or the batch ran alone |
| `plan_source` | str | Source of the plan that ran: `"user"` (the `n_jobs` of the user, or forced threads), `"profile"` (the tuned values of the machine profile), `"default"` (the built-in values) or `"nested"` (a worker thread whose pool plan is not known) |
| `profile` | str or None | Machine profile fingerprint string, or None when no profile was active |
| `reason` | str | Plain sentence stating the dispatch decision and any fallback reason |

In `result.to_dict()` and in agent tools, the dispatch record is included only when `detail_level="full"`. Existing top-level keys `details["device"]` and `details["precision"]` on GPU results remain unchanged.


## Trust score

Each result has a `TrustScore` with a rating (`HIGH`, `MODERATE`, `LOW`, `UNRELIABLE`), warnings and recommendations. It checks the aspect ratio, sweep, stall, Mach number, ground proximity and mesh.

- Stall: with tabulated section polars, the limits are the maximum Cl and the stall angle of each polar. With linear sections, the limits are a section Cl of 1.55 and an effective angle of 16 deg.
- Polar range: a section angle outside its polar table lowers the score. The end values of the table are used there.
- Low aspect ratio: the vortex lattice does not model the vortex lift of the side edges of a wing with aspect ratio below 2 at more than 10 deg, for a sweep of at most 55 deg (Polhamus 1966; Lamar 1974). This check applies to the vortex lattice only.
- Mach number: the speed of sound is the ISA value at the air density of the flight condition.
- Near field and far field: a warning when the near-field `CY` (wind axes) or `Cn` (stability axes) differs from `CY_farfield` or `Cn_trefftz` by more than 5 % and more than a floor (0.001 for `CY`, 0.0002 for `Cn`). `Cn` is compared only when each available side force, near and far, is at or below the `CY` floor. The warning does not change the score. See [Forces and moments](../theory/forces_and_moments.md).

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

## Input-uncertainty bars

`ventorum.input_error_bars` propagates the standard uncertainties (one standard deviation) of the flow inputs and of the linear section properties through the gradients of the six coefficients (first-order propagation, `u_c(f)^2 = J U J^T`). The expanded half-width is `k u_c` with the coverage factor `k` (default 2). This is layer 2 of the planned error bars (see [Trust score and error bars](../theory/trust_score)).

```python
bars = vt.input_error_bars(
    wing,
    settings=vt.SolverSettings(solver_type="vlm", n_panels=24, n_chord=2),
    uncertainties={"alpha": 0.002, "sections": {"Wing": {"a0": 0.05}}},
)
print(bars.summary())
```

Each `bars.bars[name]` holds the production value, the interval `[value - k u_c, value + k u_c]`, the status `"input_only"`, the coverage (always `None`, not calibrated) and the input layer contribution. `bars.bars[name].input_shares` gives the share of each input in `u_c^2` (they sum to 1 when `u_c > 0`; with correlations a share can be negative), and `gradient_methods` states the gradient method per input (`"adjoint"` or `"central differences"`). Outside the adjoint scope (nonlinear lifting line, tabulated airfoils, deflected controls, body rates, Fourier solver) the flow inputs use central differences and section inputs are refused. `ventorum.combine_error_bars(numerical, inputs)` combines a layer-1 result and a layer-2 result of the same case: the half-width is `sqrt(h_num^2 + h_input^2)`, centred on the fine-level value of layer 1, with status `"numerical_and_input"`.

## Mesh convergence study

`ventorum.run_mesh_convergence_study` solves the case on a set of spanwise panel counts and spacing schemes and compares each level with a fine reference mesh. It gives the minimal mesh (the smallest accurate and safe level) and the recommended mesh (a level whose error is within `robustness_margin * tolerance_pct`, at least 12 panels per semi-span; a robust `"auto"` level is preferred).

```python
study = vt.run_mesh_convergence_study(wing, tolerance_pct=0.5, panel_counts=(10, 20, 40))
print(study.summary())
```

The selection uses a deterministic cost: the total panel count, then the number of panels solved. When two levels have the same cost, the level that is evaluated first is kept (order of `spacing_schemes`, then of `panel_counts`). The measured solve times are not used, so the same case gives the same recommendation on every run and on every machine. The solve times stay in the result as information. The `efficiency_score` of a level is `(N_ref / N)^2 / (1 + e)`, with `N` the number of panels solved, `N_ref` the number of panels solved on the reference mesh and `e` the mean of the CL and CDi errors in percent.
