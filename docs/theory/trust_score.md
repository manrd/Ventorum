# Trust score and error bars

:::{warning}
**Experimental.** The rating of the lifting line, the vortex lattice and the 3D panel method comes from a calibrated error model (see *Calibrated error model* below), against a converged inviscid lifting-surface or panel solution, not against experiments. The envelope rules that act as caps are heuristic rules and are not calibrated. The uncertainty bands (`uncertainty_CL`, `uncertainty_CDi`, `uncertainty_LD`) are **not** calibrated against experimental data. Do not use them as error bars in a report.
:::

## Model

Every result carries a trust score from 0 to 1 and a rating (HIGH, MODERATE, LOW, UNRELIABLE), with the warnings and recommendations that caused it. Two parts give the score:

1. The calibrated error model. For the lifting line, the vortex lattice and the 3D panel method, a fitted model predicts the error $E$ of the result from the geometry and the mesh. The score is a function of $k E$, a probable upper bound of the error (see below). The rating is the error band of $k E$: HIGH below 2 %, MODERATE below 5 %, LOW below 15 %, UNRELIABLE at 15 % and above.
2. The envelope rules. Each rule adds a penalty when the case is near or outside a limit of the method that the error model does not cover (stall, Mach number, side edges, polar range, convergence, non-finite results, saw-tooth, thickness and viscous effects near the ground, panel validity, post-stall branches). The rules come from the envelopes in the chapters of this manual. Their penalties give a cap score $1 - \min(1, \sum \text{penalties})$. These rules are caps: they can lower the rating, but they cannot raise it. They are not calibrated.

The score is the smaller of the two, $\min(\text{score}(kE),\ 1 - \min(1, \sum \text{penalties}))$, and the rating follows from fixed bands of the score (HIGH from 0.85, MODERATE from 0.65, LOW from 0.40). A result of another solver has only the envelope rules.

## Calibrated error model

`ventorum.core.trust.predicted_error` gives the predicted error $E$ (a fraction) and the factor $k$ of each method family. The coefficients are the module constants `ERROR_*` of `ventorum.core.trust`.

| Family | Predicted error $E$ | $k$ |
| --- | --- | --- |
| Lifting line (linear and nonlinear lifting line, Fourier) | $e^{-1.1}\,AR^{-0.67}\,e^{0.063\,\Lambda_{qc}[\text{deg}]}\,e^{-0.82(1-\lambda)}$ | 1.3 |
| Vortex lattice | $e^{-2.7}\,AR^{-1}\,n_c^{-2.4} + e^{-5.5}\,e^{3.4\,\Lambda_{qc}[\text{rad}]}\,n_s^{-0.68}$, plus in ground effect $e^{-6.9}\,(h/c)^{-2.4}\,n_c^{-2.2}$, plus for each deflected chordwise control $e^{-1.7}\,\delta\,n_f^{-1.3} + e^{0.042}\,\delta\,n_s^{-0.94}$ | 1.8 |
| 3D panel method | $e^{-0.46}\,n_\text{chord}^{-1} + e^{-2.7}\,n_\text{span}^{-0.66}$ | 1.5 |

Symbols: $AR$ is the aspect ratio of the main wing, $\Lambda_{qc}$ the quarter-chord sweep (for the lattice solvers the area-weighted mean of the bound-vortex segments, `ventorum.solvers.lattice_base.quarter_chord_sweep_deg`), $\lambda$ the tip chord over the root chord of the main surface (`lattice_tip_chord_ratio` or `surface_tip_chord_ratio`). For the vortex lattice, $n_c$ is the effective chordwise panel count (1 in the section mode), $n_s$ the smallest spanwise strip count per semi-span of the surfaces, $h/c$ the smallest ground clearance over the reference chord, $\delta$ the absolute deflection of a control [rad] and $n_f = n_c (1 - x_h/c)$ the chordwise panels aft of the hinge $x_h$. For the 3D panel method, $n_\text{chord}$ is the chordwise panel count per surface and $n_\text{span}$ the smallest spanwise panel count per semi-span of the surfaces. `ventorum.core.trust.lattice_error_inputs` reads these inputs from a lattice; the CPU and the GPU paths use the same function, so they give the same rating.

For the lifting line, a spanwise-count term and a ground term were fitted and came out as zero, so the model has no such terms. The control term of the vortex lattice is calibrated on 4 geometries only (rank correlation 0.03 on held-out controls); it stays in $E$. Two or more deflected controls add their terms.

Calibration. A study of 1157 known-answer cases fitted the model. The error measure is $E_3 = \max(e_{CL}, e_{CDi}/2, e_{Cm})$: the relative errors of CL and of the Trefftz-plane CDi, and $e_{Cm} = |\Delta C_m| / |C_L|$, the shift of the centre of pressure as a fraction of the reference chord. The truth was a three-level Richardson extrapolation of the vortex lattice (a lifting surface) and a separable Richardson extrapolation of the 3D panel method with the exact kernel. The factor $k$ is the 90th percentile of the ratio of the true error to $E$ on the training cases. On 336 cases of geometries not used in the fit, the band of $k E$ gave the right rating in 76.5 % of the cases and an optimistic rating in 1.5 % (HIGH: 133 cases below 2 %, 2 at 2 to 5 %, none above 5 %). The rank correlation of $E$ with the true error was 0.96 (lifting line 0.97, vortex lattice 0.92, vortex lattice in ground effect 0.87, 3D panel method 0.99). The former penalty rules rated 62 % of the same cases in the right band and were optimistic in 31 %.

Score. `ventorum.core.trust.error_score` is a continuous, decreasing, piecewise-linear function of $\ln(kE)$: 1.0 at and below $kE = 0.2\,\%$, 0.85 at 2 %, 0.65 at 5 % and 0.40 at 15 %. Above 15 % the slope of the last segment continues down to 0. The rating from the score and the rating from the band of $kE$ therefore agree, and the score is continuous in every input of the model (no threshold jumps). The factor `error_model` is $1 - \text{score}(kE)$.

Messages. When $kE$ is 2 % or more, the result gets a warning with the predicted error and its band, and a recommendation from the term that dominates $E$: for the lifting line the vortex lattice (for swept wings, or for a low aspect ratio); for the vortex lattice more chordwise panels, more spanwise panels, chordwise panels shorter than the ground gap, or more panels at the control; for the 3D panel method more chordwise or spanwise panels.

Limits. $kE$ is a probable bound of the error against a converged inviscid solution of the same method family, in attached flow. It does not include the model error of potential flow (viscous effects, separation, compressibility); the caps handle the limits of the envelope. The model holds for the kind of cases of the study.

## Rules

`ventorum.core.trust.evaluate_aerodynamic_trust` applies the error model and these rules.

| Rule (`factors` key) | Applies to | Reason | Chapter |
| --- | --- | --- | --- |
| `error_model` | lifting line, vortex lattice, 3D panel method | The calibrated error model (see above): $1 - \text{score}(kE)$. It replaced the former rules of the aspect ratio and of the sweep of the lifting line, the slender-wing rule of the vortex lattice, the penalty of the coarse spanwise mesh, the penalty of "chordwise panels longer than the ground gap", the chordwise-count rule of the 3D panel method and the penalty of its elongated-panel rule | this chapter |
| `aspect_ratio` | all | Cap: a non-positive or undefined aspect ratio gives the full penalty | |
| `stall_proximity` | all | Cap: linear sections have no stall. With tabulated polars, each polar gives its own maximum lift and stall angle | [Numerical lifting line](numerical_lifting_line.md) |
| `polar_range` | tabulated polars | Cap: a section angle outside the table uses the end values of the table | [Numerical lifting line](numerical_lifting_line.md) |
| `side_edge_vortex` | vortex lattice | Cap: a wing of low aspect ratio at high angle of attack gets side-edge vortex lift, which potential flow does not model (Lamar 1974) | [Vortex-lattice method](vortex_lattice.md) |
| `compressibility` | all | Cap: the methods are incompressible. The Mach number uses the speed of sound of the ISA atmosphere at the given air density. When a stated Mach number and V_inf/a differ by more than 0.05, the larger of the two is rated | [Conventions](conventions.md) |
| `ground_proximity` | all in ground effect | Cap: thickness and viscous effects grow at small $h_\text{min}/c$ (penalty 0.20 below 0.3, 0.45 below 0.1). Chordwise panels longer than the gap (vortex lattice, 3D panel method) give a warning without a penalty; the ground term of the error model rates them for the vortex lattice | [Ground effect](ground_effect.md) |
| `numerical` | all | Cap: no convergence; a saw-tooth in the spanwise circulation (see *Saw-tooth rule* below). A coarse spanwise mesh (fewer than 8 panels) gives a warning without a penalty | all |
| `mesh_adequacy` | 3D panel method | Cap: panel validity: skewed panels (the median panel), warped panels, a mesh that is not watertight, folded (non-convex) panels, a surface that intersects itself or its mirror image. Elongated panels give a warning without a penalty | [Ventorum-3DP](panel3d.md) |
| `mesh_resolution` | 3D panel method | Cap: the near-field and far-field drag differ | [Ventorum-3DP](panel3d.md) |
| `non_finite_result` | all | Cap: a coefficient that is not a finite number caps the rating at LOW | |
| `post_stall_branch` | viscous coupling by angle-of-attack correction (lattice solvers and 3D panel method) | Cap: a strip is past the maximum lift of its polar (section slope at or below zero), and the branch search finds more than one converged solution: their CL differ by more than `ventorum.viscous.alpha_correction.BRANCH_CL_DIFFERENCE`. The post-stall result is not unique. The penalty is `ventorum.core.trust.POST_STALL_BRANCH_PENALTY` | [Viscous alpha correction](viscous_alpha_correction.md) |

The penalty values and the thresholds of the caps are in the code; they are design settings, not calibrated values (the drag rule and the panel shape rules of the 3D panel method come from the mesh-error study below). The cap score is $1 - \min(1, \sum \text{penalties})$ over the caps; the score is the smaller of the cap score and the error-model score. The uncertainty bands grow with $1 - \text{score}$, with fixed coefficients; this is why they are not error bars. The notes of the solver (for example a lifting line near the ground) are added to the warnings.

Three checks add a warning and no penalty:

- The near-field side force and yawing moment are compared with the far-field values of the Trefftz plane (`CY_farfield`, `Cn_trefftz`). The rule and its reason are in [Forces and moments](forces_and_moments.md) (section *Far-field lateral diagnostics*).
- Small trailing-edge clearance: the lattice solvers warn when the smallest trailing-edge height above the ground is below 0.15 of the local chord. At small trailing-edge clearance the Trefftz induced drag CDi depends on the height of the far wake.
- Mean-line lattice conditioning: on a vortex lattice on a cambered section mean line, straight horseshoe legs would leave the curved mean surface; with narrow strips they passed close to control points of other strips, and the solve was near-singular. The legs now follow the mean surface (see [Vortex kernels](vortex_kernels.md)). The rule stays as a diagnostic: the vortex lattice warns when the strip lift changes by more than 0.3 between neighbouring strips of such a surface (strips at a deflected control are left out).

## Saw-tooth rule

The rule `numerical` adds the penalty 0.40 when the spanwise circulation of a surface has a saw-tooth: the strips alternate up and down. Such an odd-even oscillation is the mode of a near-singular system. An example is the former mean-line lattice with straight horseshoe legs (see the item *Mean-line lattice conditioning* above): on narrow strips its circulation oscillated at the wing tips, and the totals were wrong.

Detection. Let $\Gamma_j$ be the circulation of strip $j$ of one surface, in spanwise order, and $d_j = \Gamma_{j+1} - \Gamma_j$ the strip-to-strip step. A step with $|d_j| \le 10^{-6} \max_j |\Gamma_j|$ has no sign. Step $j$ is a sign change when $d_j d_{j-1} < 0$. The measure is the longest run of consecutive sign changes (`ventorum.core.trust._sign_change_run`). A step without a sign is not a sign change, so it stops the run. The rule acts when the run of one surface is 3 or more (`ventorum.core.trust._SAW_TOOTH_RUN`). The warning gives the run.

Rationale. A smooth load has physical extrema: the maximum at the root, and two extrema at each edge of a part-span control, of a gap between controls or of a strong change of twist. These extrema are isolated: each gives one sign change. A load feature only one strip wide (a narrow control or gap on a coarse mesh) gives two consecutive sign changes, up then down. Three or more consecutive sign changes need two neighbouring one-strip features of opposite sign, which is the saw-tooth. The number of physical extrema grows with the number of controls, but their run does not; when the mesh is refined the features get wider and the run gets shorter. The measure therefore does not depend on the number of controls and does not grow with the mesh for a converged solution. The former rule counted all local extrema of the circulation (penalty for more than six). Each part-span control adds about two extrema per side, so a smooth solution with a flap and an aileron got the penalty, and the rating changed with the mesh.

Control edges. A control edge is a spanwise strip boundary inside one surface where the control changes: a control starts or ends there, or two controls with different names, hinges or deflections meet. This includes a control at zero deflection (for example the base case of the control derivatives): such a control still changes the lattice at its span edges (the hinge nodes and the legs of the control strips). The root of a symmetric surface with a deflected antisymmetric control is also a control edge. A jump of the deflection gives a jump of the spanwise slope of the load with a logarithmic shape, and the change of the lattice at the edge gives a small local change of the load. When the mesh resolves the edge, these features can give a small peak, a dip one strip wide and a bump next to the edge. An example is a swept wing (25 deg, taper 0.5, aspect ratio 8, NACA 2412 mean line) with a flap and an aileron at zero deflection, 96 spanwise and 32 chordwise panels, at alpha 6 deg: the inner flap edge gives a run of 3 (test `test_zero_deflection_control_edge_is_not_flagged`). This is three consecutive sign changes of a correct result, not a numerical problem. The measure therefore does not count the steps near a control edge: a step $d_j$ whose two strips $j$ and $j + 1$ are both within two strips of a control edge (`ventorum.geometry.lattice.CONTROL_EDGE_STRIPS`) has no sign, as a step below the tolerance. These are the three steps across the edge. A saw-tooth that starts at a control edge and continues beyond these steps still gives a run of 3 or more. The lattice keeps the masked steps in `VortexLattice.control_edge_step` (`ventorum.geometry.lattice.control_edge_steps`), from the controls of the strips. The threshold (3) and the penalty (0.40) do not change.

The single path (`ventorum.core.trust.lattice_spanwise_stats`, used by the lattice solvers and by the angle-of-attack coupling on a lattice), the batch path (`spanwise_stats_batch`, with the entry `control_edge_step`) and the GPU loads kernel (with the mask as an input) compute the same measure; the tests check that the paths agree. A result without a lattice (for example the 3D panel method) gets the measure without the control-edge mask. A wing with two or more controls that are each about one strip wide and have alternating deflections can still give a run of 3 or more. Such a mesh does not resolve the controls; use more spanwise panels.

## 3D panel method

`ventorum.core.trust.panel3d_mesh_checks` gives the mesh rules of Ventorum-3DP. The thresholds are the `PANEL3D_*` constants of `ventorum.core.trust`. A mesh-error study of the method calibrated them.

The calibrated error model (see above) rates the chordwise and the spanwise panel counts of the 3D panel method. `evaluate_aerodynamic_trust` calls `panel3d_mesh_checks` with `error_model=True`: the chordwise-count rule is then not applied, and the elongated-panel rule gives its warning without a penalty. Without `error_model` the function applies all the rules below, as calibrated by the earlier study.

Error levels. A *warning* means a probable mesh error above 2 % in CL or above 4 % in the Trefftz-plane CDi. A *strong warning* means a probable mesh error above 5 % in CL or above 10 % in CDi. The CDi limits are twice the CL limits because the induced drag goes with the square of the lift: one circulation error gives about twice the relative error in CDi. As a cap, a drag warning limits the result to the MODERATE band and a strong warning to the LOW band.

Indicators of the discretization error (factor `mesh_resolution`, the penalty of the worse level):

- Near-field and far-field drag. In potential flow the pressure drag of the surface and the induced drag of the Trefftz plane are equal; on a mesh they differ. The rule compares `|CD_nearfield - CDi| / |CDi|` with a warning limit and a strong limit (`PANEL3D_DRAG_WARN_REL`, `PANEL3D_DRAG_STRONG_REL`). It does not act when the difference is below an absolute floor (`PANEL3D_DRAG_ABSOLUTE_FLOOR`): at zero lift both drags are zero and the relative difference has no meaning.
- Chordwise panel count (only without the error model). The chordwise count controls most of the mesh error of CL and CDi. The rule compares `n_chord` with two minimum counts (`PANEL3D_N_CHORD_WARN`, `PANEL3D_N_CHORD_STRONG`). The panel solver gives its chordwise count per surface (`SolverSettings.n_chord`, or the default count of the solver). With the count, the warning "chordwise panels longer than the gap" also acts for the panel method, with the largest strip chord as for the vortex lattice.

Panel shape and validity (factor `mesh_adequacy`, the penalties add):

- Panel aspect ratio and skew. The study found that the largest panel aspect ratio and the largest skew do not predict the mesh error. On a normal wing mesh the largest aspect ratio comes from the trailing-edge base panels, the tip caps and the short panels of the cosine spacing, and the largest skew comes from the triangles of the tip caps and of the control-surface edges. The rules therefore use the median panel (`PANEL3D_ASPECT_RATIO_PERCENTILE`, `PANEL3D_SKEW_PERCENTILE`). At a fixed chordwise count, a large median aspect ratio means few spanwise strips and a larger error, so the aspect-ratio limit (`PANEL3D_MAX_ASPECT_RATIO`) is a calibrated warning that adds to the drag rule (with the error model: a warning without a penalty). The skew limit (`PANEL3D_MAX_SKEW`) is not calibrated: the median skew of the study meshes stayed well below it.
- Warp. The limit `PANEL3D_MAX_WARP` is not calibrated: the study meshes had no warped panels.
- A mesh that is not watertight, and folded panels. A panel that is not convex (`n_nonconvex` of `check_mesh`) comes from a folded strip, and the result of such a mesh has no value. This rule has a large penalty (`PANEL3D_PENALTY_NONCONVEX`). A surface that intersects itself or its mirror image (`n_self_intersections` of `check_mesh`) gets the same penalty; the 3D panel solver refuses such a mesh before the solve, so the rule acts only on meshes that other code checks.
- A steep root. When the first panel row of a symmetric surface has a dihedral of 60 deg or more (`ventorum.panel3d.mesh.STEEP_ROOT_DIHEDRAL_DEG`), with the root thickness scaling the factor gets a note without a penalty: the result needs a spanwise convergence study. Without the scaling the pressure coefficient at the root is singular: the factor gets a warning and the cap penalty `PANEL3D_PENALTY_STEEP_ROOT_UNSCALED` (0.40, a choice, not calibrated).

Calibration method. Wings of different aspect ratio, taper, sweep, dihedral, section (symmetric and cambered), ground clearance and flap deflection were solved on a grid of chordwise and spanwise panel counts, with the default spacings. The reference value of each wing is the Richardson extrapolation of a chordwise refinement and of a spanwise refinement (`ventorum.panel3d.convergence.order_and_extrapolation`), added with a separable error model. The same procedure was checked on a sphere and an ellipsoid against the exact solutions of `ventorum.reference.analytic3d`. Each threshold is the value that best separates the meshes above the error limit of its level from the meshes below it (the largest balanced accuracy; on a flat optimum, a round value inside it).

Recalibration. Repeat the study with the new solver version: the same wing set, the same reference procedure and the same error limits of the two levels. Set each threshold again to the value with the largest balanced accuracy. Keep the error limits, so that the meaning of a warning does not change.

Limits. The indicators give a probable error, not a bound. The thresholds hold for the cases of the study: attached flow, sections with a closed trailing edge, and the default chordwise (cosine) and spanwise spacings. For an error estimate of one result, use `panel_estimate_mesh_error` (separate chordwise and spanwise Richardson extrapolations of three mesh levels each, added with a separable error model; see [Ventorum-3DP](ventorum_3dp.md), section *Mesh convergence and Richardson extrapolation*).

## Planned error-bar layers

A calibrated error bar is planned for the force and moment coefficients. It is built in layers, so that each result gets the layers that the available data support, and each layer states its status:

### Layer 1: numerical error (implemented)

Each of the six force and moment coefficients (CL, CD, CY, Cl, Cm, Cn) of a case gets a numerical error bar: the spanwise discretisation error from three mesh levels, with the Grid Convergence Index (GCI) of P. J. Roache (1998). `ventorum.core.error_bars.numerical_error_bars` solves the case on the three levels and builds one bar per coefficient.

Mesh levels. The user mesh is the fine level: `N1 = settings.n_panels`. The coarser levels are `N2 = round(N1 / sqrt(2))` and `N3 = round(N2 / sqrt(2))`. Only the spanwise panel count changes; the chordwise count, the spacing rule and every other setting stay as the user gave them. An automatic chordwise count is resolved one time on the fine level and then used for all three levels, because in ground effect it can depend on the mesh. A surface with its own `n_panels` is scaled by `N_i / N1` (rounded, at least 4); the GCI uses the global ratios `r21` and `r32` also for it. The representative cell size is `h_i = 1 / N_i` (spanwise only). The refinement ratios are `r21 = N1 / N2` and `r32 = N2 / N3` from the real integers, not the nominal sqrt(2).

Observed order. For non-constant ratios the observed order comes from the fixed-point iteration of Celik et al. (2008), with index 1 = fine: `eps21 = f2 - f1`, `eps32 = f3 - f2`, `s = sign(eps32 / eps21)`, `p = | ln|eps32 / eps21| + q(p) | / ln(r21)`, `q(p) = ln((r21^p - s) / (r32^p - s))`. The iteration starts from `p` without `q`, runs at most 100 times and stops when the change of `p` is below 1e-10.

Convergence states. With `R = eps21 / eps32`: `"roundoff"` (the changes are at round-off level; half-width `3 * max(|eps21|, |eps32|)`), `"monotonic"` (`0 < R < 1`; GCI with safety factor 1.25 for an observed order in [0.5, 4.0], else 3.0 with the order clipped to that range), `"oscillatory"` (`R < 0`; half-width `3.0 * 0.5 * (max - min)` of the three values), `"divergent"` (`R >= 1`, or `eps32 = 0` with `eps21 != 0` whatever the sign; the change grows with refinement, so there is no interval and a note says so), and `"not_converged_order"` (monotonic values, `0 <= R < 1`, whose order iteration fails; half-width `3.0 * |eps21|`). The exact limit `R = 0` (fine equals medium, coarse different) takes the monotonic path; the order iteration then finds no order. There, `3.0 * |eps21|` would give a zero half-width, so the half-width is `3.0 * max(|eps21|, |eps32|)`. A value that is not finite gives the state `"divergent"`, no interval and a note that says so. A bar with an interval is centred on the fine-level value `f1` (`[f1 - U, f1 + U]`); the Richardson-extrapolated value is stored as information only.

The oscillatory rule, the order limits 0.5 and 4.0, and the `"roundoff"` and `"not_converged_order"` rules are project decisions (conservative choices), not results of these sources.

Output structure. An `ErrorBar` holds the coefficient name, the fine-level value, the interval bounds (`None` without an interval), a status, a coverage (always `None` here) and one `LayerContribution` per layer (here only `"numerical"`, with the half-width, method, state, observed order, safety factor, extrapolated value, levels and values). The status is one of `"statistically_calibrated"`, `"model_form_estimated"`, `"numerical_only"`, `"input_only"`, `"numerical_and_input"` or `"outside_envelope"`; layer 1 sets only `"numerical_only"` (also for a divergent coefficient, whose interval is `None`). A coverage exists only on a calibrated bar.

Limits. The bar covers the spanwise discretisation error only; the chordwise error is not in the bar. The bar is about the mesh, not about the model: it is not an error against experiment.

### Layer 2: input uncertainty (implemented)

Each of the six force and moment coefficients of a case gets an input-uncertainty bar when the user gives the standard uncertainties (one standard deviation) of the flow inputs and of the linear section properties. The method is first-order (linear) propagation of uncertainty, the law of propagation of uncertainty of the GUM (JCGM 100:2008, section 5.1, equation 10 for independent inputs and equation 13 of section 5.2 with correlations): `u_c(f)^2 = J U J^T` with `J` the gradient of the coefficient with respect to the inputs and `U` the covariance matrix of the inputs. The expanded half-width is `k u_c` with the coverage factor `k` (default 2, GUM section 6). The combination of the numerical layer and the input layer by root sum of squares of independent contributions follows Oberkampf and Roy (2010).

Inputs. `ventorum.core.input_uncertainty.input_error_bars` takes a dict `uncertainties` with optional keys `alpha` [rad], `beta` [rad], `phi` [rad], `V_inf` [m/s], `rho` [kg/m^3], `h` [m] (ground effect only), and `sections` (a dict of surface name to dict with optional `a0` [1/rad], `alpha_L0` [rad], `Cd0` [-], `Cm0` [-]). A section uncertainty of a surface applies to every strip of that surface and is fully correlated along the span: the gradient is summed over the strips of the surface before the propagation. An optional `correlations` dict gives correlation coefficients of pairs of flow inputs in [-1, 1]; the default is independent inputs.

Gradients. The gradients come from `ventorum.gradients` (discrete adjoint) when the configuration is in its scope. Outside its scope (nonlinear lifting line, tabulated airfoils, deflected controls, body rates, Fourier solver) the flow-input gradients come from central differences of the production solve with the steps alpha 0.5 deg, beta 1 deg, phi 0.5 deg, `V_inf` 1 %, `rho` 1 % and `h` 5 % of the smallest ground clearance; section inputs are then refused. The result states the method per input (`"adjoint"` or `"central differences"`). The adjoint runs on the device of the production solve, with the precision of `ventorum.gpu.get_precision()` (default float32, also on the CPU; the CPU kernels evaluate the O(n^2) partials in float64). The central differences use the production solve itself.

Output structure. The layer-2 `LayerContribution` has `layer="input"`, `half_width = k u_c`, a method that names the GUM law and the gradient method, and an empty `values` field. `ErrorBar.coverage` stays None (the bar is not calibrated); a note states that `k = 2` assumes a normal distribution of the output. Each bar also gives the share of each input in `u_c^2` (they sum to 1 when `u_c > 0`; with correlations a share can be negative), so the user sees which input dominates. The layer-2 result has status `"input_only"`. `ventorum.core.error_bars.combine_error_bars` combines a layer-1 result and a layer-2 result of the same case into one `ErrorBarResult` with status `"numerical_and_input"`: the half-width is `sqrt(h_num^2 + h_input^2)` and the interval is centred on the fine-level value of layer 1. A bar whose layer 1 has no interval keeps no interval, and a note says why.

Limits. The bar is first order: it is exact only for small uncertainties of a near-linear response. The coverage factor is not calibrated. Geometry (node) uncertainty is not covered.

The later layers are planned:

3. **Model-form uncertainty.** The difference between the model and reality, from validation comparisons that include the uncertainty of the experiment, with the method of ASME V&V 20 (2009) and Oberkampf and Roy (2010). Only graded reference data are used.
4. **Statistical coverage.** A stated probability of coverage, only where enough good reference cases exist. The calibration method is not chosen yet.

Until these layers exist, a result has only the trust score of this chapter plus layers 1 and 2.

## Implementation

- `ventorum.core.trust`: `evaluate_aerodynamic_trust`, `predicted_error`, `lifting_line_error`, `vlm_error_terms`, `panel3d_error_terms`, `error_score`, `error_band`, `lattice_error_inputs`, `polar_limits`, `polar_stats`, `isa_speed_of_sound`, `spanwise_stats_batch`, `lattice_spanwise_stats`.
- `ventorum.core.datatypes.TrustScore`.
- `ventorum.geometry.mesh_convergence`: the GCI.
- `ventorum.core.error_bars`: layer 1 (numerical error bars from three mesh levels) and the combination.
- `ventorum.core.input_uncertainty`: layer 2 (input uncertainty by linear propagation).

## Verification

The rules are tested in the test suite (for example the penalties at the limits of the polars and of the flow envelope). `tests/test_trust_error_model.py` checks the three formulas against hand-computed values, the score function, the continuity of the score, that every cap still lowers the rating, and that the GPU and the CPU paths give the same rating. The error model is calibrated against converged inviscid solutions; the score is not validated against experiments.

## References

- JCGM 100:2008 (GUM), *Evaluation of measurement data: Guide to the expression of uncertainty in measurement*, Joint Committee for Guides in Metrology, 2008.
- W. L. Oberkampf and C. J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.
- ASME V&V 20-2009, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, American Society of Mechanical Engineers, 2009.
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998.
- I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and P. E. Raad, "Procedure for estimation and reporting of uncertainty due to discretization in CFD applications", *Journal of Fluids Engineering* 130(7), 078001, 2008.
- E. C. Polhamus, "A concept of the vortex lift of sharp-edge delta wings based on a leading-edge-suction analogy", NASA TN D-3767, 1966.
- J. E. Lamar, "Extension of leading-edge-suction analogy to wings with separated flow around the side edges at subsonic speeds", NASA TR R-428, 1974.
