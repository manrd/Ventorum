# Forces and moments

## Model

The lattice solvers (vortex lattice and numerical lifting line) give the circulation of each horseshoe. The forces and moments come from four parts: the Kutta-Joukowski force on the bound vorticity (near field), the induced drag in the Trefftz plane far downstream (far field), the profile drag of the sections, and the pitching moment of the sections. The induced drag of the Trefftz plane is the reference value `CDi`; the drag component of the near-field force is kept as the diagnostic `CDi_nearfield`. The classical lifting line has its own closed forms (see [Classical lifting line](classical_lifting_line.md)).

## Near-field forces

The Kutta-Joukowski law gives the force [N] on a straight vortex segment $\mathbf l$ [m] with circulation $\Gamma$ in the local velocity $\mathbf W$ (Katz and Plotkin 2001, chapter 12; Drela 2014):

```{math}
\mathbf F = \rho\, \Gamma\, \mathbf W \times \mathbf l, \qquad \mathbf W = \mathbf V_\infty + \mathbf v .
```

$\mathbf v$ is the velocity that all horseshoes (and their images) induce at the force point.

- **Bound vortex.** The force point is on the bound vortex at the same span fraction as the control point.
- **Chordwise legs (vortex lattice only).** The legs from the bound vortex to the trailing edge lie on the surface and are bound vorticity too. Their force, taken at the mid-point of each leg, is not zero in sideslip, on non-planar surfaces and near the ground. Each leg lies on a strip edge, and the velocity that acts on it depends on the edge (`ventorum.aero.loads.leg_force_rules`):
  - *Shared edge* (the leg coincides with a leg of the neighbour strip, of the mirror strip at the root, or of a strip of a joined surface, in one core group; the strips meet without a fold). Only the velocity tangent to the strip surface acts, $\mathbf W_t = \mathbf W - (\mathbf W\cdot\hat{\mathbf n})\,\hat{\mathbf n}$ with $\hat{\mathbf n}$ the strip normal. The surface is impermeable, so its bound vorticity carries no force from a normal velocity. The lattice enforces the zero normal velocity only at its control points; at the leg points the normal velocity is a leak of the discretisation, and its force would be a spurious force in the plane of the surface (for example a side force from an aileron on a flat wing that disagrees with the far field). The force of the tangential velocity stays: it gives the rolling moment due to sideslip of a flat wing, in agreement with the measurements of Goodman and Brewer (1948).
  - *Kink edge* (a shared edge where the strips meet with a fold about the edge line: the root of a wing with dihedral, a dihedral break, a fin on a tail tip, a winglet). The velocity is projected on the edge line, so the legs of the edge carry no force and the coincident legs carry no net force. With the strip normal of each leg, the normal velocity at the kink, which is tangential to the other strip, does not cancel between the coincident legs. Their force is then spurious and grows with the number of spanwise panels: at the root of a dihedral wing in sideslip it gave a false side force, and at a fin on a tail tip a `CY_beta` that did not converge with the panels (`tests/test_farfield_lateral.py`, `tests/test_junction_legs.py`). The kinks come from the geometry definition, not from the strip normals: a change of dihedral between the sections of one surface, the root of a symmetric surface with dihedral, or the angle between two joined surfaces. Each strip has the spanwise direction of its untwisted reference line, without node displacements, in the plane normal to the untwisted chord (`SurfaceSlice.strip_span_def`); the fold of an edge is the angle between the strips that meet there. Twist, taper and sweep do not change these directions, so they never make a kink (the piecewise-flat strips of a swept, twisted wing fold slightly about every edge, by an amount that falls with the mesh size, and a rule on the strip normals made these edges kinks on fine meshes). For a fold below `LEG_KINK_BLEND_RAD` the leg keeps the tangential rule with its force scaled by one minus the fold divided by `LEG_KINK_BLEND_RAD`, so the loads are continuous in the dihedral angle, also through zero. The rules are a property of the mesh: the gradients keep them fixed.
  - *Free edge* (no other leg of the core group lies on the edge: a tip, or the root of a surface that is not mirrored or joined). The full velocity acts. Its in-plane component gives the side-edge force of the edge, the discrete form of the force on the chordwise vortex of a free edge in the Kutta-Joukowski law (Katz and Plotkin 2001, chapter 12). With it, the near-field side force of a dihedral wing in sideslip agrees with the Trefftz-plane value (`tests/test_farfield_lateral.py`).

  Two legs of different strips share an edge when they lie on each other by the measure and the tolerance of the join rule of the core groups (see [Vortex filaments](vortex_kernels.md)): the end points of each leg are near the line of the other, and the offset of the end points along that line is small. The strips of one surface share their nodes, so legs of one surface use the lower limit of the join tolerance only. A leg that matches no other leg is free only on a boundary edge of its surface that is not joined. On an interior edge (for example at the end of a control, where the chordwise panels of the two strips differ) it keeps the tangential rule; on a joined edge it keeps the tangential rule too, and the result gets a note that the model does not resolve this junction. The GPU loads use the same rules, and the gradient model (`ventorum.sensitivity.reference`) and its adjoint follow them. The lifting line has no chordwise extent, so it uses the bound vortex only, as in Phillips and Snyder (2000), and it gives almost no rolling moment due to sideslip on a flat wing.
- **Legs with hinge nodes.** In chordwise control mode the legs of a panel ahead of the hinge have a node at the hinge point of the strip edge (see [Control surfaces](control_surfaces.md)). Each leg is then two straight segments, and each segment gets its own force at its mid-point (`ventorum.aero.loads.leg_segments`). Both segments get the rule of the whole leg, found without the node. The rules of a lattice with a chordwise control come from the lattice before the flap turn (same topology) and stay fixed for every deflection; only the normal of the shared-edge rule turns with the edge node on the flap legs (the aft legs and the aft segments), so the velocity tangent to the deflected surface acts there. In the joins at the side edges of a control the warped surface tilts the bound vortices, which gives a small side force, as a dihedral does. In the matching of legs, the copies of the shared aft segment of one strip are not a fold.
- **In-plane edge forces.** The free edges carry their side-edge force through the leg rule above. In chordwise control mode, flap points and trailing edges turn about the hinge axes of the edge nodes, and a join of 0.2 local chords seals each control side edge, so neighbouring strips share their edge points. External free edges retain their full velocity. The flat rectangular wing test gives a finite positive near-field side force for right flap down at positive incidence; the force changes sign with the deflection. The Trefftz plane uses the moved trailing edges; with the joins the trace of an aileron has a continuous step at its inboard end, which gives a positive far-field side force too. DATCOM (section 6.2.3) neglects this side force for plain wings. The reported induced drag is the Trefftz-plane value; the in-plane edge forces affect only the diagnostic `CDi_nearfield`.

The total near-field force $\mathbf F_\text{near}$ is the sum over all panels. The lift and side-force coefficients use the total force with the profile drag:

```{math}
C_L = \frac{\mathbf F_\text{total}\cdot\hat{\mathbf L}}{q_\infty S_\text{ref}}, \qquad
C_Y = \frac{\mathbf F_\text{total}\cdot\hat{\mathbf Y}}{q_\infty S_\text{ref}}, \qquad
C_{Di,\text{near}} = \frac{\mathbf F_\text{near}\cdot\hat{\mathbf u}}{q_\infty S_\text{ref}} .
```

The near-field induced drag needs the leading-edge suction, which a lattice of discrete vortices resolves poorly on swept wings. It is a diagnostic only.

## Trefftz-plane induced drag

Far downstream the wake is a sheet of trailing vortices in a plane normal to the wake direction $\hat{\mathbf d}$ (the Trefftz plane; E. Trefftz 1921). In the code each strip sends two trailing vortices from its trailing-edge nodes, with $+\Gamma_s$ at the right node and $-\Gamma_s$ at the left node, where $\Gamma_s$ is the sum of the circulations of the chordwise panels of strip $s$. The nodes are projected onto the plane. Each projected vortex induces the two-dimensional velocity

```{math}
\mathbf w(\mathbf q) = \frac{\Gamma}{2\pi}\, \frac{\hat{\mathbf d}\times(\mathbf q - \mathbf p)}{|\mathbf q - \mathbf p|^2 + r_c^2},
```

with the core $r_c$ a fraction of the strip width. Between a vortex and a point of different core groups, $r_c^2$ becomes $r_c^2 + r_{c,\text{target}}^2$, the same cross-surface core as in [Vortex filaments](vortex_kernels.md). For each strip, the normal wash $w_{n,s}$ is evaluated at the point of the projected strip segment at the span fraction of the control point, along the in-plane normal of the segment. The induced drag [N] is (Katz and Plotkin 2001, chapter 8; Drela 2014, chapter 5)

```{math}
D_i = -\frac{\rho}{2} \sum_s \Gamma_s\, w_{n,s}\, \ell_s, \qquad C_{Di} = \frac{D_i}{q_\infty S_\text{ref}},
```

where $\ell_s$ [m] is the length of the projected strip segment. The section induced drag is $C_{d,i} = -\Gamma_s w_{n,s} \ell_s / (V_\infty^2 c_s w_s)$, with the strip chord $c_s$ and width $w_s$. The span efficiency is $e = C_L^2/(\pi\,AR\,C_{Di})$. In ground effect the image vortices are added in the same plane (see [Ground effect](ground_effect.md)).

## Far-field lateral diagnostics

Each lattice result also gives two far-field lateral values, next to the near-field `CY` and `Cn`. They are diagnostics: the near-field values stay the result values.

- **Side force of the trailing vortex sheet, `CY_farfield`.** Each projected strip segment $\mathrm d\mathbf s_s$ (from the left to the right trailing-edge node, on the Trefftz plane) carries the circulation $\Gamma_s$. Its Kutta-Joukowski force in the far field is $\mathbf F_s = \rho V_\infty \Gamma_s\, (\hat{\mathbf u}\times \mathrm d\mathbf s_s)$ [N], with $\hat{\mathbf u}$ the free-stream direction (Katz and Plotkin 2001, chapter 8; Drela 2014, chapter 5). `CY_farfield` is the sum of the components along the wind-axis side direction $\hat{\mathbf y}_w = \hat{\mathbf L}\times\hat{\mathbf u}$ (positive to the right), divided by $q_\infty S_\text{ref}$.
- **Yawing moment of the induced drag, `Cn_trefftz`.** The induced drag of each strip, $D'_s = -\tfrac{\rho}{2}\Gamma_s w_{n,s}\ell_s$ [N] (the distribution of the induced drag above, along $\hat{\mathbf u}$), acts at the evaluation point $\mathbf q_s$ of its segment. With the arm $\mathbf a_s$ from the reference point to $\mathbf q_s$, both projected on the Trefftz plane, the moment is $\mathbf M_T = \sum_s \mathbf a_s \times D'_s\hat{\mathbf u}$. In stability axes, $C_{n,T} = -\mathbf M_T\cdot\hat{\mathbf z}_s/(q_\infty S b)$ with $\hat{\mathbf z}_s = \hat{\mathbf L}$ (up in the geometry axes). With the lateral arm $y_s = \mathbf a_s\cdot\hat{\mathbf y}_w$ this is $C_{n,T} = \sum_s D'_s y_s/(q_\infty S b)$: more drag on the right wing gives $C_n > 0$ (nose right), the sign of `Cn`.

The trace is the trace of the induced drag. A deflected control in section mode turns only the strip properties, so the trace of a flat wing with a deflected aileron stays flat and `CY_farfield` is 0 there. In chordwise mode the control moves the trailing edge, and the trace of an antisymmetric aileron has a step (smoothed by the join) at the inboard end of each aileron: `CY_farfield` is then not 0. In ground effect the normal wash includes the image wake, as in the induced drag. A symmetric case (the half evaluation of the Trefftz plane) gives 0 exactly for both values. The GPU pipelines compute both values on the host from the strip circulation and the normal wash of the GPU, with the same equations.

The Trefftz plane has no chordwise position. A side force that acts at a chordwise distance from the reference point gives a near-field yawing moment that `Cn_trefftz` does not contain, so in sideslip the two yawing moments can differ for this reason too.

**Trust warning.** The near-field `CY` (already relative to the free stream) and the near-field `Cn`, expressed in stability axes, are compared with the far-field values. The result gets a warning when $|\text{near}-\text{far}| > 0.05\,|\text{far}|$ and $|\text{near}-\text{far}|$ is larger than a floor: 0.001 for `CY` and 0.0002 for `Cn`. The `Cn` pair is compared only when each available (finite) side force, near and far, is at or below the `CY` floor, because of the missing chordwise arm described above. Probable causes of a difference: a junction of surfaces that the mesh does not resolve (for example a fin on a tail tip, a winglet, or joined edges with chordwise grids or strip widths that do not match), or a deflected control in section mode, whose trailing-edge displacement the Trefftz-plane trace does not include (see *In-plane edge forces* above). The warning does not change the score. The stability tool of the agent interface (`stability_derivatives`) reports the warnings of all its cases (the reference case and the four perturbed cases), the score of the weakest case, and a warning when `CY_beta` of the near field and `CY_beta` of the Trefftz plane do not agree by the same rule (5 %, and a difference of the side-force change over the sideslip step above the `CY` floor).

## Section quantities

For each strip, the section lift coefficient is $c_l = 2\Gamma_s/(V_\infty c_s)$ (lower case, to keep it apart from the rolling moment $C_l$). The effective angle of attack $\alpha_\text{eff}$ comes from the nonlinear solver, or else from the linear section: $\alpha_\text{eff} = \alpha_{L0} + c_l/a_0$. With $\hat{\mathbf n}$ the geometric unit normal of the strip and $\hat{\mathbf a}$ its unit chord direction, the geometric angle of the strip is $\alpha_\text{geo} = \operatorname{atan2}(\hat{\mathbf u}\cdot\hat{\mathbf n}, \hat{\mathbf u}\cdot\hat{\mathbf a})$ and the induced angle is $\alpha_i = \alpha_\text{geo} - \alpha_\text{eff}$.

- **Profile drag.** $D_p = q_\infty c_s w_s C_d(\alpha_\text{eff})$ [N] along the free stream, applied at the quarter-chord point of the strip ($C_{d0}$ for a linear airfoil). It is added to the total force and to the moments.
- **Section pitching moment.** $q_\infty c_s^2 w_s C_m(\alpha_\text{eff})$ [N m] about the quarter chord ($C_{m0}$ for a linear airfoil), added as a pure moment about the axis $\hat{\mathbf n}_\text{lift}\times\hat{\mathbf u}$, with $\hat{\mathbf n}_\text{lift}$ the unit vector along $\hat{\mathbf u}\times\mathbf{dl}$.

## Moments

The moment about the reference point $\mathbf r_\text{ref}$ is

```{math}
\mathbf M = \sum (\mathbf r - \mathbf r_\text{ref}) \times \mathbf F + \sum \mathbf M_\text{section},
```

over all force points (bound vortices, chordwise legs, profile drag). The coefficients follow from the signs in [Conventions](conventions.md): $C_l = -M_x/(q_\infty S b)$, $C_m = M_y/(q_\infty S c)$, $C_n = -M_z/(q_\infty S b)$.

## Static stability derivatives

The agent tool `stability_derivatives` gives the static derivatives about the centre of gravity $x_{cg}$ [m] by central differences: $\pm$0.5 deg in $\alpha$ and $\pm$1 deg in $\beta$, on one mesh for all cases. The neutral point and the static margin are

```{math}
x_{np} = x_{cg} - c_\text{ref}\, \frac{C_{m\alpha}}{C_{L\alpha}}, \qquad
SM = \frac{x_{np} - x_{cg}}{c_\text{ref}},
```

with $C_{m\alpha}$ in the body axes. This is a small-angle form: it takes the lift as the force normal to the body x axis, and it ignores the drag and a vertical offset of the centre of gravity. Thus the computed neutral point moves a little with $x_{cg}$. It follows from $C_m(x) = C_m(x_{cg}) + C_L (x - x_{cg})/c_\text{ref}$ for a moment taken about a point a distance $x - x_{cg}$ aft (x aft, $C_m$ nose up), and from the definition of the neutral point, $\partial C_m / \partial\alpha = 0$ about it.

## Full aerodynamic derivative matrix

The function `derivative_matrix` combines the static, control, rate, and height derivatives into one matrix:

```{math}
\mathbf{M} = \begin{bmatrix}
\frac{\partial C_L}{\partial \alpha} & \frac{\partial C_L}{\partial \beta} & \cdots & \frac{\partial C_L}{\partial \delta_k} & \cdots & \frac{\partial C_L}{\partial \hat{p}} & \frac{\partial C_L}{\partial \hat{q}} & \frac{\partial C_L}{\partial \hat{r}} \\
\frac{\partial C_D}{\partial \alpha} & \frac{\partial C_D}{\partial \beta} & \cdots & \frac{\partial C_D}{\partial \delta_k} & \cdots & \frac{\partial C_D}{\partial \hat{p}} & \frac{\partial C_D}{\partial \hat{q}} & \frac{\partial C_D}{\partial \hat{r}} \\
\frac{\partial C_Y}{\partial \alpha} & \frac{\partial C_Y}{\partial \beta} & \cdots & \frac{\partial C_Y}{\partial \delta_k} & \cdots & \frac{\partial C_Y}{\partial \hat{p}} & \frac{\partial C_Y}{\partial \hat{q}} & \frac{\partial C_Y}{\partial \hat{r}} \\
\frac{\partial C_l}{\partial \alpha} & \frac{\partial C_l}{\partial \beta} & \cdots & \frac{\partial C_l}{\partial \delta_k} & \cdots & \frac{\partial C_l}{\partial \hat{p}} & \frac{\partial C_l}{\partial \hat{q}} & \frac{\partial C_l}{\partial \hat{r}} \\
\frac{\partial C_m}{\partial \alpha} & \frac{\partial C_m}{\partial \beta} & \cdots & \frac{\partial C_m}{\partial \delta_k} & \cdots & \frac{\partial C_m}{\partial \hat{p}} & \frac{\partial C_m}{\partial \hat{q}} & \frac{\partial C_m}{\partial \hat{r}} \\
\frac{\partial C_n}{\partial \alpha} & \frac{\partial C_n}{\partial \beta} & \cdots & \frac{\partial C_n}{\partial \delta_k} & \cdots & \frac{\partial C_n}{\partial \hat{p}} & \frac{\partial C_n}{\partial \hat{q}} & \frac{\partial C_n}{\partial \hat{r}}
\end{bmatrix}
```

The matrix has six fixed rows: $C_L, C_D, C_Y, C_l, C_m, C_n$.

The columns follow group order:
1. Static variables: $\alpha$ and $\beta$. With the height group (ground effect only), $\phi$ and $h/c_\text{ref}$ follow.
2. Control deflections: $\delta_k$ for each control name.
3. Nondimensional rates: $\hat{p} = p b / (2 V)$, $\hat{q} = q c / (2 V)$, $\hat{r} = r b / (2 V)$.

The units of each column are:
- $1/\text{rad}$ for angle and control variables ($\alpha, \beta, \phi, \delta$).
- Dimensionless ($-$) for height $h/c_\text{ref}$ and body rates $\hat{p}, \hat{q}, \hat{r}$.

Forces $C_L, C_D, C_Y$ are relative to the free stream in body, stability, and wind axes. The axes of the base condition stay fixed while the variables change. $\mathbf{R}$ is the rotation from the body axes to the target axes, and $\mathbf{L} = \operatorname{diag}(b_\text{ref}, c_\text{ref}, b_\text{ref})$.

In the static, height and control columns the variables do not depend on the axes. Only the moment rows turn, with the reference lengths:

```{math}
\begin{bmatrix} C_l \\ C_m \\ C_n \end{bmatrix}_\text{target} = \mathbf{L}^{-1} \mathbf{R} \mathbf{L} \begin{bmatrix} C_l \\ C_m \\ C_n \end{bmatrix}_\text{body}.
```

In the rate columns the rates are components of the angular velocity, so the rates turn together with the moments, as in `rate_derivatives`. With $\mathbf{F}$ the $3 \times 3$ force block (rows $C_L, C_D, C_Y$) and $\mathbf{J}$ the $3 \times 3$ moment block (rows $C_l, C_m, C_n$) of the rate columns:

```{math}
\mathbf{F}_\text{target} = \mathbf{F}_\text{body} \, \mathbf{L} \mathbf{R}^{T} \mathbf{L}^{-1}, \qquad
\mathbf{J}_\text{target} = \mathbf{L}^{-1} \mathbf{R} \mathbf{L} \, \mathbf{J}_\text{body} \, \mathbf{L} \mathbf{R}^{T} \mathbf{L}^{-1}.
```

Thus every column equals the result of its source function (`static_derivatives`, `height_derivatives`, `control_derivatives` and `rate_derivatives`) in the same axes.

## Assumptions

- The wake is straight and parallel to $\hat{\mathbf d}$ up to the Trefftz plane (no roll-up). The Trefftz plane is normal to $\hat{\mathbf d}$.
- The sections act at their quarter chord; profile drag and section moments come from two-dimensional data at $\alpha_\text{eff}$ (strip theory).

## Envelope

- `CDi_nearfield` is not reliable on swept wings; use `CDi`.
- With `wake_alignment="body"`, the Trefftz plane is normal to the body x-axis, not to the free stream; at large angles the two drag definitions differ.
- Profile drag from section data does not include interference or three-dimensional viscous effects.

## Implementation

- `ventorum.aero.loads.compute_loads` and `compute_loads_batch` (near field, section data, moments).
- `ventorum.aero.loads.trefftz_induced_drag` and `trefftz_induced_drag_batch`; the kernel `ventorum.aero.vortex.trefftz_normalwash`.
- `ventorum.aero.loads.trefftz_lateral_batch` (`CY_farfield`, `Cn_trefftz`); `ventorum.core.trust.near_far_lateral_warnings` (the near/far warning).
- `ventorum.agent.tools.stability_derivatives`.

## Verification

- [V1](../verification_report.md), [V2](../verification_report.md): $C_{L\alpha}$ and $e$ against closed forms.
- [V8](../verification_report.md): section pitching moment at zero lift.
- [V9](../verification_report.md): signs of $C_{l\beta}$, $C_{Y\beta}$ and $C_{n\beta}$.

## References

- M. W. Kutta, "Auftriebskräfte in strömenden Flüssigkeiten", *Illustrierte Aeronautische Mitteilungen* 6, 1902, p. 133; N. E. Joukowski, "Sur les tourbillons adjoints", *Travaux de la Section Physique de la Société Impériale des Amis des Sciences Naturelles* 13(2), 1906. (Cited as given in Katz and Plotkin 2001.)
- E. Trefftz, "Prandtlsche Tragflächen- und Propeller-Theorie", *Zeitschrift für Angewandte Mathematik und Mechanik* 1, 1921, pp. 206-218.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapters 8 and 12.
- M. Drela, *Flight Vehicle Aerodynamics*, MIT Press, 2014, chapter 5.
- A. Goodman and J. D. Brewer, "Investigation at low speeds of the effect of aspect ratio and sweep on static and yawing stability derivatives of untapered wings", NACA TN 1669, 1948.
- R. D. Finck, *USAF Stability and Control DATCOM*, McDonnell Douglas Corporation for the Air Force Flight Dynamics Laboratory, Wright-Patterson Air Force Base, revised 1978 (AFWAL-TR-83-3048), section 6.2.3 (side force due to asymmetric deflection of control devices).
- W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic lifting-line theory", *Journal of Aircraft* 37(4), 2000, pp. 662-670.
