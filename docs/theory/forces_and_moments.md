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
- **Chordwise legs (vortex lattice only).** The legs from the bound vortex to the trailing edge lie on the surface and are bound vorticity too. Their force, taken at the mid-point of each leg, is not zero in sideslip, on non-planar surfaces and near the ground. The leg force uses only the velocity tangent to the strip surface, $\mathbf W_t = \mathbf W - (\mathbf W\cdot\hat{\mathbf n})\,\hat{\mathbf n}$ with $\hat{\mathbf n}$ the strip normal. The surface is impermeable, so its bound vorticity carries no force from a normal velocity. The lattice enforces the zero normal velocity only at its control points; at the leg points the normal velocity is a leak of the discretisation, and its force would be a spurious force in the plane of the surface (for example a side force from an aileron on a flat wing that disagrees with the far field). The force of the tangential velocity stays: it gives the rolling moment due to sideslip of a flat wing, in agreement with the measurements of Goodman and Brewer (1948). The lifting line has no chordwise extent, so it uses the bound vortex only, as in Phillips and Snyder (2000), and it gives almost no rolling moment due to sideslip on a flat wing.
- **In-plane edge forces.** The lattice has no side-edge suction. In sideslip, momentum needs a small body side force from the side edges; without it the near-field drag is slightly lower than the Trefftz-plane drag at large sideslip. The reported induced drag is the Trefftz-plane value, so this affects only the diagnostic `CDi_nearfield`. A deflected aileron on a thin wing also gives a small side force through the side edges of the flap (proportional to the angle of attack times the deflection); the rotated-normal control model does not represent it, and the vortex lattice and the lifting lines give no side force from an antisymmetric deflection on a flat wing. DATCOM (section 6.2.3) neglects this side force for plain wings.

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

The trace is the trace of the induced drag. A deflected control turns the panel normals and does not move the trailing edge, so the trace of a flat wing with a deflected aileron is flat and `CY_farfield` is 0 there. In ground effect the normal wash includes the image wake, as in the induced drag. A symmetric case (the half evaluation of the Trefftz plane) gives 0 exactly for both values. The GPU pipelines compute both values on the host from the strip circulation and the normal wash of the GPU, with the same equations.

The Trefftz plane has no chordwise position. A side force that acts at a chordwise distance from the reference point gives a near-field yawing moment that `Cn_trefftz` does not contain, so in sideslip the two yawing moments can differ for this reason too.

**Trust warning.** The near-field `CY` (already relative to the free stream) and the near-field `Cn`, expressed in stability axes, are compared with the far-field values. The result gets a warning when $|\text{near}-\text{far}| > 0.05\,|\text{far}|$ and $|\text{near}-\text{far}|$ is larger than a floor: 0.001 for `CY` and 0.0002 for `Cn`. A probable cause of a difference is the side-edge forces that the transpiration model does not carry (see *In-plane edge forces* above). The warning does not change the score.

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
- *USAF Stability and Control DATCOM*, Air Force Flight Dynamics Laboratory, Wright-Patterson Air Force Base, section 6.2.3 (side force due to asymmetric deflection of control devices).
- W. F. Phillips and D. O. Snyder, "Modern adaptation of Prandtl's classic lifting-line theory", *Journal of Aircraft* 37(4), 2000, pp. 662-670.
