# Ground effect

## Model

Near flat ground, Ventorum adds the mirror image of every horseshoe vortex in the ground plane, with the opposite circulation. The real and image vortices together give zero velocity normal to the plane, so the plane is a solid wall. This is the method of images (C. Wieselsberger 1921; Katz and Plotkin 2001). The free stream is parallel to the ground (level flight), so the wake, which leaves the trailing edge along the free stream, is parallel to the ground too. From one flight state (`height_derivatives`) or from a sweep over height, angle of attack and bank angle, Ventorum gives the height and bank derivatives, the pitch and height aerodynamic centres, and the Irodov criterion of longitudinal static stability in ground effect.

## Ground plane

The aircraft keeps its geometry axes; the ground plane moves. With the bank angle $\phi$ [rad] (positive puts the right wing nearer to the ground), the unit normal of the ground, which points to the aircraft, is

```{math}
\hat{\mathbf k} = \hat{\mathbf L}\cos\phi + (\hat{\mathbf u}\times\hat{\mathbf L})\sin\phi ,
```

that is, the lift direction turned about the free stream by $\phi$. $\hat{\mathbf k}$ is normal to $\hat{\mathbf u}$ for every $\alpha$, $\beta$ and $\phi$, so the free stream stays parallel to the ground. The plane is the set of points $\mathbf x$ with $\mathbf x\cdot\hat{\mathbf k} = o$, and the height of a point is $h(\mathbf x) = \mathbf x\cdot\hat{\mathbf k} - o$ [m]. The offset $o$ [m] follows from the given height $h$ in one of these conventions (`height_ref`):

| `height_ref` | The point at height $h$ |
| --- | --- |
| `ref` (default) | the moment reference point |
| `min` | the lowest point of the lattice used for the ground plane |
| `qc`, `te` | the root quarter-chord or trailing-edge point of the main surface |

A geometry that touches or crosses the plane is refused (`GroundStrikeError`). The bank angle of the first contact is found by bisection, with the reference point at a constant height.

## Image method

The image of a point $\mathbf p$ and of a direction $\mathbf v$ are

```{math}
\mathbf p' = \mathbf p - 2\,h(\mathbf p)\,\hat{\mathbf k}, \qquad
\mathbf v' = \mathbf v - 2\,(\mathbf v\cdot\hat{\mathbf k})\,\hat{\mathbf k} .
```

Each horseshoe gets an image horseshoe with the reflected nodes, the reflected wake direction and the circulation $-\Gamma$.

Derivation of the wall condition: let $\mathcal R$ be the reflection in the plane. The velocity of a vortex segment from $\mathbf a$ to $\mathbf b$ at $\mathbf x$, written $\mathbf v_\Gamma(\mathbf x; \mathbf a, \mathbf b)$, contains one cross product of position differences, and its other factors depend only on lengths and dot products. A reflection changes the sign of a cross product relative to the reflected vectors, $\mathcal R\mathbf a\times\mathcal R\mathbf b = -\mathcal R(\mathbf a\times\mathbf b)$, so $\mathbf v_\Gamma(\mathcal R\mathbf x; \mathcal R\mathbf a, \mathcal R\mathbf b) = -\mathcal R\,\mathbf v_\Gamma(\mathbf x; \mathbf a, \mathbf b)$. For a point $\mathbf x$ on the plane, $\mathcal R\mathbf x = \mathbf x$. The image with circulation $-\Gamma$ then gives $\mathcal R\,\mathbf v_\Gamma(\mathbf x)$, and the sum is $\mathbf v + \mathcal R\mathbf v$, which has no component along $\hat{\mathbf k}$. The core term depends on distances only, so the regularised kernels keep this property.

The image vortices enter every part of the solution: the influence matrix (equation {eq}`eq-vlm` of [Vortex-lattice method](vortex_lattice.md)), the near-field forces and the Trefftz plane, where the image trailing vortices are projected with the real ones.

When a section mean line is given, the image points are the reflections of the real mean-surface points. The clearance check includes the bound endpoints, control points and trailing-edge points. The camber offset changes the real distance to the ground as well as the boundary normal. C. Coulliette and A. Plotkin (1996) use the real camber line for a two-dimensional discrete-vortex model. A chord-plane model with a uniformly turned normal omits this height and the chordwise camber loading. Near the ground this is an additional approximation, even when its free-air section lift agrees with the given polar. S. E. Widnall and T. M. Barrows (1970) describe the small-clearance lifting-surface limit.

A surface without all its section mean lines uses the turned-normal model over the whole surface. In ground effect the result states that its chordwise load and camber height are approximate. The note is included in the trust assessment.

In chordwise control mode, the flap points rotate about the hinge on the mean surface. The image vortices and the Trefftz plane use the deflected bound endpoints and trailing edges. The clearance check includes the deflected control points and both copies of a separated strip edge. A flap that touches or crosses the plane raises `GroundStrikeError`. A zero-deflection control keeps the same geometry and integrated coefficients. Cambered hinge moments use the mean-line hinge at zero deflection too. The single-panel vortex lattice and the lifting lines retain the thin-airfoil section control model.

For chordwise controls, the automatic panel-count probe starts with the default chordwise count and includes the deflected flap points. The ground-effect entry point then uses the selected count for its clearance, tip heights and bank-limit geometry. It refuses a strike before it calls the aerodynamic solver. The root `qc` and `te` height points rotate about the effective root-strip hinge when they lie aft of it. On a symmetric surface with an antisymmetric control, these height points belong to the original half of the main surface. Single-panel and lifting-line controls retain their section model.

The vortex lattice selects the chordwise panel count from the gap (see [Vortex-lattice method](vortex_lattice.md)). A sweep in ground effect uses one count for all its cases (the count of the case with the smallest gap), so that differences between the cases contain no mesh change. The automatic count is an integer, so $C_L(h)$ and the other results of separate solves jump at the heights where the count changes. For your own finite differences (for example $\partial C_L/\partial h$ from separate solves), set `SolverSettings.n_chord` so that all cases use one mesh.

## Height and bank derivatives

`GroundEffectSweep` solves a grid of heights $h$, angles of attack $\alpha$ and bank angles $\phi$. `GroundEffectSweepResult.compute_stability_derivatives` gives:

- $C_{L\alpha}$, $C_{m\alpha}$ [1/rad] at each height, from a least-squares line over the angles of the grid with $|\alpha| \le$ 8 deg at the bank angle of the grid nearest to 0 (with fewer than two angles in this window, all angles of the grid), and the aerodynamic centre $x_{ac} = -c\,C_{m\alpha}/C_{L\alpha}$ [m] aft of the reference point. These are at constant height of the `height_ref` point, without the chain-rule correction below; thus $x_{ac}$ can differ from $x_\alpha$.
- $C_{Lh} = \partial C_L/\partial(h/c)$ and $C_{mh} = \partial C_m/\partial(h/c)$ at constant $\alpha$, by finite differences on the height grid (`numpy.gradient`: second-order central differences inside, first-order one-sided differences at the end points).
- $C_{l\phi}$, $C_{n\phi}$, $C_{Y\phi}$ [1/rad] from a least-squares line over the bank angles with $|\phi| \le$ 5 deg (with fewer than two bank angles in this window, all bank angles of the grid), at the angle of attack of the grid nearest to 4 deg. $C_l$ and $C_n$ are the stability-axis rolling and yawing moments about the moment reference point ($C_l > 0$ right wing down, $C_n > 0$ nose right); $C_Y$ is the side force, positive to the right. The bank angle $\phi$ turns the ground about the free-stream direction; the aircraft keeps its body axes, and the `height_ref` point keeps the height of the grid. A bank-restoring rolling moment has $C_{l\phi} < 0$.

### Irodov criterion

Near the ground the lift and the pitching moment depend on two variables, $\alpha$ and $h$. Each has its own aerodynamic centre, measured aft of the moment reference point [m]. $x_\alpha$ uses `numpy.gradient` along the $\alpha$ grid (not the line fit above):

```{math}
x_\alpha = -c\, \frac{\partial C_m/\partial\alpha}{\partial C_L/\partial\alpha}, \qquad
x_h = -c\, \frac{\partial C_m/\partial(h/c)}{\partial C_L/\partial(h/c)} .
```

R. D. Irodov (1970) showed that longitudinal static stability in ground effect at constant speed needs the height aerodynamic centre upstream of the pitch aerodynamic centre (see also Rozhdestvensky 2006, who writes the criterion with the x axis upstream):

```{math}
x_h < x_\alpha, \qquad \text{irodov\_margin} = \frac{x_\alpha - x_h}{c} > 0 .
```

The criterion applies to the derivatives about the centre of gravity: the pitch derivative is a rotation about that point, and the margin changes when the point moves. Give the centre of gravity as the moment reference point. Both derivatives are about the moment reference point, with the height of that point held fixed, for every `height_ref`. A change of height at constant $\alpha$ moves every point by the same distance, so the height derivative does not depend on the convention. The $\alpha$ derivative does: with `height_ref` other than `ref`, the columns of the grid are at constant height of another point. The chain rule corrects it to constant height $h_\text{ref}$ of the reference point:

```{math}
\left.\frac{\partial F}{\partial\alpha}\right|_{h_\text{ref}} = \left.\frac{\partial F}{\partial\alpha}\right|_{h} - \frac{\partial F}{\partial (h/c)}\, \left.\frac{\partial (h_\text{ref}/c)}{\partial\alpha}\right|_{h}, \qquad F = C_L, C_m .
```

### Derivatives of one flight state

`height_derivatives` gives the static derivatives of one state in ground effect, without a grid. The variables are the angle of attack $\alpha$, the sideslip angle $\beta$, the bank angle $\phi$ [rad] and the height $h/c$ of the moment reference point (the pivot). For each variable $v$ the function solves two cases with $v \pm \Delta v$ and the other three variables held fixed, and takes the central difference

```{math}
\frac{\partial C}{\partial v} \approx \frac{C(v + \Delta v) - C(v - \Delta v)}{2\,\Delta v}, \qquad C = C_L, C_D, C_Y, C_l, C_m, C_n .
```

The pivot keeps its height in every case. A change of $\alpha$ or $\phi$ turns the ground about the pivot (the ground plane above, with the offset $o$ from the height of the pivot), and a change of $h$ moves the ground along $\hat{\mathbf k}$ at constant attitude. The derivatives are thus the partial derivatives of the Irodov criterion, about the pivot, and the criterion follows from the same matrix. The height step is a fraction of the smallest clearance $h_\text{min}$ of the base state (default $0.05\,h_\text{min}$): the ground effect varies with the gap, so the relative truncation error of the central difference is about $(\Delta h/h_\text{min})^2$ for a load that varies as $1/h_\text{min}$ (the central difference of $1/h$ is $-1/(h^2 - \Delta h^2)$). The angle steps are 0.5 deg ($\alpha$, $\phi$) and 1 deg ($\beta$). All nine cases are solved as one batch on one lattice; with an automatic chordwise count, all cases use the largest automatic count of the nine cases (normally the count of the case with the smallest gap).

The bank and height columns give the coupling of the height with the roll motion: at $\phi \neq 0$ the two tips have different gaps, so $C_{lh}$, $C_{nh}$ and $C_{Yh}$ are not zero, and $C_{l\phi}$ is the ground-effect roll stiffness at constant height of the pivot. At $\phi = 0$ and $\beta = 0$ the flow is symmetric: the lateral coefficients do not change with $\alpha$ and $h$, and the longitudinal coefficients are even functions of $\beta$ and $\phi$.

The moment rows in the stability or wind axes turn the dimensional moments with the axes of the base state, held fixed: $C' = L^{-1} R\, L\, C$ with $L = \mathrm{diag}(b, c, b)$, because $C_l$ and $C_n$ use $b$ and $C_m$ uses $c$. The Irodov centres use the body-axis $C_m$, which is equal to the stability-axis $C_m$.

## Assumptions

- Flat, rigid ground; free stream parallel to the ground (no climb or descent); steady flow.
- Thin, inviscid wing; the wake is straight and parallel to the ground (no wake deformation near the ground).
- The ground has no boundary layer (a moving ground, as in flight).
- The wake follows the free stream. In ground effect the option `wake_alignment="body"` is replaced by the free-stream direction, with a warning; the agent tools refuse it.

## Envelope

- At small gaps a real thick section can lose lift (suction under the wing) or choke the flow; the thin-wing model cannot show this. The trust score lowers the rating below fixed values of $h_\text{min}/c$ (see [Trust score](trust_score.md)).
- The lifting-line solvers are refused near the ground (see [Numerical lifting line](numerical_lifting_line.md)). The classical lifting line has no ground effect.
- The body rates $p$, $q$, $r$ are quasi-steady, at constant height of the reference point (see [Rotation rates](rotation_rates.md)). A heave rate (a vertical velocity, which tilts the free stream against the ground) and unsteady effects are not modelled; `height_derivatives` has no heave-rate column.
- The finite differences of the sweep derivatives depend on the grid of the sweep; the end points of the height grid are first order. `height_derivatives` uses central differences with fixed steps at one state.
- The Irodov margin depends on the moment reference point; it is meaningful only about the centre of gravity.

## Implementation

- `ventorum.aero.system`: `ground_normal`, `GroundPlane`, `make_ground_plane`, `build_sources` (image horseshoes), `check_ground_clearance`.
- `ventorum.aero.loads.trefftz_induced_drag_batch` (image wake in the Trefftz plane).
- `ventorum.ground_effect.analyze_ground_effect`; `ventorum.ground_effect.solver.find_bank_strike_limit` and `clearance_info`. The result also gives the force coefficient along the ground normal (`vertical_force_coefficient`), the body-axis force coefficients, and $C_l$ and $C_n$ in the stability axes.
- `ventorum.ground_effect.GroundEffectSweep`, `GroundEffectSweepResult.compute_stability_derivatives`.
- `ventorum.height_derivatives` (`ventorum/solvers/height_derivatives.py`): the derivative matrix and the Irodov terms of one state.
- `ventorum.reference.ground_effect`: the independent reference of the induced drag of an elliptic loading near the ground (see below).

## Reference: elliptic loading with an image wake

`ventorum.reference.ground_effect` gives an independent reference for the induced drag in ground effect. It does not use the production kernels, the lattice or the Trefftz-plane code. C. Wieselsberger (1921) replaced the ground by the mirror image of the wing and calculated the change of the induced drag of an elliptic loading from the vertical velocity of the image wing. This module evaluates the same model exactly in the Trefftz plane.

The wing of span $b = 2a$ [m] has the circulation $\Gamma(y) = \Gamma_0\sqrt{1 - (y/a)^2}$. Its wake is a flat sheet at the height $h$ [m], parallel to the ground. The image sheet has the circulation $-\Gamma$ at the distance $H = 2h$ below the sheet. In the Trefftz plane, with $z = y + \mathrm i\,z_v$, the complex velocity of the elliptic sheet alone is

```{math}
W(z) = u - \mathrm i\,v = \mathrm i\,w_0\left(1 - \frac{z}{\sqrt{z^2 - a^2}}\right), \qquad w_0 = \frac{\Gamma_0}{b} ,
```

with the branch of the root that is $z$ at infinity and has its cut on the sheet. $W$ goes to zero at infinity, the vertical velocity on the sheet is the uniform downwash $-w_0$, and the potential jump across the sheet is $2w_0\sqrt{a^2 - y^2} = \Gamma(y)$. Thus $W$ is the flow of the sheet (the flow of a flat plate that moves normal to itself). The image gives the upwash $v_\text{img}(y) = w_0\,(1 - \mathrm{Re}[\zeta/\sqrt{\zeta^2 - a^2}])$ on the real sheet, with $\zeta = y + \mathrm i H$. The Trefftz-plane drag $D_i = \tfrac{\rho}{2}\int \Gamma\,(w_0 - v_\text{img})\,dy$ at the same circulation (the same lift) gives the ratio

```{math}
R = \frac{k_\text{GE}}{k_\text{free}} = \frac{2}{\pi a^2}\int_{-a}^{a}\sqrt{a^2 - y^2}\;\mathrm{Re}\,\frac{\zeta}{\sqrt{\zeta^2 - a^2}}\,dy , \qquad k = \frac{C_{D_i}}{C_L^2} .
```

$R = 1 - \sigma$, where $\sigma$ is the influence coefficient of Wieselsberger (his equation 3). With $y = a\cos\theta$ the integrand is a smooth periodic function of $\theta$ for $h > 0$, and the midpoint rule converges exponentially. A second evaluation uses the Fourier transform of the loading ($\pi a J_1(ka)/|k|$) and of the interaction kernel of two parallel sheets ($\pi |k| e^{-H|k|}$):

```{math}
R = 1 - 2\int_0^\infty \frac{J_1(x)^2}{x}\,e^{-\beta x}\,dx, \qquad \beta = \frac{H}{a} = \frac{4h}{b} .
```

The two limits are $1 - R \sim (b/h)^2/32$ for $h \to \infty$ (the far field of the image is a vortex pair) and $R \sim (2\beta/\pi)(\ln(8/\beta) - 1)$ for $\beta \to 0$. These two asymptotes were derived for this reference. The tests check both evaluations against each other, the convergence of the quadrature and both asymptotes. Wieselsberger also gives Prandtl's approximation $\sigma = (1 - 0.66\,g/b)/(1.05 + 3.7\,g/b)$ with $g = 2h$, for $g/b$ from 1/15 to 1/2 (his equation 4); the tests compare it with the exact value in that range.

The reference holds for an elliptic loading. A wing near the ground changes its loading, so the comparison with the vortex lattice uses only heights at which the loading stays near elliptic. `tests/test_analytical_verification.py` compares $k_\text{GE}/k_\text{free}$ of planar elliptic wings (aspect ratios 6 and 10) with $R$ at $h/b$ = 0.5, 0.3 and 0.2. The wing is parallel to the ground ($\alpha = 0$, with the zero-lift angle of the section), so the lattice and the wake are both at the height $h$. A wing at an angle of attack has its trailing edge and its wake below the reference point, which decreases the ratio.

## Verification

- [V7](../verification_report.md): zero normal velocity on a tilted and banked plane; the free-air limit at large height; chordwise convergence of the vortex lattice; lifting line against the vortex lattice.
- `tests/test_analytical_verification.py`: the induced-drag ratio of elliptic wings against the image Trefftz-plane theory of an elliptic loading (see above). `tests/test_reference_ground_effect.py` checks the reference itself.
- Tests compare an explicit mirror wing with the image method, and the Irodov terms with direct differences.
- `tests/test_height_derivatives.py`: the derivatives of one state against single solves and against the sweep; the chain rule of the $\alpha$ derivative between two pivots; symmetry; the free-air limit; the axes transform with the reference lengths; the GPU path.

## References

- C. Wieselsberger, "Über den Flügelwiderstand in der Nähe des Bodens", *Zeitschrift für Flugtechnik und Motorluftschiffahrt* 12(10), 1921; English translation: "Wing resistance near the ground", NACA TM 77, 1922.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001 (method of images).
- C. Coulliette and A. Plotkin, ["Aerofoil ground effect revisited"](https://www.cambridge.org/core/journals/aeronautical-journal/article/abs/aerofoil-ground-effectrevisited/561E5C4DD13ED87435EF7B438859EE20), *The Aeronautical Journal* 100, 1996.
- S. E. Widnall and T. M. Barrows, ["An analytic solution for two- and three-dimensional wings in ground effect"](https://doi.org/10.1017/S0022112070000915), *Journal of Fluid Mechanics* 41, 1970, pp. 769-792.
- K. V. Rozhdestvensky, *Aerodynamics of a Lifting System in Extreme Ground Effect*, Springer, 2000.
- K. V. Rozhdestvensky, "Wing-in-ground effect vehicles", *Progress in Aerospace Sciences* 42, 2006, pp. 211-283.
- R. D. Irodov, "Criteria of longitudinal stability of ekranoplan", *Uchenye Zapiski TsAGI* 1(4), 1970, pp. 63-72 (in Russian); English machine translation "Criteria of the longitudinal stability of the ekranoplan", Foreign Technology Division, 1974, DTIC AD-A002918.
