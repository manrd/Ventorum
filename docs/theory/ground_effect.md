# Ground effect

## Model

Near flat ground, Ventorum adds the mirror image of every horseshoe vortex in the ground plane, with the opposite circulation. The real and image vortices together give zero velocity normal to the plane, so the plane is a solid wall. This is the method of images (C. Wieselsberger 1921; Katz and Plotkin 2001). The free stream is parallel to the ground (level flight), so the wake, which leaves the trailing edge along the free stream, is parallel to the ground too. From a sweep over height, angle of attack and bank angle, Ventorum gives the height and bank derivatives, the pitch and height aerodynamic centres, and the Irodov criterion of longitudinal static stability in ground effect.

## Ground plane

The aircraft keeps its geometry axes; the ground plane moves. With the bank angle $\phi$ [rad] (positive puts the right wing nearer to the ground), the unit normal of the ground, which points to the aircraft, is

```{math}
\hat{\mathbf k} = \hat{\mathbf L}\cos\phi + (\hat{\mathbf u}\times\hat{\mathbf L})\sin\phi ,
```

that is, the lift direction turned about the free stream by $\phi$. $\hat{\mathbf k}$ is normal to $\hat{\mathbf u}$ for every $\alpha$, $\beta$ and $\phi$, so the free stream stays parallel to the ground. The plane is the set of points $\mathbf x$ with $\mathbf x\cdot\hat{\mathbf k} = o$, and the height of a point is $h(\mathbf x) = \mathbf x\cdot\hat{\mathbf k} - o$ [m]. The offset $o$ [m] follows from the given height $h$ in one of these conventions (`height_ref`):

| `height_ref` | The point at height $h$ |
| --- | --- |
| `ref` (default) | the moment reference point |
| `min` | the lowest leading- or trailing-edge node |
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

The vortex lattice selects the chordwise panel count from the gap (see [Vortex-lattice method](vortex_lattice.md)). A sweep in ground effect uses one count for all its cases (the count of the case with the smallest gap), so that differences between the cases contain no mesh change.

## Height and bank derivatives

`GroundEffectSweep` solves a grid of heights $h$, angles of attack $\alpha$ and bank angles $\phi$. `GroundEffectSweepResult.compute_stability_derivatives` gives:

- $C_{L\alpha}$, $C_{m\alpha}$ [1/rad] at each height, from a least-squares line over the angles of the grid with $|\alpha| \le$ 8 deg at $\phi = 0$, and the aerodynamic centre $x_{ac} = -c\,C_{m\alpha}/C_{L\alpha}$ [m] aft of the reference point. These are at constant height of the `height_ref` point, without the chain-rule correction below; thus $x_{ac}$ can differ from $x_\alpha$.
- $C_{Lh} = \partial C_L/\partial(h/c)$ and $C_{mh} = \partial C_m/\partial(h/c)$ at constant $\alpha$, by finite differences on the height grid (`numpy.gradient`: second-order central differences inside, first-order one-sided differences at the end points).
- $C_{l\phi}$, $C_{n\phi}$, $C_{Y\phi}$ [1/rad] from a least-squares line over the bank angles with $|\phi| \le$ 5 deg, at the angle of attack of the grid nearest to 4 deg. A bank-restoring rolling moment has $C_{l\phi} < 0$; $C_l$ and $C_n$ are in the stability axes.

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

## Assumptions

- Flat, rigid ground; free stream parallel to the ground (no climb or descent); steady flow.
- Thin, inviscid wing; the wake is straight and parallel to the ground (no wake deformation near the ground).
- The ground has no boundary layer (a moving ground, as in flight).
- The wake follows the free stream. In ground effect the option `wake_alignment="body"` is replaced by the free-stream direction, with a warning; the agent tools refuse it.

## Envelope

- At small gaps a real thick section can lose lift (suction under the wing) or choke the flow; the thin-wing model cannot show this. The trust score lowers the rating below fixed values of $h_\text{min}/c$ (see [Trust score](trust_score.md)).
- The lifting-line solvers are refused near the ground (see [Numerical lifting line](numerical_lifting_line.md)). The classical lifting line has no ground effect.
- Unsteady effects (heave and pitch rates) and the derivatives with respect to rates are not modelled.
- The finite differences of the derivatives depend on the grid of the sweep; the end points of the height grid are first order.
- The Irodov margin depends on the moment reference point; it is meaningful only about the centre of gravity.

## Implementation

- `ventorum.aero.system`: `ground_normal`, `GroundPlane`, `make_ground_plane`, `build_sources` (image horseshoes), `check_ground_clearance`.
- `ventorum.aero.loads.trefftz_induced_drag_batch` (image wake in the Trefftz plane).
- `ventorum.ground_effect.analyze_ground_effect`, `find_bank_strike_limit`, `clearance_info`. The result also gives the force coefficient along the ground normal (`vertical_force_coefficient`), the body-axis force coefficients, and $C_l$ and $C_n$ in the stability axes.
- `ventorum.ground_effect.GroundEffectSweep`, `GroundEffectSweepResult.compute_stability_derivatives`.

## Verification

- [V7](../verification_report.md): zero normal velocity on a tilted and banked plane; the free-air limit at large height; chordwise convergence of the vortex lattice; lifting line against the vortex lattice.
- Tests compare an explicit mirror wing with the image method, and the Irodov terms with direct differences.

## References

- C. Wieselsberger, "Über den Flügelwiderstand in der Nähe des Bodens", *Zeitschrift für Flugtechnik und Motorluftschiffahrt* 12(10), 1921; English translation: "Wing resistance near the ground", NACA TM 77, 1922.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001 (method of images).
- K. V. Rozhdestvensky, *Aerodynamics of a Lifting System in Extreme Ground Effect*, Springer, 2000.
- K. V. Rozhdestvensky, "Wing-in-ground effect vehicles", *Progress in Aerospace Sciences* 42, 2006, pp. 211-283.
- R. D. Irodov, "Criteria of longitudinal stability of ekranoplan", *Uchenye Zapiski TsAGI* 1(4), 1970, pp. 63-74 (in Russian); English machine translation "Criteria of the longitudinal stability of the ekranoplan", Foreign Technology Division, 1974, DTIC AD-A002918.
