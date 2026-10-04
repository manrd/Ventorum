# Vortex filaments

## Model

All lattice solvers of Ventorum (the vortex-lattice method and the numerical lifting line) use one element: a bent horseshoe vortex of constant circulation $\Gamma$ [m^2/s]. The velocity that it induces comes from the Biot-Savart law for straight vortex segments and semi-infinite vortex lines (Katz and Plotkin 2001, chapters 2 and 10). A small core term in each denominator keeps the velocity finite near a filament. A second, larger core acts only between different surfaces.

## The bent horseshoe

A horseshoe of a panel has five parts:

```text
infinity --(along d)--> TE_a --(along chord)--> A --(bound)--> B --(along chord)--> TE_b --(along d)--> infinity
```

- The bound vortex $A \to B$ lies on the quarter-chord line of the panel. Positive $\Gamma$ runs from $A$ to $B$.
- The two chordwise legs $A \to \text{TE}_a$ and $B \to \text{TE}_b$ follow the strip edges to the trailing edge, in the plane of the surface.
- From the trailing edge, two semi-infinite legs go to infinity along the unit wake direction $\hat{\mathbf d}$.

The chordwise legs stay on the surface. Thus a control point never sits just above or below a trailing leg at angle of attack, and the matrix does not become nearly singular for narrow panels (see [V6](../verification_report.md)).

The wake direction is the free-stream direction $\hat{\mathbf u}$ by default (`wake_alignment="freestream"`). The option `wake_alignment="body"` gives $\hat{\mathbf d} = (1, 0, 0)$; the system matrix then does not depend on $\alpha$ and $\beta$. In ground effect the free-stream direction is always used, because the image method needs a wake parallel to the ground (see [Ground effect](ground_effect.md)).

## Biot-Savart law

For a straight segment from $\mathbf A$ to $\mathbf B$ and an evaluation point $\mathbf P$, let $\mathbf r_1 = \mathbf P - \mathbf A$, $\mathbf r_2 = \mathbf P - \mathbf B$ and $\mathbf r_0 = \mathbf B - \mathbf A$. The exact induced velocity is (Katz and Plotkin 2001, chapter 10)

```{math}
\mathbf v = \frac{\Gamma}{4\pi} \frac{\mathbf r_1 \times \mathbf r_2}{|\mathbf r_1 \times \mathbf r_2|^2}
\; \mathbf r_0 \cdot \left( \frac{\mathbf r_1}{|\mathbf r_1|} - \frac{\mathbf r_2}{|\mathbf r_2|} \right).
```

For a semi-infinite line that starts at $\mathbf A$ and goes to infinity along the unit vector $\hat{\mathbf d}$, with $\mathbf r = \mathbf P - \mathbf A$, the same law gives (derivation: put $\mathbf B = \mathbf A + L\hat{\mathbf d}$ and let $L \to \infty$; then $\mathbf r_2/|\mathbf r_2| \to -\hat{\mathbf d}$ and $|\mathbf r_1 \times \mathbf r_2| \to L\,|\hat{\mathbf d}\times\mathbf r|$)

```{math}
\mathbf v = \frac{\Gamma}{4\pi} \frac{\hat{\mathbf d} \times \mathbf r}{|\hat{\mathbf d} \times \mathbf r|^2}
\left( 1 + \frac{\hat{\mathbf d}\cdot\mathbf r}{|\mathbf r|} \right).
```

The velocity of the horseshoe is the sum of the bound segment, the two chordwise segments ($\text{TE}_a \to A$ and $B \to \text{TE}_b$), the semi-infinite line from $\text{TE}_b$ and, with the opposite sign, the semi-infinite line from $\text{TE}_a$.

## Core model

Each horseshoe has a core radius $r_c$ [m]. Ventorum uses these regularised forms:

```{math}
\mathbf v_\text{segment} = \frac{\Gamma}{4\pi} \frac{\mathbf r_1 \times \mathbf r_2}{|\mathbf r_1 \times \mathbf r_2|^2 + r_c^2 |\mathbf r_0|^2}
\; \mathbf r_0 \cdot \left( \frac{\mathbf r_1}{|\mathbf r_1|} - \frac{\mathbf r_2}{|\mathbf r_2|} \right),
\qquad
\mathbf v_\text{line} = \frac{\Gamma}{4\pi} \frac{\hat{\mathbf d} \times \mathbf r}{|\hat{\mathbf d} \times \mathbf r|^2 + r_c^2}
\left( 1 + \frac{\hat{\mathbf d}\cdot\mathbf r}{|\mathbf r|} \right).
```

For a point at the distance $\rho$ from the line of a segment, $|\mathbf r_1 \times \mathbf r_2| = |\mathbf r_1 \times \mathbf r_0| = \rho\,|\mathbf r_0|$ exactly, so both forms are the exact law multiplied by $\rho^2/(\rho^2 + r_c^2)$. When $r_c \ll \rho$ they are the exact law. At $\rho = 0$ the velocity is zero, not infinite. A positive term of this kind in the denominator of the Biot-Savart law is the regularisation of L. Rosenhead (1930); R. Krasny (1986) uses the same idea for vortex sheets. The form above and the core sizes are choices of Ventorum, which the tests check against the exact law.

The core of a horseshoe is a fixed fraction (`ventorum.geometry.lattice.CORE_RADIUS_FRACTION`) of the smaller panel dimension (strip width or chordwise panel length). The nearest control point of the same surface is half a panel dimension away, so at the control points this core changes the velocity very little; its task is to cap the velocity when a point comes close to a filament.

## Cross-surface core

A trailing vortex of one surface can pass between the control points of another surface (for example a wing wake that meets a tail). Then the velocity at the nearest control point depends on where the filament falls between two control points, and the result is not grid convergent. Ventorum adds a core at the target point when the source and the target are on different surfaces:

```{math}
r_{c,\text{total}}^2 = r_{c,\text{source}}^2 + r_{c,\text{target}}^2, \qquad
r_{c,\text{target}} = f_\text{cross}\, w_\text{target},
```

with $w_\text{target}$ [m] the strip width at the target point and $f_\text{cross}$ = `ventorum.geometry.lattice.CROSS_SURFACE_CORE_FRACTION`. The owner chose the value of $f_\text{cross}$ from a core-size study; [V10](../verification_report.md) shows its effect on split surfaces. Points and vortices of the same surface use the source core only.

Surfaces that share an edge (two halves of a V-tail, a split wing, a wing and its winglet) are one vortex sheet, so the cross-surface core must not act between them. The function `core_groups` puts two surfaces into one core group when an edge segment (the line from the leading-edge node to the trailing-edge node of one section) of one surface lies on an edge segment of the other within a tolerance, over at least half of the shorter segment. The tolerance is the smaller local strip width, between a lower and an upper limit that are fractions of the edge length. Surfaces with nearly parallel edges that are near but not joined give a warning.

## Assumptions

- The vortex filaments are straight and the wake is rigid: there is no wake roll-up and no wake relaxation.
- The wake legs are semi-infinite and straight along $\hat{\mathbf d}$.
- The fluid is inviscid and incompressible.

## Envelope

- A point on a filament gets a zero velocity from that filament (the core removes the singularity). Physical quantities are not taken at such points.
- The core changes the far-field velocity by a relative amount of order $(r_c/\rho)^2$.
- Wake-tail interaction depends on the cross-surface core size, which is a model choice. A measured wing-tail case is not yet available to calibrate it.

## Implementation

- `ventorum.aero.vortex`: `segment_velocity`, `ray_velocity` and `horseshoe_velocity_tensor` (numpy reference); `influence_matrix`, `velocity_tensor_unknowns`, `induced_velocity`.
- Compiled backends with the same equations: `ventorum.aero.vortex_numba` (Numba), `ventorum.aero.vortex_cython` (Cython), `ventorum.aero.vortex_torch` (PyTorch), and the GPU kernels in `ventorum.gpu.kernels` (NVIDIA Warp). Parity tests compare each backend with the numpy reference (`tests/test_kernels.py`).
- `ventorum.aero.vortex.HorseshoeSet` (sources) and `Targets` (core groups of the evaluation points).
- `ventorum.geometry.lattice.core_groups`, `near_miss_warnings`, `CORE_RADIUS_FRACTION`, `CROSS_SURFACE_CORE_FRACTION`.

## Verification

- [V6](../verification_report.md): conditioning and mesh convergence of the bent horseshoe.
- [V10](../verification_report.md): split surfaces and the core groups.
- Unit tests compare the regularised kernels with the exact Biot-Savart law away from the filaments.

## References

- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001: chapter 2 (Biot-Savart law), chapter 10 (vortex segment and horseshoe vortex).
- L. Rosenhead, "The spread of vorticity in the wake behind a cylinder", *Proceedings of the Royal Society of London A* 127, 1930, pp. 590-612.
- R. Krasny, "Desingularization of periodic vortex sheet roll-up", *Journal of Computational Physics* 65, 1986, pp. 292-313.
