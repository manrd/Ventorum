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
- On a strip of a chordwise control, the legs of a panel ahead of the hinge have a node at the hinge point of the strip edge: $A \to H_a \to \text{TE}_a$ and $B \to H_b \to \text{TE}_b$, one extra straight segment per leg, so the legs follow the deflected mean surface (see [Control surfaces](control_surfaces.md)). All kernel backends (numpy reference, Numba, Cython, PyTorch, NVIDIA Warp) evaluate the extra segment only for the sources that have a node; the other sources are computed as before.

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

Each horseshoe has a core radius $r_c$ [m]. The core acts on the distance $D$ [m] from the evaluation point to the finite filament:

- **Over the filament.** The foot of the perpendicular from $\mathbf P$ to the line of the segment lies on the segment ($\mathbf r_0\cdot\mathbf r_1 \ge 0$ and $\mathbf r_0\cdot\mathbf r_2 \le 0$), or the point is in front of the start of a semi-infinite line ($\hat{\mathbf d}\cdot\mathbf r \ge 0$). Then $D = \rho$, the distance from the line, and Ventorum uses

```{math}
\mathbf v_\text{segment} = \frac{\Gamma}{4\pi} \frac{\mathbf r_1 \times \mathbf r_2}{|\mathbf r_1 \times \mathbf r_2|^2 + r_c^2 |\mathbf r_0|^2}
\; \mathbf r_0 \cdot \left( \frac{\mathbf r_1}{|\mathbf r_1|} - \frac{\mathbf r_2}{|\mathbf r_2|} \right),
\qquad
\mathbf v_\text{line} = \frac{\Gamma}{4\pi} \frac{\hat{\mathbf d} \times \mathbf r}{|\hat{\mathbf d} \times \mathbf r|^2 + r_c^2}
\left( 1 + \frac{\hat{\mathbf d}\cdot\mathbf r}{|\mathbf r|} \right).
```

  For a point at the distance $\rho$ from the line of a segment, $|\mathbf r_1 \times \mathbf r_2| = \rho\,|\mathbf r_0|$ exactly, so both forms are the exact law multiplied by $\rho^2/(\rho^2 + r_c^2)$.

- **Outside the filament.** The point is before $\mathbf A$, after $\mathbf B$, or behind the start of a semi-infinite line. Then $D$ is the distance to the nearest end point ($|\mathbf r_1|$, $|\mathbf r_2|$ or $|\mathbf r|$), and the velocity is the exact law multiplied by $D^2/(D^2 + r_c^2)$. The exact law is computed in a form without cancellation:

```{math}
\mathbf v_\text{segment} = \frac{\Gamma}{4\pi} \frac{(\mathbf r_1 \times \mathbf r_2)\,(|\mathbf r_1| + |\mathbf r_2|)}{|\mathbf r_1|\,|\mathbf r_2|\,(|\mathbf r_1|\,|\mathbf r_2| + \mathbf r_1\cdot\mathbf r_2)}\,\frac{D^2}{D^2 + r_c^2},
\qquad
\mathbf v_\text{line} = \frac{\Gamma}{4\pi} \frac{\hat{\mathbf d} \times \mathbf r}{|\mathbf r|\,(|\mathbf r| - \hat{\mathbf d}\cdot\mathbf r)}\,\frac{D^2}{D^2 + r_c^2}.
```

The factor $D^2/(D^2 + r_c^2)$ is the core profile of M. P. Scully (1975), the case $n = 1$ of the family of G. H. Vatistas, V. Kozel and W. C. Mih (1991), applied here to the distance from the finite filament. When $r_c \ll D$ both forms are the exact law. At $D = 0$ (a point on the filament) the velocity is zero, not infinite. A positive term of this kind in the denominator of the Biot-Savart law is the regularisation of L. Rosenhead (1930); R. Krasny (1986) uses the same idea for vortex sheets.

The two forms agree at the end points of the filament, where $D = \rho$. $D^2$ and its first derivative are continuous there ($D^2 = \rho^2 + s^2|\mathbf r_0|^2$ outside, with $s$ the distance from the end point along the line in segment lengths), so the velocity and its derivatives with respect to the geometry have no jump. A core that acts on the distance $\rho$ from the extended line of the filament (the form of the first two equations everywhere) also removes the velocity at points near the extended line, far outside the filament, where the exact law is finite. Such points are common: the force point of a bound vortex lies on the extended line of the bound vortex of the next strip. That core changed the node gradients of the lifting line by an amount of order one against the exact law, with a length scale $r_c$ that finite differences with small steps do not show. The core on the distance from the finite filament has no such effect; the tests check the node gradients against an exact-law reference and the independence of the finite differences from the step (`tests/test_core_node_gradients.py`). The form and the core sizes are choices of Ventorum.

The core of a horseshoe is a fixed fraction (`ventorum.geometry.lattice.CORE_RADIUS_FRACTION`) of the smaller panel dimension (strip width or chordwise panel length). The nearest control point of the same surface is half a panel dimension away, so at the control points this core changes the velocity very little; its task is to cap the velocity when a point comes close to a filament.

## Cross-surface core

A trailing vortex of one surface can pass between the control points of another surface (for example a wing wake that meets a tail). Then the velocity at the nearest control point depends on where the filament falls between two control points, and the result is not grid convergent. Ventorum adds a core at the target point when the source and the target are on different surfaces:

```{math}
r_{c,\text{total}}^2 = r_{c,\text{source}}^2 + r_{c,\text{target}}^2, \qquad
r_{c,\text{target}} = f_\text{cross}\, w_\text{target},
```

with $w_\text{target}$ [m] the strip width at the target point and $f_\text{cross}$ = `ventorum.geometry.lattice.CROSS_SURFACE_CORE_FRACTION`. The owner chose the value of $f_\text{cross}$ from a core-size study; [V10](../verification_report.md) shows its effect on split surfaces. Points and vortices of the same surface use the source core only.

Surfaces that share an edge (two halves of a V-tail, a split wing, a wing and its winglet) are one vortex sheet, so the cross-surface core must not act between them. The function `core_groups` puts two surfaces into one core group when an edge segment (the line from the leading-edge node to the trailing-edge node of one section) of one surface lies on an edge segment of the other within a tolerance, over at least half of the shorter segment, and the chordwise grids of the two edges match: along the edge line, the leading-edge and trailing-edge nodes of one edge are within the tolerance of the end nodes of the other. The tolerance is the smaller local strip width, between a lower and an upper limit that are fractions of the edge length. The leg force at a joined edge needs coincident legs (see [Forces and moments](forces_and_moments.md)), so edges whose grids do not match (for example a fin root offset along the chord of the tail tip) are not joined. Warnings (`near_miss_warnings`) are given for: nearly parallel edges that are near but not joined; edges that lie on each other but whose chordwise grids do not match; and a join only through an interior strip edge of a surface (for example a tail tip on the side of a fin that passes through the tail plane), because such a join depends on the mesh. The root edge of a symmetric surface is not an interior edge for this rule.

## Assumptions

- The vortex filaments are straight and the wake is rigid: there is no wake roll-up and no wake relaxation.
- The wake legs are semi-infinite and straight along $\hat{\mathbf d}$.
- The fluid is inviscid and incompressible.

## Envelope

- A point on a filament gets a zero velocity from that filament (the core removes the singularity). Physical quantities are not taken at such points.
- The core changes the far-field velocity by a relative amount of order $(r_c/D)^2$.
- Wake-tail interaction depends on the cross-surface core size, which is a model choice. A measured wing-tail case is not yet available to calibrate it.

## Implementation

- `ventorum.aero.vortex`: `segment_velocity`, `ray_velocity` and `horseshoe_velocity_tensor` (numpy reference); `influence_matrix`, `velocity_tensor_unknowns`, `induced_velocity`.
- Compiled backends with the same equations: `ventorum.aero.vortex_numba` (Numba), `ventorum.aero.vortex_cython` (Cython), `ventorum.aero.vortex_torch` (PyTorch), and the GPU kernels in `ventorum.gpu.kernels` (NVIDIA Warp). Parity tests compare each backend with the numpy reference (`tests/test_kernels.py`).
- `ventorum.aero.vortex.HorseshoeSet` (sources) and `Targets` (core groups of the evaluation points).
- `ventorum.geometry.lattice.core_groups`, `near_miss_warnings`, `CORE_RADIUS_FRACTION`, `CROSS_SURFACE_CORE_FRACTION`.

## Verification

- [V6](../verification_report.md): conditioning and mesh convergence of the bent horseshoe.
- [V10](../verification_report.md): split surfaces and the core groups.
- Unit tests compare the regularised kernels with the exact Biot-Savart law away from the filaments and at points on and near the extended lines of the filaments (`tests/test_kernels.py`); each backend equals the numpy reference to round-off there.
- `tests/test_core_node_gradients.py`: node gradients of the linear lifting line against an exact-law reference (within 2 %), and finite differences that do not depend on the step from 1e-8 m to 1e-3 m.

## References

- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001: chapter 2 (Biot-Savart law), chapter 10 (vortex segment and horseshoe vortex).
- L. Rosenhead, "The spread of vorticity in the wake behind a cylinder", *Proceedings of the Royal Society of London A* 127, 1930, pp. 590-612.
- R. Krasny, "Desingularization of periodic vortex sheet roll-up", *Journal of Computational Physics* 65, 1986, pp. 292-313.
- M. P. Scully, "Computation of helicopter rotor wake geometry and its influence on rotor harmonic airloads", ASRL TR 178-1, Massachusetts Institute of Technology, 1975 (core velocity profile $r^2/(r^2 + r_c^2)$).
- G. H. Vatistas, V. Kozel and W. C. Mih, "A simpler model for concentrated vortices", *Experiments in Fluids* 11, 1991, pp. 73-76 (family of core profiles; $n = 1$ is the Scully profile).
