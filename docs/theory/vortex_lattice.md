# Vortex-lattice method

## Model

The vortex-lattice method (`solver_type="vlm"`, the default) models each lifting surface as a thin sheet of bound vorticity. The surface is cut into spanwise strips, and each strip into chordwise panels. Each panel carries one bent horseshoe vortex (see [Vortex filaments](vortex_kernels.md)) with its bound vortex on the quarter-chord line of the panel. The flow-tangency condition at one control point per panel gives a dense linear system for the circulations. The method goes back to V. M. Falkner (1943); the form with horseshoe vortices and the three-quarter-chord control point is the one of Katz and Plotkin (2001, section 12.1) and Drela (2014). The forces come from the Kutta-Joukowski law and the induced drag from the Trefftz plane (see [Forces and moments](forces_and_moments.md)).

## Lattice

### Spanwise strips

Each surface has `n_panels` strips per semi-span. The strip edges come from a spacing function $\eta(t)$ that maps a uniform parameter $t \in [0, 1]$ to the span fraction: the edges are at $t_k = k/n$ and the control point of strip $k$ is at the mid parameter $t = (k + \tfrac12)/n$, not at the mean of the two edges (except next to a section break, see below). The functions are

| Rule | $\eta(t)$ |
| --- | --- |
| `half-cosine` (tip clustering) | $\sin(\pi t / 2)$ |
| `cosine` (root and tip clustering) | $\tfrac12 (1 - \cos \pi t)$ |
| `root-cosine` | $1 - \cos(\pi t / 2)$ |
| `uniform` | $t$, control point at the strip centre |
| `power` | a power law with an exponent |

Over the whole span of a symmetric wing, `half-cosine` is the cosine distribution $y = \tfrac{b}{2}\cos\theta$ with uniform steps in $\theta$, and the control points are at the $\theta$ midpoints. With these control points the planar elliptic wing gives $e = 1$ with few panels (the quasi-vortex-lattice method of C. E. Lan 1974 uses a related semicircle rule in the chordwise direction; R. M. James 1972 explains the accuracy of the vortex lattice with such point placements). The `auto` rule selects `half-cosine` for planar wings and `cosine` for a symmetric surface with a dihedral kink or high sweep at the root (`ventorum.geometry.discretization.determine_optimal_spacing`). The lifting line always uses `half-cosine`.

A section where the planform has a kink or the airfoil changes is put on a strip edge, so that no strip averages across the break. This applies only to surfaces with 3 to 6 sections, and only when each neighbour strip keeps at least a quarter of its width. The two strips next to a moved edge then have their control points at the mean of their edges.

With `SolverSettings.proportional_panels`, each surface gets a panel count in proportion to its semi-span (`compute_surface_n_panels`). A surface with `mirror_y=True` is the mirror image (y to -y) of its definition; it is a separate surface, not a symmetric one.

### Chordwise panels

Each strip has `n_chord` panels, uniform or cosine-spaced in the chord. Out of ground effect the default is `ventorum.solvers.lattice_base.DEFAULT_N_CHORD`. In ground effect a panel should not be longer than the smallest gap $h_\text{min}$ [m] to the ground, so the default is

```{math}
n_\text{chord} = \left\lceil \frac{c_\text{max}}{h_\text{min}} \right\rceil,
```

kept between `DEFAULT_N_CHORD` and `MAX_AUTO_N_CHORD` ($c_\text{max}$ [m] is the largest strip chord).

### Control point and section lift slope

On a panel from chord fraction $x_0$ to $x_1$ (length $\Delta x = x_1 - x_0$), the bound vortex is at $x_0 + \tfrac14 \Delta x$ and the control point at

```{math}
x_\text{cp} = x_0 + \tfrac14 \Delta x + \tfrac12 \Delta x\, \text{CLAF},
\qquad \text{CLAF} = \frac{a_0}{2\pi},
```

with CLAF limited to a fixed interval. For $\text{CLAF} = 1$ this is the three-quarter-chord point of the panel, which gives the exact lift of a flat plate in two-dimensional flow (the rule of E. Pistolesi 1937; Katz and Plotkin 2001, section 5.5, the lumped-vortex element).

Derivation of the shift: consider one panel of chord $c$ in two-dimensional flow, with a point vortex $\Gamma$ at the quarter chord and the control point a distance $d$ behind it. Flow tangency for a flat plate at the angle $\alpha$ gives $\Gamma/(2\pi d) = V_\infty \alpha$. The lift coefficient is $C_l = 2\Gamma/(V_\infty c) = 4\pi (d/c)\, \alpha$. For $C_l = a_0 \alpha$ the distance must be $d = \tfrac{c}{2}\,\frac{a_0}{2\pi}$, which is the shift above. Ventorum applies the same shift to every chordwise panel. With one chordwise panel the derivation is exact in two-dimensional flow; with several panels it is a model choice.

### Camber

Camber enters through the zero-lift angle. The normal of the boundary condition, $\hat{\mathbf n}_{bc}$, is the geometric normal of the strip turned nose up by $-\alpha_{L0}$ about the spanwise axis in the y-z plane (the same axis as the twist). The section data are defined in streamwise sections, so camber then acts as the same incidence also on a swept surface.

### Deformed geometry

Node displacements define deformed geometry for aeroelastic analysis. A user provides displacements of the leading-edge and trailing-edge lattice nodes on the defining half of each surface in geometry axes. Displacements are added to the undeformed edge coordinates. For symmetric surfaces and mirror copies, the defining half is displaced and mirrored across the plane y = 0.

Reference values ($S_{\text{ref}}$, $b_{\text{ref}}$, $c_{\text{ref}}$) and the moment reference point stay on the undeformed geometry. This matches the standard convention for flexible aircraft. The strip twist array keeps the undeformed jig twist. The panel nodes and boundary normals follow the displaced geometry. The trailing wake leaves the deformed trailing edge along the free stream or the body axis, as in the undeformed formulation.

## Equations

The flow-tangency condition at the control point $\mathbf P_i$ of panel $i$ is

```{math}
:label: eq-vlm
\sum_j A_{ij}\,\Gamma_j = -V_\infty\, \hat{\mathbf u}\cdot\hat{\mathbf n}_{bc,i},
\qquad A_{ij} = \mathbf v_{ij}\cdot\hat{\mathbf n}_{bc,i},
```

where $\mathbf v_{ij}$ [1/m] is the velocity at $\mathbf P_i$ from the horseshoe $j$ with unit circulation, including its mirror and ground images when they apply (see [Symmetry](symmetry.md) and [Ground effect](ground_effect.md)). The system is solved with one dense LU factorisation (LAPACK through `numpy.linalg.solve`).

## Assumptions

- Thin surfaces: thickness enters only through the section data. Inviscid, incompressible, attached flow.
- Linear section data. A tabulated polar enters through its linear fit, with a warning; stall is not modelled.
- Small disturbances: the boundary condition is applied on the mean surface, and the wake is rigid and straight (along the free stream by default).

## Envelope

- No stall: for the start of stall on an unswept wing use the nonlinear lifting line.
- No leading-edge or side-edge vortex lift. The trust score lowers the rating for slender, highly swept wings and for low-aspect-ratio wings at high angle of attack (E. C. Polhamus 1966; J. E. Lamar 1974).
- No compressibility correction. Above the Mach number `ventorum.core.constants.MACH_LIMIT_INCOMPRESSIBLE` the case is outside the envelope.
- Two surfaces that overlap (control points of different surfaces closer than a small fraction of the local chord) are refused (`check_overlaps`).
- In ground effect, panels longer than the gap get a trust penalty.

## Implementation

- `ventorum.solvers.horseshoe.HorseshoeSolver` (alias `VortexLatticeSolver`), on `ventorum.solvers.lattice_base.LatticeSolver` (`resolve_n_chord`, `build`, `solve`, `solve_sweep`, `solve_batch`).
- `ventorum.geometry.lattice.build_lattice` (strips, panels, control points, normals, core groups); `ventorum.geometry.discretization` (spacing functions).
- `ventorum.solvers.core.assemble_vlm`, `solve_vlm`, `solve_vlm_batch`.
- In a sweep out of ground effect, the influence of the bound vortices and the chordwise legs does not change with the angle and is computed once (`VortexLattice.kernel_cache`); only the wake legs are computed for each angle.
- The GPU pipelines (`ventorum.gpu`) solve the same system in float32 or float64.

## Verification

- [V1](../verification_report.md), [V2](../verification_report.md): lifting-line limit of the method (high aspect ratio).
- [V3](../verification_report.md): lifting-surface checks (circular wing of W. Kinner 1937).
- [V4](../verification_report.md): grid convergence on swept wings.
- [V6](../verification_report.md): mesh convergence and conditioning.
- [V8](../verification_report.md): camber and section pitching moment.

## References

- V. M. Falkner, "The calculation of aerodynamic loading on surfaces of any shape", Aeronautical Research Council, R&M 1910, 1943.
- E. Pistolesi, "Betrachtungen über die gegenseitige Beeinflussung von Tragflügelsystemen", *Gesammelte Vorträge der Hauptversammlung 1937 der Lilienthal-Gesellschaft für Luftfahrtforschung*, 1937.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, sections 5.5 and 12.1.
- M. Drela, *Flight Vehicle Aerodynamics*, MIT Press, 2014.
- R. M. James, "On the remarkable accuracy of the vortex lattice method", *Computer Methods in Applied Mechanics and Engineering* 1, 1972, pp. 59-79.
- C. E. Lan, "A quasi-vortex-lattice method in thin wing theory", *Journal of Aircraft* 11(9), 1974, pp. 518-527.
- E. C. Polhamus, "A concept of the vortex lift of sharp-edge delta wings based on a leading-edge-suction analogy", NASA TN D-3767, 1966.
- J. E. Lamar, "Extension of leading-edge-suction analogy to wings with separated flow around the side edges at subsonic speeds", NASA TR R-428, 1974.
- W. Kinner, "Die kreisförmige Tragfläche auf potentialtheoretischer Grundlage", *Ingenieur-Archiv* 8, 1937, pp. 47-80.
