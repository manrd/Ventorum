# Symmetry

## Model

When the geometry and the flow are symmetric about the x-z plane, the circulation is symmetric too: a panel on the left half has the same circulation as its mirror panel on the right half. Ventorum then solves for the circulations of the right half only, which divides the number of unknowns by two and the work of the dense LU solve by about eight (it scales with the cube of the number of unknowns). The result is the same as the full solve, to round-off.

## Conditions

The solver uses the symmetry (`SolverSettings.use_symmetry`, on by default) only when all of these are true:

- every panel has a mirror panel: all surfaces are symmetric (`LiftingSurface.is_symmetric`);
- no sideslip, $\beta = 0$;
- the ground plane, if any, has a normal with no y component (no bank, $\phi = 0$).

Two separate surfaces that are mirror copies of each other (`mirror_y`) are solved without the reduction.

## Equations

Let $j'$ be the mirror panel of panel $j$. With $\Gamma_{j'} = \Gamma_j$ the system of the [vortex lattice](vortex_lattice.md) (or of the [lifting line](numerical_lifting_line.md)) for the right-half panels $i$ becomes

```{math}
\sum_{j \in \text{right}} \left( A_{ij} + A_{ij'} \right) \Gamma_j = b_i ,
```

that is, each unknown gets the sum of the influences of its real horseshoe, its mirror horseshoe and (in ground effect) their images. The right-hand side and the rows are those of the right-half control points.

After the solve, the velocity at a left control point is the mirror image of the velocity at its mirror point: $(v_x, v_y, v_z) \to (v_x, -v_y, v_z)$. The loads use the same rule: only the right panels are evaluated, and the left panels get the mirror image. Before it uses this fold, the code checks that the evaluation points and the circulation are symmetric to round-off; if not, it evaluates every panel.

## Assumptions

- Exact geometric symmetry of every surface about the plane y = 0.
- A symmetric solution: for a linear system with a unique solution, symmetric data give a symmetric solution. For the nonlinear lifting line past the stall, an asymmetric solution can also exist; the reduction finds only symmetric ones.

## Envelope

- Antisymmetric cases (sideslip, bank, roll) use the full solve.
- Past the maximum lift, a symmetric case can have asymmetric solutions (stall on one side). The reduction does not look for them.

## Implementation

- `ventorum.aero.system.is_symmetric_condition`, `make_unknown_map` (`UnknownMap`: which panels are unknowns and the unknown of each panel), `build_sources`.
- `ventorum.solvers.core._unknown_map`, `_panel_velocity`.
- `ventorum.aero.loads._symmetric_fold`, `_near_field_velocity`.

## Verification

- The test suite compares the folded and the full solves (lift, moments, circulation), on the CPU and the GPU paths.
- [V7](../verification_report.md) uses symmetric and banked planes.

## References

- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapter 12.
