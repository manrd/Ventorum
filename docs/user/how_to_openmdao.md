# How to use Ventorum in OpenMDAO

`ventorum.mdo` gives two OpenMDAO components that wrap one Ventorum analysis.
OpenMDAO (Gray et al. 2019) then runs Ventorum inside optimisation problems and
coupled models, for example with a structural model that gives the node
displacements.

* `VentorumComponent`: an `ExplicitComponent` with the scalar outputs (`CL`, `CD`,
  `Cm`, ...) and analytic partials from `vt.gradients`.
* `VentorumMatrixFreeComponent`: also the vector outputs `strip_force` and
  `section_cl`, with matrix-free products from `vt.gradient_vjp`.

## Installation

OpenMDAO is an optional dependency. Install the extra `mdo`:

```bash
pip install "ventorum[mdo]"
```

`import ventorum` works without OpenMDAO. `import ventorum.mdo` without OpenMDAO
raises an `ImportError` that names the extra.

## A first problem

```python
import numpy as np
import openmdao.api as om
import ventorum as vt
from ventorum.mdo import VentorumComponent

wing = vt.LiftingSurface(
    name="Wing",
    semi_span=5.0,
    sections=[
        vt.WingSection(y_frac=0.0, chord=2.0),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ],
)
aircraft = vt.Aircraft(surfaces=[wing])
condition = vt.FlightCondition(V_inf=30.0, alpha=np.radians(4.0))
settings = vt.SolverSettings(solver_type="vlm", n_panels=12, n_chord=2)

prob = om.Problem(reports=False)
prob.model.add_subsystem(
    "aero",
    VentorumComponent(aircraft, settings, condition=condition,
                      outputs=("CL", "CD", "Cm"), inputs=("alpha", "nodes")),
    promotes=["*"],
)
prob.setup(mode="rev")
prob.set_val("alpha", 5.0, units="deg")
prob.run_model()
totals = prob.compute_totals(of=["CL"], wrt=["alpha"])
print(f"CL = {prob.get_val('CL')[0]:.4f}, dCL/dalpha = {totals['CL', 'alpha'][0, 0]:.4f} per rad")
```

The component keeps a copy of the geometry, the settings and the base flight
condition. `compute` runs `vt.analyze` once with the input values.
`compute_partials` runs `vt.gradients` once: one production solve and one adjoint
solve per output give the partials of each output with respect to all inputs.
The attribute `gradient_result` holds the last `GradientResult` (notes, device,
precision), and `last_result` holds the last `SolverResult`.

## Inputs and outputs

The names are the names of `vt.gradients`. OpenMDAO converts the units that you
give with `set_val`, `add_design_var` and connections.

| Input group | OpenMDAO inputs | Units |
| --- | --- | --- |
| `"alpha"`, `"beta"`, `"phi"` | `alpha`, `beta`, `phi` | rad |
| `"V_inf"` | `V_inf` | m/s |
| `"rho"` | `rho` | kg/m**3 |
| `"h"` | `h` (needs `condition.h`, ground effect) | m |
| `"p"`, `"q"`, `"r"` | `p`, `q`, `r` (body rates) | rad/s |
| `"nodes"` | `nodes:<surface>:le`, `nodes:<surface>:te`, shape (n_edges, 3) | m |
| `"controls"` | `controls:<name>` per control name | rad |

* The node inputs are the displacements of the leading-edge and trailing-edge
  nodes of the defining half of each surface, as `vt.NodeDisplacements`. Their
  default value is the displacement of the geometry (zero when it has none).
  `vt.geometry.undeformed_nodes(aircraft, settings)` gives the node positions and
  the number of edges. The mesh topology is fixed.
* On a symmetric surface the root stays on the plane y = 0 (the lattice refuses
  other roots). The component uses 0 for the y-displacement of the root nodes
  (row 0, column 1 of `le` and `te`), so the outputs do not depend on that entry
  and its partials are 0. `vt.gradients` gives a small nonzero derivative there;
  the component does not use it. A nonzero input value there gives an OpenMDAO
  warning (the component does not raise, so a `check_partials` step on that
  entry still runs). A nonzero value in the node displacements of the geometry
  raises `ValueError` at `setup`.
* A control input sets the deflection of all controls with that name (the left
  half of an antisymmetric control moves the opposite way, as in the solver).
  Controls with one name must have one deflection in the geometry.
* OpenMDAO does not accept a dot in a variable name. A character other than a
  letter, a digit or `_` in a surface or control name becomes `_` (the surface
  `wing.main` gives `nodes:wing_main:le`). The attribute `input_names` maps each
  OpenMDAO name to its Ventorum key.
* The input groups `"sections"`, `"reference"` and `"planform"` of `vt.gradients`
  are not available in the components: they raise `ValueError` at `setup`.

Outputs: the coefficients `CL`, `CY`, `CDi`, `CDp`, `CD`, `Cl`, `Cm`, `Cn`
(dimensionless; moments in the axes `axes`: `"body"`, `"stability"` or `"wind"`).
The matrix-free component also gives `strip_force` (strip forces [N], shape
(n_strips, 3), geometry axes) and `section_cl` (section lift coefficients, shape
(n_strips,)).

## Vector outputs: the matrix-free component

```text
VentorumMatrixFreeComponent(aircraft, settings, condition=condition,
                            outputs=("CL", "strip_force"), inputs=("alpha", "nodes"))
```

In the reverse mode of OpenMDAO (`prob.setup(mode="rev")`) each product runs
`vt.gradient_vjp` once with the output seeds: one production solve and one adjoint
solve. Use the reverse mode. The forward mode needs the full Jacobian (one adjoint
solve per scalar output and one per component of each vector output); the
component keeps it for the last input values, but it is slow on large meshes.

## Accuracy, device and precision

The partials are exact derivatives of the discrete solver (the discrete adjoint,
Martins and Hwang 2013), not finite differences. The device and the precision
follow the global settings of `ventorum.gpu` (`gpu.set_device`,
`gpu.set_precision`); the component does not change them.

* In float64, `prob.check_partials(method="fd", form="central")` agrees with the
  analytic partials to 1e-6 relative. Use a small central step (1e-7 for
  coefficients). The strip forces are in N; their smallest entries need a step of
  about 1e-6 and an absolute floor relative to the largest entry.
* In float32 (the default precision on the GPU) the agreement is about 1e-3
  relative: the float32 gate of the gradients. Use a larger difference step
  (about 1e-3) when you check float32 partials.

For an optimisation that needs tight convergence, select float64 with
`ventorum.gpu.set_precision("float64")`.

## Example: minimum drag at fixed lift with twist

The script `examples/12_openmdao_twist.py` minimises `CD` at `CL = 0.4` with the
vortex lattice and linear airfoils. The design variables are the twist angles at
four spanwise control stations. A small component interpolates the twist to the
node stations and turns it into node displacements about the quarter chord (small
angles). Few control stations keep the twist smooth: a free twist per node station
lets the optimiser find a saw-tooth twist that only the mesh sees. SLSQP converges
in a few tens of iterations or less, and `CD` comes near the elliptic-loading
value `CL^2/(pi AR) + Cd0`.

```{literalinclude} ../../examples/12_openmdao_twist.py
:language: python
:start-at: "def main"
```

## References

* J. S. Gray, J. T. Hwang, J. R. R. A. Martins, K. T. Moore and B. A. Naylor,
  "OpenMDAO: an open-source framework for multidisciplinary design, analysis, and
  optimization", Structural and Multidisciplinary Optimization 59, 1075-1104, 2019.
* J. R. R. A. Martins and J. T. Hwang, "Review and unification of methods for
  computing derivatives of multidisciplinary computational models", AIAA Journal
  51(11), 2582-2599, 2013.

See also [How to compute adjoint gradients](how_to_gradients.md) and the theory
manual chapter on the adjoint gradients.
