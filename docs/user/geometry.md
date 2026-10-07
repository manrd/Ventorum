# Geometry

A geometry has three levels.

| Object | Meaning |
| --- | --- |
| `WingSection` | The cross-section at a spanwise fraction `y_frac` (0 at the root, 1 at the tip). It has the chord, the twist, an optional leading-edge position (`x_le`, `z_le`) and the airfoil data. |
| `LiftingSurface` | A wing, a tail or a fin, from root to tip, defined by a list of sections. It has the semi-span, sweep, dihedral, position and the mesh settings of the surface. |
| `Aircraft` | A set of surfaces, the reference values (`S_ref`, `b_ref`, `c_ref`) and the moment reference point (`ref_point`). |

## Symmetric and mirrored surfaces

- A symmetric surface (`is_symmetric=True`, the default) is mirrored about the plane y = 0. Its root must be on that plane.
- For a surface off that plane, for example twin fins, set `is_symmetric=False` and add `surf.mirrored()` as the left copy. Camber and twist are mirrored too.
- Surfaces that share an edge (two halves of a V-tail, a wing in two parts, a wing and its winglet) act as one vortex sheet.

## Airfoil data

- `LinearAirfoil`: lift slope `a0`, zero-lift angle `alpha_L0`, profile drag `Cd0` and pitching moment `Cm0`.
- `TabulatedAirfoil`: a polar table (alpha, Cl, Cd, Cm), for example from XFOIL or a wind tunnel. The vortex-lattice solver uses its linear part; the nonlinear lifting line uses the full table.

## Control surfaces

Attach control surfaces to a `LiftingSurface` using `ControlSurface`. Specify the hinge chord fraction, spanwise extent, and deflection sign conventions.

```python
import numpy as np
from ventorum import ControlSurface, LiftingSurface, Aircraft

elevator = ControlSurface(
    name="elevator",
    hinge_x_frac=0.75,
    eta_start=0.0,
    eta_end=1.0,
    symmetric=True,
    deflection_limit=np.radians(25.0),
)

htail = LiftingSurface(
    name="horizontal_tail",
    sections=[...],
    controls=[elevator],
)

aircraft = Aircraft(surfaces=[wing, htail])

# Set control deflection
aircraft_pitched = aircraft.set_deflection("elevator", np.radians(-2.0))
```

## Deformed geometry

A user can set displacements of the lattice nodes on a `LiftingSurface` using `NodeDisplacements`. Displacements are specified for the leading-edge and trailing-edge nodes on the defining half (root to tip):

```python
import numpy as np
from ventorum import LiftingSurface, NodeDisplacements
from ventorum.geometry import undeformed_nodes, displacements_from_section_motion

# Extract the undeformed lattice nodes for the target mesh
nodes = undeformed_nodes(aircraft, solver="vlm")["Wing"]

# Create displacements from section heave and twist
n_edges = len(nodes["eta"])
heave = 0.05 * (nodes["eta"] ** 2)
twist = np.zeros(n_edges)
disp = displacements_from_section_motion(nodes, heave=heave, twist=twist)

# Assign displacements to the surface
wing.node_displacements = disp
```

Symmetric surfaces and mirror copies mirror the displacements across the plane y = 0.

## Save and load

```python
from ventorum.core.config import save_aircraft_to_json, load_aircraft_from_json

save_aircraft_to_json(aircraft, "my_plane.json")
aircraft = load_aircraft_from_json("my_plane.json")
```
