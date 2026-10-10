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

A `ControlSurface` is a hinged flap (flap, aileron, elevator or rudder) on a `LiftingSurface`. Give it in the `controls` list of the surface. The fields are:

| Field | Meaning |
| --- | --- |
| `name` | The name of the control. Controls on different surfaces with the same name move together. |
| `eta_start`, `eta_end` | The span limits, as fractions of the semi-span (0 at the root, 1 at the tip). |
| `hinge_x_c` | The hinge position, as a fraction of the local chord from the leading edge. |
| `deflection` | The deflection [rad], positive trailing edge down (on a vertical fin: trailing edge to +y). The limit is 30 deg. |
| `symmetric` | `True` (flap, elevator): both halves deflect in the same direction. `False` (aileron): the left half deflects in the opposite direction. |

`Aircraft.set_deflection(name, deflection)` sets the deflection of all the controls with this name. It returns the number of controls that it changed. The deflection is part of the geometry, so each solve after the change uses it.

```python
import numpy as np
import ventorum as vt

# A wing with ailerons (antisymmetric) on the outer 40 % of the semi-span.
wing = vt.LiftingSurface(
    name="wing",
    semi_span=5.0,
    sections=[vt.WingSection(y_frac=0.0, chord=1.2), vt.WingSection(y_frac=1.0, chord=0.8)],
    controls=[vt.ControlSurface(name="aileron", eta_start=0.6, eta_end=1.0, hinge_x_c=0.75, symmetric=False)],
)

# A horizontal tail with an elevator on the full span.
htail = vt.LiftingSurface(
    name="htail",
    semi_span=1.6,
    position=np.array([4.0, 0.0, 0.3]),
    sections=[vt.WingSection(y_frac=0.0, chord=0.7), vt.WingSection(y_frac=1.0, chord=0.5)],
    controls=[vt.ControlSurface(name="elevator", eta_start=0.0, eta_end=1.0, hinge_x_c=0.7)],
)

aircraft = vt.Aircraft(name="demo", surfaces=[wing, htail], ref_point=np.array([0.3, 0.0, 0.0]))

# Right aileron trailing edge down 5 deg (left trailing edge up): roll to the left.
aircraft.set_deflection("aileron", np.radians(5.0))
# Elevator trailing edge up 2 deg: nose-up pitching moment.
aircraft.set_deflection("elevator", np.radians(-2.0))

result = vt.analyze(aircraft, alpha_deg=3.0, V_inf=30.0, solver="vlm")
print(f"CL = {result.totals.CL:.4f}, Cm = {result.totals.Cm:.4f}, Cl = {result.totals.Cl:.4f}")
```

The vortex-lattice solver with two or more chordwise panels turns the boundary-condition normals of the flap panels. The lifting-line solvers, and the vortex lattice with one chordwise panel, use a deflected section polar. The [theory chapter](../theory/control_surfaces.md) gives the two models. The Fourier solver does not accept a deflected control.

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
