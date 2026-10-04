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

## Save and load

```python
from ventorum.core.config import save_aircraft_to_json, load_aircraft_from_json

save_aircraft_to_json(aircraft, "my_plane.json")
aircraft = load_aircraft_from_json("my_plane.json")
```
