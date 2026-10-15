---
orphan: true
---

# Plain-flap deflection of airfoil sections

This page describes the geometric plain-flap operation of Ventorum-2D.
The function `ventorum.airfoil.flap.deflect_plain_flap` deflects the aft
part of an airfoil section about a hinge point. The 2D solver and the
polar pipeline use the deflected coordinates. This operation is geometry
only; it models no aerodynamics.

## Sign convention

The deflection in degrees is positive when the trailing edge moves down.
A positive deflection rotates the flap clockwise in the section plane,
with the chord axis to the right and the normal axis up.

This matches the control-surface convention of
{doc}`control_surfaces`: a positive deflection is a right-hand rotation
about the hinge axis, which moves the trailing edge down on a horizontal
surface. The 2D section operation applies the same physical sense.

## Construction

The input is an array of shape (N, 2) in Selig order: from the trailing
edge along the upper surface to the leading edge, then along the lower
surface back to the trailing edge. The output keeps this order.

The construction has four steps:

1. Hinge point. The function `flap_hinge_point` puts the hinge at the
   station `x_hinge` in chord units. The height is a fraction `y_hinge`
   of the local thickness, from the lower surface (0) to the upper
   surface (1). The default 0.5 is the mid-thickness.
2. Rotation. Every point with x past `x_hinge` rotates about the hinge
   by the deflection angle, as a rigid body. The trailing-edge points of
   the output are exactly the rotated input points.
3. Compressed side. On the side that the rotation compresses (the lower
   surface for trailing edge down), the moved surface overlaps the fixed
   surface. Both are cut at the crossing of their polylines nearest to
   the hinge, and the overlapping points are removed.
4. Stretched side. On the side that the rotation stretches (the upper
   surface for trailing edge down), a gap opens between the fixed and
   the moved surface. A circular arc about the hinge fills the gap, from
   the last fixed point to the first moved point. The arc radius is the
   distance from the hinge to that surface at `x_hinge`. The point count
   `n_arc` includes both end points; the default count keeps the arc
   panel lengths near the local panel length.

A zero deflection returns a copy of the input.

## Limits

- The hinge station lies in (0.05, 0.98) chord units.
- The deflection satisfies |deflection| <= 60 deg.
- The hinge height fraction lies in [0, 1].
- The result is a simple polygon. It is not an aerodynamic solution:
  the area changes slightly, and the filler arc and the cut are
  geometric blends without viscous or gap physics.

## Verification

The operation is verified in `tests/test_airfoil_flap.py` on a symmetric
12 % section and on a cambered 4412 section from the analytic NACA
formulas, with 81 cosine-spaced points per surface:

- zero deflection returns the same points;
- for a set of deflections and hinge stations, the output has positive
  area, has no self-intersection, keeps the fixed part ahead of the
  hinge, moves the trailing-edge points by the exact rotation, and turns
  the hinge-to-trailing-edge line by the deflection;
- the area change stays below 2 % at 10 deg;
- out-of-range inputs raise ValueError.

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge
  University Press, 1926, chapter VII (thin-airfoil theory of a hinged
  flap).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged
  flap", Aeronautical Research Committee, R&M 1095, 1927.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover,
  1959 (airfoil section geometry and the NACA 4-digit profile equations).
