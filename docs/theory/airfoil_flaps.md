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

The construction has five steps:

1. Hinge point. The function `flap_hinge_point` puts the hinge at the
   station `x_hinge` in chord units. The height `y_hinge` is a fraction
   of the local thickness at that station, from the lower surface (0)
   to the upper surface (1); the default 0.5 is the mid-thickness. The
   chapter {doc}`airfoils` states this definition for the whole package.
   The surfaces are split at the leading edge of the package definition
   (the point farthest from the trailing-edge midpoint). A short run of
   decreasing x near the nose (for example on a section that already
   carries a deflected flap) is accepted forward of the hinge; decreasing
   x at or aft of the hinge raises ValueError.
2. Rotation. Every point with x past `x_hinge` rotates about the hinge
   by the deflection angle, as a rigid body. The trailing-edge points of
   the output are exactly the rotated input points.
3. Compressed side. The sign of the deflection selects it: the lower
   surface for trailing edge down, the upper surface for trailing edge
   up. The moved surface overlaps the fixed surface there. Both are cut
   at the crossing of their polylines nearest to the hinge, and the
   overlapping points are removed. On coarse input the two polylines can
   miss each other (the moved surface passes the fixed corner). Then
   both parts are cut at the hinge line, the line through the hinge
   point normal to the chord, and joined there by a short segment on
   that line.
4. Stretched side. On the other side a gap opens between the fixed and
   the moved surface. A circular arc about the hinge fills the gap, from
   the last fixed point to the first moved point. The arc radius is the
   distance from the hinge to that surface at `x_hinge`. The point count
   `n_arc` includes both end points; the default count keeps the arc
   panel lengths near the local panel length.
5. Check. The output must be a simple polygon (no two edges cross or
   touch, the closing trailing-edge edge included) with a positive area.
   Otherwise the function raises `AirfoilGeometryError` and names the
   crossing edges.

Before this rule, a side counted as stretched whenever the two polylines
did not cross, so coarse input could get the filler arc on the
compressed side and a self-intersecting contour without an error.

A zero deflection returns a copy of the input.

## Limits

- The hinge station lies in (0.05, 0.98) chord units.
- The deflection satisfies |deflection| <= 60 deg.
- The hinge height fraction lies in [0, 1].
- The input must be in Selig order (positive signed area).
- The result is a simple polygon, or the function raises. It is not an
  aerodynamic solution: the area changes slightly, and the filler arc,
  the cut and the hinge-line joint are geometric blends without viscous
  or gap physics.

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
- out-of-range inputs raise ValueError;
- a sweep over NACA 0006, 0012, 2412 and 4415 with 31, 61 and 161
  points, hinges from 0.1 to 0.97, deflections from -60 to 60 deg and
  hinge heights from 0 to 1 gives a simple polygon with a positive area
  in every case (a dense grid of the same sweep carries the slow
  marker);
- the coarse cases that gave a self-intersecting contour before the
  sign rule are kept as regression cases.

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge
  University Press, 1926, chapter VII (thin-airfoil theory of a hinged
  flap).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged
  flap", Aeronautical Research Committee, R&M 1095, 1927.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover,
  1959 (airfoil section geometry and the NACA 4-digit profile equations).
