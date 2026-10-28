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

## Hinge treatments

The construction above gives the exact outline of the rigid rotation,
but the nodes near the hinge can be bad for a panel method. At small
deflections the moved surface on the compressed side does not cross the
fixed surface before the hinge line, so the hinge-line joint is a very
short panel between two corners of about 90 deg. The points added at the
hinge station can also make short panels next to the input points, and
the filler arc meets the surfaces at corners of the size of the local
surface slope plus a part of the deflection.

The option `hinge_treatment` of `deflect_plain_flap` and
`AirfoilShape.with_flap` selects what happens to these nodes. The
default is "repanel".

- "none": the outline stays as constructed (the output of the
  construction above, bit for bit).
- "merge": nodes in the hinge zone, $|x - x_h| < 0.05$ chord units, are
  removed. The trailing-edge nodes and the leading-edge node are never
  removed. Two rules apply in turn until neither removes a node.
  1. Short panels. No panel with a removable end is shorter than 0.5
     times the mean length of its two neighbour panels. The shortest
     panel (by this ratio) loses the end node whose removal leaves the
     smaller largest turning angle at its neighbours, so the removal also
     smooths the corner. At equal angles, the end node nearest to the
     line through its own neighbours goes.
  2. Corners. A node with a turning angle above 5 deg is removed when
     this lowers the largest turning angle of the node and its two
     neighbours, and the new panel is at most 1.5 times the mean length
     of its neighbour panels (the adjacent panel length limit of
     `repanel`). The node with the largest turning angle goes first, one
     node per pass. A longer panel at the hinge corner makes the viscous
     solution harder, so the stretch limit stops corner removal on
     coarse input; rule 1 still smooths the corners there.

  A removal that makes the polygon touch or cross itself is not done.
  The kept nodes do not move: the outline changes only where a chord
  replaces removed nodes. The panel count goes down by the number of
  removed nodes. A single corner can only be split into two smaller
  corners by node removal; the stretch limit of rule 2 stops it when the
  new panel would be too long.
- "repanel": rule 1 of "merge" removes the short panels, with the hinge
  node of each surface kept. The hinge node is the crossing point on a
  cut side, else the point of the surface at the hinge station. A
  parametric not-a-knot cubic spline (de Boor, 1978, chapter IV) then
  goes through the remaining points, and the nodes are distributed
  again along its arc length in one pass of the equidistribution of
  `repanel` (same weight function and default values). The trailing-edge
  points, the leading edge of the spline and the two hinge nodes are
  breakpoints of the equidistribution, so they stay nodes; the
  trailing-edge and hinge nodes are the exact input points. The
  curvature term of the weight uses the peak curvature of the fixed part
  as reference and is clipped at 1, so a corner at the hinge gets the
  node density of the leading edge without a change of the density
  elsewhere. The panel count is the input count, or the argument
  `n_panels`. The repeated passes of `repanel` are not used, because
  each new spline through the last nodes moves the surface further from
  the outline at the hinge corners.

  The spline rounds the corners at the hinge, so the surface there
  moves away from the deflected outline. The tests measure the largest
  distance from a node of the result to the outline of "none" on NACA
  0012, 2412 and 4415 sections with 120 to 240 panels and the 161
  generator points, hinges from 0.6 to 0.85 and deflections from -20 to
  20 deg: it stays below $2 \times 10^{-3}$ chord units. The deviation
  is largest at the large deflections, where the cut side has a real
  corner.

All treatments keep the trailing-edge points as the exact rotated input
points, keep the flap rotation exact, and check that the result is a
simple polygon with a positive area. "none" and "merge" keep the fixed
part ($x < x_h - 0.05$) unchanged. A zero deflection returns a copy of
the input with "none"; "merge" and "repanel" treat the input with a
node added at the hinge station of each surface, so their result changes
continuously with the deflection ("merge" then returns the input
outline with these two nodes).

The default is "repanel". A study of the method found that it makes
the viscous solution converge where "none" fails (small deflections),
and that its mesh does not depend on the input points.

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
  sign rule are kept as regression cases;
- the hinge treatments "merge" and "repanel" give valid sections with
  the exact trailing-edge rotation on NACA 0012, 2412 and 4415, hinges
  from 0.6 to 0.85 and deflections from -20 to 20 deg; "merge" keeps
  only outline nodes, keeps the fixed part and leaves no short panel in
  the hinge zone; "repanel" keeps the panel count, a node at the hinge
  on both surfaces, and stays inside the deviation bound above; both
  remove the hinge-line step of the small-deflection case.

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge
  University Press, 1926, chapter VII (thin-airfoil theory of a hinged
  flap).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged
  flap", Aeronautical Research Committee, R&M 1095, 1927.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover,
  1959 (airfoil section geometry and the NACA 4-digit profile equations).
- C. de Boor, *A Practical Guide to Splines*, Springer, 1978, chapter IV
  (not-a-knot cubic spline).
