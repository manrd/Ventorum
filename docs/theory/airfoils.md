---
orphan: true
---

# Airfoils: geometry, NACA sections, paneling and flaps

This chapter joins the three airfoil notes into one coherent chapter.
It gives the coordinate convention, the file formats, the geometry
checks, the NACA thickness and mean lines, the node distributions for
panel methods, the geometric plain-flap operation, and the single
public interface that serves the 2D and 3D solvers.

## Coordinate convention

A section is a numpy array of shape (N, 2), float64, in Selig order:
from the trailing edge along the upper surface to the leading edge,
then along the lower surface back to the trailing edge. The order is
counterclockwise, so the signed shoelace area is positive. Unit chord
has the leading edge at (0, 0) and the trailing-edge midpoint at
(1, 0). A closed trailing edge repeats the same point at the start and
the end. The `normalized` method moves any shape to this frame: it
translates the leading edge to the origin, rotates the trailing-edge
midpoint onto the x axis and scales the chord to one.

The `upper` and `lower` methods each return one surface from the
leading edge to the trailing edge, both with the leading-edge point
first. The leading edge is the stored point farthest from the
trailing-edge midpoint. Thickness is the upper y minus the lower y at
one chord station, and camber is the mean of the two, both by linear
interpolation on each surface.

The paneling functions accept coordinates in any consistent length
unit. A physical wing section uses m. A unit section uses chord units.
The shape methods of this chapter use chord units throughout.

## File formats

Three plain-text formats are read. The reader detects the format from
the first lines.

- Selig: an optional name line, then x y pairs in Selig order (M. S. Selig and co-workers, *Summary of Low-Speed Airfoil Data*, 1995).
- Lednicer: a name line, a line with the point counts of the upper and lower surfaces (two numbers larger than 1, for example `61 61`), then the upper surface from the leading edge to the trailing edge, a blank line, and the lower surface from the leading edge to the trailing edge (D. Lednicer, AIAA paper 2010-0458, 2010). The reader merges the duplicated leading-edge point.
- Plain: x y pairs with no header. The name then comes from the file stem.

Separators are spaces, tabs or commas. Lines that start with `#` are
comments and blank lines are skipped. A file in reverse order (lower
surface first, clockwise) is reversed to Selig order, so its signed
area is positive.

The writers store the same point order with a fixed number of digits
(`precision`, default 6). A file written and read again returns the
same points to the written precision.

## Geometry checks

`problems` lists the defects of a shape and is empty when the shape is
good:

- fewer than 10 points;
- repeated consecutive points (distance below 1e-12 chord units);
- clockwise order (negative signed area);
- self-intersection of the polygon (segment pair test);
- a surface whose x is not monotonic (a note, not an error).

`validate` raises `AirfoilGeometryError` when any hard defect is
present. The non-monotonic x notes do not raise, because a real
section can have a locally vertical tangent near the leading edge.

## NACA thickness distribution

The 4-digit and 5-digit series share the thickness distribution of
Abbott and von Doenhoff (1959, section 6.4). With the thickness ratio
$t$ [-] and the station $x$ [-], the half thickness $y_t$ [-] is

```{math}
y_t = 5 t (0.2969 \sqrt{x} - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 + a_4 x^4).
```

The open trailing edge uses $a_4 = -0.1015$ and keeps a finite trailing
edge thickness. The closed trailing edge uses $a_4 = -0.1036$, which
closes the trailing edge to zero thickness (Ladson, Brooks, Hill and
Sproles, NASA TM-4741, 1996).

## NACA 4-digit mean line

The 4-digit mean line follows Abbott and von Doenhoff (1959, section
6.4). With the maximum camber $m$ [-] at the position $p$ [-], the mean
line $y_c$ [-] and its slope are, for $x < p$,

```{math}
y_c = \frac{m}{p^2}(2 p x - x^2), \qquad
\frac{dy_c}{dx} = \frac{2m}{p^2}(p - x),
```

and for $x \ge p$,

```{math}
y_c = \frac{m}{(1-p)^2}((1 - 2p) + 2 p x - x^2), \qquad
\frac{dy_c}{dx} = \frac{2m}{(1-p)^2}(p - x).
```

The designation "2412" gives $m = 0.02$, $p = 0.4$ and $t = 0.12$. The
designation "0012" gives $m = 0$ and $p = 0$ (symmetric section).

## NACA 5-digit mean line

The standard (non-reflex) 5-digit mean lines follow Abbott and von
Doenhoff (1959). Jacobs, Ward and Pinkerton (NACA Report 460, 1933) and
Jacobs and Pinkerton (NACA Report 537, 1935) define the series. The
first digit sets the design lift coefficient ($0.15$ times the digit),
the second and third digits set the camber position, and the last two
digits set $t$. With the parameters $m$ [-] and $k_1$ [-], the mean
line is, for $x \le m$,

```{math}
y_c = \frac{k_1}{6}(x^3 - 3 m x^2 + m^2 (3 - m) x), \qquad
\frac{dy_c}{dx} = \frac{k_1}{6}(3 x^2 - 6 m x + m^2 (3 - m)),
```

and for $x > m$,

```{math}
y_c = \frac{k_1 m^3}{6}(1 - x), \qquad
\frac{dy_c}{dx} = -\frac{k_1 m^3}{6}.
```

The parameters for design lift 0.3 (first digit 2) are:

| Mean line | m | k1 |
| --- | --- | --- |
| 210 | 0.0580 | 361.40 |
| 220 | 0.1260 | 51.64 |
| 230 | 0.2025 | 15.957 |
| 240 | 0.2900 | 6.643 |
| 250 | 0.3910 | 3.230 |

Other design lifts scale $k_1$ linearly with the first digit: a mean
line with first digit $d$ uses $k_1(d) = k_1(2) \times d / 2$. Reflex
mean lines (third digit 1) are not built: no published table of $k_1$
and $k_2/k_1$ is loaded, so they raise an error.

## NACA coordinates

The thickness acts perpendicular to the mean line. With
$\theta = \atan(dy_c/dx)$, the upper surface is

```{math}
x_u = x - y_t \sin\theta, \qquad y_u = y_c + y_t \cos\theta,
```

and the lower surface uses the opposite signs. Cosine spacing sets
$x = (1 - \cos\beta)/2$ with uniform $\beta$ in $[0, \pi]$; uniform
spacing is also available. An odd point count gives $(n+1)/2$ stations
per surface, with the leading edge point stored once. A closed trailing
edge repeats the same point at the start and the end. The points are
not re-normalized: the perpendicular thickness moves the true leading
edge of a cambered section slightly off (0, 0), which the docstring
states.

## Panel node distributions

The paneling tools fit a parametric cubic spline through the section
and redistribute the nodes for panel methods. Each component uses a
not-a-knot cubic spline (de Boor, 1978, Chapter IV). The curvature uses
the standard differential geometry formula for a parametric plane
curve.

The `repanel` function clusters nodes at the leading edge (high
curvature) and keeps them refined at the trailing edge. The node
density follows the equidistribution of a weight function along arc
length: the nodes split the cumulative weight integral into equal
parts. The weight adds normalized curvature and Gaussian refinement
terms at the leading edge and at both trailing-edge ends. The leading
edge is always a node. The weight is smoothed until the adjacent panel
length ratio stays at most 1.5. The fit-and-equidistribute pass repeats
until the set stops moving, so repaneling an already repaneled set
returns the same nodes. A closed trailing edge stays closed; an open
trailing edge keeps its two end points exactly.

The `repanel_cosine` function resamples each surface with Chebyshev
cosine nodes in arc length from the leading edge, dense at the leading
edge and at the trailing edge. The leading edge is always a node.

## Plain-flap deflection

The function `deflect_plain_flap` deflects the aft part of a section
about a hinge point. The 2D solver and the polar pipeline use the
deflected coordinates. This operation is geometry only; it models no
aerodynamics.

The deflection in degrees is positive when the trailing edge moves
down. A positive deflection rotates the flap clockwise in the section
plane, with the chord axis to the right and the normal axis up. This
matches the control-surface convention of {doc}`control_surfaces`.

The construction has four steps:

1. Hinge point. The function `flap_hinge_point` puts the hinge at the station `x_hinge` in chord units. The height is a fraction `y_hinge` of the local thickness, from the lower surface (0) to the upper surface (1). The default 0.5 is the mid-thickness.
2. Rotation. Every point with x past `x_hinge` rotates about the hinge by the deflection angle, as a rigid body. The trailing-edge points of the output are exactly the rotated input points.
3. Compressed side. On the side that the rotation compresses (the lower surface for trailing edge down), the moved surface overlaps the fixed surface. Both are cut at the crossing of their polylines nearest to the hinge, and the overlapping points are removed.
4. Stretched side. On the side that the rotation stretches (the upper surface for trailing edge down), a gap opens between the fixed and the moved surface. A circular arc about the hinge fills the gap, from the last fixed point to the first moved point. The arc radius is the distance from the hinge to that surface at `x_hinge`. The point count `n_arc` includes both end points; the default count keeps the arc panel lengths near the local panel length.

A zero deflection returns a copy of the input. The hinge station lies
in (0.05, 0.98) chord units. The deflection satisfies
|deflection| <= 60 deg. The hinge height fraction lies in [0, 1]. The
result is a simple polygon. The area changes slightly, and the filler
arc and the cut are geometric blends without viscous or gap physics.

## Unified interface

Users and solvers get airfoils one way, through `AirfoilShape` and the
`airfoil` factory. The array functions stay public and unchanged.

- `AirfoilShape.naca(designation, **kw)` builds a NACA section.
- `AirfoilShape.from_file(path)` reads a Selig, Lednicer or plain file.
- `shape.repanel(n_panels=160, **kw)` redistributes the nodes.
- `shape.repanel_cosine(n_per_side)` resamples with cosine spacing.
- `shape.with_flap(x_hinge, deflection, y_hinge=0.5)` deflects a plain flap.
- `shape.to_file(path, fmt="selig")` writes a Selig or Lednicer file.
- `airfoil(spec)` accepts a NACA designation string, a file path, an (N, 2) array or a shape, and returns a shape. A string names a file when the path exists, else it names a NACA section.

The leading-edge definitions of the parts were compared during
integration. The shape type uses the point farthest from the
trailing-edge midpoint. The flap tools split the surfaces at the
minimum-x point. Both definitions select the same stored point on all
clean integration inputs (NACA 0012, 2412 and 23012, before and after
repaneling), so no array function was changed. The spline tool locates
the leading edge by the zero distance gradient to the trailing-edge
midpoint, starting from the farthest input point, which refines the
same definition for the smooth curve.

## Implementation

- `ventorum.airfoil.geometry.AirfoilShape`: storage, `normalized`, `upper`, `lower`, `thickness`, `camber`, `max_thickness`, `max_camber`, `area`, `problems`, `validate`, and the unified methods `naca`, `from_file`, `repanel`, `repanel_cosine`, `with_flap`, `to_file`.
- `ventorum.airfoil.airfoil`: factory over designations, files, arrays and shapes.
- `ventorum.airfoil.io.parse_airfoil`, `read_airfoil`, `write_selig`, `write_lednicer`.
- `ventorum.airfoil.naca`: `naca4_thickness`, `naca4_camber`, `naca5_camber`, `naca4`, `naca5`, `naca`.
- `ventorum.airfoil.paneling.AirfoilSpline`, `repanel`, `repanel_cosine`.
- `ventorum.airfoil.flap.deflect_plain_flap`, `flap_hinge_point`.

## References

- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, 1995.
- D. Lednicer, "A CFD evaluation of wind tunnel test sections", AIAA paper 2010-0458, 2010.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959, chapter 6 (1959 McGraw-Hill edition, section 6.4 for the NACA formulas).
- E. N. Jacobs, K. E. Ward and R. M. Pinkerton, "The characteristics of 78 related airfoil sections from tests in the variable-density wind tunnel", NACA Report 460, 1933.
- E. N. Jacobs and R. M. Pinkerton, "Tests in the variable-density wind tunnel of related airfoils having the maximum camber unusually far forward", NACA Report 537, 1935.
- C. L. Ladson, C. W. Brooks, A. S. Hill and D. W. Sproles, "Computer program to obtain ordinates for NACA airfoils", NASA TM-4741, 1996.
- C. de Boor, *A Practical Guide to Splines*, Springer, 1978, Chapter IV (not-a-knot cubic spline).
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapter 11.
- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926, chapter VII (thin-airfoil theory of a hinged flap).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged flap", Aeronautical Research Committee, R&M 1095, 1927.
