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
first. Thickness is the upper y minus the lower y at one chord station,
and camber is the mean of the two, both by linear interpolation on each
surface.

## Leading edge

The package has one definition of the leading edge: the point of the
contour farthest from the trailing-edge midpoint (the midpoint of the
first and the last point). Every tool uses it, at the resolution that
the tool works with:

- On a point polygon, the leading edge is the stored point of largest
  distance (`ventorum.airfoil.geometry.leading_edge_index`).
  `AirfoilShape.leading_edge_index`, `upper`, `lower`, `normalized` and
  the surface split of `flap_hinge_point` and `deflect_plain_flap` use
  this point.
- On the parametric spline, the leading edge is the continuous maximum
  of the same distance, where the tangent is normal to the line to the
  trailing-edge midpoint (`AirfoilSpline.s_leading_edge`). The root
  search starts from the stored point above and accepts only a change
  from moving away to moving back (a maximum). A curve without such a
  maximum raises an error instead of a silent fallback. `repanel` and
  `repanel_cosine` put a node at this point.

On a clean section the stored point and the minimum-x point agree. On a
deflected section they can differ: the trailing-edge midpoint moves, so
the farthest point moves up the nose. The flap tools therefore split the
surfaces at the farthest point and accept a short run of decreasing x
forward of the hinge.

The paneling functions accept coordinates in any consistent length
unit. A physical wing section uses m. A unit section uses chord units.
The shape methods of this chapter use chord units throughout.

## File formats

Three plain-text formats are read. The reader detects the format from
the first lines.

- Selig: an optional name line, then x y pairs in Selig order (M. S. Selig and co-workers, *Summary of Low-Speed Airfoil Data*, 1995).
- Lednicer: a name line, a line with the point counts of the upper and lower surfaces (two numbers larger than 1, for example `61 61`), then the upper surface from the leading edge to the trailing edge, a blank line, and the lower surface from the leading edge to the trailing edge (D. Lednicer, AIAA paper 2010-0458, 2010). The reader merges the duplicated leading-edge point. A file without the name line is read when its first line holds two whole numbers of at least 2 whose sum equals the number of data lines that follow; two numbers that fail this count stay a data point.
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
- self-intersection of the polygon (segment pair test on the closed ring, the trailing-edge edge included; edges that cross or touch fail);
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

The parameters for design lift 0.3 (first digit 2) come from the
five-digit mean-line table of Abbott and von Doenhoff (1959, section 6.5,
NACA five-digit wing sections):

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

## NACA designations

The generator accepts four or five digits, with an optional "NACA"
prefix in any letter case and optional spaces between the prefix and
the digits: "2412", "NACA 2412", "naca2412", "23012", "NACA 23012", or
the integer 2412. Hyphens, underscores and spaces inside the digits
raise an error. A designation with zero camber and a nonzero position
digit ("0412", "01012", "03012") gives the symmetric section of the same
thickness with a warning, because its position digits have no effect.

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
length ratio stays at most 1.5. The limit of the smoothing is the
uniform weight; when even the uniform weight cannot meet the 1.5 bound
(very few panels), the function warns and returns the nodes. The
fit-and-equidistribute pass repeats until the set stops moving, so
repaneling an already repaneled set returns the same nodes; it fits a
spline through the nodes, so `n_panels` is at least 3. A closed trailing
edge stays closed; an open trailing edge keeps its two end points
exactly.

The accuracy of the nodes depends on the input sampling. For an ellipse
of thickness ratio 0.12 sampled uniformly in eccentric angle (721
points), the nodes of 160 panels lie on the exact ellipse within 1e-6
chord. For the coarse input of 61 uniform-in-x points per surface, the
spline cannot follow the blunt ends of the ellipse, where the slope is
infinite and the input points are far apart along the curve. The test
`test_coarse_ellipse_accuracy_limit` measures node distances of 1.3e-3
chord near the blunt trailing end (x > 0.95), 3.0e-4 chord near the
leading edge (x < 0.05) and 1.2e-5 chord between, and checks the bounds
1.5e-3, 4e-4 and 2e-5 chord. Sections with a sharp trailing edge and
denser input points near the leading edge are less sensitive.

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

Hinge height. This is the one definition of the package: `y_hinge` is
a fraction of the local thickness at the hinge station `x_hinge`, from
the lower surface (0) to the upper surface (1). The default 0.5 is the
mid-thickness point. The hinge point is
$(x_h,\; y_l(x_h) + y_{hinge}\,[y_u(x_h) - y_l(x_h)])$ with the upper and
lower surface heights $y_u$ and $y_l$ at $x_h$. `flap_hinge_point`,
`deflect_plain_flap` and `AirfoilShape.with_flap` all use it.

The construction has five steps:

1. Hinge point. The function `flap_hinge_point` puts the hinge at the station `x_hinge` in chord units, at the height `y_hinge` above. The surfaces are split at the leading edge of the package definition.
2. Rotation. Every point with x past `x_hinge` rotates about the hinge by the deflection angle, as a rigid body. The trailing-edge points of the output are exactly the rotated input points.
3. Compressed side. The sign of the deflection selects it: the lower surface for trailing edge down, the upper surface for trailing edge up. The moved surface overlaps the fixed surface there. Both are cut at the crossing of their polylines nearest to the hinge, and the overlapping points are removed. When the polylines do not cross (on coarse input the moved surface can pass the fixed corner), both parts are cut at the hinge line, the line through the hinge point normal to the chord, and joined there.
4. Stretched side. On the other side a gap opens between the fixed and the moved surface. A circular arc about the hinge fills the gap, from the last fixed point to the first moved point. The arc radius is the distance from the hinge to that surface at `x_hinge`. The point count `n_arc` includes both end points; the default count keeps the arc panel lengths near the local panel length.
5. Check. The output must be a simple polygon with a positive area; otherwise the function raises `AirfoilGeometryError`.

Hinge treatment. The option `hinge_treatment` selects what happens to
the nodes near the hinge after the construction: "none" (default) keeps
the outline as constructed; "merge" removes nodes near the hinge until
no panel there is shorter than half the mean length of its neighbours
and no isolated sharp corner remains, and the kept nodes do not move;
"repanel" fits a spline through the outline and distributes the nodes
again along it, with nodes kept at the hinge on both surfaces, at the
leading edge and at the trailing edge (panel count of the input, or
`n_panels`). The page {doc}`airfoil_flaps` gives the rules, the limits
and the tested deviation bound of "repanel".

A zero deflection returns a copy of the input. The input must be in
Selig order. The hinge station lies in (0.05, 0.98) chord units. The
deflection satisfies |deflection| <= 60 deg. The hinge height fraction
lies in [0, 1]. The area changes slightly, and the filler arc, the cut
and the hinge-line joint are geometric blends without viscous or gap
physics. A sweep over NACA 0006, 0012, 2412 and 4415 with 31, 61 and 161
points, hinges from 0.1 to 0.97, deflections from -60 to 60 deg and hinge
heights from 0 to 1 checks that every output is a simple polygon with a
positive area.

## Unified interface

Users and solvers get airfoils one way, through `AirfoilShape` and the
`airfoil` factory. The array functions stay public and unchanged.

- `AirfoilShape.naca(designation, **kw)` builds a NACA section.
- `AirfoilShape.from_file(path)` reads a Selig, Lednicer or plain file.
- `shape.repanel(n_panels=160, **kw)` redistributes the nodes.
- `shape.repanel_cosine(n_per_side)` resamples with cosine spacing.
- `shape.with_flap(x_hinge, deflection, y_hinge=0.5, hinge_treatment="repanel")` deflects a plain flap. The result holds the reference geometry of the section before the deflection (see below).
- `shape.to_file(path, fmt="selig")` writes a Selig or Lednicer file.
- `airfoil(spec)` accepts a NACA designation string, a file path, an (N, 2) array or a shape, and returns a shape.

The factory applies these rules in order. A shape returns a copy. An
integer, or a string of the NACA designation form (four or five digits
with an optional "NACA" prefix), builds the NACA section; a designation
always wins over a file of the same name in the current folder. A
`pathlib.Path` reads that file. A string reads a file only when it names
an existing file with an extension (for example "naca2412.dat"). An
(N, 2) array stores the points. Any other string raises an error. The
keyword `name` applies to every input; the generator keywords
`n_points`, `spacing` and `closed_te` apply only to a NACA designation,
and any keyword that does not apply raises TypeError.

## Reference geometry

A shape can hold a reference geometry, `shape.reference`, a
`SectionReference`: the leading edge and the trailing-edge midpoint of
the chord line and the reference chord (by default the distance between
the two points), in chord units. Its quarter-chord point lies 0.25 of
the way from the leading edge to the trailing-edge midpoint. The 2D
solvers take the default reference chord, moment reference point and
pivot from it; explicit `chord_ref` and `moment_ref` arguments win.

- `with_flap` stores the reference geometry of the section before the
  deflection: the stored one, or the one derived from the points. A
  plain flap moves the trailing edge, so the deflected outline is
  shorter than the undeflected chord and its chord line is tilted; the
  reference of thin-airfoil theory is the undeflected chord (Glauert
  1927).
- `repanel`, `repanel_cosine` and the factory copy keep the stored
  reference geometry; `normalized` maps it with the points.
- A shape without a stored reference geometry derives it from its points
  (`shape.reference_geometry()`, `SectionReference.from_coords`): the
  leading edge of the package definition, the trailing-edge midpoint and
  their distance.
- The Selig and Lednicer files hold the points only: the writers do not
  store the reference geometry.

## Implementation

- `ventorum.airfoil.geometry.AirfoilShape`: storage, `normalized`, `upper`, `lower`, `thickness`, `camber`, `max_thickness`, `max_camber`, `area`, `problems`, `validate`, `reference_geometry`, and the unified methods `naca`, `from_file`, `repanel`, `repanel_cosine`, `with_flap`, `to_file`.
- `ventorum.airfoil.geometry.SectionReference`: reference geometry (chord line and reference chord), `from_coords`, `quarter_chord`.
- `ventorum.airfoil.geometry.leading_edge_index` (the leading-edge definition) and `polygon_self_intersection` (the simple-polygon test).
- `ventorum.airfoil.airfoil`: factory over designations, files, arrays and shapes.
- `ventorum.airfoil.io.parse_airfoil`, `read_airfoil`, `write_selig`, `write_lednicer`.
- `ventorum.airfoil.naca`: `naca4_thickness`, `naca4_camber`, `naca5_camber`, `naca4`, `naca5`, `naca`, `is_naca_designation`.
- `ventorum.airfoil.paneling.AirfoilSpline`, `repanel`, `repanel_cosine`.
- `ventorum.airfoil.flap.deflect_plain_flap`, `flap_hinge_point`.

## References

- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, 1995.
- D. Lednicer, "A CFD evaluation of wind tunnel test sections", AIAA paper 2010-0458, 2010.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959, chapter 6 (1959 McGraw-Hill edition, section 6.4 for the thickness distribution and the 4-digit mean line, section 6.5 for the 5-digit mean lines and the table of $m$ and $k_1$).
- E. N. Jacobs, K. E. Ward and R. M. Pinkerton, "The characteristics of 78 related airfoil sections from tests in the variable-density wind tunnel", NACA Report 460, 1933.
- E. N. Jacobs and R. M. Pinkerton, "Tests in the variable-density wind tunnel of related airfoils having the maximum camber unusually far forward", NACA Report 537, 1935.
- C. L. Ladson, C. W. Brooks, A. S. Hill and D. W. Sproles, "Computer program to obtain ordinates for NACA airfoils", NASA TM-4741, 1996.
- C. de Boor, *A Practical Guide to Splines*, Springer, 1978, Chapter IV (not-a-knot cubic spline).
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapter 11.
- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge University Press, 1926, chapter VII (thin-airfoil theory of a hinged flap).
- H. Glauert, "Theoretical relationships for an aerofoil with hinged flap", Aeronautical Research Committee, R&M 1095, 1927.
