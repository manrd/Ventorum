---
orphan: true
---

# Airfoil geometry

This chapter gives the coordinate convention, the file formats and the geometry checks of the 2D airfoil tools.

## Coordinate convention

A section is a numpy array of shape (N, 2), float64, in Selig order: from the trailing edge along the upper surface to the leading edge, then along the lower surface back to the trailing edge. The order is counterclockwise, so the signed shoelace area is positive. Unit chord has the leading edge at (0, 0) and the trailing-edge midpoint at (1, 0). A closed trailing edge repeats the same point at the start and the end. The `normalized` method moves any shape to this frame: it translates the leading edge to the origin, rotates the trailing-edge midpoint onto the x axis and scales the chord to one.

The `upper` and `lower` methods each return one surface from the leading edge to the trailing edge, both with the leading-edge point first. The leading edge is the stored point farthest from the trailing-edge midpoint (`ventorum.airfoil.geometry.leading_edge_index`); this is the one leading-edge definition of the package, and {doc}`airfoils` explains where each tool uses it. Thickness is the upper y minus the lower y at one chord station, and camber is the mean of the two, both by linear interpolation on each surface.

## File formats

Three plain-text formats are read. The reader detects the format from the first lines.

- Selig: an optional name line, then x y pairs in Selig order (M. S. Selig and co-workers, *Summary of Low-Speed Airfoil Data*, 1995).
- Lednicer: a name line, a line with the point counts of the upper and lower surfaces (two numbers larger than 1, for example `61 61`), then the upper surface from the leading edge to the trailing edge, a blank line, and the lower surface from the leading edge to the trailing edge (D. Lednicer, AIAA paper 2010-0458, 2010). The reader merges the duplicated leading-edge point. A Lednicer file without the name line is also read: its first line holds two whole numbers of at least 2 whose sum equals the number of data lines that follow. Two numbers that fail this count stay a data point.
- Plain: x y pairs with no header. The name then comes from the file stem.

Separators are spaces, tabs or commas. Lines that start with `#` are comments and blank lines are skipped. A file in reverse order (lower surface first, clockwise) is reversed to Selig order, so its signed area is positive.

The writers store the same point order with a fixed number of digits (`precision`, default 6). A file written and read again returns the same points to the written precision.

## Geometry checks

`problems` lists the defects of a shape and is empty when the shape is good:

- fewer than 10 points;
- repeated consecutive points (distance below 1e-12 chord units);
- clockwise order (negative signed area);
- self-intersection of the polygon (segment pair test on the closed ring, the trailing-edge edge from the last to the first point included; edges that cross or touch fail);
- a surface whose x is not monotonic (a note, not an error).

`validate` raises `AirfoilGeometryError` when any hard defect is present. The non-monotonic x notes do not raise, because a real section can have a locally vertical tangent near the leading edge.

## Implementation

- `ventorum.airfoil.geometry.AirfoilShape`: storage, `normalized`, `upper`, `lower`, `thickness`, `camber`, `max_thickness`, `max_camber`, `area`, `problems`, `validate`.
- `ventorum.airfoil.io.parse_airfoil`, `read_airfoil`, `write_selig`, `write_lednicer`.

## References

- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, 1995.
- D. Lednicer, "A CFD evaluation of wind tunnel test sections", AIAA paper 2010-0458, 2010.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959, chapter 6.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapter 11.
