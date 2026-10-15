# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""One public interface over airfoil readers, NACA generators, paneling and flaps.

The array functions stay public and unchanged. The shape methods call
them and return new shapes. The factory :func:`airfoil` builds a shape
from a NACA designation, a file path, a point array or a shape.

Coordinate convention: a numpy array of shape (N, 2), float64, in Selig
order: from the trailing edge along the upper surface to the leading
edge, then along the lower surface back to the trailing edge
(counterclockwise, positive signed area). Unit chord has the leading
edge at (0, 0) and the trailing-edge midpoint at (1, 0), unless the
function says otherwise. A closed trailing edge repeats the same point
at the start and the end.

Sources: Selig and co-workers, *Summary of Low-Speed Airfoil Data*,
1995 (Selig file format); Lednicer, AIAA paper 2010-0458, 2010
(Lednicer file format); Abbott and von Doenhoff, *Theory of Wing
Sections*, 1959, section 6.4 (NACA thickness and 4-digit mean line);
Jacobs, Ward and Pinkerton, NACA Report 460, 1933, and Jacobs and
Pinkerton, NACA Report 537, 1935 (5-digit series); Ladson, Brooks, Hill
and Sproles, NASA TM-4741, 1996 (closed trailing edge coefficient);
de Boor, *A Practical Guide to Splines*, 1978, Chapter IV (not-a-knot
cubic spline); Glauert, *The Elements of Aerofoil and Airscrew Theory*,
1926, chapter VII, and Glauert, R&M 1095, 1927 (hinged plain flap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ventorum.airfoil.flap import deflect_plain_flap, flap_hinge_point
from ventorum.airfoil.geometry import AirfoilGeometryError, AirfoilShape
from ventorum.airfoil.io import parse_airfoil, read_airfoil, write_lednicer, write_selig
from ventorum.airfoil.naca import naca, naca4, naca4_camber, naca4_thickness, naca5, naca5_camber
from ventorum.airfoil.paneling import AirfoilSpline, repanel, repanel_cosine

__all__ = [
    "AirfoilGeometryError",
    "AirfoilShape",
    "AirfoilSpline",
    "airfoil",
    "deflect_plain_flap",
    "flap_hinge_point",
    "naca",
    "naca4",
    "naca4_camber",
    "naca4_thickness",
    "naca5",
    "naca5_camber",
    "parse_airfoil",
    "read_airfoil",
    "repanel",
    "repanel_cosine",
    "write_lednicer",
    "write_selig",
]


def airfoil(
    spec: AirfoilShape | str | int | Path | np.ndarray, **kwargs: Any
) -> AirfoilShape:
    """Build an airfoil shape from a designation, a file, an array or a shape.

    A NACA designation string (for example "2412" or "NACA 23012") calls
    the NACA generator. An integer designation (for example 2412) works
    the same way. A file path (string or Path) reads the file when
    the path exists. An (N, 2) array stores the points. A shape returns
    a copy. The array functions stay available for direct use.

    Parameters
    ----------
    spec : AirfoilShape or str or int or pathlib.Path or numpy.ndarray
        NACA designation [-], file path, point array [chord units] of
        shape (N, 2) in Selig order, or shape to copy.
    **kwargs : Any
        Extra arguments of the NACA generator (``n_points`` [-],
        ``spacing`` [-], ``closed_te`` [-]) when ``spec`` is a NACA
        designation. ``name`` [str] sets the shape name for any input
        kind.

    Returns
    -------
    AirfoilShape
        Shape with coordinates [chord units] in Selig order.

    Raises
    ------
    ValueError
        If a string is neither a readable file path nor a valid NACA
        designation, or the array has a bad shape.
    """
    name = kwargs.pop("name", None)
    if isinstance(spec, AirfoilShape):
        out = AirfoilShape(name=spec.name, coords=np.array(spec.coords, dtype=float, copy=True))
        if name is not None:
            out.name = str(name)
        return out
    if isinstance(spec, bool):
        raise ValueError(f"Invalid airfoil spec {spec!r}.")
    if isinstance(spec, (int, np.integer)):
        shape = AirfoilShape.naca(int(spec), **kwargs)
        if name is not None:
            shape.name = str(name)
        return shape
    if isinstance(spec, (str, Path)):
        candidate = Path(spec)
        if candidate.exists() and candidate.is_file():
            shape = read_airfoil(candidate)
            if name is not None:
                shape.name = str(name)
            return shape
        if isinstance(spec, Path):
            raise ValueError(f"Airfoil file not found: {spec!r}.")
        text = str(spec)
        try:
            shape = AirfoilShape.naca(text, **kwargs)
        except ValueError:
            raise ValueError(
                f"Invalid airfoil spec {spec!r}: no such file and not a NACA designation."
            ) from None
        if name is not None:
            shape.name = str(name)
        return shape
    arr = np.asarray(spec, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Airfoil array must have shape (N, 2); got {arr.shape}.")
    shape_name = str(name) if name is not None else "airfoil"
    return AirfoilShape(name=shape_name, coords=arr)
