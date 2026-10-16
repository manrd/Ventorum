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
from ventorum.airfoil.naca import (
    is_naca_designation,
    naca,
    naca4,
    naca4_camber,
    naca4_thickness,
    naca5,
    naca5_camber,
)
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


#: Keyword arguments of the NACA generator that the factory passes on.
_NACA_KEYWORDS = frozenset({"n_points", "spacing", "closed_te"})


def airfoil(
    spec: AirfoilShape | str | int | Path | np.ndarray, **kwargs: Any
) -> AirfoilShape:
    """Build an airfoil shape from a designation, a file, an array or a shape.

    The input kind follows these rules, in this order:

    1. An :class:`AirfoilShape` returns a copy.
    2. An integer, or a string of the NACA designation form (four or five
       digits with an optional "NACA" prefix, see
       :func:`ventorum.airfoil.naca.is_naca_designation`), calls the NACA
       generator. A NACA designation always wins over a file of the same
       name in the current folder.
    3. A :class:`pathlib.Path` reads that file. A string reads a file only
       when it names an existing file with an extension (for example
       "naca2412.dat").
    4. An (N, 2) array stores the points.

    Any other string raises ValueError.

    Parameters
    ----------
    spec : AirfoilShape or str or int or pathlib.Path or numpy.ndarray
        NACA designation [-], file path, point array [chord units] of
        shape (N, 2) in Selig order, or shape to copy.
    **kwargs : Any
        ``name`` [str] sets the shape name for any input kind. The NACA
        generator also takes ``n_points`` [-], ``spacing`` [-] and
        ``closed_te`` [-]. Other keywords raise TypeError.

    Returns
    -------
    AirfoilShape
        Shape with coordinates [chord units] in Selig order.

    Raises
    ------
    TypeError
        If a keyword does not apply to the input kind (for example
        ``n_points`` with a file or an array).
    ValueError
        If a string is neither a NACA designation nor an existing file
        with an extension, a path does not exist, or the array has a
        bad shape.
    """
    name = kwargs.pop("name", None)
    if isinstance(spec, bool):
        raise ValueError(f"Invalid airfoil spec {spec!r}.")
    naca_input = isinstance(spec, (int, np.integer)) or (
        isinstance(spec, str) and is_naca_designation(spec)
    )
    allowed = _NACA_KEYWORDS if naca_input else frozenset()
    extra = sorted(set(kwargs) - allowed)
    if extra:
        kind = "a NACA designation" if naca_input else f"an input of type {type(spec).__name__}"
        raise TypeError(
            f"airfoil() got keyword arguments {extra} that do not apply to {kind}; "
            "only 'name' applies to files, arrays and shapes."
        )
    if isinstance(spec, AirfoilShape):
        out = AirfoilShape(name=spec.name, coords=np.array(spec.coords, dtype=float, copy=True))
    elif naca_input:
        out = AirfoilShape.naca(spec if isinstance(spec, str) else int(spec), **kwargs)
    elif isinstance(spec, Path):
        if not spec.is_file():
            raise ValueError(f"Airfoil file not found: {spec!r}.")
        out = read_airfoil(spec)
    elif isinstance(spec, str):
        candidate = Path(spec)
        if candidate.suffix and candidate.is_file():
            out = read_airfoil(candidate)
        else:
            raise ValueError(
                f"Invalid airfoil spec {spec!r}: not a NACA designation (four or five "
                "digits with an optional 'NACA' prefix) and not an existing file with "
                "an extension. Give a file as a pathlib.Path or as a name such as "
                "'section.dat'."
            )
    else:
        arr = np.asarray(spec, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Airfoil array must have shape (N, 2); got {arr.shape}.")
        out = AirfoilShape(name="airfoil", coords=arr)
    if name is not None:
        out.name = str(name)
    return out
