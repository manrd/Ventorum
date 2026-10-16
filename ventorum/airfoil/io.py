# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Readers and writers of airfoil coordinate files (Selig, Lednicer, plain).

Format detection
----------------
Selig files hold an optional name line and then x y pairs in Selig order
(from the trailing edge along the upper surface to the leading edge and back
along the lower surface). Lednicer files hold a name line, a line with the
point counts of the upper and lower surfaces, the upper surface from the
leading edge to the trailing edge, a blank line, and the lower surface from
the leading edge to the trailing edge. A Lednicer file without the name line
is detected when its first line holds two whole numbers of at least 2 whose
sum equals the number of data lines that follow. Plain files hold x y pairs
with no header. Separators are spaces, tabs or commas. Lines that start with ``#``
are comments. A file in reverse order (lower surface first) is reversed to
Selig order. Duplicate leading-edge points of Lednicer files are merged.

References
----------
- M. S. Selig, J. J. Guglielmo, A. P. Broeren and P. Giguere, *Summary of
  Low-Speed Airfoil Data*, Vol. 1, SoarTech Publications, 1995 (Selig
  coordinate file format).
- D. Lednicer, "A CFD evaluation of wind tunnel test sections", AIAA
  paper 2010-0458, 2010 (Lednicer coordinate file format).
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover,
  1959, chapter 6 (section geometry).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ventorum.airfoil.geometry import AirfoilGeometryError, AirfoilShape


def _numbers(line: str) -> list[float] | None:
    """Parse *line* as numbers, or return None when it holds text."""
    tokens = line.replace(",", " ").split()
    if not tokens:
        return None
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _filtered_lines(text: str) -> list[tuple[int, str]]:
    """Return the non-blank, non-comment lines with their line numbers."""
    out: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((number, stripped))
    return out


def parse_airfoil(text: str, name: str | None = None) -> AirfoilShape:
    """Parse airfoil coordinates from a string and detect the format.

    Parameters
    ----------
    text : str
        Full content of the file.
    name : str or None
        Name of the shape. An explicit value wins over the header line.
        Without a value, the header line is used, or ``"airfoil"`` when
        the file has no header.

    Returns
    -------
    AirfoilShape
        Shape with coordinates [chord units] in Selig order.

    Raises
    ------
    AirfoilGeometryError
        If a data line holds one number or text, or the file holds fewer
        than 3 points.
    """
    kept = _filtered_lines(text)
    if not kept:
        raise AirfoilGeometryError("Cannot parse airfoil: fewer than 3 points (no data lines).")
    first_values = _numbers(kept[0][1])
    header: str | None = None
    if first_values is not None and len(first_values) == 2:
        data_lines = kept
        if _is_lednicer_counts(first_values, len(kept) - 1):
            return _parse_lednicer(None, data_lines, name)
    else:
        header = kept[0][1]
        data_lines = kept[1:]
        if not data_lines:
            raise AirfoilGeometryError(
                f"Cannot parse airfoil {header!r}: fewer than 3 points (no data lines)."
            )
        second_values = _numbers(data_lines[0][1])
        if second_values is not None and len(second_values) == 2:
            if second_values[0] > 1.0 and second_values[1] > 1.0:
                return _parse_lednicer(header, data_lines, name)
    coords = _parse_point_lines(data_lines, header)
    if coords.shape[0] < 3:
        who = header if header is not None else "airfoil"
        raise AirfoilGeometryError(
            f"Cannot parse airfoil {who!r}: fewer than 3 points (got {coords.shape[0]})."
        )
    coords = _to_selig_order(coords)
    final_name = name if name is not None else (header if header is not None else "airfoil")
    return AirfoilShape(name=final_name, coords=coords)


def _parse_point_lines(data_lines: list[tuple[int, str]], header: str | None) -> np.ndarray:
    """Parse Selig or plain data lines into a point array [chord units]."""
    pts: list[list[float]] = []
    for number, line in data_lines:
        values = _numbers(line)
        if values is None:
            raise AirfoilGeometryError(f"Line {number}: text in the data ({line!r}).")
        if len(values) == 1:
            raise AirfoilGeometryError(f"Line {number}: one number, need x and y ({line!r}).")
        if len(values) != 2:
            raise AirfoilGeometryError(f"Line {number}: need x and y pairs ({line!r}).")
        pts.append([values[0], values[1]])
    return np.array(pts, dtype=np.float64)


def _is_lednicer_counts(values: list[float], n_following: int) -> bool:
    """Return True when a first line of two numbers is a Lednicer counts line.

    A Lednicer file without a name line starts with the counts line. The
    line holds two whole numbers of at least 2, and their sum equals the
    number of data lines that follow. A data point that only looks like
    counts (for example a first point (6, 6) of a file in other units)
    fails the sum check and stays a data point.
    """
    a, b = values
    whole = a == round(a) and b == round(b)
    return bool(whole and a >= 2.0 and b >= 2.0 and int(a) + int(b) == n_following)


def _parse_lednicer(
    header: str | None, data_lines: list[tuple[int, str]], name: str | None
) -> AirfoilShape:
    """Parse a Lednicer block (counts line first) into Selig order.

    *header* is the name line, or None for a file without a name line.
    """
    who = header if header is not None else "airfoil"
    count_values = _numbers(data_lines[0][1])
    assert count_values is not None and len(count_values) == 2
    n_up = int(round(count_values[0]))
    n_low = int(round(count_values[1]))
    if n_up < 2 or n_low < 2:
        raise AirfoilGeometryError(
            f"Cannot parse airfoil {who!r}: bad Lednicer counts ({data_lines[0][1]!r})."
        )
    point_lines = data_lines[1:]
    if len(point_lines) != n_up + n_low:
        raise AirfoilGeometryError(
            f"Cannot parse airfoil {who!r}: Lednicer counts say "
            f"{n_up} + {n_low} points but found {len(point_lines)} data lines."
        )
    pts = _parse_point_lines(point_lines, header)
    upper = pts[:n_up]
    lower = pts[n_up:]
    if np.linalg.norm(upper[0] - lower[0]) <= 1e-12:
        lower = lower[1:]
    selig = np.vstack([upper[::-1], lower])
    selig = _to_selig_order(selig)
    final_name = name if name is not None else who
    return AirfoilShape(name=final_name, coords=selig)


def _to_selig_order(coords: np.ndarray) -> np.ndarray:
    """Reverse the points when they run clockwise (lower surface first)."""
    x = coords[:, 0]
    y = coords[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area < 0.0:
        return coords[::-1].copy()
    return coords


def read_airfoil(path: str | Path) -> AirfoilShape:
    """Read an airfoil file in Selig, Lednicer or plain format.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the file.

    Returns
    -------
    AirfoilShape
        Shape with coordinates [chord units] in Selig order. The name comes
        from the header line, or from the file stem when the file has no
        header line.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    shape = parse_airfoil(text)
    kept = _filtered_lines(text)
    if kept:
        first_values = _numbers(kept[0][1])
        if first_values is not None and len(first_values) == 2:
            shape = AirfoilShape(name=file_path.stem, coords=shape.coords)
    else:
        shape = AirfoilShape(name=file_path.stem, coords=shape.coords)
    return shape


def write_selig(shape: AirfoilShape, path: str | Path, precision: int = 6) -> None:
    """Write a shape in Selig format (name line, then x y pairs).

    Parameters
    ----------
    shape : AirfoilShape
        Shape to write (coordinates [chord units]).
    path : str or pathlib.Path
        Output file path.
    precision : int
        Digits after the decimal point.
    """
    file_path = Path(path)
    lines = [shape.name]
    fmt = f"{{:.{int(precision)}f}} {{:.{int(precision)}f}}"
    for x, y in shape.coords.tolist():
        lines.append(fmt.format(x, y))
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lednicer(shape: AirfoilShape, path: str | Path, precision: int = 6) -> None:
    """Write a shape in Lednicer format (counts line, upper, blank, lower).

    Parameters
    ----------
    shape : AirfoilShape
        Shape to write (coordinates [chord units]).
    path : str or pathlib.Path
        Output file path.
    precision : int
        Digits after the decimal point.
    """
    file_path = Path(path)
    upper = shape.upper()
    lower = shape.lower()
    fmt = f"{{:.{int(precision)}f}} {{:.{int(precision)}f}}"
    lines = [shape.name, f"{int(upper.shape[0])} {int(lower.shape[0])}"]
    for x, y in upper.tolist():
        lines.append(fmt.format(x, y))
    lines.append("")
    for x, y in lower.tolist():
        lines.append(fmt.format(x, y))
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
