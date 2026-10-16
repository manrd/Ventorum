# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""NACA 4-digit and 5-digit airfoil generators.

Build unit-chord airfoil coordinates from the published thickness and mean
line formulas. The thickness distribution follows Abbott and von Doenhoff
(1959, section 6.4) with the finite trailing edge coefficient of Ladson,
Brooks, Hill and Sproles (NASA TM-4741, 1996). The 4-digit mean line follows
Abbott and von Doenhoff (1959, section 6.4). The standard (non-reflex)
5-digit mean lines 210, 220, 230, 240 and 250 use the parameters m and k1
of the five-digit mean-line table of Abbott and von Doenhoff (1959,
section 6.5, NACA five-digit wing sections), given there for the design
lift coefficient 0.3. The design lift scales k1 linearly with the first
digit as Jacobs, Ward and Pinkerton (NACA Report 460, 1933) and Jacobs and
Pinkerton (NACA Report 537, 1935) define it.

Designations
------------
The accepted forms are four or five digits, with an optional "NACA"
prefix in any letter case and optional spaces between the prefix and the
digits: "2412", "NACA 2412", "naca2412", "23012", "NACA 23012", or the
integer 2412. Other separators (hyphens, underscores, spaces inside the
digits) raise ValueError. A designation with zero camber and a nonzero
position digit ("0412", "01012", "03012") is accepted with a UserWarning:
the mean line is flat, so the position digits have no effect and the
section equals the symmetric section of the same thickness.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

__all__ = [
    "is_naca_designation",
    "naca",
    "naca4",
    "naca4_camber",
    "naca4_thickness",
    "naca5",
    "naca5_camber",
]

#: Thickness polynomial coefficients of Abbott and von Doenhoff (1959).
_A0 = 0.2969
_A1 = -0.1260
_A2 = -0.3516
_A3 = 0.2843
_A4_OPEN = -0.1015

#: Last coefficient for a closed trailing edge (Ladson et al., 1996).
_A4_CLOSED = -0.1036

#: Standard 5-digit mean lines: position code to (m, k1 for design lift 0.3),
#: from the five-digit mean-line table of Abbott and von Doenhoff (1959,
#: section 6.5).
_FIVE_DIGIT_TABLE = {
    "10": (0.0580, 361.40),
    "20": (0.1260, 51.64),
    "30": (0.2025, 15.957),
    "40": (0.2900, 6.643),
    "50": (0.3910, 3.230),
}


#: Accepted designation text: optional "NACA" prefix, then the digits.
_DESIGNATION = re.compile(r"^\s*(?:NACA\s*)?(\d+)\s*$", re.IGNORECASE)


def _clean(designation: str | int) -> str:
    """Return the digit string of a designation.

    Parameters
    ----------
    designation : str or int
        Designation such as "NACA 2412", "naca2412", "2412" or 2412. The
        accepted forms are listed in the module docstring.

    Returns
    -------
    str
        Digits only, for example "2412".
    """
    if isinstance(designation, bool):
        raise ValueError(f"Invalid NACA designation: {designation!r}.")
    if isinstance(designation, (int, np.integer)):
        text = str(int(designation))
    elif isinstance(designation, str):
        text = designation
    else:
        raise ValueError(f"Invalid NACA designation: {designation!r}.")
    match = _DESIGNATION.match(text)
    if match is None:
        raise ValueError(
            f"Invalid NACA designation: {designation!r}. Use four or five digits "
            "with an optional 'NACA' prefix, for example 'NACA 2412' or '23012'."
        )
    return match.group(1)


def is_naca_designation(spec: object) -> bool:
    """Return True when *spec* has the form of a 4-digit or 5-digit designation.

    The test checks the form only (an optional "NACA" prefix and four or
    five digits, see the module docstring). A string of this form can
    still be an invalid section, for example "1012" (camber without a
    position), which :func:`naca` rejects.

    Parameters
    ----------
    spec : object
        Candidate designation [-]: a string or an integer.

    Returns
    -------
    bool
        True for a string or an integer of the designation form.
    """
    if isinstance(spec, bool):
        return False
    if isinstance(spec, (int, np.integer)):
        return 1000 <= int(spec) <= 99999
    if not isinstance(spec, str):
        return False
    match = _DESIGNATION.match(spec)
    return match is not None and len(match.group(1)) in (4, 5)


def _warn_flat_mean_line(designation: str | int, digits: str) -> None:
    """Warn that a zero-camber designation has a nonzero position digit."""
    warnings.warn(
        f"NACA designation {designation!r} has zero camber and a nonzero position "
        f"digit ({digits}): the mean line is flat, so the position digits have no "
        "effect and the section is the symmetric section of the same thickness.",
        UserWarning,
        stacklevel=3,
    )


def naca4_thickness(
    x: np.ndarray | list[float] | float,
    t: float,
    closed_te: bool = False,
) -> np.ndarray:
    """Return the half thickness distribution of a NACA 4-digit airfoil.

    Parameters
    ----------
    x : array_like
        Chordwise stations as a fraction of chord [-], with values in [0, 1].
    t : float
        Maximum thickness ratio [-], for example 0.12.
    closed_te : bool
        Use the closed trailing edge coefficient -0.1036. Default False.

    Returns
    -------
    numpy.ndarray
        Half thickness y_t as a fraction of chord [-].
    """
    xc = np.asarray(x, dtype=float)
    if float(t) < 0.0:
        raise ValueError(f"Thickness ratio must be zero or positive, got {t!r}.")
    if np.any(xc < 0.0) or np.any(xc > 1.0):
        raise ValueError("Stations x must lie in [0, 1].")
    a4 = _A4_CLOSED if closed_te else _A4_OPEN
    yt = (
        5.0
        * float(t)
        * (
            _A0 * np.sqrt(xc)
            + _A1 * xc
            + _A2 * xc**2
            + _A3 * xc**3
            + a4 * xc**4
        )
    )
    yt = np.asarray(yt, dtype=float)
    if closed_te:
        yt = np.where(xc == 1.0, 0.0, yt)
    return yt


def naca4_camber(
    x: np.ndarray | list[float] | float,
    m: float,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean line and slope of a NACA 4-digit airfoil.

    Parameters
    ----------
    x : array_like
        Chordwise stations as a fraction of chord [-], with values in [0, 1].
    m : float
        Maximum mean line height as a fraction of chord [-].
    p : float
        Position of the maximum mean line height [-], in [0, 1].

    Returns
    -------
    y_c : numpy.ndarray
        Mean line height as a fraction of chord [-].
    dyc_dx : numpy.ndarray
        Mean line slope dy_c/dx [-].
    """
    xc = np.asarray(x, dtype=float)
    mm = float(m)
    pp = float(p)
    if np.any(xc < 0.0) or np.any(xc > 1.0):
        raise ValueError("Stations x must lie in [0, 1].")
    if not 0.0 <= pp <= 1.0:
        raise ValueError(f"Camber position p must lie in [0, 1], got {p!r}.")
    if mm == 0.0:
        zeros = np.zeros_like(xc, dtype=float)
        return zeros, zeros.copy()
    if pp == 0.0 or pp == 1.0:
        raise ValueError("Camber position p of 0 or 1 needs m = 0.")
    front = xc < pp
    yc = np.where(
        front,
        mm / pp**2 * (2.0 * pp * xc - xc**2),
        mm / (1.0 - pp) ** 2 * ((1.0 - 2.0 * pp) + 2.0 * pp * xc - xc**2),
    )
    dyc = np.where(
        front,
        2.0 * mm / pp**2 * (pp - xc),
        2.0 * mm / (1.0 - pp) ** 2 * (pp - xc),
    )
    return np.asarray(yc, dtype=float), np.asarray(dyc, dtype=float)


def _five_digit_parameters(prefix: str, name: str | int | None = None) -> tuple[float, float]:
    """Return the mean line parameters m and k1 of a 3-digit prefix.

    Parameters
    ----------
    prefix : str
        Three digits such as "230".
    name : str or int or None
        Designation as the caller gave it, for the warning text. None
        uses the prefix.

    Returns
    -------
    tuple[float, float]
        Mean line parameters m [-] and k1 [-].
    """
    if len(prefix) != 3 or not prefix.isdigit():
        raise ValueError(f"Invalid 5-digit mean line: {prefix!r}.")
    if prefix[2] == "1":
        raise ValueError(
            f"Reflex 5-digit mean line {prefix!r} is not available: "
            "no published table of k1 and k2/k1 is loaded. "
            "Use a standard mean line 210, 220, 230, 240 or 250."
        )
    pos = prefix[1:3]
    lift_digit = int(prefix[0])
    if pos not in _FIVE_DIGIT_TABLE:
        raise ValueError(
            f"Unknown 5-digit mean line {prefix!r}: "
            "use 210, 220, 230, 240 or 250 (scaled by the first digit)."
        )
    m, k1_base = _FIVE_DIGIT_TABLE[pos]
    k1 = k1_base * lift_digit / 2.0
    if lift_digit == 0:
        _warn_flat_mean_line(prefix if name is None else name, prefix[1:])
    return m, k1


def naca5_camber(
    x: np.ndarray | list[float] | float,
    designation: str | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean line and slope of a standard NACA 5-digit airfoil.

    Parameters
    ----------
    x : array_like
        Chordwise stations as a fraction of chord [-], with values in [0, 1].
    designation : str or int
        Full designation such as "23012" or a 3-digit mean line such as
        "230". Reflex mean lines (third digit 1) raise ValueError. A first
        digit 0 (zero design lift) gives a flat mean line with a
        UserWarning.

    Returns
    -------
    y_c : numpy.ndarray
        Mean line height as a fraction of chord [-].
    dyc_dx : numpy.ndarray
        Mean line slope dy_c/dx [-].
    """
    xc = np.asarray(x, dtype=float)
    if np.any(xc < 0.0) or np.any(xc > 1.0):
        raise ValueError("Stations x must lie in [0, 1].")
    digits = _clean(designation)
    if len(digits) == 3:
        prefix = digits
    elif len(digits) == 5:
        prefix = digits[:3]
        if digits[3:5] == "00":
            raise ValueError(f"Invalid NACA designation: {designation!r}.")
    else:
        raise ValueError(f"Invalid NACA designation: {designation!r}.")
    m, k1 = _five_digit_parameters(prefix, designation)
    if k1 == 0.0:
        zeros = np.zeros_like(xc, dtype=float)
        return zeros, zeros.copy()
    front = xc <= m
    quad = m**2 * (3.0 - m)
    yc = np.where(
        front,
        k1 / 6.0 * (xc**3 - 3.0 * m * xc**2 + quad * xc),
        k1 * m**3 / 6.0 * (1.0 - xc),
    )
    dyc = np.where(
        front,
        k1 / 6.0 * (3.0 * xc**2 - 6.0 * m * xc + quad),
        np.full_like(xc, -k1 * m**3 / 6.0, dtype=float),
    )
    return np.asarray(yc, dtype=float), np.asarray(dyc, dtype=float)


def _parse_naca4(designation: str | int) -> tuple[float, float, float]:
    """Return the camber m, position p and thickness t of a 4-digit name.

    Parameters
    ----------
    designation : str or int
        Designation such as "2412" or 2412.

    Returns
    -------
    tuple[float, float, float]
        Maximum camber m [-], camber position p [-], thickness ratio t [-].
    """
    digits = _clean(designation)
    if len(digits) != 4:
        raise ValueError(f"Invalid NACA 4-digit designation: {designation!r}.")
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:4]) / 100.0
    if t == 0.0:
        raise ValueError(f"Invalid NACA 4-digit designation: {designation!r}.")
    if p == 0.0 and m != 0.0:
        raise ValueError(f"Invalid NACA 4-digit designation: {designation!r}.")
    if m == 0.0 and p != 0.0:
        _warn_flat_mean_line(designation, digits[:2])
    return m, p, t


def _parse_naca5(designation: str | int) -> tuple[str, float]:
    """Return the digits and the thickness t of a 5-digit name.

    Parameters
    ----------
    designation : str or int
        Designation such as "23012" or 23012.

    Returns
    -------
    tuple[str, float]
        All five digits such as "23012" [-] and thickness ratio t [-].
    """
    digits = _clean(designation)
    if len(digits) != 5:
        raise ValueError(f"Invalid NACA 5-digit designation: {designation!r}.")
    if digits[3:5] == "00":
        raise ValueError(f"Invalid NACA 5-digit designation: {designation!r}.")
    t = int(digits[3:5]) / 100.0
    return digits, t


def _stations(n_points: int, spacing: str) -> np.ndarray:
    """Return the chordwise stations from 0 to 1.

    Parameters
    ----------
    n_points : int
        Total points of the airfoil loop. It must be odd.
    spacing : str
        "cosine" or "uniform".

    Returns
    -------
    numpy.ndarray
        Stations as a fraction of chord [-], from 0 to 1.
    """
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, bool):
        raise ValueError(f"n_points must be an odd integer, got {n_points!r}.")
    if int(n_points) % 2 == 0 or int(n_points) < 3:
        raise ValueError(f"n_points must be odd and at least 3, got {n_points!r}.")
    n_half = (int(n_points) + 1) // 2
    if spacing == "cosine":
        beta = np.linspace(0.0, np.pi, n_half)
        return (1.0 - np.cos(beta)) / 2.0
    if spacing == "uniform":
        return np.linspace(0.0, 1.0, n_half)
    raise ValueError(f"Unknown spacing {spacing!r}: use 'cosine' or 'uniform'.")


def _assemble(
    xc: np.ndarray,
    yt: np.ndarray,
    yc: np.ndarray,
    dyc: np.ndarray,
) -> np.ndarray:
    """Return the Selig-ordered loop from stations and thickness.

    Parameters
    ----------
    xc : numpy.ndarray
        Chordwise stations as a fraction of chord [-], from 0 to 1.
    yt : numpy.ndarray
        Half thickness as a fraction of chord [-].
    yc : numpy.ndarray
        Mean line height as a fraction of chord [-].
    dyc : numpy.ndarray
        Mean line slope [-].

    Returns
    -------
    numpy.ndarray
        Points (N, 2) as a fraction of chord [-], in Selig order.
    """
    theta = np.arctan(dyc)
    st = np.sin(theta)
    ct = np.cos(theta)
    xu = xc - yt * st
    yu = yc + yt * ct
    xl = xc + yt * st
    yl = yc - yt * ct
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower]).astype(float)


def naca4(
    designation: str | int,
    n_points: int = 161,
    spacing: str = "cosine",
    closed_te: bool = False,
) -> np.ndarray:
    """Return the coordinates of a NACA 4-digit airfoil.

    Parameters
    ----------
    designation : str or int
        Designation such as "2412", "NACA 2412" or 2412.
    n_points : int
        Total points of the loop. It must be odd. Default 161.
    spacing : str
        "cosine" or "uniform". Default "cosine".
    closed_te : bool
        Close the trailing edge. Default False.

    Returns
    -------
    numpy.ndarray
        Points (N, 2) as a fraction of chord [-], in Selig order: from the
        trailing edge along the upper surface to the leading edge, then
        along the lower surface back to the trailing edge. The points are
        not re-normalized: the perpendicular thickness moves the true
        leading edge slightly off (0, 0) for cambered sections.
    """
    m, p, t = _parse_naca4(designation)
    xc = _stations(n_points, spacing)
    yt = naca4_thickness(xc, t, closed_te=closed_te)
    yc, dyc = naca4_camber(xc, m, p)
    return _assemble(xc, yt, yc, dyc)


def naca5(
    designation: str | int,
    n_points: int = 161,
    spacing: str = "cosine",
    closed_te: bool = False,
) -> np.ndarray:
    """Return the coordinates of a standard NACA 5-digit airfoil.

    Parameters
    ----------
    designation : str or int
        Designation such as "23012", "NACA 23012" or 23012. Reflex mean
        lines (third digit 1) raise ValueError.
    n_points : int
        Total points of the loop. It must be odd. Default 161.
    spacing : str
        "cosine" or "uniform". Default "cosine".
    closed_te : bool
        Close the trailing edge. Default False.

    Returns
    -------
    numpy.ndarray
        Points (N, 2) as a fraction of chord [-], in Selig order. The
        points are not re-normalized (see :func:`naca4`).
    """
    digits, t = _parse_naca5(designation)
    xc = _stations(n_points, spacing)
    yt = naca4_thickness(xc, t, closed_te=closed_te)
    yc, dyc = naca5_camber(xc, digits)
    return _assemble(xc, yt, yc, dyc)


def naca(
    designation: str | int,
    n_points: int = 161,
    spacing: str = "cosine",
    closed_te: bool = False,
) -> np.ndarray:
    """Return the coordinates of a NACA 4-digit or 5-digit airfoil.

    Parameters
    ----------
    designation : str or int
        Designation such as "2412", "NACA 2412", 2412, "23012" or
        "NACA 23012". Four digits select the 4-digit series, five digits
        select the 5-digit series. Hyphens, underscores and spaces inside
        the digits raise ValueError (see the module docstring).
    n_points : int
        Total points of the loop. It must be odd. Default 161.
    spacing : str
        "cosine" or "uniform". Default "cosine".
    closed_te : bool
        Close the trailing edge. Default False.

    Returns
    -------
    numpy.ndarray
        Points (N, 2) as a fraction of chord [-], in Selig order.
    """
    digits = _clean(designation)
    if len(digits) == 4:
        return naca4(digits, n_points=n_points, spacing=spacing, closed_te=closed_te)
    if len(digits) == 5:
        return naca5(digits, n_points=n_points, spacing=spacing, closed_te=closed_te)
    raise ValueError(f"Invalid NACA designation: {designation!r}.")
