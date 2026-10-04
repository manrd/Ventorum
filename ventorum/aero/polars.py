# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Airfoil polar interpolation and management utilities.

Provides helpers for loading polar data from XFOIL-format files and CSV,
and for blending polars between spanwise stations.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from ventorum.core.datatypes import TabulatedAirfoil


# ═══════════════════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════════════════

_RE_PATTERN = re.compile(r"\bRe\s*=\s*([0-9]*\.?[0-9]+)\s*(?:[eE]\s*([+-]?\s*\d+))?")

def load_xfoil_polar(filepath: str | Path) -> TabulatedAirfoil:
    """Load a polar from an XFOIL-format file.

    XFOIL polar files have a multi-line header (lines starting with ``-``
    or containing text) followed by columnar data with columns:
    ``alpha  CL  CD  CDp  CM  Top_Xtr  Bot_Xtr``.

    Parameters
    ----------
    filepath : str or Path
        Path to the XFOIL polar file.

    Returns
    -------
    TabulatedAirfoil
    """
    filepath = Path(filepath)
    alphas, cls, cds, cms = [], [], [], []
    re_val = None
    name = filepath.stem

    with open(filepath) as f:
        in_data = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Reynolds number from the header, e.g. "Re =     1.000 e 6" or "Re = 3e5".
            if re_val is None:
                m = _RE_PATTERN.search(line)
                if m:
                    mant = float(m.group(1))
                    exp = int(m.group(2)) if m.group(2) is not None else 0
                    re_val = mant * 10.0 ** exp

            # Header-data boundary: a line of dashes
            if stripped.startswith("---"):
                in_data = True
                continue

            if in_data:
                cols = stripped.split()
                if len(cols) >= 5:
                    try:
                        alphas.append(float(cols[0]))
                        cls.append(float(cols[1]))
                        cds.append(float(cols[2]))
                        cms.append(float(cols[4]))
                    except ValueError:
                        continue

    # XFOIL writes the points in the order it ran them (for example two
    # sweeps from alpha = 0), so sort them. TabulatedAirfoil merges repeated alphas.
    order = np.argsort(np.array(alphas, dtype=float), kind="stable")
    return TabulatedAirfoil(
        name=name,
        alpha=np.radians(np.array(alphas, dtype=float)[order]),
        Cl_data=np.array(cls, dtype=float)[order],
        Cd_data=np.array(cds, dtype=float)[order],
        Cm_data=np.array(cms, dtype=float)[order],
        Re=re_val,
    )


def load_csv_polar(
    filepath: str | Path,
    alpha_col: str = "alpha",
    cl_col: str = "Cl",
    cd_col: str = "Cd",
    cm_col: str | None = "Cm",
    alpha_in_degrees: bool = True,
    delimiter: str = ",",
) -> TabulatedAirfoil:
    """Load a polar from a CSV file.

    Parameters
    ----------
    filepath : str or Path
    alpha_col, cl_col, cd_col, cm_col : str
        Column header names.
    alpha_in_degrees : bool
        If *True*, the alpha column is converted from degrees to radians.
    delimiter : str
        CSV delimiter character.

    Returns
    -------
    TabulatedAirfoil
    """
    filepath = Path(filepath)
    data: dict[str, list[float]] = {}

    with open(filepath) as f:
        header = [h.strip() for h in f.readline().strip().split(delimiter)]
        for h in header:
            data[h] = []
        for line_no, line in enumerate(f, start=2):
            if not line.strip():
                continue
            cols = line.strip().split(delimiter)
            if len(cols) != len(header):
                raise ValueError(f"{filepath}: line {line_no} has {len(cols)} values, the header has {len(header)}.")
            for h, v in zip(header, cols):
                try:
                    data[h].append(float(v.strip()))
                except ValueError as exc:
                    raise ValueError(f"{filepath}: line {line_no}, column {h!r}: {v.strip()!r} is not a number.") from exc

    for col in (alpha_col, cl_col, cd_col):
        if col not in data:
            raise ValueError(f"{filepath}: column {col!r} not found (columns: {', '.join(header)}).")

    alphas = np.array(data[alpha_col])
    if alpha_in_degrees:
        alphas = np.radians(alphas)

    cm_arr = None
    if cm_col:
        if cm_col in data:
            cm_arr = np.array(data[cm_col])
        else:
            warnings.warn(f"{filepath}: no {cm_col!r} column; the section pitching moment is taken as zero.",
                          RuntimeWarning, stacklevel=2)

    order = np.argsort(alphas)
    return TabulatedAirfoil(
        name=filepath.stem,
        alpha=alphas[order],
        Cl_data=np.array(data[cl_col])[order],
        Cd_data=np.array(data[cd_col])[order],
        Cm_data=None if cm_arr is None else cm_arr[order],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Polar blending
# ═══════════════════════════════════════════════════════════════════════════════

def blend_tabulated(
    af1: TabulatedAirfoil,
    af2: TabulatedAirfoil,
    weight: float,
) -> TabulatedAirfoil:
    """Linearly blend two tabulated polars.

    Parameters
    ----------
    af1, af2 : TabulatedAirfoil
        The two polars to blend.
    weight : float
        Blending weight in [0, 1].  ``0`` → pure *af1*, ``1`` → pure *af2*.

    Returns
    -------
    TabulatedAirfoil
        Blended polar evaluated on the common alpha range. If the input polars
        have different alpha ranges, a RuntimeWarning is issued giving both
        ranges and the kept range in degrees.
    """
    w = float(np.clip(weight, 0.0, 1.0))
    a1_min, a1_max = float(af1.alpha.min()), float(af1.alpha.max())
    a2_min, a2_max = float(af2.alpha.min()), float(af2.alpha.max())
    a_min = max(a1_min, a2_min)
    a_max = min(a1_max, a2_max)
    if a_max <= a_min:
        raise ValueError(f"Polars {af1.name!r} and {af2.name!r} have no common angle-of-attack range.")
    # Warn if the ranges differ.
    if not (np.isclose(a1_min, a2_min) and np.isclose(a1_max, a2_max)):
        warnings.warn(
            f"Blending polars with different alpha ranges: {af1.name!r} [{np.degrees(a1_min):.1f}, {np.degrees(a1_max):.1f}] deg, "
            f"{af2.name!r} [{np.degrees(a2_min):.1f}, {np.degrees(a2_max):.1f}] deg. "
            f"Kept range: [{np.degrees(a_min):.1f}, {np.degrees(a_max):.1f}] deg.",
            RuntimeWarning, stacklevel=2,
        )
    # Common grid: every tabulated point of both polars inside the range they
    # share (no resampling, so a peak at a tabulated point is kept).
    alpha_common = np.unique(np.concatenate([af1.alpha, af2.alpha]))
    alpha_common = alpha_common[(alpha_common >= a_min) & (alpha_common <= a_max)]

    def mix(f1, f2):
        return (1.0 - w) * f1(alpha_common) + w * f2(alpha_common)

    Cl = mix(af1.Cl, af2.Cl)
    Cd = mix(af1.Cd, af2.Cd)
    Cm = mix(af1.Cm, af2.Cm) if (af1.Cm_data is not None and af2.Cm_data is not None) else None
    if (af1.Cm_data is None) != (af2.Cm_data is None):
        warnings.warn("Only one of the blended polars has Cm data; the blend has no Cm.", RuntimeWarning, stacklevel=2)
    re_val = af1.Re if af1.Re == af2.Re else None

    return TabulatedAirfoil(
        name=f"blend({af1.name},{af2.name},w={w:.2f})",
        alpha=alpha_common,
        Cl_data=np.asarray(Cl, dtype=float),
        Cd_data=np.asarray(Cd, dtype=float),
        Cm_data=None if Cm is None else np.asarray(Cm, dtype=float),
        Re=re_val,
    )

