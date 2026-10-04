# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Airfoil polar interpolation and management utilities.

Provides helpers for loading polar data from XFOIL-format files and CSV,
and for blending polars between spanwise stations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ventorum.legacy.core.datatypes import LinearAirfoil, TabulatedAirfoil, AirfoilType


# ═══════════════════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════════════════

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

            # Try to extract Reynolds number from header
            if "Re =" in line or "Re=" in line:
                parts = line.replace("Re =", "Re=").split("Re=")
                if len(parts) > 1:
                    re_str = parts[1].strip().split()[0].replace(",", "")
                    try:
                        # XFOIL often uses e.g. "1.000 e 6"
                        re_val = float(re_str)
                    except ValueError:
                        pass

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

    return TabulatedAirfoil(
        name=name,
        alpha=np.radians(np.array(alphas)),
        Cl_data=np.array(cls),
        Cd_data=np.array(cds),
        Cm_data=np.array(cms),
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
        header = f.readline().strip().split(delimiter)
        header = [h.strip() for h in header]
        for h in header:
            data[h] = []
        for line in f:
            cols = line.strip().split(delimiter)
            for h, v in zip(header, cols):
                try:
                    data[h].append(float(v.strip()))
                except ValueError:
                    data[h].append(0.0)

    alphas = np.array(data[alpha_col])
    if alpha_in_degrees:
        alphas = np.radians(alphas)

    cm_arr = np.array(data[cm_col]) if cm_col and cm_col in data else None

    return TabulatedAirfoil(
        name=filepath.stem,
        alpha=alphas,
        Cl_data=np.array(data[cl_col]),
        Cd_data=np.array(data[cd_col]),
        Cm_data=cm_arr,
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
        Blended polar evaluated on a common alpha grid.
    """
    w = np.clip(weight, 0.0, 1.0)
    # Common alpha grid: union of both ranges
    a_min = max(af1.alpha.min(), af2.alpha.min())
    a_max = min(af1.alpha.max(), af2.alpha.max())
    n = max(len(af1.alpha), len(af2.alpha))
    alpha_common = np.linspace(a_min, a_max, n)

    Cl = (1 - w) * np.interp(alpha_common, af1.alpha, af1.Cl_data) \
         + w * np.interp(alpha_common, af2.alpha, af2.Cl_data)
    Cd = (1 - w) * np.interp(alpha_common, af1.alpha, af1.Cd_data) \
         + w * np.interp(alpha_common, af2.alpha, af2.Cd_data)

    Cm = None
    if af1.Cm_data is not None and af2.Cm_data is not None:
        Cm = (1 - w) * np.interp(alpha_common, af1.alpha, af1.Cm_data) \
             + w * np.interp(alpha_common, af2.alpha, af2.Cm_data)

    return TabulatedAirfoil(
        name=f"blend({af1.name},{af2.name},w={w:.2f})",
        alpha=alpha_common,
        Cl_data=Cl,
        Cd_data=Cd,
        Cm_data=Cm,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: get section Cl for any airfoil type
# ═══════════════════════════════════════════════════════════════════════════════

def section_Cl(airfoil: AirfoilType, alpha: float) -> float:
    """Get Cl for any airfoil type at the given angle of attack [rad]."""
    return float(np.atleast_1d(airfoil.Cl(alpha)).item())


def section_Cd(airfoil: AirfoilType, alpha: float) -> float:
    """Get Cd for any airfoil type at the given angle of attack [rad]."""
    return float(np.atleast_1d(airfoil.Cd(alpha)).item())
