# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Parse AVL output files.

The parsers read the ``FT`` (total forces) and ``ST`` (stability
derivatives) files that ``avl.exe`` writes. They know only the text
format shown in the AVL user documentation and in real program output.
A run is good only when its output files exist and parse completely:
a missing or truncated value raises ``ValueError`` (never a silent
zero).
"""

from __future__ import annotations

import re

#: Floating-point number in AVL output (handles 0.7758E-01, -0.00000).
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def _find(text: str, pattern: str, label: str) -> float:
    """Return the first float after *label* in *text*, or raise."""
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"AVL output is truncated or invalid: {label} not found.")
    try:
        value = float(match.group(1))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"AVL output is truncated or invalid: {label} is not a number.") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"AVL output is invalid: {label} is not finite.")
    return value


def parse_ft(text: str) -> dict[str, float]:
    """Parse an AVL total-forces (``FT``) file.

    Parameters
    ----------
    text : str
        Full content of the file.

    Returns
    -------
    dict
        Forces, moments and reference values. Moment keys ``Cl``,
        ``Cm`` and ``Cn`` are the AVL standard-axis body moments;
        ``Cl_prim`` and ``Cn_prim`` are the stability-axis values
        (primed in the file). ``CDff`` is the Trefftz-plane induced
        drag.

    Raises
    ------
    ValueError
        If any expected value is missing or not a number.
    """
    get = lambda pat, label: _find(text, pat, label)  # noqa: E731
    out = {
        "Alpha": get(r"Alpha\s*=\s*(" + _FLOAT + r")", "Alpha"),
        "Beta": get(r"Beta\s*=\s*(" + _FLOAT + r")", "Beta"),
        "Mach": get(r"Mach\s*=\s*(" + _FLOAT + r")", "Mach"),
        "CXtot": get(r"CXtot\s*=\s*(" + _FLOAT + r")", "CXtot"),
        "CYtot": get(r"CYtot\s*=\s*(" + _FLOAT + r")", "CYtot"),
        "CZtot": get(r"CZtot\s*=\s*(" + _FLOAT + r")", "CZtot"),
        "Cltot": get(r"Cltot\s*=\s*(" + _FLOAT + r")", "Cltot"),
        "Cmtot": get(r"Cmtot\s*=\s*(" + _FLOAT + r")", "Cmtot"),
        "Cntot": get(r"Cntot\s*=\s*(" + _FLOAT + r")", "Cntot"),
        "Cl_prim": get(r"Cl'tot\s*=\s*(" + _FLOAT + r")", "Cl'tot"),
        "Cn_prim": get(r"Cn'tot\s*=\s*(" + _FLOAT + r")", "Cn'tot"),
        "CLtot": get(r"CLtot\s*=\s*(" + _FLOAT + r")", "CLtot"),
        "CDtot": get(r"CDtot\s*=\s*(" + _FLOAT + r")", "CDtot"),
        "CDvis": get(r"CDvis\s*=\s*(" + _FLOAT + r")", "CDvis"),
        "CDind": get(r"CDind\s*=\s*(" + _FLOAT + r")", "CDind"),
        "CLff": get(r"CLff\s*=\s*(" + _FLOAT + r")", "CLff"),
        "CDff": get(r"CDff\s*=\s*(" + _FLOAT + r")", "CDff"),
        "CYff": get(r"CYff\s*=\s*(" + _FLOAT + r")", "CYff"),
        "e": get(r"\be\s*=\s*(" + _FLOAT + r")", "span efficiency e"),
        "Sref": get(r"Sref\s*=\s*(" + _FLOAT + r")", "Sref"),
        "Cref": get(r"Cref\s*=\s*(" + _FLOAT + r")", "Cref"),
        "Bref": get(r"Bref\s*=\s*(" + _FLOAT + r")", "Bref"),
        "Xref": get(r"Xref\s*=\s*(" + _FLOAT + r")", "Xref"),
        "Yref": get(r"Yref\s*=\s*(" + _FLOAT + r")", "Yref"),
        "Zref": get(r"Zref\s*=\s*(" + _FLOAT + r")", "Zref"),
    }
    return out


#: Stability-derivative names in the ``ST`` file and their plain keys.
_ST_NAMES = (
    "CLa", "CLb", "CYa", "CYb", "CDa", "CDb",
    "Cla", "Clb", "Cma", "Cmb", "Cna", "Cnb",
    "CLp", "CLq", "CLr", "CYp", "CYq", "CYr",
    "CDp", "CDq", "CDr", "Clp", "Clq", "Clr",
    "Cmp", "Cmq", "Cmr", "Cnp", "Cnq", "Cnr",
)


def parse_st(text: str) -> dict[str, float]:
    """Parse an AVL stability-derivatives (``ST``) file.

    The file starts with the total-forces block, so the result holds
    every key of :func:`parse_ft` plus the alpha, beta and rate
    derivatives and the neutral point ``Xnp`` [m].

    Parameters
    ----------
    text : str
        Full content of the file.

    Returns
    -------
    dict
        Total forces plus derivatives (per rad for the angle
        derivatives).

    Raises
    ------
    ValueError
        If any expected value is missing or not a number.
    """
    out = parse_ft(text)
    for name in _ST_NAMES:
        out[name] = _find(text, r"\b" + name + r"\s*=\s*(" + _FLOAT + r")", name)
    out["Xnp"] = _find(text, r"Xnp\s*=\s*(" + _FLOAT + r")", "Neutral point Xnp")
    return out


def parse_banner(stdout: str) -> str:
    """Return the AVL version text from the program banner.

    Parameters
    ----------
    stdout : str
        Captured standard output of ``avl.exe``.

    Returns
    -------
    str
        For example ``"Athena Vortex Lattice Program Version 3.52"``.

    Raises
    ------
    ValueError
        If the banner is not found (the program never started).
    """
    match = re.search(r"Athena Vortex Lattice\s+Program\s+Version\s+([\d.]+)", stdout)
    if match is None:
        raise ValueError("AVL banner not found in the program output.")
    return f"Athena Vortex Lattice Program Version {match.group(1)}"
