# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run XFOIL and read its polars as tabulated airfoils.

XFOIL is driven as a black box through its documented interactive commands
(M. Drela and H. Youngren, XFOIL 6.99 user guide): the NACA generator, LOAD,
PPAR, PANE, GDES with FLAP, OPER with VPAR, PACC, ASEQ and ALFA. No XFOIL
source code is read or used.

Sources: Drela, M., 1989, XFOIL: An analysis and design system for low
Reynolds number airfoils; Drela, M. and Youngren, H., XFOIL 6.99 user guide
(documented commands only).
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ventorum.aero.polars import load_xfoil_polar
from ventorum.core.datatypes import TabulatedAirfoil


def resolve_xfoil_executable(executable: str | None = None) -> str:
    """Resolve the XFOIL executable name or path.

    Parameters
    ----------
    executable : str or None
        Explicit executable. None uses the environment variable
        ``VENTORUM_XFOIL``, else ``xfoil`` on PATH.

    Returns
    -------
    str
        The executable to run. No machine path is stored in the code.
    """
    if executable is not None:
        return executable
    return os.environ.get("VENTORUM_XFOIL", "xfoil")


@functools.lru_cache(maxsize=8)
def xfoil_version(executable: str | None = None) -> str:
    """Read the XFOIL version line from its start-up banner.

    Parameters
    ----------
    executable : str or None
        Explicit executable (see :func:`resolve_xfoil_executable`).

    Returns
    -------
    str
        The banner line that holds ``Version`` (for example
        ``"XFOIL Version 6.99"``), or ``"unknown"`` when XFOIL does not run.
    """
    exe = resolve_xfoil_executable(executable)
    try:
        result = subprocess.run(
            [exe],
            input="quit\n",
            text=True,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in result.stdout.splitlines():
        if "Version" in line:
            return line.strip()
    return "unknown"


def _write_selig_file(path: str | Path, name: str, coordinates: np.ndarray) -> None:
    """Write airfoil coordinates to a Selig-format file for XFOIL LOAD.

    Parameters
    ----------
    path : str or Path
        Output file path.
    name : str
        Airfoil name written on the first line.
    coordinates : np.ndarray
        Points (N, 2) in Selig order: from the trailing edge along the upper
        surface to the leading edge, then along the lower surface back to the
        trailing edge. A closed trailing edge repeats the first point last.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Coordinates must have shape (N, 2), got {coords.shape}."
        )
    if coords.shape[0] < 2:
        raise ValueError("Coordinates must hold at least 2 points.")
    lines = [str(name)]
    lines.extend(f"{float(x):.10f} {float(y):.10f}" for x, y in coords)
    Path(path).write_text("\n".join(lines) + "\n")


#: Answer to the XFOIL FLAP prompt for the hinge y location that makes
#: XFOIL ask for the hinge height as a fraction of the local thickness
#: (XFOIL 6.99 user guide, GDES menu, FLAP command).
XFOIL_FLAP_Y_OVER_T_CODE = "999"


def check_flap(flap: object) -> tuple[float, float, float] | None:
    """Check a plain-flap tuple and return it as floats.

    Every section polar source of Ventorum uses the same flap tuple
    ``(x_hinge, y_hinge, deflection_deg)``.

    Parameters
    ----------
    flap : sequence of 3 numbers or None
        ``x_hinge`` [-]: hinge position as a fraction of the chord, in
        (0, 1). ``y_hinge`` [-]: hinge height as a fraction of the local
        thickness, from the lower surface (0) to the upper surface (1).
        ``deflection_deg`` [deg]: flap deflection, positive trailing edge
        down, magnitude below 90 deg. None means no flap.

    Returns
    -------
    tuple of float or None
        ``(x_hinge, y_hinge, deflection_deg)``, or None.

    Raises
    ------
    ValueError
        If the tuple does not hold 3 finite numbers in range.
    """
    if flap is None:
        return None
    if isinstance(flap, (str, bytes)):
        raise ValueError(
            "flap must be (x_hinge, y_hinge, deflection_deg), got a string."
        )
    try:
        values = [float(v) for v in flap]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise ValueError(
            f"flap must be (x_hinge, y_hinge, deflection_deg), got {flap!r}."
        ) from None
    if len(values) != 3:
        raise ValueError(
            "flap must hold 3 numbers (x_hinge, y_hinge, deflection_deg), "
            f"got {len(values)}."
        )
    x_hinge, y_hinge, deflection_deg = values
    if not all(np.isfinite(values)):
        raise ValueError(f"flap values must be finite, got {values}.")
    if not 0.0 < x_hinge < 1.0:
        raise ValueError(f"flap x_hinge must lie in (0, 1), got {x_hinge}.")
    if not 0.0 <= y_hinge <= 1.0:
        raise ValueError(
            "flap y_hinge is a fraction of the local thickness and must lie "
            f"in [0, 1], got {y_hinge}."
        )
    if abs(deflection_deg) >= 90.0:
        raise ValueError(
            f"flap deflection_deg must be below 90 deg in magnitude, got {deflection_deg}."
        )
    return (x_hinge, y_hinge, deflection_deg)


def _geometry_commands(
    airfoil: str,
    coord_file: str | None,
    n_panels: int | None,
    flap: tuple[float, float, float] | None,
) -> list[str]:
    """Build the XFOIL command lines that set the geometry and panel it.

    The flap uses the documented FLAP command of the GDES menu. XFOIL asks
    for the hinge x location, then the hinge y location; the answer 999
    makes it ask for the hinge height as a fraction of the local thickness
    (y/t), then for the deflection [deg], positive down.
    """
    if coord_file is not None:
        cmds = [f"load {coord_file}"]
    elif airfoil.lower().startswith("naca"):
        cmds = [f"{airfoil}"]
    else:
        cmds = [f"load {airfoil}"]
    if n_panels is not None:
        cmds.extend(["ppar", f"n {int(n_panels)}", "", ""])
    if flap is not None:
        x_hinge, y_over_t, deflection_deg = flap
        cmds.extend(
            [
                "gdes",
                "flap",
                f"{float(x_hinge)}",
                XFOIL_FLAP_Y_OVER_T_CODE,
                f"{float(y_over_t)}",
                f"{float(deflection_deg)}",
                "",
                "",
            ]
        )
    cmds.append("pane")
    return cmds


def _oper_commands(
    Re: float,
    mach: float,
    max_iter: int,
    ncrit: float,
    xtr_top: float,
    xtr_bottom: float,
) -> list[str]:
    """Build the XFOIL command lines that set the viscous operating conditions."""
    return [
        "oper",
        "vpar",
        f"n {float(ncrit)}",
        f"xtr {float(xtr_top)} {float(xtr_bottom)}",
        "",
        f"iter {max_iter}",
        f"mach {mach}",
        f"visc {Re}",
    ]


def _execute(commands: list[str], executable: str, timeout: float | None) -> str:
    """Send the command lines to XFOIL and return its standard output.

    Parameters
    ----------
    commands : list[str]
        XFOIL command lines, without the trailing newline.
    executable : str
        Resolved XFOIL executable.
    timeout : float or None
        Largest run time [s]. None means no limit.

    Returns
    -------
    str
        The standard output of XFOIL.
    """
    cmd_str = "\n".join(commands) + "\n"
    try:
        result = subprocess.run(
            [executable],
            input=cmd_str,
            text=True,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"XFOIL execution failed with exit code {e.returncode}:\n{e.stdout}\n{e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"XFOIL did not finish in {timeout} s and was stopped."
        ) from e
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "XFOIL executable not found. Ensure 'xfoil' is in your system PATH."
        ) from e
    return result.stdout


def _run_leg(
    geometry: list[str],
    oper: list[str],
    sweep_cmd: str,
    polar_file: Path,
    executable: str,
    timeout: float | None,
) -> tuple[TabulatedAirfoil | None, str]:
    """Run one XFOIL alpha sweep and read its polar file.

    Returns
    -------
    TabulatedAirfoil or None
        The polar, or None when XFOIL wrote no usable point.
    str
        The standard output of XFOIL.
    """
    stdout = _execute(
        geometry + oper + ["pacc", str(polar_file), "", sweep_cmd, "pacc", "", "quit"],
        executable,
        timeout,
    )
    if not polar_file.exists():
        return None, stdout
    polar = load_xfoil_polar(polar_file)
    if len(polar.alpha) == 0:
        return None, stdout
    return polar, stdout


def _merge_runs(
    polars: list[TabulatedAirfoil],
    name: str,
    lo_deg: float,
    hi_deg: float,
) -> TabulatedAirfoil:
    """Merge the polars of the sweep legs, sorted, clipped and deduplicated.

    Parameters
    ----------
    polars : list[TabulatedAirfoil]
        One polar per sweep leg.
    name : str
        Polar name.
    lo_deg, hi_deg : float
        Requested angle range [deg]. Points outside are dropped, so a sweep
        that starts at 0 keeps only the requested range.

    Returns
    -------
    TabulatedAirfoil
        The merged polar. Angles shared by two legs (for example alpha = 0)
        are replaced by their mean.
    """
    prepared = [p.tables() for p in polars]
    alphas = np.concatenate(
        [np.asarray(t[0], dtype=float).ravel() for t in prepared]
    )
    cls = np.concatenate(
        [np.asarray(t[1], dtype=float).ravel() for t in prepared]
    )
    cds = np.concatenate(
        [np.asarray(t[2], dtype=float).ravel() for t in prepared]
    )
    cms_list = [t[3] for t in prepared]
    cms = (
        np.concatenate([np.asarray(c, dtype=float).ravel() for c in cms_list])
        if all(c is not None for c in cms_list)
        else None
    )
    lo, hi = np.radians(min(lo_deg, hi_deg)), np.radians(max(lo_deg, hi_deg))
    keep = (alphas >= lo - 1e-12) & (alphas <= hi + 1e-12)
    alphas, cls, cds = alphas[keep], cls[keep], cds[keep]
    cms = None if cms is None else cms[keep]
    order = np.argsort(alphas, kind="stable")
    alphas, cls, cds = alphas[order], cls[order], cds[order]
    cms = None if cms is None else cms[order]
    uniq, inverse, counts = np.unique(alphas, return_inverse=True, return_counts=True)
    if uniq.size < alphas.size:
        cls = np.bincount(inverse, weights=cls) / counts
        cds = np.bincount(inverse, weights=cds) / counts
        if cms is not None:
            cms = np.bincount(inverse, weights=cms) / counts
        alphas = uniq
    return TabulatedAirfoil(
        name=name,
        alpha=np.asarray(alphas, dtype=float),
        Cl_data=np.asarray(cls, dtype=float),
        Cd_data=np.asarray(cds, dtype=float),
        Cm_data=None if cms is None else np.asarray(cms, dtype=float),
        Re=polars[0].Re,
    )


def run_xfoil(
    airfoil: str,
    Re: float,
    alpha_start: float = -5.0,
    alpha_end: float = 15.0,
    alpha_step: float = 0.5,
    mach: float = 0.0,
    max_iter: int = 150,
    timeout: float | None = 300.0,
    coordinates: np.ndarray | None = None,
    ncrit: float = 9.0,
    xtr_top: float = 1.0,
    xtr_bottom: float = 1.0,
    n_panels: int | None = None,
    flap: tuple[float, float, float] | None = None,
    executable: str | None = None,
    sweep: str = "linear",
) -> TabulatedAirfoil:
    """Run XFOIL to generate a polar and return it as a TabulatedAirfoil.

    Parameters
    ----------
    airfoil : str
        The airfoil specification. If it starts with 'naca' (e.g., 'naca 2412'),
        XFOIL's internal NACA generator is used. Otherwise, it is treated as a
        file path to a coordinate file to LOAD. When *coordinates* is given,
        this string only names the polar. It must be one line: a line break
        or another control character is refused, because XFOIL reads its
        commands line by line.
    Re : float
        Reynolds number for the viscous analysis.
    alpha_start : float
        Starting angle of attack [deg].
    alpha_end : float
        Ending angle of attack [deg].
    alpha_step : float
        Angle of attack step size [deg]. The sign follows the sweep direction.
    mach : float
        Mach number.
    max_iter : int
        Maximum viscous iterations per angle of attack.
    timeout : float or None
        Largest run time of each XFOIL run [s] (default 300). XFOIL can stop
        and wait for input when it does not converge; the run is then stopped.
        *None* means no limit.
    coordinates : np.ndarray or None
        Points (N, 2) in Selig order, written to a temporary Selig file and
        loaded instead of *airfoil*. None uses *airfoil*.
    ncrit : float
        Critical amplification factor for transition. Default 9.0.
    xtr_top, xtr_bottom : float
        Forced transition locations (fraction of chord) on the upper and
        lower surface. Default 1.0 (free transition).
    n_panels : int or None
        Panel count set with XFOIL PPAR before paneling. None keeps the
        XFOIL default (160).
    flap : tuple or None
        ``(x_hinge, y_hinge, deflection_deg)``: hinge position as a fraction
        of the chord [-], hinge height as a fraction of the local thickness
        [-] (0 lower surface, 1 upper surface) and deflection [deg],
        positive down. Applied with the documented FLAP command of the
        XFOIL geometry design menu (hinge height given as y/t). See
        :func:`check_flap`.
    executable : str or None
        XFOIL executable (see :func:`resolve_xfoil_executable`).
    sweep : str
        Order of the angle-of-attack runs [deg]. ``"linear"`` (default, the
        behaviour of earlier versions) runs one sweep from *alpha_start*
        to *alpha_end*. ``"from_zero"`` runs alpha from 0 up to
        *alpha_end* and, in a new XFOIL start, from 0 down to
        *alpha_start*, then merges and sorts (better convergence near
        stall). :class:`ventorum.aero.polar_source.XfoilPolarSource` uses
        ``"from_zero"`` by default.

    Returns
    -------
    TabulatedAirfoil
        The parsed aerodynamic polar. Points that did not converge are absent.

    Raises
    ------
    FileNotFoundError
        If the XFOIL executable is not found.
    RuntimeError
        If XFOIL execution fails, takes longer than *timeout*, or fails to
        produce a polar file.
    ValueError
        If *sweep* is not ``"from_zero"`` or ``"linear"``, if *airfoil*
        holds a control character, or if *flap* is not valid.
    """
    if sweep not in ("from_zero", "linear"):
        raise ValueError(
            f"Unknown sweep {sweep!r}; use 'from_zero' or 'linear'."
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(airfoil)):
        raise ValueError(
            "The airfoil text must be one line without control characters."
        )
    flap = check_flap(flap)
    exe = resolve_xfoil_executable(executable)
    step_mag = abs(float(alpha_step))
    if step_mag == 0.0:
        raise ValueError("alpha_step must not be zero.")
    if airfoil.lower().startswith("naca"):
        name = airfoil.upper()
    else:
        name = Path(airfoil).stem

    if sweep == "linear":
        step = step_mag if alpha_end >= alpha_start else -step_mag
        legs = [f"aseq {alpha_start} {alpha_end} {step}"]
    else:
        legs = []
        if alpha_end != 0.0:
            legs.append(
                f"aseq 0.0 {alpha_end} {step_mag if alpha_end > 0.0 else -step_mag}"
            )
        if alpha_start != 0.0:
            legs.append(
                f"aseq 0.0 {alpha_start} {step_mag if alpha_start > 0.0 else -step_mag}"
            )
        if not legs:
            legs = [f"aseq 0.0 0.0 {step_mag}"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        coord_file = None
        if coordinates is not None:
            coord_file = str(tmp_path / "coordinates.dat")
            _write_selig_file(coord_file, airfoil, coordinates)
        geometry = _geometry_commands(airfoil, coord_file, n_panels, flap)
        oper = _oper_commands(Re, mach, max_iter, ncrit, xtr_top, xtr_bottom)
        polars: list[TabulatedAirfoil] = []
        last_stdout = ""
        for index, leg in enumerate(legs):
            polar_file = tmp_path / f"polar_{index}.txt"
            polar, last_stdout = _run_leg(
                geometry, oper, leg, polar_file, exe, timeout
            )
            if polar is not None:
                polars.append(polar)
        if not polars:
            raise RuntimeError(
                "XFOIL did not generate a polar file. It likely failed to "
                f"converge.\nXFOIL Output:\n{last_stdout}"
            )
        merged = _merge_runs(polars, name, alpha_start, alpha_end)
        if len(merged.alpha) == 0:
            raise RuntimeError(
                "XFOIL generated an empty polar file. It failed to converge "
                f"for any point.\nXFOIL Output:\n{last_stdout}"
            )
        return merged


def _run_case(case: dict) -> TabulatedAirfoil:
    """Run one batch case (a keyword dictionary for :func:`run_xfoil`)."""
    return run_xfoil(**dict(case))


def run_xfoil_batch(cases: list[dict], workers: int | None = None) -> list:
    """Run several XFOIL cases in parallel XFOIL processes.

    Parameters
    ----------
    cases : list[dict]
        One keyword dictionary for :func:`run_xfoil` per case.
    workers : int or None
        Thread pool size (each thread starts its own XFOIL subprocess).
        None uses the CPU count minus one.

    Returns
    -------
    list
        One entry per case, in input order: the TabulatedAirfoil, or the
        exception object when that case failed. A failed case does not stop
        the batch.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}.")
    out: list = [None] * len(cases)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_case, case) for case in cases]
        for index, future in enumerate(futures):
            try:
                out[index] = future.result()
            except Exception as exc:
                out[index] = exc
    return out
