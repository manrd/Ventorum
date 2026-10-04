# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run XFOIL and read its polars as tabulated airfoils."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ventorum.aero.polars import load_xfoil_polar
from ventorum.core.datatypes import TabulatedAirfoil


def run_xfoil(
    airfoil: str,
    Re: float,
    alpha_start: float = -5.0,
    alpha_end: float = 15.0,
    alpha_step: float = 0.5,
    mach: float = 0.0,
    max_iter: int = 150,
    timeout: float | None = 300.0,
) -> TabulatedAirfoil:
    """
    Run XFOIL to generate a polar and return it as a TabulatedAirfoil.

    Parameters
    ----------
    airfoil : str
        The airfoil specification. If it starts with 'naca' (e.g., 'naca 2412'),
        XFOIL's internal NACA generator is used. Otherwise, it is treated as a
        file path to a coordinate file to LOAD.
    Re : float
        Reynolds number for the viscous analysis.
    alpha_start : float
        Starting angle of attack [deg].
    alpha_end : float
        Ending angle of attack [deg].
    alpha_step : float
        Angle of attack step size [deg].
    mach : float
        Mach number.
    max_iter : int
        Maximum viscous iterations per angle of attack.
    timeout : float or None
        Largest run time of XFOIL [s] (default 300). XFOIL can stop and
        wait for input when it does not converge; the run is then stopped.
        *None* means no limit.

    Returns
    -------
    TabulatedAirfoil
        The parsed aerodynamic polar.

    Raises
    ------
    FileNotFoundError
        If the 'xfoil' executable is not found in the system PATH.
    RuntimeError
        If XFOIL execution fails, takes longer than *timeout*, or fails to
        produce a polar file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        polar_file = tmp_path / "polar.txt"

        # Prepare XFOIL commands
        cmds = []

        # 1. Load Geometry
        if airfoil.lower().startswith("naca"):
            cmds.append(f"{airfoil}")
        else:
            # Assume it's a file path to coordinates
            cmds.append(f"load {airfoil}")

        # 2. Smooth/pane the airfoil
        cmds.append("pane")

        # 3. Enter OPER menu and set conditions
        cmds.extend([
            "oper",
            f"iter {max_iter}",
            f"mach {mach}",
            f"visc {Re}",
        ])

        # 4. Start polar accumulation
        cmds.extend([
            "pacc",
            str(polar_file),
            "",  # skip dump file
        ])

        # 5. Run alpha sweep
        cmds.append(f"aseq {alpha_start} {alpha_end} {alpha_step}")

        # 6. Stop accumulation, exit oper, quit
        cmds.extend([
            "pacc",
            "",
            "quit"
        ])

        cmd_str = "\n".join(cmds) + "\n"

        # Execute XFOIL
        try:
            result = subprocess.run(
                ["xfoil"],
                input=cmd_str,
                text=True,
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"XFOIL execution failed with exit code {e.returncode}:\n{e.stdout}\n{e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"XFOIL did not finish in {timeout} s and was stopped.") from e
        except FileNotFoundError as e:
            raise FileNotFoundError("XFOIL executable not found. Ensure 'xfoil' is in your system PATH.") from e

        # Verify and load polar
        if not polar_file.exists():
            raise RuntimeError(f"XFOIL did not generate a polar file. It likely failed to converge.\nXFOIL Output:\n{result.stdout}")

        polar = load_xfoil_polar(polar_file)

        # Rename the polar appropriately
        if airfoil.lower().startswith("naca"):
            polar.name = airfoil.upper()
        else:
            polar.name = Path(airfoil).stem

        # If XFOIL failed to converge for any points, the polar might be empty
        if len(polar.alpha) == 0:
            raise RuntimeError(f"XFOIL generated an empty polar file. It failed to converge for any point.\nXFOIL Output:\n{result.stdout}")

        return polar
