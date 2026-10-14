# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Ventorum adapter of the speed harness (in-process reference)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import ventorum as vt
from ventorum.core.datatypes import FlightCondition, SolverSettings

from validation.speed.base import SweepResult, failed_result, median_seconds
from validation.speed.cases import N_CHORD, V_INF

#: Tool name in the reports.
TOOL_NAME = "ventorum"

#: This adapter is always available (no external program).
ENV_VARS: tuple[str, ...] = ()

#: Ventorum supports ground effect (kept for the runner hook).
SUPPORTS_GROUND_EFFECT = True

#: Spanwise meshes of the convergence study (per semi-span).
MESHES = [20, 40, 80, 160]


def is_available(explicit: str | None = None) -> bool:
    """Return True (Ventorum runs in this process)."""
    return True


def find_tool(explicit: str | None = None) -> None:
    """Return None (there is no external program to find)."""
    return None


def describe_mesh(mesh: int) -> str:
    """Return a human readable mesh label."""
    return f"n_panels {mesh}, n_chord {N_CHORD}"


def _settings(mesh: int) -> SolverSettings:
    """Return the solver settings of one mesh."""
    return SolverSettings(solver_type="vlm", n_panels=int(mesh), n_chord=N_CHORD)


def run_sweep(aircraft, mesh: int, angles_deg: list[float],
              workdir: str | Path | None = None, timeout: float | None = None) -> SweepResult:
    """Run one Ventorum sweep and return the result with its wall time [s].

    Parameters
    ----------
    aircraft : Aircraft
        Case geometry.
    mesh : int
        Spanwise panels per semi-span.
    angles_deg : list of float
        Sweep angles [deg].
    workdir : str, Path or None
        Unused (kept so all adapters share one call shape).
    timeout : float or None
        Unused (kept so all adapters share one call shape).

    Returns
    -------
    SweepResult
        CL [-] and CDi [-] per angle with the wall time [s].
    """
    angles = [float(a) for a in angles_deg]
    t0 = time.perf_counter()
    try:
        results = vt.analyze_sweep(
            aircraft, np.array(angles, dtype=float), settings=_settings(mesh),
        )
        cl = [float(r.totals.CL) for r in results]
        cdi = [float(r.totals.CDi) for r in results]
    except Exception as exc:  # noqa: BLE001 - a failed solve is a failed row
        from validation.speed.base import scrub_home

        return failed_result(TOOL_NAME, getattr(aircraft, "name", ""), describe_mesh(mesh),
                             angles, scrub_home(exc))
    elapsed = time.perf_counter() - t0
    return SweepResult(
        tool=TOOL_NAME, case_id=getattr(aircraft, "name", ""), mesh_label=describe_mesh(mesh),
        mesh=int(mesh), angles_deg=angles, CL=cl, CDi=cdi,
        ok=True, skipped=False, reason=None, elapsed_s=elapsed,
    )


def solve_point(aircraft, mesh: int, alpha_deg: float) -> tuple[float, float]:
    """Solve one point and return (CL [-], CDi [-])."""
    condition = FlightCondition(V_inf=V_INF, alpha=float(np.radians(alpha_deg)))
    result = vt.analyze(aircraft, condition=condition, settings=_settings(mesh))
    return float(result.totals.CL), float(result.totals.CDi)


def time_sweeps(aircraft, mesh: int, angles_deg: list[float],
                repeats: int) -> tuple[float, list[float]]:
    """Time *repeats* sweeps after one untimed warm-up. Return (median, samples)."""
    angles = [float(a) for a in angles_deg]
    run_sweep(aircraft, mesh, angles, None, None)
    samples: list[float] = []
    for _ in range(int(repeats)):
        samples.append(run_sweep(aircraft, mesh, angles, None, None).elapsed_s)
    return median_seconds(samples), samples


def check_available() -> SweepResult | None:
    """Return None (Ventorum never skips)."""
    return None
