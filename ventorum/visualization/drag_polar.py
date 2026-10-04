# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot the drag polar and the CL-alpha curve of a sweep."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    SolverResult,
    SolverSettings,
)


def alpha_sweep(
    aircraft: Aircraft,
    condition: FlightCondition,
    settings: SolverSettings,
    alpha_range: np.ndarray | None = None,
    n_jobs: int | str = "auto",
    backend: str = "auto",
    progress: bool = False,
) -> list[SolverResult]:
    """Run the solver over a range of angles of attack.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Base condition (speed, density, sideslip and height are kept; the
        angle of attack is replaced).
    settings : SolverSettings
    alpha_range : np.ndarray or None
        Angles of attack [rad]. Defaults to -2 deg to 12 deg in 1 deg steps.
    n_jobs : int or str
        Cases in parallel for the vortex-lattice solver (see
        :func:`ventorum.utils.parallel.plan_parallel`): ``"auto"`` uses the
        tuned plan of this machine or the default, and solves a small lattice
        (size class "small") as one batch (``solve_sweep``). The kernel
        threads of each case follow from it. The lifting-line solvers solve all angles
        as one batch (``solve_sweep``) on all kernel threads; the nonlinear
        solver starts each angle from the solution of the previous one.
    backend : {"auto", "thread", "serial"}
        Pool type (threads share the lattice).
    progress : bool
        If True, prints a progress bar to the terminal.

    Returns
    -------
    list[SolverResult]
        One result per angle of attack; moments are about ``aircraft.ref_point``.
    """
    from ventorum.hardware.profile import size_class
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.solvers.lattice_base import LatticeSolver
    from ventorum.utils.parallel import case_executor, plan_parallel

    if alpha_range is None:
        alpha_range = np.radians(np.arange(-2, 13, 1.0))
    alpha_range = np.asarray(alpha_range, dtype=float)
    b = (backend or "auto").lower()
    if b not in ("auto", "thread", "serial"):
        raise ValueError(f"backend={backend!r}: use 'auto', 'thread' or 'serial'.")

    canonical = resolve_solver_type(settings.solver_type)
    solver = make_solver(canonical)

    pb = None
    if progress:
        from ventorum.utils.progress import ProgressBar
        pb = ProgressBar(total=len(alpha_range), title=f"Alpha Sweep ({solver.name})", unit="pts")

    # The lifting-line solvers solve all angles as one batch (solve_sweep):
    # faster than a pool of workers, and each angle gets the same result as
    # a single solve.
    serial = (b == "serial" or not isinstance(solver, LatticeSolver) or condition.h is not None
              or getattr(solver, "collocation", "vlm") == "llt")
    if not serial:
        # One lattice for all angles; the plan depends on its size.
        from ventorum.utils.validation import validate_aircraft

        validate_aircraft(aircraft)
        aircraft.compute_reference_values()
        rp = aircraft.moment_reference()
        lattice = solver.build(aircraft, settings, condition, None, rp)
        plan = plan_parallel(len(alpha_range), lattice.n_panels, n_jobs)
        # A small lattice: one batch (solve_sweep) is faster than a pool of
        # workers; larger lattices gain from cases in parallel.
        auto = n_jobs is None or (isinstance(n_jobs, str) and n_jobs.lower() == "auto")
        serial = plan.workers == 1 or (auto and size_class(lattice.n_panels) == "small")
    if serial:
        results = solver.solve_sweep(aircraft, condition, settings, alpha_range)
        if pb:
            pb.update(len(alpha_range))
            pb.finish("Sweep complete")
        return results

    # Parallel: one lattice, the angles shared among the workers. The fixed
    # part of the vortex system is computed once (see solve_sweep).
    lattice.kernel_cache = {}

    def one(a: float) -> SolverResult:
        cond = FlightCondition(
            V_inf=condition.V_inf, alpha=float(a), beta=condition.beta,
            rho=condition.rho, h=condition.h, phi=getattr(condition, "phi", 0.0),
        )
        return solver.solve_lattice(lattice, cond, settings, aircraft.S_ref, aircraft.b_ref, aircraft.c_ref,
                                    ref_point=rp, main_surface=aircraft.main_surface_index())

    with case_executor(plan) as ex:
        results = list(ex.map(one, alpha_range))
    if pb:
        pb.update(len(alpha_range))
        pb.finish("Sweep complete")
    return results


def plot_drag_polar(
    results: list[SolverResult],
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot CL vs CDi (drag polar).

    Parameters
    ----------
    results : list[SolverResult]
        Results from an alpha sweep.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    CLs = [r.totals.CL for r in results]
    CDis = [r.totals.CDi for r in results]

    ax.plot(CDis, CLs, "o-", markersize=4, color="#2563eb", linewidth=2,
            label="CL vs CDi")

    # Add total drag if available
    if results[0].totals.CD_total is not None:
        CDs = [r.totals.CD_total for r in results]
        ax.plot(CDs, CLs, "s--", markersize=4, color="#dc2626", linewidth=1.5,
                label="CL vs CD_total")

    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title("Drag Polar")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cl_vs_alpha(
    results: list[SolverResult],
    alpha_range: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot CL vs α.

    Parameters
    ----------
    results : list[SolverResult]
    alpha_range : np.ndarray or None
        α values [rad] corresponding to *results*.
    ax : plt.Axes or None

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    CLs = [r.totals.CL for r in results]

    if alpha_range is not None:
        alphas_deg = np.degrees(alpha_range)
    else:
        alphas_deg = np.arange(len(CLs))

    ax.plot(alphas_deg, CLs, "o-", markersize=4, color="#2563eb", linewidth=2)

    # Annotate CL_alpha (slope from linear region)
    if len(CLs) >= 3:
        # Use central region
        mid = len(CLs) // 2
        lo, hi = max(0, mid - 2), min(len(CLs), mid + 3)
        if hi - lo >= 2:
            slope = np.polyfit(np.radians(alphas_deg[lo:hi]), CLs[lo:hi], 1)[0]
            ax.annotate(
                f"CL_α ≈ {slope:.3f} /rad\n({slope * np.pi / 180:.4f} /°)",
                xy=(alphas_deg[mid], CLs[mid]),
                xytext=(alphas_deg[mid] + 2, CLs[mid] - 0.15),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
            )

    ax.set_xlabel("α [°]")
    ax.set_ylabel("CL")
    ax.set_title("Lift Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep_summary(
    results: list[SolverResult],
    alpha_range: np.ndarray,
) -> Figure:
    """Create a 1×2 summary: CL-α and drag polar side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_cl_vs_alpha(results, alpha_range, ax=axes[0])
    plot_drag_polar(results, ax=axes[1])
    fig.suptitle("Alpha Sweep Summary", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
