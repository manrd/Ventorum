# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Analyse the aerodynamics of wings and aircraft with Ventorum.

Vortex-lattice method (default), numerical lifting line (Phillips and
Snyder) and the classical Fourier lifting line, for incompressible, attached
flow, with flat-ground effect. See README.md for the valid envelope.

Quick start
-----------
>>> import ventorum as vt
>>> wing = vt.LiftingSurface(semi_span=5.0, sections=[
...     vt.WingSection(y_frac=0.0, chord=2.0),
...     vt.WingSection(y_frac=1.0, chord=1.0),
... ])
>>> result = vt.analyze(wing, alpha_deg=5.0)
>>> print(f"CL = {result.totals.CL:.4f}, CDi = {result.totals.CDi:.6f}")
"""

from __future__ import annotations

import numpy as np

# --- re-export public API ----------------------------------------------------
from ventorum.core.datatypes import (  # noqa: F401
    Aircraft,
    ControlSurface,
    DiscretizedSurface,
    FlightCondition,
    IntegratedResult,
    LinearAirfoil,
    LiftingSurface,
    SolverResult,
    SolverSettings,
    SpanwiseResult,
    TabulatedAirfoil,
    TrustScore,
    WingSection,
    aircraft_from_json,
    aircraft_to_json,
)
from ventorum.core.trust import evaluate_aerodynamic_trust  # noqa: F401
from ventorum.solvers import (  # noqa: F401
    BaseSolver,
    FourierSolver,
    HorseshoeSolver,
    VortexLatticeSolver,
    LinearLLTSolver,
    LinearSolver,
    NonlinearSolver,
)
from ventorum.solvers.trimming import (  # noqa: F401
    TrimResult,
    trim,
)
from ventorum.aero.polars import (  # noqa: F401
    load_xfoil_polar,
    load_csv_polar,
    blend_tabulated,
)
from ventorum.aero.loads import (  # noqa: F401
    compute_loads,
    trefftz_induced_drag,
)
from ventorum.aero.system import (  # noqa: F401
    GroundPlane,
    make_ground_plane,
    freestream_direction,
    ground_normal,
)
from ventorum.geometry.lattice import (  # noqa: F401
    VortexLattice,
    build_lattice,
)
from ventorum.solvers.lattice_base import assemble_system_matrix  # noqa: F401
from ventorum.geometry.processing import (  # noqa: F401
    discretize_surface,
    discretize_aircraft_surfaces,
)
from ventorum.geometry.discretization import (  # noqa: F401
    get_spacing,
    cosine_spacing,
    half_cosine_spacing,
    root_cosine_spacing,
    uniform_spacing,
    power_spacing,
    determine_optimal_spacing,
    compute_surface_n_panels,
)
from ventorum.aero.xfoil_runner import run_xfoil  # noqa: F401
from ventorum.visualization.distributions import (  # noqa: F401
    plot_lift_distribution,
    plot_cl_distribution,
    plot_alpha_distribution,
    plot_induced_drag_distribution,
    plot_all_distributions,
    plot_fourier_spectrum,
    plot_component_breakdown,
)
from ventorum.visualization.drag_polar import (  # noqa: F401
    alpha_sweep,
    plot_drag_polar,
    plot_cl_vs_alpha,
    plot_sweep_summary,
)
from ventorum.visualization.geometry_plot import (  # noqa: F401
    plot_geometry,
    plot_surface_results,
    plot_panel_results,
)
from ventorum.visualization.convergence import (  # noqa: F401
    plot_convergence,
    plot_mesh_convergence,
)
from ventorum.geometry.mesh_convergence import (  # noqa: F401
    run_mesh_convergence_study,
    MeshConvergenceResult,
    MeshConvergencePoint,
    GeneralizationGuideline,
)
from ventorum.visualization.wake_plot import (  # noqa: F401
    plot_vortex_wake,
    plot_trefftz_plane,
)
from ventorum.visualization.planform_plot import (  # noqa: F401
    plot_planform_2d,
    plot_span_loading,
)
from ventorum.visualization.performance_plot import (  # noqa: F401
    plot_efficiency_curves,
    plot_pitching_moment,
    plot_airfoil_polar,
)
from ventorum.visualization.dashboard import plot_aircraft_dashboard  # noqa: F401
from ventorum.instance import (  # noqa: F401
    Ventorum,
    VentorumInstance,
    VentorumCaseManager,
    run_parallel_instances,
)
from ventorum.hardware import tune_machine  # noqa: F401
from ventorum.agent import (  # noqa: F401
    call_tool,
    get_tool_schemas,
    wing_analysis,
    polar_sweep,
    ground_effect,
    stability_derivatives,
    batch_evaluate,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Top-level convenience function
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition | None = None,
    settings: SolverSettings | None = None,
    *,
    alpha_deg: float | None = None,
    V_inf: float = 50.0,
    solver: str | None = None,
    n_panels: int = 80,
    use_symmetry: bool | None = None,
) -> SolverResult:
    """One-call entry point for an analysis.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        If a single :class:`LiftingSurface` is given, it is wrapped into an
        :class:`Aircraft` automatically.
    condition : FlightCondition or None
        If *None*, one is built from *alpha_deg* and *V_inf*. If
        ``condition.h`` is set, the ground plane is ``h`` below
        ``Aircraft.ref_point`` (default: the origin), parallel to the free
        stream when phi = 0 (see :mod:`ventorum.aero.system`).
        For other height conventions use :func:`analyze_ground_effect`.
    settings : SolverSettings or None
        If *None*, defaults are used with *solver* and *n_panels*.
    alpha_deg : float or None
        Angle of attack in **degrees** (convenience shorthand, default 5).
    V_inf : float
        Free-stream velocity [m/s].
    solver : str or None
        ``"auto"`` (default) is always ``"vlm"``. Also ``"vlm"`` (alias
        ``"horseshoe"``), ``"linear"`` (lifting line), ``"nonlinear"`` (select
        it to use tabulated polars past the linear range) and ``"fourier"``.
    n_panels : int
        Spanwise panels per semi-span (when *settings* is None).
    use_symmetry : bool or None
        Use the y = 0 symmetry plane when the geometry and flow allow it.

    Returns
    -------
    SolverResult
    """
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.validation import validate_flight_condition, validate_solver_settings

    aircraft = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry

    if condition is None:
        alpha_rad = np.radians(alpha_deg) if alpha_deg is not None else np.radians(5.0)
        condition = FlightCondition(V_inf=V_inf, alpha=alpha_rad)

    if settings is None:
        settings = SolverSettings(solver_type=solver or "auto", n_panels=n_panels)
    elif solver is not None:
        settings = settings.clone()
        settings.solver_type = solver
    if use_symmetry is not None:
        settings = settings.clone()
        settings.use_symmetry = bool(use_symmetry)

    validate_flight_condition(condition)
    validate_solver_settings(settings)

    canonical = resolve_solver_type(settings.solver_type)
    result = make_solver(canonical).solve(aircraft, condition, settings)
    result.condition = condition
    return result


def analyze_sweep(
    geometry: Aircraft | LiftingSurface,
    alpha_deg_range: np.ndarray,
    V_inf: float = 50.0,
    solver: str | None = None,
    n_panels: int = 80,
    spacing: str = "auto",
    n_jobs: int | str = "auto",
    backend: str = "auto",
    progress: bool = False,
    settings: SolverSettings | None = None,
    use_symmetry: bool | None = None,
) -> list[SolverResult]:
    """Run the solver over a range of angles of attack.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
    alpha_deg_range : np.ndarray
        Array of angles of attack in degrees.
    V_inf : float
        Free-stream velocity [m/s].
    solver : str or None
    n_panels : int
    spacing : str
        Panel distribution (default ``"auto"``, the same as
        :class:`SolverSettings` and :func:`analyze`).
    n_jobs : int or str
        Number of parallel workers for the vortex-lattice solver (1 = serial,
        >1 = parallel, -1 = all cores, ``"auto"``). The lifting-line solvers
        solve all angles as one batch on all kernel threads; the nonlinear
        solver starts each angle from the solution of the previous one.
    backend : str
        Pool for the parallel cases: ``"auto"`` (threads), ``"thread"`` or
        ``"serial"``.
    progress : bool
        If True, displays live progress in the terminal.
    settings : SolverSettings or None
        Custom solver settings if desired.
    use_symmetry : bool or None
        Use the y = 0 symmetry plane when the geometry and flow allow it.

    Returns
    -------
    list[SolverResult]
    """
    if isinstance(geometry, LiftingSurface):
        aircraft = Aircraft(name="SingleWing", surfaces=[geometry])
    else:
        aircraft = geometry

    condition = FlightCondition(V_inf=V_inf, alpha=0.0)

    if settings is None:
        settings = SolverSettings(solver_type=solver or "auto", n_panels=n_panels, spacing=spacing)
    elif solver is not None:
        settings = settings.clone()
        settings.solver_type = solver
    if use_symmetry is not None:
        settings = settings.clone()
        settings.use_symmetry = bool(use_symmetry)

    alpha_range = np.radians(alpha_deg_range)

    return alpha_sweep(
        aircraft, condition, settings, alpha_range,
        n_jobs=n_jobs, backend=backend, progress=progress,
    )

from ventorum.ground_effect import (  # noqa: F401, E402
    GroundEffectCondition,
    GroundEffectResult,
    analyze_ground_effect,
    GroundEffectSweep,
    GroundEffectSweepResult,
    sweep_height,
    sweep_roll,
    sweep_alpha,
    plot_height_sweep,
    plot_roll_effect,
    plot_pitch_stability,
    plot_asymmetric_distributions,
    plot_ground_effect_matrix,
    plot_clearance_envelope,
)

__version__ = "0.3.0"

