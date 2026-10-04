# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Original Ventorum solvers and acceleration paths (first upload, commit 9283a5f).

This package keeps the original code so that its optimisations can be
studied, measured and ported to the verified core. Known physics errors
(recovery audit): the horseshoe solver ignores camber and over-predicts
induced drag; the nonlinear solver gives about half the lift; the GPU
solvers repeat both errors. The linear lifting line and the Fourier solver
agree with theory within about 1 %. Do not use the results of the other
solvers until they are corrected.
"""

from __future__ import annotations

import numpy as np

# --- re-export public API ----------------------------------------------------
from ventorum.legacy.core.datatypes import (  # noqa: F401
    Aircraft,
    DiscretizedSurface,
    FlightCondition,
    FourierStations,
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
from ventorum.legacy.core.trust import evaluate_aerodynamic_trust  # noqa: F401
from ventorum.legacy.solvers import (  # noqa: F401
    BaseSolver,
    FourierSolver,
    HorseshoeSolver,
    LinearLLTSolver,
    LinearSolver,
    NonlinearSolver,
    GPUHorseshoeSolver,
    GPUNonlinearSolver,
    has_cuda,
)


def analyze(
    geometry: Aircraft | LiftingSurface,
    condition: FlightCondition | None = None,
    settings: SolverSettings | None = None,
    *,
    alpha_deg: float | None = None,
    V_inf: float = 50.0,
    solver: str | None = None,
    n_panels: int = 80,
    backend: str = "auto",
    use_symmetry: bool | None = None,
) -> SolverResult:
    """One-call entry point for an LLT analysis.

    Parameters
    ----------
    geometry : Aircraft or LiftingSurface
        If a single :class:`LiftingSurface` is given, it is wrapped into an
        :class:`Aircraft` automatically.
    condition : FlightCondition or None
        If *None*, one is built from *alpha_deg* and *V_inf*.
    settings : SolverSettings or None
        If *None*, defaults are used.  *solver* and *n_panels* override fields
        in the default settings.
    alpha_deg : float or None
        Angle of attack in **degrees** (convenience shorthand).
    V_inf : float
        Free-stream velocity [m/s].
    solver : str or None
        ``"horseshoe"``, ``"linear"`` (or ``"linear_llt"``), ``"fourier"``, or ``"gpu"`` / ``"gpu_nonlinear"``. If *None*,
        auto-selects based on geometry complexity.
    n_panels : int
        Panels per semi-span.
    backend : str
        Execution backend: ``"auto"`` (default), ``"gpu"``, ``"thread"``, or ``"process"``.
        When ``"auto"``, automatically dispatches to GPU for high discretizations (N >= 100)
        and retains optimized CPU BLAS for small meshes (N < 100) to avoid PCIe latency.
    use_symmetry : bool or None
        Whether to enforce the Y=0 symmetry plane to compute on half-mesh and mirror the other.
        If None, uses settings.use_symmetry (defaults to True, automatically deactivating on asymmetry).

    Returns
    -------
    SolverResult
    """
    # --- wrap single surface --------------------------------------------------
    if isinstance(geometry, LiftingSurface):
        aircraft = Aircraft(name="SingleWing", surfaces=[geometry])
    else:
        aircraft = geometry

    # --- build condition ------------------------------------------------------
    if condition is None:
        alpha_rad = np.radians(alpha_deg) if alpha_deg is not None else np.radians(5.0)
        condition = FlightCondition(V_inf=V_inf, alpha=alpha_rad)

    # --- build settings -------------------------------------------------------
    if settings is None:
        # Auto-select solver
        if solver is None:
            multi_surface = len(aircraft.surfaces) > 1
            has_sweep = any(
                abs(s.sweep_le) > np.radians(0.5) or
                any(sec.x_le is not None and abs(sec.x_le) > 1e-6
                    for sec in s.sections)
                for s in aircraft.surfaces
            )
            has_dihedral = any(
                abs(s.dihedral) > np.radians(0.5)
                for s in aircraft.surfaces
            )
            if multi_surface or has_sweep or has_dihedral:
                solver = "horseshoe"
            else:
                solver = "horseshoe"  # default to horseshoe for generality
        settings = SolverSettings(solver_type=solver, n_panels=n_panels)
    elif solver is not None:
        settings.solver_type = solver
    if use_symmetry is not None:
        settings.use_symmetry = bool(use_symmetry)

    from ventorum.legacy.utils.validation import validate_flight_condition, validate_solver_settings
    validate_flight_condition(condition)
    validate_solver_settings(settings)

    # --- detect nonlinear polars ----------------------------------------------
    has_tabulated = any(
        isinstance(sec.airfoil, TabulatedAirfoil)
        for surf in aircraft.surfaces
        for sec in surf.sections
    )

    # --- dispatch to the right solver with adaptive hybrid routing -----------
    from ventorum.legacy.utils.linalg import get_optimal_hardware_backend
    from ventorum.legacy.aero.gpu_influence import has_cuda

    n_surfaces = len(aircraft.surfaces)
    is_ge = condition.h is not None
    opt_backend = get_optimal_hardware_backend(
        n_panels=settings.n_panels,
        n_surfaces=n_surfaces,
        is_sweep=False,
        is_ground_effect=is_ge,
        is_nonlinear=has_tabulated,
        user_backend=backend,
    )
    use_gpu = (
        settings.solver_type in ("gpu", "gpu_horseshoe", "gpu_nonlinear")
        or opt_backend == "gpu"
    )

    if settings.solver_type == "fourier":
        from ventorum.legacy.solvers.fourier import FourierSolver
        result = FourierSolver().solve(aircraft, condition, settings)
    elif settings.solver_type in ("linear", "linear_llt"):
        from ventorum.legacy.solvers.linear import LinearLLTSolver
        result = LinearLLTSolver().solve(aircraft, condition, settings)
    elif settings.solver_type in ("horseshoe", "vlm", "gpu_horseshoe"):
        if use_gpu and has_cuda():
            from ventorum.legacy.solvers.gpu_horseshoe import GPUHorseshoeSolver
            result = GPUHorseshoeSolver().solve(aircraft, condition, settings)
        else:
            from ventorum.legacy.solvers.horseshoe import HorseshoeSolver
            result = HorseshoeSolver().solve(aircraft, condition, settings)
    elif has_tabulated:
        if use_gpu and has_cuda():
            from ventorum.legacy.solvers.gpu_nonlinear import GPUNonlinearSolver
            result = GPUNonlinearSolver().solve(aircraft, condition, settings)
        else:
            from ventorum.legacy.solvers.nonlinear import NonlinearSolver
            result = NonlinearSolver().solve(aircraft, condition, settings)
    else:
        if use_gpu and has_cuda():
            from ventorum.legacy.solvers.gpu_horseshoe import GPUHorseshoeSolver
            result = GPUHorseshoeSolver().solve(aircraft, condition, settings)
        else:
            from ventorum.legacy.solvers.horseshoe import HorseshoeSolver
            result = HorseshoeSolver().solve(aircraft, condition, settings)

    result.condition = condition
    return result


def analyze_sweep(
    geometry: Aircraft | LiftingSurface,
    alpha_deg_range: np.ndarray,
    V_inf: float = 50.0,
    solver: str | None = None,
    n_panels: int = 80,
    spacing: str = "cosine",
    n_jobs: int | str = "auto",
    backend: str = "auto",
    progress: bool = False,
    settings: SolverSettings | None = None,
    use_symmetry: bool | None = None,
) -> list[SolverResult]:
    """Run the solver over a range of angles of attack with adaptive acceleration.
    
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
        Panel distribution ('cosine' or 'uniform').
    n_jobs : int
        Number of parallel workers (1 = serial, >1 = parallel, -1 = all cores).
    backend : str
        Concurrency backend ('auto', 'gpu', 'thread', or 'process'). Defaults to 'auto'.
    progress : bool
        If True, displays live progress in the terminal.
    settings : SolverSettings or None
        Custom solver settings if desired.
    use_symmetry : bool or None
        Whether to enforce the Y=0 symmetry plane.
    
    Returns
    -------
    list[SolverResult]
    """
    from ventorum.legacy.drag_polar import alpha_sweep
    
    if isinstance(geometry, LiftingSurface):
        aircraft = Aircraft(name="SingleWing", surfaces=[geometry])
    else:
        aircraft = geometry
        
    condition = FlightCondition(V_inf=V_inf, alpha=0.0)
    
    if settings is None:
        if solver is None:
            solver = "horseshoe"
        settings = SolverSettings(solver_type=solver, n_panels=n_panels, spacing=spacing)
    elif solver is not None:
        settings.solver_type = solver
    if use_symmetry is not None:
        settings.use_symmetry = bool(use_symmetry)
        
    alpha_range = np.radians(alpha_deg_range)
    
    return alpha_sweep(
        aircraft, condition, settings, alpha_range,
        n_jobs=n_jobs, backend=backend, progress=progress,
    )

from ventorum.legacy.utils.benchmark import ( # noqa: F401
    run_performance_benchmark,
    run_sweep_benchmark,
    run_ground_effect_benchmark,
    run_multithread_benchmark,
    run_multi_instance_benchmark,
    run_gpu_benchmark,
    run_symmetry_benchmark,
)
from ventorum.legacy.core.symmetry import (  # noqa: F401
    can_use_symmetry,
    is_symmetry_compatible,
    mirror_discretized_surface,
    extract_half_mesh_surface,
    reconstruct_full_circulation,
    reconstruct_full_downwash,
)

from ventorum.legacy.hardware import (  # noqa: F401
    HardwareInfo,
    MachineConfig,
    scan_hardware,
    get_machine_fingerprint,
    tune_machine,
    load_machine_config,
    save_machine_config,
    clear_machine_config,
    is_current_machine_optimized,
    get_active_setting,
    is_autotune_disabled,
)


def ensure_machine_tuned(
    quick: bool = False,
    verbose: bool = True,
    force: bool = False,
) -> MachineConfig:
    """Ensure that the local machine has been scanned and calibrated.

    If no valid machine profile exists for this machine (or force=True), executes
    the hardware scanner and benchmark suite to establish machine-optimized defaults.
    """
    from ventorum.legacy.hardware import is_current_machine_optimized, load_machine_config, tune_machine
    if not force and is_current_machine_optimized():
        cfg, _ = load_machine_config(verbose_mismatch=False)
        if cfg is not None:
            return cfg
    return tune_machine(quick=quick, verbose=verbose, save=True)


__version__ = "0.1.0"


