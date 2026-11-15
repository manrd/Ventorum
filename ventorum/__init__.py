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
    NodeDisplacements,
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
from ventorum.core.error_bars import (  # noqa: F401
    ErrorBar,
    ErrorBarResult,
    combine_error_bars,
    numerical_error_bars,
)
from ventorum.core.input_uncertainty import input_error_bars  # noqa: F401
from ventorum.solvers import (  # noqa: F401
    BaseSolver,
    FourierSolver,
    VortexLatticeSolver,
    LinearLLTSolver,
    NonlinearLLTSolver,
)
from ventorum.solvers.trimming import (  # noqa: F401
    TrimResult,
    trim,
)
from ventorum.solvers.control_derivatives import (  # noqa: F401
    ControlDerivatives,
    ControlDerivativesResult,
    HingeMomentDerivatives,
    control_derivatives,
)
from ventorum.aero.hinge import (  # noqa: F401
    HingeMoment,
    hinge_moments,
)
from ventorum.solvers.rate_derivatives import (  # noqa: F401
    RateDerivativesResult,
    rate_axes_transform,
    rate_derivatives,
)
from ventorum.solvers.height_derivatives import (  # noqa: F401
    HeightDerivativesResult,
    height_derivatives,
)
from ventorum.solvers.static_derivatives import (  # noqa: F401
    StaticDerivativesResult,
    static_derivatives,
)
from ventorum.solvers.derivative_matrix import (  # noqa: F401
    DerivativeMatrixAllDict,
    DerivativeMatrixResult,
    derivative_matrix,
)
# The gradient functions load PyTorch: the module __getattr__ (end of this
# file) gives them on first access, so that ``import ventorum`` stays light.
from ventorum.sensitivity.errors import GradientScopeError  # noqa: F401
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
    V_inf: float | None = None,
    solver: str | None = None,
    n_panels: int | None = None,
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
        Angle of attack in **degrees** (convenience shorthand, default 5
        deg when *condition* is None).
    V_inf : float or None
        Free-stream velocity [m/s] (convenience shorthand, default 50 m/s
        when *condition* is None).
    solver : str or None
        ``"auto"`` (default) is always ``"vlm"``. Also ``"vlm"``,
        ``"linear"`` (lifting line), ``"nonlinear"`` (select
        it to use tabulated polars past the linear range) and ``"fourier"``.
        With *settings*, it replaces ``settings.solver_type``.
    n_panels : int or None
        Spanwise panels per semi-span (convenience shorthand, default 80
        when *settings* is None).
    use_symmetry : bool or None
        Use the y = 0 symmetry plane when the geometry and flow allow it.
        With *settings*, it replaces ``settings.use_symmetry``.

    Returns
    -------
    SolverResult

    Raises
    ------
    InputError
        A shorthand argument is given together with the object that holds
        the same value, and the two values are different: *alpha_deg* or
        *V_inf* with *condition* (``condition.alpha`` [rad],
        ``condition.V_inf`` [m/s]), or *n_panels* with *settings*
        (``settings.n_panels``). The object does not take the shorthand
        value, so the call is refused instead of ignoring the shorthand. An
        equal value is accepted.
    """
    from ventorum.solvers.factory import make_solver, resolve_solver_type
    from ventorum.utils.deprecation import canonical_settings
    from ventorum.utils.validation import validate_flight_condition, validate_solver_settings

    from ventorum.solvers.lattice_base import LatticeSolver

    aircraft = Aircraft(name="SingleWing", surfaces=[geometry]) if isinstance(geometry, LiftingSurface) else geometry
    _check_shorthand(condition, settings, alpha_deg, V_inf, n_panels)

    if condition is None:
        alpha_rad = np.radians(alpha_deg) if alpha_deg is not None else np.radians(5.0)
        condition = FlightCondition(V_inf=50.0 if V_inf is None else V_inf, alpha=alpha_rad)

    if settings is None:
        settings = SolverSettings(solver_type=solver or "auto", n_panels=80 if n_panels is None else n_panels)
    elif solver is not None:
        settings = settings.clone()
        settings.solver_type = solver
    if use_symmetry is not None:
        settings = settings.clone()
        settings.use_symmetry = bool(use_symmetry)

    validate_flight_condition(condition)
    settings = canonical_settings(settings)
    validate_solver_settings(settings)

    canonical = resolve_solver_type(settings.solver_type)
    solver_obj = make_solver(canonical)
    if isinstance(solver_obj, LatticeSolver):
        # The condition is checked above: the solver does not check it again,
        # so the warnings of the check are given once.
        result = solver_obj.solve(aircraft, condition, settings, _condition_checked=True)
    else:
        result = solver_obj.solve(aircraft, condition, settings)
    result.condition = condition
    return result


def _check_shorthand(condition: FlightCondition | None, settings: SolverSettings | None,
                     alpha_deg: float | None, V_inf: float | None, n_panels: int | None) -> None:
    """Refuse a shorthand argument of :func:`analyze` that the given object would override.

    Parameters
    ----------
    condition : FlightCondition or None
        Flight condition of the call.
    settings : SolverSettings or None
        Solver settings of the call.
    alpha_deg : float or None
        Shorthand angle of attack [deg].
    V_inf : float or None
        Shorthand free-stream speed [m/s].
    n_panels : int or None
        Shorthand spanwise panel count.

    Raises
    ------
    InputError
        When a shorthand value is given and differs from the value of the
        object that is used instead.
    """
    import math

    from ventorum.agent.schemas import InputError

    def _differ(a: float, b: float) -> bool:
        try:
            return not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15)
        except (TypeError, ValueError):
            return True

    if condition is not None:
        if alpha_deg is not None and _differ(np.radians(alpha_deg), condition.alpha):
            raise InputError(
                f"analyze: alpha_deg={alpha_deg!r} deg differs from condition.alpha="
                f"{float(np.degrees(condition.alpha))!r} deg. Give the angle in one place only: "
                "the condition is used, and alpha_deg would be ignored.")
        if V_inf is not None and _differ(V_inf, condition.V_inf):
            raise InputError(
                f"analyze: V_inf={V_inf!r} m/s differs from condition.V_inf={condition.V_inf!r} m/s. "
                "Give the speed in one place only: the condition is used, and V_inf would be ignored.")
    if settings is not None and n_panels is not None and _differ(n_panels, settings.n_panels):
        raise InputError(
            f"analyze: n_panels={n_panels!r} differs from settings.n_panels={settings.n_panels!r}. "
            "Give the panel count in one place only: the settings are used, and n_panels would be ignored.")


def _check_sweep_conditions(condition: FlightCondition, alpha_rad: np.ndarray) -> None:
    """Apply the flight-condition check of :func:`analyze` to the cases of a sweep.

    Parameters
    ----------
    condition : FlightCondition
        Base condition of the sweep.
    alpha_rad : numpy.ndarray
        Angles of attack of the sweep [rad].

    Raises
    ------
    ValueError
        With the same text as :func:`analyze` for the same value.
    """
    from ventorum.utils.validation import _check_angle, validate_flight_condition

    validate_flight_condition(condition)
    for a in np.atleast_1d(np.asarray(alpha_rad, dtype=float)):
        _check_angle("alpha", float(a))


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

    Raises
    ------
    ValueError
        The flight condition or an angle of the sweep is not valid (the
        same check and text as :func:`analyze`).
    """
    if isinstance(geometry, LiftingSurface):
        aircraft = Aircraft(name="SingleWing", surfaces=[geometry])
    else:
        aircraft = geometry

    condition = FlightCondition(V_inf=V_inf, alpha=0.0)
    alpha_range = np.radians(alpha_deg_range)
    _check_sweep_conditions(condition, alpha_range)

    if settings is None:
        settings = SolverSettings(solver_type=solver or "auto", n_panels=n_panels, spacing=spacing)
    elif solver is not None:
        settings = settings.clone()
        settings.solver_type = solver
    if use_symmetry is not None:
        settings = settings.clone()
        settings.use_symmetry = bool(use_symmetry)

    from ventorum.utils.deprecation import canonical_settings

    settings = canonical_settings(settings)

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

__version__ = "0.5.0"

# Names of ventorum.sensitivity that load PyTorch (given by __getattr__).
_LAZY_SENSITIVITY = ("GradientResult", "check_gradients", "gradient_vjp", "gradients")


def __getattr__(name: str):
    """Return a lazy gradient name, or an old solver class name with a deprecation warning."""
    if name in _LAZY_SENSITIVITY:
        import ventorum.sensitivity as sensitivity

        value = getattr(sensitivity, name)
        globals()[name] = value
        return value

    from ventorum.utils.deprecation import DEPRECATED_CLASS_NAMES, warn_deprecated_class

    if name in DEPRECATED_CLASS_NAMES:
        import ventorum.solvers as solvers

        warn_deprecated_class(name, DEPRECATED_CLASS_NAMES[name])
        return getattr(solvers, DEPRECATED_CLASS_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the module names, with the lazy gradient names included."""
    return sorted(set(globals()) | set(_LAZY_SENSITIVITY))
