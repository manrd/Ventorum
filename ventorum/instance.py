# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Provide independent Ventorum instances and a manager that runs them in parallel.

The module supports:

1. Independent Ventorum instances. Each instance has its own geometry,
   flight condition, solver settings, case specification and worker pool.
2. Concurrent runs of many instances in threads. The instances do not
   share state.
3. Nested concurrency. Each instance can use many workers for its own
   computations (for example angle-of-attack sweeps).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Literal
from collections.abc import Sequence

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    FlightCondition,
    LiftingSurface,
    SolverResult,
    SolverSettings,
)
from ventorum.utils.progress import MultiInstanceProgressTracker


def _lift_to_drag(res: SolverResult) -> float:
    """Return the lift-to-drag ratio of one result (NaN if the drag is zero)."""
    tot = res.totals
    cd = tot.CD_total if tot.CD_total is not None else tot.CDi
    return float(tot.CL / cd) if cd > 1e-12 else float("nan")


class Ventorum:
    """An independent instance of the Ventorum solver.

    Each instance is a self-contained execution context that encapsulates its own
    aerodynamic geometry, flight conditions, solver settings, and dedicated internal
    worker pool. Multiple instances can execute concurrently in threads with
    guaranteed thread safety and complete absence of inter-instance interference.

    The instance keeps the results of its last run. ``last_run_mode`` is ``"single"``
    after :meth:`analyze`, ``"sweep"`` after :meth:`analyze_sweep`, and ``None``
    before the first run. Every run clears the results of the other mode, so
    :meth:`get_summary`, :meth:`get_polar` and :meth:`get_L_over_D` always return
    the results of the last run, in the mode of that run.

    Parameters
    ----------
    name : str
        Human-readable case/instance name.
    geometry : Aircraft or LiftingSurface or None
        The aerodynamic geometry for this instance. If a LiftingSurface is given,
        it is automatically wrapped in an Aircraft. If None, can be assigned later.
    condition : FlightCondition or None
        Baseline flight condition for this instance.
    settings : SolverSettings or None
        Solver settings (n_panels, solver_type, etc.). The instance keeps a
        copy. Only the arguments *solver*, *n_panels* and *use_symmetry* that
        you give explicitly (not None) change these settings.
    alpha_sweep_deg : np.ndarray or Sequence[float] or None
        If provided, configuring this instance to run an angle-of-attack sweep.
    n_jobs : int or str
        Number of cases in parallel for the sweeps of this instance
        (1 = serial within instance, >1 = parallel cases, -1 = one per
        core, ``"auto"`` = the automatic plan). The default is 1.
    backend : str
        Concurrency backend for this instance's internal workers: 'auto' or
        'thread' (ThreadPoolExecutor), or 'serial'.
    solver : str or None
        Solver algorithm: 'auto' (the vortex-lattice method, the same default
        as :func:`ventorum.analyze`), 'vlm', 'linear',
        'nonlinear' or 'fourier'. None: 'auto' if
        *settings* is None, otherwise the solver of *settings*.
    n_panels : int or None
        Panels per semi-span. None: 80 if *settings* is None, otherwise the
        value of *settings*.
    V_inf : float
        Free-stream velocity [m/s] (default 50.0).
    alpha_deg : float or None
        Baseline angle of attack in degrees (default 5.0).
    use_symmetry : bool or None
        Use the y = 0 symmetry plane. None: True if *settings* is None,
        otherwise the value of *settings*.
    case_id : str or None
        Unique identifier. If None, a short UUID is automatically assigned.
    metadata : dict or None
        Optional dictionary of user-defined tags, parameters, or descriptions.
    """

    def __init__(
        self,
        name: str = "Ventorum_Case",
        geometry: Aircraft | LiftingSurface | None = None,
        condition: FlightCondition | None = None,
        settings: SolverSettings | None = None,
        *,
        alpha_sweep_deg: np.ndarray | Sequence[float] | None = None,
        n_jobs: int | str = 1,
        backend: str = "auto",
        solver: str | None = None,
        n_panels: int | None = None,
        V_inf: float = 50.0,
        alpha_deg: float | None = None,
        use_symmetry: bool | None = None,
        case_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        from ventorum.utils.deprecation import warn_solver_alias

        warn_solver_alias(solver if solver is not None else (
            settings.solver_type if settings is not None else None))
        self.case_id = case_id or f"case_{uuid.uuid4().hex[:8]}"
        self.name = name

        self.n_jobs = 1 if n_jobs is None else n_jobs
        self.backend = "thread" if backend is None or backend.lower() == "auto" else backend.lower()

        self.metadata = dict(metadata or {})
        self._lock = threading.RLock()

        # Isolate and prepare geometry
        if geometry is not None:
            if isinstance(geometry, LiftingSurface):
                self.aircraft = Aircraft(name=f"{name}_Aircraft", surfaces=[geometry.clone()])
            else:
                self.aircraft = geometry.clone()
            self.aircraft.compute_reference_values()
        else:
            self.aircraft = None

        # Build flight condition
        if condition is not None:
            self.condition = condition.clone()
        else:
            alpha_rad = np.radians(alpha_deg) if alpha_deg is not None else np.radians(5.0)
            self.condition = FlightCondition(V_inf=V_inf, alpha=alpha_rad)

        # Build solver settings. The defaults apply only when no settings are given.
        if settings is not None:
            self.settings = settings.clone()
        else:
            self.settings = SolverSettings(solver_type="auto", n_panels=80, use_symmetry=True)
        if solver is not None:
            self.settings.solver_type = solver
        if n_panels is not None:
            self.settings.n_panels = int(n_panels)
        if use_symmetry is not None:
            self.settings.use_symmetry = bool(use_symmetry)

        # Sweep configuration
        if alpha_sweep_deg is not None:
            self.alpha_sweep_deg = np.asarray(alpha_sweep_deg, dtype=float)
        else:
            self.alpha_sweep_deg = None

        # Results & Execution status
        self.result: SolverResult | None = None  # Result of a single-point run
        self.sweep_results: list[SolverResult] | None = None
        self.sweep_alphas_deg: np.ndarray | None = None  # Angles [deg] of sweep_results
        self.last_run_mode: str | None = None  # 'single', 'sweep', or None before the first run
        self.last_alpha_deg: float | None = None  # Angle [deg] of the single-point run
        self.execution_time: float = 0.0
        self.status: str = "idle"  # 'idle', 'running', 'completed', 'failed'
        self.error: str | None = None

    def analyze(
        self,
        condition: FlightCondition | None = None,
        settings: SolverSettings | None = None,
        *,
        alpha_deg: float | None = None,
        V_inf: float | None = None,
        use_symmetry: bool | None = None,
    ) -> SolverResult:
        """Execute a single-point solve on this instance.

        The run becomes the result of the instance: ``last_run_mode`` is
        ``"single"`` and the results of an earlier sweep are cleared.
        """
        from ventorum import analyze as package_analyze

        if self.aircraft is None:
            raise ValueError(f"Instance '{self.name}' has no geometry assigned.")

        cond = condition.clone() if condition is not None else self.condition.clone()
        if alpha_deg is not None:
            cond.alpha = float(np.radians(alpha_deg))
        if V_inf is not None:
            cond.V_inf = float(V_inf)

        sett = settings.clone() if settings is not None else self.settings.clone()
        if use_symmetry is not None:
            sett.use_symmetry = bool(use_symmetry)
        res = package_analyze(self.aircraft, condition=cond, settings=sett)
        with self._lock:
            self.result = res
            self.sweep_results = None
            self.sweep_alphas_deg = None
            self.last_run_mode = "single"
            self.last_alpha_deg = float(np.degrees(cond.alpha))
        return res

    def analyze_sweep(
        self,
        alpha_deg_range: np.ndarray | Sequence[float] | None = None,
        *,
        V_inf: float | None = None,
        n_jobs: int | str | None = None,
        backend: str | None = None,
        progress: bool = False,
        use_symmetry: bool | None = None,
    ) -> list[SolverResult]:
        """Execute an angle-of-attack sweep using this instance's worker pool.

        Parameters
        ----------
        alpha_deg_range : array-like or None
            Array of angles of attack in degrees. If None, uses `self.alpha_sweep_deg`.
        V_inf : float or None
            Free-stream velocity [m/s].
        n_jobs : int, str or None
            Cases in parallel for this sweep. Defaults to `self.n_jobs`.
        backend : str or None
            Concurrency backend ('auto', 'thread' or 'serial'). Defaults to `self.backend`.
        progress : bool
            Whether to display terminal progress for this instance.
        use_symmetry : bool or None
            Override symmetry plane usage for this sweep.

        Notes
        -----
        The sweep becomes the result of the instance: ``last_run_mode`` is
        ``"sweep"`` and the result of an earlier single-point run is cleared.
        """
        from ventorum.visualization.drag_polar import alpha_sweep

        if self.aircraft is None:
            raise ValueError(f"Instance '{self.name}' has no geometry assigned.")

        alphas = (
            np.asarray(alpha_deg_range, dtype=float)
            if alpha_deg_range is not None
            else self.alpha_sweep_deg
        )
        if alphas is None:
            raise ValueError("No alpha range provided for sweep.")

        workers = n_jobs if n_jobs is not None else self.n_jobs
        b_end = backend.lower() if backend is not None else self.backend

        cond = self.condition.clone()
        if V_inf is not None:
            cond.V_inf = float(V_inf)

        sett = self.settings.clone()
        if use_symmetry is not None:
            sett.use_symmetry = bool(use_symmetry)

        alpha_rad = np.radians(alphas)
        results = alpha_sweep(
            self.aircraft,
            condition=cond,
            settings=sett,
            alpha_range=alpha_rad,
            n_jobs=workers,
            backend=b_end,
            progress=progress,
        )

        with self._lock:
            self.sweep_results = results
            self.sweep_alphas_deg = np.array(alphas, dtype=float)
            self.result = None
            self.last_run_mode = "sweep"
            self.last_alpha_deg = None
        return results

    def run(self, progress: bool = False) -> SolverResult | list[SolverResult]:
        """Execute the primary case configured for this instance.

        If `alpha_sweep_deg` is configured, runs a multi-angle sweep using
        `self.n_jobs`. Otherwise, executes a single-point analysis.
        Thread-safe, self-timed, and updates `self.status` and `self.execution_time`.
        """
        with self._lock:
            self.status = "running"
            self.error = None

        t0 = time.perf_counter()
        try:
            if self.alpha_sweep_deg is not None:
                res = self.analyze_sweep(
                    alpha_deg_range=self.alpha_sweep_deg,
                    n_jobs=self.n_jobs,
                    backend=self.backend,
                    progress=progress,
                )
            else:
                res = self.analyze()

            elapsed = time.perf_counter() - t0
            with self._lock:
                self.execution_time = elapsed
                self.status = "completed"
            return res

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.execution_time = elapsed
                self.status = "failed"
                self.error = str(exc)
            raise

    def get_summary(self) -> dict[str, Any]:
        """Return a structured summary of this instance's case and results.

        The results section describes the last run only: ``single_point``
        after :meth:`analyze` and ``sweep`` after :meth:`analyze_sweep`. Before
        the first run the summary has no results section.
        """
        with self._lock:
            summary: dict[str, Any] = {
                "case_id": self.case_id,
                "name": self.name,
                "status": self.status,
                "execution_time_s": self.execution_time,
                "n_jobs": self.n_jobs,
                "backend": self.backend,
                "solver_type": self.settings.solver_type,
                "n_panels": self.settings.n_panels,
                "error": self.error,
            }

            if self.aircraft is not None:
                summary["aircraft_name"] = self.aircraft.name
                summary["S_ref"] = self.aircraft.S_ref
                summary["b_ref"] = self.aircraft.b_ref
                summary["c_ref"] = self.aircraft.c_ref

            if self.last_run_mode == "single" and self.result is not None:
                tot = self.result.totals
                summary["single_point"] = {
                    "CL": tot.CL,
                    "CDi": tot.CDi,
                    "e": tot.e,
                    "Cl": tot.Cl,
                    "Cm": tot.Cm,
                    "Cn": tot.Cn,
                }

            elif self.last_run_mode == "sweep" and self.sweep_results and self.sweep_alphas_deg is not None:
                summary["sweep"] = {
                    "n_alphas": len(self.sweep_results),
                    "alpha_deg_min": float(np.min(self.sweep_alphas_deg)),
                    "alpha_deg_max": float(np.max(self.sweep_alphas_deg)),
                    "CL_max": max(r.totals.CL for r in self.sweep_results),
                    "CL_min": min(r.totals.CL for r in self.sweep_results),
                }

            return summary

    def get_polar(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (alphas_deg, CL, CDi) arrays of the last run.

        A single-point run gives arrays of length 1, at the angle of attack
        that was solved. The angles of a sweep are the angles of that sweep,
        which can be different from the configured ``alpha_sweep_deg``.

        Raises
        ------
        RuntimeError
            If the instance has not run yet.
        """
        with self._lock:
            if self.last_run_mode == "sweep" and self.sweep_results is not None and self.sweep_alphas_deg is not None:
                alphas = self.sweep_alphas_deg.copy()
                CLs = np.array([r.totals.CL for r in self.sweep_results])
                CDis = np.array([r.totals.CDi for r in self.sweep_results])
                return alphas, CLs, CDis
            if self.last_run_mode == "single" and self.result is not None:
                tot = self.result.totals
                return (
                    np.array([self.last_alpha_deg], dtype=float),
                    np.array([tot.CL], dtype=float),
                    np.array([tot.CDi], dtype=float),
                )
            raise RuntimeError(
                f"Instance '{self.name}' has no results. Call analyze(), analyze_sweep() or run() first."
            )

    def get_L_over_D(self) -> float | np.ndarray:
        """Return the lift-to-drag ratio L/D of the last run.

        A single-point run gives one float. A sweep gives one value per point
        of that sweep. A result without drag gives NaN.

        Raises
        ------
        RuntimeError
            If the instance has not run yet.
        """
        with self._lock:
            if self.last_run_mode == "sweep" and self.sweep_results is not None:
                return np.array([_lift_to_drag(r) for r in self.sweep_results])
            if self.last_run_mode == "single" and self.result is not None:
                return _lift_to_drag(self.result)
            raise RuntimeError(
                f"Instance '{self.name}' has no results. Call analyze(), analyze_sweep() or run() first."
            )

    def run_mesh_convergence(
        self,
        tolerance_pct: float = 0.5,
        *,
        cl_tolerance_pct: float | None = None,
        cdi_tolerance_pct: float | None = None,
        target_metric: Literal["both", "CL", "CDi", "circulation"] = "both",
        spacing_schemes: Sequence[str] | str = ("auto", "half-cosine", "cosine", "uniform"),
        panel_counts: Sequence[int] | None = None,
        ref_n_panels: int = 160,
        ref_spacing: str = "auto",
        proportional_panels: bool = True,
        robustness_margin: float = 0.8,
        evaluate_sweep: bool | None = None,
        progress: bool = False,
        apply_to_case: bool = False,
    ) -> Any:
        """Execute an automated mesh convergence study on this instance.

        Determines the minimal and recommended mesh parameters, verifies accuracy across
        the instance's flight conditions and sweep envelope, and provides generalizable guidelines.
        """
        from ventorum.geometry.mesh_convergence import run_mesh_convergence_study

        if self.aircraft is not None:
            self._ensure_reference_values()
        return run_mesh_convergence_study(
            case=self,
            tolerance_pct=tolerance_pct,
            cl_tolerance_pct=cl_tolerance_pct,
            cdi_tolerance_pct=cdi_tolerance_pct,
            target_metric=target_metric,
            spacing_schemes=spacing_schemes,
            panel_counts=panel_counts,
            ref_n_panels=ref_n_panels,
            ref_spacing=ref_spacing,
            proportional_panels=proportional_panels,
            robustness_margin=robustness_margin,
            evaluate_sweep=evaluate_sweep,
            progress=progress,
            apply_to_case=apply_to_case,
        )

    def _ensure_reference_values(self) -> None:
        """Fill *S_ref*, *b_ref* and *c_ref* if the aircraft does not have them yet.

        A solve calls ``compute_reference_values()`` on the aircraft of the
        instance (see :meth:`analyze`). A mesh convergence study reads these
        values from a copy of that aircraft, before it solves anything, so the
        study needs them on the aircraft of the instance.
        """
        with self._lock:
            ac = self.aircraft
            if ac is None:
                return
            if ac.S_ref is None or ac.b_ref is None or ac.c_ref is None:
                ac.compute_reference_values()

    def apply_mesh_recommendation(self, study: Any) -> Ventorum:
        """Apply the recommendations of a mesh convergence study to this instance.

        The method applies the recommended settings. It also applies the
        recommended panel counts to the surfaces that have their own
        ``n_panels``. It returns this instance.
        """
        with self._lock:
            self.settings = study.recommended_settings.clone()
            if self.aircraft is not None:
                for surf, n_s in zip(self.aircraft.surfaces, getattr(study, "recommended_surface_n_panels", [])):
                    if n_s is not None:
                        surf.n_panels = n_s
        return self

    def copy(self, deep: bool = True) -> Ventorum:
        """Create an independent copy of this instance."""
        with self._lock:
            new_inst = Ventorum(
                name=f"{self.name}_copy",
                geometry=self.aircraft.clone() if (deep and self.aircraft is not None) else self.aircraft,
                condition=self.condition.clone() if (deep and self.condition is not None) else self.condition,
                settings=self.settings.clone() if (deep and self.settings is not None) else self.settings,
                alpha_sweep_deg=self.alpha_sweep_deg.copy() if (deep and self.alpha_sweep_deg is not None) else self.alpha_sweep_deg,
                n_jobs=self.n_jobs,
                backend=self.backend,
                metadata=dict(self.metadata) if deep else self.metadata,
            )
            return new_inst

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"<Ventorum instance='{self.name}' id='{self.case_id}' status='{self.status}' "
                f"n_jobs={self.n_jobs} ({self.backend}) time={self.execution_time:.3f}s>"
            )


# Alias for explicit clarity
VentorumInstance = Ventorum


def run_parallel_instances(
    instances: Sequence[Ventorum],
    max_concurrent_instances: int | None = None,
    instance_backend: str = "thread",
    show_progress: bool = True,
    raise_on_error: bool = False,
) -> list[Ventorum]:
    """Execute multiple independent Ventorum instances concurrently in parallel.

    Each instance executes its designated aerodynamic case, using its own dedicated
    worker pool (`instance.n_jobs`), providing full hierarchical / nested
    parallelism with verified zero-crosstalk.

    Parameters
    ----------
    instances : Sequence[Ventorum]
        Collection of independent Ventorum instances to run.
    max_concurrent_instances : int or None
        Maximum number of instances executing simultaneously. Defaults to
        min(len(instances), os.cpu_count() or 1).
    instance_backend : str
        Concurrency backend for scheduling the instances: 'auto' or 'thread'
        (ThreadPoolExecutor), or 'serial'. A process pool is not available,
        because the instances keep their results in memory.
    show_progress : bool
        If True, displays a thread-safe synchronized multi-instance progress tracker.
    raise_on_error : bool
        If True, raises any exception encountered during an instance execution.
        If False (default), the failing instance records status='failed' and error,
        allowing other instances to complete.

    Returns
    -------
    list[Ventorum]
        The completed instances in original input order with results populated.
    """
    if not instances:
        return []

    total_instances = len(instances)
    if max_concurrent_instances is not None and max_concurrent_instances != "auto":
        max_workers = int(max_concurrent_instances)
    else:
        max_workers = min(total_instances, os.cpu_count() or 1)
    max_workers = max(1, max_workers)

    effective_backend = "thread" if instance_backend.lower() == "auto" else instance_backend.lower()
    if effective_backend not in ("thread", "serial"):
        raise ValueError(
            f"instance_backend={instance_backend!r}: use 'thread' or 'serial'. The instances keep their "
            "results in memory, so a process pool cannot return them."
        )

    tracker = (
        MultiInstanceProgressTracker(total_instances, title="Ventorum Instances")
        if show_progress
        else None
    )

    def _execute_instance(inst: Ventorum) -> Ventorum:
        if tracker:
            tracker.on_instance_start(inst.name)
        try:
            # Silence inner progress bars during concurrent instance runs to avoid screen flicker
            inst.run(progress=False)
        except Exception:
            if raise_on_error:
                raise
        finally:
            if tracker:
                tracker.on_instance_finish(inst.name)
        return inst

    from ventorum.utils.parallel import run_cases

    # Instances run side by side; each one runs its own cases inside its
    # worker, with the worker's share of the cores (no nested pools).
    if max_workers == 1 or effective_backend == "serial":
        for inst in instances:
            _execute_instance(inst)
    else:
        run_cases(_execute_instance, instances, None, max_workers)

    if tracker:
        tracker.finish("All instances executed.")

    return list(instances)


class VentorumCaseManager:
    """High-level batch manager and orchestrator for Ventorum multi-instance studies.

    Provides a convenient interface to register multiple aerodynamic cases,
    dispatch them across parallel instances with customizable worker allocations,
    and aggregate results into structured tables.
    """

    def __init__(self, name: str = "Ventorum_Study"):
        self.name = name
        self.instances: list[Ventorum] = []

    def create_case(
        self,
        name: str,
        geometry: Aircraft | LiftingSurface,
        alpha_sweep_deg: np.ndarray | Sequence[float] | None = None,
        n_jobs: int | str = 1,
        backend: str = "auto",
        condition: FlightCondition | None = None,
        settings: SolverSettings | None = None,
        solver: str | None = None,
        n_panels: int | None = None,
    ) -> Ventorum:
        """Construct and register a new Ventorum instance (see :class:`Ventorum`)."""
        inst = Ventorum(
            name=name,
            geometry=geometry,
            condition=condition,
            settings=settings,
            alpha_sweep_deg=alpha_sweep_deg,
            n_jobs=n_jobs,
            backend=backend,
            solver=solver,
            n_panels=n_panels,
        )
        self.instances.append(inst)
        return inst

    def run_all(
        self,
        max_concurrent_instances: int | None = None,
        instance_backend: str = "thread",
        show_progress: bool = True,
    ) -> list[Ventorum]:
        """Execute all registered instances in parallel."""
        return run_parallel_instances(
            self.instances,
            max_concurrent_instances=max_concurrent_instances,
            instance_backend=instance_backend,
            show_progress=show_progress,
        )

    def summary(self) -> list[dict[str, Any]]:
        """Return summary records for all cases."""
        return [inst.get_summary() for inst in self.instances]

    def summary_table(self) -> str:
        """Return a formatted text table summarizing all cases and their primary results."""
        headers = ["Case Name", "Status", "Solver", "Panels", "Time (s)", "CL", "CDi", "L/Di", "e"]
        rows: list[list[str]] = []
        for inst in self.instances:
            s = inst.get_summary()
            sp = s.get("single_point")
            sw = s.get("sweep")
            if sp:
                cl_str = f"{sp['CL']:.4f}"
                cdi_str = f"{sp['CDi']:.5f}"
                ld_str = f"{sp['CL'] / sp['CDi']:.2f}" if sp["CDi"] > 1e-12 else "N/A"
                e_str = f"{sp['e']:.3f}"
            elif sw:
                cl_str = f"[{sw['CL_min']:.2f}..{sw['CL_max']:.2f}]"
                cdi_str = "sweep"
                ld_str = "sweep"
                e_str = "sweep"
            else:
                cl_str, cdi_str, ld_str, e_str = "-", "-", "-", "-"

            rows.append([
                s["name"],
                s["status"],
                str(s.get("solver_type", "-")),
                str(s.get("n_panels", "-")),
                f"{s['execution_time_s']:.3f}",
                cl_str,
                cdi_str,
                ld_str,
                e_str,
            ])

        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        row_lines = [" | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) for r in rows]
        return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)

    def to_dataframe(self) -> Any:
        """Convert all case summaries to a pandas DataFrame if pandas is installed.

        Returns
        -------
        pandas.DataFrame or list[dict]
            A DataFrame of case metrics if pandas is available, otherwise raw summary dicts.
        """
        records = []
        for inst in self.instances:
            s = inst.get_summary()
            flat: dict[str, Any] = {
                "name": s["name"],
                "case_id": s["case_id"],
                "status": s["status"],
                "time_s": s["execution_time_s"],
                "workers": s["n_jobs"],
                "backend": s["backend"],
                "solver": s.get("solver_type"),
                "panels": s.get("n_panels"),
            }
            if "single_point" in s:
                sp = s["single_point"]
                flat["CL"] = sp["CL"]
                flat["CDi"] = sp["CDi"]
                flat["e"] = sp["e"]
                flat["L_over_Di"] = sp["CL"] / sp["CDi"] if sp["CDi"] > 1e-12 else float("nan")
                flat["Cm"] = sp["Cm"]
            if "sweep" in s:
                sw = s["sweep"]
                flat["sweep_n"] = sw["n_alphas"]
                flat["CL_max"] = sw["CL_max"]
                flat["CL_min"] = sw["CL_min"]
            records.append(flat)

        try:
            import pandas as pd
            return pd.DataFrame(records)
        except ImportError:
            return records

    def find_optimal(
        self,
        metric: str = "L_over_D",
        objective: str = "max",
    ) -> tuple[Ventorum, float, dict[str, Any]]:
        """Identify the optimal case in the study based on a specified aerodynamic metric.

        Parameters
        ----------
        metric : str
            Metric name ('L_over_D', 'CL', 'CDi', 'e').
        objective : str
            'max' or 'min'.

        Returns
        -------
        tuple[Ventorum, float, dict[str, Any]]
            (best_instance, best_metric_value, summary_dict)
        """
        best_inst = None
        best_val = -float("inf") if objective == "max" else float("inf")

        for inst in self.instances:
            if inst.status != "completed":
                continue
            val = None
            if metric == "L_over_D":
                ld = inst.get_L_over_D()
                val = float(np.max(ld)) if isinstance(ld, np.ndarray) else float(ld)
            elif metric == "CL":
                if inst.sweep_results:
                    val = float(max(r.totals.CL for r in inst.sweep_results))
                elif inst.result:
                    val = float(inst.result.totals.CL)
            elif metric == "CDi":
                if inst.sweep_results:
                    val = float(min(r.totals.CDi for r in inst.sweep_results))
                elif inst.result:
                    val = float(inst.result.totals.CDi)
            elif metric == "e":
                if inst.sweep_results:
                    val = float(max(r.totals.e for r in inst.sweep_results))
                elif inst.result:
                    val = float(inst.result.totals.e)

            if val is not None:
                if objective == "max" and val > best_val:
                    best_val = val
                    best_inst = inst
                elif objective == "min" and val < best_val:
                    best_val = val
                    best_inst = inst

        if best_inst is None:
            raise ValueError(f"No completed cases found with metric '{metric}'.")

        return best_inst, best_val, best_inst.get_summary()

    def get_all_polars(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Retrieve (alpha_deg, CL, CDi) polar curves for all sweep cases in the study."""
        polars = {}
        for inst in self.instances:
            if inst.sweep_results is not None:
                polars[inst.name] = inst.get_polar()
        return polars
