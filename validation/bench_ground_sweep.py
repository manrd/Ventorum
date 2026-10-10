# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Benchmark ground sweeps and Trefftz loads on the CPU."""

from __future__ import annotations

import subprocess
import time

import numpy as np

import ventorum as vt
from ventorum.ground_effect.sweep import GroundEffectSweep


def _get_git_commit() -> str:
    """Return the short git commit hash or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _median_time(fn, repeats: int = 5) -> float:
    """Return the median wall time (s) of *fn*, with one warm-up call."""
    fn()  # warm-up call
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _wing_tail() -> vt.Aircraft:
    """Return a wing and horizontal stabilizer aircraft."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    tail = vt.LiftingSurface(
        name="Tail",
        semi_span=1.5,
        position=np.array([3.5, 0.0, 0.5]),
        sections=[
            vt.WingSection(y_frac=0.0, chord=0.6),
            vt.WingSection(y_frac=1.0, chord=0.4),
        ],
    )
    return vt.Aircraft(name="WingTail", surfaces=[wing, tail])


def _symmetric_wing() -> vt.Aircraft:
    """Return a single symmetric rectangular wing."""
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=5.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    return vt.Aircraft(name="Wing", surfaces=[wing])


def main() -> None:
    """Run the benchmarks and print a Markdown table."""
    commit = _get_git_commit()
    version = vt.__version__

    wt_ac = _wing_tail()
    sym_wing = _symmetric_wing()

    alphas_9 = np.linspace(0.0, 8.0, 9)
    phis_1 = np.array([0.0])

    # (a) VLM ground effect sweep: 6 heights x 9 alphas x 1 phi
    heights_vlm = np.linspace(0.5, 3.0, 6)

    def run_vlm_20():
        sw = GroundEffectSweep(wt_ac, settings=vt.SolverSettings(solver_type="vlm", n_panels=20))
        return sw.run_sweep(heights_vlm, alphas_9, phis_1)

    def run_vlm_40():
        sw = GroundEffectSweep(wt_ac, settings=vt.SolverSettings(solver_type="vlm", n_panels=40))
        return sw.run_sweep(heights_vlm, alphas_9, phis_1)

    t_vlm_20 = _median_time(run_vlm_20)
    t_vlm_40 = _median_time(run_vlm_40)

    # (b) Linear lifting line sweep: heights at h/c >= 1
    heights_llt = np.linspace(1.5, 3.5, 6)

    def run_llt_20():
        sw = GroundEffectSweep(wt_ac, settings=vt.SolverSettings(solver_type="linear", n_panels=20))
        return sw.run_sweep(heights_llt, alphas_9, phis_1)

    def run_llt_40():
        sw = GroundEffectSweep(wt_ac, settings=vt.SolverSettings(solver_type="linear", n_panels=40))
        return sw.run_sweep(heights_llt, alphas_9, phis_1)

    t_llt_20 = _median_time(run_llt_20)
    t_llt_40 = _median_time(run_llt_40)

    # (c) 33 single analyze calls of a symmetric wing in free air
    alphas_33 = np.linspace(-2.0, 10.0, 33)

    def run_33_single():
        for a in alphas_33:
            vt.analyze(sym_wing, alpha_deg=float(a))

    t_trefftz_33 = _median_time(run_33_single)

    print("# Ground sweep and Trefftz load benchmark")
    print()
    print(f"- Ventorum version: `{version}`")
    print(f"- Git commit: `{commit}`")
    print(f"- Module file: `{vt.__file__}`")
    print()
    print("| Case | Solver | Panels | Grid | Time (ms) |")
    print("| --- | --- | --- | --- | ---:|")
    print(f"| Wing + tail sweep | VLM | 20 | 6 x 9 x 1 | {1e3 * t_vlm_20:.1f} |")
    print(f"| Wing + tail sweep | VLM | 40 | 6 x 9 x 1 | {1e3 * t_vlm_40:.1f} |")
    print(f"| Wing + tail sweep | LLT linear | 20 | 6 x 9 x 1 | {1e3 * t_llt_20:.1f} |")
    print(f"| Wing + tail sweep | LLT linear | 40 | 6 x 9 x 1 | {1e3 * t_llt_40:.1f} |")
    print(f"| Symmetric wing free air | VLM (default) | 80 | 33 solves | {1e3 * t_trefftz_33:.1f} |")


if __name__ == "__main__":
    main()
