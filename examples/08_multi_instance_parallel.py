# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Ventorum Example 8: Multi-Instance Parallel Execution with Nested Workers.

Demonstrates running multiple completely independent Ventorum instances simultaneously,
where each instance handles its own unique aerodynamic case and utilizes dedicated
internal worker threads to execute its computations, with verified zero interference.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import ventorum as vt


def main():
    print("=" * 80)
    print("Ventorum Example 8: Parallel Independent Instances with Nested Concurrency")
    print("=" * 80)

    # 1. Define 4 independent aerodynamic cases with distinct geometries and conditions
    alphas_deg = np.linspace(-2.0, 10.0, 25)  # 25 angles of attack per case

    # Case 1: High-AR Sailplane Wing
    wing_sailplane = vt.LiftingSurface(
        name="Sailplane Wing",
        semi_span=8.0,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.2),
            vt.WingSection(y_frac=0.7, chord=0.9),
            vt.WingSection(y_frac=1.0, chord=0.5, twist=np.radians(-1.5)),
        ],
    )

    # Case 2: Swept Wing with Washout
    wing_swept = vt.LiftingSurface(
        name="Swept Transport Wing",
        semi_span=6.5,
        sweep_le=np.radians(25.0),
        dihedral=np.radians(3.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=2.4),
            vt.WingSection(y_frac=1.0, chord=0.8, twist=np.radians(-2.5)),
        ],
    )

    # Case 3: Low-AR Cropped Delta Wing
    wing_delta = vt.LiftingSurface(
        name="Cropped Delta Wing",
        semi_span=3.5,
        sweep_le=np.radians(45.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=3.2),
            vt.WingSection(y_frac=1.0, chord=0.6),
        ],
    )

    # Case 4: General Aviation Wing with Dihedral
    wing_ga = vt.LiftingSurface(
        name="General Aviation Wing",
        semi_span=5.0,
        dihedral=np.radians(5.0),
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.6),
            vt.WingSection(y_frac=1.0, chord=1.1, twist=np.radians(-1.0)),
        ],
    )

    # 2. Package into independent Ventorum instances, each utilizing 2 internal workers
    # Total active threads: 4 instances * 2 workers = 8 concurrent compute workers
    instances = [
        vt.Ventorum("Sailplane", geometry=wing_sailplane, alpha_sweep_deg=alphas_deg, n_workers=2, n_panels=30),
        vt.Ventorum("Swept_Transport", geometry=wing_swept, alpha_sweep_deg=alphas_deg, n_workers=2, n_panels=30),
        vt.Ventorum("Cropped_Delta", geometry=wing_delta, alpha_sweep_deg=alphas_deg, n_workers=2, n_panels=30),
        vt.Ventorum("GA_Dihedral", geometry=wing_ga, alpha_sweep_deg=alphas_deg, n_workers=2, n_panels=30),
    ]

    print(f"\nConfigured {len(instances)} independent Ventorum instances:")
    for inst in instances:
        print(f" - [{inst.name}]: {len(alphas_deg)} alphas, {inst.n_workers} internal workers, N={inst.settings.n_panels}")

    # 3. Execute all instances concurrently in parallel
    print("\nExecuting instances in parallel...")
    t0 = time.perf_counter()
    completed = vt.run_parallel_instances(
        instances,
        max_concurrent_instances=4,
        instance_backend="thread",
        show_progress=True,
    )
    t_parallel = time.perf_counter() - t0
    total_solves = len(instances) * len(alphas_deg)

    print(f"\nExecution Finished in {t_parallel:.2f}s ({total_solves} aerodynamic points, {total_solves / t_parallel:.1f} solves/s)")

    # 4. Summary Table
    print("\n" + "=" * 80)
    print(f"{'Case Name':<20} | {'Status':<10} | {'Time (s)':>8} | {'CL max':>8} | {'CDi min':>8} | {'e (span eff)':>12}")
    print("-" * 80)
    for inst in completed:
        alphas_out, cls, cdis = inst.get_polar()
        cl_max = np.max(cls)
        cdi_min = np.min(cdis)
        e_val = inst.sweep_results[len(alphas_deg) // 2].totals.e
        print(f"{inst.name:<20} | {inst.status:<10} | {inst.execution_time:>8.3f} | {cl_max:>8.4f} | {cdi_min:>8.5f} | {e_val:>12.3f}")
    print("=" * 80)

    # 5. Plot the multi-case comparative drag polars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for inst, color in zip(completed, colors):
        alphas_out, cls, cdis = inst.get_polar()
        # CL vs Alpha
        ax1.plot(alphas_out, cls, "o-", label=inst.name, color=color, markersize=3, linewidth=1.8)
        # Drag Polar: CL vs CDi
        ax2.plot(cdis, cls, "s-", label=inst.name, color=color, markersize=3, linewidth=1.8)

    ax1.set_xlabel("Angle of Attack $\\alpha$ [deg]")
    ax1.set_ylabel("Lift Coefficient $C_L$")
    ax1.set_title("Multi-Case Lift Curves ($C_L$ vs $\\alpha$)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_xlabel("Induced Drag Coefficient $C_{Di}$")
    ax2.set_ylabel("Lift Coefficient $C_L$")
    ax2.set_title("Multi-Case Drag Polars ($C_L$ vs $C_{Di}$)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "multi_instance_drag_polars.png", dpi=150)
    print(f"\nSaved multi-case comparative polar plot to '{out_dir / 'multi_instance_drag_polars.png'}'")
    # plt.show()


if __name__ == "__main__":
    main()
