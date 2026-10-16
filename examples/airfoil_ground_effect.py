# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""NACA 4412 in ground effect: Cl against h/c and Cp pictures.

The script solves the 2D inviscid flow over a NACA 4412 section at 4 deg
angle of attack at heights h/c of 1.0, 0.3 and 0.1 above the ground. It
saves the Cp picture of the three heights and the Cl against h/c picture
as PNG files in the output folder.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ventorum.airfoil import AirfoilShape
from ventorum.panel2d.solver import solve_inviscid
from ventorum.visualization.airfoil2d import plot_ground_effect_cp


def parse_args(argv=None):
    """Parse the output folder argument."""
    parser = argparse.ArgumentParser(
        description="NACA 4412 in ground effect: save Cp and Cl PNG files."
    )
    parser.add_argument(
        "outdir",
        nargs="?",
        default=None,
        help="Output folder for the PNG files (default: system temp folder).",
    )
    parser.add_argument(
        "--outdir",
        dest="outdir_opt",
        default=None,
        help="Output folder for the PNG files (same as the positional argument).",
    )
    return parser.parse_args(argv)


def main(outdir=None):
    """Run the ground-effect sweep and save the PNG files. Return the paths."""
    if outdir is None:
        outdir = tempfile.gettempdir()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    shape = AirfoilShape.naca("4412", n_points=161)
    alpha_deg = 4.0
    heights = [1.0, 0.3, 0.1]
    results = {}
    for height in heights:
        results[height] = solve_inviscid(
            shape.coords, alpha_deg, height=height, backend="numpy"
        )
        print(f"h/c = {height:.1f}: Cl = {float(results[height].Cl):.4f}, "
              f"Cm = {float(results[height].Cm):.4f}")

    _, ax = plt.subplots(figsize=(8.0, 5.0))
    plot_ground_effect_cp(results, ax=ax)
    ax.set_title(
        f"NACA 4412 ground effect on Cp at alpha = {alpha_deg:.2f} deg"
    )
    cp_path = out / "naca4412_ground_effect_cp.png"
    ax.figure.savefig(cp_path, dpi=150)
    plt.close(ax.figure)

    cls = np.array([float(results[h].Cl) for h in heights])
    _, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(heights, cls, "o-", color="#2563eb", linewidth=1.8)
    ax.set_xlabel("h/c [-]")
    ax.set_ylabel("Cl [-]")
    ax.set_title(
        f"NACA 4412 lift rise near the ground at alpha = {alpha_deg:.2f} deg"
    )
    ax.grid(True, alpha=0.3)
    cl_path = out / "naca4412_cl_vs_height.png"
    ax.figure.savefig(cl_path, dpi=150)
    plt.close(ax.figure)

    print(f"Wrote {cp_path}")
    print(f"Wrote {cl_path}")
    return [cp_path, cl_path]


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.outdir_opt or args.outdir)
