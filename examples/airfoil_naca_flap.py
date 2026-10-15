# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""NACA 2412 with a 15 deg plain flap: inviscid Cp picture.

The script builds a NACA 2412 section, deflects a plain flap about a hinge
at 75 percent chord, solves the 2D inviscid flow at 5 deg angle of attack,
and saves the geometry picture and the Cp picture (flapped section against
the clean section) as PNG files in the output folder.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ventorum.airfoil import AirfoilShape
from ventorum.panel2d.solver import solve_inviscid
from ventorum.visualization.airfoil2d import plot_airfoil, plot_cp


def parse_args(argv=None):
    """Parse the output folder argument."""
    parser = argparse.ArgumentParser(
        description="NACA 2412 with plain flap: save geometry and Cp PNG files."
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
    """Run the flap case and save the PNG files. Return the file paths."""
    if outdir is None:
        outdir = tempfile.gettempdir()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    clean = AirfoilShape.naca("2412", n_points=161)
    flapped = clean.with_flap(x_hinge=0.75, deflection=15.0)
    alpha_deg = 5.0
    res_clean = solve_inviscid(clean.coords, alpha_deg, backend="numpy")
    res_flap = solve_inviscid(flapped.coords, alpha_deg, backend="numpy")

    print(f"NACA 2412 clean:  Cl = {float(res_clean.Cl):.4f}, "
          f"Cm = {float(res_clean.Cm):.4f}")
    print(f"NACA 2412 flap 15 deg: Cl = {float(res_flap.Cl):.4f}, "
          f"Cm = {float(res_flap.Cm):.4f}")

    _, ax = plt.subplots(figsize=(9.0, 3.8))
    plot_airfoil(clean, ax=ax)
    plot_airfoil(flapped, ax=ax)
    ax.set_title("NACA 2412 clean and with 15 deg plain flap (hinge x/c = 0.75)")
    geom_path = out / "naca2412_flap_geometry.png"
    ax.figure.savefig(geom_path, dpi=150)
    plt.close(ax.figure)

    _, ax = plt.subplots(figsize=(8.0, 5.0))
    plot_cp(res_flap, compare=res_clean, ax=ax)
    ax.set_title(
        f"NACA 2412 with 15 deg flap at alpha = {alpha_deg:.2f} deg "
        f"(flap Cl = {float(res_flap.Cl):.4f}, "
        f"clean Cl = {float(res_clean.Cl):.4f})"
    )
    cp_path = out / "naca2412_flap_cp.png"
    ax.figure.savefig(cp_path, dpi=150)
    plt.close(ax.figure)

    print(f"Wrote {geom_path}")
    print(f"Wrote {cp_path}")
    return [geom_path, cp_path]


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.outdir_opt or args.outdir)
