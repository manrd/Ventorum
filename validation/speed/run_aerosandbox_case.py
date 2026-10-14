# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run one AeroSandbox vortex-lattice sweep of the speed harness.

This script runs in a subprocess (the interpreter comes from
``--aerosandbox-python`` or ``VENTORUM_AEROSANDBOX_PYTHON``) so the
AeroSandbox import time and threads stay out of the Ventorum process.
It reads ``case.json`` (see :mod:`validation.speed.case_spec`) from
its working folder, solves every angle with the documented
``VortexLatticeMethod`` analysis class, and writes ``result.json``.
The timed region covers only the solve loop; the import time is
measured separately and reported. It opens no window. Usage::

    aerosandbox-python run_aerosandbox_case.py

Settings (each from the public AeroSandbox documentation): the
vortex-lattice analysis class with documented spanwise and chordwise
resolution and cosine spacing (both spacing functions keep their
documented default, cosine); one operating point per angle at the
case velocity; symmetric wings built as right halves; reference
values from the case; default symmetric airfoil.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

#: Chordwise panels of every run (documented default of the analysis class).
CHORDWISE_RESOLUTION = 10


def build_airplane(spec: dict):
    """Build the AeroSandbox airplane of the case *spec*."""
    import aerosandbox as asb

    wings = []
    for surf in spec["surfaces"]:
        if not surf["symmetric"] or surf["mirror_y"]:
            raise ValueError(
                f"Surface {surf['name']!r} is not a symmetric right half; "
                "the harness maps only symmetric surfaces."
            )
        xsecs = []
        for station in surf["stations"]:
            xsecs.append(
                asb.WingXSec(
                    xyz_le=[float(v) for v in station["le_m"]],
                    chord=float(station["chord_m"]),
                )
            )
        wings.append(asb.Wing(name=str(surf["name"]), symmetric=True, xsecs=xsecs))
    return asb.Airplane(
        name=str(spec["case_id"]),
        xyz_ref=[0.0, 0.0, 0.0],
        wings=wings,
        s_ref=float(spec["s_ref_m2"]),
        c_ref=float(spec["c_ref_m"]),
        b_ref=float(spec["b_ref_m"]),
    )


def run_sweep(spec: dict) -> dict:
    """Solve every angle of *spec*. Return the result mapping."""
    t0 = time.perf_counter()
    import aerosandbox as asb

    import_done = time.perf_counter()
    airplane = build_airplane(spec)
    mesh = int(spec["mesh"])
    cl: list[float] = []
    cdi: list[float] = []
    angles = [float(a) for a in spec["angles_deg"]]
    for alpha in angles:
        op = asb.OperatingPoint(velocity=float(spec["v_inf_m_s"]), alpha=alpha)
        analysis = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op,
            spanwise_resolution=mesh,
            chordwise_resolution=CHORDWISE_RESOLUTION,
            # The symmetric option stays off: AeroSandbox 4.2.10 raises
            # NotImplementedError ("VLM with symmetry detection not yet
            # implemented") when it is on (tested 2026-10-14).
            run_symmetric_if_possible=False,
        )
        out = analysis.run()
        cl.append(float(out["CL"]))
        cdi.append(float(out["CD"]))
    t1 = time.perf_counter()
    return {
        "tool": "aerosandbox",
        "tool_version": str(asb.__version__),
        "case_id": spec["case_id"],
        "mesh": mesh,
        "angles_deg": angles,
        "CL": cl,
        "CDi": cdi,
        "elapsed_solve_s": t1 - import_done,
        "import_s": import_done - t0,
        "settings": {
            "analysis": "VortexLatticeMethod",
            "spanwise_resolution": mesh,
            "chordwise_resolution": CHORDWISE_RESOLUTION,
            "spanwise_spacing": "cosine (documented default)",
            "chordwise_spacing": "cosine (documented default)",
            "run_symmetric_if_possible": False,
            "airfoil": "default symmetric (documented default)",
            "velocity_m_s": float(spec["v_inf_m_s"]),
        },
    }


def main() -> int:
    """Entry point: read ``case.json``, write ``result.json``."""
    folder = Path.cwd()
    spec = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    result = run_sweep(spec)
    (folder / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"AeroSandbox {result['tool_version']}: {len(result['CL'])} points OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
