# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Run one MachUpX lifting-line sweep of the speed harness.

This script runs in a subprocess (the interpreter comes from
``--machupx-python`` or ``VENTORUM_MACHUPX_PYTHON``) so the MachUpX
import time and threads stay out of the Ventorum process. It reads
``case.json`` (see :mod:`validation.speed.case_spec`) from its working
folder, solves every angle through the documented ``Scene`` class, and
writes ``result.json``. The timed region covers only the solve loop;
the import time is measured separately and reported. It opens no
window. Usage::

    machupx-python run_machupx_case.py

Settings (each from the public MachUpX documentation): a scene from a
dict input with SI units; the linear solver of the speed comparison;
default flat-plate airfoil (thin airfoil theory, no profile drag, so
the drag coefficient is the induced drag); cosine clustering of the
grid (documented default); mesh refinement by the documented grid
count ``N`` per semispan.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def build_aircraft_dict(spec: dict) -> dict:
    """Return the MachUpX aircraft dict of the case *spec*."""
    wings = {}
    main_id = 0
    main_key = ""
    for index, surf in enumerate(spec["surfaces"], start=1):
        if not surf["symmetric"] or surf["mirror_y"]:
            raise ValueError(
                f"Surface {surf['name']!r} is not a symmetric right half; "
                "the harness maps only symmetric surfaces."
            )
        key = f"wing_{index}"
        if not main_key or surf["semi_span_m"] > wings[main_key]["semispan"]:
            main_id = index
            main_key = key
        (root, tip) = surf["stations"]
        if float(surf["dihedral_deg"]) != 0.0:
            # The MachUpX documentation gives the body frame (x forward,
            # y right, z down) but not the sign of the dihedral input.
            raise ValueError(
                f"Surface {surf['name']!r} has dihedral; the MachUpX dihedral "
                "sign is not documented, so the harness maps only flat surfaces."
            )
        wings[f"wing_{index}"] = {
            "ID": index,
            "side": "both",
            "is_main": False,
            "semispan": float(surf["semi_span_m"]),
            "twist": 0.0,
            "dihedral": float(surf["dihedral_deg"]),
            "sweep": float(surf["sweep_resolved_deg"]),
            "chord": [[0.0, float(root["chord_m"])], [1.0, float(tip["chord_m"])]],
            "grid": {"N": int(spec["mesh"]), "distribution": "cosine_cluster"},
        }
    wings[main_key]["is_main"] = True
    for index, surf in enumerate(spec["surfaces"], start=1):
        if index == main_id:
            continue
        pos = [float(v) for v in surf["position_m"]]
        wings[f"wing_{index}"]["connect_to"] = {
            "ID": main_id,
            "location": "root",
            "dx": -pos[0],
            "dy": pos[1],
            "dz": -pos[2],
        }
    return {
        "CG": [0.0, 0.0, 0.0],
        "weight": 100.0,
        "reference": {
            "area": float(spec["s_ref_m2"]),
            "longitudinal_length": float(spec["c_ref_m"]),
            "lateral_length": float(spec["b_ref_m"]),
        },
        "wings": wings,
    }


def run_sweep(spec: dict, folder: Path) -> dict:
    """Solve every angle of *spec*. Return the result mapping."""
    t0 = time.perf_counter()
    import machupX as MX
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    import_done = time.perf_counter()
    try:
        tool_version = str(pkg_version("machupx"))
    except PackageNotFoundError:
        tool_version = "unknown"
    aircraft_path = str(folder / "aircraft.json")
    folder.joinpath("aircraft.json").write_text(
        json.dumps(build_aircraft_dict(spec), indent=1), encoding="utf-8"
    )
    cl: list[float] = []
    cdi: list[float] = []
    angles = [float(a) for a in spec["angles_deg"]]
    for alpha in angles:
        scene_input = {
            "solver": {"type": "linear"},
            "units": "SI",
            "scene": {
                "atmosphere": {},
                "aircraft": {
                    "harness": {
                        "file": aircraft_path,
                        "state": {
                            "velocity": float(spec["v_inf_m_s"]),
                            "alpha": alpha,
                            "beta": 0.0,
                        },
                    }
                },
            },
        }
        scene_path = folder / "scene.json"
        scene_path.write_text(json.dumps(scene_input, indent=1), encoding="utf-8")
        scene = MX.Scene(str(scene_path))
        total = scene.solve_forces(
            dimensional=False, non_dimensional=True, verbose=False
        )["harness"]["total"]
        cl.append(float(total["CL"]))
        cdi.append(float(total["CD"]))
    t1 = time.perf_counter()
    return {
        "tool": "machupx",
        "tool_version": tool_version,
        "case_id": spec["case_id"],
        "mesh": int(spec["mesh"]),
        "angles_deg": angles,
        "CL": cl,
        "CDi": cdi,
        "elapsed_solve_s": t1 - import_done,
        "import_s": import_done - t0,
        "settings": {
            "solver": "linear (speed comparison; the documentation "
            "recommends nonlinear for accuracy)",
            "units": "SI",
            "airfoil": "default flat plate (documented default)",
            "grid_N": int(spec["mesh"]),
            "grid_distribution": "cosine_cluster (documented default)",
            "velocity_m_s": float(spec["v_inf_m_s"]),
        },
    }


def main() -> int:
    """Entry point: read ``case.json``, write ``result.json``."""
    folder = Path.cwd()
    spec = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    result = run_sweep(spec, folder)
    (folder / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"MachUpX {result['tool_version']}: {len(result['CL'])} points OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
