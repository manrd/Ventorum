# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Build the OpenVSP .vsp3 file of one speed-harness case.

This script runs under the OpenVSP Python interpreter (``py -3.13``
on the harness machine, given through the ``VENTORUM_VSP_PYTHON``
variable). It reads the case description (see
:mod:`validation.speed.case_spec`), builds one WING geometry per
lifting surface with the documented tessellation settings, and writes
the ``.vsp3`` file plus a small JSON record with the tool version and
the applied settings. It opens no window. Usage::

    vsp-python vsp_make_geom.py --spec case.json --out model.vsp3 --info geom_info.json

Only documented OpenVSP API calls are used (wing creation and
tessellation as in the published ``Wing.vspscript`` and
``TestAnalysisVSPAERO.vspscript`` examples; X-Z planar symmetry as in
the published ``Swept_Wing_API`` tutorial).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_case(spec: dict, vsp3_path: str) -> dict[str, int]:
    """Build every surface of *spec* and write the .vsp3 file.

    Returns the applied spanwise tessellation per surface name.
    """
    import openvsp as vsp

    vsp.ClearVSPModel()
    tess: dict[str, int] = {}
    for surf in spec["surfaces"]:
        if not surf["symmetric"] or surf["mirror_y"]:
            raise ValueError(
                f"Surface {surf['name']!r} is not a symmetric right half; "
                "the harness maps only symmetric surfaces."
            )
        stations = surf["stations"]
        if len(stations) != 2:
            raise ValueError(
                f"Surface {surf['name']!r} has {len(stations)} stations; "
                "the harness maps only root and tip stations."
            )
        wid = vsp.AddGeom("WING", "")
        vsp.SetGeomName(wid, surf["name"])
        # Half model (right half only) when the analysis uses X-Z symmetry.
        vsp.SetParmVal(wid, "Sym_Planar_Flag", "Sym", 0.0 if spec.get("half_model") else 2.0)
        vsp.SetParmVal(wid, "CapUMinOption", "EndCap", 0)
        (root, tip) = stations
        vsp.SetParmVal(wid, "Span", "XSec_1", float(surf["semi_span_m"]))
        vsp.SetParmVal(wid, "Root_Chord", "XSec_1", float(root["chord_m"]))
        vsp.SetParmVal(wid, "Tip_Chord", "XSec_1", float(tip["chord_m"]))
        vsp.SetParmVal(wid, "Sweep", "XSec_1", float(surf["sweep_resolved_deg"]))
        vsp.SetParmVal(wid, "Sweep_Location", "XSec_1", 0.0)
        vsp.SetParmVal(wid, "Dihedral", "XSec_1", float(surf["dihedral_deg"]))
        count = int(surf["tess_u"])
        vsp.SetParmVal(wid, "SectTess_U", "XSec_1", count)
        vsp.Update()
        for index, station in enumerate(stations):
            vsp.SetParmVal(wid, "Twist_Location", f"XSec_{index}", 0.0)
            vsp.SetParmVal(wid, "Twist", f"XSec_{index}", float(station["twist_deg"]))
        vsp.Update()
        for index in range(2):
            vsp.SetParmVal(wid, "ThickChord", f"XSecCurve_{index}", 0.12)
            vsp.SetParmVal(wid, "Camber", f"XSecCurve_{index}", 0.0)
        vsp.Update()
        pos = surf["position_m"]
        vsp.SetParmVal(wid, "X_Rel_Location", "XForm", float(pos[0]))
        vsp.SetParmVal(wid, "Y_Rel_Location", "XForm", float(pos[1]))
        vsp.SetParmVal(wid, "Z_Rel_Location", "XForm", float(pos[2]))
        vsp.Update()
        tess[str(surf["name"])] = count
    vsp.WriteVSPFile(vsp3_path, vsp.SET_ALL)
    return tess


def main(argv: list[str] | None = None) -> int:
    """Entry point: read the spec, write the .vsp3 file and the record."""
    parser = argparse.ArgumentParser(description="Build the .vsp3 of one harness case.")
    parser.add_argument("--spec", required=True, help="Input case JSON file.")
    parser.add_argument("--out", required=True, help="Output .vsp3 file.")
    parser.add_argument("--info", required=True, help="Output JSON record file.")
    args = parser.parse_args(argv)

    import openvsp as vsp

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    tess = build_case(spec, args.out)
    info = {
        "tool": "vspaero",
        "tool_version": str(vsp.GetVSPVersion()),
        "case_id": spec["case_id"],
        "mesh": int(spec["mesh"]),
        "tess_u": tess,
        "s_ref_m2": float(spec["s_ref_m2"]),
        "b_ref_m": float(spec["b_ref_m"]),
        "c_ref_m": float(spec["c_ref_m"]),
    }
    Path(args.info).write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"Wrote {args.out} ({info['tool_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
