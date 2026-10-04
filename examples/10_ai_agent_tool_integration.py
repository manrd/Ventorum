# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Example 10: AI agent tools
==========================

This example shows the strict agent tools of Ventorum:

1. One analysis through ``call_tool`` with strict keys (units in the names).
2. A refused input: the key ``alpha`` has no unit, so the tool returns an error.
3. A polar sweep.
4. Stability derivatives about the centre of gravity (CG ahead of and behind
   the neutral point).
5. Ground effect with one height that strikes the ground.
6. A batch with one invalid candidate.

Every payload is strict JSON. The trust bands are heuristic; they are not
calibrated error bounds.
"""

import json

from ventorum.agent import call_tool, get_tool_schemas

FAST = {"n_panels": 20}
WING = {"span_m": 12.0, "root_chord_m": 1.6, "tip_chord_m": 0.8, "sweep_le_deg": 4.0,
        "tip_twist_deg": -2.0, "airfoil": {"cd0": 0.008}}


def show(title, payload):
    print(f"\n--- {title} ---")
    json.dumps(payload, allow_nan=False)  # The payload is strict JSON.
    if payload["status"] == "success":
        print(payload.get("executive_summary", "(no summary)"))
    else:
        print(f"ERROR [{payload['error']['type']}]: {payload['error']['message']}")
    return payload


def main():
    print("Tools:", ", ".join(s["name"] for s in get_tool_schemas("anthropic")))

    # 1. One analysis.
    p = show("1. Wing analysis", call_tool("ventorum_wing_analysis", {
        "wing": WING,
        "flight_condition": {"V_inf_m_s": 45.0, "alpha_deg": 5.0},
        "settings": FAST,
        "detail_level": "summary",
    }))
    t = p["trust"]
    print(f"Trust {t['score']} ({t['rating']}); heuristic bands (not calibrated): "
          f"{t['heuristic_bands_not_calibrated']}")

    # 2. A refused input.
    show("2. Refused input", call_tool("ventorum_wing_analysis", {
        "wing": WING, "flight_condition": {"alpha": 5.0}}))

    # 3. Polar sweep.
    p = show("3. Polar sweep", call_tool("ventorum_polar_sweep", {
        "wing": WING, "alpha_start_deg": -2.0, "alpha_end_deg": 10.0, "alpha_step_deg": 2.0,
        "flight_condition": {"V_inf_m_s": 45.0}, "settings": FAST, "detail_level": "standard",
    }))
    for row in p["polar_table"]:
        print(f"   alpha {row['alpha_deg']:5.1f} deg  CL {row['CL']:.4f}  CD {row['CD']:.5f}  L/D {row['L_over_D']}")

    # 4. Stability: CG ahead of and behind the neutral point.
    for x_cg in (0.2, 0.8):
        p = show(f"4. Stability, x_cg = {x_cg} m", call_tool("ventorum_stability_derivatives", {
            "wing": WING, "flight_condition": {"alpha_deg": 4.0}, "x_cg_m": x_cg, "settings": FAST,
            "detail_level": "summary",
        }))
        print("   Pitch:", p["stability_assessment"]["pitch"])

    # 5. Ground effect; the lowest height strikes the ground.
    p = show("5. Ground effect", call_tool("ventorum_ground_effect", {
        "wing": WING, "heights_m": [0.05, 0.5, 1.0, 2.0, 4.0], "alpha_deg": 4.0,
        "settings": FAST, "detail_level": "standard",
    }))
    for row in p["rows"]:
        if row["status"] == "ok":
            print(f"   h {row['h_m']:4.2f} m  CL ratio {row['CL_ratio']:.3f}  "
                  f"induced-drag factor ratio {row['induced_drag_factor_ratio']:.3f}  "
                  f"bank strike limit {row['phi_strike_limit_deg']} deg")
        else:
            print(f"   h {row['h_m']:4.2f} m  {row['status']}: {row['message']}")
    for row in p["irodov"]["rows"]:
        print(f"   Irodov h {row['h_m']:4.2f} m: margin {row['irodov_margin']} -> {row['verdict']}")

    # 6. Batch with one invalid candidate.
    p = show("6. Batch", call_tool("ventorum_batch_evaluate", {
        "candidates": [
            {"name": "AR6", "span_m": 6.0, "chord_m": 1.0, "airfoil": {"cd0": 0.008}},
            {"name": "AR10", "span_m": 10.0, "chord_m": 1.0, "airfoil": {"cd0": 0.008}},
            {"name": "bad", "span_m": 8.0, "chord_m": "1 m"},
        ],
        "flight_condition": {"V_inf_m_s": 40.0, "alpha_deg": 5.0},
        "objective": "max_L_over_D",
        "settings": FAST,
    }))
    for row in p["rankings"]:
        print(f"   #{row['rank']} {row['name']}: L/D {row['L_over_D']}")
    for f in p["failed"]:
        print(f"   failed: {f['name']} [{f['error']['type']}] {f['error']['message']}")


if __name__ == "__main__":
    main()
