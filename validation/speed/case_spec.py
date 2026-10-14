# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Shared case description of the cross-tool speed harness.

This module converts a Ventorum aircraft into plain data (nested dicts
of floats, strings and booleans) that the external-tool helper scripts
read. It runs no tool and imports no tool package, so the unit tests
use it with fake tools only. Each helper script takes one JSON file of
this shape and writes back plain numbers.
"""

from __future__ import annotations

import math
from typing import Any

from ventorum.geometry.lattice import surface_reference_line

#: Free-stream velocity [m/s] of every case (same value as the cases module).
V_INF_M_S = 50.0

#: Smallest spanwise count used on a secondary surface (keeps panels usable).
MIN_SECONDARY_COUNT = 4


def is_uniform_angles(angles_deg: list[float], tol: float = 1e-9) -> bool:
    """Return True when *angles_deg* are uniformly spaced [deg].

    VSPAERO sweeps a uniform AlphaStart/AlphaEnd/AlphaNpts range in one
    run; any other angle list needs one single-point run per angle.
    """
    angles = [float(a) for a in angles_deg]
    if len(angles) < 3:
        return True
    step = (angles[-1] - angles[0]) / (len(angles) - 1)
    return all(
        abs((b - a) - step) <= tol * max(1.0, abs(step)) for a, b in zip(angles[:-1], angles[1:])
    )


def _resolved_stations(surf) -> list[dict[str, Any]]:
    """Return the resolved leading-edge stations of one surface."""
    import numpy as np

    secs = sorted(surf.sections, key=lambda s: s.y_frac)
    if len(secs) != 2:
        raise ValueError(
            f"Surface {surf.name!r} has {len(secs)} sections; "
            "the speed harness maps only two-section surfaces "
            "(root and tip). Add sections or split the surface."
        )
    eta = np.array([s.y_frac for s in secs], dtype=float)
    x_le, y, z = surface_reference_line(surf, eta)
    pos = np.asarray(surf.position, dtype=float)
    out = []
    for sec, xl, yy, zz in zip(secs, x_le, y, z):
        out.append(
            {
                "y_frac": float(sec.y_frac),
                "chord_m": float(sec.chord),
                "twist_deg": float(math.degrees(sec.twist + surf.incidence)),
                "le_m": [float(xl + pos[0]), float(yy + pos[1]), float(zz + pos[2])],
            }
        )
    return out


def _surface_sweep_deg(stations: list[dict[str, Any]]) -> float:
    """Return the leading-edge sweep angle [deg] from resolved stations."""
    (r, t) = stations
    dx = t["le_m"][0] - r["le_m"][0]
    dy = t["le_m"][1] - r["le_m"][1]
    if abs(dy) <= 1e-12:
        return 0.0
    return float(math.degrees(math.atan2(dx, abs(dy))))


def aircraft_spec(aircraft, mesh: int, angles_deg: list[float],
                  v_inf_m_s: float = V_INF_M_S) -> dict[str, Any]:
    """Return the plain-data description of one case.

    Parameters
    ----------
    aircraft : Aircraft
        Case geometry. Reference values are computed when missing.
    mesh : int
        Mesh key (meaning depends on the tool; see each adapter).
    angles_deg : list of float
        Sweep angles [deg].
    v_inf_m_s : float
        Free-stream velocity [m/s].

    Returns
    -------
    dict
        JSON-serialisable mapping with case id, mesh, angles, reference
        values and one entry per lifting surface (resolved stations).
    """
    if getattr(aircraft, "S_ref", None) is None:
        aircraft.compute_reference_values()
    surfaces = []
    for surf in aircraft.surfaces:
        stations = _resolved_stations(surf)
        surfaces.append(
            {
                "name": str(surf.name),
                "semi_span_m": float(surf.semi_span),
                "symmetric": bool(surf.is_symmetric),
                "mirror_y": bool(surf.mirror_y),
                "position_m": [float(v) for v in surf.position],
                "incidence_deg": float(math.degrees(surf.incidence)),
                "sweep_le_deg": float(math.degrees(surf.sweep_le)),
                "sweep_resolved_deg": _surface_sweep_deg(stations),
                "dihedral_deg": float(math.degrees(surf.dihedral)),
                "stations": stations,
            }
        )
    return {
        "case_id": str(getattr(aircraft, "name", "")),
        "mesh": int(mesh),
        "angles_deg": [float(a) for a in angles_deg],
        "v_inf_m_s": float(v_inf_m_s),
        "s_ref_m2": float(aircraft.S_ref),
        "b_ref_m": float(aircraft.b_ref),
        "c_ref_m": float(aircraft.c_ref),
        "surfaces": surfaces,
    }


def scaled_counts(spec: dict[str, Any], mesh: int) -> dict[str, int]:
    """Return a spanwise count per surface name.

    The largest surface (by semi-span) gets *mesh*; smaller surfaces get
    a count scaled by semi-span (uniform spanwise density, at least
    :data:`MIN_SECONDARY_COUNT`). The VSPAERO adapter uses this mapping
    (equal spanwise density keeps small-surface panels usable); the
    AeroSandbox and MachUpX adapters use one count everywhere.
    """
    spans = {s["name"]: float(s["semi_span_m"]) for s in spec["surfaces"]}
    if not spans:
        raise ValueError("The case has no lifting surfaces.")
    ref = max(spans.values())
    if ref <= 0.0:
        raise ValueError("The reference semi-span is not positive.")
    return {
        name: (int(mesh) if span >= ref * (1.0 - 1e-12)
               else max(MIN_SECONDARY_COUNT, int(round(mesh * span / ref))))
        for name, span in spans.items()
    }


def check_result_payload(payload: object, angles: list[float]) -> tuple[list[float], list[float], str, float]:
    """Return (CL, CDi, version, solve seconds) of a case payload.

    Raise ValueError when the mapping misses CL/CDi lists of the right
    length, holds non-numeric values, or misses the version or solve
    time. A bad payload is a failed run, never silent zeros.
    """
    if not isinstance(payload, dict):
        raise ValueError("result.json holds no mapping.")
    cl = payload.get("CL")
    cdi = payload.get("CDi")
    if not isinstance(cl, list) or not isinstance(cdi, list):
        raise ValueError("result.json holds no CL/CDi lists.")
    if len(cl) != len(angles) or len(cdi) != len(angles):
        raise ValueError(
            f"result.json holds {len(cl)}/{len(cdi)} points for {len(angles)} angles."
        )
    if any(not isinstance(v, (int, float)) for v in cl + cdi):
        raise ValueError("result.json holds non-numeric CL/CDi values.")
    version = payload.get("tool_version", "unknown")
    elapsed = payload.get("elapsed_solve_s", 0.0)
    if not isinstance(version, str) or not isinstance(elapsed, (int, float)):
        raise ValueError("result.json holds no version or solve time.")
    return [float(v) for v in cl], [float(v) for v in cdi], version, float(elapsed)
