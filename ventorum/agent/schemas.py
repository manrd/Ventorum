# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Strict input schemas and parsers for the agent tools.

Rules:

* Every key with a unit has the unit in its name (``span_m``, ``alpha_deg``).
* Unknown keys are refused. The error message gives the allowed keys.
* Only the schema keys are accepted. There are no aliases: ``span`` or
  ``chord`` (no unit) are refused with a hint to ``span_m`` or ``chord_m``.
  Keys without a clear unit or meaning (``alpha``, ``aoa``, ``washout``,
  ``aspect_ratio``) are also refused with a hint.
* The mesh has at most :data:`MAX_TOTAL_PANELS` vortex-lattice panels
  (spanwise strips of all surfaces x chordwise panels). A larger mesh is
  refused, and an automatic ``n_chord`` is reduced to stay at the limit.
* Numbers must be JSON numbers (int or float). Strings such as ``"60 m/s"``
  are refused. Booleans must be JSON booleans.

The JSON schemas that :func:`get_tool_schemas` exports come from the same
key tables that the parsers use.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import math
import re
from typing import Any, Literal

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    ControlSurface,
    FlightCondition,
    LinearAirfoil,
    LiftingSurface,
    NodeDisplacements,
    SolverSettings,
    WingSection,
)
from ventorum.core.constants import RHO_SL
from ventorum.core.errors import VentorumError
from ventorum.utils.validation import MIN_AREA_M2, MIN_CHORD_M, MIN_SPAN_M, MIN_SPEED_M_S


class InputError(VentorumError):
    """The tool input does not agree with the schema."""


# Largest mesh that a tool solves: spanwise strips of all surfaces (both halves) x chordwise panels.
# A dense vortex-lattice solve of this size takes up to some tens of seconds and a few hundred MB.
MAX_TOTAL_PANELS = 4000

# Largest half-mesh panel count that the 3D panel tool solves [-].
MAX_PANEL_PANELS = 15000

# Largest total work of one tool call, in work units. One solve of a
# lattice with N panels costs N^2 units. The work of a call is the sum
# over all solves that the call will run (sweep points, heights, mesh
# levels, batch cases, derivative steps). This is about 12 solves of
# 4000 panels. A call above this budget is refused before any solve.
MAX_CALL_WORK = 2.0e8

# Largest total work of one 3D panel tool call, in operations. The dense
# factorization costs N^3 / 3 operations for N solved panels per configuration (the half
# mesh when the solver folds the symmetry, else the full mesh).
MAX_PANEL_CALL_WORK = 2.0e12

# Smallest step of a polar sweep [deg].
MIN_ALPHA_STEP_DEG = 0.01

_MESH_LIMIT_TXT = (f"The total mesh (spanwise strips of all surfaces, both halves, x chordwise panels) "
                   f"must be at most {MAX_TOTAL_PANELS} panels; a larger mesh is refused.")


# ═══════════════════════════════════════════════════════════════════════════════
# Key tables (one source for the parsers and the exported JSON schemas)
# ═══════════════════════════════════════════════════════════════════════════════

def _num(desc: str, **extra: Any) -> dict[str, Any]:
    return {"type": "number", "description": desc, **extra}


def _int(desc: str, **extra: Any) -> dict[str, Any]:
    return {"type": "integer", "description": desc, **extra}


def _vec3(desc: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": desc}


AIRFOIL_PROPS: dict[str, dict[str, Any]] = {
    "a0_per_rad": _num("Section lift-curve slope [1/rad]. Default 2*pi.", exclusiveMinimum=0, maximum=15),
    "alpha_L0_deg": _num("Section zero-lift angle [deg]. Negative for positive camber. Default 0.",
                         minimum=-20, maximum=20),
    "cd0": _num("Section profile drag coefficient (constant). Default 0 (no profile drag).",
                minimum=0, maximum=1),
    "cm0": _num("Section pitching moment coefficient about the quarter chord. Default 0.",
                minimum=-1, maximum=1),
}

# Limits of the 2D airfoil tools. One source for the schemas, the tools
# and the agent guide.
#: Largest panel count of one airfoil solve [-].
MAX_AIRFOIL_PANELS = 1000
#: Largest number of 2D solves (angles x flaps x heights) of one polar call.
MAX_AIRFOIL_SOLVES = 61
#: Largest number of viscous iterations per angle [-].
MAX_AIRFOIL_ITER = 200
#: Smallest and largest height above the ground in chords [-].
MIN_AIRFOIL_H_C = 0.05
MAX_AIRFOIL_H_C = 500.0
#: Largest number of points of an input coordinate text [-].
MAX_AIRFOIL_INPUT_POINTS = 2000
#: Largest length of the section text [characters].
MAX_AIRFOIL_TEXT_CHARS = 200_000
#: Run-time limit of each XFOIL start of the polar tool [s]. Fixed by the server.
AIRFOIL_XFOIL_TIMEOUT_S = 120.0
#: Smallest and largest reference chord of the polar tool, in chords of the section [-].
MIN_AIRFOIL_CHORD_REF = 0.1
MAX_AIRFOIL_CHORD_REF = 10.0
#: Largest absolute coordinate of the moment reference point of the polar tool, in chords [-].
MAX_AIRFOIL_MOMENT_REF = 10.0

_SECTION_PROP = {
    "type": "string",
    "maxLength": MAX_AIRFOIL_TEXT_CHARS,
    "description": "NACA designation (for example 'NACA 2412') or coordinate text in Selig or "
                   f"Lednicer format (at most {MAX_AIRFOIL_INPUT_POINTS} points).",
}

#: Keys of the plain-flap object of the airfoil tools.
AIRFOIL_FLAP_PROPS: dict[str, dict[str, Any]] = {
    "x_hinge": _num("Hinge position, fraction of the chord [-].", minimum=0.05, maximum=0.98),
    "y_hinge": _num("Hinge height, fraction of the local thickness [-]: 0 lower surface, "
                    "1 upper surface. Default 0.5.", minimum=0.0, maximum=1.0),
    "deflection_deg": _num("Flap deflection [deg], positive trailing edge down.",
                           minimum=-60.0, maximum=60.0),
}

_FLAP_PROP = {
    "type": "object",
    "description": "Optional plain flap: x_hinge [-], y_hinge [-] (fraction of the local thickness) "
                   "and deflection_deg [deg].",
    "properties": AIRFOIL_FLAP_PROPS,
    "required": ["x_hinge", "deflection_deg"],
    "additionalProperties": False,
}

AIRFOIL_GEOMETRY_PROPS: dict[str, dict[str, Any]] = {
    "section": _SECTION_PROP,
    "n_panels": _int("Optional panel count for repaneling [-].", minimum=10, maximum=MAX_AIRFOIL_PANELS),
    "flap": _FLAP_PROP,
}

AIRFOIL_POLAR_PROPS: dict[str, dict[str, Any]] = {
    "section": _SECTION_PROP,
    "Re": _num("Reynolds number [-].", minimum=1.0, maximum=1.0e9),
    "alphas_deg": {
        "type": "array",
        "items": {"type": "number", "minimum": -30.0, "maximum": 30.0},
        "minItems": 1,
        "maxItems": MAX_AIRFOIL_SOLVES,
        "description": "List of angles of attack [deg]. Give this list or the three range keys.",
    },
    "alpha_start_deg": _num("First angle of attack [deg].", minimum=-30.0, maximum=30.0),
    "alpha_end_deg": _num("Last angle of attack [deg].", minimum=-30.0, maximum=30.0),
    "alpha_step_deg": _num(f"Step [deg], at least {MIN_ALPHA_STEP_DEG}.", minimum=MIN_ALPHA_STEP_DEG, maximum=30.0),
    "flap": _FLAP_PROP,
    "h_c": _num("Height of the quarter chord above the ground, in chords [-]. Omit for free air. "
                "Only source 'ventorum2d'.", minimum=MIN_AIRFOIL_H_C, maximum=MAX_AIRFOIL_H_C),
    "source": {
        "type": "string",
        "enum": ["ventorum2d", "xfoil"],
        "default": "ventorum2d",
        "description": "Polar solver: 'ventorum2d' (default) or 'xfoil' (the XFOIL of the server).",
    },
    "n_panels": _int("Panel count [-]. Default 160.", minimum=10, maximum=MAX_AIRFOIL_PANELS),
    "ncrit": _num("Critical amplification exponent of free transition [-]. Default 9.", minimum=1.0, maximum=20.0),
    "xtr_top": _num("Forced transition on the upper surface, fraction of the chord [-]. Default 1 (free).",
                    minimum=0.0, maximum=1.0),
    "xtr_bottom": _num("Forced transition on the lower surface, fraction of the chord [-]. Default 1 (free).",
                       minimum=0.0, maximum=1.0),
    "max_iter": _int("Viscous iterations per angle [-]. Default 50 (ventorum2d) or 100 (xfoil).",
                     minimum=1, maximum=MAX_AIRFOIL_ITER),
    "chord_ref": _num("Reference chord of the coefficients and of the Reynolds number, in chords of the "
                      "section [-]. Default: the chord of the section (with a flap, the chord before the "
                      "deflection). Only source 'ventorum2d'.",
                      minimum=MIN_AIRFOIL_CHORD_REF, maximum=MAX_AIRFOIL_CHORD_REF),
    "moment_ref": {
        "type": "array",
        "items": {"type": "number", "minimum": -MAX_AIRFOIL_MOMENT_REF, "maximum": MAX_AIRFOIL_MOMENT_REF},
        "minItems": 2,
        "maxItems": 2,
        "description": "Moment reference point [x, y] in chords [-], in the frame of the normalized section "
                       "(leading edge at the origin, trailing-edge midpoint at (1, 0)). Default: the "
                       "quarter-chord point on the chord line (with a flap, of the section before the "
                       "deflection). Only source 'ventorum2d'.",
    },
}


SECTION_PROPS: dict[str, dict[str, Any]] = {
    "y_frac": _num("Span station as a fraction of the semi-span: 0 = root, 1 = tip.", minimum=0, maximum=1),
    "chord_m": _num("Local chord [m].", exclusiveMinimum=0),
    "twist_deg": _num("Local geometric twist [deg]. Positive = leading edge up. Default 0.",
                      minimum=-30, maximum=30),
    "x_le_m": _num("Leading-edge x offset from the root leading edge [m], x aft. "
                   "Default: from sweep_le_deg."),
    "z_le_m": _num("Leading-edge z offset from the root leading edge [m], z up. "
                   "Default: from dihedral_deg. If set, the semi-span is the projected (y) span."),
    "airfoil": {"type": "object", "properties": AIRFOIL_PROPS, "additionalProperties": False,
                "description": "Linear airfoil model of this section. Default: the surface airfoil."},
    "shape": {"type": "string",
              "description": "Airfoil section shape (e.g. NACA designation '0012' or '2412'). Default: NACA 0012."},
}

CONTROL_PROPS: dict[str, dict[str, Any]] = {
    "name": {"type": "string",
             "description": "Control name (string, no unit). Required. Must be unique on the surface. "
                            "Controls on different surfaces with the same name move together."},
    "eta_start": _num("Start of the control as a fraction of the semi-span [-]: 0 = root, 1 = tip. "
                      "Default 0.", minimum=0, maximum=1),
    "eta_end": _num("End of the control as a fraction of the semi-span [-]: 0 = root, 1 = tip. "
                    "Default 1.", minimum=0, maximum=1),
    "hinge_x_c": {"type": "number", "exclusiveMinimum": 0, "maximum": 1,
                  "description": "Hinge position as a fraction of the local chord from the leading edge [-]. "
                                 "Default 0.75. Must be larger than 0 and smaller than 1."},
    "deflection_deg": _num("Deflection [deg]. Positive = right-hand rotation about the hinge axis "
                           "(trailing edge down on a horizontal surface, trailing edge to +y on a fin). "
                           "Range -30 to 30. Default 0.", minimum=-30, maximum=30),
    "symmetric": {"type": "boolean",
                  "description": "True (default, boolean): the same deflection on both halves (flap, "
                                 "elevator). False: the opposite deflection on the left copy (aileron)."},
}

_DISP_VECTOR = {
    "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
    "description": "One displacement [dx, dy, dz] in metres [m], geometry axes (x aft, y right, z up).",
}

NODE_DISPLACEMENT_PROPS: dict[str, dict[str, Any]] = {
    "le_m": {"type": "array", "minItems": 2, "maxItems": 401, "items": _DISP_VECTOR,
             "description": "Leading-edge node displacements in metres [m]: one [dx, dy, dz] triple per "
                            "strip edge of the defining half, from the root to the tip. Must have the same "
                            "length as te_m. The number of items must equal the number of strip edges of the "
                            "defining half for the solver and settings of the call; use "
                            "ventorum_undeformed_nodes to get them."},
    "te_m": {"type": "array", "minItems": 2, "maxItems": 401, "items": _DISP_VECTOR,
             "description": "Trailing-edge node displacements in metres [m]: one [dx, dy, dz] triple per "
                            "strip edge of the defining half, from the root to the tip. Must have the same "
                            "length as le_m. The number of items must equal the number of strip edges of the "
                            "defining half for the solver and settings of the call; use "
                            "ventorum_undeformed_nodes to get them."},
    "eta": {"type": "array", "minItems": 2, "maxItems": 401, "items": {"type": "number"},
            "description": "Spanwise stations [-] as fractions of the semi-span, from "
                           "ventorum_undeformed_nodes. Optional. When given, the solver checks that the mesh "
                           "has the same stations."},
}

SURFACE_PROPS: dict[str, dict[str, Any]] = {
    "name": {"type": "string", "description": "Surface name."},
    "span_m": _num("Tip-to-tip span [m] of a symmetric surface, measured along the dihedral line. "
                   "Give span_m or semi_span_m, not both. Not allowed when symmetric is false.",
                   exclusiveMinimum=0),
    "semi_span_m": _num("Root-to-tip length [m] of one half, measured along the dihedral line.",
                        exclusiveMinimum=0),
    "chord_m": _num("Constant chord [m] (rectangular planform). Do not combine with "
                    "root_chord_m, tip_chord_m or sections.", exclusiveMinimum=0),
    "root_chord_m": _num("Root chord [m]. Give it with tip_chord_m.", exclusiveMinimum=0),
    "tip_chord_m": _num("Tip chord [m]. Give it with root_chord_m.", exclusiveMinimum=0),
    "sections": {
        "type": "array", "minItems": 2, "maxItems": 50,
        "items": {"type": "object", "properties": SECTION_PROPS,
                  "required": ["y_frac", "chord_m"], "additionalProperties": False},
        "description": "Explicit sections from root (y_frac 0) to tip (y_frac 1). Replaces chord_m, "
                       "root_chord_m, tip_chord_m and tip_twist_deg.",
    },
    "sweep_le_deg": _num("Leading-edge sweep [deg]. Default 0.", minimum=-75, maximum=75),
    "dihedral_deg": _num("Dihedral [deg], positive = tip up. Default 0. 90 with symmetric false is a fin.",
                         minimum=-90, maximum=90),
    "tip_twist_deg": _num("Tip twist relative to the root [deg], linear along the span. Positive = leading "
                          "edge up (wash-in). Washout is a NEGATIVE value. Default 0.",
                          minimum=-30, maximum=30),
    "incidence_deg": _num("Surface incidence [deg], positive = leading edge up. Default 0.",
                          minimum=-30, maximum=30),
    "position_m": _vec3("Root leading-edge position [x, y, z] in metres, geometry axes (x aft, y right, "
                        "z up). Default [0, 0, 0]. A symmetric surface must have y = 0."),
    "symmetric": {"type": "boolean",
                  "description": "True (default): the surface has a left and a right half. False: one "
                                 "half only (for example a fin)."},
    "mirror": {"type": "boolean",
               "description": "True: also add the mirror image (y -> -y) of this surface. Needs "
                              "symmetric false. Default false."},
    "n_panels": _int("Spanwise panels per semi-span for this surface. Default: the settings value. "
                     + _MESH_LIMIT_TXT, minimum=4, maximum=400),
    "airfoil": {"type": "object", "properties": AIRFOIL_PROPS, "additionalProperties": False,
                "description": "Linear airfoil model for all sections. Default: thin airfoil "
                               "(a0 = 2*pi, no camber, no profile drag)."},
    "controls": {
        "type": "array", "minItems": 1, "maxItems": 10,
        "items": {"type": "object", "properties": CONTROL_PROPS,
                  "required": ["name"], "additionalProperties": False},
        "description": "Hinged control surfaces (flaps, ailerons, elevators, rudders), 1 to 10. "
                       "Deflections are given in degrees [deg]; span limits are fractions [-] of the "
                       "semi-span. A surface with 'mirror': true gives the controls to both copies.",
    },
    "node_displacements": {
        "type": "object", "properties": NODE_DISPLACEMENT_PROPS,
        "required": ["le_m", "te_m"], "additionalProperties": False,
        "description": "Deformed shape of the surface in metres [m]: displacements of the leading-edge "
                       "and trailing-edge lattice nodes of the defining half. The number of items must "
                       "equal the number of strip edges of the defining half for the solver and settings "
                       "of the call; use ventorum_undeformed_nodes to get them.",
    },
    "shape": {"type": "string",
              "description": "Airfoil section shape for all sections of this surface (e.g. '0012' or '2412'). "
                             "Default: NACA 0012."},
}

AIRCRAFT_PROPS: dict[str, dict[str, Any]] = {
    "name": {"type": "string", "description": "Configuration name."},
    "surfaces": {"type": "array", "minItems": 1, "maxItems": 20,
                 "items": {"type": "object", "properties": SURFACE_PROPS, "additionalProperties": False},
                 "description": "Lifting surfaces. The surface with the largest projected planform area "
                                "is the main wing (reference values)."},
    "ref_point_m": _vec3("Moment reference point [x, y, z] in metres, geometry axes. Default [0, 0, 0]."),
    "S_ref_m2": _num("Reference area [m^2]. Default: from the main wing.", minimum=MIN_AREA_M2),
    "b_ref_m": _num("Reference span [m]. Default: from the main wing.", minimum=MIN_SPAN_M),
    "c_ref_m": _num("Reference chord [m]. Default: mean aerodynamic chord of the main wing.",
                    minimum=MIN_CHORD_M),
}

CONDITION_PROPS: dict[str, dict[str, Any]] = {
    "V_inf_m_s": _num("Free-stream speed [m/s]. Default 50.", minimum=MIN_SPEED_M_S, maximum=340),
    "alpha_deg": _num("Angle of attack [deg]. Default 5.", minimum=-30, maximum=30),
    "beta_deg": _num("Sideslip [deg], positive = wind from the right. Default 0.", minimum=-30, maximum=30),
    "rho_kg_m3": _num("Air density [kg/m^3]. Default 1.225 (RHO_SL). Do not combine with altitude_m.",
                      exclusiveMinimum=0, maximum=2),
    "altitude_m": _num("Pressure altitude [m] for the ISA density (troposphere). Do not combine with "
                       "rho_kg_m3.", minimum=-500, maximum=11000),
    "h_m": _num("Height [m] of the moment reference point above flat ground. Omit for free air.",
                exclusiveMinimum=0),
}

SETTINGS_PROPS: dict[str, dict[str, Any]] = {
    "solver": {"type": "string", "enum": ["auto", "vlm", "linear", "nonlinear", "fourier"],
               "description": "Solver. 'auto' (default) = vortex-lattice method (vlm). 'linear' and "
                              "'nonlinear' are lifting-line methods; 'fourier' is the classical series "
                              "(unswept planar wing, no ground effect)."},
    "n_panels": _int("Spanwise panels per semi-span. Default 80. " + _MESH_LIMIT_TXT, minimum=4, maximum=400),
    "n_chord": _int("Chordwise panels of the vortex-lattice method. Default: automatic (4 in free air, "
                    "more near the ground; the tool reduces the automatic value if the mesh is larger "
                    "than the limit). " + _MESH_LIMIT_TXT, minimum=1, maximum=64),
    "wake_alignment": {"type": "string", "enum": ["freestream", "body"],
                       "description": "Wake direction. 'freestream' (default) or 'body' (as in AVL; not "
                                      "allowed in ground effect)."},
}

DETAIL_LEVEL = {"type": "string", "enum": ["summary", "standard", "full"],
                "description": "summary: key numbers only. standard: also geometry and tables. "
                               "full: also arrays."}

AXES_VALUES = ("body", "stability", "wind", "all")

AXES_SCHEMA = {
    "type": "string",
    "enum": ["body", "stability", "wind", "all"],
    "description": "Moment axes: 'body' (default, fixed to the aircraft), 'stability' (turned by "
                   "alpha about y) or 'wind' (turned by beta about z, x along the free stream). "
                   "'all' returns the three sets. CL, CD and CY are relative to the free stream "
                   "in every set.",
}

# Keys that are refused with a specific hint.
_HINTS = {
    "span": "use span_m (metres, tip to tip)",
    "semi_span": "use semi_span_m (metres, root to tip)",
    "chord": "use chord_m (metres)",
    "root_chord": "use root_chord_m (metres)",
    "tip_chord": "use tip_chord_m (metres)",
    "sweep_deg": "use sweep_le_deg (leading-edge sweep, degrees)",
    "v_inf": "use V_inf_m_s (m/s)",
    "rho": "use rho_kg_m3 (kg/m^3)",
    "alpha": "use alpha_deg (degrees)",
    "aoa": "use alpha_deg (degrees)",
    "alpha_rad": "use alpha_deg (degrees)",
    "beta": "use beta_deg (degrees)",
    "washout": "use tip_twist_deg; washout is a negative tip_twist_deg",
    "washout_deg": "use tip_twist_deg; washout is a negative tip_twist_deg",
    "twist": "use tip_twist_deg (positive = leading edge up)",
    "twist_deg": "use tip_twist_deg (positive = leading edge up) or sections[].twist_deg",
    "mean_chord": "use chord_m for a constant chord, or root_chord_m and tip_chord_m",
    "aspect_ratio": "aspect ratio is a result; give span_m and the chords",
    "area": "area is a result; give span_m and the chords (or set S_ref_m2 in an aircraft spec)",
    "taper_ratio": "give root_chord_m and tip_chord_m",
    "sweep": "use sweep_le_deg (degrees)",
    "dihedral": "use dihedral_deg (degrees)",
    "incidence": "use incidence_deg (degrees)",
    "is_symmetric": "use symmetric (boolean)",
    "velocity": "use V_inf_m_s (m/s)",
    "speed": "use V_inf_m_s (m/s)",
    "airspeed": "use V_inf_m_s (m/s)",
    "density": "use rho_kg_m3 (kg/m^3)",
    "altitude": "use altitude_m (ISA density) or h_m (height above the ground)",
    "h": "use h_m (height of the moment reference point above the ground, metres)",
    "height": "use h_m (height of the moment reference point above the ground, metres)",
    "panels": "use n_panels",
    "wingspan": "use span_m",
    "deflection": "use deflection_deg (degrees)",
    "hinge": "use hinge_x_c (fraction of the chord)",
    "flaps": "use controls",
    "x_cg": "use x_cg_m",
    "heights": "use heights_m",
    "chord_root": "use root_chord_m",
    "chord_tip": "use tip_chord_m",
    "n_workers": "use n_jobs",
    "x_start": "use position_m ([x, y, z] of the root leading edge, metres)",
    "y_start": "use position_m ([x, y, z] of the root leading edge, metres)",
    "z_start": "use position_m ([x, y, z] of the root leading edge, metres)",
    "position": "use position_m ([x, y, z] of the root leading edge, metres)",
    "x_le": "use position_m ([x, y, z] of the root leading edge, metres)",
    "offset": "use position_m ([x, y, z] of the root leading edge, metres)",
    "location": "use position_m ([x, y, z] of the root leading edge, metres)",
    "root_le": "use position_m",
    "solver_type": "use solver in settings",
    "method": "use solver in settings",
    "cg": "use x_cg_m (stability tool) or aircraft.ref_point_m",
    "cg_m": "use x_cg_m (stability tool) or aircraft.ref_point_m",
    "mach": "Mach number is not an input (incompressible flow); give V_inf_m_s",
    "wings": "use the 'aircraft' argument with 'surfaces'",
    "components": "use the 'aircraft' argument with 'surfaces'",
    "parts": "use the 'aircraft' argument with 'surfaces'",
    "tail": "add it as a surface in aircraft.surfaces",
    "stabilizer": "add it as a surface in aircraft.surfaces",
    "htail": "add it as a surface in aircraft.surfaces",
    "vtail": "add it as a surface in aircraft.surfaces",
    "fin": "add it as a surface in aircraft.surfaces",
    "surfaces": "use the 'aircraft' argument for several surfaces or for reference values",
    "S_ref_m2": "use the 'aircraft' argument for several surfaces or for reference values",
    "b_ref_m": "use the 'aircraft' argument for several surfaces or for reference values",
    "c_ref_m": "use the 'aircraft' argument for several surfaces or for reference values",
    "ref_point_m": "use the 'aircraft' argument for several surfaces or for reference values",
}

_ACTIVE_TOOL: contextvars.ContextVar[str | None] = contextvars.ContextVar("_ACTIVE_TOOL", default=None)


@contextlib.contextmanager
def active_tool_context(tool_name: str):
    """Set the active tool context for key placement hints.

    Parameters
    ----------
    tool_name : str
        Name of the tool being executed or validated.
    """
    token = _ACTIVE_TOOL.set(tool_name)
    try:
        yield
    finally:
        _ACTIVE_TOOL.reset(token)


def _extract_container_keys(schema: dict[str, Any], container_name: str, result: dict[str, set[str]]) -> None:
    if not isinstance(schema, dict):
        return
    props = schema.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            result.setdefault(k, set()).add(container_name)
            if isinstance(v, dict):
                if v.get("type") == "object":
                    _extract_container_keys(v, k, result)
                elif v.get("type") == "array" and isinstance(v.get("items"), dict):
                    if v["items"].get("type") == "object":
                        _extract_container_keys(v["items"], k, result)


def _get_tool_key_map(tool_name: str | None) -> dict[str, set[str]]:
    if not tool_name:
        return {}
    for tool_def in AGENT_TOOL_DEFINITIONS:
        if tool_def["name"] == tool_name:
            result: dict[str, set[str]] = {}
            params = tool_def.get("parameters", {})
            props = params.get("properties", {})
            for top_k, top_v in props.items():
                result.setdefault(top_k, set()).add("top level")
                if isinstance(top_v, dict):
                    if top_v.get("type") == "object":
                        _extract_container_keys(top_v, top_k, result)
                    elif top_v.get("type") == "array" and isinstance(top_v.get("items"), dict):
                        if top_v["items"].get("type") == "object":
                            _extract_container_keys(top_v["items"], top_k, result)
            return result
    return {}


def _container_from_where(where: str) -> str:
    if not where or where.endswith("arguments"):
        return "top level"
    cleaned = re.sub(r"\[\d+\]", "", where)
    parts = cleaned.split(".")
    last = parts[-1].strip()
    return last if last else "top level"


def _get_placement_hint(key: str, where: str, tool: str | None = None) -> str | None:
    tool_name = tool or _ACTIVE_TOOL.get()
    key_map = _get_tool_key_map(tool_name)
    if not key_map or key not in key_map:
        return None
    containers = key_map[key]
    current = _container_from_where(where)
    other = [c for c in sorted(containers) if c != current]
    if not other:
        return None

    parts: list[str] = []
    if "top level" in other:
        parts.append("at the top level")
    named = [f"'{c}'" for c in other if c != "top level"]
    if named:
        parts.append(f"in {' or '.join(named)}")
    return f"'{key}' belongs {' or '.join(parts)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Primitive checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_keys(obj: Any, allowed: dict[str, Any] | set[str] | list[str], where: str,
               *, tool: str | None = None) -> dict[str, Any]:
    """Check that *obj* is a dict with known keys only; return a copy of it.

    A key with the value None (JSON null) is refused: omit the key to get
    its default. This applies to the tool arguments (through the
    dispatcher) and to every nested object.
    """
    if not isinstance(obj, dict):
        raise InputError(f"{where} must be a JSON object, got {type(obj).__name__}.")
    allowed_set = set(allowed)
    out: dict[str, Any] = {}
    for key, val in obj.items():
        if not isinstance(key, str):
            raise InputError(f"{where}: keys must be strings, got {key!r}.")
        if key not in allowed_set:
            hints: list[str] = []
            placement_hint = _get_placement_hint(key, where, tool)
            if placement_hint:
                hints.append(placement_hint)
            unit_hint = _HINTS.get(key) or _HINTS.get(key.lower())
            if unit_hint:
                hints.append(unit_hint)

            msg = f"{where}: unknown key '{key}'."
            if hints:
                msg += f" Hint: {'; '.join(hints)}."
            msg += f" Allowed keys: {', '.join(sorted(allowed_set))}."
            raise InputError(msg)
        if val is None:
            raise InputError(f"{where}: '{key}' is null. Omit the key to use the default; "
                             "explicit null is refused.")
        out[key] = val
    return out


def number(val: Any, name: str, *, minimum: float | None = None, maximum: float | None = None,
           exclusive_min: float | None = None) -> float:
    """Return *val* as a float. Refuse strings, booleans, NaN, infinity and values out of range."""
    if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
        raise InputError(f"'{name}' must be a number (JSON number, no unit text), got {val!r}.")
    v = float(val)
    if not math.isfinite(v):
        raise InputError(f"'{name}' must be finite, got {val!r}.")
    if exclusive_min is not None and v <= exclusive_min:
        raise InputError(f"'{name}' must be > {exclusive_min}, got {v}.")
    if minimum is not None and v < minimum:
        raise InputError(f"'{name}' must be >= {minimum}, got {v}.")
    if maximum is not None and v > maximum:
        raise InputError(f"'{name}' must be <= {maximum}, got {v}.")
    return v


def integer(val: Any, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Return *val* as an int. A float is accepted only if it has no fraction (16.0)."""
    if isinstance(val, bool):
        raise InputError(f"'{name}' must be an integer, got {val!r}.")
    if isinstance(val, (float, np.floating)) and math.isfinite(float(val)) and float(val).is_integer():
        val = int(val)
    if not isinstance(val, (int, np.integer)):
        raise InputError(f"'{name}' must be an integer, got {val!r}.")
    v = int(val)
    if minimum is not None and v < minimum:
        raise InputError(f"'{name}' must be >= {minimum}, got {v}.")
    if maximum is not None and v > maximum:
        raise InputError(f"'{name}' must be <= {maximum}, got {v}.")
    return v


def boolean(val: Any, name: str) -> bool:
    """Return *val* if it is a JSON boolean. Strings such as "false" are refused."""
    if not isinstance(val, (bool, np.bool_)):
        raise InputError(f"'{name}' must be a boolean (true or false, not a string), got {val!r}.")
    return bool(val)


def string(val: Any, name: str, choices: list[str] | tuple[str, ...] | None = None) -> str:
    """Return *val* if it is a string (and one of *choices* if given)."""
    if not isinstance(val, str):
        raise InputError(f"'{name}' must be a string, got {val!r}.")
    if choices is not None and val not in choices:
        raise InputError(f"'{name}' must be one of {list(choices)}, got {val!r}.")
    return val


def vector3(val: Any, name: str) -> np.ndarray:
    """Return a 3-component vector of numbers. A 1-D numpy array is accepted."""
    if isinstance(val, np.ndarray):
        if val.ndim != 1:
            raise InputError(f"'{name}' must be a list of 3 numbers [x, y, z] in metres, got {val!r}.")
        val = val.tolist()
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        raise InputError(f"'{name}' must be a list of 3 numbers [x, y, z] in metres, got {val!r}.")
    return np.array([number(v, f"{name}[{i}]") for i, v in enumerate(val)], dtype=float)


def number_list(val: Any, name: str, *, min_len: int = 1, max_len: int = 100, **limits: Any) -> list[float]:
    """Return a list of numbers. A 1-D numpy array is accepted."""
    if isinstance(val, np.ndarray):
        if val.ndim != 1:
            raise InputError(f"'{name}' must be a list of numbers, got {val!r}.")
        val = val.tolist()
    if not isinstance(val, (list, tuple)):
        raise InputError(f"'{name}' must be a list of numbers, got {val!r}.")
    if not (min_len <= len(val) <= max_len):
        raise InputError(f"'{name}' must have {min_len} to {max_len} items, got {len(val)}.")
    return [number(v, f"{name}[{i}]", **limits) for i, v in enumerate(val)]


def _limits(prop: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "minimum" in prop:
        out["minimum"] = prop["minimum"]
    if "maximum" in prop:
        out["maximum"] = prop["maximum"]
    if "exclusiveMinimum" in prop:
        out["exclusive_min"] = prop["exclusiveMinimum"]
    return out


def _num_key(d: dict[str, Any], key: str, props: dict[str, dict[str, Any]], where: str,
             default: float | None = None) -> float | None:
    if key not in d:
        return default
    if d[key] is None:
        raise InputError(f"'{where}.{key}' is null. Omit the key to use the default; "
                         "explicit null is refused.")
    return number(d[key], f"{where}.{key}", **_limits(props[key]))


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry parsers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_airfoil(spec: Any, where: str) -> LinearAirfoil:
    """Build a :class:`LinearAirfoil` from a strict airfoil object."""
    d = check_keys(spec, AIRFOIL_PROPS, where)
    return LinearAirfoil(
        a0=_num_key(d, "a0_per_rad", AIRFOIL_PROPS, where, 2.0 * np.pi),
        alpha_L0=float(np.radians(_num_key(d, "alpha_L0_deg", AIRFOIL_PROPS, where, 0.0))),
        Cd0=_num_key(d, "cd0", AIRFOIL_PROPS, where, 0.0),
        Cm0=_num_key(d, "cm0", AIRFOIL_PROPS, where, 0.0),
    )


def _parse_controls(val: Any, where: str) -> list[ControlSurface]:
    """Build control surfaces from a strict controls list. Deflections are stored in rad."""
    if not isinstance(val, list) or not (1 <= len(val) <= 10):
        raise InputError(f"'{where}' must be a list of 1 to 10 control objects.")
    out: list[ControlSurface] = []
    seen: set[str] = set()
    for i, item in enumerate(val):
        w = f"{where}[{i}]"
        d = check_keys(item, CONTROL_PROPS, w)
        if "name" not in d:
            raise InputError(f"{w}: 'name' is required.")
        name = string(d["name"], f"{w}.name")
        if name in seen:
            raise InputError(f"{w}: duplicate control name {name!r}. 'name' must be unique on the surface.")
        seen.add(name)
        eta_start = _num_key(d, "eta_start", CONTROL_PROPS, w, 0.0)
        eta_end = _num_key(d, "eta_end", CONTROL_PROPS, w, 1.0)
        hinge = _num_key(d, "hinge_x_c", CONTROL_PROPS, w, 0.75)
        if hinge >= 1.0:
            raise InputError(f"'{w}.hinge_x_c' must be < 1, got {hinge}.")
        deflection_deg = _num_key(d, "deflection_deg", CONTROL_PROPS, w, 0.0)
        symmetric = boolean(d["symmetric"], f"{w}.symmetric") if "symmetric" in d else True
        out.append(ControlSurface(
            name=name,
            eta_start=float(eta_start),
            eta_end=float(eta_end),
            hinge_x_c=float(hinge),
            deflection=float(np.radians(deflection_deg)),
            symmetric=symmetric,
        ))
    return out


def _parse_disp_array(val: Any, where: str) -> list[list[float]]:
    """Return a displacement array as a list of [dx, dy, dz] triples in metres."""
    if isinstance(val, np.ndarray):
        if val.ndim != 2:
            raise InputError(f"'{where}' must be a list of 2 to 401 [dx, dy, dz] triples in metres, "
                             f"got an array of shape {val.shape}.")
        val = val.tolist()
    if not isinstance(val, (list, tuple)) or not (2 <= len(val) <= 401):
        n = len(val) if isinstance(val, (list, tuple)) else val
        raise InputError(f"'{where}' must be a list of 2 to 401 [dx, dy, dz] triples in metres, got {n}.")
    out: list[list[float]] = []
    for i, item in enumerate(val):
        w = f"{where}[{i}]"
        if isinstance(item, np.ndarray):
            item = item.tolist() if item.ndim == 1 else item
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            got = len(item) if isinstance(item, (list, tuple)) else repr(item)
            raise InputError(f"'{w}' must be a list of exactly 3 numbers [dx, dy, dz] in metres, "
                             f"got {got}.")
        out.append([number(v, f"{w}[{k}]") for k, v in enumerate(item)])
    return out


def _parse_node_displacements(val: Any, where: str) -> NodeDisplacements:
    """Build node displacements in metres from a strict node_displacements object."""
    d = check_keys(val, NODE_DISPLACEMENT_PROPS, where)
    for req in ("le_m", "te_m"):
        if req not in d:
            raise InputError(f"{where}: '{req}' is required.")
    le = _parse_disp_array(d["le_m"], f"{where}.le_m")
    te = _parse_disp_array(d["te_m"], f"{where}.te_m")
    if len(le) != len(te):
        raise InputError(f"{where}: 'le_m' and 'te_m' must have the same length, got {len(le)} and "
                         f"{len(te)}.")
    eta = None
    if "eta" in d:
        stations = number_list(d["eta"], f"{where}.eta", min_len=2, max_len=401)
        if len(stations) != len(le):
            raise InputError(f"{where}: 'eta' must have the same length as 'le_m' and 'te_m', got "
                             f"{len(stations)} stations for {len(le)} edges.")
        eta = np.array(stations, dtype=float)
    return NodeDisplacements(le=np.array(le, dtype=float), te=np.array(te, dtype=float), eta=eta)


def parse_surface(spec: Any, where: str = "wing", index: int = 0) -> list[LiftingSurface]:
    """Build one surface (two if ``mirror`` is true) from a strict surface object."""
    d = check_keys(spec, SURFACE_PROPS, where)
    name = string(d["name"], f"{where}.name") if "name" in d else ("Wing" if index == 0 else f"Surface_{index + 1}")
    symmetric = boolean(d["symmetric"], f"{where}.symmetric") if "symmetric" in d else True
    mirror = boolean(d["mirror"], f"{where}.mirror") if "mirror" in d else False
    if mirror and symmetric:
        raise InputError(f"{where}: 'mirror': true needs 'symmetric': false (a symmetric surface already "
                         "has both halves).")

    # Span
    if ("span_m" in d) == ("semi_span_m" in d):
        raise InputError(f"{where}: give exactly one of 'span_m' (tip to tip) or 'semi_span_m' (root to tip).")
    if "span_m" in d:
        if not symmetric:
            raise InputError(f"{where}: 'span_m' is for a symmetric surface. For 'symmetric': false give "
                             "'semi_span_m' (root-to-tip length).")
        semi_span = _num_key(d, "span_m", SURFACE_PROPS, where) / 2.0
    else:
        semi_span = _num_key(d, "semi_span_m", SURFACE_PROPS, where)

    default_af = parse_airfoil(d["airfoil"], f"{where}.airfoil") if "airfoil" in d else LinearAirfoil()
    default_shape = string(d["shape"], f"{where}.shape") if "shape" in d else None

    # Chords and twist
    has_const = "chord_m" in d
    has_rt = "root_chord_m" in d or "tip_chord_m" in d
    has_sec = "sections" in d
    if has_const + has_rt + has_sec != 1:
        raise InputError(f"{where}: give exactly one chord definition: 'chord_m' (constant chord), or "
                         "'root_chord_m' with 'tip_chord_m', or 'sections'.")
    if has_sec:
        if "tip_twist_deg" in d:
            raise InputError(f"{where}: with 'sections', give the twist in each section ('twist_deg'), "
                             "not 'tip_twist_deg'.")
        sections = _parse_sections(d["sections"], f"{where}.sections", default_af, default_shape)
    else:
        if has_const:
            c_root = c_tip = _num_key(d, "chord_m", SURFACE_PROPS, where)
        else:
            if "root_chord_m" not in d or "tip_chord_m" not in d:
                raise InputError(f"{where}: give both 'root_chord_m' and 'tip_chord_m' (or 'chord_m' for a "
                                 "constant chord).")
            c_root = _num_key(d, "root_chord_m", SURFACE_PROPS, where)
            c_tip = _num_key(d, "tip_chord_m", SURFACE_PROPS, where)
        tip_twist = float(np.radians(_num_key(d, "tip_twist_deg", SURFACE_PROPS, where, 0.0)))
        sections = [
            WingSection(y_frac=0.0, chord=c_root, twist=0.0, airfoil=default_af.clone(), shape=default_shape),
            WingSection(y_frac=1.0, chord=c_tip, twist=tip_twist, airfoil=default_af.clone(), shape=default_shape),
        ]

    position = vector3(d["position_m"], f"{where}.position_m") if "position_m" in d else np.zeros(3)
    if symmetric and abs(position[1]) > 1e-12:
        raise InputError(f"{where}: a symmetric surface must have its root on y = 0 (position_m[1] = 0). "
                         "For an off-centre surface use 'symmetric': false with 'mirror': true.")
    n_panels = (integer(d["n_panels"], f"{where}.n_panels", minimum=4, maximum=400)
                if "n_panels" in d else None)

    surf = LiftingSurface(
        name=name,
        semi_span=float(semi_span),
        sections=sections,
        is_symmetric=symmetric,
        position=position,
        incidence=float(np.radians(_num_key(d, "incidence_deg", SURFACE_PROPS, where, 0.0))),
        sweep_le=float(np.radians(_num_key(d, "sweep_le_deg", SURFACE_PROPS, where, 0.0))),
        dihedral=float(np.radians(_num_key(d, "dihedral_deg", SURFACE_PROPS, where, 0.0))),
        n_panels=n_panels,
    )
    if "controls" in d:
        surf.controls = _parse_controls(d["controls"], f"{where}.controls")
    if "node_displacements" in d:
        surf.node_displacements = _parse_node_displacements(d["node_displacements"], f"{where}.node_displacements")
    if surf.controls:
        from ventorum.geometry.controls import validate_controls

        try:
            validate_controls(surf)
        except InputError:
            raise
        except ValueError as exc:
            raise InputError(f"{where}: {exc}") from exc
    out = [surf]
    if mirror:
        out.append(surf.mirrored())
    return out


def _parse_sections(val: Any, where: str, default_af: LinearAirfoil, default_shape: Any = None) -> list[WingSection]:
    if not isinstance(val, list) or not (2 <= len(val) <= 50):
        raise InputError(f"'{where}' must be a list of 2 to 50 section objects.")
    out: list[WingSection] = []
    for i, item in enumerate(val):
        w = f"{where}[{i}]"
        d = check_keys(item, SECTION_PROPS, w)
        for req in ("y_frac", "chord_m"):
            if req not in d:
                raise InputError(f"{w}: '{req}' is required.")
        x_le = _num_key(d, "x_le_m", SECTION_PROPS, w)
        z_le = _num_key(d, "z_le_m", SECTION_PROPS, w)
        shape = string(d["shape"], f"{w}.shape") if "shape" in d else default_shape
        out.append(WingSection(
            y_frac=_num_key(d, "y_frac", SECTION_PROPS, w),
            chord=_num_key(d, "chord_m", SECTION_PROPS, w),
            twist=float(np.radians(_num_key(d, "twist_deg", SECTION_PROPS, w, 0.0))),
            x_le=x_le,
            z_le=z_le,
            airfoil=parse_airfoil(d["airfoil"], f"{w}.airfoil") if "airfoil" in d else default_af.clone(),
            shape=shape,
        ))
    fr = [s.y_frac for s in out]
    if fr[0] != 0.0 or fr[-1] != 1.0 or any(b <= a for a, b in zip(fr[:-1], fr[1:])):
        raise InputError(f"'{where}': y_frac must start at 0, end at 1 and increase strictly. Got {fr}.")
    return out


def build_aircraft_from_spec(
    wing: Any = None,
    aircraft: Any = None,
    where: str | None = None,
) -> Aircraft:
    """Build an :class:`Aircraft` from a strict wing or aircraft spec.

    Give exactly one of *wing* (one surface) or *aircraft* (several surfaces,
    with ``surfaces``). An :class:`Aircraft` or :class:`LiftingSurface` instance
    is also accepted from Python callers under either name.
    """
    if (wing is None) == (aircraft is None):
        msg = "Give exactly one of 'wing' (one surface) or 'aircraft' (several surfaces, with 'surfaces')."
        prefix = f"{where}: " if where else ""
        raise InputError(f"{prefix}{msg}")

    if aircraft is not None:
        spec = aircraft
        w_name = where or "aircraft"
        if isinstance(spec, Aircraft):
            ac = spec.clone()
        elif isinstance(spec, LiftingSurface):
            ac = Aircraft(name=spec.name, surfaces=[spec.clone()])
        elif isinstance(spec, dict):
            d = check_keys(spec, AIRCRAFT_PROPS, w_name)
            if "surfaces" not in d:
                raise InputError(f"'{w_name}': 'surfaces' is required.")
            surfs_in = d["surfaces"]
            if not isinstance(surfs_in, list) or not (1 <= len(surfs_in) <= 20):
                raise InputError(f"'{w_name}.surfaces' must be a list of 1 to 20 surface objects.")
            surfaces: list[LiftingSurface] = []
            for i, s in enumerate(surfs_in):
                surfaces.extend(parse_surface(s, f"{w_name}.surfaces[{i}]", i))
            ac = Aircraft(
                name=string(d["name"], f"{w_name}.name") if "name" in d else "Aircraft",
                surfaces=surfaces,
                S_ref=_num_key(d, "S_ref_m2", AIRCRAFT_PROPS, w_name),
                b_ref=_num_key(d, "b_ref_m", AIRCRAFT_PROPS, w_name),
                c_ref=_num_key(d, "c_ref_m", AIRCRAFT_PROPS, w_name),
                ref_point=vector3(d["ref_point_m"], f"{w_name}.ref_point_m") if "ref_point_m" in d else None,
            )
        else:
            raise InputError(f"'{w_name}' must be a JSON object, got {type(spec).__name__}.")
    else:
        spec = wing
        w_name = where or "wing"
        if isinstance(spec, Aircraft):
            ac = spec.clone()
        elif isinstance(spec, LiftingSurface):
            ac = Aircraft(name=spec.name, surfaces=[spec.clone()])
        elif isinstance(spec, dict):
            surfaces = parse_surface(spec, w_name, 0)
            ac = Aircraft(name=surfaces[0].name, surfaces=surfaces)
        else:
            raise InputError(f"'{w_name}' must be a JSON object, got {type(spec).__name__}.")
    from ventorum.utils.validation import validate_surface

    for surf in ac.surfaces:
        try:
            validate_surface(surf)
        except InputError:
            raise
        except ValueError as exc:
            raise InputError(f"{where}: {exc}") from exc
    ac.compute_reference_values()
    return ac


# ═══════════════════════════════════════════════════════════════════════════════
# Flight condition and settings
# ═══════════════════════════════════════════════════════════════════════════════

def isa_density(altitude_m: float) -> float:
    """ISA density [kg/m^3] in the troposphere (altitude below 11 000 m)."""
    return RHO_SL * (1.0 - 2.25577e-5 * altitude_m) ** 4.25588


def parse_condition(spec: Any, *, allow: tuple[str, ...] | None = None,
                    where: str = "flight_condition") -> dict[str, Any]:
    """Parse a strict flight-condition object.

    Returns a dict with ``V_inf_m_s``, ``alpha_deg``, ``beta_deg``,
    ``rho_kg_m3``, ``density_source`` and ``h_m`` (None in free air).
    *allow* limits the keys (for example a polar sweep does not take
    ``alpha_deg``).
    """
    props = CONDITION_PROPS if allow is None else {k: v for k, v in CONDITION_PROPS.items() if k in allow}
    d = check_keys({} if spec is None else spec, props, where)
    if "rho_kg_m3" in d and "altitude_m" in d:
        raise InputError(f"{where}: give 'rho_kg_m3' or 'altitude_m', not both.")
    out = {
        "V_inf_m_s": _num_key(d, "V_inf_m_s", CONDITION_PROPS, where, 50.0),
        "alpha_deg": _num_key(d, "alpha_deg", CONDITION_PROPS, where, 5.0),
        "beta_deg": _num_key(d, "beta_deg", CONDITION_PROPS, where, 0.0),
        "h_m": _num_key(d, "h_m", CONDITION_PROPS, where, None),
    }
    if "altitude_m" in d:
        alt = _num_key(d, "altitude_m", CONDITION_PROPS, where)
        out["rho_kg_m3"] = isa_density(alt)
        out["density_source"] = f"ISA at altitude {alt:g} m"
    else:
        out["rho_kg_m3"] = _num_key(d, "rho_kg_m3", CONDITION_PROPS, where, RHO_SL)
        out["density_source"] = "given" if "rho_kg_m3" in d else "default (sea level)"
    return out


def make_flight_condition(c: dict[str, Any], alpha_deg: float | None = None,
                          beta_deg: float | None = None) -> FlightCondition:
    """Build a :class:`FlightCondition` from a parsed condition dict."""
    a = c["alpha_deg"] if alpha_deg is None else alpha_deg
    b = c["beta_deg"] if beta_deg is None else beta_deg
    return FlightCondition(V_inf=c["V_inf_m_s"], alpha=float(np.radians(a)), beta=float(np.radians(b)),
                           rho=c["rho_kg_m3"], h=c["h_m"])


def build_flight_condition_from_spec(spec: Any) -> FlightCondition:
    """Build a :class:`FlightCondition` from a strict flight-condition object."""
    return make_flight_condition(parse_condition(spec))


def parse_settings(spec: Any, where: str = "settings") -> SolverSettings:
    """Parse a strict settings object into :class:`SolverSettings`."""
    d = check_keys({} if spec is None else spec, SETTINGS_PROPS, where)
    s = SolverSettings()
    if "solver" in d:
        s.solver_type = string(d["solver"], f"{where}.solver", SETTINGS_PROPS["solver"]["enum"])
    if "n_panels" in d:
        s.n_panels = integer(d["n_panels"], f"{where}.n_panels", minimum=4, maximum=400)
    if "n_chord" in d:
        if d["n_chord"] is None:
            raise InputError(f"'{where}.n_chord' is null. Omit the key for the automatic value; "
                             "explicit null is refused.")
        s.n_chord = integer(d["n_chord"], f"{where}.n_chord", minimum=1, maximum=64)
    if "wake_alignment" in d:
        s.wake_alignment = string(d["wake_alignment"], f"{where}.wake_alignment",
                                  SETTINGS_PROPS["wake_alignment"]["enum"])
    return s


def parse_detail_level(val: Any) -> str:
    """Check the detail level."""
    return string(val, "detail_level", DETAIL_LEVEL["enum"])


def parse_axes(val: Any) -> str:
    """Check the moment axes. None (omitted) gives the default ``"body"``."""
    if val is None:
        return "body"
    return string(val, "axes", list(AXES_VALUES))


# ═══════════════════════════════════════════════════════════════════════════════
# Tool definitions (JSON schema)
# ═══════════════════════════════════════════════════════════════════════════════

_SURFACE_SCHEMA = {
    "type": "object",
    "properties": SURFACE_PROPS,
    "additionalProperties": False,
    "description": "Give exactly one of 'wing' (one surface) or 'aircraft' (several surfaces, with 'surfaces'). "
                   "One lifting surface. Give span_m or semi_span_m, and chord_m, or root_chord_m with "
                   "tip_chord_m, or sections.",
}
_AIRCRAFT_SCHEMA = {
    "type": "object",
    "properties": AIRCRAFT_PROPS,
    "required": ["surfaces"],
    "additionalProperties": False,
    "description": "Give exactly one of 'wing' (one surface) or 'aircraft' (several surfaces, with 'surfaces'). "
                   "Several lifting surfaces with reference values and a moment reference point.",
}


def _obj(props: dict[str, Any], desc: str) -> dict[str, Any]:
    return {"type": "object", "properties": props, "additionalProperties": False, "description": desc}


CONDITION_SCHEMA = _obj(CONDITION_PROPS, "Flight condition. All keys are optional.")
SETTINGS_SCHEMA = _obj(SETTINGS_PROPS, "Solver settings. All keys are optional.")
_POLAR_CONDITION_SCHEMA = _obj({k: v for k, v in CONDITION_PROPS.items() if k != "alpha_deg"},
                               "Flight condition without alpha_deg (the sweep sets alpha).")

_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "wing": _SURFACE_SCHEMA,
        "aircraft": _AIRCRAFT_SCHEMA,
    },
    "additionalProperties": False,
    "description": "Give exactly one of 'wing' (one surface) or 'aircraft' (several surfaces, with 'surfaces').",
}

_AXES_NO_ALL_SCHEMA = {
    "type": "string",
    "enum": ["body", "stability", "wind"],
    "description": "Moment axes: 'body' (default, fixed to the aircraft), 'stability' (turned by "
                   "alpha about y) or 'wind' (turned by beta about z, x along the free stream). "
                   "'all' is refused. CL, CD and CY are relative to the free stream in every set.",
}

#: Section properties of the layer-2 input uncertainties (agent units).
INPUT_UNCERTAINTY_SECTION_PROPS: dict[str, dict[str, Any]] = {
    "a0_per_rad": _num("Standard uncertainty of the section lift-curve slope [1/rad].", minimum=0),
    "alpha_L0_deg": _num("Standard uncertainty of the section zero-lift angle [deg].", minimum=0),
    "Cd0": _num("Standard uncertainty of the section profile drag coefficient [-].", minimum=0),
    "Cm0": _num("Standard uncertainty of the section pitching moment coefficient [-].", minimum=0),
}

#: Flow inputs of the layer-2 input uncertainties (agent units).
INPUT_UNCERTAINTY_FLOW_PROPS: dict[str, dict[str, Any]] = {
    "alpha_deg": _num("Standard uncertainty of the angle of attack [deg].", minimum=0),
    "beta_deg": _num("Standard uncertainty of the sideslip angle [deg].", minimum=0),
    "phi_deg": _num("Standard uncertainty of the bank angle [deg].", minimum=0),
    "V_inf_m_s": _num("Standard uncertainty of the free-stream speed [m/s].", minimum=0),
    "rho_kg_m3": _num("Standard uncertainty of the air density [kg/m^3].", minimum=0),
    "h_m": _num("Standard uncertainty of the height above the ground [m]. Ground effect only.", minimum=0),
}

INPUT_UNCERTAINTIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Layer-2 standard uncertainties (one standard deviation, never negative). Flow inputs "
                   "and, in 'sections', the linear section data per surface name (fully correlated along "
                   "the span). The layer-2 bars are combined with the numerical bars by root sum of squares.",
    "properties": {
        **INPUT_UNCERTAINTY_FLOW_PROPS,
        "sections": {
            "type": "object",
            "description": "Section uncertainties keyed by surface name.",
            "additionalProperties": {
                "type": "object",
                "properties": INPUT_UNCERTAINTY_SECTION_PROPS,
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

PANEL_SETTINGS_PROPS: dict[str, dict[str, Any]] = {
    "n_chord": _int("Chordwise panels per surface [-]. Default 40.", minimum=4, maximum=200),
    "n_panels": _int("Spanwise panels per semi-span [-]. Default 80.", minimum=4, maximum=400),
    "panel_kutta": {
        "type": "string",
        "enum": ["morino", "pressure"],
        "description": "Kutta condition: 'morino' (default) or 'pressure' (not exposed yet).",
    },
    "panel_backend": {
        "type": "string",
        "enum": ["auto", "numpy", "numba", "gpu"],
        "description": "Kernel backend. Default 'auto'.",
    },
    "panel_ground_solver": {
        "type": "string",
        "enum": ["auto", "direct", "krylov"],
        "description": "Ground linear solver: 'auto' (default), 'direct' or 'krylov'.",
    },
    "panel_cache": {
        "type": "boolean",
        "description": "True (default) to cache factorized systems and meshes.",
    },
}

PANEL_SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Solver settings for the 3D panel method.",
    "properties": PANEL_SETTINGS_PROPS,
    "additionalProperties": False,
}

AGENT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "ventorum_wing_analysis",
        "description": "Aerodynamic coefficients of a wing or a set of surfaces at one flight condition "
                       "(vortex-lattice method by default). Moments are about the moment reference point: "
                       "Cl > 0 right wing down, Cm > 0 nose up, Cn > 0 nose right.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "flight_condition": CONDITION_SCHEMA,
                "settings": SETTINGS_SCHEMA,
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "axes": {**AXES_SCHEMA, "default": "body"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_polar_sweep",
        "description": "Sweep of the angle of attack: CL, CDi, CD, Cm and L/D per angle, the lift slope "
                       "from a linear fit and the maximum L/D. When every angle fails, the top-level "
                       "error copies the first row's error.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "alpha_start_deg": _num("First angle of attack [deg].", minimum=-30, maximum=30),
                "alpha_end_deg": _num("Last angle of attack [deg] (>= alpha_start_deg).", minimum=-30, maximum=30),
                "alpha_step_deg": _num(f"Step [deg], at least {MIN_ALPHA_STEP_DEG}. At most 61 angles.",
                                       minimum=MIN_ALPHA_STEP_DEG, maximum=30),
                "flight_condition": _POLAR_CONDITION_SCHEMA,
                "settings": SETTINGS_SCHEMA,
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "axes": {**AXES_SCHEMA, "default": "body"},
            },
            "required": ["alpha_start_deg", "alpha_end_deg", "alpha_step_deg"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_ground_effect",
        "description": "Ground effect at a list of heights compared with free air, with the same "
                       "chordwise mesh for all cases. Reports ground strikes per height, the bank angle "
                       "for a wingtip strike and, with 2 or more valid heights, the Irodov height-pitch "
                       "stability margin (valid when ref_point_m is the centre of gravity). Each row carries trust_score, trust_rating, warnings and "
                       "converged; the summary gives the lowest rating. Two heights closer than "
                       "1e-6 * c_ref are refused. Fourier solver and wake_alignment 'body' are not allowed.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "heights_m": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0},
                              "minItems": 1, "maxItems": 20,
                              "description": "Heights above the ground [m] of the point that height_ref "
                                             "selects. Metres, not h/b or h/c."},
                "alpha_deg": _num("Angle of attack = pitch attitude to the ground [deg]. Default 5.",
                                  minimum=-30, maximum=30),
                "phi_deg": _num("Bank angle [deg], positive = right wing down. Default 0.",
                                minimum=-60, maximum=60),
                "beta_deg": _num("Sideslip [deg]. Default 0.", minimum=-30, maximum=30),
                "height_ref": {"type": "string", "enum": ["ref", "min", "qc", "te"],
                               "description": "Point whose height is heights_m: 'ref' (default) the moment "
                                              "reference point, 'min' the lowest point of the geometry, "
                                              "'qc' / 'te' the root quarter chord / trailing edge of the "
                                              "main wing."},
                "V_inf_m_s": _num("Free-stream speed [m/s]. Default 50.", minimum=MIN_SPEED_M_S, maximum=340),
                "rho_kg_m3": _num("Air density [kg/m^3]. Default 1.225 (RHO_SL).", exclusiveMinimum=0, maximum=2),
                "ref_point_m": _vec3("Moment reference point [x, y, z] in metres. Default: the aircraft "
                                     "ref_point_m, else [0, 0, 0]."),
                "settings": SETTINGS_SCHEMA,
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "axes": {**AXES_SCHEMA, "default": "body"},
            },
            "required": ["heights_m"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_stability_derivatives",
        "description": "Static stability derivatives by central differences (alpha +/- 0.5 deg, beta "
                       "+/- 1 deg) about the centre of gravity: CL_alpha, Cm_alpha, neutral point, static "
                       "margin, Cl_beta, Cn_beta, CY_beta. With flight_condition.h_m the derivatives are "
                       "in ground effect at constant height.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "flight_condition": CONDITION_SCHEMA,
                "x_cg_m": _num("Centre of gravity x position [m], geometry axes (x aft). Default: the "
                               "aircraft ref_point_m x, else 0."),
                "settings": SETTINGS_SCHEMA,
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "axes": {**AXES_SCHEMA, "default": "body",
                         "description": "Moment axes: 'body' (default, fixed to the aircraft), 'stability' "
                                        "(turned by alpha about y) or 'wind' (turned by beta about z, x along "
                                        "the free stream). 'all' returns the three sets. The point moments "
                                        "and the moment derivatives (Cm_alpha, Cl_beta, Cn_beta) use the "
                                        "selected axes; the derivatives are taken in the axes of the "
                                        "reference condition, held fixed while alpha and beta change. CL, "
                                        "CD and CY are relative to the free stream in every set. The static "
                                        "margin and the neutral point use the body-axis Cm_alpha."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_batch_evaluate",
        "description": "Analyse several candidate geometries at one flight condition and rank them. "
                       "Candidates with invalid input are listed with their error.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "items": _CANDIDATE_SCHEMA, "minItems": 1, "maxItems": 50,
                               "description": "Candidate geometries: array of objects, each with exactly one "
                                              "of 'wing' or 'aircraft'. Use 'name' to label them."},
                "flight_condition": CONDITION_SCHEMA,
                "objective": {"type": "string", "enum": ["max_L_over_D", "min_CDi", "max_CL"],
                              "description": "Ranking objective. Default max_L_over_D (uses CD with "
                                             "profile drag when the airfoils give cd0, else CDi)."},
                "settings": SETTINGS_SCHEMA,
                "n_jobs": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 64},
                                        {"type": "string", "enum": ["auto"]}],
                              "description": "Parallel threads. Default 'auto'."},
                "axes": {**AXES_SCHEMA, "default": "body"},
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_mesh_convergence",
        "description": "Spanwise mesh convergence study against a fine reference mesh. Gives the "
                       "smallest mesh within the tolerance and a recommended mesh. When no level meets "
                       "the tolerance it reports converged false with no recommended mesh and no "
                       "scaling guideline. Each level "
                       "reports n_panels and panels_solved. alpha_tested_deg lists the angles that "
                       "were really solved: exactly the requested alpha_sweep_deg angles, plus the "
                       "condition alpha when a single requested angle differs from it.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "flight_condition": CONDITION_SCHEMA,
                "tolerance_pct": _num("Allowed error against the reference [%]. Default 0.5.",
                                      exclusiveMinimum=0, maximum=50),
                "target_metric": {"type": "string", "enum": ["both", "CL", "CDi", "circulation"],
                                  "description": "Quantity for convergence. Default 'both' (CL and CDi)."},
                "panel_counts": {"type": "array", "items": {"type": "integer", "minimum": 4, "maximum": 400},
                                 "minItems": 2, "maxItems": 12,
                                 "description": "Spanwise panels per semi-span to test. Default "
                                                "10, 15, 20, 30, 40, 60, 80."},
                "spacing_schemes": {"type": "array", "minItems": 1, "maxItems": 5,
                                    "items": {"type": "string",
                                              "enum": ["auto", "half-cosine", "cosine", "uniform", "root"]},
                                    "description": "Spanwise spacings to test. Default auto, half-cosine, "
                                                   "cosine, uniform."},
                "alpha_sweep_deg": {"type": "array", "items": {"type": "number", "minimum": -30, "maximum": 30},
                                    "minItems": 1, "maxItems": 15,
                                    "description": "Optional angles of attack [deg] at which the mesh "
                                                   "must also converge."},
                "ref_n_panels": _int("Panels per semi-span of the reference mesh. Default 160. Must be "
                                     "larger than all panel_counts.", minimum=20, maximum=400),
                "solver": SETTINGS_PROPS["solver"],
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_undeformed_nodes",
        "description": "Strip-edge stations and undeformed lattice nodes of each surface: the eta stations "
                       "[-] and the leading-edge and trailing-edge node coordinates [m] of the mesh that "
                       "the solver builds with these settings. Use them to build node_displacements. "
                       "The Fourier solver takes no node displacements and is refused with invalid_method. "
                       "No solve runs, so the work budget does not apply.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "settings": SETTINGS_SCHEMA,
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_trim",
        "description": "Trim an aircraft with Newton's method (Etkin and Reid; Dennis and Schnabel): the "
                       "angle of attack [deg] and the control deflections [deg] for a target lift "
                       "coefficient [-] with zero aerodynamic moments. A target that is not reached is "
                       "not an error: the payload gives trim_status and trimmed false. The solves run on "
                       "the selected device; the payload states the device and the precision.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "CL_target": _num("Target lift coefficient [-]. Range -3 to 5.", minimum=-3, maximum=5),
                "pitch_control": {"type": "string",
                                  "description": "Name of the pitch control (string, must match a name of "
                                                 "the controls of the wing)."},
                "roll_control": {"type": "string",
                                 "description": "Name of the roll control (string, must match a name of "
                                                "the controls). Optional. Give both roll_control and "
                                                "yaw_control, or neither (lateral trim)."},
                "yaw_control": {"type": "string",
                                "description": "Name of the yaw control (string, must match a name of "
                                               "the controls). Optional. Give both roll_control and "
                                               "yaw_control, or neither (lateral trim)."},
                "flight_condition": CONDITION_SCHEMA,
                "alpha_bounds_deg": {"type": "array",
                                     "items": {"type": "number", "minimum": -30, "maximum": 30},
                                     "minItems": 2, "maxItems": 2,
                                     "description": "Lower and upper bounds of the angle of attack [deg]. "
                                                    "Must increase. Default [-10, 20]."},
                "max_iterations": _int("Maximum Newton iterations [-] (integer). Range 1 to 50. "
                                       "Default 20.", minimum=1, maximum=50),
                "settings": SETTINGS_SCHEMA,
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "axes": {**AXES_SCHEMA, "default": "body"},
            },
            "required": ["CL_target", "pitch_control"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_error_bars",
        "description": "Layer-1 numerical error bars of the six force and moment coefficients from three "
                       "spanwise mesh levels (Grid Convergence Index, Roache; observed order, Celik et al.). "
                       "The user mesh is the fine level. With input_uncertainties, layer-2 input uncertainty "
                       "is combined by root sum of squares. n_panels below 16 is refused.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "flight_condition": CONDITION_SCHEMA,
                "settings": SETTINGS_SCHEMA,
                "axes": {**_AXES_NO_ALL_SCHEMA, "default": "body"},
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "input_uncertainties": INPUT_UNCERTAINTIES_SCHEMA,
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_derivative_matrix",
        "description": "Full aerodynamic derivative matrix of one flight state: sensitivity of the six "
                       "force and moment coefficients (CL, CD, CY, Cl, Cm, Cn) to static variables (alpha, "
                       "beta, and in ground effect phi and h/c), control deflections, and body rates "
                       "(p_hat, q_hat, r_hat).",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "flight_condition": CONDITION_SCHEMA,
                "settings": SETTINGS_SCHEMA,
                "groups": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["static", "control", "rate", "height"]},
                    "description": "Derivative groups to include. Default ['static', 'control', 'rate']. "
                                   "In ground effect, 'height' is added by default.",
                },
                "controls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of control names to differentiate. Default: all controls.",
                },
                "axes": {**AXES_SCHEMA, "default": "body"},
                "x_cg_m": _num("Centre of gravity x position [m], geometry axes (x aft). Default: the "
                               "aircraft ref_point_m x, else 0."),
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_machine_capabilities",
        "description": "Hardware capabilities of this machine: public hardware data (never a machine "
                       "identifier or a fingerprint), available kernel backends, Cython thread mode, "
                       "torch device, GPU data, GPU pipeline device and precision, and tuning profile "
                       "status, with one sentence of advice. Takes no input.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_tune_machine",
        "description": "Measure this machine and write its tuning profile (same result as ventorum-tune). "
                       "Takes about half a minute with quick true and up to two minutes with quick false. "
                       "Tuning changes the speed; the device choice can change results at the float32 "
                       "round-off level (about 1e-7 to 1e-6 relative). This tool is outside the work "
                       "budget: its duration is stated here instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "quick": {"type": "boolean", "default": True,
                          "description": "Skip the large cases (about 3 times faster). Default true."},
                "save": {"type": "boolean", "default": True,
                         "description": "Write the profile to the user configuration folder. "
                                        "Default true."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_airfoil_geometry",
        "description": "Normalized geometry, thickness, camber, trailing-edge gap and defects of an "
                       "airfoil section (NACA designation or coordinate text in Selig or Lednicer format).",
        "parameters": {
            "type": "object",
            "properties": AIRFOIL_GEOMETRY_PROPS,
            "required": ["section"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_airfoil_polar",
        "description": "Lift, drag and pitching moment polar of an airfoil section (ventorum2d viscous "
                       "panel solver, also in ground effect, or the XFOIL of the server), with transition "
                       f"positions and convergence status. Budget: at most {MAX_AIRFOIL_SOLVES} angles, and "
                       "angles x max_iter x n_panels^2 at most "
                       f"{int(MAX_CALL_WORK)} work units per call.",
        "parameters": {
            "type": "object",
            "properties": AIRFOIL_POLAR_PROPS,
            "required": ["section", "Re"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ventorum_panel_analysis",
        "description": "Aerodynamic analysis with the 3D source-doublet potential-flow panel method "
                       "(Ventorum-3DP, Morino Kutta condition). Computes CL, CD (pressure), CDi (Trefftz), "
                       "CY, Cl, Cm and Cn, the trust score, and the execution dispatch record. Ground effect "
                       "is supported through flight_condition.h_m. Factorization work is N^3 / 3 operations "
                       f"for N solved panels (the full mesh with sideslip or an asymmetric surface). Refuses "
                       f"meshes above {MAX_PANEL_PANELS} half-mesh panels "
                       f"and calls above {int(MAX_PANEL_CALL_WORK):.0e} operations. Not exposed yet: "
                       "the pressure Kutta condition and control surfaces are not supported and are refused.",
        "parameters": {
            "type": "object",
            "properties": {
                "wing": _SURFACE_SCHEMA,
                "aircraft": _AIRCRAFT_SCHEMA,
                "flight_condition": CONDITION_SCHEMA,
                "n_chord": _int("Chordwise panel count per surface [-]. Default 40.", minimum=4, maximum=200),
                "n_span": _int("Spanwise panels per semi-span [-]. Default 80.", minimum=4, maximum=400),
                "cp": {
                    "type": "boolean",
                    "description": "True to include the panel pressure coefficients Cp and centroids in the result. Default false.",
                },
                "settings": PANEL_SETTINGS_SCHEMA,
                "detail_level": {**DETAIL_LEVEL, "default": "standard"},
                "axes": {**AXES_SCHEMA, "default": "body"},
            },
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = [t["name"] for t in AGENT_TOOL_DEFINITIONS]


# Schema keywords of the OpenAPI 3.0 subset that Gemini function declarations accept.
GEMINI_SCHEMA_KEYS = ("type", "format", "description", "nullable", "enum", "properties", "required",
                      "items", "minimum", "maximum", "minItems", "maxItems")


def _merge_any_of(options: list[dict[str, Any]], desc: str | None) -> dict[str, Any]:
    """Make one schema from the options of an ``anyOf`` (Gemini does not accept ``anyOf``)."""
    if all(o.get("type") == "object" for o in options):
        props: dict[str, Any] = {}
        for o in options:
            props.update(o.get("properties", {}))
        alt = " Alternatives: " + " OR ".join(o.get("description", "") for o in options)
        return {"type": "object", "properties": props, "description": (desc or "") + alt}
    # Different types: keep the first option and name the others in the description.
    first = dict(options[0])
    others = [", ".join(repr(v) for v in o["enum"]) if "enum" in o else f"a {o.get('type')}" for o in options[1:]]
    first["description"] = f"{desc or ''} Also accepted: {'; '.join(others)}.".strip()
    return first


def _to_gemini(schema: Any) -> Any:
    """Translate a JSON schema to the OpenAPI 3.0 subset of Gemini (:data:`GEMINI_SCHEMA_KEYS`).

    ``anyOf`` is merged into one schema, ``exclusiveMinimum`` becomes
    ``minimum`` plus a sentence in the description, and other keywords
    (``additionalProperties``, ``default``) are removed. The tools still
    check the full rules.
    """
    if isinstance(schema, list):
        return [_to_gemini(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    if "anyOf" in schema:
        schema = _merge_any_of(schema["anyOf"], schema.get("description"))
    out: dict[str, Any] = {}
    for key, val in schema.items():
        if key == "properties":
            out[key] = {name: _to_gemini(sub) for name, sub in val.items()}
        elif key == "items":
            out[key] = _to_gemini(val)
        elif key in GEMINI_SCHEMA_KEYS:
            out[key] = copy.deepcopy(val)
    if "exclusiveMinimum" in schema:
        lim = schema["exclusiveMinimum"]
        out.setdefault("minimum", lim)
        out["description"] = f"{out.get('description', '')} Must be larger than {lim}.".strip()
    return out


def get_tool_schemas(format: Literal["openai", "anthropic", "gemini", "mcp"] = "openai") -> list[dict[str, Any]]:
    """Tool schemas for an LLM framework.

    ``openai``: function-calling tools. ``anthropic``: tools with
    ``input_schema``. ``mcp``: tools with ``inputSchema``. ``gemini``:
    function declarations in the OpenAPI 3.0 subset of Gemini
    (:data:`GEMINI_SCHEMA_KEYS`); see :func:`_to_gemini`. The tools still
    refuse unknown keys and values out of range.
    """
    out: list[dict[str, Any]] = []
    for tool in AGENT_TOOL_DEFINITIONS:
        params = copy.deepcopy(tool["parameters"])
        if format == "openai":
            out.append({"type": "function", "function": {
                "name": tool["name"], "description": tool["description"], "parameters": params}})
        elif format == "anthropic":
            out.append({"name": tool["name"], "description": tool["description"], "input_schema": params})
        elif format == "gemini":
            out.append({"name": tool["name"], "description": tool["description"],
                        "parameters": _to_gemini(params)})
        elif format == "mcp":
            out.append({"name": tool["name"], "description": tool["description"], "inputSchema": params})
        else:
            raise InputError(f"format must be 'openai', 'anthropic', 'gemini' or 'mcp', got {format!r}.")
    return out
