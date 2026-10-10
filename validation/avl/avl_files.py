# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Write AVL input files from a Ventorum aircraft.

The writer maps a :class:`ventorum.core.datatypes.Aircraft` to the
AVL geometry file format described in the AVL user documentation
(``avl_doc.txt``). It also builds the AVL command stream for one run
case. It never reads AVL source code; it only writes text files that
``avl.exe`` reads as a black box.

Mapping rules (task T-0055, fixed decisions):

* Units are metres. ``Sref``, ``Cref`` and ``Bref`` come from the
  Ventorum aircraft after ``compute_reference_values()``.
  ``Xref``, ``Yref`` and ``Zref`` come from ``moment_reference()``.
  Mach is 0 and profile drag is 0 (induced drag only).
* One AVL SURFACE per Ventorum surface. A symmetric surface is written
  as its right half with the ``YDUPLICATE`` keyword about y = 0.
  A mirror copy (``mirror_y=True``) is written as its own surface with
  mirrored coordinates, ordered from tip to root so that the sections
  still run from left to right (increasing y). The spanwise spacing
  parameter changes sign for that order, because AVL applies it from
  the first section to the last section. A mirror copy and its partner
  surface get the same ``COMPONENT`` index, as ``YDUPLICATE`` gives to
  the two halves of a duplicated surface.
* One AVL SECTION per Ventorum ``WingSection``. The leading-edge point
  comes from ``surface_edge_geometry``. The incidence angle [deg] is
  the section twist plus the surface incidence minus the zero-lift
  angle of the section airfoil. In ground effect the run angle of
  attack is added as well (see :func:`build_case`).
* The section lift slope factor (AVL ``CLAF`` keyword) is
  ``a0 / (2 pi)``.
* The AVL spanwise vortex count per surface (per half for a duplicated
  surface) is the Ventorum ``n_panels``. The chordwise count is the
  Ventorum ``n_chord`` (4 when automatic, out of ground effect).

The writer refuses anything it cannot map exactly with a ``ValueError``
that names the surface and the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ventorum.core.datatypes import (
    Aircraft,
    LinearAirfoil,
    SolverSettings,
    TabulatedAirfoil,
)


#: AVL spanwise spacing parameter for each resolved Ventorum spacing.
#: From the spacing table of ``avl_doc.txt``: 1.0 is cosine (clustered
#: at root and tip), -2.0 is the mirrored sine (clustered at the tip
#: when the surface runs from root to tip), 0.0 is uniform and 2.0 is
#: sine (clustered at the root). Ventorum ``"power"`` (gentle tip
#: clustering) has no AVL equivalent; it maps to -2.0 with a note.
SPACING_MAP = {
    "half-cosine": -2.0,
    "cosine": 1.0,
    "uniform": 0.0,
    "root-cosine": 2.0,
    "power": -2.0,
}

#: AVL chordwise spacing parameter for each Ventorum chord spacing.
CHORD_SPACING_MAP = {"uniform": 0.0, "cosine": 1.0}

#: Chordwise panel count when the Ventorum settings leave it automatic.
#: This is the out-of-ground-effect default of the Ventorum solver.
DEFAULT_N_CHORD = 4


@dataclass(frozen=True)
class AvlCase:
    """One AVL run case: geometry text and run conditions.

    Attributes
    ----------
    avl_text : str
        Content of the ``.avl`` geometry file.
    run_alpha_deg : float
        Angle of attack [deg] to set in AVL before the solve.
    run_beta_deg : float
        Sideslip angle [deg] to set in AVL before the solve.
    zsym : float or None
        Value of ``Zsym`` [m] written in the header (None in free air).
    ground : bool
        True when the case uses the ground-effect rotation method.
    notes : tuple of str
        Warnings about approximations (for example ``allow_cm0``).
    n_surfaces : int or None
        Expected AVL surface count (a duplicated surface counts two).
        None skips the check of the ``FT`` header.
    n_strips : int or None
        Expected AVL strip count (spanwise vortex count of all
        surfaces). None skips the check.
    n_vortices : int or None
        Expected AVL horseshoe-vortex count. None skips the check.
    """

    avl_text: str
    run_alpha_deg: float = 0.0
    run_beta_deg: float = 0.0
    zsym: float | None = None
    ground: bool = False
    notes: tuple[str, ...] = ()
    n_surfaces: int | None = None
    n_strips: int | None = None
    n_vortices: int | None = None


def _num(value: float) -> str:
    """Format a float with full round-trip precision for the AVL file."""
    return repr(float(value))


def _rotate_about_y(points: np.ndarray, ref: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate points nose up by *angle_rad* [rad] about *ref* [m].

    The rotation axis is the y axis through the reference point. A point
    aft of the reference (x > ref x) moves down, so the nose goes up.
    """
    points = np.asarray(points, dtype=float)
    ref = np.asarray(ref, dtype=float).reshape(3)
    ca, sa = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    d = points - ref
    out = points.copy()
    out[:, 0] = ref[0] + d[:, 0] * ca + d[:, 2] * sa
    out[:, 1] = points[:, 1]
    out[:, 2] = ref[2] - d[:, 0] * sa + d[:, 2] * ca
    return out


def _surface_name(surf: Any, index: int) -> str:
    """Return a one-line AVL surface name for *surf*."""
    name = str(getattr(surf, "name", "") or "").strip().replace("\n", " ")
    return name if name else f"Surface{index + 1}"


def _component_indices(surfaces: list[Any]) -> list[int]:
    """Return the AVL ``COMPONENT`` index [-] of each surface.

    Each surface gets its own index (1, 2, ...), except a mirror copy
    (``mirror_y=True``), which gets the index of its partner surface
    (the same geometry with the opposite mirror flag). AVL does not use
    the finite-core model between surfaces of one component; the two
    halves of a ``YDUPLICATE`` surface also share one index.
    """
    from ventorum.core.datatypes import _is_mirror_pair

    index = [k + 1 for k in range(len(surfaces))]
    for k, surf in enumerate(surfaces):
        if not bool(getattr(surf, "mirror_y", False)):
            continue
        for j, other in enumerate(surfaces):
            partner = not bool(getattr(other, "mirror_y", False))
            if j != k and partner and _is_mirror_pair(other, surf):
                index[k] = index[j]
                break
    return index


def _section_airfoil_props(surf: Any) -> None:
    """Refuse airfoils that AVL flat sections cannot map exactly."""
    for sec in surf.sections:
        af = sec.airfoil
        if isinstance(af, TabulatedAirfoil):
            raise ValueError(
                f"[{surf.name}] section at y_frac={sec.y_frac}: TabulatedAirfoil "
                f"{getattr(af, 'name', '')!r} cannot be mapped to an AVL flat section."
            )


def build_case(
    aircraft: Aircraft,
    settings: SolverSettings | None = None,
    *,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    ground_h: float | None = None,
    allow_cm0: bool = False,
) -> AvlCase:
    """Build the AVL geometry text and run conditions for one case.

    Parameters
    ----------
    aircraft : Aircraft
        Ventorum aircraft (synthetic geometry only).
    settings : SolverSettings or None
        Mesh settings. ``None`` selects the defaults (80 spanwise
        panels, automatic chordwise count and spacing).
    alpha_deg : float
        Angle of attack [deg].
    beta_deg : float
        Sideslip angle [deg].
    ground_h : float or None
        Height [m] of the moment reference point above the ground.
        ``None`` means free air. When given, the geometry-rotation
        method is used: the whole geometry turns nose up by alpha
        about the reference point, the AVL z-image plane is a solid
        wall at ``Zsym = Zref - h``, and AVL runs at alpha = 0.
    allow_cm0 : bool
        When True, a nonzero section ``Cm0`` is kept with a warning
        in the case notes. When False (default) it is refused, because
        AVL flat sections cannot carry it.

    Returns
    -------
    AvlCase
        Geometry text, run angles and notes.

    Raises
    ------
    ValueError
        If a surface cannot be mapped exactly (names the surface and
        the field).
    """
    from ventorum.geometry.lattice import resolve_spacing, surface_edge_geometry

    settings = settings or SolverSettings()
    ac = aircraft.clone()
    ac.compute_reference_values()
    ref = ac.moment_reference()
    notes: list[str] = []

    ground = ground_h is not None
    if ground and float(ground_h) <= 0.0:
        raise ValueError(f"Ground height h={ground_h} must be strictly positive.")
    alpha_rad = float(np.radians(alpha_deg))
    run_alpha = 0.0 if ground else float(alpha_deg)

    if ground:
        header_sym = f"0 1 {_num(ref[2] - float(ground_h))}"
        zsym: float | None = float(ref[2] - float(ground_h))
    else:
        header_sym = "0 0 0.0"
        zsym = None

    lines = [
        f"Ventorum case alpha={alpha_deg:g} beta={beta_deg:g}",
        "0.0",
        header_sym,
        f"{_num(ac.S_ref)} {_num(ac.c_ref)} {_num(ac.b_ref)}",
        f"{_num(ref[0])} {_num(ref[1])} {_num(ref[2])}",
    ]

    components = _component_indices(list(ac.surfaces))
    n_surfaces = 0
    n_strips = 0
    n_vortices = 0
    for s_idx, surf in enumerate(ac.surfaces):
        name = _surface_name(surf, s_idx)
        _section_airfoil_props(surf)
        if bool(getattr(surf, "is_symmetric", True)) and bool(getattr(surf, "mirror_y", False)):
            raise ValueError(f"[{name}] mirror_y=True needs is_symmetric=False.")

        secs = sorted(surf.sections, key=lambda s: s.y_frac)
        if len(secs) < 2:
            raise ValueError(f"[{name}] needs at least two sections.")
        y_frac = np.array([s.y_frac for s in secs], dtype=float)
        geo = surface_edge_geometry(surf, y_frac)
        le = np.asarray(geo["le"], dtype=float)
        chords = np.asarray(geo["chord"], dtype=float)

        if ground:
            le = _rotate_about_y(le, ref, alpha_rad)

        mirror = bool(getattr(surf, "mirror_y", False))
        symmetric = bool(getattr(surf, "is_symmetric", True))
        if mirror:
            le = le * np.array([1.0, -1.0, 1.0])
            order = list(range(len(secs) - 1, -1, -1))
            notes.append(
                f"[{name}] mirror copy written tip to root (left-to-right order)."
            )
        else:
            order = list(range(len(secs)))
        if symmetric:
            root_y = float(le[0][1]) if not mirror else float(le[-1][1])
            if abs(root_y) > 1e-9 * max(1.0, float(surf.semi_span)):
                raise ValueError(
                    f"[{name}] is symmetric but its root is at y={root_y:.6g} m, "
                    "not on the plane y=0."
                )

        n_span = surf.n_panels if surf.n_panels is not None else settings.n_panels
        spacing_key = resolve_spacing(
            surf.spacing if surf.spacing is not None else settings.spacing, surf, "vlm"
        )
        if spacing_key not in SPACING_MAP:
            raise ValueError(f"[{name}] spacing={spacing_key!r} has no AVL mapping.")
        if spacing_key == "power":
            notes.append(f"[{name}] Ventorum 'power' spacing mapped to AVL -2.0.")
        s_space = SPACING_MAP[spacing_key]
        if mirror and s_space != 0.0:
            # AVL applies the spacing from the first section to the last.
            # The mirror copy is written tip to root, so the sign changes.
            s_space = -s_space
        chord_spacing = str(getattr(settings, "chord_spacing", "uniform")).lower()
        if chord_spacing not in CHORD_SPACING_MAP:
            raise ValueError(f"[{name}] chord_spacing={chord_spacing!r} has no AVL mapping.")
        n_chord = settings.n_chord if settings.n_chord is not None else DEFAULT_N_CHORD

        lines += ["SURFACE", name,
                  f"{int(n_chord)} {CHORD_SPACING_MAP[chord_spacing]:.1f} "
                  f"{int(n_span)} {s_space:.1f}",
                  "COMPONENT", str(components[s_idx])]
        if symmetric:
            lines += ["YDUPLICATE", "0.0"]
        halves = 2 if symmetric else 1
        n_surfaces += halves
        n_strips += halves * int(n_span)
        n_vortices += halves * int(n_span) * int(n_chord)
        for k in order:
            sec = secs[k]
            af = sec.airfoil
            if not isinstance(af, LinearAirfoil):  # checked above; kept for clarity
                raise ValueError(f"[{name}] section: TabulatedAirfoil cannot be mapped.")
            a0 = float(af.a0)
            alpha_l0 = float(af.alpha_L0)
            cm0 = float(af.Cm0)
            cd0 = float(af.Cd0)
            if abs(cm0) > 0.0:
                if not allow_cm0:
                    raise ValueError(
                        f"[{name}] section at y_frac={sec.y_frac}: Cm0={cm0:g} "
                        "cannot be mapped to an AVL flat section "
                        "(use allow_cm0=True to keep it with a warning)."
                    )
                notes.append(f"[{name}] section at y_frac={sec.y_frac}: Cm0={cm0:g} ignored.")
            if abs(cd0) > 0.0:
                notes.append(
                    f"[{name}] section at y_frac={sec.y_frac}: Cd0={cd0:g} ignored "
                    "(induced drag only)."
                )
            claf = a0 / (2.0 * float(np.pi))
            ainc = float(np.degrees(sec.twist + surf.incidence - alpha_l0))
            if ground:
                ainc += float(alpha_deg)
            x, y, z = (float(v) for v in le[k])
            lines += [
                "SECTION",
                f"{_num(x)} {_num(y)} {_num(z)} {_num(float(chords[k]))} {_num(ainc)}",
                "CLAF",
                _num(claf),
            ]

    return AvlCase(
        avl_text="\n".join(lines) + "\n",
        run_alpha_deg=run_alpha,
        run_beta_deg=float(beta_deg),
        zsym=zsym,
        ground=ground,
        notes=tuple(notes),
        n_surfaces=n_surfaces,
        n_strips=n_strips,
        n_vortices=n_vortices,
    )


#: Blank lines before ``QUIT``. Each blank line leaves one menu level,
#: so the program gets back to the top level even when a prompt took a
#: line that the stream did not plan for.
QUIT_PADDING = 4


def _oper_block(run_alpha_deg: float, run_beta_deg: float, ft_name: str,
                st_name: str) -> list[str]:
    """Return the OPER commands of one run case (enter, solve, write, leave)."""
    return [
        "OPER",
        f"A A {float(run_alpha_deg):.6f}",
        f"B B {float(run_beta_deg):.6f}",
        "X",
        "FT",
        ft_name,
        "ST",
        st_name,
        "",
    ]


def command_stream(
    run_alpha_deg: float,
    run_beta_deg: float = 0.0,
    ft_name: str = "ft.txt",
    st_name: str = "st.txt",
) -> str:
    """Build the AVL command stream for one run case.

    The stream enters the OPER menu, sets alpha and beta with direct
    constraints, executes the case, writes the total forces (``FT``)
    and the stability derivatives (``ST``) to files, leaves the menus
    and quits. Each command that prompts for more input (``FT``,
    ``ST``) is followed by its file name on the next line. A blank
    line leaves the OPER menu, as the AVL documentation requires.
    :data:`QUIT_PADDING` more blank lines come before ``QUIT``, so the
    stream gets back to the top level even if a prompt took one line
    more than planned.

    Parameters
    ----------
    run_alpha_deg : float
        Angle of attack [deg].
    run_beta_deg : float
        Sideslip angle [deg].
    ft_name : str
        File name for the total forces (relative to the run folder).
    st_name : str
        File name for the stability derivatives.

    Returns
    -------
    str
        Command stream text for the standard input of ``avl.exe``.
    """
    lines = _oper_block(run_alpha_deg, run_beta_deg, ft_name, st_name)
    return "\n".join(lines + [""] * QUIT_PADDING + ["QUIT", ""])


def session_command_stream(
    entries: list[tuple[str | None, float, float, str, str]],
) -> str:
    """Build one AVL command stream that runs several cases in one session.

    Parameters
    ----------
    entries : list of tuple
        One ``(avl_name, alpha_deg, beta_deg, ft_name, st_name)`` per
        run case, in run order. ``avl_name`` is the geometry file to
        read with the ``LOAD`` command before the case (a ground-effect
        sweep has one rotated geometry per angle), or None to keep the
        geometry that is already loaded. Angles in [deg].

    Returns
    -------
    str
        Command stream text for the standard input of ``avl.exe``.
    """
    lines: list[str] = []
    for avl_name, alpha, beta, ft_name, st_name in entries:
        if avl_name is not None:
            lines += ["LOAD", avl_name]
        lines += _oper_block(alpha, beta, ft_name, st_name)
    return "\n".join(lines + [""] * QUIT_PADDING + ["QUIT", ""])
