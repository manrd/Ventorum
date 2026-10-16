# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Plot airfoil shapes and 2D panel-method pressure distributions.

This module provides the standard pictures of Ventorum-2D results: the
airfoil contour with its panel nodes, the surface pressure distribution Cp
in free air and in ground effect, and the comparison of one Cp curve with a
second result or with reference data. Each function draws on given axes (or
new axes) and returns the axes. No function calls ``plt.show()``.

Coordinate convention: point arrays have shape (N, 2), float64, in Selig
order (from the trailing edge along the upper surface to the leading edge,
then along the lower surface back to the trailing edge). Unit-chord
sections place the leading edge at (0, 0) and the trailing-edge midpoint
at (1, 0) chord units.

Frames: in free air the 2D solver keeps the input nodes and turns the
free stream, so the node x of a result is the body-frame x. In ground
effect the solver turns the section by the angle of attack (clockwise,
about a pivot) and moves it to the height, so the stored nodes are in
the solved frame. :func:`plot_ground_effect_cp` turns the nodes back by
the angle of attack and measures x from the leading edge (the node
farthest from the trailing-edge midpoint) in the body frame.
:func:`plot_cp` plots the node x of the result as stored and labels the
axis with that frame.

References
----------
- J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
  University Press, 2001, chapter 11 (panel-method surface velocity and
  the pressure coefficient Cp = 1 - (V/V_inf)^2).
- I. H. Abbott and A. E. von Doenhoff, Theory of Wing Sections, Dover,
  1959, chapter 6 (NACA section geometry and pressure plots).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ventorum.airfoil.geometry import AirfoilShape, leading_edge_index

if TYPE_CHECKING:
    from ventorum.panel2d.solver import Panel2DResult

__all__ = ["plot_airfoil", "plot_cp", "plot_ground_effect_cp"]


def _to_numpy(values: Any) -> np.ndarray:
    """Return the values as a float64 numpy array.

    Torch tensors use detach, move to CPU and convert. Numpy arrays,
    lists and scalars use a plain conversion.
    """
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=float)
    detach = getattr(values, "detach", None)
    if callable(detach):
        return np.asarray(values.detach().cpu().numpy(), dtype=float)
    return np.asarray(values, dtype=float)


def _check_index(index: int) -> int:
    """Return the angle index after type and range checks."""
    try:
        idx = operator.index(index)
    except TypeError:
        raise TypeError(
            f"index must be an integer, got {index!r}."
        ) from None
    if idx < 0:
        raise IndexError(f"index must be zero or positive, got {idx}.")
    return idx


def _scalar_at(values: Any, index: int) -> float:
    """Return one scalar of a scalar or sweep quantity."""
    arr = _to_numpy(values)
    if arr.ndim == 0:
        if index != 0:
            raise IndexError(
                f"index {index} is out of range for a single-angle result."
            )
        return float(arr)
    if arr.ndim == 1 and arr.size == 1:
        if index != 0:
            raise IndexError(
                f"index {index} is out of range for a single-angle result."
            )
        return float(arr[0])
    if index >= arr.shape[0]:
        raise IndexError(
            f"index {index} is out of range for {arr.shape[0]} angles."
        )
    return float(arr[index])


def _body_cp_list(result: Panel2DResult, index: int) -> list[np.ndarray]:
    """Return the Cp row of each body at the angle index."""
    cp = result.Cp
    if isinstance(cp, list):
        out: list[np.ndarray] = []
        for body in cp:
            arr = _to_numpy(body)
            if arr.ndim == 2:
                if index >= arr.shape[0]:
                    raise IndexError(
                        f"index {index} is out of range "
                        f"for {arr.shape[0]} angles."
                    )
                out.append(np.asarray(arr[index], dtype=float))
            elif arr.ndim == 1:
                if index != 0:
                    raise IndexError(
                        f"index {index} is out of range "
                        "for a single-angle result."
                    )
                out.append(arr)
            else:
                raise ValueError(
                    "Cp of each body must have shape (N,) or (n_angles, N)."
                )
        return out
    arr = _to_numpy(cp)
    if arr.ndim == 2:
        if index >= arr.shape[0]:
            raise IndexError(
                f"index {index} is out of range for {arr.shape[0]} angles."
            )
        return [np.asarray(arr[index], dtype=float)]
    if arr.ndim == 1:
        if index != 0:
            raise IndexError(
                f"index {index} is out of range for a single-angle result."
            )
        return [arr]
    raise ValueError("Cp must have shape (N,) or (n_angles, N).")


def _n_angles(result: Panel2DResult) -> int:
    """Return the angle count of a result from its Cp shape."""
    cp = result.Cp
    if isinstance(cp, list):
        n = 1
        for body in cp:
            arr = _to_numpy(body)
            if arr.ndim == 2:
                n = max(n, arr.shape[0])
        return n
    arr = _to_numpy(cp)
    return arr.shape[0] if arr.ndim == 2 else 1


def _body_coords_list(
    result: Panel2DResult, index: int, n_bodies: int
) -> list[np.ndarray]:
    """Return the node coordinates of each body at the angle index.

    The solver stores one coordinate set per angle in ground effect and a
    single set in free air. Multi-body results store one entry per body.
    """
    coords = result.coordinates
    if isinstance(coords, (list, tuple)) and coords:
        first = coords[0]
        if isinstance(first, (list, tuple)):
            picked = coords[index]
            return [_to_numpy(picked[b]) for b in range(n_bodies)]
        if n_bodies == 1:
            if index >= len(coords):
                raise IndexError(
                    f"index {index} is out of range "
                    f"for {len(coords)} coordinate sets."
                )
            return [_to_numpy(coords[index])]
        out: list[np.ndarray] = []
        for body in coords:
            arr = _to_numpy(body)
            if arr.ndim == 3:
                if index >= arr.shape[0]:
                    raise IndexError(
                        f"index {index} is out of range "
                        f"for {arr.shape[0]} angles."
                    )
                out.append(np.asarray(arr[index], dtype=float))
            elif arr.ndim == 2:
                if index >= _n_angles(result):
                    raise IndexError(
                        f"index {index} is out of range "
                        f"for {_n_angles(result)} angles."
                    )
                out.append(arr)
            else:
                raise ValueError(
                    "Coordinates of each body must have shape (N, 2)."
                )
        return out
    arr = _to_numpy(coords)
    if arr.ndim == 3:
        if index >= arr.shape[0]:
            raise IndexError(
                f"index {index} is out of range for {arr.shape[0]} angles."
            )
        return [np.asarray(arr[index], dtype=float)]
    if arr.ndim == 2:
        if index >= _n_angles(result):
            raise IndexError(
                f"index {index} is out of range "
                f"for {_n_angles(result)} angles."
            )
        return [arr]
    raise ValueError("Coordinates must have shape (N, 2).")


def _body_frame_x(xy_list: list[np.ndarray], alpha_deg: float) -> list[np.ndarray]:
    """Return the body-frame x [chord units] of solved ground-effect nodes.

    The solver turns the bodies clockwise by the angle of attack. The
    function turns the nodes back (counterclockwise by ``alpha_deg``
    [deg]) and moves x so that the leading-edge node of the first body
    is at x = 0. A turn and a shift keep the shape, so the result is the
    input x up to that shift.
    """
    a = np.radians(float(alpha_deg))
    cos_a = float(np.cos(a))
    sin_a = float(np.sin(a))
    turned = [
        np.column_stack([xy[:, 0] * cos_a - xy[:, 1] * sin_a,
                         xy[:, 0] * sin_a + xy[:, 1] * cos_a])
        for xy in xy_list
    ]
    x_le = float(turned[0][leading_edge_index(turned[0]), 0])
    return [t[:, 0] - x_le for t in turned]


def _compare_series(
    compare: Any, index: int
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Return the (x, Cp, label) series of the compare argument.

    A result object uses the same angle index; an index out of its range
    raises IndexError. An (x, Cp) pair of arrays plots as one point
    series.
    """
    if hasattr(compare, "Cp") and hasattr(compare, "coordinates"):
        n_compare = _n_angles(compare)
        if index >= n_compare:
            raise IndexError(
                f"compare holds {n_compare} angle(s), so angle index {index} is "
                "out of range. Give a compare result with the same angles, or "
                "the (x, Cp) arrays of the wanted angle."
            )
        cp_list = _body_cp_list(compare, index)
        xy_list = _body_coords_list(compare, index, len(cp_list))
        out = []
        for b, (xy, cp) in enumerate(zip(xy_list, cp_list)):
            label = f"compare body {b}" if len(cp_list) > 1 else "compare"
            out.append((np.asarray(xy[:, 0], dtype=float), cp, label))
        return out
    try:
        x_ref, cp_ref = compare
    except (TypeError, ValueError):
        raise TypeError(
            "compare must be a Panel2DResult or an (x, Cp) pair of arrays."
        ) from None
    x_arr = _to_numpy(x_ref).ravel()
    cp_arr = _to_numpy(cp_ref).ravel()
    if x_arr.shape != cp_arr.shape:
        raise ValueError(
            f"compare arrays must share one shape, got {x_arr.shape} "
            f"and {cp_arr.shape}."
        )
    return [(x_arr, cp_arr, "reference")]


def plot_airfoil(
    shape_or_coords: AirfoilShape | np.ndarray,
    ax: plt.Axes | None = None,
    show_nodes: bool = False,
    ground_height: float | None = None,
) -> plt.Axes:
    """Plot an airfoil section contour, with panel nodes and ground.

    Parameters
    ----------
    shape_or_coords : AirfoilShape or numpy.ndarray
        Section shape or point coordinates [chord units], shape (N, 2),
        in Selig order.
    ax : matplotlib Axes or None
        Axes to draw on. None creates new axes.
    show_nodes : bool
        True marks the panel nodes with circles. Default False.
    ground_height : float or None
        Height of the ground line [chord units], in the same units as the
        coordinates. None draws no ground. Default None.

    Returns
    -------
    matplotlib Axes
        The axes with the airfoil contour (one line), the node markers
        when requested (one more line) and the ground line when requested
        (one more line).
    """
    if isinstance(shape_or_coords, AirfoilShape):
        coords = np.asarray(shape_or_coords.coords, dtype=float)
        name = shape_or_coords.name
    else:
        coords = np.asarray(shape_or_coords, dtype=float)
        name = "airfoil"
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Airfoil coordinates must have shape (N, 2), got {coords.shape}."
        )
    if coords.shape[0] < 2:
        raise ValueError("Airfoil needs at least two points.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 3.8))

    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color="#2563eb",
        linewidth=1.8,
        label=name,
    )
    if show_nodes:
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color="#dc2626",
            marker="o",
            linestyle="None",
            markersize=3,
            label="nodes",
        )
    if ground_height is not None:
        ground = float(ground_height)
        ax.axhline(
            ground, color="#059669", linestyle="--", linewidth=1.5,
            label="ground",
        )
        lo = min(float(np.min(coords[:, 1])), ground)
        hi = float(np.max(coords[:, 1]))
        pad = 0.08 * max(hi - lo, 1e-12)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("x [chord units]")
    ax.set_ylabel("y [chord units]")
    ax.set_title(f"Airfoil shape: {name}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    return ax


def plot_cp(
    result: Panel2DResult,
    index: int = 0,
    ax: plt.Axes | None = None,
    invert: bool = True,
    compare: Any = None,
) -> plt.Axes:
    """Plot the Cp distribution of a 2D panel result against x.

    The curve uses the node Cp values of Katz and Plotkin (2001,
    chapter 11): Cp = 1 - (V/V_inf)^2. The x stations [chord units] are
    the node x coordinates of the selected angle as the result stores
    them. A free-air result stores the input nodes (body frame). A
    ground-effect result stores the nodes turned by the angle of attack
    (solved frame); :func:`plot_ground_effect_cp` shows those results in
    the body frame. The axis label names the frame.

    Parameters
    ----------
    result : Panel2DResult
        Inviscid panel result in free air or in ground effect.
    index : int
        Angle index [-] for a sweep result. A single-angle result needs
        index 0. Default 0.
    ax : matplotlib Axes or None
        Axes to draw on. None creates new axes.
    invert : bool
        True puts suction up (inverted Cp axis). Default True.
    compare : Panel2DResult or (x, Cp) pair or None
        Second result at the same angle index, or reference stations
        x [chord units] with Cp values [-]. A compare result with fewer
        angles than ``index + 1`` raises IndexError. The compare data
        draws as points. None draws no compare data. Default None.

    Returns
    -------
    matplotlib Axes
        The axes with one line per body, plus one point series per
        compare body (or one point series for array data).
    """
    idx = _check_index(index)
    cp_list = _body_cp_list(result, idx)
    xy_list = _body_coords_list(result, idx, len(cp_list))
    alpha = _scalar_at(result.alpha, idx)
    cl = _scalar_at(result.Cl, idx)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 5.0))

    for b, (xy, cp) in enumerate(zip(xy_list, cp_list)):
        label = f"body {b}" if len(cp_list) > 1 else "panel method"
        ax.plot(
            xy[:, 0], cp, color="#2563eb", linewidth=1.8, label=label,
        )

    if compare is not None:
        for x_c, cp_c, label in _compare_series(compare, idx):
            ax.plot(
                x_c, cp_c, color="black", marker="o", linestyle="None",
                markersize=4, label=label,
            )

    ax.set_xlabel("x [chord units], node frame of the result")
    ax.set_ylabel("Cp [-]")
    ax.set_title(
        f"Cp distribution at alpha = {alpha:.2f} deg (Cl = {cl:.4f})"
    )
    if invert:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    return ax


def plot_ground_effect_cp(
    results_by_height: dict[float, Panel2DResult] | list[tuple[float, Panel2DResult]],
    index: int = 0,
    ax: plt.Axes | None = None,
    invert: bool = True,
) -> plt.Axes:
    """Plot Cp curves of several ground heights on one axes.

    Each height holds the panel result at that height above the ground.
    The curves share the angle index, so one call shows how the suction
    peak and the loading grow near the ground.

    The x stations are in the body frame. The solver stores the nodes
    turned clockwise by the angle of attack; this function turns them
    back by the angle of attack of each result and measures x from the
    leading-edge node of the first body (the node farthest from its
    trailing-edge midpoint). For a section with the leading edge at
    (0, 0) and unit chord, this x equals the input x/c.

    Parameters
    ----------
    results_by_height : dict or list of pairs
        Height h/c [-] (the height in chord units of a unit-chord
        section) mapped to its Panel2DResult. A list holds (h/c, result)
        pairs. Heights sort in increasing order.
    index : int
        Angle index [-] for sweep results. A single-angle result needs
        index 0. Default 0.
    ax : matplotlib Axes or None
        Axes to draw on. None creates new axes.
    invert : bool
        True puts suction up (inverted Cp axis). Default True.

    Returns
    -------
    matplotlib Axes
        The axes with one line per height (times the body count).
    """
    if hasattr(results_by_height, "items"):
        items = list(results_by_height.items())
    else:
        items = list(results_by_height)
    if not items:
        raise ValueError("results_by_height must hold at least one result.")
    pairs: list[tuple[float, Panel2DResult]] = []
    for entry in items:
        try:
            height, res = entry
        except (TypeError, ValueError):
            raise ValueError(
                "results_by_height must map heights to results, "
                "or hold (height, result) pairs."
            ) from None
        if not hasattr(res, "Cp") or not hasattr(res, "coordinates"):
            raise TypeError("Each value must be a Panel2DResult.")
        pairs.append((float(height), res))
    pairs.sort(key=lambda p: p[0])

    idx = _check_index(index)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 5.0))

    alpha = _scalar_at(pairs[0][1].alpha, idx)
    for height, res in pairs:
        cp_list = _body_cp_list(res, idx)
        xy_list = _body_coords_list(res, idx, len(cp_list))
        x_body = _body_frame_x(xy_list, _scalar_at(res.alpha, idx))
        for b, (x_b, cp) in enumerate(zip(x_body, cp_list)):
            if len(cp_list) > 1:
                label = f"h/c = {height:g}, body {b}"
            else:
                label = f"h/c = {height:g}"
            ax.plot(x_b, cp, linewidth=1.8, label=label)

    ax.set_xlabel("x/c [-], body frame, from the leading edge")
    ax.set_ylabel("Cp [-]")
    ax.set_title(
        f"Ground effect on Cp at alpha = {alpha:.2f} deg"
    )
    if invert:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, title="height")
    return ax
