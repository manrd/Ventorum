# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Common interface for 2D section polar sources.

XFOIL provides the first implementation, driven as a black box through its
documented interactive commands (M. Drela and H. Youngren, XFOIL 6.99 user
guide). The in-house 2D solver follows the same protocol
(:mod:`ventorum.aero.polar_source_v2d`).

The plain-flap tuple is the same in every source:
``(x_hinge, y_hinge, deflection_deg)``, with the hinge position as a
fraction of the chord [-], the hinge height as a fraction of the local
thickness [-] (0 lower surface, 1 upper surface) and the deflection [deg],
positive trailing edge down.

Sources: Drela, M., 1989, XFOIL: An analysis and design system for low
Reynolds number airfoils; Drela, M. and Youngren, H., XFOIL 6.99 user guide
(documented commands only).
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.xfoil_runner import (
    check_flap,
    resolve_xfoil_executable,
    run_xfoil,
    xfoil_version,
)
from ventorum.core.datatypes import TabulatedAirfoil

#: Inputs of :func:`run_xfoil` that :meth:`XfoilPolarSource.polar` sets
#: itself from its named arguments. They are not accepted as options.
_NAMED_INPUTS = frozenset(
    {
        "airfoil",
        "Re",
        "alpha_start",
        "alpha_end",
        "alpha_step",
        "mach",
        "flap",
        "coordinates",
    }
)

#: Defaults of the :func:`run_xfoil` options (``ncrit``, ``xtr_top``,
#: ``xtr_bottom``, ``n_panels``, ``max_iter``, ``timeout``, ``sweep``,
#: ``executable``), read from its signature so that the cache key always
#: covers every input of the run.
_RUN_XFOIL_OPTION_DEFAULTS: dict[str, Any] = {
    name: param.default
    for name, param in inspect.signature(run_xfoil).parameters.items()
    if name not in _NAMED_INPUTS and param.default is not inspect.Parameter.empty
}

#: Sweep order that :class:`XfoilPolarSource` uses when the caller does
#: not give one (better convergence near stall than one linear sweep).
SOURCE_DEFAULT_SWEEP = "from_zero"


def _json_value(value: object) -> object:
    """Return *value* as a plain JSON type for the cache key."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in np.asarray(value, dtype=object).ravel()]
    return repr(value)


@runtime_checkable
class SectionPolarSource(Protocol):
    """Protocol for 2D section polar providers."""

    def polar(
        self,
        section: str | np.ndarray,
        Re: float,
        alphas_deg: Sequence[float] | np.ndarray,
        *,
        mach: float = 0.0,
        flap: tuple[float, float, float] | None = None,
        **options: object,
    ) -> TabulatedAirfoil:
        """Return the section polar that covers the requested angles.

        Parameters
        ----------
        section : str or np.ndarray
            A NACA name (for example ``"naca 0012"``) or points (N, 2) in
            Selig order: from the trailing edge along the upper surface to
            the leading edge, then along the lower surface back to the
            trailing edge.
        Re : float
            Reynolds number [-].
        alphas_deg : sequence of float
            Requested angles of attack [deg]. The returned polar covers this
            range; intermediate angles come from its interpolation.
        mach : float
            Mach number [-].
        flap : tuple or None
            ``(x_hinge, y_hinge, deflection_deg)``: hinge position as a
            fraction of the chord [-], hinge height as a fraction of the
            local thickness [-] (0 lower surface, 1 upper surface) and
            deflection [deg], positive down.
        **options
            Extra provider keywords (for XFOIL: ``ncrit``, ``xtr_top``,
            ``xtr_bottom``, ``n_panels``, ``sweep``, ``timeout``,
            ``max_iter``, ``executable``). A source that supports ground
            effect also takes ``height`` [-] (height above the ground in
            chords).

        Returns
        -------
        TabulatedAirfoil
            The section polar over the requested range.
        """
        ...


class XfoilPolarSource:
    """Section polar source that runs XFOIL, with an optional disk cache.

    Parameters
    ----------
    cache : bool, None or PolarCache
        True (default) uses a :class:`PolarCache` in the default folder.
        A PolarCache instance uses that folder. False or None disables
        caching.
    **xfoil_options
        Default keywords forwarded to :func:`run_xfoil` (``ncrit``,
        ``xtr_top``, ``xtr_bottom``, ``n_panels``, ``max_iter``,
        ``timeout``, ``sweep``, ``executable``). Per-call keywords of
        :meth:`polar` override them. The sweep order is ``"from_zero"``
        unless given.

    Notes
    -----
    The cache key holds every input of the XFOIL run: the section, the
    Reynolds number, the Mach number, the flap, the angle grid, every
    :func:`run_xfoil` option and the version line of the XFOIL executable
    that runs the case. The version is read for the executable of each
    call, so a call with another executable never reads a polar of the
    first one. When the version of the executable is unknown (XFOIL does
    not start or prints no version line), the cache is not used for that
    call.
    """

    def __init__(self, cache: bool | None | PolarCache = True, **xfoil_options: object) -> None:
        if isinstance(cache, PolarCache):
            self._cache: PolarCache | None = cache
        elif cache is True:
            self._cache = PolarCache()
        elif cache is False or cache is None:
            self._cache = None
        else:
            raise TypeError(
                f"cache must be True, False, None or a PolarCache, got {cache!r}."
            )
        self._check_option_names(xfoil_options)
        self._xfoil_options = dict(xfoil_options)

    @staticmethod
    def _check_option_names(options: dict[str, object]) -> None:
        """Refuse option names that are named inputs or unknown to run_xfoil."""
        clash = _NAMED_INPUTS & set(options)
        if clash:
            raise ValueError(
                f"Option(s) {sorted(clash)} clash with the named inputs of polar(); "
                "pass them as named inputs instead."
            )
        unknown = set(options) - set(_RUN_XFOIL_OPTION_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown XFOIL option(s) {sorted(unknown)}; allowed: "
                f"{sorted(_RUN_XFOIL_OPTION_DEFAULTS)}."
            )

    def effective_options(self, **options: object) -> dict[str, Any]:
        """Return every :func:`run_xfoil` option that a call would use.

        Parameters
        ----------
        **options
            Per-call options (see :meth:`polar`).

        Returns
        -------
        dict
            The function defaults, then the constructor options, then
            *options*. The sweep default of the source is ``"from_zero"``.
        """
        self._check_option_names(options)
        settings: dict[str, Any] = dict(_RUN_XFOIL_OPTION_DEFAULTS)
        settings["sweep"] = SOURCE_DEFAULT_SWEEP
        settings.update(self._xfoil_options)
        settings.update(options)
        return settings

    def describe(self, **options: object) -> dict[str, Any]:
        """Return the name, the options and the XFOIL version of this source.

        Polar tables record this dictionary as their provenance.

        Parameters
        ----------
        **options
            Per-call options to include (see :meth:`polar`).

        Returns
        -------
        dict
            ``{"source": "XfoilPolarSource", "options": {...}, "version":
            str}``. The options hold every :func:`run_xfoil` option except
            the executable; the version is the banner line of the XFOIL
            executable that the source runs (``"unknown"`` when it does not
            start).
        """
        settings = self.effective_options(**options)
        exe = resolve_xfoil_executable(settings.get("executable"))
        opts = {k: v for k, v in settings.items() if k != "executable"}
        return {
            "source": type(self).__name__,
            "options": opts,
            "version": xfoil_version(exe),
        }

    def polar(
        self,
        section: str | np.ndarray,
        Re: float,
        alphas_deg: Sequence[float] | np.ndarray,
        *,
        mach: float = 0.0,
        flap: tuple[float, float, float] | None = None,
        **options: object,
    ) -> TabulatedAirfoil:
        """Return the XFOIL section polar that covers the requested angles.

        Parameters
        ----------
        section : str or np.ndarray
            A NACA name (for example ``"naca 0012"``) or points (N, 2) in
            Selig order.
        Re : float
            Reynolds number [-].
        alphas_deg : sequence of float
            Requested angles of attack [deg]. XFOIL runs a uniform sweep
            from the smallest to the largest value; the returned polar
            covers that range and interpolates between its points.
        mach : float
            Mach number [-].
        flap : tuple or None
            ``(x_hinge, y_hinge, deflection_deg)``: hinge position as a
            fraction of the chord [-], hinge height as a fraction of the
            local thickness [-] and deflection [deg], positive down.
        **options
            Extra :func:`run_xfoil` keywords (``ncrit``, ``xtr_top``,
            ``xtr_bottom``, ``n_panels``, ``sweep``, ``timeout``,
            ``max_iter``, ``executable``). They override the constructor
            defaults, but must not repeat the named arguments above.

        Returns
        -------
        TabulatedAirfoil
            The section polar over the requested range.
        """
        settings = self.effective_options(**options)
        alphas = np.atleast_1d(np.asarray(alphas_deg, dtype=float))
        if alphas.size == 0:
            raise ValueError("alphas_deg must hold at least one angle [deg].")
        if not np.all(np.isfinite(alphas)):
            raise ValueError("alphas_deg must hold finite values only.")
        flap_tuple = check_flap(flap)
        if isinstance(section, str):
            airfoil = section
            coordinates = None
            key_section: object = section.strip().lower()
        else:
            coordinates = np.asarray(section, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise ValueError(
                    "Section coordinates must have shape (N, 2), "
                    f"got {coordinates.shape}."
                )
            airfoil = "coordinates"
            key_section = [
                [round(float(v), 9) for v in row] for row in coordinates
            ]
        lo, hi = float(alphas.min()), float(alphas.max())
        uniq = np.unique(alphas)
        step = 0.5 if uniq.size == 1 else float(np.diff(uniq).min())
        payload: dict[str, object] | None = None
        if self._cache is not None:
            exe = resolve_xfoil_executable(settings.get("executable"))
            version = xfoil_version(exe)
            if version != "unknown":
                payload = {
                    "source": type(self).__name__,
                    "section": key_section,
                    "Re": float(Re),
                    "mach": float(mach),
                    "flap": None if flap_tuple is None else list(flap_tuple),
                    "alphas_deg": sorted(float(v) for v in alphas),
                    "alpha_step": step,
                    "options": {
                        k: _json_value(v)
                        for k, v in sorted(settings.items())
                        if k != "executable"
                    },
                    "xfoil_version": version,
                }
                hit = self._cache.get(payload)
                if hit is not None:
                    return hit
        polar = run_xfoil(
            airfoil,
            float(Re),
            lo,
            hi,
            step,
            mach=mach,
            flap=flap_tuple,
            coordinates=coordinates,
            **settings,
        )
        if payload is not None and self._cache is not None:
            self._cache.put(payload, polar)
        return polar
