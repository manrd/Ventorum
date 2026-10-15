# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Common interface for 2D section polar sources.

XFOIL provides the first implementation, driven as a black box through its
documented interactive commands (M. Drela and H. Youngren, XFOIL 6.99 user
guide). A second implementation for the in-house 2D solver will follow the
same protocol later.

Sources: Drela, M., 1989, XFOIL: An analysis and design system for low
Reynolds number airfoils; Drela, M. and Youngren, H., XFOIL 6.99 user guide
(documented commands only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.xfoil_runner import run_xfoil, xfoil_version
from ventorum.core.datatypes import TabulatedAirfoil

#: Defaults of :func:`run_xfoil` repeated here so the cache key covers them.
_KEY_DEFAULTS: dict[str, object] = {
    "ncrit": 9.0,
    "xtr_top": 1.0,
    "xtr_bottom": 1.0,
    "n_panels": None,
    "sweep": "from_zero",
}


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
            Reynolds number.
        alphas_deg : sequence of float
            Requested angles of attack [deg]. The returned polar covers this
            range; intermediate angles come from its interpolation.
        mach : float
            Mach number.
        flap : tuple or None
            ``(x_hinge, y_hinge, deflection_deg)``: hinge position in
            airfoil chord units and deflection [deg], positive down.
        **options
            Extra provider keywords (for XFOIL: ``ncrit``, ``xtr_top``,
            ``xtr_bottom``, ``n_panels``, ``sweep``, ``timeout``,
            ``max_iter``, ``executable``).

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
        Default keywords forwarded to :func:`run_xfoil` (for example
        ``ncrit``, ``sweep``, ``timeout``). Per-call keywords of
        :meth:`polar` override them.
    """

    def __init__(self, cache: bool | None | PolarCache = True, **xfoil_options: object) -> None:
        if isinstance(cache, PolarCache):
            self._cache = cache
        elif cache is True:
            self._cache = PolarCache()
        elif cache is False or cache is None:
            self._cache = None
        else:
            raise TypeError(
                f"cache must be True, False, None or a PolarCache, got {cache!r}."
            )
        self._xfoil_options = dict(xfoil_options)
        self._version: str | None = None

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
            Reynolds number.
        alphas_deg : sequence of float
            Requested angles of attack [deg]. XFOIL runs a uniform sweep
            from the smallest to the largest value; the returned polar
            covers that range and interpolates between its points.
        mach : float
            Mach number.
        flap : tuple or None
            ``(x_hinge, y_hinge, deflection_deg)``: hinge position in
            airfoil chord units and deflection [deg], positive down.
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
        clash = {
            "airfoil",
            "Re",
            "alpha_start",
            "alpha_end",
            "alpha_step",
            "mach",
            "flap",
            "coordinates",
        } & set(options)
        if clash:
            raise ValueError(
                f"Option(s) {sorted(clash)} clash with the named inputs of polar(); "
                "pass them as named inputs instead."
            )
        alphas = np.atleast_1d(np.asarray(alphas_deg, dtype=float))
        if alphas.size == 0:
            raise ValueError("alphas_deg must hold at least one angle [deg].")
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
        settings = dict(self._xfoil_options)
        settings.update(options)
        payload: dict[str, object] | None = None
        if self._cache is not None:
            if self._version is None:
                self._version = xfoil_version(settings.get("executable"))
            payload = {
                "section": key_section,
                "Re": float(Re),
                "mach": float(mach),
                "ncrit": settings.get("ncrit", _KEY_DEFAULTS["ncrit"]),
                "xtr_top": settings.get("xtr_top", _KEY_DEFAULTS["xtr_top"]),
                "xtr_bottom": settings.get(
                    "xtr_bottom", _KEY_DEFAULTS["xtr_bottom"]
                ),
                "flap": None
                if flap is None
                else [float(v) for v in flap],
                "alphas_deg": sorted(float(v) for v in alphas),
                "n_panels": settings.get("n_panels", _KEY_DEFAULTS["n_panels"]),
                "sweep": settings.get("sweep", _KEY_DEFAULTS["sweep"]),
                "xfoil_version": self._version,
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
            flap=flap,
            coordinates=coordinates,
            **settings,
        )
        if payload is not None and self._cache is not None:
            self._cache.put(payload, polar)
        return polar
