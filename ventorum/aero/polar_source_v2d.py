# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Section polar source driven by the Ventorum-2D viscous panel solver.

This module wraps the in-house two-dimensional coupled viscous-inviscid
panel solver into the SectionPolarSource protocol. It evaluates clean or
flapped section polars in free air or in ground effect.

Sources: Drela, M. and Giles, M. B., 1987, Viscous-inviscid analysis of
transonic and low Reynolds number airfoils, AIAA Journal; Drela, M., 1989,
XFOIL: An analysis and design system for low Reynolds number airfoils;
Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed. (ground effect
by method of images); Glauert, H., 1927, Theoretical relationships for an
aerofoil with hinged flap, R&M 1095.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ventorum.aero.polar_cache import PolarCache
from ventorum.airfoil import airfoil
from ventorum.core.datatypes import TabulatedAirfoil
from ventorum.panel2d.viscous import solve_viscous

__all__ = ["Ventorum2DPolarSource"]


class Ventorum2DPolarSource:
    """Section polar source that runs the Ventorum-2D panel solver.

    Parameters
    ----------
    cache : bool, None or PolarCache
        True (default) uses a :class:`PolarCache` in the default folder.
        A PolarCache instance uses that folder. False or None disables
        caching.
    n_panels : int
        Number of panels [-] for repaneling the section. Default is 160.
    ncrit : float
        Critical amplification exponent [-] for transition. Default is 9.0.
    xtr : tuple of (float, float)
        Forced transition locations (upper, lower) as fraction of chord [-].
        Default is (1.0, 1.0).
    backend : {"numpy"}
        Computational backend. Default is "numpy".
    """

    supports_ground: bool = True

    def __init__(
        self,
        cache: bool | None | PolarCache = True,
        n_panels: int = 160,
        ncrit: float = 9.0,
        xtr: tuple[float, float] = (1.0, 1.0),
        backend: str = "numpy",
    ) -> None:
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
        try:
            n_pan = int(n_panels)
        except (TypeError, ValueError):
            raise ValueError(
                f"n_panels must be an integer, got {n_panels!r}."
            ) from None
        if n_pan < 2:
            raise ValueError(f"n_panels must be at least 2, got {n_pan}.")
        try:
            nc = float(ncrit)
        except (TypeError, ValueError):
            raise ValueError(f"ncrit must be a float, got {ncrit!r}.") from None
        if not np.isfinite(nc) or nc <= 0.0:
            raise ValueError(f"ncrit must be positive, got {ncrit}.")
        try:
            xtr_tup = (float(xtr[0]), float(xtr[1]))
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"xtr must be a sequence of 2 floats, got {xtr!r}."
            ) from None
        if not (0.0 <= xtr_tup[0] <= 1.0 and 0.0 <= xtr_tup[1] <= 1.0):
            raise ValueError(f"xtr elements must lie in [0, 1], got {xtr_tup}.")
        if backend != "numpy":
            raise ValueError(f"backend must be 'numpy', got {backend!r}.")
        self._n_panels = n_pan
        self._ncrit = nc
        self._xtr = xtr_tup
        self._backend = str(backend)
        self.last_dropped: int = 0

    def polar(
        self,
        *args: Any,
        mach: float = 0.0,
        flap: tuple[float, float, float] | tuple[float, float] | None = None,
        height: float | None = None,
        **options: object,
    ) -> TabulatedAirfoil:
        """Return the Ventorum-2D section polar for the requested angles.

        Parameters
        ----------
        *args : object
            Positional inputs: ``section``, ``Re``, and ``alphas_deg``.
            ``section`` is a NACA designation string, coordinates of shape
            (N, 2) in Selig order, or an AirfoilShape instance.
            ``Re`` is the Reynolds number [-].
            ``alphas_deg`` is a sequence of angles of attack [deg].
        mach : float
            Mach number [-]. Must be 0.0.
        flap : tuple or None
            ``(x_hinge, y_hinge, deflection_deg)`` or
            ``(x_hinge, deflection_deg)``: hinge chord position [-],
            optional thickness fraction [-], and deflection [deg]
            (positive down).
        height : float or None
            Section height above the ground normalized by chord [-],
            or None for free air.
        **options
            Extra keywords forwarded to :func:`solve_viscous` (for example
            ``ncrit``, ``xtr``, ``max_iter``, ``tol``, ``wake_length``).

        Returns
        -------
        TabulatedAirfoil
            The section polar over the converged requested angles. Angles
            are in radians. Non-converged angles are dropped.
        """
        clash_candidates = {"section", "Re", "alphas_deg", "mach", "flap", "height"}
        clash = clash_candidates & set(options)
        if clash:
            raise ValueError(
                f"Option(s) {sorted(clash)} clash with the named inputs of polar(); "
                "pass them as named inputs instead."
            )
        if len(args) < 3:
            raise TypeError(
                f"polar() takes at least 3 arguments (section, Re, alphas_deg), got {len(args)}."
            )
        section = args[0]
        Re = args[1]
        alphas_deg = args[2]

        try:
            mach_val = float(mach)
        except (TypeError, ValueError):
            raise ValueError(f"mach must be a float, got {mach!r}.") from None
        if abs(mach_val) > 1e-12:
            raise ValueError(
                f"Only mach=0.0 is supported in Ventorum2DPolarSource, got mach={mach}."
            )
        try:
            re_val = float(Re)
        except (TypeError, ValueError):
            raise ValueError(f"Re must be a float, got {Re!r}.") from None
        if not np.isfinite(re_val) or re_val <= 0.0:
            raise ValueError(f"Re must be positive, got {Re}.")
        alphas = np.atleast_1d(np.asarray(alphas_deg, dtype=float))
        if alphas.size == 0:
            raise ValueError("alphas_deg must hold at least one angle [deg].")
        if not np.all(np.isfinite(alphas)):
            raise ValueError("alphas_deg must hold finite values only.")
        h_val: float | None = None
        if height is not None:
            try:
                h_val = float(height)
            except (TypeError, ValueError):
                raise ValueError(
                    f"height must be a float, got {height!r}."
                ) from None
            if not np.isfinite(h_val) or h_val <= 0.0:
                raise ValueError(f"height must be positive, got {height}.")

        flap_tup: tuple[float, float, float] | None = None
        if flap is not None:
            try:
                if len(flap) == 3:
                    flap_tup = (float(flap[0]), float(flap[1]), float(flap[2]))
                elif len(flap) == 2:
                    flap_tup = (float(flap[0]), 0.5, float(flap[1]))
                else:
                    raise ValueError(
                        f"flap must hold 2 or 3 numbers, got {flap}."
                    )
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"Invalid flap argument: {exc}.") from exc

        if isinstance(section, str):
            shape = airfoil(section, n_points=self._n_panels + 1)
            key_section: object = section.strip().lower()
        else:
            shape = airfoil(section)
            key_section = [
                [round(float(v), 9) for v in row] for row in shape.coords
            ]

        if len(shape.coords) != self._n_panels + 1:
            shape = shape.repanel(n_panels=self._n_panels)

        if flap_tup is not None:
            shape = shape.with_flap(
                x_hinge=flap_tup[0],
                deflection=flap_tup[2],
                y_hinge=flap_tup[1],
            )

        settings: dict[str, Any] = {
            "ncrit": float(options.get("ncrit", self._ncrit)),
            "xtr": tuple(options.get("xtr", self._xtr)),
            "backend": str(options.get("backend", self._backend)),
        }
        solver_options = dict(options)
        solver_options.pop("ncrit", None)
        solver_options.pop("xtr", None)
        solver_options.pop("backend", None)
        solver_options.setdefault("tol", 1e-4)
        solver_options.setdefault("max_iter", 50)

        unique_req = np.sort(np.unique(alphas))
        payload: dict[str, Any] | None = None
        if self._cache is not None:
            payload = {
                "source": "Ventorum2DPolarSource",
                "section": key_section,
                "Re": float(re_val),
                "mach": float(mach_val),
                "flap": None if flap_tup is None else list(flap_tup),
                "height": h_val,
                "alphas_deg": [round(float(v), 9) for v in unique_req],
                "n_panels": self._n_panels,
                "ncrit": settings["ncrit"],
                "xtr": list(settings["xtr"]),
                "backend": settings["backend"],
            }
            for k, v in sorted(solver_options.items()):
                payload[k] = v
            hit = self._cache.get(payload)
            if hit is not None:
                return hit

        sweep_alphas = unique_req

        res = solve_viscous(
            shape.coords,
            sweep_alphas,
            re_val,
            mach=0.0,
            ncrit=settings["ncrit"],
            xtr=settings["xtr"],
            height=h_val,
            backend=settings["backend"],
            **solver_options,
        )

        if np.ndim(res.alpha) == 0:
            res_alphas = np.array([float(res.alpha)])
            res_cls = np.array([float(res.Cl)])
            res_cds = np.array([float(res.Cd)])
            res_cms = np.array([float(res.Cm)])
            res_conv = np.array([bool(res.converged)])
        else:
            res_alphas = np.asarray(res.alpha, dtype=float)
            res_cls = np.asarray(res.Cl, dtype=float)
            res_cds = np.asarray(res.Cd, dtype=float)
            res_cms = np.asarray(res.Cm, dtype=float)
            res_conv = np.asarray(res.converged, dtype=bool)

        kept_alphas: list[float] = []
        kept_cls: list[float] = []
        kept_cds: list[float] = []
        kept_cms: list[float] = []

        for a in unique_req:
            matches = np.flatnonzero(np.isclose(res_alphas, a, atol=1e-6))
            if matches.size > 0:
                idx = int(matches[0])
                if res_conv[idx]:
                    kept_alphas.append(float(res_alphas[idx]))
                    kept_cls.append(float(res_cls[idx]))
                    kept_cds.append(float(res_cds[idx]))
                    kept_cms.append(float(res_cms[idx]))

        n_requested = unique_req.size
        dropped = n_requested - len(kept_alphas)
        self.last_dropped = dropped

        h_str = "free air" if h_val is None else f"{h_val:g}"
        name = (
            f"Ventorum-2D {shape.name} Re={re_val:.6g} h={h_str} "
            f"({dropped} of {n_requested} alpha dropped)"
        )
        polar = TabulatedAirfoil(
            name=name,
            alpha=np.radians(np.asarray(kept_alphas, dtype=float)),
            Cl_data=np.asarray(kept_cls, dtype=float),
            Cd_data=np.asarray(kept_cds, dtype=float),
            Cm_data=np.asarray(kept_cms, dtype=float),
            Re=re_val,
        )

        if payload is not None and self._cache is not None:
            self._cache.put(payload, polar)

        return polar
