# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Section polar source driven by the Ventorum-2D viscous panel solver.

This module wraps the in-house two-dimensional coupled viscous-inviscid
panel solver into the SectionPolarSource protocol. It evaluates clean or
flapped section polars in free air or in ground effect.

The flap tuple is the same as in every section polar source:
``(x_hinge, y_hinge, deflection_deg)``, with the hinge position as a
fraction of the chord [-], the hinge height as a fraction of the local
thickness [-] (0 lower surface, 1 upper surface) and the deflection [deg],
positive trailing edge down. By default the coefficients and the Reynolds
number use the reference chord of the section, and the moment is taken
about the quarter-chord point on its chord line; for a flapped section
both belong to the section before the flap deflection (the reference of
thin-airfoil theory, Glauert 1927): the source passes the flapped
:class:`ventorum.airfoil.AirfoilShape`, which holds that reference
geometry. The options ``chord_ref`` and ``moment_ref`` override them.

The cache key holds the package version, the solver revision
(``ventorum.panel2d.SOLVER_REVISION``) and a sha256 fingerprint of the source
files of the 2D packages (:func:`solver_fingerprint`), so a polar of an
older solver is never read back.

Sources: Drela, M. and Giles, M. B., 1987, Viscous-inviscid analysis of
transonic and low Reynolds number airfoils, AIAA Journal; Drela, M., 1989,
XFOIL: An analysis and design system for low Reynolds number airfoils;
Katz, J. and Plotkin, A., 2001, Low-Speed Aerodynamics, 2nd ed. (ground effect
by method of images); Glauert, H., 1927, Theoretical relationships for an
aerofoil with hinged flap, R&M 1095.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ventorum.aero.polar_cache import PolarCache
from ventorum.aero.xfoil_runner import check_flap
from ventorum.airfoil import airfoil
from ventorum.airfoil.naca import is_naca_designation
from ventorum.core.datatypes import TabulatedAirfoil
from ventorum.panel2d.solver import check_point
from ventorum.panel2d.viscous import check_chord_ref, solve_viscous

__all__ = ["Ventorum2DPolarSource", "solver_fingerprint", "solver_revision"]

#: Packages whose source files form the solver fingerprint.
_FINGERPRINT_PACKAGES = ("panel2d", "ibl2d", "airfoil")

#: Inputs of the 2D solver that :meth:`Ventorum2DPolarSource.polar` sets
#: itself; they are not accepted as options.
_NAMED_SOLVER_INPUTS = frozenset({"bodies", "alpha_deg", "Re", "mach", "height"})


def solver_revision() -> str:
    """Return the revision of the 2D viscous solver.

    Returns
    -------
    str
        ``ventorum.panel2d.SOLVER_REVISION``, which the 2D package raises
        when a change of the solver can change a result.
    """
    import ventorum.panel2d as panel2d

    return str(panel2d.SOLVER_REVISION)


@functools.lru_cache(maxsize=1)
def solver_fingerprint() -> str:
    """Return a sha256 fingerprint of the source files of the 2D solver.

    The fingerprint covers the Python files of ``ventorum.panel2d``,
    ``ventorum.ibl2d`` and ``ventorum.airfoil`` (names and contents), so
    every code change of the solver, the boundary layer or the geometry
    gives a new cache key.

    Returns
    -------
    str
        Hex digest.
    """
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in _FINGERPRINT_PACKAGES:
        for path in sorted((root / package).glob("*.py")):
            digest.update(f"{package}/{path.name}\n".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_version() -> str:
    """Return the version of the ventorum package."""
    import ventorum

    return str(getattr(ventorum, "__version__", "unknown"))


def _solver_option_names() -> set[str]:
    """Return the keyword options that the 2D viscous solver accepts."""
    params = inspect.signature(solve_viscous).parameters
    return {
        name
        for name, p in params.items()
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name not in _NAMED_SOLVER_INPUTS
    }


#: Keyword options of the 2D viscous solver, read from its signature.
_SOLVER_OPTIONS = frozenset(_solver_option_names())


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


def _reference_key(shape: Any) -> list[float] | None:
    """Return the stored reference geometry of a shape for the cache key, or None."""
    ref = getattr(shape, "reference", None)
    if ref is None:
        return None
    return [*ref.leading_edge, *ref.trailing_edge, float(ref.chord_ref)]


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

    Attributes
    ----------
    supports_ground : bool
        True: :meth:`polar` takes the ``height`` keyword.
    last_dropped : int
        Number of requested angles that did not converge in the last
        :meth:`polar` call that ran the solver (the polar name also
        states it).
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

    def _settings(self, options: dict[str, object]) -> dict[str, Any]:
        """Return the solver keywords of one call (defaults, then *options*)."""
        unknown = set(options) - _SOLVER_OPTIONS
        if unknown:
            raise ValueError(
                f"Unknown Ventorum-2D option(s) {sorted(unknown)}; allowed: "
                f"{sorted(_SOLVER_OPTIONS)}."
            )
        settings: dict[str, Any] = {
            "ncrit": self._ncrit,
            "xtr": self._xtr,
            "backend": self._backend,
            "tol": 1e-4,
            "max_iter": 50,
        }
        settings.update(options)
        settings["ncrit"] = float(settings["ncrit"])
        settings["xtr"] = tuple(float(v) for v in settings["xtr"])  # type: ignore[union-attr]
        settings["backend"] = str(settings["backend"])
        # The reference values: None is the default (left out of the key);
        # a given value is checked and stored as plain floats.
        if settings.get("chord_ref") is None:
            settings.pop("chord_ref", None)
        else:
            settings["chord_ref"] = check_chord_ref(settings["chord_ref"])
        if settings.get("moment_ref") is None:
            settings.pop("moment_ref", None)
        else:
            settings["moment_ref"] = check_point(settings["moment_ref"], "moment_ref")
        return settings

    def describe(self, **options: object) -> dict[str, Any]:
        """Return the name, the options and the version of this source.

        Polar tables record this dictionary as their provenance.

        Parameters
        ----------
        **options
            Per-call solver options to include (see :meth:`polar`).

        Returns
        -------
        dict
            ``{"source": "Ventorum2DPolarSource", "options": {...},
            "version": str}``. The version holds the package version, the
            solver revision and the first 12 characters of the solver
            fingerprint.
        """
        settings = self._settings(dict(options))
        opts = {"n_panels": self._n_panels}
        opts.update({k: _json_value(v) for k, v in sorted(settings.items())})
        version = (
            f"ventorum {_package_version()}, solver revision {solver_revision()}, "
            f"fingerprint {solver_fingerprint()[:12]}"
        )
        return {"source": type(self).__name__, "options": opts, "version": version}

    def polar(
        self,
        section: Any,
        Re: float,
        alphas_deg: Sequence[float] | np.ndarray,
        *,
        mach: float = 0.0,
        flap: tuple[float, float, float] | None = None,
        height: float | None = None,
        **options: object,
    ) -> TabulatedAirfoil:
        """Return the Ventorum-2D section polar for the requested angles.

        Parameters
        ----------
        section : str, np.ndarray or AirfoilShape
            A NACA designation string, coordinates of shape (N, 2) in
            Selig order [chord units], or an AirfoilShape instance.
        Re : float
            Reynolds number [-].
        alphas_deg : sequence of float
            Angles of attack [deg]. The solver sweeps from 0 in both
            directions with continuation (its own sweep rule).
        mach : float
            Mach number [-]. Must be 0.0.
        flap : tuple or None
            ``(x_hinge, y_hinge, deflection_deg)``: hinge position as a
            fraction of the chord [-], hinge height as a fraction of the
            local thickness [-] (0 lower surface, 1 upper surface) and
            deflection [deg], positive down.
        height : float or None
            Height above the ground in chords [-] of the reference point
            of the 2D solver (its default pivot, the quarter-chord point on
            the chord line of the section before a flap deflection), for
            a section of unit chord. None means free air.
        **options
            Extra keywords forwarded to :func:`solve_viscous` (for example
            ``ncrit``, ``xtr``, ``max_iter``, ``tol``, ``wake_length``,
            ``chord_ref`` and ``moment_ref``). ``chord_ref`` [chord units]
            is the reference chord of the coefficients and of the
            Reynolds number; ``moment_ref`` [chord units] is the moment
            reference point (x, y) in the frame of the section as given
            (before the repaneling and the flap deflection, which do not
            move the frame). Both win over the reference geometry of the
            section and are part of the cache key. Unknown names are
            refused. A named input given twice (for example ``Re``) is a
            TypeError of the call itself.

        Returns
        -------
        TabulatedAirfoil
            The section polar over the converged requested angles. Angles
            are in radians. Non-converged angles are dropped; the polar
            name and :attr:`last_dropped` give their number.
        """
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

        flap_tup = check_flap(flap)
        settings = self._settings(dict(options))

        if isinstance(section, str) and is_naca_designation(section):
            shape = airfoil(section, n_points=self._n_panels + 1)
            key_section: object = section.strip().lower()
        else:
            # Coordinates or a coordinate file: the factory takes no generator
            # options, and the cache key uses the points, not the file name.
            shape = airfoil(section)
            key_section = [
                [round(float(v), 9) for v in row] for row in shape.coords
            ]

        reference_key = _reference_key(shape)
        if len(shape.coords) != self._n_panels + 1:
            shape = shape.repanel(n_panels=self._n_panels)

        # The flapped shape holds the reference geometry of the section
        # before the deflection (reference chord and moment point).
        if flap_tup is not None:
            shape = shape.with_flap(
                x_hinge=flap_tup[0],
                deflection=flap_tup[2],
                y_hinge=flap_tup[1],
            )

        unique_req = np.sort(np.unique(alphas))
        payload: dict[str, Any] | None = None
        if self._cache is not None:
            payload = {
                "source": type(self).__name__,
                "package_version": _package_version(),
                "solver_revision": solver_revision(),
                "solver_fingerprint": solver_fingerprint(),
                "section": key_section,
                "reference": reference_key,
                "Re": float(re_val),
                "mach": float(mach_val),
                "flap": None if flap_tup is None else list(flap_tup),
                "height": h_val,
                "alphas_deg": [round(float(v), 9) for v in unique_req],
                "n_panels": self._n_panels,
                "options": {k: _json_value(v) for k, v in sorted(settings.items())},
            }
            hit = self._cache.get(payload)
            if hit is not None:
                return hit

        res = solve_viscous(
            shape,
            unique_req,
            re_val,
            mach=0.0,
            height=h_val,
            **settings,
        )

        res_alphas = np.atleast_1d(np.asarray(res.alpha, dtype=float))
        res_cls = np.atleast_1d(np.asarray(res.Cl, dtype=float))
        res_cds = np.atleast_1d(np.asarray(res.Cd, dtype=float))
        res_cms = np.atleast_1d(np.asarray(res.Cm, dtype=float))
        res_conv = np.atleast_1d(np.asarray(res.converged, dtype=bool))

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
