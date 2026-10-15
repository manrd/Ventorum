# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Disk cache for airfoil section polars.

The key is the sha256 of a canonical JSON of the inputs. The value is the
TabulatedAirfoil stored as JSON (no pickle), so cache files stay readable
and safe to share.

Sources: the cache design follows the standard content-addressable store
pattern; no aerodynamic source is needed for this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ventorum.core.datatypes import TabulatedAirfoil


def _polar_to_dict(polar: TabulatedAirfoil) -> dict[str, Any]:
    """Serialise a TabulatedAirfoil to plain JSON types (angles in [rad])."""
    return {
        "name": polar.name,
        "alpha": np.asarray(polar.alpha, dtype=float).tolist(),
        "Cl": np.asarray(polar.Cl_data, dtype=float).tolist(),
        "Cd": np.asarray(polar.Cd_data, dtype=float).tolist(),
        "Cm": (
            None
            if polar.Cm_data is None
            else np.asarray(polar.Cm_data, dtype=float).tolist()
        ),
        "Re": polar.Re,
    }


def _polar_from_dict(data: dict[str, Any]) -> TabulatedAirfoil:
    """Rebuild a TabulatedAirfoil from :func:`_polar_to_dict` output."""
    return TabulatedAirfoil(
        name=data.get("name", ""),
        alpha=np.asarray(data["alpha"], dtype=float),
        Cl_data=np.asarray(data["Cl"], dtype=float),
        Cd_data=np.asarray(data["Cd"], dtype=float),
        Cm_data=(
            None if data.get("Cm") is None else np.asarray(data["Cm"], dtype=float)
        ),
        Re=data.get("Re"),
    )


class PolarCache:
    """Content-addressed disk cache for section polars.

    Parameters
    ----------
    directory : str, Path or None
        Cache folder. None uses the environment variable ``VENTORUM_CACHE``
        plus ``/polars``, else ``~/.cache/ventorum/polars``.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        if directory is not None:
            self._directory = Path(directory)
        elif os.environ.get("VENTORUM_CACHE"):
            self._directory = Path(os.environ["VENTORUM_CACHE"]) / "polars"
        else:
            self._directory = Path.home() / ".cache" / "ventorum" / "polars"
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        """Cache folder."""
        return self._directory

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        """Return the sha256 of the canonical JSON of *payload*.

        Parameters
        ----------
        payload : Mapping[str, Any]
            JSON-serialisable inputs (coordinates rounded to 1e-9 or the
            NACA string, Re, Mach, ncrit, xtr, flap, alpha grid, n_panels,
            sweep, and the XFOIL version line).

        Returns
        -------
        str
            Hex digest of the canonical JSON (sorted keys, compact).
        """
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def path_for(self, payload: Mapping[str, Any]) -> Path:
        """Return the cache file path for *payload*."""
        return self._directory / (self.key(payload) + ".json")

    def get(self, payload: Mapping[str, Any]) -> TabulatedAirfoil | None:
        """Return the cached polar for *payload*, or None on a miss.

        Parameters
        ----------
        payload : Mapping[str, Any]
            JSON-serialisable inputs (see :meth:`key`).

        Returns
        -------
        TabulatedAirfoil or None
            The stored polar, or None when absent or unreadable.
        """
        try:
            data = json.loads(self.path_for(payload).read_text())
        except (OSError, ValueError):
            return None
        try:
            return _polar_from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, payload: Mapping[str, Any], polar: TabulatedAirfoil) -> Path:
        """Store *polar* under *payload* and return the file path.

        Parameters
        ----------
        payload : Mapping[str, Any]
            JSON-serialisable inputs (see :meth:`key`).
        polar : TabulatedAirfoil
            The polar to store.

        Returns
        -------
        Path
            The cache file written.
        """
        path = self.path_for(payload)
        path.write_text(json.dumps(_polar_to_dict(polar), indent=2))
        return path

    def clear(self) -> int:
        """Remove all cached polars and return the file count.

        Returns
        -------
        int
            Number of cache files removed.
        """
        count = 0
        for path in self._directory.glob("*.json"):
            path.unlink()
            count += 1
        return count
