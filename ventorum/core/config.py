# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Save an aircraft to a JSON file and load it again."""

from __future__ import annotations

from pathlib import Path

from ventorum.core.datatypes import Aircraft, aircraft_from_json, aircraft_to_json

__all__ = ["aircraft_from_json", "aircraft_to_json", "load_aircraft_from_json", "save_aircraft_to_json"]


def save_aircraft_to_json(aircraft: Aircraft, filepath: str | Path) -> None:
    """Write *aircraft* to the JSON file *filepath*."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(aircraft_to_json(aircraft))


def load_aircraft_from_json(filepath: str | Path) -> Aircraft:
    """Read an aircraft from the JSON file *filepath*."""
    with open(filepath, encoding="utf-8") as f:
        return aircraft_from_json(f.read())
