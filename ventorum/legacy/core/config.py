# Author: Manuel Alejandro Rodriguez Diaz, PhD
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Any

from ventorum.legacy.core.datatypes import (
    Aircraft,
    LiftingSurface,
    WingSection,
    LinearAirfoil,
    TabulatedAirfoil,
    aircraft_to_json,
    aircraft_from_json,
)

class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def save_aircraft_to_json(aircraft: Aircraft, filepath: str | Path):
    """
    Serialize an Aircraft object and save it to a JSON file.
    
    Parameters
    ----------
    aircraft : Aircraft
        The aircraft to serialize.
    filepath : str | Path
        The output file path.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(aircraft_to_json(aircraft))


def load_aircraft_from_json(filepath: str | Path) -> Aircraft:
    """
    Load an Aircraft object from a JSON file.
    
    Parameters
    ----------
    filepath : str | Path
        The input file path.
        
    Returns
    -------
    Aircraft
        The loaded aircraft object.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return aircraft_from_json(f.read())



def dict_to_aircraft(data: dict[str, Any]) -> Aircraft:
    """
    Convert a dictionary to an Aircraft object.
    """
    surfaces = []
    for surf_data in data.get('surfaces', []):
        sections = []
        for sec_data in surf_data.get('sections', []):
            af_data = sec_data.pop('airfoil', {})
            af_type = af_data.pop('_type', 'LinearAirfoil')
            
            if af_type == 'TabulatedAirfoil':
                # Convert list back to numpy arrays
                for k in ['alpha', 'Cl', 'Cd', 'Cm']:
                    if k in af_data and af_data[k] is not None:
                        af_data[k] = np.array(af_data[k])
                airfoil = TabulatedAirfoil(**af_data)
            else:
                airfoil = LinearAirfoil(**af_data)
                
            sec_data['airfoil'] = airfoil
            sections.append(WingSection(**sec_data))
            
        surf_data['sections'] = sections
        
        # Convert position to numpy array
        if 'position' in surf_data and surf_data['position'] is not None:
            surf_data['position'] = np.array(surf_data['position'])
            
        surfaces.append(LiftingSurface(**surf_data))
        
    data['surfaces'] = surfaces
    return Aircraft(**data)
