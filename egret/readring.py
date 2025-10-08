# readring.py
#
# Copyright (C) 2025 Hirokazu Maesaka (RIKEN SPring-8 Center)
#
# This file is part of Egret: Engine for General Research in
# Energetic-beam Tracking.
#
# Egret is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .ring import Ring
from .lattice import Lattice
from .element import Element
from .drift import Drift
from .dipole import Dipole
from .quadrupole import Quadrupole
from .sextupole import Sextupole
from .octupole import Octupole

from pathlib import Path
import latticejson
from typing import Dict, List, Any

def read_ring(path: str | Path) -> Ring:
    '''
    Read a ring configuration from a Lattice JSON file and return a Ring object.

    Args:
        path (str or Path): Path to the Lattice JSON file.
        
    Returns:
        Ring: The constructed Ring object.
    '''
    config = latticejson.load(path)
    rootname = config['root']
    elements = _make_elements(config)
    return Ring(rootname, [_make_lattice(config, name, elements) for name in config['lattices'][rootname]], config['energy']*1.e9, config['info'])

def _make_lattice(config: Dict[str, Any], name: str, elems: List[Lattice | Element]) \
    -> Lattice | Element:
    '''
    Recursively construct a Lattice or Element based on the lattice configuration.

    Args:
        config (dict): The lattice configuration dictionary.
        name (str): The name of the lattice or element to construct.
        elems (dict): A dictionary of Element objects.

    Returns:
        Lattice or Element: The constructed Lattice if 'name' is in lattices, otherwise the Element.
    '''
    if name in config['lattices']:
        return Lattice(name, [_make_lattice(config, k, elems) for k in config['lattices'][name]])
    return elems[name]

def _make_elements(config: Dict[str, Any]) -> Dict[str, Element]:
    '''
    Create a dictionary of Element objects from the lattice configuration.

    Args:
        config (dict): The lattice configuration dictionary.
        
    Returns:
        dict: A dictionary mapping element names to Element objects.

    Raises:
        KeyError: If an unknown element type is encountered.
    '''
    elements = {}
    for name,val in config['elements'].items():
        key = val[0]
        dat = val[1]
        match key:
            case 'Drift':
                elements[name] = Drift(name, dat['length'])
            case 'Dipole':
                elements[name] = Dipole(name, dat['length'], dat['angle'], dat['k1'])
            case 'Quadrupole':
                elements[name] = Quadrupole(name, dat['length'], dat['k1'])
            case 'Sextupole':
                elements[name] = Sextupole(name, dat['length'], dat['k2'])
            case 'Octupole':
                elements[name] = Octupole(name, dat['length'], dat['k3'])
            case _:
                raise KeyError(key)
    return elements
