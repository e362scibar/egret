# cpp/ring.py
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

from __future__ import annotations
from logging import config
from os import path
from unicodedata import name
from ..base.ring import Ring as RingABC
from egret.cppegret import Ring as RingCPP
from .element import Element
from .coordinate import Coordinate
from .envelope import Envelope
from .dispersion import Dispersion
from .lattice import Lattice
from .drift import Drift
from .dipole import Dipole
from .quadrupole import Quadrupole
from .sextupole import Sextupole
from .octupole import Octupole
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any
import latticejson

class Ring(RingABC, Element):
    '''
    Ring accelerator class.
    '''

    def __init__(self, name: str, elements: List[Element], energy: float,
                 info: str = '', **kwargs) -> None:
        '''
        Initialize ring accelerator.

        Args:
            name str: Ring name.
            elements List[Element, ...]: List of elements in the ring.
            energy float: Beam energy [eV].
            info str: Additional information.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            cpp_elements = [elem.instance for elem in elements]
            from egret.cppegret import Ring as RingCPP
            self.instance = RingCPP(name, cpp_elements, energy, info)
        super().__init__(None, None, None, instance=self.instance)

    @property
    def energy(self) -> float:
        '''
        Beam energy [eV].
        '''
        return self.instance.energy

    @property
    def tune_x(self) -> float:
        '''
        Horizontal tune of the ring.
        '''
        return self.instance.tune_x

    @property
    def tune_y(self) -> float:
        '''
        Vertical tune of the ring.
        '''
        return self.instance.tune_y

    @property
    def cood0(self) -> Coordinate:
        '''
        Initial coordinate of the closed orbit.
        '''
        return Coordinate(instance=self.instance.cood0)

    @property
    def evlp0(self) -> Envelope:
        '''
        Initial beam envelope of the closed orbit.
        '''
        return Envelope(None, None, instance=self.instance.evlp0)

    @property
    def disp0(self) -> Dispersion:
        '''
        Initial dispersion of the closed orbit.
        '''
        return Dispersion(None, None, instance=self.instance.disp0)

    @property
    def emittance_x(self) -> float:
        '''
        Horizontal equilibrium emittance [m.rad].
        '''
        return self.instance.emittance_x

    @property
    def emittance_y(self) -> float:
        '''
        Vertical equilibrium emittance [m.rad].
        '''
        return self.instance.emittance_y

    @property
    def Jx(self) -> float:
        '''
        Horizontal damping partition number.
        '''
        return self.instance.Jx

    @property
    def Jy(self) -> float:
        '''
        Vertical damping partition number.
        '''
        return self.instance.Jy

    @property
    def Jz(self) -> float:
        '''
        Longitudinal damping partition number.
        '''
        return self.instance.Jz

    @property
    def I2(self) -> float:
        '''
        Radiation integral I2.
        '''
        return self.instance.I2

    @property
    def I4(self) -> float:
        '''
        Radiation integral I4.
        '''
        return self.instance.I4

    @property
    def I4u(self) -> float:
        '''
        Radiation integral I4u.
        '''
        return self.instance.I4u

    @property
    def I4v(self) -> float:
        '''
        Radiation integral I4v.
        '''
        return self.instance.I4v

    @property
    def I5u(self) -> float:
        '''
        Radiation integral I5u.
        '''
        return self.instance.I5u

    @property
    def I5v(self) -> float:
        '''
        Radiation integral I5v.
        '''
        return self.instance.I5v

    def update(self, delta: float = 0.):
        '''
        Update transfer matrix, dispersion, and emittance.

        Args:
            delta float: Relative momentum deviation (default: 0.).
        '''
        self.instance.update(delta)

    def find_initial_coordinate_of_closed_orbit(self, guess: Coordinate = None) -> None:
        '''
        Find initial coordinate of the closed orbit using Newton-Raphson method.

        Args:
            guess Coordinate: Initial guess of the closed orbit.
        '''
        if guess is None:
            self.instance.find_initial_coordinate_of_closed_orbit()
        else:
            self.instance.find_initial_coordinate_of_closed_orbit(guess.instance)

    @classmethod
    def read_json(cls, path: str) -> Ring:
        '''
        Read ring from a LatticeJSON file.

        Args:
            path str: Path to the LatticeJSON file.
        Returns:
            Ring: Ring object.
        '''
        config = latticejson.load(path)
        rootname = config['root']
        elements = cls._make_elements(config)
        return Ring(rootname, [cls._make_lattice(config, name, elements) for name in config['lattices'][rootname]], config['energy']*1.e9, config['info'])

    @classmethod
    def _make_lattice(cls, config: Dict[str, Any], name: str, elems: List[Lattice | Element]) \
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
            return Lattice(name, [cls._make_lattice(config, k, elems) for k in config['lattices'][name]])
        return elems[name]

    @classmethod
    def _make_elements(cls, config: Dict[str, Any]) -> Dict[str, Element]:
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

    def copy(self) -> Ring:
        '''
        Create a copy of the ring.

        Returns:
            Ring: A copy of the ring.
        '''
        elements = [elem.copy() for elem in self.elements]
        return Ring(self.instance.name, elements,
                    self.instance.energy, self.instance.info)
