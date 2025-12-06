# cpp/lattice.py
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
from ..base.lattice import Lattice as LatticeABC
from egret.cppegret import Lattice as LatticeCPP
from .element import Element
from typing import List

class Lattice(LatticeABC, Element):
    '''
    Lattice element class.
    '''

    def __init__(self, name: str, elements: List[Element],
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize lattice element.

        Args:
            name str: Element name.
            elements List[Element, ...]: List of elements in the lattice.
            dx float: Horizontal offset of the magnetic center [m].
            dy float: Vertical offset of the magnetic center [m].
            ds float: Longitudinal offset of the magnetic center [m].
            tilt float: Rotation angle around the beam axis [rad].
            info str: Additional information.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            cpp_elements = [elem.instance for elem in elements]
            self.instance = LatticeCPP(name, cpp_elements,
                                       dx, dy, ds, tilt, info)
        super().__init__(None, None, None, instance=self.instance)

    @classmethod
    def length_of(cls, elements: List[Element]) -> float:
        '''
        Calculate total length of the lattice.

        Args:
            elements List[Element, ...]: List of elements in the lattice.

        Returns:
            float: Total length of the lattice [m].
        '''
        return LatticeCPP.length_of([elem.instance for elem in elements])

    @classmethod
    def angle_of(cls, elements: List[Element]) -> float:
        '''
        Calculate total bending angle of the lattice.

        Args:
            elements List[Element, ...]: List of elements in the lattice.

        Returns:
            float: Total bending angle of the lattice [rad].
        '''
        return LatticeCPP.angle_of([elem.instance for elem in elements])

    def copy(self) -> Lattice:
        '''
        Create a copy of the lattice element.

        Returns:
            Lattice: A copy of the lattice element.
        '''
        elements = [elem.copy() for elem in self.elements]
        return Lattice(self.instance.name, elements,
                       self.instance.dx, self.instance.dy, self.instance.ds,
                       self.instance.tilt, self.instance.info)
