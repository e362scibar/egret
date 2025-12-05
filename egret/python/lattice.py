# python/lattice.py
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
from .element import Element
import copy
from typing import List

class Lattice(LatticeABC, Element):
    '''
    Lattice element class.
    '''

    def __init__(self, name: str, elements: List[Element],
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the lattice.
            elements list of Element: List of elements in the lattice.
            dx float: Horizontal offset of the lattice [m].
            dy float: Vertical offset of the lattice [m].
            ds float: Longitudinal offset of the lattice [m].
            tilt float: Tilt angle of the lattice [rad].
            info str: Additional information.
        '''
        length = self.length_of(elements)
        angle = self.angle_of(elements)
        super().__init__(name, length, angle, dx, dy, ds, tilt, info)
        self._elements = copy.deepcopy(elements)

    @classmethod
    def length_of(cls, elements: List[Element]) -> float:
        '''
        Calculate total length of the lattice.

        Args:
            elements List[Element, ...]: List of elements in the lattice.

        Returns:
            float: Total length of the lattice [m].
        '''
        return sum(elem.length for elem in elements)

    @classmethod
    def angle_of(cls, elements: List[Element]) -> float:
        '''
        Calculate total bending angle of the lattice.

        Args:
            elements List[Element, ...]: List of elements in the lattice.

        Returns:
            float: Total bending angle of the lattice [rad].
        '''
        return sum(elem.angle for elem in elements)

    def copy(self) -> Lattice:
        '''
        Return a copy of the lattice.

        Returns:
            Lattice: Copy of the lattice.
        '''
        return Lattice(self.name, self.elements,
                       self.dx, self.dy, self.ds,
                       self.tilt, self.info)
