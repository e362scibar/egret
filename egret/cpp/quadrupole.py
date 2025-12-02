# base/quadrupole.py
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
from ..base.quadrupole import Quadrupole as QuadrupoleABC
from egret.cppegret import Quadrupole as QuadrupoleCPP
from .element import Element

class Quadrupole(QuadrupoleABC, Element):
    '''
    Quadrupole magnet class.
    '''

    def __init__(self, name: str, length: float, k1: float,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize a quadrupole magnet.

        Args:
            name (str): Name of the quadrupole magnet.
            length (float): Length [m].
            k1 (float): Normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
            dx (float, optional): Horizontal offset [m]. Defaults to 0.0.
            dy (float, optional): Vertical offset [m]. Defaults to 0.0.
            ds (float, optional): Longitudinal offset [m]. Defaults to 0.0.
            tilt (float, optional): Tilt angle [rad]. Defaults to 0.0.
            info (str, optional): Additional information. Defaults to ''.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = QuadrupoleCPP(name, length, k1, dx, dy, ds, tilt, info)
        super().__init__(None, None, None, instance=self.instance)

    @property
    def k1(self) -> float:
        '''
        Normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
        '''
        return self.instance.k1

    @k1.setter
    def k1(self, k1: float) -> None:
        '''
        Set normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)

        Args:
            k1 (float): Normalized gradient [1/m^2].
        '''
        self.instance.k1 = k1
