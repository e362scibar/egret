# cpp/dipole.py
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
from ..base.dipole import Dipole as DipoleABC
from .element import Element
from egret.cppegret import Dipole as DipoleCPP
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray
import numpy as np
import numpy.typing as npt
from typing import Tuple

class Dipole(DipoleABC, Element):
    '''
    Dipole magnet class.
    '''

    def __init__(self, name: str, length: float, angle: float, k1: float = 0.0,
                 e1: float = 0.0, e2: float = 0.0,
                 h1: float = 0.0, h2: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize dipole magnet.

        Args:
            name (str): Name of the dipole magnet.
            length (float): Length of the dipole magnet [m].
            angle (float): Bending angle of the dipole [rad].
            k1 (float, optional): Quadrupole component [1/m^2]. Defaults to 0.0.
            e1 (float, optional): Entrance edge angle [rad]. Defaults to 0.0.
            e2 (float, optional): Exit edge angle [rad]. Defaults to 0.0.
            h1 (float, optional): Entrance pole-face curvature [1/m]. Defaults to 0.0.
            h2 (float, optional): Exit pole-face curvature [1/m]. Defaults to 0.0.
            dx (float, optional): Horizontal offset [m]. Defaults to 0.0.
            dy (float, optional): Vertical offset [m]. Defaults to 0.0.
            ds (float, optional): Longitudinal offset [m]. Defaults to 0.0.
            tilt (float, optional): Tilt angle [rad]. Defaults to 0.0.
            info (str, optional): Additional information. Defaults to ''.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = DipoleCPP(name, length, angle, k1, e1, e2,
                                      h1, h2, dx, dy, ds, tilt, info)
        super().__init__(None, None, None, instance=self.instance)

    @property
    def rho(self) -> float:
        '''
        Bending radius of the dipole [m].
        '''
        return self.instance.rho

    @property
    def k1(self) -> float:
        '''
        Quadrupole component [1/m^2].
        '''
        return self.instance.k1

    @property
    def e1(self) -> float:
        '''
        Entrance edge angle [rad].
        '''
        return self.instance.e1

    @property
    def e2(self) -> float:
        '''
        Exit edge angle [rad].
        '''
        return self.instance.e2

    @property
    def h1(self) -> float:
        '''
        Entrance pole-face curvature [1/m].
        '''
        return self.instance.h1

    @property
    def h2(self) -> float:
        '''
        Exit pole-face curvature [1/m].
        '''
        return self.instance.h2

    @k1.setter
    def k1(self, k1: float) -> None:
        '''
        Set quadrupole component [1/m^2].

        Args:
            k1 (float): Quadrupole component [1/m^2].
        '''
        self.instance.k1 = k1

    @e1.setter
    def e1(self, e1: float) -> None:
        '''
        Set entrance edge angle [rad].

        Args:
            e1 (float): Entrance edge angle [rad].
        '''
        self.instance.e1 = e1

    @e2.setter
    def e2(self, e2: float) -> None:
        '''
        Set exit edge angle [rad].

        Args:
            e2 (float): Exit edge angle [rad].
        '''
        self.instance.e2 = e2

    @h1.setter
    def h1(self, h1: float) -> None:
        '''
        Set entrance pole-face curvature [1/m].

        Args:
            h1 (float): Entrance pole-face curvature [1/m].
        '''
        self.instance.h1 = h1

    @h2.setter
    def h2(self, h2: float) -> None:
        '''
        Set exit pole-face curvature [1/m].

        Args:
            h2 (float): Exit pole-face curvature [1/m].
        '''
        self.instance.h2 = h2

    def copy(self) -> Dipole:
        '''
        Create a copy of the dipole element.

        Returns:
            Dipole: Copied dipole element.
        '''
        return Dipole(self.instance.name, self.instance.length, self.instance.angle,
                      self.instance.k1, self.instance.e1, self.instance.e2,
                      self.instance.h1, self.instance.h2,
                      self.instance.dx, self.instance.dy, self.instance.ds,
                      self.instance.tilt, self.instance.info)
