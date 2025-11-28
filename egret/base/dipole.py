# base/dipole.py
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
from abc import abstractmethod
from .element import Element
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray
import numpy as np
import numpy.typing as npt
from typing import Tuple

class Dipole(Element):
    '''
    Dipole magnet.
    '''

    @property
    @abstractmethod
    def angle(self) -> float:
        '''
        Bending angle of the dipole [rad].
        '''
        pass

    @property
    @abstractmethod
    def radius(self) -> float:
        '''
        Bending radius of the dipole [m].
        '''
        pass

    @property
    @abstractmethod
    def k1(self) -> float:
        '''
        Quadrupole component [1/m^2].
        '''
        pass

    @property
    @abstractmethod
    def e1(self) -> float:
        '''
        Entrance edge angle [rad].
        '''
        pass

    @property
    @abstractmethod
    def e2(self) -> float:
        '''
        Exit edge angle [rad].
        '''
        pass

    @property
    @abstractmethod
    def h1(self) -> float:
        '''
        Entrance pole-face curvature [1/m].
        '''
        pass

    @property
    @abstractmethod
    def h2(self) -> float:
        '''
        Exit pole-face curvature [1/m].
        '''
        pass

    @angle.setter
    @abstractmethod
    def angle(self, value: float) -> None:
        '''
        Set bending angle of the dipole [rad].
        '''
        pass

    @radius.setter
    @abstractmethod
    def radius(self, value: float) -> None:
        '''
        Set bending radius of the dipole [m].
        '''
        pass

    @k1.setter
    @abstractmethod
    def k1(self, value: float) -> None:
        '''
        Set quadrupole component [1/m^2].
        '''
        pass

    @e1.setter
    @abstractmethod
    def e1(self, value: float) -> None:
        '''
        Set entrance edge angle [rad].
        '''
        pass

    @e2.setter
    @abstractmethod
    def e2(self, value: float) -> None:
        '''
        Set exit edge angle [rad].
        '''
        pass

    @h1.setter
    @abstractmethod
    def h1(self, value: float) -> None:
        '''
        Set entrance pole-face curvature [1/m].
        '''
        pass

    @h2.setter
    @abstractmethod
    def h2(self, value: float) -> None:
        '''
        Set exit pole-face curvature [1/m].
        '''
        pass

    @abstractmethod
    def copy(self) -> Dipole:
        '''
        Return a copy of the dipole element.

        Returns:
            Dipole: Copied dipole element.
        '''
        pass
