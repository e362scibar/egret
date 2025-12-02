# base/coordinate.py
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
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Coordinate(ABC):
    '''
    Base class of phase-space coordinates.
    '''

    @property
    @abstractmethod
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        Phase-space coordinate vector [x, x', y, y'].
        '''
        pass

    @property
    @abstractmethod
    def x(self) -> float:
        '''
        Horizontal position.
        '''
        pass

    @property
    @abstractmethod
    def xp(self) -> float:
        '''
        Horizontal angle.
        '''
        pass

    @property
    @abstractmethod
    def y(self) -> float:
        '''
        Vertical position.
        '''
        pass

    @property
    @abstractmethod
    def yp(self) -> float:
        '''
        Vertical angle.
        '''
        pass

    @property
    @abstractmethod
    def s(self) -> float:
        '''
        Longitudinal position.
        '''
        pass

    @property
    @abstractmethod
    def z(self) -> float:
        '''
        Longitudinal coordinate.
        '''
        pass

    @property
    @abstractmethod
    def delta(self) -> float:
        '''
        Relative momentum deviation.
        '''
        pass

    @vector.setter
    @abstractmethod
    def vector(self, vector: npt.NDArray[np.floating]):
        '''
        Set the phase-space coordinate vector [x, x', y, y'].

        Args:
            vector npt.NDArray[np.floating]: Phase-space coordinate vector.
        '''
        pass

    @x.setter
    @abstractmethod
    def x(self, x: float):
        '''
        Set the horizontal position.

        Args:
            x float: Horizontal position.
        '''
        pass

    @xp.setter
    @abstractmethod
    def xp(self, xp: float):
        '''
        Set the horizontal angle.

        Args:
            xp float: Horizontal angle.
        '''
        pass

    @y.setter
    @abstractmethod
    def y(self, y: float):
        '''
        Set the vertical position.

        Args:
            y float: Vertical position.
        '''
        pass

    @yp.setter
    @abstractmethod
    def yp(self, yp: float):
        '''
        Set the vertical angle.

        Args:
            yp float: Vertical angle.
        '''
        pass

    @s.setter
    @abstractmethod
    def s(self, s: float):
        '''
        Set the longitudinal position.

        Args:
            s float: Longitudinal position.
        '''
        pass

    @z.setter
    @abstractmethod
    def z(self, z: float):
        '''
        Set the longitudinal displacement.

        Args:
            z float: Longitudinal displacement.
        '''
        pass

    @delta.setter
    @abstractmethod
    def delta(self, delta: float):
        '''
        Set the relative momentum deviation.

        Args:
            delta float: Relative momentum deviation.
        '''
        pass

    @abstractmethod
    def copy(self) -> Coordinate:
        '''
        Returns:
            Coordinate: A copy of the coordinate object.
        '''
        pass
