# base/coordinatearray.py
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
from .basearray import BaseArray
from .coordinate import Coordinate
import numpy as np
import numpy.typing as npt

class CoordinateArray(BaseArray):
    '''
    Base class for phase-space coordinate array.
    '''
    index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3}

    @property
    @abstractmethod
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        pass

    @property
    @abstractmethod
    def x(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal position array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def xp(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal angle array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def y(self) -> npt.NDArray[np.floating]:
        '''
        Vertical position array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def yp(self) -> npt.NDArray[np.floating]:
        '''
        Vertical angle array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def z(self) -> npt.NDArray[np.floating]:
        '''
        Longitudinal displacement array [m] with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def delta(self) -> npt.NDArray[np.floating]:
        '''
        Relative momentum deviation array with shape (N,).
        '''
        pass

    @vector.setter
    @abstractmethod
    def vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Set 4xN array of 4D phase-space vectors [x, x', y, y'].

        Args:
            vector npt.NDArray[np.floating]: 4xN array of phase-space vectors.
        '''
        pass

    @x.setter
    @abstractmethod
    def x(self, x: npt.NDArray[np.floating]) -> None:
        '''
        Set horizontal position array with shape (N,).

        Args:
            x float: Horizontal position array.
        '''
        pass

    @xp.setter
    @abstractmethod
    def xp(self, xp: npt.NDArray[np.floating]) -> None:
        '''
        Set horizontal angle array with shape (N,).

        Args:
            xp float: Horizontal angle array.
        '''
        pass

    @y.setter
    @abstractmethod
    def y(self, y: npt.NDArray[np.floating]) -> None:
        '''
        Set vertical position array with shape (N,).

        Args:
            y float: Vertical position array.
        '''
        pass

    @yp.setter
    @abstractmethod
    def yp(self, yp: npt.NDArray[np.floating]) -> None:
        '''
        Set vertical angle array with shape (N,).

        Args:
            yp float: Vertical angle array.
        '''
        pass

    @z.setter
    @abstractmethod
    def z(self, z: npt.NDArray[np.floating]) -> None:
        '''
        Set longitudinal displacement array [m] with shape (N,).

        Args:
            z float: Longitudinal displacement array.
        '''
        pass

    @delta.setter
    @abstractmethod
    def delta(self, delta: npt.NDArray[np.floating]) -> None:
        '''
        Set relative momentum deviation array with shape (N,).

        Args:
            delta float: Relative momentum deviation array.
        '''
        pass

    @abstractmethod
    def copy(self) -> CoordinateArray:
        '''
        Create a copy.

        Returns:
            CoordinateArray: A copy of the coordinate array object.
        '''
        pass

    @abstractmethod
    def append(self, cood: CoordinateArray) -> None:
        '''
        Append another coordinate array to this one.

        Args:
            cood CoordinateArray: Coordinate array to append.
        '''
        pass

    @abstractmethod
    def from_s(self, s: float) -> Coordinate:
        '''
        Get coordinate at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m]

        Returns:
            Coordinate: Coordinate at the specified position.
        '''
        pass
