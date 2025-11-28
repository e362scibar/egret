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
from abc import ABC, abstractmethod
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
    def vector(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        pass

    @z.setter
    @abstractmethod
    def z(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set longitudinal displacement array [m] with shape (N,).
        '''
        pass

    @delta.setter
    @abstractmethod
    def delta(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set relative momentum deviation array with shape (N,).
        '''
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.

        Returns:
            NDArray: Value of the coordinate corresponding to the key.
        '''
        try:
            return self.vector[self.index[key]]
        except KeyError:
            match key:
                case 's':
                    return self.s
                case 'z':
                    return self.z
                case 'delta':
                    return self.delta
                case _:
                    raise KeyError(f'Invalid key: {key}')

    @abstractmethod
    def __setitem__(self, key: str, value: float) -> None:
        '''
        Set coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.
            value NDArray: Value to set.
        '''
        try:
            self.vector[self.index[key]] = value
        except KeyError:
            match key:
                case 's':
                    self.s = value
                case 'z':
                    self.z = value
                case 'delta':
                    self.delta = value
                case _:
                    raise KeyError(f'Invalid key: {key}')

    @abstractmethod
    def copy(self) -> CoordinateArray:
        '''
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
