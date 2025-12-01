# cpp/coordinatearray.py
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
from ..base.coordinatearray import CoordinateArray as CoordinateArrayABC
from egret.cppegret import CoordinateArray as CoordinateArrayCPP
from .coordinate import Coordinate
import numpy as np
import numpy.typing as npt

class CoordinateArray(CoordinateArrayABC):
    '''
    Class for phase-space coordinate array.
    '''

    def __init__(self, vector: npt.NDArray[np.floating],
                 s: npt.NDArray[np.floating],
                 z: npt.NDArray[np.floating] = np.array([]),
                 delta: npt.NDArray[np.floating] = np.array([]), **kwargs):
        '''
        4xN array of 4D phase-space vectors [x, x', y, y'].

        Args:
            vector npt.NDArray[np.floating]: 4xN array of 4D phase-space vectors [x, x', y, y'].
            s npt.NDArray[np.floating]: Longitudinal position array [m] with shape (N,).
            z npt.NDArray[np.floating]: Longitudinal displacement array [m] with shape (N,).
            delta npt.NDArray[np.floating]: Relative momentum deviation array with shape (N,).
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = CoordinateArrayCPP(vector, s, z, delta)

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        return self.instance.vector

    @property
    def s(self) -> npt.NDArray[np.floating]:
        '''
        Longitudinal position array [m] with shape (N,).
        '''
        return self.instance.s

    @property
    def z(self) -> npt.NDArray[np.floating]:
        '''
        Longitudinal displacement array [m] with shape (N,).
        '''
        return self.instance.z

    @property
    def delta(self) -> npt.NDArray[np.floating]:
        '''
        Relative momentum deviation array with shape (N,).
        '''
        return self.instance.delta

    @vector.setter
    def vector(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        self.instance.vector = value

    @s.setter
    def s(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set longitudinal position array [m] with shape (N,).
        '''
        self.instance.s = value

    @z.setter
    def z(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set longitudinal displacement array [m] with shape (N,).
        '''
        self.instance.z = value

    @delta.setter
    def delta(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set relative momentum deviation array with shape (N,).
        '''
        pass

    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.

        Returns:
            NDArray: Value of the coordinate corresponding to the key.
        '''
        match key:
            case 'x':
                return self.instance.x_array
            case 'xp':
                return self.instance.xp_array
            case 'y':
                return self.instance.y_array
            case 'yp':
                return self.instance.yp_array
            case 's':
                return self.instance.s_array
            case 'z':
                return self.instance.z_array
            case 'delta':
                return self.instance.delta_array
            case _:
                raise KeyError(f'Invalid key: {key}')

    def __setitem__(self, key: str, value: float) -> None:
        '''
        Set coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.
            value NDArray: Value to set.
        '''
        match key:
            case 'x':
                self.instance.x_array = value
            case 'xp':
                self.instance.xp_array = value
            case 'y':
                self.instance.y_array = value
            case 'yp':
                self.instance.yp_array = value
            case 's':
                self.instance.s_array = value
            case 'z':
                self.instance.z_array = value
            case 'delta':
                self.instance.delta_array = value
            case _:
                raise KeyError(f'Invalid key: {key}')

    def copy(self) -> CoordinateArray:
        '''
        Returns:
            CoordinateArray: A copy of the coordinate array object.
        '''
        return CoordinateArray(self.instance.vector,
                               self.instance.s,
                               self.instance.z,
                               self.instance.delta)

    def append(self, cood: CoordinateArray) -> None:
        '''
        Append another coordinate array to this one.

        Args:
            cood CoordinateArray: Coordinate array to append.
        '''
        self.instance.append(cood.instance)

    def from_s(self, s: float) -> Coordinate:
        '''
        Get coordinate at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m]

        Returns:
            Coordinate: Coordinate at the specified position.
        '''
        coord_cpp = self.instance.from_s(s)
        return Coordinate(None, None, instance=coord_cpp)
