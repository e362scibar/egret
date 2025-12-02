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
from .basearray import BaseArray
from .coordinate import Coordinate
import numpy as np
import numpy.typing as npt

class CoordinateArray(CoordinateArrayABC, BaseArray):
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
        super().__init__(None, instance=self.instance)

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        return self.instance.vector_array

    @property
    def x(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal position array [m] with shape (N,).
        '''
        return self.instance.x_array

    @property
    def xp(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal angle array [rad] with shape (N,).
        '''
        return self.instance.xp_array

    @property
    def y(self) -> npt.NDArray[np.floating]:
        '''
        Vertical position array [m] with shape (N,).
        '''
        return self.instance.y_array

    @property
    def yp(self) -> npt.NDArray[np.floating]:
        '''
        Vertical angle array [rad] with shape (N,).
        '''
        return self.instance.yp_array

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
    def vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Set 4xN array of 4D phase-space vectors.

        Args:
            vector npt.NDArray[np.floating]: 4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        self.instance.vector = vector

    @x.setter
    def x(self, x: npt.NDArray[np.floating]) -> None:
        '''
        Set horizontal position array [m] with shape (N,).

        Args:
            x npt.NDArray[np.floating]: Horizontal position array [m] with shape (N,).
        '''
        self.instance.x_array = x

    @xp.setter
    def xp(self, xp: npt.NDArray[np.floating]) -> None:
        '''
        Set horizontal angle array [rad] with shape (N,).

        Args:
            xp npt.NDArray[np.floating]: Horizontal angle array [rad] with shape (N,).
        '''
        self.instance.xp_array = xp

    @y.setter
    def y(self, y: npt.NDArray[np.floating]) -> None:
        '''
        Set vertical position array [m] with shape (N,).

        Args:
            y npt.NDArray[np.floating]: Vertical position array [m] with shape (N,).
        '''
        self.instance.y_array = y

    @yp.setter
    def yp(self, yp: npt.NDArray[np.floating]) -> None:
        '''
        Set vertical angle array [rad] with shape (N,).

        Args:
            yp npt.NDArray[np.floating]: Vertical angle array [rad] with shape (N,).
        '''
        self.instance.yp_array = yp

    @z.setter
    def z(self, z: npt.NDArray[np.floating]) -> None:
        '''
        Set longitudinal displacement array [m] with shape (N,).

        Args:
            z npt.NDArray[np.floating]: Longitudinal displacement array [m] with shape (N,).
        '''
        self.instance.z_array = z

    @delta.setter
    def delta(self, delta: npt.NDArray[np.floating]) -> None:
        '''
        Set relative momentum deviation array with shape (N,).

        Args:
            delta npt.NDArray[np.floating]: Relative momentum deviation array with shape (N,).
        '''
        self.instance.delta_array = delta

    def copy(self) -> CoordinateArray:
        '''
        Returns:
            CoordinateArray: A copy of the coordinate array object.
        '''
        return CoordinateArray(self.instance.vector_array,
                               self.instance.s_array,
                               self.instance.z_array,
                               self.instance.delta_array)

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
