# python/coordinatearray.py
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
from .basearray import BaseArray
from .coordinate import Coordinate
import numpy as np
import numpy.typing as npt

class CoordinateArray(CoordinateArrayABC, BaseArray):
    '''
    Phase-space coordinate array class.
    '''

    def __init__(self, vector: npt.NDArray[np.floating], s: npt.NDArray[np.floating],
                 z: npt.NDArray[np.floating] = None, delta: npt.NDArray[np.floating] = None):
        '''
        Args:
            vector npt.NDArray[np.floating]: 4xN 4D phase-space vectors [x, x', y, y'].
            s npt.NDArray[np.floating]: Longitudinal position array along the reference orbit [m] with shape (N,).
            z npt.NDArray[np.floating]: Longitudinal displacement array [m] with shape (N,).
            delta npt.NDArray[np.floating]: Relative momentum deviation array with shape (N,).
        '''
        super().__init__(s)
        self._vector = vector.copy()
        if z is None:
            self._z = np.zeros_like(s)
        else:
            self._z = z.copy()
        if delta is None:
            self._delta = np.zeros_like(s)
        else:
            self._delta = delta.copy()

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D phase-space vectors [x, x', y, y'].
        '''
        return self._vector

    @property
    def z(self) -> npt.NDArray[np.floating]:
        '''
        Longitudinal displacement array [m] with shape (N,).
        '''
        return self._z

    @property
    def delta(self) -> npt.NDArray[np.floating]:
        '''
        Relative momentum deviation array with shape (N,).
        '''
        return self._delta

    @property
    def x(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal position array with shape (N,).
        '''
        return self._vector[0, :]

    @property
    def xp(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal angle array with shape (N,).
        '''
        return self._vector[1, :]

    @property
    def y(self) -> npt.NDArray[np.floating]:
        '''
        Vertical position array with shape (N,).
        '''
        return self._vector[2, :]

    @property
    def yp(self) -> npt.NDArray[np.floating]:
        '''
        Vertical angle array with shape (N,).
        '''
        return self._vector[3, :]

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Set the 4xN array of 4D phase-space vectors [x, x', y, y'].

        Args:
            vector npt.NDArray[np.floating]: New 4xN array of 4D phase-space vectors.
        '''
        self._vector = vector.copy()

    @z.setter
    def z(self, z: npt.NDArray[np.floating]) -> None:
        '''
        Set the longitudinal displacement array [m] with shape (N,).

        Args:
            z npt.NDArray[np.floating]: New longitudinal displacement array.
        '''
        self._z = z.copy()

    @delta.setter
    def delta(self, delta: npt.NDArray[np.floating]) -> None:
        '''
        Set the relative momentum deviation array with shape (N,).

        Args:
            delta npt.NDArray[np.floating]: New relative momentum deviation array.
        '''
        self._delta = delta.copy()

    @x.setter
    def x(self, x: npt.NDArray[np.floating]) -> None:
        '''
        Set the horizontal position array with shape (N,).

        Args:
            x npt.NDArray[np.floating]: New horizontal position array.
        '''
        self._vector[0, :] = x

    @xp.setter
    def xp(self, xp: npt.NDArray[np.floating]) -> None:
        '''
        Set the horizontal angle array with shape (N,).

        Args:
            xp npt.NDArray[np.floating]: New horizontal angle array.
        '''
        self._vector[1, :] = xp

    @y.setter
    def y(self, y: npt.NDArray[np.floating]) -> None:
        '''
        Set the vertical position array with shape (N,).

        Args:
            y npt.NDArray[np.floating]: New vertical position array.
        '''
        self._vector[2, :] = y

    @yp.setter
    def yp(self, yp: npt.NDArray[np.floating]) -> None:
        '''
        Set the vertical angle array with shape (N,).

        Args:
            yp npt.NDArray[np.floating]: New vertical angle array.
        '''
        self._vector[3, :] = yp

    def copy(self) -> CoordinateArray:
        '''
        Returns:
            CoordinateArray: A copy of the coordinate array object.
        '''
        return CoordinateArray(self._vector, self._s, self._z, self._delta)

    def append(self, cood: CoordinateArray) -> None:
        '''
        Append another coordinate array to this one.

        Args:
            cood CoordinateArray: Coordinate array to append.
        '''
        super().append(cood)
        self._vector = np.hstack((self._vector, cood._vector))
        self._z = np.hstack((self._z, cood._z))
        self._delta = np.hstack((self._delta, cood._delta))

    def from_s(self, s: float) -> Coordinate:
        '''
        Get coordinate at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m]

        Returns:
            Coordinate: Coordinate at the specified position.
        '''
        idx = self.index_from_s(s)
        s0, s1 = self._s[idx], self._s[idx+1]
        ds = s1 - s0
        a = np.array([(s1-s)/ds, (s-s0)/ds]) if ds != 0. else np.array([0.5, 0.5])
        vec = np.sum(self._vector[:,idx:idx+2] * a[np.newaxis, :], axis=1)
        z = np.sum(self._z[idx:idx+2] * a)
        delta = np.sum(self._delta[idx:idx+2] * a)
        return Coordinate(vec, s, z, delta)
