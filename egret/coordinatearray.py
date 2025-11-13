# coordinatearray.py
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

from .coordinate import Coordinate

import numpy as np
import numpy.typing as npt

class CoordinateArray:
    '''
    Phase-space coordinate array.
    '''
    index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3}

    def __init__(self, vector: npt.NDArray[np.floating], s: npt.NDArray[np.floating],
                 z: npt.NDArray[np.floating] = None, delta: npt.NDArray[np.floating] = None):
        '''
        Args:
            vector npt.NDArray[np.floating]: 4xN 4D phase-space vectors [x, x', y, y'].
            s npt.NDArray[np.floating]: Longitudinal position array along the reference orbit [m] with shape (N,).
            z npt.NDArray[np.floating]: Longitudinal displacement array [m] with shape (N,).
            delta npt.NDArray[np.floating]: Relative momentum deviation array with shape (N,).
        '''
        self.vector = vector.copy()
        self.s = s.copy()
        if z is None:
            self.z = np.zeros_like(s)
        else:
            self.z = z.copy()
        if delta is None:
            self.delta = np.zeros_like(s)
        else:
            self.delta = delta.copy()

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

    def copy(self) -> CoordinateArray:
        '''
        Returns:
            CoordinateArray: A copy of the coordinate array object.
        '''
        return CoordinateArray(self.vector, self.s, self.z, self.delta)

    def append(self, cood: CoordinateArray) -> None:
        '''
        Append another coordinate array to this one.

        Args:
            cood CoordinateArray: Coordinate array to append.
        '''
        self.vector = np.hstack((self.vector, cood.vector))
        self.s = np.hstack((self.s, cood.s))
        self.z = np.hstack((self.z, cood.z))
        self.delta = np.hstack((self.delta, cood.delta))

    def from_s(self, s: float) -> Coordinate:
        '''
        Get coordinate at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m]

        Returns:
            Coordinate: Coordinate at the specified position.
        '''
        idx = np.searchsorted(self.s, s)
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx == len(self.s) - 1:
            raise ValueError(f'Out of range: s={s}, max={self.s[-1]}')
        s0, s1 = self.s[idx], self.s[idx+1]
        ds = s1 - s0
        a = np.array([(s1-s)/ds, (s-s0)/ds]) if ds != 0. else np.array([0.5, 0.5])
        vec = np.sum(self.vector[:,idx:idx+2] * a[np.newaxis, :], axis=1)
        z = np.sum(self.z[idx:idx+2] * a)
        delta = np.sum(self.delta[idx:idx+2] * a)
        return Coordinate(vec, s, z, delta)
