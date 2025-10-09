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

import numpy as np
import numpy.typing as npt

class CoordinateArray:
    '''
    Phase-space coordinate array.
    ''' 
    index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3}

    def __init__(self, x: npt.NDArray[np.floating], xp: npt.NDArray[np.floating],
                 y: npt.NDArray[np.floating], yp: npt.NDArray[np.floating],
                 s: npt.NDArray[np.floating],
                 z: npt.NDArray[np.floating] = None, delta: npt.NDArray[np.floating] = None):
        '''
        Args:
            x NDArray: Horizontal position [m].
            xp NDArray: Horizontal angle [rad].
            y NDArray: Vertical position [m].
            yp NDArray: Vertical angle [rad].
            s NDArray: Longitudinal position along the reference orbit [m].
            z NDArray: Longitudinal displacement [m].
            delta NDArray: Relative momentum deviation.
        '''
        self.vector = np.array([x, xp, y, yp])
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
        return CoordinateArray(self.vector[0], self.vector[1], self.vector[2], self.vector[3],
                               self.s, self.z, self.delta)
