# cpp/coordinate.py
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
from ..base.coordinate import Coordinate as CoordinateABC
from egret.cppegret import Coordinate as CoordinateCPP
import numpy as np
import numpy.typing as npt

class Coordinate(CoordinateABC):
    '''
    Class for phase-space coordinates.
    '''

    def __init__(self, vector: npt.NDArray[np.floating] = np.zeros(4),
                 s: float = 0.0, z: float = 0.0, delta: float = 0.0, **kwargs):
        '''
        Args:
            vector npt.NDArray[np.floating]: Phase-space vector [x, x', y, y'].
            s float: Longitudinal position.
            z float: Longitudinal displacement.
            delta float: Relative energy deviation.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = CoordinateCPP(vector, s, z, delta)

    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.

        Returns:
            float: Value of the coordinate corresponding to the key.
        '''
        match key:
            case 'x':
                return self.instance.x
            case 'xp':
                return self.instance.xp
            case 'y':
                return self.instance.y
            case 'yp':
                return self.instance.yp
            case 's':
                return self.instance.s
            case 'z':
                return self.instance.z
            case 'delta':
                return self.instance.delta
            case _:
                raise KeyError(f'Invalid key: {key}')

    def __setitem__(self, key: str, value: float) -> None:
        '''
        Set coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.
            value float: Value to set.
        '''
        match key:
            case 'x':
                self.instance.x = value
            case 'xp':
                self.instance.xp = value
            case 'y':
                self.instance.y = value
            case 'yp':
                self.instance.yp = value
            case 's':
                self.instance.s = value
            case 'z':
                self.instance.z = value
            case 'delta':
                self.instance.delta = value
            case _:
                raise KeyError(f'Invalid key: {key}')

    def copy(self) -> Coordinate:
        '''
        Returns:
            Coordinate: A copy of the coordinate object.
        '''
        return Coordinate(self.instance.vector, self.instance.s,
                          self.instance.z, self.instance.delta)
