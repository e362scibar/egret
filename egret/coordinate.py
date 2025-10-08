# coordinate.py
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

import numpy as np

class Coordinate:
    '''
    Phase-space coordinate.
    ''' 
    index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3}

    def __init__(self, x: float = 0., xp: float = 0., y: float = 0., yp: float = 0.,
                 z: float = 0., delta: float = 0., s: float = 0.):
        '''
        Args:
            x float: Horizontal position [m].
            xp float: Horizontal angle [rad].
            y float: Vertical position [m].
            yp float: Vertical angle [rad].
            z float: Longitudinal displacement [m].
            delta float: Relative momentum deviation.
            s float: Longitudinal position along the reference orbit [m].
        '''
        self.vector = np.array([x, xp, y, yp])
        self.z = z
        self.delta = delta
        self.s = s

    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.
        
        Returns:
            float: Value of the coordinate corresponding to the key.
        '''
        try:
            return self.vector[self.index[key]]
        except KeyError:
            match key:
                case 'z':
                    return self.z
                case 'delta':
                    return self.delta
                case 's':
                    return self.s
                case _:
                    raise KeyError(f'Invalid key: {key}')

    def __setitem__(self, key: str, value: float) -> None:
        '''
        Set coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', 'z', 'delta', or 's'.
            value float: Value to set.
        '''
        try:
            self.vector[self.index[key]] = value
        except KeyError:
            match key:
                case 'z':
                    self.z = value
                case 'delta':
                    self.delta = value
                case 's':
                    self.s = value
                case _:
                    raise KeyError(f'Invalid key: {key}')
