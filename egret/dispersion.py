# dispersion.py
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

class Dispersion:
    '''
    Energy dispersion.
    '''
    index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3}

    def __init__(self, vector: npt.NDArray[np.floating], s: float = 0.):
        '''
        Args:
            vector npt.NDArray[np.floating]: 4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
            s float: Longitudinal position along the reference orbit [m].
        '''
        self.vector = vector.copy()
        self.s = s

    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', or 's'.

        Returns:
            float: Value of the coordinate corresponding to the key.
        '''
        try:
            return self.vector[self.index[key]]
        except KeyError:
            match key:
                case 's':
                    return self.s
                case _:
                    raise KeyError(f'Invalid key: {key}')

    def __setitem__(self, key: str, value: float) -> None:
        '''
        Set coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', or 's'.
            value float: Value to set.
        '''
        try:
            self.vector[self.index[key]] = value
        except KeyError:
            match key:
                case 's':
                    self.s = value
                case _:
                    raise KeyError(f'Invalid key: {key}')

    def copy(self) -> Dispersion:
        '''
        Returns:
            Dispersion: A copy of the coordinate object.
        '''
        return Dispersion(self.vector, self.s)
