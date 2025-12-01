# cpp/dispersion.py
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
from ..base.dispersion import Dispersion as DispersionABC
from egret.cppegret import Dispersion as DispersionCPP
import numpy as np
import numpy.typing as npt

class Dispersion(DispersionABC):
    '''
    Base class for energy dispersion.
    '''

    def __init__(self, vector: npt.NDArray[np.floating],
                 s: float, **kwargs):
        '''
        Initialize Dispersion object.

        Args:
            vector npt.NDArray[np.floating]: 4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
            s float: Longitudinal position [m].
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = DispersionCPP(vector, s)

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        return self.instance.vector

    @property
    def s(self) -> float:
        '''
        Longitudinal position [m].
        '''
        return self.instance.s

    @vector.setter
    def vector(self, value: npt.NDArray[np.floating]) -> None:
        self.instance.vector = value

    @s.setter
    def s(self, value: float) -> None:
        self.instance.s = value

    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', or 's'.

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
                self.s = value
            case _:
                raise KeyError(f'Invalid key: {key}')

    def copy(self) -> Dispersion:
        '''
        Returns:
            Dispersion: A copy of the coordinate object.
        '''
        return Dispersion(self.instance.vector, self.instance.s)
