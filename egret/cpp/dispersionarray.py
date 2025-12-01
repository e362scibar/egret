# cpp/dispersionarray.py
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
from ..base.dispersionarray import DispersionArray as DispersionArrayABC
from egret.cppegret import DispersionArray as DispersionArrayCPP
from .dispersion import Dispersion
import numpy as np
import numpy.typing as npt

class DispersionArray(DispersionArrayABC):
    '''
    Class for energy dispersion array.
    '''

    def __init__(self, vector: npt.NDArray[np.floating],
                 s: npt.NDArray[np.floating], **kwargs):
        '''
        Initialize DispersionArray object.

        Args:
            vector npt.NDArray[np.floating]: 4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].
            s npt.NDArray[np.floating]: 1D array of longitudinal positions [m].
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = DispersionArrayCPP(vector, s)

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].
        '''
        return self.instance.vector

    @vector.setter
    def vector(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].

        Args:
            value NDArray: 4xN array of dispersion vectors.
        '''
        self.instance.vector = value

    def __getitem__(self, key: str) -> float:
        '''
        Get coordinate value by key.

        Args:
            key str: Key of the coordinate. 'x', 'xp', 'y', 'yp', or 's'.

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
                return self.s
            case _:
                raise KeyError(f'Invalid key: {key}')

    def copy(self) -> DispersionArray:
        '''
        Returns:
            DispersionArray: A copy of the dispersion array object.
        '''
        return DispersionArray(self.instance.vector, self.instance.s)

    def append(self, disp: DispersionArray):
        '''
        Append another dispersion array to this one.

        Args:
            disp DispersionArray: Another dispersion array to append.
        '''
        self.instance.append(disp.instance)

    def from_s(self, s: float) -> Dispersion:
        '''
        Get dispersion at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m]

        Returns:
            Coordinate: Coordinate at the specified position.
        '''
        instance = self.instance.from_s(s)
        return Dispersion(instance=instance)
