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
from .basearray import BaseArray
from .dispersion import Dispersion
import numpy as np
import numpy.typing as npt

class DispersionArray(DispersionArrayABC, BaseArray):
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
        super().__init__(None, instance=self.instance)

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].
        '''
        return self.instance.vector

    @property
    def x(self) -> npt.NDArray[np.floating]:
        '''
        Array of horizontal dispersion eta_x [m].
        '''
        return self.instance.x_array

    @property
    def xp(self) -> npt.NDArray[np.floating]:
        '''
        Array of horizontal dispersion angle eta'_x [rad].
        '''
        return self.instance.xp_array

    @property
    def y(self) -> npt.NDArray[np.floating]:
        '''
        Array of vertical dispersion eta_y [m].
        '''
        return self.instance.y_array

    @property
    def yp(self) -> npt.NDArray[np.floating]:
        '''
        Array of vertical dispersion angle eta'_y [rad].
        '''
        return self.instance.yp_array

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]):
        '''
        Set the 4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].

        Args:
            vector npt.NDArray[np.floating]: 4xN array of 4D dispersion vectors.
        '''
        self.instance.vector_array = vector

    @x.setter
    def x(self, x: npt.NDArray[np.floating]):
        '''
        Set the array of horizontal dispersion eta_x [m].

        Args:
            x npt.NDArray[np.floating]: Array of horizontal dispersion.
        '''
        self.instance.x_array = x

    @xp.setter
    def xp(self, xp: npt.NDArray[np.floating]):
        '''
        Set the array of horizontal dispersion angle eta'_x [rad].

        Args:
            xp npt.NDArray[np.floating]: Array of horizontal dispersion angle.
        '''
        self.instance.xp_array = xp

    @y.setter
    def y(self, y: npt.NDArray[np.floating]):
        '''
        Set the array of vertical dispersion eta_y [m].

        Args:
            y npt.NDArray[np.floating]: Array of vertical dispersion.
        '''
        self.instance.y_array = y

    @yp.setter
    def yp(self, yp: npt.NDArray[np.floating]):
        '''
        Set the array of vertical dispersion angle eta'_y [rad].

        Args:
            yp npt.NDArray[np.floating]: Array of vertical dispersion angle.
        '''
        self.instance.yp_array = yp

    def copy(self) -> DispersionArray:
        '''
        Returns:
            DispersionArray: A copy of the dispersion array object.
        '''
        return DispersionArray(self.instance.vector_array, self.instance.s_array)

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
