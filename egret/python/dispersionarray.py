# python/dispersionarray.py
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
from .basearray import BaseArray
from .dispersion import Dispersion
import numpy as np
import numpy.typing as npt

class DispersionArray(DispersionArrayABC, BaseArray):
    '''
    Energy dispersion array.
    '''

    def __init__(self, vector: npt.NDArray[np.floating], s: npt.NDArray[np.floating]):
        '''
        Args:
            vector npt.NDArray[np.floating]: 4xN 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].
            s npt.NDArray[np.floating]: Longitudinal position array along the reference orbit [m] with shape (N,).
        '''
        super().__init__(s)
        self._vector = vector.copy()

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].
        '''
        return self._vector

    @property
    def x(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal dispersion array with shape (N,).
        '''
        return self._vector[0, :]

    @property
    def xp(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal angle dispersion array with shape (N,).
        '''
        return self._vector[1, :]

    @property
    def y(self) -> npt.NDArray[np.floating]:
        '''
        Vertical dispersion array with shape (N,).
        '''
        return self._vector[2, :]

    @property
    def yp(self) -> npt.NDArray[np.floating]:
        '''
        Vertical angle dispersion array with shape (N,).
        '''
        return self._vector[3, :]

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]):
        '''
        Set the 4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].

        Args:
            vector npt.NDArray[np.floating]: 4xN array of dispersion vectors.
        '''
        self._vector = vector.copy()

    @x.setter
    def x(self, x: npt.NDArray[np.floating]):
        '''
        Set the horizontal dispersion array with shape (N,).

        Args:
            x npt.NDArray[np.floating]: Horizontal dispersion array.
        '''
        self._vector[0, :] = x.copy()

    @xp.setter
    def xp(self, xp: npt.NDArray[np.floating]):
        '''
        Set the horizontal angle dispersion array with shape (N,).

        Args:
            xp npt.NDArray[np.floating]: Horizontal angle dispersion array.
        '''
        self._vector[1, :] = xp.copy()

    @y.setter
    def y(self, y: npt.NDArray[np.floating]):
        '''
        Set the vertical dispersion array with shape (N,).

        Args:
            y npt.NDArray[np.floating]: Vertical dispersion array.
        '''
        self._vector[2, :] = y.copy()

    @yp.setter
    def yp(self, yp: npt.NDArray[np.floating]):
        '''
        Set the vertical angle dispersion array with shape (N,).

        Args:
            yp npt.NDArray[np.floating]: Vertical angle dispersion array.
        '''
        self._vector[3, :] = yp.copy()

    def copy(self) -> DispersionArray:
        '''
        Returns:
            DispersionArray: A copy of the dispersion array object.
        '''
        return DispersionArray(self._vector, self._s)

    def append(self, disp: DispersionArray):
        '''
        Append another dispersion array to this one.

        Args:
            disp DispersionArray: Another dispersion array to append.
        '''
        BaseArray.append(self, disp)
        self._vector = np.hstack((self._vector, disp._vector))

    def from_s(self, s: float) -> Dispersion:
        '''
        Get dispersion at the specified longitudinal position by linear interpolation.

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
        return Dispersion(vec, s)
