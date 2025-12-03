# python/dispersion.py
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
import numpy as np
import numpy.typing as npt

class Dispersion(DispersionABC):
    '''
    Energy dispersion class.
    '''

    def __init__(self, vector: npt.NDArray[np.floating] = np.zeros(4) , s: float = 0.):
        '''
        Args:
            vector npt.NDArray[np.floating]: 4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
            s float: Longitudinal position along the reference orbit [m].
        '''
        self._vector = vector.copy()
        self._s = s

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        Returns:
            npt.NDArray[np.floating]: 4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        return self._vector

    @property
    def s(self) -> float:
        '''
        Returns:
            float: Longitudinal position along the reference orbit [m].
        '''
        return self._s

    @property
    def x(self) -> float:
        '''
        Returns:
            float: Horizontal dispersion [m].
        '''
        return self._vector[0]

    @property
    def xp(self) -> float:
        '''
        Returns:
            float: Horizontal angle dispersion [rad].
        '''
        return self._vector[1]

    @property
    def y(self) -> float:
        '''
        Returns:
            float: Vertical dispersion [m].
        '''
        return self._vector[2]

    @property
    def yp(self) -> float:
        '''
        Returns:
            float: Vertical angle dispersion [rad].
        '''
        return self._vector[3]

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Set the 4D dispersion vector.

        Args:
            vector npt.NDArray[np.floating]: 4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        self._vector = vector.copy()

    @s.setter
    def s(self, s: float) -> None:
        '''
        Set the longitudinal position along the reference orbit.

        Args:
            s float: Longitudinal position [m].
        '''
        self._s = s

    @x.setter
    def x(self, x: float) -> None:
        '''
        Set horizontal dispersion.

        Args:
            x float: Horizontal dispersion [m].
        '''
        self._vector[0] = x

    @xp.setter
    def xp(self, xp: float) -> None:
        '''
        Set horizontal angle dispersion.

        Args:
            xp float: Horizontal angle dispersion [rad].
        '''
        self._vector[1] = xp

    @y.setter
    def y(self, y: float) -> None:
        '''
        Set vertical dispersion.

        Args:
            y float: Vertical dispersion [m].
        '''
        self._vector[2] = y

    @yp.setter
    def yp(self, yp: float) -> None:
        '''
        Set vertical angle dispersion.

        Args:
            yp float: Vertical angle dispersion [rad].
        '''
        self._vector[3] = yp

    def copy(self) -> Dispersion:
        '''
        Returns:
            Dispersion: A copy of the coordinate object.
        '''
        return Dispersion(self._vector, self._s)
