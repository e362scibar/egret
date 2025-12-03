# python/coordinate.py
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
import numpy as np
import numpy.typing as npt

class Coordinate(CoordinateABC):
    '''
    Phase-space coordinates.
    '''

    def __init__(self, vector: npt.NDArray[np.floating] = np.zeros(4),
                 s: float = 0., z: float = 0., delta: float = 0.):
        '''
        Args:
            vector npt.NDArray[np.floating]: 4D phase-space vector [x, x', y, y'].
            s float: Longitudinal position along the reference orbit [m].
            z float: Longitudinal displacement [m].
            delta float: Relative momentum deviation.
        '''
        self._vector = vector.copy()
        self._s = s
        self._z = z
        self._delta = delta

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        Returns:
            npt.NDArray[np.floating]: 4D phase-space vector [x, x', y, y'].
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
    def z(self) -> float:
        '''
        Returns:
            float: Longitudinal displacement [m].
        '''
        return self._z

    @property
    def delta(self) -> float:
        '''
        Returns:
            float: Relative momentum deviation.
        '''
        return self._delta

    @property
    def x(self) -> float:
        '''
        Returns:
            float: Horizontal position [m].
        '''
        return self._vector[0]

    @property
    def xp(self) -> float:
        '''
        Returns:
            float: Horizontal angle [rad].
        '''
        return self._vector[1]

    @property
    def y(self) -> float:
        '''
        Returns:
            float: Vertical position [m].
        '''
        return self._vector[2]

    @property
    def yp(self) -> float:
        '''
        Returns:
            float: Vertical angle [rad].
        '''
        return self.vector[3]

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Set 4D phase-space vector.

        Args:
            vector npt.NDArray[np.floating]: 4D phase-space vector [x, x', y, y'].
        '''
        self._vector = vector.copy()

    @s.setter
    def s(self, s: float) -> None:
        '''
        Set longitudinal position along the reference orbit.

        Args:
            s float: Longitudinal position [m].
        '''
        self._s = s

    @z.setter
    def z(self, z: float) -> None:
        '''
        Set longitudinal displacement.

        Args:
            z float: Longitudinal displacement [m].
        '''
        self._z = z

    @delta.setter
    def delta(self, delta: float) -> None:
        '''
        Set relative momentum deviation.

        Args:
            delta float: Relative momentum deviation.
        '''
        self._delta = delta

    @x.setter
    def x(self, x: float) -> None:
        '''
        Set horizontal position.

        Args:
            x float: Horizontal position [m].
        '''
        self._vector[0] = x

    @xp.setter
    def xp(self, xp: float) -> None:
        '''
        Set horizontal angle.

        Args:
            xp float: Horizontal angle [rad].
        '''
        self._vector[1] = xp

    @y.setter
    def y(self, y: float) -> None:
        '''
        Set vertical position.

        Args:
            y float: Vertical position [m].
        '''
        self._vector[2] = y

    @yp.setter
    def yp(self, yp: float) -> None:
        '''
        Set vertical angle.

        Args:
            yp float: Vertical angle [rad].
        '''
        self._vector[3] = yp

    def copy(self) -> Coordinate:
        '''
        Returns:
            Coordinate: A copy of the coordinate object.
        '''
        return Coordinate(self._vector, self._s, self._z, self._delta)
