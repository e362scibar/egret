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

    @property
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        Phase-space vector [x, x', y, y'].
        '''
        return self.instance.vector

    @property
    def x(self) -> float:
        '''
        Horizontal position.
        '''
        return self.instance.x

    @property
    def xp(self) -> float:
        '''
        Horizontal angle.
        '''
        return self.instance.xp

    @property
    def y(self) -> float:
        '''
        Vertical position.
        '''
        return self.instance.y

    @property
    def yp(self) -> float:
        '''
        Vertical angle.
        '''
        return self.instance.yp

    @property
    def s(self) -> float:
        '''
        Longitudinal position.
        '''
        return self.instance.s

    @property
    def z(self) -> float:
        '''
        Longitudinal displacement.
        '''
        return self.instance.z

    @property
    def delta(self) -> float:
        '''
        Relative energy deviation.
        '''
        return self.instance.delta

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]):
        '''
        Set the phase-space vector.

        Args:
            vector npt.NDArray[np.floating]: Phase-space vector. [x, x', y, y']
        '''
        self.instance.vector = vector

    @x.setter
    def x(self, x: float):
        '''
        Set the horizontal position.

        Args:
            x float: Horizontal position.
        '''
        self.instance.x = x

    @xp.setter
    def xp(self, xp: float):
        '''
        Set the horizontal angle.

        Args:
            xp float: Horizontal angle.
        '''
        self.instance.xp = xp

    @y.setter
    def y(self, y: float):
        '''
        Set the vertical position.

        Args:
            y float: Vertical position.
        '''
        self.instance.y = y

    @yp.setter
    def yp(self, yp: float):
        '''
        Set the vertical angle.

        Args:
            yp float: Vertical angle.
        '''
        self.instance.yp = yp

    @s.setter
    def s(self, s: float):
        '''
        Set the longitudinal position.

        Args:
            s float: Longitudinal position.
        '''
        self.instance.s = s

    @z.setter
    def z(self, z: float):
        '''
        Set the longitudinal displacement.

        Args:
            z float: Longitudinal displacement.
        '''
        self.instance.z = z

    @delta.setter
    def delta(self, delta: float):
        '''
        Set the relative energy deviation.

        Args:
            delta float: Relative energy deviation.
        '''
        self.instance.delta = delta

    def copy(self) -> Coordinate:
        '''
        Returns:
            Coordinate: A copy of the coordinate object.
        '''
        return Coordinate(self.instance.vector, self.instance.s,
                          self.instance.z, self.instance.delta)
