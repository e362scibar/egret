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

    @property
    def x(self) -> float:
        '''
        Horizontal dispersion [m].
        '''
        return self.instance.x

    @property
    def xp(self) -> float:
        '''
        Horizontal angle dispersion [rad].
        '''
        return self.instance.xp

    @property
    def y(self) -> float:
        '''
        Vertical dispersion [m].
        '''
        return self.instance.y

    @property
    def yp(self) -> float:
        '''
        Vertical angle dispersion [rad].
        '''
        return self.instance.yp

    @vector.setter
    def vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Set the 4D dispersion vector.

        Args:
            vector npt.NDArray[np.floating]: 4D dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        self.instance.vector = vector

    @s.setter
    def s(self, s: float) -> None:
        '''
        Set the longitudinal position [m].

        Args:
            s float: Longitudinal position
        '''
        self.instance.s = s

    @x.setter
    def x(self, x: float) -> None:
        '''
        Set the horizontal dispersion [m].

        Args:
            x float: Horizontal dispersion [m].
        '''
        self.instance.x = x

    @xp.setter
    def xp(self, xp: float) -> None:
        '''
        Set the horizontal angle dispersion [rad].

        Args:
            xp float: Horizontal angle dispersion [rad].
        '''
        self.instance.xp = xp

    @y.setter
    def y(self, y: float) -> None:
        '''
        Set the vertical dispersion [m].

        Args:
            y float: Vertical dispersion [m].
        '''
        self.instance.y = y

    @yp.setter
    def yp(self, yp: float) -> None:
        '''
        Set the vertical angle dispersion [rad].

        Args:
            yp float: Vertical angle dispersion [rad].
        '''
        self.instance.yp = yp

    def copy(self) -> Dispersion:
        '''
        Returns:
            Dispersion: A copy of the coordinate object.
        '''
        return Dispersion(self.instance.vector, self.instance.s)
