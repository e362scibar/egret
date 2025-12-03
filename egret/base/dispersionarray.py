# base/dispersionarray.py
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
from abc import abstractmethod
from .basearray import BaseArray
from .dispersion import Dispersion
import numpy as np
import numpy.typing as npt

class DispersionArray(BaseArray):
    '''
    Base class for energy dispersion array.
    '''
    index = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3}

    @property
    @abstractmethod
    def vector(self) -> npt.NDArray[np.floating]:
        '''
        4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].
        '''
        pass

    @property
    @abstractmethod
    def x(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal dispersion array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def xp(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal angle dispersion array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def y(self) -> npt.NDArray[np.floating]:
        '''
        Vertical dispersion array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def yp(self) -> npt.NDArray[np.floating]:
        '''
        Vertical angle dispersion array with shape (N,).
        '''
        pass

    @vector.setter
    @abstractmethod
    def vector(self, vector: npt.NDArray[np.floating]):
        '''
        Set the 4xN array of 4D dispersion vectors [eta_x, eta'_x, eta_y, eta'_y].

        Args:
            vector npt.NDArray[np.floating]: 4xN array of dispersion vectors.
        '''
        pass

    @x.setter
    @abstractmethod
    def x(self, x: npt.NDArray[np.floating]):
        '''
        Set the horizontal dispersion array with shape (N,).

        Args:
            x npt.NDArray[np.floating]: Horizontal dispersion array.
        '''
        pass

    @xp.setter
    @abstractmethod
    def xp(self, xp: npt.NDArray[np.floating]):
        '''
        Set the horizontal angle dispersion array with shape (N,).

        Args:
            xp npt.NDArray[np.floating]: Horizontal angle dispersion array.
        '''
        pass

    @y.setter
    @abstractmethod
    def y(self, value: npt.NDArray[np.floating]):
        '''
        Set the vertical dispersion array with shape (N,).

        Args:
            y npt.NDArray[np.floating]: Vertical dispersion array.
        '''
        pass

    @yp.setter
    @abstractmethod
    def yp(self, value: npt.NDArray[np.floating]):
        '''
        Set the vertical angle dispersion array with shape (N,).

        Args:
            yp npt.NDArray[np.floating]: Vertical angle dispersion array.
        '''
        pass

    @abstractmethod
    def copy(self) -> DispersionArray:
        '''
        Create a copy

        Returns:
            DispersionArray: A copy of the dispersion array object.
        '''
        pass

    @abstractmethod
    def append(self, disp: DispersionArray):
        '''
        Append another dispersion array to this one.

        Args:
            disp DispersionArray: Another dispersion array to append.
        '''
        pass

    @abstractmethod
    def from_s(self, s: float) -> Dispersion:
        '''
        Get dispersion at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m]

        Returns:
            Coordinate: Coordinate at the specified position.
        '''
        pass
