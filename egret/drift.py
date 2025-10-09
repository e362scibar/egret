# drift.py
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

from .element import Element
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Drift(Element):
    '''
    Drift space element.
    '''

    def __init__(self, name: str, length: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)

    def transfer_matrix(self, cood0: Coordinate = None) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the drift class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        tmat = np.eye(4)
        tmat[0, 1] = self.length
        tmat[2, 3] = self.length
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the drift class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
        tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
        tmat[:, 0, 1] = s
        tmat[:, 2, 3] = s
        return tmat, s

    def coordinate_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
        -> CoordinateArray:
        tmat, s = self.transfer_matrix_array(cood0, ds, endpoint)
        cood = np.matmul(tmat, cood0.vector)
        return CoordinateArray(cood[:, 0], cood[:, 1], cood[:, 2], cood[:, 3],
                               s + cood0['s'],
                               np.full_like(s, cood0['z']), np.full_like(s, cood0['delta']))

    def tmatarray(self, ds: float = 0.01, endpoint: bool = False) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix array along the drift space.

        Args:
            ds float: Step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            tmat npt.NDArray[np.floating]: Transfer matrix array.
            s npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
        tmat = np.repeat(self.tmat[np.newaxis, :, :], len(s), axis=0)
        tmat[:, 0, 1] = s
        tmat[:, 2, 3] = s
        return tmat, s
