# octupole.py
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

from .element import Element
from .coordinate import Coordinate
from .drift import Drift

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Octupole(Element):
    '''
    Octupole magnet.
    '''

    def __init__(self, name: str, length: float, k3: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            k3 float: Octupole strength [1/m^4].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k3 = k3

    def copy(self) -> Octupole:
        '''
        Return a copy of the octupole.

        Returns:
            Octupole: Copy of the octupole.
        '''
        return Octupole(self.name, self.length, self.k3,
                        self.dx, self.dy, self.ds, self.tilt, self.info)

    def transfer_matrix(self, cood0: Coordinate):
        '''
        Transfer matrix of the octupole.

        Args:
            cood0 Coordinate: Initial coordinate

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        # temporarily set to drift
        return Drift.transfer_matrix_from_length(self.length)

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False):
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal position array of shape (N,).
        '''
        # temporarily set to drift
        return Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)
