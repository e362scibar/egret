# cpp/drift.py
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
from ..base.drift import Drift as DriftABC
from .element import Element
from egret.cppegret import Drift as DriftCPP
import numpy as np
import numpy.typing as npt
from typing import Tuple

class Drift(DriftABC, Element):
    '''
    Drift space element.
    '''

    def __init__(self, name: str, length: float, dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize a drift space element.

        Args:
            name str: Name of the drift space element.
            length float: Length of the drift space [m].
            dx float: Horizontal offset of the drift space [m].
            dy float: Vertical offset of the drift space [m].
            ds float: Longitudinal offset of the drift space [m].
            tilt float: Tilt angle of the drift space [rad].
            info str: Additional information.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = DriftCPP(name, length, dx, dy, ds, tilt, info)
        super().__init__(None, None, None, instance=self.instance)

    @classmethod
    def transfer_matrix_from_length(cls, length: float) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the drift space.

        Args:
            length float: Length of the drift space [m].

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        return DriftCPP.transfer_matrix_from_length(length)

    @classmethod
    def transfer_matrix_array_from_length(cls, length: float, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the drift space.

        Args:
            length float: Length of the drift space [m].
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        tmat, s = DriftCPP.transfer_matrix_array_from_length(length, ds, endpoint)
        return np.array(tmat), s
