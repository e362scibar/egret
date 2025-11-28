# base/drift.py
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
from .element import Element
import numpy as np
import numpy.typing as npt
from typing import Tuple

class Drift(Element):
    '''
    Base class of a drift space element.
    '''

    @classmethod
    @abstractmethod
    def transfer_matrix_from_length(cls, length: float) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the drift space.

        Args:
            length float: Length of the drift space [m].

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        pass

    @classmethod
    @abstractmethod
    def transfer_matrix_array_from_length(cls, length: float, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the drift space.

        Args:
            length float: Length of the drift space [m].
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        pass
