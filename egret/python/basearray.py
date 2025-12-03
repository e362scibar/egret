# python/basearray.py
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
from ..base.basearray import BaseArray as BaseArrayABC
import numpy as np
import numpy.typing as npt

class BaseArray(BaseArrayABC):
    '''
    Base class for array types.
    '''

    def __init__(self, s: npt.NDArray[np.floating]) -> None:
        '''
        Initialize the BaseArray.

        Args:
            s npt.NDArray[np.floating]: Array of longitudinal positions.
        '''
        self._s = s.copy()

    @property
    def s(self) -> npt.NDArray[np.floating]:
        '''
        Array of longitudinal positions
        '''
        return self._s

    @s.setter
    def s(self, s: npt.NDArray[np.floating]) -> None:
        '''
        Set the array of longitudinal positions.

        Args:
            s npt.NDArray[np.floating]: Array of longitudinal positions.
        '''
        self._s = s.copy()

    def __len__(self) -> int:
        '''
        Return the length of the array.
        '''
        return len(self._s)

    def copy(self):
        '''
        Return a copy of the array.
        '''
        return BaseArray(self._s.copy())

    def ds(self) -> float:
        '''
        Return the step size of the array.
        '''
        if len(self._s) < 2:
            raise ValueError("Array must have at least two elements to compute step size.")
        return self._s[1] - self._s[0]

    def append(self, other: BaseArray) -> None:
        '''
        Append another item to the array.
        '''
        self._s = np.concatenate((self._s, other._s))

    def index_from_s(self, s: float) -> int:
        '''
        Return the index corresponding to the given s position.

        Args:
            s float: s position.

        Returns:
            int: Index corresponding to the s position.
        '''
        idx = np.searchsorted(self._s, s)
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx == len(self._s) - 1:
            raise ValueError(f'Out of range: s={s}, max={self._s[-1]}')
        return idx
