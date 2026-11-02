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

def read_only_view(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    '''
    Return a read-only view of an array that shares its memory.

    Args:
        arr npt.NDArray[np.floating]: Array to expose without allowing writes.

    Returns:
        npt.NDArray[np.floating]: Read-only view of arr.
    '''
    view = arr.view()
    view.flags.writeable = False
    return view


class BaseArray(BaseArrayABC):
    '''
    Base class for array types.
    '''
    TOL = 1.0e-6

    def __init__(self, s: npt.NDArray[np.floating]) -> None:
        '''
        Initialize the BaseArray.

        Args:
            s npt.NDArray[np.floating]: Array of longitudinal positions.

        Raises:
            ValueError: If s is not non-decreasing.
        '''
        self._check_non_decreasing(s)
        self._s = s.copy()

    @property
    def s(self) -> npt.NDArray[np.floating]:
        '''
        Array of longitudinal positions (read-only view).
        '''
        return read_only_view(self._s)

    @s.setter
    def s(self, s: npt.NDArray[np.floating]) -> None:
        '''
        Set the array of longitudinal positions. The length must not change.

        Args:
            s npt.NDArray[np.floating]: Array of longitudinal positions.

        Raises:
            ValueError: If the length changes or s is not non-decreasing.
        '''
        if len(s) != len(self._s):
            raise ValueError(f"s must keep its current length {len(self._s)}; got {len(s)}.")
        self._check_non_decreasing(s)
        self._s = s.copy()

    @staticmethod
    def _check_non_decreasing(s: npt.NDArray[np.floating]) -> None:
        '''
        Check that longitudinal positions are non-decreasing.

        Args:
            s npt.NDArray[np.floating]: Array of longitudinal positions.

        Raises:
            ValueError: If s is not non-decreasing.
        '''
        if np.any(np.diff(s) < 0.0):
            raise ValueError("s must be non-decreasing.")

    def _check_length(self, n: int, name: str) -> None:
        '''
        Check that a per-element array has the same length as s.

        Args:
            n int: Length of the array to check.
            name str: Name of the array used in the error message.

        Raises:
            ValueError: If n does not match the length of s.
        '''
        if n != len(self._s):
            raise ValueError(f"{name} must have {len(self._s)} elements to match s; got {n}.")

    def _check_vector(self, vector: npt.NDArray[np.floating]) -> None:
        '''
        Check that a 4xN vector array has one column per element of s.

        Args:
            vector npt.NDArray[np.floating]: 4xN array.

        Raises:
            ValueError: If vector is not 2-D or its number of columns does not match s.
        '''
        if vector.ndim != 2:
            raise ValueError(f"vector must be 2-D; got shape {vector.shape}.")
        self._check_length(vector.shape[1], "vector")

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
        if s < self._s[0] - self.TOL or s > self._s[-1] + self.TOL:
            raise ValueError(f'Out of range: s={s}, range=({self._s[0]}, {self._s[-1]})')
        #idx = np.searchsorted(self._s, s)
        idx = np.searchsorted(self._s, s, side='right')
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx <= 0:
            idx = 1
        if idx >= len(self._s):
            idx = len(self._s) - 1
        return idx - 1
