# cpp/basearray.py
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

from ..base import BaseArray as BaseArrayABC
from egret.cppegret import BaseArray as BaseArrayCPP
import numpy as np
import numpy.typing as npt

class BaseArray(BaseArrayABC):
    '''
    Base class for C++ array types.
    '''

    def __init__(self, s_array: npt.NDArray[np.floating], **kwargs):
        '''
        Initialize the BaseArray.

        Args:
            s_array (npt.NDArray[np.floating]): Array of s positions.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = BaseArrayCPP(s_array)

    @property
    def s(self) -> npt.NDArray[np.floating]:
        '''
        Array of s positions
        '''
        return self.instance.s_array

    @s.setter
    def s(self, s: npt.NDArray[np.floating]):
        '''
        Set the array of s positions.

        Args:
            s npt.NDArray[np.floating]: Array of s positions.
        '''
        self.instance.s_array = s

    def __len__(self) -> int:
        '''
        Return the length of the array.
        '''
        return len(self.instance)

    def copy(self) -> "BaseArray":
        '''
        Return a copy of the array.
        '''
        return BaseArray(self.s_array)


    def ds(self) -> float:
        '''
        Return the step size of the array.
        '''
        return self.instance.ds()

    def append(self, other: "BaseArray"):
        '''
        Append another item to the array.
        '''
        self.instance.append(other.instance)

    def index_from_s(self, s: float) -> int:
        '''
        Return the index corresponding to the given s position.

        Args:
            s float: s position.

        Returns:
            int: Index corresponding to the s position.
        '''
        return self.instance.index_from_s(s)
