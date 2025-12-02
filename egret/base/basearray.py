# base/basearray.py
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

from abc import ABC, abstractmethod

class BaseArray(ABC):
    '''
    Abstract base class for array types.
    '''

    @property
    @abstractmethod
    def s(self):
        '''
        Array of longitudinal positions
        '''
        pass

    @s.setter
    @abstractmethod
    def s(self, s):
        '''
        Set the array of longitudinal positions.

        Args:
            s float: Array of longitudinal positions.
        '''
        pass

    @abstractmethod
    def __len__(self) -> int:
        '''
        Return the length of the array.
        '''
        pass

    @abstractmethod
    def copy(self):
        '''
        Return a copy of the array.
        '''
        pass


    @abstractmethod
    def ds(self) -> float:
        '''
        Return the step size of the array.
        '''

    @abstractmethod
    def append(self, other):
        '''
        Append another item to the array.
        '''
        pass

    @abstractmethod
    def index_from_s(self, s: float) -> int:
        '''
        Return the index corresponding to the given s position.

        Args:
            s float: s position.

        Returns:
            int: Index corresponding to the s position.
        '''
        pass
