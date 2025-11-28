# base/object.py
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

class Object(ABC):
    '''
    Base class of any objects.
    '''

    @property
    @abstractmethod
    def name(self) -> str:
        '''
        Get the name of the object.

        Returns:
            str: name of the object
        '''
        pass

    @name.setter
    @abstractmethod
    def name(self, value: str) -> None:
        '''
        Set the name of the object.

        Args:
            value str: name of the object
        '''
        pass
