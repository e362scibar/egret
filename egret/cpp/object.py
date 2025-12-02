# cpp/object.py
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

from ..base.object import Object as ObjectABC
from egret.cppegret import Object as ObjectCPP

class Object(ObjectABC):
    '''
    Base class of any objects.
    '''

    def __init__(self, name: str, **kwargs) -> None:
        '''
        Initialize the object.

        Args:
            name (str): name of the object
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = ObjectCPP(name)

    @property
    def name(self) -> str:
        '''
        Get the name of the object.

        Returns:
            str: name of the object
        '''
        return self.instance.name

    @name.setter
    def name(self, name: str) -> None:
        '''
        Set the name of the object.

        Args:
            name str: name of the object
        '''
        self.instance.name = name
