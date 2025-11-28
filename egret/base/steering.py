# base/steering.py
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

class Steering(Element):
    '''
    Base class of a steering magnet.
    '''

    @property
    @abstractmethod
    def dxp(self) -> float:
        '''
        Horizontal deflection angle [rad].
        '''
        pass

    @property
    @abstractmethod
    def dyp(self) -> float:
        '''
        Vertical deflection angle [rad].
        '''
        pass

    @dxp.setter
    @abstractmethod
    def dxp(self, value: float) -> None:
        '''
        Set horizontal deflection angle.

        Args:
            value float: Horizontal deflection angle [rad].
        '''
        pass

    @dyp.setter
    @abstractmethod
    def dyp(self, value: float) -> None:
        '''
        Set vertical deflection angle.

        Args:
            value float: Vertical deflection angle [rad].
        '''
        pass

    @abstractmethod
    def set_steering(self, dxp: float, dyp: float) -> None:
        '''
        Set the steering angles.

        Args:
            dxp float: Horizontal deflection angle [rad].
            dyp float: Vertical deflection angle [rad].
        '''
        self.dxp = dxp
        self.dyp = dyp
