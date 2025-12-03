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
from typing import Tuple

class Steering(Element):
    '''
    Base class of a steering magnet.
    '''

    @property
    @abstractmethod
    def kick_x(self) -> float:
        '''
        Horizontal kick angle [rad].
        '''
        pass

    @property
    @abstractmethod
    def kick_y(self) -> float:
        '''
        Vertical kick angle [rad].
        '''
        pass

    @property
    @abstractmethod
    def kick(self) -> Tuple[float, float]:
        '''
        Return a tuple of horizontal and vertical kick angles [rad].
        '''
        pass

    @kick_x.setter
    @abstractmethod
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        pass

    @kick_y.setter
    @abstractmethod
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle.
        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        pass

    @kick.setter
    @abstractmethod
    def kick(self, kick_x: float, kick_y: float) -> None:
        '''
        Set the steering angles.
        '''
        pass

    @abstractmethod
    def set_steering(self, kick_x: float, kick_y: float) -> None:
        '''
        Set the steering angles.

        Args:
            kick_x float: Horizontal kick angle [rad].
            kick_y float: Vertical kick angle [rad].
        '''
        pass

    @abstractmethod
    def copy(self) -> Steering:
        '''
        Return a copy of the steering element.
        '''
        pass
