# cpp/steering.py
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
from ..base.steering import Steering as SteeringABC
from egret.cppegret import Steering as SteeringCPP
from .element import Element
from typing import Tuple

class Steering(SteeringABC, Element):
    '''
    Steering magnet class.
    '''

    def __init__(self, name: str, length: float,
                 kick_x: float = 0.0, kick_y: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = "", **kwargs) -> None:
        '''
        Initialize a steering magnet.

        Args:
            name str: Name of the steering magnet.
            length float: Length of the steering magnet [m].
            kick_x float: Horizontal kick angle [rad].
            kick_y float: Vertical kick angle [rad].
            dx float: Horizontal offset of the magnetic center [m].
            dy float: Vertical offset of the magnetic center [m].
            ds float: Longitudinal offset of the magnetic center [m].
            tilt float: Tilt angle around the beam axis [rad].
            info str: Additional information.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = SteeringCPP(name, length, kick_x, kick_y,
                                        dx, dy, ds, tilt, info)
        super().__init__(None, None, None, instance=self.instance)

    @property
    def kick_x(self) -> float:
        '''
        Horizontal kick angle [rad].
        '''
        return self.instance.kick_x

    @property
    def kick_y(self) -> float:
        '''
        Vertical kick angle [rad].
        '''
        return self.instance.kick_y

    @property
    def kick(self) -> Tuple[float, float]:
        '''
        Return a tuple of horizontal and vertical kick angles [rad].
        '''
        return self.instance.kick

    @kick_x.setter
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        self.instance.kick_x = kick_x

    @kick_y.setter
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle.

        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        self.instance.kick_y = kick_y

    @kick.setter
    def kick(self, kick_x: float, kick_y: float) -> None:
        '''
        Set the steering angles.

        Args:
            kick_x float: Horizontal kick angle [rad].
            kick_y float: Vertical kick angle [rad].
        '''
        self.instance.kick = kick_x, kick_y

    def set_steering(self, kick_x: float | None = None,
                     kick_y: float | None = None) -> None:
        '''
        Set the steering angles.

        Args:
            kick_x float: Horizontal kick angle [rad] or None to leave unchanged.
            kick_y float: Vertical kick angle [rad] or None to leave unchanged.
        '''
        self.instance.set_steering(kick_x, kick_y)
