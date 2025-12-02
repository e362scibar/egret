# base/nonlinearmultipole.py
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
from .coordinate import Coordinate

class NonlinearMultipole(Element):
    '''
    Base class for nonlinear multipole magnets.
    '''

    @property
    @abstractmethod
    def k0x(self) -> float:
        '''
        Horizontal steering strength [1/m].
        '''
        pass

    @property
    @abstractmethod
    def k0y(self) -> float:
        '''
        Vertical steering strength [1/m].
        '''
        pass

    @property
    @abstractmethod
    def kick_x(self) -> float:
        '''
        Horizontal kick angle of the steering coil [rad].
        '''
        pass

    @property
    @abstractmethod
    def kick_y(self) -> float:
        '''
        Vertical kick angle of the steering coil [rad].
        '''
        pass

    @k0x.setter
    @abstractmethod
    def k0x(self, value: float) -> None:
        '''
        Set horizontal steering strength.

        Args:
            value float: Horizontal steering strength [1/m].
        '''
        pass

    @k0y.setter
    @abstractmethod
    def k0y(self, value: float) -> None:
        '''
        Set vertical steering strength.

        Args:
            value float: Vertical steering strength [1/m].
        '''
        pass

    @kick_x.setter
    @abstractmethod
    def kick_x(self, value: float) -> None:
        '''
        Set horizontal kick angle of the steering coil.

        Args:
            value float: Horizontal kick angle [rad].
        '''
        pass

    @kick_y.setter
    @abstractmethod
    def kick_y(self, value: float) -> None:
        '''
        Set vertical kick angle of the steering coil.

        Args:
            value float: Vertical kick angle [rad].
        '''
        pass

    @abstractmethod
    def set_steering(self, kick_x: float = None, kick_y: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
        '''
        pass

    @abstractmethod
    def get_k(self, cood: Coordinate) -> float:
        '''
        Get quadrupole strength at given coordinate.

        Args:
            cood Coordinate: Coordinate.
        '''
        pass
