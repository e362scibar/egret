# base/dispersion.py
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

from abc import ABC, abstractmethod

class Dispersion(ABC):
    '''
    Base class for energy dispersion.
    '''

    @property
    @abstractmethod
    def x(self) -> float:
        '''
        Horizontal dispersion [m].
        '''
        pass

    @property
    @abstractmethod
    def xp(self) -> float:
        '''
        Horizontal angle dispersion [rad].
        '''
        pass

    @property
    @abstractmethod
    def y(self) -> float:
        '''
        Vertical dispersion [m].
        '''
        pass

    @property
    @abstractmethod
    def yp(self) -> float:
        '''
        Vertical angle dispersion [rad].
        '''
        pass

    @property
    @abstractmethod
    def s(self) -> float:
        '''
        Longitudinal position [m].
        '''
        pass

    @x.setter
    @abstractmethod
    def x(self, x: float) -> None:
        '''
        Set horizontal dispersion.

        Args:
            x float: Horizontal dispersion [m].
        '''
        pass

    @xp.setter
    @abstractmethod
    def xp(self, xp: float) -> None:
        '''
        Set horizontal angle dispersion.

        Args:
            xp float: Horizontal angle dispersion [rad].
        '''
        pass

    @y.setter
    @abstractmethod
    def y(self, y: float) -> None:
        '''
        Set vertical dispersion.

        Args:
            y float: Vertical dispersion [m].
        '''
        pass

    @yp.setter
    @abstractmethod
    def yp(self, yp: float) -> None:
        '''
        Set vertical angle dispersion.

        Args:
            yp float: Vertical angle dispersion [rad].
        '''
        pass

    @s.setter
    @abstractmethod
    def s(self, s: float) -> None:
        '''
        Set longitudinal position.

        Args:
            s float: Longitudinal position [m].
        '''
        pass

    @abstractmethod
    def copy(self) -> Dispersion:
        '''
        Returns:
            Dispersion: A copy of the coordinate object.
        '''
        pass
