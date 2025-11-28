# base/quadrupole.py
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

class Quadrupole(Element):
    '''
    Base class of a quadrupole magnet.
    '''

    @property
    @abstractmethod
    def k1(self) -> float:
        '''
        Normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
        '''
        pass

    @k1.setter
    @abstractmethod
    def k1(self, value: float) -> None:
        '''
        Set normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
        '''
        pass
