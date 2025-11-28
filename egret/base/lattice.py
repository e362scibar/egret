# base/lattice.py
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
from .envelope import Envelope
from .dispersion import Dispersion
import copy
import numpy as np
import numpy.typing as npt
from typing import Tuple, List

class Lattice(Element):
    '''
    Lattice element.
    '''

    @property
    @abstractmethod
    def elements(self) -> List[Element]:
        '''
        List of elements in the lattice.
        '''
        pass

    @elements.setter
    @abstractmethod
    def elements(self, value: List[Element]) -> None:
        '''
        Set list of elements in the lattice.
        '''
        pass

    @abstractmethod
    def update(self):
        '''
        Update bending angle of the lattice.
        '''
        pass

    @abstractmethod
    def get_element(self, key: int | Tuple[int, ...]) -> Element:
        '''
        Get element by index or tuple of indices.

        Args:
            key int or tuple of int: Index or tuple of indices.

        Returns:
            Element: Element at the specified index.
        '''
        pass

    @abstractmethod
    def get_s(self, key: int | Tuple[int, ...]) -> float:
        '''
        Get longitudinal position by index or tuple of indices.

        Args:
            key int or tuple of int: Index or tuple of indices.

        Returns:
            float: Longitudinal position [m].
        '''
        pass

    @abstractmethod
    def find_index(self, name: str | Tuple[str, ...]) -> Tuple[int, ...]:
        '''
        Find indices of elements starting with a given name.

        Args:
            name str | tuple of str: Name of the element.

        Returns:
            tuple of int: Tuple of indices of the element.
        '''
        pass
