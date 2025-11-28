# base/ring.py
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
import scipy
from typing import Tuple, List

class Ring(Element):
    '''
    Base class of a ring accelerator.
    '''

    @property
    @abstractmethod
    def elements(self) -> List[Element]:
        '''
        List of elements in the ring.
        '''
        pass

    @elements.setter
    @abstractmethod
    def elements(self, value: List[Element]) -> None:
        '''
        Set list of elements in the ring.
        '''
        pass

    @property
    @abstractmethod
    def energy(self) -> float:
        '''
        Beam energy [eV].
        '''
        pass

    @energy.setter
    @abstractmethod
    def energy(self, value: float) -> None:
        '''
        Set beam energy [eV].
        '''
        pass

    @property
    @abstractmethod
    def tune(self) -> npt.NDArray[np.floating]:
        '''
        Tune of the ring.
        '''
        pass

    @property
    @abstractmethod
    def emittance(self) -> npt.NDArray[np.floating]:
        '''
        Equilibrium emittance [m.rad].
        '''
        pass

    @property
    @abstractmethod
    def cood0(self) -> Coordinate:
        '''
        Initial coordinate of the closed orbit.
        '''
        pass

    @property
    @abstractmethod
    def evlp0(self) -> Envelope:
        '''
        Initial beam envelope of the closed orbit.
        '''
        pass

    @property
    @abstractmethod
    def disp0(self) -> Dispersion:
        '''
        Initial dispersion of the closed orbit.
        '''
        pass

    @property
    @abstractmethod
    def Jx(self) -> float:
        '''
        Horizontal damping partition number.
        '''
        pass

    @property
    @abstractmethod
    def Jy(self) -> float:
        '''
        Vertical damping partition number.
        '''
        pass

    @property
    @abstractmethod
    def Jz(self) -> float:
        '''
        Longitudinal damping partition number.
        '''
        pass

    @abstractmethod
    def update(self, delta: float = 0.):
        '''
        Update transfer matrix, dispersion, and emittance.

        Args:
            delta float: Relative momentum deviation (default: 0.).
        '''
        pass

    @classmethod
    @abstractmethod
    def read_json(cls, path: str) -> Ring:
        '''
        Read ring from a LatticeJSON file.

        Args:
            path str: Path to the LatticeJSON file.
        Returns:
            Ring: Ring object.
        '''
        pass
