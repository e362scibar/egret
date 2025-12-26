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
import numpy as np
import numpy.typing as npt
from typing import Final

class Ring(Element):
    '''
    Base class of a ring accelerator.
    '''

    C_q: Final = 3.8319e-13  # Quantum excitation factor [m]
    m_e_eV: Final = 510998.9461  # Electron rest mass [eV/c^2]

    tol_cod = 1.0e-12  # Tolerance for closed orbit calculation [m]
    max_iter_cod = 50  # Maximum iteration for closed orbit calculation

    @property
    @abstractmethod
    def energy(self) -> float:
        '''
        Beam energy [eV].
        '''
        pass

    @property
    @abstractmethod
    def tune_x(self) -> float:
        '''
        Horizontal tune of the ring.
        '''
        pass

    @property
    @abstractmethod
    def tune_y(self) -> float:
        '''
        Vertical tune of the ring.
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
    def emittance_x(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal equilibrium emittance [m.rad].
        '''
        pass

    @property
    @abstractmethod
    def emittance_y(self) -> npt.NDArray[np.floating]:
        '''
        Vertical equilibrium emittance [m.rad].
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
    def update(self, delta: float = 0., method: str = 'symplectic4') -> None:
        '''
        Update transfer matrix, dispersion, and emittance.

        Args:
            delta float: Relative momentum deviation (default: 0.).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').
        '''
        pass

    @abstractmethod
    def find_initial_coordinate_of_closed_orbit(guess: Coordinate = None, method: str = 'symplectic4') -> None:
        '''
        Find initial coordinate of the closed orbit using Newton-Raphson method.

        Args:
            guess Coordinate: Initial guess of the closed orbit.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').
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
