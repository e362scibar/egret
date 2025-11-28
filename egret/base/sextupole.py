# base/sextupole.py
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
from .drift import Drift
from .quadrupole import Quadrupole
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Sextupole(Element):
    '''
    Base class of a sextupole magnet.
    '''

    @property
    @abstractmethod
    def k2(self) -> float:
        '''
        Normalized sextupole strength [1/m^3].
        '''
        pass

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
    def dxp(self) -> float:
        '''
        Horizontal kick angle of the steering coil [rad].
        '''
        pass

    @property
    @abstractmethod
    def dyp(self) -> float:
        '''
        Vertical kick angle of the steering coil [rad].
        '''
        pass

    @k2.setter
    @abstractmethod
    def k2(self, value: float) -> None:
        '''
        Set normalized sextupole strength.

        Args:
            value float: Normalized sextupole strength [1/m^3].
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

    @dxp.setter
    @abstractmethod
    def dxp(self, value: float) -> None:
        '''
        Set horizontal kick angle of the steering coil.

        Args:
            value float: Horizontal kick angle [rad].
        '''
        pass

    @dyp.setter
    @abstractmethod
    def dyp(self, value: float) -> None:
        '''
        Set vertical kick angle of the steering coil.

        Args:
            value float: Vertical kick angle [rad].
        '''
        pass

    @abstractmethod
    def transfer_matrix_by_midpoint_method(self, cood0: Coordinate, ds: float = 0.1,
                                           tmatflag: bool = True, dispflag: bool = False) \
        -> Tuple[npt.NDArray[np.floating], Coordinate, npt.NDArray[np.floating]]:
        '''
        Calculate a single step transfer matrix using the midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Step size [m] for integration.
            tmatflag bool: Calculate transfer matrix if true. (default: True)
            dispflag bool: Calculate additive dispersion if True. (default: False)

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix, if tmatflag is True, else None.
            Coordinate: Final coordinate after the step.
            npt.NDArray[np.floating]: Additive dispersion if dispflag is True, else None.
        '''
        pass

    @abstractmethod
    def set_steering(self, dxp: float = None, dyp: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            dxp float: Horizontal kick angle of the steering coil [rad].
            dyp float: Vertical kick angle of the steering coil [rad].
        '''
        if dxp is not None:
            self.dxp = dxp
            self.k0x = - self.dxp / self.length
        if dyp is not None:
            self.dyp = dyp
            self.k0y = - self.dyp / self.length
