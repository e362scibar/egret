# base/octupole.py
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

class Octupole(Element):
    '''
    Base class of an octupole magnet.
    '''

    @property
    @abstractmethod
    def k3(self) -> float:
        '''
        Normalized octupole strength [1/m^4].
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

    @property
    @abstractmethod
    def k1(self) -> float:
        '''
        Additional quadrupole strength [1/m^2].
        '''
        pass

    @property
    @abstractmethod
    def tilt_quad(self) -> float:
        '''
        Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
        '''
        pass

    @k3.setter
    @abstractmethod
    def k3(self, value: float) -> None:
        '''
        Set normalized octupole strength.

        Args:
            value float: Normalized octupole strength [1/m^4].
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

    @k1.setter
    @abstractmethod
    def k1(self, value: float) -> None:
        '''
        Set additional quadrupole strength.

        Args:
            value float: Additional quadrupole strength [1/m^2].
        '''
        pass

    @tilt_quad.setter
    @abstractmethod
    def tilt_quad(self, value: float) -> None:
        '''
        Set tilt angle of the additional quadrupole.

        Args:
            value float: Tilt angle [rad] (pi/4 for skew quad).
        '''
        pass

    @abstractmethod
    def transfer_matrix_by_midpoint_method(self, cood0: Coordinate, ds: float = 0.1,
                                    tmat_flag: bool = True, disp_flag: bool = False) \
    -> Tuple[npt.NDArray[np.floating], Coordinate, npt.NDArray[np.floating]]:
        '''
        Transfer matrix by the midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
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
        pass

    @abstractmethod
    def set_quadrupole(self, k1: float = None, tilt_quad: float = None) -> None:
        '''
        Set additional quadrupole strength and tilt angle.

        Args:
            k1 float: Additional quadrupole strength [1/m^2].
            tilt_quad float: Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
        '''
        pass
