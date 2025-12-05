# python/octupole.py
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
from ..base.octupole import Octupole as OctupoleABC
from .nonlinearmultipole import NonlinearMultipole
from .coordinate import Coordinate
import numpy as np
from typing import Tuple

class Octupole(OctupoleABC, NonlinearMultipole):
    '''
    Octupole magnet class.
    '''

    def __init__(self, name: str, length: float, k3: float,
                 k1: float = 0., tilt_quad: float = 0.,
                 kick_x: float = 0., kick_y: float = 0.,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = '') -> None:
        '''
        Initialize octupole magnet.

        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            k3 float: Octupole strength [1/m^4].
            k1 float: Additional quadrupole strength [1/m^2].
            tilt_quad float: Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, kick_x, kick_y, dx, dy, ds, tilt, info)
        self._k3 = k3
        self.set_quadrupole(k1, tilt_quad)

    @property
    def k3(self) -> float:
        '''
        Normalized octupole strength [1/m^4].
        '''
        return self._k3

    @property
    def k1(self) -> float:
        '''
        Additional quadrupole strength [1/m^2].
        '''
        return self._k1

    @property
    def tilt_quad(self) -> float:
        '''
        Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
        '''
        return self._tilt_quad

    @k3.setter
    def k3(self, k3: float) -> None:
        '''
        Set normalized octupole strength.

        Args:
            k3 float: Normalized octupole strength [1/m^4].
        '''
        self._k3 = k3

    @k1.setter
    def k1(self, k1: float) -> None:
        '''
        Set additional quadrupole strength.

        Args:
            k1 float: Additional quadrupole strength [1/m^2].
        '''
        self._k1 = k1

    @tilt_quad.setter
    def tilt_quad(self, tilt_quad: float) -> None:
        '''
        Set tilt angle of the additional quadrupole.

        Args:
            tilt_quad float: Tilt angle [rad] (pi/4 for skew quad).
        '''
        self._tilt_quad = tilt_quad

    def copy(self) -> Octupole:
        '''
        Return a copy of the octupole.

        Returns:
            Octupole: Copy of the octupole.
        '''
        return Octupole(self._name, self._length, self._k3,
                        self._k1, self._tilt_quad, self._kick_x, self._kick_y,
                        self._dx, self._dy, self._ds, self._tilt, self._info)

    def get_k(self, cood: Coordinate) -> Tuple[complex, complex]:
        '''
        Calculate dipole and quadrupole strengths at given coordinate.
        x' + j y' = - k0 L - k1 L (x - j y)

        Args:
            cood Coordinate: Coordinate.

        Returns:
            complex: Dipole strength [1/m].
            complex: Quadrupole strength [1/m^2].
        '''
        k3 = self._k3 / (1. + cood.delta)
        k0x, k0y = self.k0x / (1. + cood.delta), self.k0y / (1. + cood.delta)
        k1 = self._k1 / (1. + cood.delta)
        x, y = cood.x, cood.y
        k0 = k3 * (x**3 / 6. - 0.5 * x * y**2 + 1.j * (y**3 / 6. - 0.5 * x**2 * y)) \
            + k0x + 1.j * k0y + k1 * np.exp(2.j * self._tilt_quad) * (x - 1.j * y)
        k1 = k3 * (0.5 * (x**2 - y**2) - 1.j * x * y) + k1 * np.exp(2.j * self._tilt_quad)
        return k0, k1

    def set_quadrupole(self, k1: float = None, tilt_quad: float = None) -> None:
        '''
        Set additional quadrupole strength and tilt angle.

        Args:
            k1 float: Additional quadrupole strength [1/m^2] or None to leave unchanged.
            tilt_quad float: Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad) or None to leave unchanged.
        '''
        if k1 is not None:
            self._k1 = k1
        if tilt_quad is not None:
            self._tilt_quad = tilt_quad
