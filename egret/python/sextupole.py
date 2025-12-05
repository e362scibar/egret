# python/sextupole.py
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
from ..base.sextupole import Sextupole as SextupoleABC
from .nonlinearmultipole import NonlinearMultipole
from .coordinate import Coordinate
from typing import Tuple

class Sextupole(SextupoleABC, NonlinearMultipole):
    '''
    Sextupole magnet class.
    '''

    def __init__(self, name: str, length: float, k2: float,
                 kick_x: float = 0., kick_y: float = 0.,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            k2 float: Normalized sextupole strength [1/m^3].
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, kick_x, kick_y, dx, dy, ds, tilt, info)
        self._k2 = k2

    @property
    def k2(self) -> float:
        '''
        Normalized sextupole strength [1/m^3].
        '''
        return self._k2

    @k2.setter
    def k2(self, k2: float) -> None:
        '''
        Set normalized sextupole strength [1/m^3].

        Args:
            k2 float: Normalized sextupole strength [1/m^3].
        '''
        self._k2 = k2

    def copy(self) -> Sextupole:
        '''
        Return a copy of the sextupole.

        Returns:
            Sextupole: Copy of the sextupole.
        '''
        return Sextupole(self._name, self._length, self._k2, self._kick_x, self._kick_y,
                         self._dx, self._dy, self._ds, self._tilt, self._info)

    def get_k(self, cood: Coordinate) -> Tuple[complex, complex]:
        '''
        Calculate dipole and quadrupole strengthis at given coordinate.
        x' + j y' = - k0 L - K1 L (x - j y)

        Args:
            cood Coordinate: Particle coordinate.

        Returns:
            float:
        '''
        k2 = self._k2 / (1. + cood.delta)
        k0x, k0y = self.k0x / (1. + cood.delta), self.k0y / (1. + cood.delta)
        x0, y0 = cood.x, cood.y, cood.xp, cood.yp
        # dipole strength
        k0 = k2 * (0.5 * (x0**2 - y0**2) - 1.j * x0 * y0) + k0x + 1.j * k0y
        # quadrupole strength
        k1 = k2 * (x0 - 1.j * y0)
        return k0, k1
