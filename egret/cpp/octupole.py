# cpp/octupole.py
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
from egret.cppegret import Octupole as OctupoleCPP
from .nonlinearmultipole import NonlinearMultipole
from .coordinate import Coordinate
from typing import Tuple

class Octupole(NonlinearMultipole):
    '''
    Octupole magnet class.
    '''

    def __init__(self, name: str, length: float, k3: float = 0.0,
                 k1: float = 0.0, tilt_quad: float = 0.0,
                 kick_x: float = 0.0, kick_y: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize octupole magnet.

        Args:
            name str: Element name.
            length float: Element length [m].
            k3 float: Normalized octupole strength [1/m^4].
            k1 float: Additional quadrupole strength [1/m^2].
            tilt_quad float: Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
            dx float: Horizontal offset of the magnetic center [m].
            dy float: Vertical offset of the magnetic center [m].
            ds float: Longitudinal offset of the magnetic center [m].
            tilt float: Rotation angle around the beam axis [rad].
            info str: Additional information.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = OctupoleCPP(name, length, k3, k1, tilt_quad,
                                        kick_x, kick_y, dx, dy, ds, tilt, info)
        super().__init__(None, None, instance=self.instance)

    @property
    def k3(self) -> float:
        '''
        Normalized octupole strength [1/m^4].
        '''
        return self.instance.k3

    @property
    def k1(self) -> float:
        '''
        Additional quadrupole strength [1/m^2].
        '''
        return self.instance.k1

    @property
    def tilt_quad(self) -> float:
        '''
        Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
        '''
        return self.instance.tilt_quad

    @k3.setter
    def k3(self, k3: float) -> None:
        '''
        Set normalized octupole strength.

        Args:
            k3 float: Normalized octupole strength [1/m^4].
        '''
        self.instance.k3 = k3

    @k1.setter
    def k1(self, k1: float) -> None:
        '''
        Set additional quadrupole strength.

        Args:
            k1 float: Additional quadrupole strength [1/m^2].
        '''
        self.instance.k1 = k1

    @tilt_quad.setter
    def tilt_quad(self, tilt_quad: float) -> None:
        '''
        Set tilt angle of the additional quadrupole.

        Args:
            tilt_quad float: Tilt angle [rad] (pi/4 for skew quad).
        '''
        self.instance.tilt_quad = tilt_quad

    def get_k(self, cood: Coordinate) -> Tuple[complex, complex]:
        '''
        Calculate dipole and quadrupole strengths at given coordinate.
        x' + j y' = - k0 L - k1 L (x - j y)

        Args:
            cood Coordinate: Particle coordinate.

        Returns:
            complex: Dipole strength [1/m].
            complex: Quadrupole strength [1/m^2].
        '''
        return self.instance.get_k(cood.instance)

    def set_quadrupole(self, k1: float = None, tilt_quad: float = None) -> None:
        '''
        Set additional quadrupole strength and tilt angle.

        Args:
            k1 float: Additional quadrupole strength [1/m^2].
            tilt_quad float: Tilt angle of the additional quadrupole [rad] (pi/4 for skew quad).
        '''
        self.instance.set_quadrupole(k1, tilt_quad)

    def copy(self) -> Octupole:
        '''
        Create a copy of this octupole.

        Returns:
            Octupole: Copied octupole instance.
        '''
        return Octupole(self.instance.name, self.instance.length,
            self.instance.k3, self.instance.k1, self.instance.tilt_quad,
            self.instance.kick_x, self.instance.kick_y,
            self.instance.dx, self.instance.dy, self.instance.ds,
            self.instance.tilt, self.instance.info)
