# cpp/nonlinearmultipole.py
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
from ..base.nonlinearmultipole import NonlinearMultipole as NonlinearMultipoleABC
from egret.cppegret import NonlinearMultipole as NonlinearMultipoleCPP
from .element import Element
from .coordinate import Coordinate
from typing import Tuple

class NonlinearMultipole(NonlinearMultipoleABC, Element):
    '''
    Base class for nonlinear multipole magnets.
    '''

    def __init__(self, name: str, length: float,
                 kick_x: float = 0.0, kick_y: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize nonlinear multipole magnet.

        Args:
            name str: Element name.
            length float: Element length [m].
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
            self.instance = NonlinearMultipoleCPP(name, length, kick_x, kick_y,
                                                  dx, dy, ds, tilt, info)
        super().__init__(None, None, None, instance=self.instance)

    @property
    def k0x(self) -> float:
        '''
        Horizontal steering strength [1/m].
        '''
        return self.instance.k0x

    @property
    def k0y(self) -> float:
        '''
        Vertical steering strength [1/m].
        '''
        return self.instance.k0y

    @property
    def kick_x(self) -> float:
        '''
        Horizontal kick angle of the steering coil [rad].
        '''
        return self.instance.kick_x

    @property
    def kick_y(self) -> float:
        '''
        Vertical kick angle of the steering coil [rad].
        '''
        return self.instance.kick_y

    @k0x.setter
    def k0x(self, k0x: float) -> None:
        '''
        Set horizontal steering strength.

        Args:
            k0x float: Horizontal steering strength [1/m].
        '''
        self.instance.k0x = k0x

    @k0y.setter
    def k0y(self, k0y: float) -> None:
        '''
        Set vertical steering strength.

        Args:
            k0y float: Vertical steering strength [1/m].
        '''
        self.instance.k0y = k0y

    @kick_x.setter
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle of the steering coil.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        self.instance.kick_x = kick_x

    @kick_y.setter
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle of the steering coil.

        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        self.instance.kick_y = kick_y

    def set_steering(self, kick_x: float = None, kick_y: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            kick_x float: Horizontal kick angle of the steering coil [rad] or None to keep current value.
            kick_y float: Vertical kick angle of the steering coil [rad] or None to keep current value.
        '''
        self.instance.set_steering(kick_x, kick_y)

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
        return self.instance.get_k(cood.instance)

    def copy(self) -> NonlinearMultipole:
        '''
        Create a copy of this element.

        Returns:
            NonlinearMultipole: Copied element.
        '''
        return NonlinearMultipole(self.instance.name, self.instance.length,
            self.instance.kick_x, self.instance.kick_y,
            self.instance.dx, self.instance.dy, self.instance.ds,
            self.instance.tilt, self.instance.info)