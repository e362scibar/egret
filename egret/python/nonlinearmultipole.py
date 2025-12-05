# base/nonlinearmultipole.py
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

from anyio import value
from ..base.nonlinearmultipole import NonlinearMultipole as NonlinearMultipoleABC
from .element import Element
from .coordinate import Coordinate

class NonlinearMultipole(NonlinearMultipoleABC, Element):
    '''
    Base class for nonlinear multipole magnets.
    '''

    def __init__(self, name: str, length: float, kick_x: float = 0.0, kick_y: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0., info: float = '') -> None:
        '''
        Initialize nonlinear multipole magnet class.

        Args:
            name str: Element name.
            length float: Element length [m].
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
            dx float: Horizontal offset of the element center [m].
            dy float: Vertical offset of the element center [m].
            ds float: Longitudinal offset of the element center [m].
            tilt float: Tilt angle around the beam axis [rad].
            info str: Additional information.
        '''
        super().__init__(self, name, length, 0.0, dx, dy, ds, tilt, info)
        self._kick_x = kick_x
        self._kick_y = kick_y

    @property
    def k0x(self) -> float:
        '''
        Horizontal steering strength [1/m].
        '''
        return self._kick_x / self._length

    @property
    def k0y(self) -> float:
        '''
        Vertical steering strength [1/m].
        '''
        return self._kick_y / self._length

    @property
    def kick_x(self) -> float:
        '''
        Horizontal kick angle of the steering coil [rad].
        '''
        return self._kick_x

    @property
    def kick_y(self) -> float:
        '''
        Vertical kick angle of the steering coil [rad].
        '''
        return self._kick_y

    @k0x.setter
    def k0x(self, k0x: float) -> None:
        '''
        Set horizontal steering strength.

        Args:
            k0x float: Horizontal steering strength [1/m].
        '''
        self._kick_x = k0x * self._length

    @k0y.setter
    def k0y(self, k0y: float) -> None:
        '''
        Set vertical steering strength.

        Args:
            k0y float: Vertical steering strength [1/m].
        '''
        self._kick_y = k0y * self._length

    @kick_x.setter
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle of the steering coil.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        self._kick_x = kick_x

    @kick_y.setter
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle of the steering coil.

        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        self._kick_y = kick_y

    def set_steering(self, kick_x: float = None, kick_y: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            kick_x float: Horizontal kick angle of the steering coil [rad] or None to leave unchanged.
            kick_y float: Vertical kick angle of the steering coil [rad] or None to leave unchanged.
        '''
        if kick_x is not None:
            self._kick_x = kick_x
        if kick_y is not None:
            self._kick_y = kick_y

    def get_k(self, cood: Coordinate) -> float:
        '''
        Get quadrupole strength at given coordinate.

        Args:
            cood Coordinate: Coordinate.
        '''
        raise NotImplementedError('get_k method is not implemented in NonlinearMultipole base class.')
