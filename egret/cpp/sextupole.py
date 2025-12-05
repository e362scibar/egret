# cpp/sextupole.py
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
from egret.cppegret import Sextupole as SextupoleCPP
from .nonlinearmultipole import NonlinearMultipole

class Sextupole(SextupoleABC, NonlinearMultipole):
    '''
    Sextupole magnet class.
    '''

    def __init__(self, name: str, length: float, k2: float = 0.0,
                 kick_x: float = 0.0, kick_y: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = '', **kwargs) -> None:
        '''
        Initialize sextupole magnet.

        Args:
            name str: Element name.
            length float: Element length [m].
            k2 float: Normalized sextupole strength [1/m^3].
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
            self.instance = SextupoleCPP(name, length, k2, kick_x, kick_y,
                                         dx, dy, ds, tilt, info)
        super().__init__(None, None, instance=self.instance)

    @property
    def k2(self) -> float:
        '''
        Normalized sextupole strength [1/m^3].
        '''
        return self.instance.k2

    @k2.setter
    def k2(self, k2: float) -> None:
        '''
        Set normalized sextupole strength.

        Args:
            k2 float: Normalized sextupole strength [1/m^3].
        '''
        self.instance.k2 = k2
