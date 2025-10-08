# octupole.py
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

from .element import Element

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Octupole(Element):
    '''
    Octupole magnet.
    '''

    def __init__(self, name: str, length: float, k3: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            k3 float: Octupole strength [1/m^4].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k3 = k3
        self.update()
    
    def update(self):
        '''
        Update transfer matrix and dispersion.
        '''
        # temporarilly set to drift
        self.tmat[0, 1] = self.length
        self.tmat[2, 3] = self.length

    def tmatarray(self, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal position array of shape (N,).
        '''
        # temporarilly set to drift
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
        tmat = np.repeat(self.tmat[np.newaxis, :, :], len(s), axis=0)
        tmat[:, 0, 1] = s
        tmat[:, 2, 3] = s
        return tmat, s
