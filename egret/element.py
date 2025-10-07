# element.py
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

from .object import Object
from .betafunc import BetaFunc

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Element(Object):
    '''
    Base class of an accelerator element.
    ''' 
    def __init__(self, name:str, length:float,
                 dx:float=0., dy:float=0., ds:float=0., tilt:float=0., info:str=''):
        super().__init__(name)
        self.length = length
        self.dx = dx
        self.dy = dy
        self.ds = ds
        self.tilt = tilt
        self.info = info
        self.tmat = np.eye(6)
        self.disp = np.zeros(6)

    def tmatarray(self, ds:float=0.01, endpoint:bool=False)->npt.NDArray[np.floating]:
        s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1)
        print('Element.tmatarray()', self.name, self.length, len(s))
        return np.repeat(self.tmat[np.newaxis,:,:], len(s), axis=0), s

    def betafunc(self, b0:BetaFunc, ds:float=0.01, endpoint:bool=False)->BetaFunc:
        tmat, s = self.tmatarray(ds, endpoint)
        return b0.transfer(tmat, s)

    def dispersion(self, ds:float=0.01, endpoint:bool=False)->Tuple[npt.NDArray[np.floating],npt.NDArray[np.floating]]:
        n = int(self.length//ds)+int(endpoint)+1
        s = np.linspace(0., self.length, n, endpoint)
        return np.zeros((6, n)), s

    def etafunc(self, eta0:npt.NDArray[np.floating], ds:float=0.01, endpoint:bool=False)->Tuple[npt.NDArray[np.floating],npt.NDArray[np.floating]]:
        disp, s = self.dispersion(ds, endpoint)
        tmat, _ = self.tmatarray(ds, endpoint)
        return np.matmul(tmat, eta0).T + disp, s

    def radiation_integrals(self, beta0:BetaFunc, eta0:npt.NDArray[np.floating], ds:float=0.1)->Tuple[float,float,float]:
        '''
        Calculate radiation integrals.

        Args:
            beta0 BetaFunc: Initial Twiss parameters.
            eta0 npt.NDArray[np.floating]: Initial dispersion [eta_x, eta_x', eta_y, eta_y'].
            ds float: Step size for numerical integration.

        Returns:
            float, float, float: Radiation integrals I2, I4, and I5.
        '''
        return 0., 0., 0.
