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

class Element(Object):
    """
    Base class of an accelerator element.
    """
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
        return np.repeat(self.tmat[np.newaxis,:,:], n, axis=0), s
    
    def betafunc(self, b0:BetaFunc, ds:float=0.01, endpoint:bool=False)->BetaFunc:
        return b0.transfer(*self.tmatarray(ds, endpoint))
