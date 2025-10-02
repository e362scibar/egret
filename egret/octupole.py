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

class Octupole(Element):
    """
    Octupole magnet.
    """
    def __init__(self, name, length, k3, dx=0., dy=0., ds=0., tilt=0., info=''):
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k3 = k3
        self.update()
    
    def update(self):
        # temporarilly set to drift
        self.tmat[0,1] = self.length
        self.tmat[2,3] = self.length

    def tmatarray(self, ds:float=0.01, endpoint:bool=False)->npt.NDArray[np.floating]:
        # temporarilly set to drift
        s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1, endpoint)
        tmat = np.repeat(self.tmat[np.newaxis,:,:], len(s), axis=0)
        tmat[:,0,1] = s
        tmat[:,2,3] = s
        return tmat
