# quadrupole.py
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

class Quadrupole(Element):
    """
    Quadrupole magnet.
    """
    def __init__(self, name:str, length:float, k1:float,
                 dx:float=0., dy:float=0., ds:float=0., tilt:float=0., info:float=''):
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k1 = k1
        self.update()

    def update(self):
        k = np.abs(self.k1)
        psi = np.sqrt(k) * self.length
        if k == 0.: # drift
            self.tmat[0,1] = self.length
            self.tmat[2,3] = self.length
            return
        mf = np.array([[np.cos(psi), np.sin(psi)/np.sqrt(k)],
                       [-np.sqrt(k)*np.sin(psi), np.cos(psi)]])
        md = np.array([[np.cosh(psi), np.sinh(psi)/np.sqrt(k)],
                       [np.sqrt(k)*np.sinh(psi), np.cosh(psi)]])
        if self.k1 < 0.: # defocusing quadrupole
            self.tmat[0:2,0:2] = md
            self.tmat[2:4,2:4] = mf
        else: # focusing quadrupole
            self.tmat[0:2,0:2] = mf
            self.tmat[2:4,2:4] = md

    def tmatarray(self, ds:float=0.01, endpoint:bool=False)->npt.NDArray[np.floating]:
        k = np.abs(self.k1)
        s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1, endpoint)
        tmat = np.repeat(np.eye(6)[np.newaxis,:,:], len(s), axis=0)
        if k == 0.: # drift
            tmat[:,0,1] = s
            tmat[:,2,3] = s
            return tmat, s
        psi = np.sqrt(k) * s
        mf = np.moveaxis(np.array([[np.cos(psi), np.sin(psi)/np.sqrt(k)],
                                   [-np.sqrt(k)*np.sin(psi), np.cos(psi)]]), 2, 0)
        md = np.moveaxis(np.array([[np.cosh(psi), np.sinh(psi)/np.sqrt(k)],
                                   [np.sqrt(k)*np.sinh(psi), np.cosh(psi)]]), 2, 0)
        if self.k1 < 0.: # defocusing quadrupole
            tmat[:,0:2,0:2] = md
            tmat[:,2:4,2:4] = mf
        else: # focusing quadrupole
            tmat[:,0:2,0:2] = mf
            tmat[:,2:4,2:4] = md
        return tmat, s
