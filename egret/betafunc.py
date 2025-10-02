# betafunc.py
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

from .object import Object

import numpy as np
import numpy.typing as npt

class BetaFunc:
    """
    Beta function object.
    """
    index = {'bx': 0, 'ax': 1, 'gx': 2, 'by': 3, 'ay': 4, 'gy': 5}

    def __init__(self, bx=1., ax=0., by=1., ay=0., s=0.):
        gx = (1. + ax**2) / bx
        gy = (1. + ay**2) / by
        self.vector = np.array([bx, ax, gx, by, ay, gy])
        self.s = s

    def __getitem__(self, key):
        return self.vector[self.index[key]]

    @classmethod
    def tmat(cls, tmat:npt.NDArray[np.floating])->npt.NDArray[np.floating]:
        Cx = tmat[...,0,0]
        Sx = tmat[...,0,1]
        Cpx = tmat[...,1,0]
        Spx = tmat[...,1,1]
        Cy = tmat[...,2,2]
        Sy = tmat[...,2,3]
        Cpy = tmat[...,3,2]
        Spy = tmat[...,3,3]
        tmatbx = np.array([[Cx**2, -2.*Cx*Sx, Sx**2],
                           [-Cx*Cpx, Cx*Spx+Cpx*Sx, -Sx*Spx],
                           [Cpx**2, -2.*Cpx*Spx, Spx**2]])
        tmatby = np.array([[Cy**2, -2.*Cy*Sy, Sy**2],
                           [-Cy*Cpy, Cx*Spy+Cpy*Sy, -Sy*Spy],
                           [Cpy**2, -2.*Cpy*Spy, Spy**2]])
        if Cx.ndim == 0:
            tmatb = np.zeros((6,6))
            tmatb[0:3,0:3] = tmatbx
            tmatb[3:6,3:6] = tmatby
        else:
            tmatb = np.zeros((Cx.shape[0],6,6))
            tmatb[...,0:3,0:3] = np.moveaxis(tmatbx, 2, 0)
            tmatb[...,3:6,3:6] = np.moveaxis(tmatby, 2, 0)
        return tmatb

    def transfer(self, tmat:npt.NDArray[np.floating], s)->BetaFunc:
        tmat = self.tmat(tmat)
        beta = np.matmul(tmat, self.vector)
        return BetaFunc(beta[...,0], beta[...,1], beta[...,3], beta[...,4], self.s+s)

    def append(self, beta:BetaFunc):
        if self.vector.ndim == 1:
            self.vector = self.vector[:,np.newaxis]
            self.s = np.array([self.s])
        if beta.vector.ndim == 1:
            self.vector = np.hstack((self.vector, beta.vector[:,np.newaxis]))
            self.s = np.hstack((self.s, np.array([beta.s])))
        else:
            self.vector = np.hstack((self.vector, beta.vector))
            self.s = np.hstack((self.s, beta.s))
