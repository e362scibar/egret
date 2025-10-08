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
    '''
    Beta function object.
    '''
    index = {'bx': 0, 'ax': 1, 'gx': 2, 'by': 3, 'ay': 4, 'gy': 5}

    def __init__(self, bx: float | npt.NDArray[np.floating] = 1., ax: float | npt.NDArray[np.floating] = 0.,
                 by: float | npt.NDArray[np.floating] = 1., ay: float | npt.NDArray[np.floating] = 0.,
                 s: float | npt.NDArray[np.floating] = 0.):
        '''
        Args:
            bx float or array-like: Horizontal beta function [m].
            ax float or array-like: Horizontal alpha function.
            by float or array-like: Vertical beta function [m].
            ay float or array-like: Vertical alpha function.
            s float or array-like: Longitudinal position [m].
        '''
        gx = (1. + ax**2) / bx
        gy = (1. + ay**2) / by
        self.vector = np.array([bx, ax, gx, by, ay, gy])
        self.s = s

    def __getitem__(self, key):
        '''
        Get beta function value by key.

        Args:
            key str: Key of the beta function. 'bx', 'ax', 'gx', 'by', 'ay', or 'gy'.
        
        Returns:
            float or npt.NDArray[np.floating]: Value of the beta function corresponding to the key.
        '''
        return self.vector[self.index[key]]

    @classmethod
    def tmat(cls, tmat: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        '''
        Compute the transformation matrix for the beta function from the 4x4 transfer matrix.
        
        Args:
            tmat (npt.NDArray[np.floating]): 4x4 or (N,4,4) transfer matrix.
        
        Returns:
            npt.NDArray[np.floating]: 6x6 or (N,6,6) transformation matrix for the beta function.
        '''
        Cx = tmat[..., 0, 0]
        Sx = tmat[..., 0, 1]
        Cpx = tmat[..., 1, 0]
        Spx = tmat[..., 1, 1]
        Cy = tmat[..., 2, 2]
        Sy = tmat[..., 2, 3]
        Cpy = tmat[..., 3, 2]
        Spy = tmat[..., 3, 3]
        tmatbx = np.array([[Cx**2, -2.*Cx*Sx, Sx**2],
                           [-Cx*Cpx, Cx*Spx+Cpx*Sx, -Sx*Spx],
                           [Cpx**2, -2.*Cpx*Spx, Spx**2]])
        tmatby = np.array([[Cy**2, -2.*Cy*Sy, Sy**2],
                           [-Cy*Cpy, Cy*Spy+Cpy*Sy, -Sy*Spy],
                           [Cpy**2, -2.*Cpy*Spy, Spy**2]])
        if Cx.ndim == 0:
            tmatb = np.zeros((6, 6))
            tmatb[0:3, 0:3] = tmatbx
            tmatb[3:6, 3:6] = tmatby
        else:
            tmatb = np.zeros((Cx.shape[0],6,6))
            tmatb[..., 0:3, 0:3] = np.moveaxis(tmatbx, 2, 0)
            tmatb[..., 3:6, 3:6] = np.moveaxis(tmatby, 2, 0)
        return tmatb

    def transfer(self, tmat: npt.NDArray[np.floating], s: float | npt.NDArray[np.floating]) -> BetaFunc:
        '''
        Compute the beta function after applying the transfer matrix.
        
        Args:
            tmat (npt.NDArray[np.floating]): 4x4 or (N,4,4) transfer matrix.
            s (float or npt.NDArray[np.floating]): Longitudinal position increment [m].
        '''
        tmat = self.tmat(tmat)
        beta = np.matmul(tmat, self.vector)
        return BetaFunc(beta[..., 0], beta[..., 1], beta[..., 3], beta[..., 4], self.s + s)

    def append(self, beta: BetaFunc):
        '''
        Append another BetaFunc object to this one.
        
        Args:
            beta (BetaFunc): Another BetaFunc object to append.
        '''
        if self.vector.ndim == 1:
            self.vector = self.vector[:, np.newaxis]
            self.s = np.array([self.s])
        if beta.vector.ndim == 1:
            self.vector = np.hstack((self.vector, beta.vector[:, np.newaxis]))
            self.s = np.hstack((self.s, np.array([beta.s])))
        else:
            self.vector = np.hstack((self.vector, beta.vector))
            self.s = np.hstack((self.s, beta.s))
