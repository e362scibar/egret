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

from __future__ import annotations

from .element import Element
from .coordinate import Coordinate
from .drift import Drift

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Quadrupole(Element):
    '''
    Quadrupole magnet.
    '''

    def __init__(self, name: str, length: float, k1: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: float = ''):
        '''
        Args:
            name str: Name of the quadrupole.
            length float: Length of the quadrupole [m].
            k1 float: Normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
            dx float: Horizontal offset of the quadrupole [m].
            dy float: Vertical offset of the quadrupole [m].
            ds float: Longitudinal offset of the quadrupole [m].
            tilt float: Tilt angle of the quadrupole [rad]. (+pi/4: skew quadrupole)
            info str: Additional information.
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k1 = k1

    def copy(self) -> Quadrupole:
        '''
        Return a copy of the quadrupole element.

        Returns:
            Quadrupole: Copied quadrupole element.
        '''
        return Quadrupole(self.name, self.length, self.k1,
                          self.dx, self.dy, self.ds, self.tilt, self.info)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m] for integration. (Not used in the Quadrupole class.)

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        delta = 0. if cood0 is None else cood0.delta
        k = self.k1 / (1. + delta)
        tmat = np.eye(4)
        if k == 0.: # drift
            tmat[0, 1] = self.length
            tmat[2, 3] = self.length
            return tmat
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * self.length
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        mf = np.array([[cospsi, sinpsi/sqrtk],
                       [-sqrtk*sinpsi, cospsi]])
        md = np.array([[coshpsi, sinhpsi/sqrtk],
                       [sqrtk*sinhpsi, coshpsi]])
        if k < 0.: # defocusing quadrupole
            tmat[0:2, 0:2] = md
            tmat[2:4, 2:4] = mf
        else: # focusing quadrupole
            tmat[0:2, 0:2] = mf
            tmat[2:4, 2:4] = md
        if self.tilt != 0.:
            ct = np.cos(self.tilt)
            st = np.sin(self.tilt)
            rmat = np.array([[ct, 0., st, 0.],
                             [0., ct, 0., st],
                             [-st, 0., ct, 0.],
                             [0., -st, 0., ct]])
            tmat = rmat.T @ tmat @ rmat
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate. (Not used in the Quadrupole class.)
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        delta = 0. if cood0 is None else cood0.delta
        k = self.k1 / (1. + delta)
        s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1, endpoint)
        tmat = np.repeat(np.eye(4)[:,:,np.newaxis], len(s), axis=2)
        if k == 0.: # drift
            tmat[0,1,:] = s
            tmat[2,3,:] = s
            return tmat, s
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * s
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        mf = np.array([[cospsi, sinpsi/sqrtk], [-sqrtk*sinpsi, cospsi]])
        md = np.array([[coshpsi, sinhpsi/sqrtk], [sqrtk*sinhpsi, coshpsi]])
        if k < 0.: # defocusing quadrupole
            tmat[0:2,0:2,:] = md
            tmat[2:4,2:4,:] = mf
        else: # focusing quadrupole
            tmat[0:2,0:2,:] = mf
            tmat[2:4,2:4,:] = md
        if self.tilt != 0.:
            ct = np.cos(self.tilt)
            st = np.sin(self.tilt)
            rmat = np.array([[ct, 0., st, 0.],
                             [0., ct, 0., st],
                             [-st, 0., ct, 0.],
                             [0., -st, 0., ct]])
            tmat = np.einsum('ji,jkn,kl->iln', rmat, tmat, rmat)
        return tmat, s

    def dispersion(self, cood0: Coordinate) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: 4-element dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        k = self.k1 / (1. + cood0.delta)
        disp = np.zeros(4)
        if k == 0.: # drift
            return disp
        cood0vec = cood0.vector.copy()
        cood0vec[0] -= self.dx
        cood0vec[2] -= self.dy
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * self.length
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        Mf1 = np.array([[sinpsi, -cospsi/sqrtk], [sqrtk*cospsi, sinpsi]]) * 0.5 * self.length * sqrtk
        Mf2 = np.array([[0., sinpsi/sqrtk], [sqrtk*sinpsi, 0.]]) * 0.5
        Md1 = np.array([[-sinhpsi, -coshpsi/sqrtk], [-sqrtk*coshpsi, -sinhpsi]]) * 0.5 * self.length * sqrtk
        Md2 = np.array([[0., sinhpsi/sqrtk], [-sqrtk*sinhpsi, 0.]]) * 0.5
        if k < 0.: # defocusing quadrupole
            disp[0:2] = np.dot(Md1 + Md2, cood0vec[0:2])
            disp[2:4] = np.dot(Mf1 + Mf2, cood0vec[2:4])
        else: # focusing quadrupole
            disp[0:2] = np.dot(Mf1 + Mf2, cood0vec[0:2])
            disp[2:4] = np.dot(Md1 + Md2, cood0vec[2:4])
        return disp

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1, endpoint)
        k = self.k1 / (1. + cood0.delta)
        disp = np.zeros((4, len(s)))
        if k == 0.: # drift
            return disp, s
        cood0vec = cood0.vector.copy()
        cood0vec[0] -= self.dx
        cood0vec[2] -= self.dy
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * s
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        Mf1 = np.array([[sinpsi, -cospsi/sqrtk], [sqrtk*cospsi, sinpsi]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtk
        Mf2 = np.array([[np.zeros_like(s), sinpsi/sqrtk], [sinpsi*sqrtk, np.zeros_like(s)]]) * 0.5
        Md1 = np.array([[-sinhpsi, -coshpsi/sqrtk], [-sqrtk*coshpsi, -sinhpsi]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtk
        Md2 = np.array([[np.zeros_like(s), sinhpsi/sqrtk], [-sinhpsi*sqrtk, np.zeros_like(s)]]) * 0.5
        if k < 0.: # defocusing quadrupole
            disp[0:2,:] = np.matmul((Md1 + Md2).transpose(2,0,1), cood0vec[0:2]).T
            disp[2:4,:] = np.matmul((Mf1 + Mf2).transpose(2,0,1), cood0vec[2:4]).T
        else: # focusing quadrupole
            disp[0:2,:] = np.matmul((Mf1 + Mf2).transpose(2,0,1), cood0vec[0:2]).T
            disp[2:4,:] = np.matmul((Md1 + Md2).transpose(2,0,1), cood0vec[2:4]).T
        return disp, s
