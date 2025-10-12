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
            tilt float: Tilt angle of the quadrupole [rad].
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

    def transfer_matrix(self, cood0: Coordinate = None) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate. (Not used in the Quadrupole class.)

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        k = np.abs(self.k1)
        tmat = np.eye(4)
        if k == 0.: # drift
            tmat[0, 1] = self.length
            tmat[2, 3] = self.length
            return tmat
        psi = np.sqrt(k) * self.length
        mf = np.array([[np.cos(psi), np.sin(psi)/np.sqrt(k)],
                       [-np.sqrt(k)*np.sin(psi), np.cos(psi)]])
        md = np.array([[np.cosh(psi), np.sinh(psi)/np.sqrt(k)],
                       [np.sqrt(k)*np.sinh(psi), np.cosh(psi)]])
        if self.k1 < 0.: # defocusing quadrupole
            tmat[0:2, 0:2] = md
            tmat[2:4, 2:4] = mf
        else: # focusing quadrupole
            tmat[0:2, 0:2] = mf
            tmat[2:4, 2:4] = md
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate. (Not used in the Quadrupole class.)
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        k = np.abs(self.k1)
        s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1, endpoint)
        tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
        if k == 0.: # drift
            tmat[:, 0, 1] = s
            tmat[:, 2, 3] = s
            return tmat, s
        psi = np.sqrt(k) * s
        mf = np.moveaxis(np.array([[np.cos(psi), np.sin(psi)/np.sqrt(k)],
                                   [-np.sqrt(k)*np.sin(psi), np.cos(psi)]]), 2, 0)
        md = np.moveaxis(np.array([[np.cosh(psi), np.sinh(psi)/np.sqrt(k)],
                                   [np.sqrt(k)*np.sinh(psi), np.cosh(psi)]]), 2, 0)
        if self.k1 < 0.: # defocusing quadrupole
            tmat[:, 0:2, 0:2] = md
            tmat[:, 2:4, 2:4] = mf
        else: # focusing quadrupole
            tmat[:, 0:2, 0:2] = mf
            tmat[:, 2:4, 2:4] = md
        return tmat, s

    def dispersion(self, cood0: Coordinate) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: 4-element dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        tmat = self.transfer_matrix() - Drift.transfer_matrix_from_length(self.length)
        return np.matmul(tmat, cood0.vector)

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
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
        tmat, s = self.transfer_matrix_array(ds=ds, endpoint=endpoint)
        tmat_drift, _ = Drift.transfer_matrix_array_from_length(self.length, ds=ds, endpoint=endpoint)
        disp = np.matmul(tmat - tmat_drift, cood0.vector).T
        return disp, s
