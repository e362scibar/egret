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

# Optional Numba acceleration (same-file placement)
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

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
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        # try numeric helper (numba or pure-python)
        try:
            tmat, s = _quad_transfer_matrix_array(self.k1 if cood0 is None else self.k1 / (1. + cood0.delta),
                                                  self.length, ds, endpoint, self.tilt)
            return tmat, s
        except Exception:
            # fallback to original implementation
            delta = 0. if cood0 is None else cood0.delta
            k = self.k1 / (1. + delta)
            s = np.linspace(0., self.length, int(self.length//ds)+int(endpoint)+1, endpoint)
            tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
            if k == 0.: # drift
                tmat[:, 0, 1] = s
                tmat[:, 2, 3] = s
                return tmat, s
            sqrtk = np.sqrt(np.abs(k))
            psi = sqrtk * s
            cospsi, sinpsi = np.cos(psi), np.sin(psi)
            coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
            mf = np.array([[cospsi, sinpsi/sqrtk],
                           [-sqrtk*sinpsi, cospsi]]).transpose(2, 0, 1)
            md = np.array([[coshpsi, sinhpsi/sqrtk],
                           [sqrtk*sinhpsi, coshpsi]]).transpose(2, 0, 1)
            if k < 0.: # defocusing quadrupole
                tmat[:, 0:2, 0:2] = md
                tmat[:, 2:4, 2:4] = mf
            else: # focusing quadrupole
                tmat[:, 0:2, 0:2] = mf
                tmat[:, 2:4, 2:4] = md
            if self.tilt != 0.:
                ct = np.cos(self.tilt)
                st = np.sin(self.tilt)
                rmat = np.array([[ct, 0., st, 0.],
                                 [0., ct, 0., st],
                                 [-st, 0., ct, 0.],
                                 [0., -st, 0., ct]])
                tmat = np.einsum('ji,kjl,lm->kim', rmat, tmat, rmat)
            return tmat, s

    def dispersion(self, cood0: Coordinate) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: 4-element dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        tmat = Drift.transfer_matrix_from_length(self.length) - self.transfer_matrix(cood0)
        return np.matmul(tmat, cood0.vector)

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
        tmat, s = self.transfer_matrix_array(cood0, ds, endpoint)
        tmat_drift, _ = Drift.transfer_matrix_array_from_length(self.length, ds=ds, endpoint=endpoint)
        disp = np.matmul(tmat_drift - tmat, cood0.vector).T
        return disp, s

def _quad_transfer_matrix_array_py(k: float, length: float, ds: float, endpoint: bool, tilt: float):
    if ds <= 0.0:
        raise ValueError('ds must be positive')
    n = int(length // ds) + (1 if endpoint else 0) + 1
    s = np.empty(n, dtype=np.float64)
    if n == 1:
        s[0] = 0.0
    else:
        for i in range(n):
            s[i] = length * i / (n - 1)
    tmat = np.repeat(np.eye(4)[np.newaxis,:,:], n, axis=0)
    if k == 0.:
        tmat[:, 0, 1] = s
        tmat[:, 2, 3] = s
        return tmat, s
    sqrtk = np.sqrt(np.abs(k))
    psi = sqrtk * s
    cospsi, sinpsi = np.cos(psi), np.sin(psi)
    coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
    mf = np.array([[cospsi, sinpsi/sqrtk],
                   [-sqrtk*sinpsi, cospsi]]).transpose(2, 0, 1)
    md = np.array([[coshpsi, sinhpsi/sqrtk],
                   [sqrtk*sinhpsi, coshpsi]]).transpose(2, 0, 1)
    if k < 0.:
        tmat[:, 0:2, 0:2] = md
        tmat[:, 2:4, 2:4] = mf
    else:
        tmat[:, 0:2, 0:2] = mf
        tmat[:, 2:4, 2:4] = md
    if tilt != 0.:
        ct = np.cos(tilt)
        st = np.sin(tilt)
        rmat = np.array([[ct, 0., st, 0.],
                         [0., ct, 0., st],
                         [-st, 0., ct, 0.],
                         [0., -st, 0., ct]])
        tmat = np.einsum('ij,kjl,lm->kim', rmat.T, tmat, rmat)
    return tmat, s


# bind and optionally njit
_quad_transfer_matrix_array = _quad_transfer_matrix_array_py
if _NUMBA_AVAILABLE:
    try:
        _quad_transfer_matrix_array = njit(_quad_transfer_matrix_array_py, cache=True)
    except Exception:
        _quad_transfer_matrix_array = _quad_transfer_matrix_array_py
