# envelopearray.py
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

import numpy as np
import numpy.typing as npt

class EnvelopeArray:
    '''
    Beam envelope array.
    '''

    def __init__(self, cov: npt.NDArray[np.floating], s: npt.NDArray[np.floating], T: npt.NDArray[np.floating] = None):
        '''
        Args:
            cov npt.NDArray[np.floating]: 4x4xN positive-definite covariance matrices with the determinant of unity.
            s npt.NDArray[np.floating]: Longitudinal positions [m] with shape (N,).
            T npt.NDArray[np.floating]: 2x2xN coordinate transformation matrices for eigenmode. (Optional)
        '''
        self.set(cov, s, T)

    def __getitem__(self, key: str) -> npt.NDArray[np.floating]:
        '''
        Get beam envelope value by key.

        Args:
            key str: Key of the coordinate. 'bx', 'ax', 'gx', 'by', 'ay', 'gy', 'bu', 'au', 'gu', 'bv', 'av', 'gv', or 's'.

        Returns:
            NDArray: Value of the coordinate corresponding to the key.
        '''
        match key:
            case 'bx':
                return self.cov[0, 0, :]
            case 'ax':
                return -0.5 * (self.cov[0, 1, :] + self.cov[1, 0, :])
            case 'gx':
                return self.cov[1, 1, :]
            case 'by':
                return self.cov[2, 2, :]
            case 'ay':
                return -0.5 * (self.cov[2, 3, :] + self.cov[3, 2, :])
            case 'gy':
                return self.cov[3, 3, :]
            case 'bu':
                return self.U[0, 0, :]
            case 'au':
                return -0.5 * (self.U[0, 1, :] + self.U[1, 0, :])
            case 'gu':
                return self.U[1, 1, :]
            case 'bv':
                return self.V[0, 0, :]
            case 'av':
                return -0.5 * (self.V[0, 1, :] + self.V[1, 0, :])
            case 'gv':
                return self.V[1, 1, :]
            case 's':
                return self.s
            case _:
                raise KeyError(f'Invalid key: {key}')

    def set(self, cov: npt.NDArray[np.floating], s: npt.NDArray[np.floating], T: npt.NDArray[np.floating] = None) -> None:
        '''
        Set beam envelope parameters.

        Args:
            cov npt.NDArray[np.floating]: 4x4xN positive-definite covariance matrices with the determinant of unity.
            s npt.NDArray[np.floating]: Longitudinal positions [m] with shape (N,).
            T npt.NDArray[np.floating]: 2x2xN coordinate transformation matrices for eigenmode. (Optional)
        '''
        self.cov = cov.copy()
        self.s = s.copy()
        self.calc_eigenmode(T)

    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None) -> None:
        '''
        Calculate eigenmode.

        Args:
            T npt.NDArray[np.floating]: 2x2xN coordinate transformation matrices for eigenmode. (Optional)
        '''
        Sxx, Sxy, Syy = self.cov[0:2, 0:2, :], self.cov[0:2, 2:4, :], self.cov[2:4, 2:4, :]
        if T is not None:
            self.T = T.copy()
        else:
            N = self.cov.shape[2]
            self.T = np.zeros((2, 2, N), dtype=self.cov.dtype)
            # Calculate T from the covariance matrix
            mat = np.array([[-Sxx[0,0,:], -Sxx[0,1,:]-Syy[0,1,:], 0., Syy[0,0,:]],
                            [0., -Syy[1,1,:], -Sxx[0,0,:], -Sxx[0,1,:]+Syy[0,1,:]],
                            [-Sxx[0,1,:]+Syy[0,1,:], -Sxx[1,1,:], -Syy[0,0,:], 0.],
                            [Syy[1,1,:], 0., -Sxx[0,1,:]-Syy[0,1,:], -Sxx[1,1,:]]]).transpose(2, 0, 1)  # (N, 4, 4)
            vec = Sxy.reshape(4, N).T  # (N, 4)
            try:
                res = np.linalg.solve(mat, vec)
            except np.linalg.LinAlgError:
                res = np.zeros((N, 4))
            T = res.T.reshape(2, 2, N)
            self.T = T
        T_ = np.array([[T[1,1,:], -T[0,1,:]], [-T[1,0,:], T[0,0,:]]])
        tau = np.sqrt(1. - np.linalg.det(T.transpose(2,0,1)))
        chi = 1. / (2. * tau**2 - 1.)
        sqrtchi = np.sqrt(chi)
        self.U = sqrtchi * (tau**2 * Sxx - np.matmul(T_.transpose(2, 0, 1), np.matmul(Syy.transpose(2, 0, 1), T_.transpose(2, 0, 1)))).transpose(1,2,0)
        self.V = sqrtchi * (tau**2 * Syy - np.matmul(T.transpose(2, 0, 1), np.matmul(Sxx.transpose(2, 0, 1), T.transpose(2, 0, 1)))).transpose(1,2,0)

    def copy(self) -> EnvelopeArray:
        '''
        Returns:
            EnvelopeArray: A copy of the envelope array object.
        '''
        return EnvelopeArray(self.cov, self.s, self.T)

    def append(self, evlp: EnvelopeArray) -> None:
        '''
        Append another envelope array to this one.

        Args:
            evlp EnvelopeArray: Another envelope array to append.
        '''
        self.cov = np.dstack((self.cov, evlp.cov))
        self.s = np.hstack((self.s, evlp.s))
