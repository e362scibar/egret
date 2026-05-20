# python/envelopearray.py
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
from ..base.envelopearray import EnvelopeArray as EnvelopeArrayABC
from .basearray import BaseArray
from .envelope import Envelope
import numpy as np
import numpy.typing as npt
import scipy

class EnvelopeArray(EnvelopeArrayABC, BaseArray):
    '''
    Beam envelope array class.
    '''

    def __init__(self, cov: npt.NDArray[np.floating], s: npt.NDArray[np.floating], T: npt.NDArray[np.floating] = None,
                 psix: npt.NDArray[np.floating] = None, psiy: npt.NDArray[np.floating] = None) -> None:
        '''
        Args:
            cov npt.NDArray[np.floating]: Nx4x4 positive-definite covariance matrices with the determinant of unity.
            s npt.NDArray[np.floating]: Longitudinal positions [m] with shape (N,).
            T npt.NDArray[np.floating]: Nx2x2 coordinate transformation matrices for eigenmode. (Optional)
            psix npt.NDArray[np.floating]: Horizontal phase advance array [rad] with shape (N,). (Optional)
            psiy npt.NDArray[np.floating]: Vertical phase advance array [rad] with shape (N,). (Optional)
        '''
        super().__init__(s)
        self._cov = cov.copy()
        self._psix = psix
        self._psiy = psiy
        self.calc_eigenmode(T)

    @property
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        Nx4x4 array of 4D positive-definite covariance matrices with the determinant of unity.
        '''
        return self._cov

    @property
    def T(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 coordinate transformation matrices for eigenmode.
        '''
        return self._T

    @property
    def U(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 array of 2D positive-definite covariance matrices for eigenmode U with the determinant of unity.
        '''
        return self._U

    @property
    def V(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 array of 2D positive-definite covariance matrices for eigenmode V with the determinant of unity.
        '''
        return self._V

    @property
    def tau(self) -> npt.NDArray[np.floating]:
        '''
        Array of coupling parameters tau with shape (N,).
        '''
        return self._tau

    @property
    def bx(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal beta function array [m] with shape (N,).
        '''
        return self._cov[:, 0, 0]

    @property
    def ax(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal alpha function array with shape (N,).
        '''
        return -0.5 * (self._cov[:, 0, 1] + self._cov[:, 1, 0])

    @property
    def gx(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal gamma function array [1/m] with shape (N,).
        '''
        return self._cov[:, 1, 1]

    @property
    def by(self) -> npt.NDArray[np.floating]:
        '''
        Vertical beta function array [m] with shape (N,).
        '''
        return self._cov[:, 2, 2]

    @property
    def ay(self) -> npt.NDArray[np.floating]:
        '''
        Vertical alpha function array with shape (N,).
        '''
        return -0.5 * (self._cov[:, 2, 3] + self._cov[:, 3, 2])

    @property
    def gy(self) -> npt.NDArray[np.floating]:
        '''
        Vertical gamma function array [1/m] with shape (N,).
        '''
        return self._cov[:, 3, 3]

    @property
    def bu(self) -> npt.NDArray[np.floating]:
        '''
        Eigenmode U beta function array [m] with shape (N,).
        '''
        return self._U[:, 0, 0]

    @property
    def au(self) -> npt.NDArray[np.floating]:
        '''
        Eigenmode U alpha function array with shape (N,).
        '''
        return -0.5 * (self._U[:, 0, 1] + self._U[:, 1, 0])

    @property
    def gu(self) -> npt.NDArray[np.floating]:
        '''
        Eigenmode U gamma function array [1/m] with shape (N,).
        '''
        return self._U[:, 1, 1]

    @property
    def bv(self) -> npt.NDArray[np.floating]:
        '''
        Eigenmode V beta function array [m] with shape (N,).
        '''
        return self._V[:, 0, 0]

    @property
    def av(self) -> npt.NDArray[np.floating]:
        '''
        Eigenmode V alpha function array with shape (N,).
        '''
        return -0.5 * (self._V[:, 0, 1] + self._V[:, 1, 0])

    @property
    def gv(self) -> npt.NDArray[np.floating]:
        '''
        Eigenmode V gamma function array [1/m] with shape (N,).
        '''
        return self._V[:, 1, 1]

    @property
    def psix(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal phase advance array [rad] with shape (N,).
        '''
        return self._psix

    @property
    def psiy(self) -> npt.NDArray[np.floating]:
        '''
        Vertical phase advance array [rad] with shape (N,).
        '''
        return self._psiy

    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None) -> None:
        '''
        Calculate eigenmode.

        Args:
            T npt.NDArray[np.floating]: Nx2x2 coordinate transformation matrices for eigenmode. (Optional)
        '''
        Sxx, Sxy, Syy = self._cov[:, 0:2, 0:2], self._cov[:, 0:2, 2:4], self._cov[:, 2:4, 2:4]
        if T is not None:
            self._T = T.copy()
        else:
            N = self._cov.shape[0]
            self._T = np.zeros((N, 2, 2), dtype=self._cov.dtype)
            # Calculate T from the covariance matrix
            zeros = np.zeros_like(self._s)
            mat = np.array([[-Sxx[:,0,0], -Sxx[:,0,1]-Syy[:,0,1], zeros, Syy[:,0,0]],
                            [zeros, -Syy[:,1,1], -Sxx[:,0,0], -Sxx[:,0,1]+Syy[:,0,1]],
                            [-Sxx[:,0,1]+Syy[:,0,1], -Sxx[:,1,1], -Syy[:,0,0], zeros],
                            [Syy[:,1,1], zeros, -Sxx[:,0,1]-Syy[:,0,1], -Sxx[:,1,1]]]).transpose(2, 0, 1)  # (N, 4, 4)
            vec = Sxy.transpose(1, 2, 0).reshape(4, N).T[:,:,np.newaxis]  # (N, 4, 1)
            try:
                res = np.linalg.solve(mat, vec)
            except np.linalg.LinAlgError:
                res = np.zeros((N, 4))
            T = res.reshape(N, 2, 2)
            self._T = T
        T_ = np.array([[T[:,1,1], -T[:,0,1]], [-T[:,1,0], T[:,0,0]]]).transpose(2,0,1)
        self._tau = np.sqrt(1. - np.linalg.det(T))
        tau = self._tau[:, np.newaxis, np.newaxis]
        sqrtchi = 1. / np.sqrt(2. * tau**2 - 1.)
        self._U = sqrtchi * (tau**2 * Sxx - np.einsum('nij,njk,nlk->nil', T_, Syy, T_))
        self._V = sqrtchi * (tau**2 * Syy - np.einsum('nij,njk,nlk->nil', T, Sxx, T))
        if self._psix is None:
            if len(self._s) < 2:
                self._psix = np.zeros_like(self._s)
                return
            beta = self._U[:,0,0]
            betap = self._U[:,0,1] + self._U[:,1,0] # beta' = -2 alpha
            f = scipy.interpolate.CubicHermiteSpline(self._s, 1./beta, -betap/beta**2)
            seg = np.array([f.integrate(self._s[i], self._s[i+1]) for i in range(len(self._s)-1)])
            self._psix = np.concatenate(([0.], np.cumsum(seg)))
        if self._psiy is None:
            if len(self._s) < 2:
                self._psiy = np.zeros_like(self._s)
                return
            beta = self._V[:,0,0]
            betap = self._V[:,0,1] + self._V[:,1,0] # beta' = -2 alpha
            f = scipy.interpolate.CubicHermiteSpline(self._s, 1./beta, -betap/beta**2)
            seg = np.array([f.integrate(self._s[i], self._s[i+1]) for i in range(len(self._s)-1)])
            self._psiy = np.concatenate(([0.], np.cumsum(seg)))

    def copy(self) -> EnvelopeArray:
        '''
        Returns:
            EnvelopeArray: A copy of the envelope array object.
        '''
        return EnvelopeArray(self._cov, self._s, self._T, self._psix, self._psiy)

    def append(self, evlp: EnvelopeArray) -> None:
        '''
        Append another envelope array to this one.

        Args:
            evlp EnvelopeArray: Another envelope array to append.
        '''
        BaseArray.append(self, evlp)
        self._cov = np.concatenate((self._cov, evlp._cov))
        self._T = np.concatenate((self._T, evlp._T))
        self._tau = np.hstack((self._tau, evlp._tau))
        self._U = np.concatenate((self._U, evlp._U))
        self._V = np.concatenate((self._V, evlp._V))
        self._psix = np.hstack((self._psix, evlp._psix))
        self._psiy = np.hstack((self._psiy, evlp._psiy))

    @classmethod
    def transport(cls, evlp0: Envelope, tmat: npt.NDArray[np.floating], s: npt.NDArray[np.floating]) -> EnvelopeArray:
        '''
        Transport the envelope array using the given transfer matrix.

        Args:
            evlp0 Envelope: Initial envelope.
            tmat npt.NDArray[np.floating]: Nx4x4 transfer matrices.
            s npt.NDArray[np.floating]: Longitudinal positions [m] from evlp0.s with shape (N,).

        Returns:
            EnvelopeArray: Transported envelope array.
        '''
        cov = np.einsum('nij,jk,nlk->nil', tmat, evlp0.cov, tmat)
        Mxx, Mxy, Myx, Myy = tmat[:,0:2,0:2], tmat[:,0:2,2:4], tmat[:,2:4,0:2], tmat[:,2:4,2:4]
        Mxx_ = np.array([[Mxx[:,1,1], -Mxx[:,0,1]], [-Mxx[:,1,0], Mxx[:,0,0]]]).transpose(2,0,1)
        Mxy_ = np.array([[Mxy[:,1,1], -Mxy[:,0,1]], [-Mxy[:,1,0], Mxy[:,0,0]]]).transpose(2,0,1)
        T0, tau0 = evlp0.T, evlp0.tau
        T0_ = np.array([[T0[1,1], -T0[0,1]], [-T0[1,0], T0[0,0]]])
        tauMu = tau0 * Mxx - np.matmul(Mxy, T0)
        tauMv = tau0 * Myy + np.matmul(Myx, T0_)
        tau = np.sqrt(0.5 * (np.linalg.det(tauMu) + np.linalg.det(tauMv)))
        tau_ = tau[:, np.newaxis, np.newaxis]
        Mu, Mv = tauMu / tau_, tauMv / tau_
        Mu_ = np.array([[Mu[:,1,1], -Mu[:,0,1]], [-Mu[:,1,0], Mu[:,0,0]]]).transpose(2,0,1)
        Mv_T1 = tau0 * Mxy_ + np.matmul(T0, Mxx_)
        T1Mu = -tau0 * Myx + np.matmul(Myy, T0)
        T = 0.5 * (np.matmul(Mv, Mv_T1) + np.matmul(T1Mu, Mu_))
        dpsix = np.arctan2(Mu[:,0,1], evlp0.bu*Mu[:,0,0]-evlp0.au*Mu[:,0,1])
        dpsiy = np.arctan2(Mv[:,0,1], evlp0.bv*Mv[:,0,0]-evlp0.av*Mv[:,0,1])
        psix = evlp0.psix + dpsix
        psiy = evlp0.psiy + dpsiy
        U = np.einsum('nij,jk,nlk->nil', Mu, evlp0.U, Mu)
        V = np.einsum('nij,jk,nlk->nil', Mv, evlp0.V, Mv)
        return cls(cov, evlp0.s + s, T, psix, psiy)

    def from_s(self, s: float) -> Envelope:
        '''
        Get the envelope at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Envelope: Interpolated envelope at the specified longitudinal position.
        '''
        if len(self._s) < 2:
            raise ValueError("Envelope array must have at least two elements for interpolation.")
        idx = self.index_from_s(s)
        s0, s1 = self._s[idx], self._s[idx+1]
        ds = s1 - s0
        if s < s0 - self.TOL or s > s1 + self.TOL:
            raise ValueError(f"s = {s} is out of range of s_array: [{s0}, {s1}]")
        if ds == 0.0:
            a = np.array([0.5, 0.5])
        else:
            a = np.array([(s1-s)/ds, (s-s0)/ds])
        cov = np.sum(self._cov[idx:idx+2, :, :] * a[:, np.newaxis, np.newaxis], axis=0)
        T = np.sum(self._T[idx:idx+2, :, :] * a[:, np.newaxis, np.newaxis], axis=0)
        psix = np.sum(self._psix[idx:idx+2] * a)
        psiy = np.sum(self._psiy[idx:idx+2] * a)
        return Envelope(cov, s, T, psix, psiy)

    def T_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Get the coordinate transformation matrices for eigenmode.

        Returns:
            npt.NDArray[np.floating]: Nx4x4 coordinate transformation matrices for eigenmode.
        '''
        T = self._T
        T_ = np.array([[T[:,1,1], -T[:,0,1]], [-T[:,1,0], T[:,0,0]]]).transpose(2,0,1)
        mat = np.eye(4)[np.newaxis, :, :] * self._tau[:, np.newaxis, np.newaxis]
        mat[:, 2:4, 0:2] = T
        mat[:, 0:2, 2:4] = -T_
        return mat
