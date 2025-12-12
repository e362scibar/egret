# python/envelope.py
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
from ..base.envelope import Envelope as EnvelopeABC
import numpy as np
import numpy.typing as npt

class Envelope(EnvelopeABC):
    '''
    Beam envelope class.
    '''

    def __init__(self, cov: npt.NDArray[np.floating] = np.eye(4), s: float = 0.,
                 T: npt.NDArray[np.floating] = None, psix: float = 0., psiy: float = 0.):
        '''
        Args:
            cov npt.NDArray[np.floating]: 4x4 positive-definite covariance matrix with the determinant of unity.
            s float: Longitudinal position [m].
            T npt.NDArray[np.floating]: 2x2 coordinate transformation matrix for eigenmode. (Optional)
            psix float: Betatron phase in horizontal plane [rad]. (Optional)
            psiy float: Betatron phase in vertical plane [rad]. (Optional)
        '''
        self._cov = cov.copy()
        self._s = s
        self._psix = psix
        self._psiy = psiy
        self.calc_eigenmode(T)

    @property
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        Get the 4x4 covariance matrix.

        Returns:
            npt.NDArray[np.floating]: 4x4 positive-definite covariance matrix with the determinant of unity.
        '''
        return self._cov

    @property
    def s(self) -> float:
        '''
        Get the longitudinal position.

        Returns:
            float: Longitudinal position [m].
        '''
        return self._s

    @property
    def T(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode transformation matrix.

        Returns:
            npt.NDArray[np.floating]: 2x2 coordinate transformation matrix for eigenmode.
        '''
        return self._T

    @property
    def U(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode covariance matrix U.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode covariance matrix U.
        '''
        return self._U

    @property
    def V(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode covariance matrix V.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode covariance matrix V.
        '''
        return self._V

    @property
    def tau(self) -> float:
        '''
        Get the tau parameter for eigenmode calculation.

        Returns:
            float: Tau parameter for eigenmode calculation.
        '''
        return self._tau

    @property
    def bx(self) -> float:
        '''
        Get the horizontal beta function.

        Returns:
            float: Horizontal beta function.
        '''
        return self._cov[0, 0]

    @property
    def ax(self) -> float:
        '''
        Get the horizontal alpha function.

        Returns:
            float: Horizontal alpha function.
        '''
        return -0.5 * (self._cov[0, 1] + self._cov[1, 0])

    @property
    def gx(self) -> float:
        '''
        Get the horizontal gamma function.

        Returns:
            float: Horizontal gamma function.
        '''
        return self._cov[1, 1]

    @property
    def by(self) -> float:
        '''
        Get the vertical beta function.

        Returns:
            float: Vertical beta function.
        '''
        return self._cov[2, 2]

    @property
    def ay(self) -> float:
        '''
        Get the vertical alpha function.

        Returns:
            float: Vertical alpha function.
        '''
        return -0.5 * (self._cov[2, 3] + self._cov[3, 2])

    @property
    def gy(self) -> float:
        '''
        Get the vertical gamma function.

        Returns:
            float: Vertical gamma function.
        '''
        return self._cov[3, 3]

    @property
    def bu(self) -> float:
        '''
        Get the beta function for eigenmode U.

        Returns:
            float: Beta function for eigenmode U.
        '''
        return self._U[0, 0]

    @property
    def au(self) -> float:
        '''
        Get the alpha function for eigenmode U.

        Returns:
            float: Alpha function for eigenmode U.
        '''
        return -0.5 * (self._U[0, 1] + self._U[1, 0])

    @property
    def gu(self) -> float:
        '''
        Get the gamma function for eigenmode U.

        Returns:
            float: Gamma function for eigenmode U.
        '''
        return self._U[1, 1]

    @property
    def bv(self) -> float:
        '''
        Get the beta function for eigenmode V.

        Returns:
            float: Beta function for eigenmode V.
        '''
        return self._V[0, 0]

    @property
    def av(self) -> float:
        '''
        Get the alpha function for eigenmode V.

        Returns:
            float: Alpha function for eigenmode V.
        '''
        return -0.5 * (self._V[0, 1] + self._V[1, 0])

    @property
    def gv(self) -> float:
        '''
        Get the gamma function for eigenmode V.

        Returns:
            float: Gamma function for eigenmode V.
        '''
        return self._V[1, 1]

    @property
    def psix(self) -> float:
        '''
        Get the horizontal betatron phase. (Eigenmode U)

        Returns:
            float: Horizontal betatron phase [rad].
        '''
        return self._psix

    @property
    def psiy(self) -> float:
        '''
        Get the vertical betatron phase. (Eigenmode V)

        Returns:
            float: Vertical betatron phase [rad].
        '''
        return self._psiy

    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None):
        '''
        Calculate eigenmode transformation matrix.

        Args:
            T npt.NDArray[np.floating]: 4x4 coordinate transformation matrix for eigenmode. (Optional)
        '''
        Sxx, Sxy, Syy = self.cov[0:2, 0:2], self.cov[0:2, 2:4], self.cov[2:4, 2:4]
        if T is not None:
            self._T = T.copy()
        else:
            # Calculate T from the covariance matrix
            mat = np.array([[-Sxx[0,0], -Sxx[0,1]-Syy[0,1], 0., Syy[0,0]],
                            [0., -Syy[1,1], -Sxx[0,0], -Sxx[0,1]+Syy[0,1]],
                            [-Sxx[0,1]+Syy[0,1], -Sxx[1,1], -Syy[0,0], 0.],
                            [Syy[1,1], 0., -Sxx[0,1]-Syy[0,1], -Sxx[1,1]]])
            vec = Sxy.reshape(4)
            try:
                res = np.linalg.solve(mat, vec)
            except np.linalg.LinAlgError:
                res = np.zeros(4)
            T = res.reshape(2,2)
            self._T = T
        T_ = np.array([[T[1,1], -T[0,1]], [-T[1,0], T[0,0]]])
        tau = np.sqrt(1. - np.linalg.det(T))
        chi = 1. / (2. * tau**2 - 1.)
        sqrtchi = np.sqrt(chi)
        self._U = sqrtchi * (tau**2 * Sxx - T_ @ Syy @ T_.T)
        self._V = sqrtchi * (tau**2 * Syy - T @ Sxx @ T.T)
        self._tau = tau

    def copy(self) -> Envelope:
        '''
        Create a copy of the envelope.

        Returns:
            Envelope: A copy of the envelope object.
        '''
        return Envelope(self._cov, self._s, self._T, self._psix, self._psiy)

    def transfer(self, tmat: npt.NDArray[np.floating], length: float) -> None:
        '''
        Transfer the envelope using the given transfer matrix.

        Args:
            tmat npt.NDArray[np.floating]: 4x4 transfer matrix.
            length float: Length of the element [m].
        '''
        self._cov = tmat @ self._cov @ tmat.T
        Mxx, Mxy, Myx, Myy = tmat[0:2,0:2], tmat[0:2,2:4], tmat[2:4,0:2], tmat[2:4,2:4]
        Mxx_ = np.array([[Mxx[1,1], -Mxx[0,1]], [-Mxx[1,0], Mxx[0,0]]])
        Mxy_ = np.array([[Mxy[1,1], -Mxy[0,1]], [-Mxy[1,0], Mxy[0,0]]])
        T0, tau0 = self._T, self._tau
        T0_ = np.array([[T0[1,1], -T0[0,1]], [-T0[1,0], T0[0,0]]])
        tauMu, tauMv = tau0 * Mxx - Mxy @ T0, tau0 * Myy + Myx @ T0_
        tau = np.sqrt(0.5 * (np.linalg.det(tauMu) + np.linalg.det(tauMv)))
        Mu, Mv = tauMu / tau, tauMv / tau
        Mu_ = np.array([[Mu[1,1], -Mu[0,1]], [-Mu[1,0], Mu[0,0]]])
        Mv_T1, T1Mu = tau0 * Mxy_ + T0 @ Mxx_, -tau0 * Myx + Myy @ T0
        self._T = 0.5 * (Mv @ Mv_T1 + T1Mu @ Mu_)
        self._tau = tau
        bu0, bv0 = self.bu, self.bv
        au0, av0 = self.au, self.av
        self._U = Mu @ self._U @ Mu.T
        self._V = Mv @ self._V @ Mv.T
        bu1, bv1 = self.bu, self.bv
        au1, av1 = self.au, self.av
        Au = np.array([[np.sqrt(bu1/bu0), au0*np.sqrt(bu1/bu0)],
                       [0., np.sqrt(bu0*bu1)],
                       [(au0-au1)/np.sqrt(bu0*bu1), -(1.+au0*au1)/np.sqrt(bu0*bu1)],
                       [np.sqrt(bu0/bu1), -au1*np.sqrt(bu0/bu1)]]),
        Av = np.array([[np.sqrt(bv1/bv0), av0*np.sqrt(bv1/bv0)],
                       [0., np.sqrt(bv0*bv1)],
                       [(av0-av1)/np.sqrt(bv0*bv1), -(1.+av0*av1)/np.sqrt(bv0*bv1)],
                       [np.sqrt(bv0/bv1), -av1*np.sqrt(bv0/bv1)]]),
        CSu = np.linalg.pinv(Au) @ Mu.flatten().reshape(4,1)
        CSv = np.linalg.pinv(Av) @ Mv.flatten().reshape(4,1)
        dpsix, dpsiy = np.arctan2(CSu[1], CSu[0]), np.arctan2(CSv[1], CSv[0])
        if dpsix < 0.:
            dpsix += 2. * np.pi
        if dpsiy < 0.:
            dpsiy += 2. * np.pi
        self._psix += dpsix
        self._psiy += dpsiy
        self._s += length

    def T_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Get the 4x4 transformation matrix for eigenmode.

        Returns:
            npt.NDArray[np.floating]: 4x4 transformation matrix.
        '''
        mat = np.eye(4) * self._tau
        mat[2:4,0:2] = self._T
        mat[0:2,2:4] = -np.array([[self._T[1,1], -self._T[0,1]], [-self._T[1,0], self._T[0,0]]])
        return mat
