# cpp/envelopearray.py
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
from egret.cppegret import EnvelopeArray as EnvelopeArrayCPP
from .basearray import BaseArray
from .envelope import Envelope
import numpy as np
import numpy.typing as npt

class EnvelopeArray(EnvelopeArrayABC, BaseArray):
    '''
    Class for beam envelope array.
    '''

    def __init__(self, cov: npt.NDArray[np.floating], s: npt.NDArray[np.floating],
                 T: npt.NDArray[np.floating] = None, **kwargs):
        '''
        Args:
            cov npt.NDArray[np.floating]: Nx4x4 positive-definite covariance matrices with the determinant of unity.
            s npt.NDArray[np.floating]: Longitudinal positions [m] with shape (N,).
            T npt.NDArray[np.floating]: Nx2x2 coordinate transformation matrices for eigenmode. (Optional)
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            cov_list = [cov[i, :, :] for i in range(cov.shape[0])]
            T_list = [T[i, :, :] for i in range(T.shape[0])] if T is not None else None
            self.instance = EnvelopeArrayCPP(cov_list, s, T_list)
        super().__init__(None, instance=self.instance)

    @property
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        Nx4x4 array of 4D positive-definite covariance matrices with the determinant of unity.
        '''
        return np.array(self.instance.cov_array)

    @property
    def T(self) -> npt.NDArray[np.floating]:
        '''
        2x2xN coordinate transformation matrices for eigenmode.
        '''
        return np.array(self.instance.T_array)

    @property
    def U(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 array of 2D positive-definite covariance matrices for eigenmode U with the determinant of unity.
        '''
        return np.array(self.instance.U_array)

    @property
    def V(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 array of 2D positive-definite covariance matrices for eigenmode V with the determinant of unity.
        '''
        return np.array(self.instance.V_array)

    @property
    def tau(self) -> npt.NDArray[np.floating]:
        '''
        Array of coupling parameters tau with shape (N,).
        '''
        return self.instance.tau_array

    @property
    def bx(self) -> npt.NDArray[np.floating]:
        '''
        Array of horizontal beta function bx [m].
        '''
        return self.instance.bx_array

    @property
    def ax(self) -> npt.NDArray[np.floating]:
        '''
        Array of horizontal alpha function ax.
        '''
        return self.instance.ax_array

    @property
    def gx(self) -> npt.NDArray[np.floating]:
        '''
        Array of horizontal gamma function gx [1/m].
        '''
        return self.instance.gx_array

    @property
    def by(self) -> npt.NDArray[np.floating]:
        '''
        Array of vertical beta function by [m].
        '''
        return self.instance.by_array

    @property
    def ay(self) -> npt.NDArray[np.floating]:
        '''
        Array of vertical alpha function ay.
        '''
        return self.instance.ay_array

    @property
    def gy(self) -> npt.NDArray[np.floating]:
        '''
        Array of vertical gamma function gy [1/m].
        '''
        return self.instance.gy_array

    @property
    def bu(self) -> npt.NDArray[np.floating]:
        '''
        Array of eigenmode U beta function bu [m].
        '''
        return self.instance.bu_array

    @property
    def au(self) -> npt.NDArray[np.floating]:
        '''
        Array of eigenmode U alpha function au.
        '''
        return self.instance.au_array

    @property
    def gu(self) -> npt.NDArray[np.floating]:
        '''
        Array of eigenmode U gamma function gu [1/m].
        '''
        return self.instance.gu_array

    @property
    def bv(self) -> npt.NDArray[np.floating]:
        '''
        Array of eigenmode V beta function bv [m].
        '''
        return self.instance.bv_array

    @property
    def av(self) -> npt.NDArray[np.floating]:
        '''
        Array of eigenmode V alpha function av.
        '''
        return self.instance.av_array

    @property
    def gv(self) -> npt.NDArray[np.floating]:
        '''
        Array of eigenmode V gamma function gv [1/m].
        '''
        return self.instance.gv_array

    def copy(self) -> EnvelopeArray:
        '''
        Returns:
            EnvelopeArray: A copy of the envelope array object.
        '''
        return EnvelopeArray(cov=self.instance.cov_array, s=self.instance.s_array, T=self.instance.T_array)

    def append(self, evlp: EnvelopeArray) -> None:
        '''
        Append another envelope array to this one.

        Args:
            evlp EnvelopeArray: Another envelope array to append.
        '''
        self.instance.append(evlp.instance)

    @classmethod
    def transport(cls, evlp0: Envelope, tmat: npt.NDArray[np.floating], s: npt.NDArray[np.floating]) -> EnvelopeArray:
        '''
        Transport the envelope array using the given transfer matrix.

        Args:
            evlp0 Envelope: Initial envelope.
            tmat npt.NDArray[np.floating]: 4x4xN transfer matrices.
            s npt.NDArray[np.floating]: Longitudinal positions [m] from evlp0.s with shape (N,).

        Returns:
            EnvelopeArray: Transported envelope array.
        '''
        tmat_list = [tmat[i, :, :] for i in range(tmat.shape[0])]
        instance = EnvelopeArrayCPP.transport(evlp0.instance, tmat_list, s)
        return EnvelopeArray(None, None, instance=instance)

    def from_s(self, s: float) -> Envelope:
        '''
        Get the envelope at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Envelope: Interpolated envelope at the specified longitudinal position.
        '''
        instance = self.instance.from_s(s)
        return Envelope(None, None, instance=instance)

    def T_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Get the coordinate transformation matrices for eigenmode.

        Returns:
            npt.NDArray[np.floating]: 4x4xN coordinate transformation matrices for eigenmode.
        '''
        return np.array(self.instance.T_matrix())
