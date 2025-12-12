# base/envelopearray.py
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
from abc import abstractmethod
from .basearray import BaseArray
from .envelope import Envelope
import numpy as np
import numpy.typing as npt

class EnvelopeArray(BaseArray):
    '''
    Base class for beam envelope array.
    '''

    @property
    @abstractmethod
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        Nx4x4 array of 4D positive-definite covariance matrices with the determinant of unity.
        '''
        pass

    @property
    @abstractmethod
    def T(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 coordinate transformation matrices for eigenmode.
        '''
        pass

    @property
    @abstractmethod
    def U(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 array of 2D positive-definite covariance matrices for eigenmode U with the determinant of unity.
        '''
        pass

    @property
    @abstractmethod
    def V(self) -> npt.NDArray[np.floating]:
        '''
        Nx2x2 array of 2D positive-definite covariance matrices for eigenmode V with the determinant of unity.
        '''
        pass

    @property
    @abstractmethod
    def tau(self) -> npt.NDArray[np.floating]:
        '''
        Array of coupling parameters tau with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def bx(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal beta function array [m] with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def ax(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal alpha function array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def gx(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal gamma function array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def by(self) -> npt.NDArray[np.floating]:
        '''
        Vertical beta function array [m] with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def ay(self) -> npt.NDArray[np.floating]:
        '''
        Vertical alpha function array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def gy(self) -> npt.NDArray[np.floating]:
        '''
        Vertical gamma function array with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def bu(self) -> npt.NDArray[np.floating]:
        '''
        Beta function array for eigenmode U [m] with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def au(self) -> npt.NDArray[np.floating]:
        '''
        Alpha function array for eigenmode U with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def gu(self) -> npt.NDArray[np.floating]:
        '''
        Gamma function array for eigenmode U with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def bv(self) -> npt.NDArray[np.floating]:
        '''
        Beta function array for eigenmode V [m] with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def av(self) -> npt.NDArray[np.floating]:
        '''
        Alpha function array for eigenmode V with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def gv(self) -> npt.NDArray[np.floating]:
        '''
        Gamma function array for eigenmode V with shape (N,).
        '''
        pass

    @property
    @abstractmethod
    def psix(self) -> npt.NDArray[np.floating]:
        '''
        Horizontal batatron phase array [rad] with shape (N,). (Eigenmode U)
        '''
        pass

    @property
    @abstractmethod
    def psiy(self) -> npt.NDArray[np.floating]:
        '''
        Vertical betatron phase array [rad] with shape (N,). (Eigenmode V)
        '''
        pass

    @abstractmethod
    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None) -> None:
        '''
        Calculate eigenmode.

        Args:
            T npt.NDArray[np.floating]: Nx2x2 coordinate transformation matrices for eigenmode. (Optional)
        '''
        pass

    @abstractmethod
    def copy(self) -> EnvelopeArray:
        '''
        Returns:
            EnvelopeArray: A copy of the envelope array object.
        '''
        pass

    @abstractmethod
    def append(self, evlp: EnvelopeArray) -> None:
        '''
        Append another envelope array to this one.

        Args:
            evlp EnvelopeArray: Another envelope array to append.
        '''
        pass

    @classmethod
    @abstractmethod
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
        pass

    @abstractmethod
    def from_s(self, s: float) -> Envelope:
        '''
        Get the envelope at the specified longitudinal position by linear interpolation.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Envelope: Interpolated envelope at the specified longitudinal position.
        '''
        pass

    @abstractmethod
    def T_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Get the coordinate transformation matrices for eigenmode.

        Returns:
            npt.NDArray[np.floating]: Nx4x4 coordinate transformation matrices for eigenmode.
        '''
        pass
