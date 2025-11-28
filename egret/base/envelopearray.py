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
from abc import ABC, abstractmethod
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
        4x4xN array of 4D positive-definite covariance matrices with the determinant of unity.
        '''
        pass

    @property
    @abstractmethod
    def T(self) -> npt.NDArray[np.floating]:
        '''
        2x2xN coordinate transformation matrices for eigenmode.
        '''
        pass

    @property
    @abstractmethod
    def U(self) -> npt.NDArray[np.floating]:
        '''
        2x2xN array of 2D positive-definite covariance matrices for eigenmode U with the determinant of unity.
        '''
        pass

    @property
    @abstractmethod
    def V(self) -> npt.NDArray[np.floating]:
        '''
        2x2xN array of 2D positive-definite covariance matrices for eigenmode V with the determinant of unity.
        '''
        pass

    @property
    @abstractmethod
    def tau(self) -> npt.NDArray[np.floating]:
        '''
        Array of coupling parameters tau with shape (N,).
        '''
        pass

    @cov.setter
    @abstractmethod
    def cov(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 4x4xN array of 4D positive-definite covariance matrices with the determinant of unity.

        Args:
            value NDArray: 4x4xN array of covariance matrices.
        '''
        pass

    @T.setter
    @abstractmethod
    def T(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 2x2xN coordinate transformation matrices for eigenmode.

        Args:
            value NDArray: 2x2xN array of coordinate transformation matrices.
        '''
        pass

    @U.setter
    @abstractmethod
    def U(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 2x2xN array of 2D positive-definite covariance matrices for eigenmode U with the determinant of unity.

        Args:
            value NDArray: 2x2xN array of covariance matrices.
        '''
        pass

    @V.setter
    @abstractmethod
    def V(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set 2x2xN array of 2D positive-definite covariance matrices for eigenmode V with the determinant of unity.

        Args:
            value NDArray: 2x2xN array of covariance matrices.
        '''
        pass

    @tau.setter
    @abstractmethod
    def tau(self, value: npt.NDArray[np.floating]) -> None:
        '''
        Set array of coupling parameters tau with shape (N,).

        Args:
            value NDArray: Array of coupling parameters tau.
        '''
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None) -> None:
        '''
        Calculate eigenmode.

        Args:
            T npt.NDArray[np.floating]: 2x2xN coordinate transformation matrices for eigenmode. (Optional)
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
            tmat npt.NDArray[np.floating]: 4x4xN transfer matrices.
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
            npt.NDArray[np.floating]: 4x4xN coordinate transformation matrices for eigenmode.
        '''
        pass
