# base/envelope.py
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

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Envelope:
    '''
    Base class for beam envelope object.
    '''

    @abstractmethod
    def __getitem__(self, key):
        '''
        Get beta function value by key.

        Args:
            key str: Key of the beta function. 'bx', 'ax', 'gx', 'by', 'ay', 'gy', 'bu', 'au', 'gu', 'bv', 'av', 'gv', or 's'.

        Returns:
            float: Value of the beta function corresponding to the key.
        '''
        match key:
            case 'bx':
                return self.cov[0, 0]
            case 'ax':
                return -0.5 * (self.cov[0, 1] + self.cov[1, 0])
            case 'gx':
                return self.cov[1, 1]
            case 'by':
                return self.cov[2, 2]
            case 'ay':
                return -0.5 * (self.cov[2, 3] + self.cov[3, 2])
            case 'gy':
                return self.cov[3, 3]
            case 'bu':
                return self.U[0, 0]
            case 'au':
                return -0.5 * (self.U[0, 1] + self.U[1, 0])
            case 'gu':
                return self.U[1, 1]
            case 'bv':
                return self.V[0, 0]
            case 'av':
                return -0.5 * (self.V[0, 1] + self.V[1, 0])
            case 'gv':
                return self.V[1, 1]
            case 's':
                return self.s
            case _:
                raise KeyError(f'Invalid key: {key}')

    @property
    @abstractmethod
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        Get the 4x4 covariance matrix.

        Returns:
            npt.NDArray[np.floating]: 4x4 covariance matrix.
        '''
        pass

    @property
    @abstractmethod
    def s(self) -> float:
        '''
        Get the longitudinal position.

        Returns:
            float: Longitudinal position.
        '''
        pass

    @property
    @abstractmethod
    def T(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode transformation matrix.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode transformation matrix.
        '''
        pass

    @property
    @abstractmethod
    def U(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode covariance matrix U.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode covariance matrix U.
        '''
        pass

    @property
    @abstractmethod
    def V(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode covariance matrix V.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode covariance matrix V.
        '''
        pass

    @property
    @abstractmethod
    def tau(self) -> float:
        '''
        Get the eigenmode coupling parameter tau.

        Returns:
            float: Eigenmode coupling parameter tau.
        '''
        pass

    @abstractmethod
    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None):
        '''
        Calculate eigenmode transformation matrix.

        Args:
            T npt.NDArray[np.floating]: 4x4 coordinate transformation matrix for eigenmode. (Optional)
        '''
        pass

    @abstractmethod
    def copy(self):
        '''
        Create a copy of the envelope.
        '''
        pass

    @abstractmethod
    def transfer(self, tmat: npt.NDArray[np.floating], length: float) -> None:
        '''
        Transfer the envelope using the given transfer matrix.

        Args:
            tmat npt.NDArray[np.floating]: 4x4 transfer matrix.
            length float: Length of the element [m].
        '''
        pass

    @abstractmethod
    def T_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Get the 4x4 transformation matrix for eigenmode.

        Returns:
            npt.NDArray[np.floating]: 4x4 transformation matrix.
        '''
        pass
