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

class Envelope(ABC):
    '''
    Base class for beam envelope object.
    '''

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

    @property
    @abstractmethod
    def bx(self) -> float:
        '''
        Horizontal beta function [m].
        '''
        pass

    @property
    @abstractmethod
    def ax(self) -> float:
        '''
        Horizontal alpha function.
        '''
        pass

    @property
    @abstractmethod
    def gx(self) -> float:
        '''
        Horizontal gamma function [1/m].
        '''
        pass

    @property
    @abstractmethod
    def by(self) -> float:
        '''
        Vertical beta function [m].
        '''
        pass

    @property
    @abstractmethod
    def ay(self) -> float:
        '''
        Vertical alpha function.
        '''
        pass

    @property
    @abstractmethod
    def gy(self) -> float:
        '''
        Vertical gamma function [1/m].
        '''
        pass

    @property
    @abstractmethod
    def bu(self) -> float:
        '''
        Eigenmode U beta function [m].
        '''
        pass

    @property
    @abstractmethod
    def au(self) -> float:
        '''
        Eigenmode U alpha function.
        '''
        pass

    @property
    @abstractmethod
    def gu(self) -> float:
        '''
        Eigenmode U gamma function [1/m].
        '''
        pass

    @property
    @abstractmethod
    def bv(self) -> float:
        '''
        Eigenmode V beta function [m].
        '''
        pass

    @property
    @abstractmethod
    def av(self) -> float:
        '''
        Eigenmode V alpha function.
        '''
        pass

    @property
    @abstractmethod
    def gv(self) -> float:
        '''
        Eigenmode V gamma function [1/m].
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
