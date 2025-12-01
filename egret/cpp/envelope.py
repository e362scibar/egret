# cpp/envelope.py
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
from egret.cppegret import Envelope as EnvelopeCPP
import numpy as np
import numpy.typing as npt

class Envelope(EnvelopeABC):
    '''
    Class for beam envelope object.
    '''

    def __init__(self, cov: npt.NDArray[np.floating], s: float, **kwargs):
        '''
        Args:
            cov npt.NDArray[np.floating]: 4x4 covariance matrix.
            s float: Longitudinal position.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = EnvelopeCPP(cov, s)

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
                return self.instance.bx
            case 'ax':
                return self.instance.ax
            case 'gx':
                return self.instance.gx
            case 'by':
                return self.instance.by
            case 'ay':
                return self.instance.ay
            case 'gy':
                return self.instance.gy
            case 'bu':
                return self.instance.bu
            case 'au':
                return self.instance.au
            case 'gu':
                return self.instance.gu
            case 'bv':
                return self.instance.bv
            case 'av':
                return self.instance.av
            case 'gv':
                return self.instance.gv
            case 's':
                return self.instance.s
            case _:
                raise KeyError(f'Invalid key: {key}')

    @property
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        Get the 4x4 covariance matrix.

        Returns:
            npt.NDArray[np.floating]: 4x4 covariance matrix.
        '''
        return self.instance.cov

    @property
    def s(self) -> float:
        '''
        Get the longitudinal position.

        Returns:
            float: Longitudinal position.
        '''
        return self.instance.s

    @property
    def T(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode transformation matrix.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode transformation matrix.
        '''
        return self.instance.T

    @property
    def U(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode covariance matrix U.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode covariance matrix U.
        '''
        return self.instance.U

    @property
    def V(self) -> npt.NDArray[np.floating]:
        '''
        Get the 2x2 eigenmode covariance matrix V.

        Returns:
            npt.NDArray[np.floating]: 2x2 eigenmode covariance matrix V.
        '''
        return self.instance.V

    @property
    def tau(self) -> float:
        '''
        Get the eigenmode coupling parameter tau.

        Returns:
            float: Eigenmode coupling parameter tau.
        '''
        return self.instance.tau

    def calc_eigenmode(self, T: npt.NDArray[np.floating] = None):
        '''
        Calculate eigenmode transformation matrix.

        Args:
            T npt.NDArray[np.floating]: 4x4 coordinate transformation matrix for eigenmode. (Optional)
        '''
        self.instance.calc_eigenmode(T)

    def copy(self):
        '''
        Create a copy of the envelope.
        '''
        return Envelope(cov=self.instance.cov, s=self.instance.s)

    def transfer(self, tmat: npt.NDArray[np.floating], length: float) -> None:
        '''
        Transfer the envelope using the given transfer matrix.

        Args:
            tmat npt.NDArray[np.floating]: 4x4 transfer matrix.
            length float: Length of the element [m].
        '''
        self.instance.transfer(tmat, length)

    def T_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Get the 4x4 transformation matrix for eigenmode.

        Returns:
            npt.NDArray[np.floating]: 4x4 transformation matrix.
        '''
        return self.instance.T_matrix()
