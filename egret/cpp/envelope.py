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

    @property
    def cov(self) -> npt.NDArray[np.floating]:
        '''
        4x4 covariance matrix.
        '''
        return self.instance.cov

    @property
    def s(self) -> float:
        '''
        Longitudinal position.
        '''
        return self.instance.s

    @property
    def T(self) -> npt.NDArray[np.floating]:
        '''
        2x2 eigenmode transformation matrix.
        '''
        return self.instance.T

    @property
    def tau(self) -> float:
        '''
        Eigenmode coupling parameter tau.
        '''
        return self.instance.tau

    @property
    def U(self) -> npt.NDArray[np.floating]:
        '''
        2x2 eigenmode covariance matrix U.
        '''
        return self.instance.U

    @property
    def V(self) -> npt.NDArray[np.floating]:
        '''
        2x2 eigenmode covariance matrix V.
        '''
        return self.instance.V

    @property
    def bx(self) -> float:
        '''
        Horizontal beta function [m].
        '''
        return self.instance.bx

    @property
    def ax(self) -> float:
        '''
        Horizontal alpha function [m].
        '''
        return self.instance.ax

    @property
    def gx(self) -> float:
        '''
        Horizontal gamma function [1/m].
        '''
        return self.instance.gx

    @property
    def by(self) -> float:
        '''
        Vertical beta function [m].
        '''
        return self.instance.by

    @property
    def ay(self) -> float:
        '''
        Vertical alpha function [m].
        '''
        return self.instance.ay

    @property
    def gy(self) -> float:
        '''
        Vertical gamma function [1/m].
        '''
        return self.instance.gy

    @property
    def bu(self) -> float:
        '''
        Eigenmode U beta function [m].
        '''
        return self.instance.bu

    @property
    def au(self) -> float:
        '''
        Eigenmode U alpha function [m].
        '''
        return self.instance.au

    @property
    def gu(self) -> float:
        '''
        Eigenmode U gamma function [1/m].
        '''
        return self.instance.gu

    @property
    def bv(self) -> float:
        '''
        Eigenmode V beta function [m].
        '''
        return self.instance.bv

    @property
    def av(self) -> float:
        '''
        Eigenmode V alpha function [m].
        '''
        return self.instance.av

    @property
    def gv(self) -> float:
        '''
        Eigenmode V gamma function [1/m].
        '''
        return self.instance.gv

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
