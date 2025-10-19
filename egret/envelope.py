# envelope.py
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

class Envelope:
    '''
    Beam envelope object.
    '''

    def __init__(self, cov: npt.NDArray[np.floating] = np.eye(4), s: float = 0.):
        '''
        Args:
            cov npt.NDArray[np.floating]: 4x4 positive-definite covariance matrix with the determinant of unity.
            s float: Longitudinal position [m].
        '''
        self.cov = cov.copy()
        self.s = s

    def __getitem__(self, key):
        '''
        Get beta function value by key.

        Args:
            key str: Key of the beta function. 'bx', 'ax', 'gx', 'by', 'ay', or 'gy'.

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
            case 's':
                return self.s
            case _:
                raise KeyError(f'Invalid key: {key}')

    def __setitem__(self, key, value):
        '''
        Set beta function value by key.

        Args:
            key str: Key of the beta function. 'bx', 'ax', 'gx', 'by', 'ay', or 'gy'.
            value float: Value to set.
        '''
        match key:
            case 'bx':
                self.cov[0, 0] = value
                self.cov[1, 1] = (1. + self.cov[1, 0] * self.cov[0, 1]) / value
            case 'ax':
                self.cov[0, 1] = -value
                self.cov[1, 0] = -value
                self.cov[1, 1] = (1. + value**2) / self.cov[0, 0]
            case 'gx':
                self.cov[1, 1] = value
                self.cov[0, 0] = (1. + self.cov[1, 0] * self.cov[0, 1]) / value
            case 'by':
                self.cov[2, 2] = value
                self.cov[3, 3] = (1. + self.cov[3, 2] * self.cov[2, 3]) / value
            case 'ay':
                self.cov[2, 3] = -value
                self.cov[3, 2] = -value
                self.cov[3, 3] = (1. + value**2) / self.cov[2, 2]
            case 'gy':
                self.cov[3, 3] = value
                self.cov[2, 2] = (1. + self.cov[3, 2] * self.cov[2, 3]) / value
            case 's':
                self.s = value
            case _:
                raise KeyError(f'Invalid key: {key}')

    def copy(self) -> Envelope:
        '''
        Create a copy of the envelope.

        Returns:
            Envelope: A copy of the envelope object.
        '''
        return Envelope(self.cov, self.s)
