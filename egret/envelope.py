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

class Envelope:
    '''
    Beam envelope object.
    '''
    index = {'bx': 0, 'ax': 1, 'gx': 2, 'by': 3, 'ay': 4, 'gy': 5}

    def __init__(self, bx: float = 1., ax: float = 0., by: float = 1., ay: float = 0., s: float = 0.):
        '''
        Args:
            bx float: Horizontal beta function [m].
            ax float: Horizontal alpha function.
            by float: Vertical beta function [m].
            ay float: Vertical alpha function.
            s float: Longitudinal position [m].
        '''
        gx = (1. + ax**2) / bx
        gy = (1. + ay**2) / by
        self.vector = np.array([bx, ax, gx, by, ay, gy])
        self.s = s

    def __getitem__(self, key):
        '''
        Get beta function value by key.

        Args:
            key str: Key of the beta function. 'bx', 'ax', 'gx', 'by', 'ay', or 'gy'.

        Returns:
            float: Value of the beta function corresponding to the key.
        '''
        try:
            return self.vector[self.index[key]]
        except KeyError:
            match key:
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
        try:
            self.vector[self.index[key]] = value
        except KeyError:
            match key:
                case 's':
                    self.s = value
                case _:
                    raise KeyError(f'Invalid key: {key}')
        match key:
            case 'bx':
                self.vector[2] = (1. + self.vector[1]**2) / value
            case 'ax':
                self.vector[2] = (1. + value**2) / self.vector[0]
            case 'gx':
                self.vector[0] = (1. + self.vector[1]**2) / value
            case 'by':
                self.vector[5] = (1. + self.vector[4]**2) / value
            case 'ay':
                self.vector[5] = (1. + value**2) / self.vector[4]
            case 'gy':
                self.vector[3] = (1. + self.vector[4]**2) / value
            case _:
                pass

    def copy(self) -> Envelope:
        '''
        Create a copy of the envelope.

        Returns:
            Envelope: A copy of the envelope object.
        '''
        return Envelope(self.vector[0], self.vector[1], self.vector[3], self.vector[4], self.s)
