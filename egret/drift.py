# drift.py
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

from .element import Element
from .coordinate import Coordinate

import numpy as np
import numpy.typing as npt
from typing import Tuple

# Optional Numba acceleration colocated in this file
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

class Drift(Element):
    '''
    Drift space element.
    '''

    def __init__(self, name: str, length: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)

    def copy(self) -> Drift:
        '''
        Return a copy of the element.

        Returns:
            Drift: A copy of the element.
        '''
        return Drift(self.name, self.length, self.dx, self.dy, self.ds, self.tilt, self.info)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the drift class).
            ds float: Maximum step size [m] for integration. (not used in the drift class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        tmat = np.eye(4)
        tmat[0, 1] = self.length
        tmat[2, 3] = self.length
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the drift class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        # try numba accelerated kernel first
        if _NUMBA_AVAILABLE:
            try:
                tmat, s = _drift_transfer_matrix_array_numba(self.length, ds, endpoint)
                return tmat, s
            except Exception:
                pass

        # try numba-accelerated numeric kernel (or pure-python numeric helper)
        try:
            tmat, s = _drift_transfer_matrix_array(self.length, ds, endpoint)
            return tmat, s
        except Exception:
            # fall back to original implementation
            s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
            tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
            tmat[:, 0, 1] = s
            tmat[:, 2, 3] = s
            return tmat, s

    @classmethod
    def transfer_matrix_from_length(cls, length: float) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the drift space.

        Args:
            length float: Length of the drift space [m].

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        tmat = np.eye(4)
        tmat[0, 1] = length
        tmat[2, 3] = length
        return tmat

    @classmethod
    def transfer_matrix_array_from_length(cls, length: float, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the drift space.

        Args:
            length float: Length of the drift space [m].
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., length, int(length//ds) + int(endpoint) + 1, endpoint)
        tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
        tmat[:, 0, 1] = s
        tmat[:, 2, 3] = s
        return tmat, s

def _drift_transfer_matrix_array_py(length: float, ds: float, endpoint: bool):
    # create s array similar to np.linspace used in original implementation
    if ds <= 0.0:
        raise ValueError('ds must be positive')
    n = int(length // ds) + (1 if endpoint else 0) + 1
    s = np.empty(n, dtype=np.float64)
    if n == 1:
        s[0] = 0.0
    else:
        for i in range(n):
            s[i] = length * i / (n - 1)

    tmat = np.empty((n, 4, 4), dtype=np.float64)
    for i in range(n):
        # identity
        tmat[i, 0, 0] = 1.0
        tmat[i, 1, 1] = 1.0
        tmat[i, 2, 2] = 1.0
        tmat[i, 3, 3] = 1.0
        # zero others
        tmat[i, 0, 1] = s[i]
        tmat[i, 1, 0] = 0.0
        tmat[i, 0, 2] = 0.0
        tmat[i, 0, 3] = 0.0
        tmat[i, 1, 2] = 0.0
        tmat[i, 1, 3] = 0.0
        tmat[i, 2, 0] = 0.0
        tmat[i, 2, 1] = 0.0
        tmat[i, 2, 3] = s[i]
        tmat[i, 3, 0] = 0.0
        tmat[i, 3, 1] = 0.0
        tmat[i, 3, 2] = 0.0
    return tmat, s

# bind and optionally njit
_drift_transfer_matrix_array = _drift_transfer_matrix_array_py
if _NUMBA_AVAILABLE:
    try:
        _drift_transfer_matrix_array = njit(_drift_transfer_matrix_array_py, cache=True)
    except Exception:
        _drift_transfer_matrix_array = _drift_transfer_matrix_array_py

def _drift_transfer_matrix_array_py(length: float, ds: float, endpoint: bool):
    if ds <= 0.0:
        raise ValueError('ds must be positive')
    n = int(length // ds) + int(endpoint) + 1
    s = np.empty(n, dtype=np.float64)
    if n == 1:
        s[0] = 0.0
    else:
        for i in range(n):
            s[i] = length * i / (n - 1)
    tmat = np.empty((n, 4, 4), dtype=np.float64)
    for i in range(n):
        tmat[i, :, :] = np.eye(4)
        tmat[i, 0, 1] = s[i]
        tmat[i, 2, 3] = s[i]
    return tmat, s

if _NUMBA_AVAILABLE:
    try:
        _drift_transfer_matrix_array_numba = njit(_drift_transfer_matrix_array_py, cache=True)
    except Exception:
        _drift_transfer_matrix_array_numba = _drift_transfer_matrix_array_py
