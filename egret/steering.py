# steering.py
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
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray
from .drift import Drift

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Steering(Element):
    '''
    Steering magnet.
    '''

    def __init__(self, name: str, length: float, dxp: float = 0., dyp: float = 0.,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            dxp float: Horizontal deflection angle [rad].
            dyp float: Vertical deflection angle [rad].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.dxp = dxp
        self.dyp = dyp

    def copy(self) -> Steering:
        '''
        Return a copy of the steering magnet.

        Returns:
            Steering: Copied steering magnet.
        '''
        return Steering(self.name, self.length, self.dxp, self.dyp,
                        self.dx, self.dy, self.ds, self.tilt, self.info)

    def _tilted_kick(self, delta) -> Tuple[float, float]:
        '''
        Get the tilted kick angles.

        Args:
            delta float: Relative energy deviation.

        Returns:
            Tuple[float, float]: Tilted kick angles (dxp_tilted, dyp_tilted).
        '''
        cosphi, sinphi = np.cos(self.tilt), np.sin(self.tilt)
        rotmat = np.array([[cosphi, -sinphi], [sinphi, cosphi]])
        dxyp_tilted = np.dot(rotmat, np.array([self.dxp, self.dyp])) / (1. + delta)
        return dxyp_tilted[0], dxyp_tilted[1]

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the steering magnet).
            ds float: Maximum step size [m] for integration. (not used in the steering magnet).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        return Drift.transfer_matrix_from_length(self.length)

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the steering magnet).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        return Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)

    def dispersion(self, cood0: Coordinate) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion function at the end of the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: Additive dispersion function [eta_x, eta_x', eta_y, eta_y'].
        '''
        dxp, dyp = self._tilted_kick(cood0.delta)
        eta_xp, eta_yp = -dxp, -dyp
        eta_x = -0.5 * dxp * self.length
        eta_y = -0.5 * dyp * self.length
        return np.array([eta_x, eta_xp, eta_y, eta_yp])

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion function along the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint) if np.abs(self.length) > 0. else np.array([0.])
        dxp, dyp = self._tilted_kick(cood0.delta)
        if np.abs(self.length) > 0.:
            eta_xp = -dxp * s / self.length
            eta_yp = -dyp * s / self.length
            eta_x = -0.5 * dxp * s**2 / self.length
            eta_y = -0.5 * dyp * s**2 / self.length
        else:
            eta_xp = np.array([-dxp])
            eta_yp = np.array([-dyp])
            eta_x = np.array([0.])
            eta_y = np.array([0.])
        disp = np.array([eta_x, eta_xp, eta_y, eta_yp])
        return disp, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the Steering class).

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        dxp, dyp = self._tilted_kick(cood0.delta)
        dx, dy = 0.5 * self.length * dxp, 0.5 * self.length * dyp
        tmat = self.transfer_matrix(cood0)
        cood = np.dot(tmat, cood0.vector) + np.array([dx, dxp, dy, dyp])
        cood1 = Coordinate(cood, cood0.s + self.length, cood0.z, cood0.delta)
        if evlp0 is not None:
            evlp1 = evlp0.copy()
            evlp1.transfer(tmat, self.length)
        else:
            evlp1 = None
        if disp0 is not None:
            disp = np.dot(tmat, disp0.vector) + self.dispersion(cood0)
            disp1 = Dispersion(disp, disp0.s + self.length)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        dxp, dyp = self._tilted_kick(cood0.delta)
        tmat, s = self.transfer_matrix_array(cood0, ds, endpoint)
        if np.abs(self.length) > 0.:
            x, y = 0.5 * s**2 * dxp / self.length, 0.5 * s**2 * dyp / self.length
            xp, yp = s * dxp / self.length, s * dyp / self.length
        else:
            x, y = np.array([0.]), np.array([0.])
            xp, yp = np.array([dxp]), np.array([dyp])
        cood = np.matmul(tmat.transpose(2,0,1), cood0.vector).T + np.array([x, xp, y, yp])
        cood1 = CoordinateArray(cood, s + cood0.s,
                                np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
        if evlp0 is not None:
            evlp1 = EnvelopeArray.transport(evlp0, tmat, s)
        else:
            evlp1 = None
        if disp0 is not None:
            disp_add, _ = self.dispersion_array(cood0, ds, endpoint)
            disp = np.matmul(tmat.transpose(2,0,1), disp0.vector).T + disp_add
            disp1 = DispersionArray(disp, s + disp0.s)
        else:
            disp1 = None
        return cood1, evlp1, disp1
