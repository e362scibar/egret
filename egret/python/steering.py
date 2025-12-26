# python/steering.py
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
from ..base.steering import Steering as SteeringABC
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

class Steering(SteeringABC, Element):
    '''
    Steering magnet class.
    '''

    def __init__(self, name: str, length: float,
                 kick_x: float = 0., kick_y: float = 0.,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            kick_x float: Horizontal kick angle [rad].
            kick_y float: Vertical kick angle [rad].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, 0.0, dx, dy, ds, tilt, info)
        self._kick_x = kick_x
        self._kick_y = kick_y

    @property
    def kick_x(self) -> float:
        '''
        Horizontal kick angle [rad].
        '''
        return self._kick_x

    @property
    def kick_y(self) -> float:
        '''
        Vertical kick angle [rad].
        '''
        return self._kick_y

    @property
    def kick(self) -> Tuple[float, float]:
        '''
        Return a tuple of horizontal and vertical kick angles [rad].
        '''
        return self._kick_x, self._kick_y

    @kick_x.setter
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        self._kick_x = kick_x

    @kick_y.setter
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle.

        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        self._kick_y = kick_y

    @kick.setter
    def kick(self, kick_x: float, kick_y: float) -> None:
        '''
        Set the steering angles.

        Args:
            kick_x float: Horizontal kick angle [rad].
            kick_y float: Vertical kick angle [rad].
        '''
        self._kick_x = kick_x
        self._kick_y = kick_y

    def copy(self) -> Steering:
        '''
        Return a copy of the steering magnet.

        Returns:
            Steering: Copied steering magnet.
        '''
        return Steering(self._name, self.length, self._kick_x, self._kick_y,
                        self._dx, self._dy, self._ds, self._tilt, self._info)

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
        kick_tilted = np.dot(rotmat, np.array([self.kick_x, self.kick_y])) / (1. + delta)
        return kick_tilted[0], kick_tilted[1]

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the steering magnet).
            ds float: Maximum step size [m] for integration. (not used in the steering magnet).
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the steering magnet).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        return Drift.transfer_matrix_from_length(self._length)

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the steering magnet).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the steering magnet).

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        return Drift.transfer_matrix_array_from_length(self._length, ds, endpoint)

    def dispersion(self, cood0: Coordinate, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Additive dispersion function at the end of the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate. (Only delta is used here.)
            ds float: Maximum step size for integration [m]. (Not used in this method.)
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Steering class.)

        Returns:
            npt.NDArray[np.floating]: Additive dispersion function [eta_x, eta_x', eta_y, eta_y'].
        '''
        kick_x, kick_y = self._tilted_kick(cood0.delta)
        eta_xp, eta_yp = -kick_x, -kick_y
        eta_x = -0.5 * kick_x * self._length
        eta_y = -0.5 * kick_y * self._length
        return np.array([eta_x, eta_xp, eta_y, eta_yp])

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion function along the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Steering class.)

        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = self.s_array(ds, endpoint)
        kick_x, kick_y = self._tilted_kick(cood0.delta)
        if np.abs(self._length) > 0.:
            eta_xp = -kick_x * s / self._length
            eta_yp = -kick_y * s / self._length
            eta_x = -0.5 * kick_x * s**2 / self._length
            eta_y = -0.5 * kick_y * s**2 / self._length
        else:
            eta_xp = np.array([-kick_x])
            eta_yp = np.array([-kick_y])
            eta_x = np.array([0.])
            eta_y = np.array([0.])
        disp = np.array([eta_x, eta_xp, eta_y, eta_yp])
        return disp, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1, method: str = 'symplectic4') \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the Steering class).
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Steering class.)

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        kick_x, kick_y = self._tilted_kick(cood0.delta)
        dx, dy = 0.5 * self._length * kick_x, 0.5 * self._length * kick_y
        tmat = self.transfer_matrix(cood0, ds, method)
        cood = np.dot(tmat, cood0.vector) + np.array([dx, kick_x, dy, kick_y])
        cood1 = Coordinate(cood, cood0.s + self._length, cood0.z, cood0.delta)
        if evlp0 is not None:
            evlp1 = evlp0.copy()
            evlp1.transfer(tmat, self._length)
        else:
            evlp1 = None
        if disp0 is not None:
            disp = np.dot(tmat, disp0.vector) + self.dispersion(cood0, ds, method)
            disp1 = Dispersion(disp, disp0.s + self._length)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True, method: str = 'symplectic4') \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the steering magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Steering class.)

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        kick_x, kick_y = self._tilted_kick(cood0.delta)
        tmat, s = self.transfer_matrix_array(cood0, ds, endpoint, method)
        if np.abs(self._length) > 0.:
            x, y = 0.5 * s**2 * kick_x / self._length, 0.5 * s**2 * kick_y / self._length
            xp, yp = s * kick_x / self._length, s * kick_y / self._length
        else:
            x, y = np.array([0.]), np.array([0.])
            xp, yp = np.array([kick_x]), np.array([kick_y])
        cood = np.matmul(tmat, cood0.vector).T + np.array([x, xp, y, yp])
        cood1 = CoordinateArray(cood, s + cood0.s,
                                np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
        if evlp0 is not None:
            evlp1 = EnvelopeArray.transport(evlp0, tmat, s)
        else:
            evlp1 = None
        if disp0 is not None:
            disp_add, _ = self.dispersion_array(cood0, ds, endpoint, method)
            disp = np.matmul(tmat, disp0.vector).T + disp_add
            disp1 = DispersionArray(disp, s + disp0.s)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def set_steering(self, kick_x: float | None = None, kick_y: float | None = None) -> None:
        '''
        Set the steering angles.

        Args:
            kick_x float: Horizontal deflection angle [rad].
            kick_y float: Vertical deflection angle [rad].
        '''
        if kick_x is not None:
            self._kick_x = kick_x
        if kick_y is not None:
            self._kick_y = kick_y
