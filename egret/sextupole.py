# sextupole.py
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
from .drift import Drift
from .quadrupole import Quadrupole

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Sextupole(Element):
    '''
    Sextupole magnet.
    '''

    def __init__(self, name: str, length: float, k2: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = '', dxp: float = 0., dyp: float = 0.):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            k2 float: Normalized sextupole strength [1/m^3].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
            dxp float: Horizontal kick angle of the steering coil [rad].
            dyp float: Vertical kick angle of the steering coil [rad].
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k2 = k2
        self.dxp = dxp
        self.dyp = dyp

    def copy(self) -> Sextupole:
        '''
        Return a copy of the sextupole.

        Returns:
            Sextupole: Copy of the sextupole.
        '''
        return Sextupole(self.name, self.length, self.k2,
                         self.dx, self.dy, self.ds,
                         self.tilt, self.info)

    def transfer_matrix(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the sextupole calculated by RK4 method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        k0x = self.dxp / self.length
        k0y = self.dyp / self.length
        x0, y0, xp0, yp0 = cood0['x'], cood0['y'], cood0['xp'], cood0['yp']
        for s in np.arange(0, self.length, s_step):
            # dipole strength at the entrance (x'+jy' = k0 L)
            k0 = self.k2 * (- 0.5 * (x0**2 - y0**2) + 1.j * x0 * y0) + k0x + 1.j * k0y
            # quadrupole strength at the entrance
            k1 = self.k2 * (x0 - 1.j * y0)
            # tilt angle of the quadrupole
            tilt = np.angle(k1) * 0.5
            # transverse offset to generate dipole kick
            offset = np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            # get first quad
            quad1 = Quadrupole(self.name+'_quad1h', s_step*0.5, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            # 1st half step
            cood1 = quad1.transfer(Coordinate(0., xp0, 0., yp0))
            # dipole strength at the middle 1
            x1, y1, xp1, yp1 = cood1['x']+x0, cood1['y']+y0, cood1['xp'], cood1['yp']
            k0 = self.k2 * (- 0.5 * (x1**2 - y1**2) + 1.j * x1 * y1)
            k1 = self.k2 * (x1 - 1.j * y1)
            tilt = np.angle(k1) * 0.5
            offset = np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            quad2 = Quadrupole(self.name+'_quad2', s_step*0.5, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            cood2 = quad2.transfer(Coordinate(0., xp0, 0., yp0))
            # dipole strength at the middle 2
            x2, y2, xp2, yp2 = cood2['x']+x0, cood2['y']+y0, cood2['xp'], cood2['yp']
            k0 = self.k2 * (- 0.5 * (x2**2 - y2**2) + 1.j * x2 * y2)
            k1 = self.k2 * (x2 - 1.j * y2)
            tilt = np.angle(k1) * 0.5
            offset = np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            quad3 = Quadrupole(self.name+'_quad3', s_step*0.5, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            cood3 = quad3.transfer(Coordinate(0., xp2, 0., yp2))
            # dipole strength at the exit
            x3, y3, xp3, yp3 = cood3['x']+x0, cood3['y']+y0, cood3['xp'], cood3['yp']
            k0 = self.k2 * (- 0.5 * (x3**2 - y3**2) + 1.j * x3 * y3)
            k1 = self.k2 * (x3 - 1.j * y3)
            tilt = np.angle(k1) * 0.5
            offset = np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            quad4 = Quadrupole(self.name+'_quad4', s_step, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            cood4 = quad4.transfer(Coordinate(0., xp3, 0., yp3))
            # update for next step
            x0, y0, xp0, yp0 = cood4['x']+x0, cood4['y']+y0, cood4['xp'], cood4['yp']


        # temporarily set to drift
        return Drift.transfer_matrix_from_length(self.length)

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal position array of shape (N,).
        '''
        # temporarily set to drift
        return Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)
