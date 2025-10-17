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
from .drift import Drift
from .quadrupole import Quadrupole
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray

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
        self.set_steering(dxp, dyp)

    def copy(self) -> Sextupole:
        '''
        Return a copy of the sextupole.

        Returns:
            Sextupole: Copy of the sextupole.
        '''
        return Sextupole(self.name, self.length, self.k2,
                         self.dx, self.dy, self.ds,
                         self.tilt, self.info)

    def transfer_matrix_by_midpoint_method(self, cood0: Coordinate, ds: float = 0.1) \
        -> Tuple[npt.NDArray[np.floating], Coordinate]:
        '''
        Calculate a single step transfer matrix using the midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
            Coordinate: Final coordinate after the step.
        '''
        x0, y0, xp0, yp0 = cood0['x'], cood0['y'], cood0['xp'], cood0['yp']
        # dipole strength at the entrance (x'+jy' = k0 L)
        k0a = self.k2 * (0.5 * (x0**2 - y0**2) - 1.j * x0 * y0) + self.k0x + 1.j * self.k0y
        # quadrupole strength at the entrance
        k1a = self.k2 * (x0 - 1.j * y0)
        # tilt angle of the quadrupole
        tilt = np.angle(k1a) * 0.5
        if np.abs(k1a) < 1.e-20:
            # no quadrupole, just dipole kick
            x1, y1 = x0 + (xp0 - 0.5*k0a.real*ds) * ds, y0 + (yp0 - 0.5*k0a.imag*ds) * ds
        else:
            # transverse offset to generate dipole kick
            offset = - np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0a) / np.abs(k1a)
            # get first quad
            quad1 = Quadrupole(self.name+'_quad1', ds, np.abs(k1a), dx=offset.real, dy=offset.imag, tilt=tilt)
            # get coordinate after first quad
            cood1, _, _ = quad1.transfer(Coordinate(0., xp0, 0., yp0))
            x1, y1 = cood1['x'] + x0, cood1['y'] + y0
        # dipole strength after the first quad
        k0b = self.k2 * (0.5 * (x1**2 - y1**2) - 1.j * x1 * y1) + self.k0x + 1.j * self.k0y
        # quadrupole strength after the first quad
        k1b = self.k2 * (x1 - 1.j * y1)
        # get average dipole strength
        k0 = 0.5 * (k0a + k0b)
        # get average quadrupole strength
        k1 = 0.5 * (k1a + k1b)
        # tilt angle of the quadrupole
        tilt = np.angle(k1) * 0.5
        if np.abs(k1) < 1.e-20:
            # no quadrupole, just dipole kick
            cood2 = Coordinate(x0 + (xp0 - 0.5*k0.real*ds) * ds, xp0 - k0.real * ds,
                               y0 + (yp0 - 0.5*k0.imag*ds) * ds, yp0 - k0.imag * ds)
            tmat = Drift.transfer_matrix_from_length(ds)
        else:
            # transverse offset to generate dipole kick
            offset = - np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            # get second quad
            quad2 = Quadrupole(self.name+'_quad2', ds, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            # get coordinate after second quad
            cood2, _, _ = quad2.transfer(Coordinate(0., xp0, 0., yp0))
            cood2['x'] += x0
            cood2['y'] += y0
            # get transfer matrix of the second quad
            tmat = quad2.transfer_matrix()
        return tmat, cood2

    def transfer_matrix(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the sextupole calculated by midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        cood = cood0.copy()
        tmat = np.eye(4)
        for _ in range(n_step):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
        return tmat

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
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        s = np.linspace(0., self.length, n_step + int(endpoint), endpoint=endpoint)
        cood = cood0.copy()
        tmat = np.eye(4)
        tmat_list = [tmat.copy()]
        for _ in range(n_step - int(not endpoint)):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
            tmat_list.append(tmat.copy())
        return np.array(tmat_list), s

    def dispersion(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector at the exit of the sextupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        cood = cood0.copy()
        for _ in range(n_step):
            _, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
        return Drift.transfer_matrix_from_length(self.length) @ cood0.vector - cood.vector

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the sextupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Additive dispersion array [eta_x, eta_x', eta_y, eta_y'].
            npt.NDArray[np.floating]: Longitudinal position array [s].
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        s = np.linspace(0., self.length, n_step + int(endpoint), endpoint=endpoint)
        cood = cood0.copy()
        tmat = np.eye(4)
        cood_list = [np.zeros(4)]
        for _ in range(n_step - int(not endpoint)):
            _, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            cood_list.append(cood.vector.copy())
        tmat_drift, _ = Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)
        disp = np.matmul(tmat_drift, cood0.vector) - np.array(cood_list)
        return disp, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the sextupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration.

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood0err = Coordinate(cood0['x'] - self.dx, cood0['xp'],
                              cood0['y'] - self.dy, cood0['yp'],
                              cood0['s'] - self.ds, cood0['z'], cood0['delta'])
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        cood = cood0err.copy()
        tmat = np.eye(4)
        for _ in range(n_step):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
        cood1 = Coordinate(cood['x'] + self.dx, cood['xp'], cood['y'] + self.dy, cood['yp'],
                           cood0['s'] + self.length, cood0['z'], cood0['delta'])
        if evlp0 is not None:
            evlp = np.dot(self.envelope_transfer_matrix(tmat), evlp0.vector)
            evlp1 = Envelope(evlp[0], evlp[1], evlp[3], evlp[4], cood0['s'] + self.length)
        else:
            evlp1 = None
        if disp0 is not None:
            disp = Drift.transfer_matrix_from_length(self.length) @ cood0err.vector - cood.vector
            disp += np.dot(tmat, disp0.vector)
            disp1 = Dispersion(disp[0], disp[1], disp[2], disp[3], cood0['s'] + self.length)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate, envelope, and dispersion arrays along the sextupole.

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
        cood0err = Coordinate(cood0['x'] - self.dx, cood0['xp'],
                              cood0['y'] - self.dy, cood0['yp'],
                              cood0['s'] - self.ds, cood0['z'], cood0['delta'])
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        s = np.linspace(0., self.length, n_step + int(endpoint), endpoint=endpoint)
        cood = cood0err.copy()
        tmat = np.eye(4)
        cood_list = [cood.vector.copy()]
        tmat_list = [tmat.copy()]
        for _ in range(n_step - int(not endpoint)):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
            cood_list.append(cood.vector.copy())
            tmat_list.append(tmat.copy())
        cood_array = np.array(cood_list)
        cood_array = CoordinateArray(cood_array[:, 0] + self.dx, cood_array[:, 1],
                                     cood_array[:, 2] + self.dy, cood_array[:, 3],
                                     cood0['s'] + s, np.full_like(s, cood0['z']), np.full_like(s, cood0['delta']))
        if evlp0 is not None:
            evlp_array = np.matmul(self.envelope_transfer_matrix_array(np.array(tmat_list)), evlp0.vector)
            evlp_array = EnvelopeArray(evlp_array[:, 0], evlp_array[:, 1], evlp_array[:, 3], evlp_array[:, 4],
                                       s + cood0['s'])
        else:
            evlp_array = None
        if disp0 is not None:
            disp_array = np.matmul(np.array(tmat_list), disp0.vector)
            tmat_drift, _ = Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)
            disp_array += np.matmul(tmat_drift, cood0err.vector) - np.array(cood_list)
            disp_array = DispersionArray(disp_array[:, 0], disp_array[:, 1], disp_array[:, 2], disp_array[:, 3],
                                         s + cood0['s'])
        else:
            disp_array = None
        return cood_array, evlp_array, disp_array

    def set_steering(self, dxp: float = None, dyp: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            dxp float: Horizontal kick angle of the steering coil [rad].
            dyp float: Vertical kick angle of the steering coil [rad].
        '''
        if dxp is not None:
            self.dxp = dxp
            self.k0x = - self.dxp / self.length
        if dyp is not None:
            self.dyp = dyp
            self.k0y = - self.dyp / self.length
