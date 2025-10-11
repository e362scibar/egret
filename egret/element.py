# element.py
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

from .object import Object
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray

import numpy as np
import numpy.typing as npt
from typing import Tuple

class Element(Object):
    '''
    Base class of an accelerator element.
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
        super().__init__(name)
        self.length = length
        self.dx = dx
        self.dy = dy
        self.ds = ds
        self.tilt = tilt
        self.info = info

    def transfer_matrix(self, cood0: Coordinate = None) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        return np.eye(4)

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
        return np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0), s

    @classmethod
    def envelope_transfer_matrix(cls, tmat: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        '''
        Compute the transformation matrix for the beam envelope from the 4x4 transfer matrix.

        Args:
            tmat (npt.NDArray[np.floating]): 4x4 transfer matrix.

        Returns:
            npt.NDArray[np.floating]: 6x6 transformation matrix for the beam envelope.
        '''
        Cx = tmat[0, 0]
        Sx = tmat[0, 1]
        Cpx = tmat[1, 0]
        Spx = tmat[1, 1]
        Cy = tmat[2, 2]
        Sy = tmat[2, 3]
        Cpy = tmat[3, 2]
        Spy = tmat[3, 3]
        tmatb = np.zeros((6, 6))
        tmatb[0:3, 0:3] = np.array([[Cx**2, -2.*Cx*Sx, Sx**2],
                                    [-Cx*Cpx, Cx*Spx+Cpx*Sx, -Sx*Spx],
                                    [Cpx**2, -2.*Cpx*Spx, Spx**2]])
        tmatb[3:6, 3:6] = np.array([[Cy**2, -2.*Cy*Sy, Sy**2],
                                    [-Cy*Cpy, Cy*Spy+Cpy*Sy, -Sy*Spy],
                                    [Cpy**2, -2.*Cpy*Spy, Spy**2]])
        return tmatb

    @classmethod
    def envelope_transfer_matrix_array(cls, tmat: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        '''
        Compute the transformation matrix array for the beam envelope from the 4x4 transfer matrix array.

        Args:
            tmat npt.NDArray[np.floating]: Nx4x4 transfer matrix.

        Returns:
            npt.NDArray[np.floating]: Nx6x6 transformation matrix array for the beam envelope.
        '''
        Cx = tmat[:, 0, 0]
        Sx = tmat[:, 0, 1]
        Cpx = tmat[:, 1, 0]
        Spx = tmat[:, 1, 1]
        Cy = tmat[:, 2, 2]
        Sy = tmat[:, 2, 3]
        Cpy = tmat[:, 3, 2]
        Spy = tmat[:, 3, 3]
        tmatb = np.zeros((tmat.shape[0], 6, 6))
        tmatb[:, 0:3, 0:3] = np.moveaxis(np.array([[Cx**2, -2.*Cx*Sx, Sx**2],
                                                   [-Cx*Cpx, Cx*Spx+Cpx*Sx, -Sx*Spx],
                                                   [Cpx**2, -2.*Cpx*Spx, Spx**2]]), 2, 0)
        tmatb[:, 3:6, 3:6] = np.moveaxis(np.array([[Cy**2, -2.*Cy*Sy, Sy**2],
                                                   [-Cy*Cpy, Cy*Spy+Cpy*Sy, -Sy*Spy],
                                                   [Cpy**2, -2.*Cpy*Spy, Spy**2]]), 2, 0)
        return tmatb

    def dispersion(self, cood0: Coordinate = None) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        return np.zeros(4)

    def dispersion_array(self, cood0: Coordinate = None, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        n = int(self.length//ds) + int(endpoint) + 1
        s = np.linspace(0., self.length, n, endpoint)
        return np.zeros((4, n)), s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        if self.dx != 0. or self.dy != 0.:
            print(f'transfer: Element: {self.name}: dx={self.dx}, dy={self.dy}.')
            print(self.transfer_matrix())
        cood0err = Coordinate(cood0['x'] - self.dx, cood0['xp'],
                              cood0['y'] - self.dy, cood0['yp'],
                              cood0['s'] - self.ds, cood0['z'], cood0['delta'])
        if hasattr(self, 'elements'):
            cood = cood0err
            evlp = evlp0.copy() if evlp0 is not None else None
            disp = disp0.copy() if disp0 is not None else None
            for elem in self.elements:
                cood, evlp, disp = elem.transfer(cood, evlp, disp)
            cood1 = Coordinate(cood['x'] + self.dx, cood['xp'],
                               cood['y'] + self.dy, cood['yp'],
                               cood['s'] + self.ds, cood['z'], cood['delta'])
            disp1, evlp1 = disp, evlp
        else:
            tmat = self.transfer_matrix(cood0err)
            cood = np.dot(tmat, cood0err.vector)
            cood1 = Coordinate(cood[0] + self.dx, cood[1], cood[2] + self.dy, cood[3],
                               cood0['s'] + self.length, cood0['z'], cood0['delta'])
            if evlp0 is not None:
                evlp = np.dot(self.envelope_transfer_matrix(tmat), evlp0.vector)
                evlp1 = Envelope(evlp[0], evlp[1], evlp[3], evlp[4], cood0['s'] + self.length)
            else:
                evlp1 = None
            if disp0 is not None:
                disp = np.dot(tmat, disp0.vector) + self.dispersion(cood0err)
                disp1 = Dispersion(disp[0], disp[1], disp[2], disp[3], cood0['s'] + self.length)
            else:
                disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            CoordinateArray: Coordinate array along the element.
        '''
        if self.dx != 0. or self.dy != 0.:
            print(f'transfer_array: Element: {self.name}: dx={self.dx}, dy={self.dy}.')
            print(self.transfer_matrix())
        cood0err = Coordinate(cood0['x'] - self.dx, cood0['xp'],
                              cood0['y'] - self.dy, cood0['yp'],
                              cood0['s'] - self.ds, cood0['z'], cood0['delta'])
        if hasattr(self, 'elements'):
            cood = cood0err
            evlp = evlp0.copy() if evlp0 is not None else None
            disp = disp0.copy() if disp0 is not None else None
            cood1, evlp1, disp1 = None, None, None
            for elem in self.elements:
                coodarray, evlparray, disparray = elem.transfer_array(cood, evlp, disp, ds, False)
                if cood1 is None:
                    cood1 = coodarray
                else:
                    cood1.append(coodarray)
                if evlp0 is not None:
                    if evlp1 is None:
                        evlp1 = evlparray
                    else:
                        evlp1.append(evlparray)
                if disp0 is not None:
                    if disp1 is None:
                        disp1 = disparray
                    else:
                        disp1.append(disparray)
                cood, evlp, disp = elem.transfer(cood, evlp, disp)
            cood1 = CoordinateArray(cood1['x'] + self.dx, cood1['xp'],
                                    cood1['y'] + self.dy, cood1['yp'],
                                    cood1['s'] + self.ds, cood1['z'], cood1['delta'])
            if endpoint:
                cood1.append(CoordinateArray(np.array([cood['x'] + self.dx]), np.array([cood['xp']]),
                                             np.array([cood['y'] + self.dy]), np.array([cood['yp']]),
                                             np.array([cood0['s'] + self.length]),
                                             np.array([cood0['z']]), np.array([cood0['delta']])))
                if evlp0 is not None:
                    evlp1.append(EnvelopeArray(np.array([evlp['bx']]), np.array([evlp['ax']]),
                                               np.array([evlp['by']]), np.array([evlp['ay']]),
                                               np.array([evlp0['s'] + self.length])))
                if disp0 is not None:
                    disp1.append(DispersionArray(np.array([disp['x']]), np.array([disp['xp']]),
                                                 np.array([disp['y']]), np.array([disp['yp']]),
                                                 np.array([disp0['s'] + self.length])))
        else:
            tmat, s = self.transfer_matrix_array(cood0err, ds, endpoint)
            cood = np.matmul(tmat, cood0err.vector)
            cood1 = CoordinateArray(cood[:, 0] + self.dx, cood[:, 1], cood[:, 2] + self.dy, cood[:, 3], s + cood0['s'],
                                    np.full_like(s, cood0['z']), np.full_like(s, cood0['delta']))
            if evlp0 is not None:
                tmatb = self.envelope_transfer_matrix_array(tmat)
                evlp = np.matmul(tmatb, evlp0.vector)
                evlp1 = EnvelopeArray(evlp[:, 0], evlp[:, 1], evlp[:, 3], evlp[:, 4], s + cood0['s'])
            else:
                evlp1 = None
            if disp0 is not None:
                disp_add, _ = self.dispersion_array(cood0err, ds, endpoint)
                disp = np.matmul(tmat, disp0.vector) + disp_add.T
                disp1 = DispersionArray(disp[:, 0], disp[:, 1], disp[:, 2], disp[:, 3], s + cood0['s'])
            else:
                disp1 = None
        return cood1, evlp1, disp1

    def radiation_integrals(self, cood0: Coordinate, evlp0: Envelope, disp0: Dispersion, ds: float = 0.1) \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial envelope.
            disp0 Dispersion: Initial dispersion.
            ds float: Maximum step size [m].

        Returns:
            float, float, float: Radiation integrals I2, I4, and I5.
        '''
        return 0., 0., 0.
