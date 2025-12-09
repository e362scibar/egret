# base/nonlinearmultipole.py
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
from ..base.nonlinearmultipole import NonlinearMultipole as NonlinearMultipoleABC
from .element import Element
from .drift import Drift
from .quadrupole import Quadrupole
from .coordinate import Coordinate
from .envelope import Envelope
from .dispersion import Dispersion
from .coordinatearray import CoordinateArray
from .envelopearray import EnvelopeArray
from .dispersionarray import DispersionArray
import numpy as np
import numpy.typing as npt
from typing import Tuple

class NonlinearMultipole(NonlinearMultipoleABC, Element):
    '''
    Base class for nonlinear multipole magnets.
    '''

    def __init__(self, name: str, length: float, kick_x: float = 0.0, kick_y: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0., info: float = '') -> None:
        '''
        Initialize nonlinear multipole magnet class.

        Args:
            name str: Element name.
            length float: Element length [m].
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
            dx float: Horizontal offset of the element center [m].
            dy float: Vertical offset of the element center [m].
            ds float: Longitudinal offset of the element center [m].
            tilt float: Tilt angle around the beam axis [rad].
            info str: Additional information.
        '''
        super().__init__(name, length, 0.0, dx, dy, ds, tilt, info)
        self._kick_x = kick_x
        self._kick_y = kick_y

    @property
    def k0x(self) -> float:
        '''
        Horizontal steering strength [1/m].
        '''
        return -self._kick_x / self._length

    @property
    def k0y(self) -> float:
        '''
        Vertical steering strength [1/m].
        '''
        return -self._kick_y / self._length

    @property
    def kick_x(self) -> float:
        '''
        Horizontal kick angle of the steering coil [rad].
        '''
        return self._kick_x

    @property
    def kick_y(self) -> float:
        '''
        Vertical kick angle of the steering coil [rad].
        '''
        return self._kick_y

    @k0x.setter
    def k0x(self, k0x: float) -> None:
        '''
        Set horizontal steering strength.

        Args:
            k0x float: Horizontal steering strength [1/m].
        '''
        self._kick_x = -k0x * self._length

    @k0y.setter
    def k0y(self, k0y: float) -> None:
        '''
        Set vertical steering strength.

        Args:
            k0y float: Vertical steering strength [1/m].
        '''
        self._kick_y = -k0y * self._length

    @kick_x.setter
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle of the steering coil.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        self._kick_x = kick_x

    @kick_y.setter
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle of the steering coil.

        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        self._kick_y = kick_y

    def set_steering(self, kick_x: float = None, kick_y: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            kick_x float: Horizontal kick angle of the steering coil [rad] or None to leave unchanged.
            kick_y float: Vertical kick angle of the steering coil [rad] or None to leave unchanged.
        '''
        if kick_x is not None:
            self._kick_x = kick_x
        if kick_y is not None:
            self._kick_y = kick_y

    def get_k(self, cood: Coordinate) -> Tuple[complex, complex]:
        '''
        Calculate quadrupole strength at given coordinate.
        x' + j y' = - k0 L - k1 L (x - j y)

        Args:
            cood Coordinate: Particle oordinate.

        Returns:
            complex: Dipole strength [1/m].
            complex: Quadrupole strength [1/m^2].
        '''
        raise NotImplementedError('get_k method is not implemented in NonlinearMultipole base class.')

    def transfer_matrix_by_midpoint_method(self, cood0: Coordinate, ds: float = 0.1,
                                           tmatflag: bool = True, dispflag: bool = False) \
        -> Tuple[npt.NDArray[np.floating], Coordinate, npt.NDArray[np.floating]]:
        '''
        Calculate a single step transfer matrix using the midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Step size [m] for integration.
            tmatflag bool: Calculate transfer matrix if true. (default: True)
            dispflag bool: Calculate additive dispersion if True. (default: False)

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix, if tmatflag is True, else None.
            Coordinate: Final coordinate after the step.
            npt.NDArray[np.floating]: Additive dispersion if dispflag is True, else None.
        '''
        x0, y0, xp0, yp0 = cood0.x, cood0.y, cood0.xp, cood0.yp
        # dipole and quadrupole strengths at the entrance
        k0a, k1a = self.get_k(cood0)
        # tilt angle of the quadrupole
        tilt = np.angle(k1a) * 0.5
        if np.abs(k1a) < 1.e-20:
            # no quadrupole, just dipole kick
            x1, y1 = x0 + (xp0 - 0.5*k0a.real*ds) * ds, y0 + (yp0 - 0.5*k0a.imag*ds) * ds
            cood1 = Coordinate(np.array([x1, xp0 - k0a.real * ds, y1, yp0 - k0a.imag * ds]),
                               cood0.s + ds, cood0.z, cood0.delta)
        else:
            # transverse offset to generate dipole kick
            # offset = - np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0a) / np.abs(k1a)
            offset = - np.exp(2.j*tilt) * np.conj(k0a) / np.abs(k1a)
            # get first quad
            quad1 = Quadrupole(self._name+'_quad1', ds, np.abs(k1a), dx=offset.real, dy=offset.imag, tilt=tilt)
            # get coordinate after first quad
            cood1, _, _ = quad1.transfer(Coordinate(np.array([0., xp0, 0., yp0]), cood0.s, cood0.z, delta=0.))
            cood1.x += x0
            cood1.y += y0
            cood1.delta = cood0.delta
        # dipole and quadrupole strengths after the first quad
        k0b, k1b = self.get_k(cood1)
        # get average dipole and quadrupole strengths
        k0, k1 = 0.5 * (k0a + k0b), 0.5 * (k1a + k1b)
        # tilt angle of the quadrupole
        tilt = np.angle(k1) * 0.5
        # calculate final coordinate after the step
        tmat, disp = None, None
        if np.abs(k1) < 1.e-20:
            # no quadrupole, just dipole kick
            cood2 = Coordinate(np.array([x0 + (xp0 - 0.5*k0.real*ds) * ds, xp0 - k0.real * ds,
                                         y0 + (yp0 - 0.5*k0.imag*ds) * ds, yp0 - k0.imag * ds]),
                               cood0.s + ds, cood0.z, cood0.delta)
            if tmatflag:
                tmat = Drift.transfer_matrix_from_length(ds)
            if dispflag:
                disp = np.array([0.5 * k0.real * ds**2, k0.real * ds, 0.5 * k0.imag * ds**2, k0.imag * ds])
        else:
            # transverse offset to generate dipole kick
            # offset = - np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            offset = - np.exp(2.j*tilt) * np.conj(k0) / np.abs(k1)
            # get second quad
            quad2 = Quadrupole(self._name+'_quad2', ds, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            # get coordinate after second quad
            cood = Coordinate(np.array([0., xp0, 0., yp0]), cood0.s, cood0.z, delta=0.)
            cood2, _, _ = quad2.transfer(cood)
            cood2.x += x0
            cood2.y += y0
            cood2.delta = cood0.delta
            # get transfer matrix and additive dispersion of the second quad
            if tmatflag:
                tmat = quad2.transfer_matrix()
            if dispflag:
                cood = Coordinate(np.array([-offset.real, xp0, -offset.imag, yp0]), cood0.s, cood0.z, delta=0.)
                disp = quad2.dispersion(cood)
        return tmat, cood2, disp

    def get_step(self, ds):
        '''
        Calculate the number of steps and step size for integration.
        
        Args:
            ds float: Maximum step size [m] for integration.
            
        Returns:
            int: Number of steps.
            float: Step size [m].
        '''
        n_step = int(np.ceil(self._length / ds))
        s_step = self._length / n_step
        return n_step, s_step

    def transfer_matrix(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the multipole magnet calculated by midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        n_step, s_step = self.get_step(ds)
        cood = cood0.copy()
        tmat = np.eye(4)
        for _ in range(n_step):
            tmat_step, cood, _ = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the multipole magnet.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal position array of shape (N,).
        '''
        n_step, s_step = self.get_step(ds)
        s = self.s_array(ds, endpoint)
        cood = cood0.copy()
        tmat = np.eye(4)
        tmat_list = [tmat.copy()]
        for _ in range(n_step - int(not endpoint)):
            tmat_step, cood, _ = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
            tmat_list.append(tmat.copy())
        return np.array(tmat_list), s

    def dispersion(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector at the exit of the multipole magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        n_step, s_step = self.get_step(ds)
        cood = cood0.copy()
        dispout = np.zeros(4)
        for _ in range(n_step):
            tmat, cood, disp = self.transfer_matrix_by_midpoint_method(cood, s_step, dispflag=True)
            dispout = np.dot(tmat, dispout) + disp
        return dispout

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the multipole magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: 4xN Additive dispersion array [eta_x, eta_x', eta_y, eta_y'].
            npt.NDArray[np.floating]: Longitudinal position array [s].
        '''
        n_step, s_step = self.get_step(ds)
        s = self.s_array(ds, endpoint)
        cood = cood0.copy()
        disp_list = [np.zeros(4)]
        for _ in range(n_step - int(not endpoint)):
            tmat, cood, disp = self.transfer_matrix_by_midpoint_method(cood, s_step, dispflag=True)
            disp_list.append(np.dot(tmat, disp_list[-1]) + disp)
        return np.array(disp_list).T, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the multipole magnet.

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
        cood = cood0.copy()
        cood.x -= self._dx
        cood.y -= self._dy
        cood.s -= self._ds
        n_step, s_step = self.get_step(ds)
        tmat = np.eye(4) if evlp0 is not None else None
        dispvec = disp0.vector.copy() if disp0 is not None else None
        for _ in range(n_step):
            tmat_step, cood, disp = self.transfer_matrix_by_midpoint_method(cood, s_step, dispflag=(disp0 is not None))
            if evlp0 is not None:
                tmat = tmat_step @ tmat
            if disp0 is not None:
                dispvec = np.dot(tmat_step, dispvec) + disp
        cood1 = cood
        cood1.x += self._dx
        cood1.y += self._dy
        cood1.s += self._ds
        if evlp0 is not None:
            evlp1 = evlp0.copy()
            evlp1.transfer(tmat, self._length)
        else:
            evlp1 = None
        if disp0 is not None:
            disp1 = Dispersion(dispvec, disp0.s + self._length)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate, envelope, and dispersion arrays along the multipole magnet.

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
        cood = cood0.copy()
        cood.x -= self._dx
        cood.y -= self._dy
        cood.s -= self._ds
        n_step, s_step = self.get_step(ds)
        s = self.s_array(ds, endpoint)
        cood_list = [cood.vector.copy()]
        tmat, tmat_list = np.eye(4), [np.eye(4)] if evlp0 is not None else None
        disp_list = [disp0.vector.copy()] if disp0 is not None else None
        for _ in range(n_step - int(not endpoint)):
            tmat_step, cood, disp = self.transfer_matrix_by_midpoint_method(cood, s_step, dispflag=(disp0 is not None))
            cood_list.append(cood.vector.copy())
            if evlp0 is not None:
                tmat = tmat_step @ tmat
                tmat_list.append(tmat.copy())
            if disp0 is not None:
                disp_list.append(np.dot(tmat_step, disp_list[-1]) + disp)
        cood_array = np.array(cood_list).T
        cood_array[0] += self._dx
        cood_array[2] += self._dy
        cood1 = CoordinateArray(cood_array, s + cood0.s + self._ds,
                                np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
        if evlp0 is not None:
            evlp1 = EnvelopeArray.transport(evlp0, np.array(tmat_list), s)
        else:
            evlp1 = None
        if disp0 is not None:
            disp1 = DispersionArray(np.array(disp_list).T, s + disp0.s)
        else:
            disp1 = None
        return cood1, evlp1, disp1
