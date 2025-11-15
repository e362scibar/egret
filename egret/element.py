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

from __future__ import annotations

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

    def copy(self) -> Element:
        '''
        Create a copy of the element.

        Returns:
            Element: A copy of the element.
        '''
        return Element(self.name, self.length, self.dx, self.dy, self.ds, self.tilt, self.info)

    def set_index(self, index: Tuple[int, ...] | None = None) -> None:
        '''
        Set the index of the element in the lattice.

        Args:
            index Tuple[int, ...] | None: Index tuple representing the position of the element in the lattice.
        '''
        self.index = index
        if hasattr(self, 'elements'):
            for i, elem in enumerate(self.elements):
                next_index = index + (i,) if index is not None else (i,)
                elem.set_index(next_index)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        return np.eye(4)

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = np.linspace(0., self.length, int(self.length / ds) + int(endpoint) + 1, endpoint)
        return np.repeat(np.eye(4)[:,:,np.newaxis], len(s), axis=2), s

    def dispersion(self, cood0: Coordinate = None) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        return np.zeros(4)

    def dispersion_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False) \
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
        n = int(self.length / ds) + int(endpoint) + 1
        s = np.linspace(0., self.length, n, endpoint)
        return np.zeros((4, n)), s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood0err = cood0.copy()
        cood0err.vector[0] -= self.dx
        cood0err.vector[2] -= self.dy
        cood0err.s -= self.ds
        if hasattr(self, 'elements'):
            cood = cood0err
            evlp = evlp0.copy() if evlp0 is not None else None
            disp = disp0.copy() if disp0 is not None else None
            for elem in self.elements:
                cood, evlp, disp = elem.transfer(cood, evlp, disp)
            cood1, evlp1, disp1 = cood, evlp, disp
        else:
            tmat = self.transfer_matrix(cood0err)
            cood = np.dot(tmat, cood0err.vector)
            cood1 = Coordinate(cood, cood0err.s + self.length, cood0err.z, cood0err.delta)
            if evlp0 is not None:
                evlp1 = evlp0.copy()
                evlp1.transfer(tmat, self.length)
            else:
                evlp1 = None
            if disp0 is not None:
                disp = np.dot(tmat, disp0.vector) + self.dispersion(cood0err)
                disp1 = Dispersion(disp, disp0.s + self.length)
            else:
                disp1 = None
        cood1.vector[0] += self.dx
        cood1.vector[2] += self.dy
        cood1.s += self.ds
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.

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
        cood0err = cood0.copy()
        cood0err.vector[0] -= self.dx
        cood0err.vector[2] -= self.dy
        cood0err.s -= self.ds
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
            if endpoint:
                cood1.append(CoordinateArray(cood.vector[:, np.newaxis], np.array([cood.s])))
                if evlp0 is not None:
                    evlp1.append(EnvelopeArray(evlp.cov[:, :, np.newaxis], np.array([evlp.s]), evlp.T[:, :, np.newaxis]))
                if disp0 is not None:
                    disp1.append(DispersionArray(disp.vector[:, np.newaxis], np.array([disp.s])))
            cood1.vector[0] += self.dx
            cood1.vector[2] += self.dy
            cood1.s += self.ds
        else:
            tmat, s = self.transfer_matrix_array(cood0err, ds, endpoint)
            # Use compiled bulk-apply helper when available to speed up applying (4,4,N) to vectors
            try:
                from ._pyegret_bridge import apply_transfer_matrix_array
                cood = apply_transfer_matrix_array(tmat, cood0err.vector)
                # apply_transfer_matrix_array returns (4,N) for single vector or (4,M,N) for multiple
                # Ensure result has shape (4, N)
                if cood.ndim == 2 and cood.shape[0] == 4 and cood.shape[1] == s.shape[0]:
                    pass
                elif cood.ndim == 3 and cood.shape[0] == 4:
                    # multiple vectors; choose first by default to preserve original behavior
                    cood = cood[:,0,:]
                else:
                    # fallback to numpy matmul if shapes unexpected
                    cood = np.matmul(tmat.transpose(2,0,1), cood0err.vector).T
            except Exception:
                cood = np.matmul(tmat.transpose(2,0,1), cood0err.vector).T
            cood[0] += self.dx
            cood[2] += self.dy
            cood1 = CoordinateArray(cood, s + cood0.s + self.ds,
                                    np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
            if evlp0 is not None:
                evlp1 = EnvelopeArray.transport(evlp0, tmat, s)
            else:
                evlp1 = None
            if disp0 is not None:
                disp_add, _ = self.dispersion_array(cood0err, ds, endpoint)
                disp = np.matmul(tmat.transpose(2,0,1), disp0.vector).T + disp_add
                disp1 = DispersionArray(disp, s + disp0.s)
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
            Tuple[float, float, float, float, float, float]: Radiation integrals I2, I4, I5u, I5v, I4u, and I4v.
        '''
        return 0., 0., 0., 0., 0., 0.

    def get_element_from_s(self, s: float) -> Tuple[Element, float]:
        '''
        Get element and local longitudinal position by longitudinal position.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Element: Element at the specified longitudinal position.
            float: Local longitudinal position in the element [m].
        '''
        if s < 0. or s >= self.length:
            raise ValueError('Longitudinal position out of range.')
        if hasattr(self, 'elements'):
            s0 = 0.
            for elem in self.elements:
                if s < s0 + elem.length:
                    return elem.get_element_from_s(s - s0)
                s0 += elem.length
            raise ValueError('Longitudinal position out of range.')
        else:
            return self, s

    def transfer_matrix_from_s(self, s: float, cood0: Coordinate = Coordinate(), ds: float = 0.1) \
        -> npt.NDArray[np.floating]:
        '''
        Transfer matrices from the given longitudinal position to the end of the element.

        Args:
            s float: Longitudinal position [m].
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix from s to the end of the element.
        '''
        if s < 0. or s > self.length:
            raise ValueError('Longitudinal position out of range.')
        if hasattr(self, 'elements'):
            s0 = 0.
            cood = cood0.copy()
            for elem in self.elements:
                if s >= s0 and s < s0 + elem.length:
                    tmat = elem.transfer_matrix_from_s(s - s0, cood, ds)
                    coodvec = np.dot(tmat, cood.vector)
                    cood = Coordinate(coodvec, cood.s + elem.length - (s - s0), cood.z, cood.delta)
                elif s < s0:
                    tmat_elem = elem.transfer_matrix(cood, ds)
                    tmat = np.dot(tmat_elem, tmat)
                    cood, _, _ = elem.transfer(cood)
                s0 += elem.length
            return tmat
        else:
            elem = self.copy()
            elem.length -= s
            return elem.transfer_matrix(cood0, ds)
