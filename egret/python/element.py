# python/element.py
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
from ..base.element import Element as ElementABC
from .object import Object
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

class Element(ElementABC, Object):
    '''
    Base class of an accelerator element.
    '''

    def __init__(self, name: str, length: float, angle: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Initialize the element.

        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            angle float: Bending angle of the element [rad].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        super().__init__(name)
        self._length = length
        self._angle = angle
        self._dx = dx
        self._dy = dy
        self._ds = ds
        self._tilt = tilt
        self._info = info
        self._indices = None
        self._elements = None

    @property
    def length(self) -> float:
        '''
        Length of the element [m].
        '''
        return self._length

    @property
    def angle(self) -> float:
        '''
        Bending angle of the element [rad].
        '''
        return self._angle

    @property
    def dx(self) -> float:
        '''
        Horizontal offset of the element [m].
        '''
        return self._dx

    @property
    def dy(self) -> float:
        '''
        Vertical offset of the element [m].
        '''
        return self._dy

    @property
    def ds(self) -> float:
        '''
        Longitudinal offset of the element [m].
        '''
        return self._ds

    @property
    def tilt(self) -> float:
        '''
        Tilt angle of the element [rad].
        '''
        return self._tilt

    @property
    def info(self) -> str:
        '''
        Additional information.
        '''
        return self._info

    @property
    def indices(self) -> Tuple[int, ...] | None:
        '''
        Index tuple representing the position of the element in the lattice.
        None for the top element.
        '''
        return self._indices

    @property
    def elements(self) -> List[Element] | None:
        '''
        List of elements (None for single element).
        '''
        return self._elements

    @length.setter
    def length(self, length: float) -> None:
        '''
        Set the length of the element.

        Args:
            length float: Length of the element [m].
        '''
        self._length = length

    @angle.setter
    def angle(self, angle: float) -> None:
        '''
        Set the bending angle of the element.

        Args:
            angle float: Bending angle of the element [rad].
        '''
        self._angle = angle

    @dx.setter
    def dx(self, dx: float) -> None:
        '''
        Set the horizontal offset of the element.

        Args:
            dx float: Horizontal offset of the element [m].
        '''
        self._dx = dx

    @dy.setter
    def dy(self, dy: float) -> None:
        '''
        Set the vertical offset of the element.

        Args:
            dy float: Vertical offset of the element [m].
        '''
        self._dy = dy

    @ds.setter
    def ds(self, ds: float) -> None:
        '''
        Set the longitudinal offset of the element.

        Args:
            ds float: Longitudinal offset of the element [m].
        '''
        self._ds = ds

    @tilt.setter
    def tilt(self, tilt: float) -> None:
        '''
        Set the tilt angle of the element.

        Args:
            tilt float: Tilt angle of the element [rad].
        '''
        self._tilt = tilt

    @info.setter
    def info(self, info: str) -> None:
        '''
        Set the additional information.

        Args:
            info str: Additional information.
        '''
        self._info = info

    def copy(self) -> Element:
        '''
        Create a copy of the element.

        Returns:
            Element: A copy of the element.
        '''
        return Element(self._name, self._length, self._angle, self._dx, self._dy, self._ds,
                       self._tilt, self._info)

    def set_indices(self, indices: Tuple[int, ...] | None = None) -> None:
        '''
        Set the index of the element in the lattice.

        Args:
            indices Tuple[int, ...] | None: Index tuple representing the position of the element in the lattice.
        '''
        self._indices = indices
        if self._elements is not None:
            for i, elem in enumerate(self._elements):
                next_indices = indices + (i,) if indices is not None else (i,)
                elem.set_indices(next_indices)

    @classmethod
    def s_array_from_length(cls, length: float, ds: float = 0.1, endpoint: bool = True) -> npt.NDArray[np.floating]:
        '''
        Generate longitudinal position array from length.

        Args:
            length float: Length of the element [m].
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        '''
        if length == 0.0:
            return np.array([0.])
        n = int(np.ceil(length / ds)) + int(endpoint)
        return np.linspace(0., length, n, endpoint)

    def s_array(self, ds: float = 0.1, endpoint: bool = True) -> npt.NDArray[np.floating]:
        '''
        Generate longitudinal position array of the element.

        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Longitudinal position array [m].
        '''
        return self.s_array_from_length(self._length, ds, endpoint)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.
        Start point is always the beginning of the element, and the end point is always the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        if self._elements is None:
            raise NotImplementedError('transfer_matrix method not implemented in the base class.')
        tmat = np.eye(4)
        if cood0 is not None:
            cood = cood0.copy()
            cood, _, _ = self.drift_transfer(self._ds, cood, None, None)
            cood.x -= self._dx
            cood.y -= self._dy
        else:
            cood = None
        for elem in self._elements:
            tmat_elem = elem.transfer_matrix(cood, ds, method)
            tmat = np.dot(tmat_elem, tmat)
            if cood is not None:
                cood, _, _ = elem.transfer(cood)
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = True, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.
        Start point is always the beginning of the element, and the end point is always the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        if self._elements is None:
            raise NotImplementedError('transfer_matrix_array method not implemented in the base class.')
        cood = cood0.copy() if cood0 is not None else Coordinate()
        cood, _, _ = self.drift_transfer(self._ds, cood, None, None)
        cood.x -= self._dx
        cood.y -= self._dy
        tmat = np.eye(4)
        tmat_arrays = []
        s_arrays = []
        s0 = 0.0
        for i, elem in enumerate(self._elements):
            tmat_elem, s_elem = elem.transfer_matrix_array(cood, ds, False, method)
            if tmat_elem.shape[0] != s_elem.shape[0]:
                raise ValueError(f'transfer_matrix_array: shape mismatch name={elem.name}, len(tmat)={tmat_elem.shape[0]}, len(s)={s_elem.shape[0]}')
            tmat_arrays.append(np.matmul(tmat_elem, tmat))
            s_arrays.append(s_elem + s0)
            s0 += elem.length
            tmat = elem.transfer_matrix(cood, ds) @ tmat
            if cood is not None:
                cood, _, _ = elem.transfer(cood)
        if endpoint:
            tmat_arrays.append(tmat[np.newaxis, :, :])
            s_arrays.append(np.array([s0]))
        return np.array(tmat_arrays), np.hstack(s_arrays)

    def dispersion(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the element.
        Start point is always the beginning of the element, and the end point is always the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate (optional).
            ds float: Maximum step size [m] for integration (optional).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        if self._elements is None:
            return np.zeros(4)
        cood = cood0.copy() if cood0 is not None else Coordinate()
        disp = Dispersion()
        cood, _, _ = self.drift_transfer(self._ds, cood, None, None)
        cood.x -= self._dx
        cood.y -= self._dy
        for elem in self._elements:
            cood, _, disp = elem.transfer(cood, None, disp, ds, method)
        _, _, disp = self.drift_transfer(-self._ds, None, None, disp)
        return disp.vector

    def dispersion_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the element.
        Start point is always the beginning of the element, and the end point is always the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        if self._elements is None:
            s = self.s_array(ds, endpoint)
            return np.zeros((4, len(s))), s
        cood = cood0.copy() if cood0 is not None else Coordinate()
        disp = Dispersion()
        cood, _, _ = self.drift_transfer(self._ds, cood, None, None)
        cood.x -= self._dx
        cood.y -= self._dy
        s0 = 0.
        disparray = []
        sarray = []
        for elem in self._elements:
            disp_elem, s_elem = elem.dispersion_array(cood, ds, False, method)
            tmat, _ = elem.transfer_matrix_array(cood, ds, False, method)
            disparray.append(disp_elem + np.matmul(tmat, disp.vector).T)
            sarray.append(s_elem + s0)
            s0 += elem.length
            cood, _, disp = elem.transfer(cood, None, disp, ds, method)
        if endpoint:
            disparray.append(disp.vector[:, np.newaxis])
            sarray.append(np.array([s0]))
        return np.hstack(disparray), np.hstack(sarray)

    @classmethod
    def drift_transfer(cls, length: float, cood0: Coordinate = None, evlp0: Envelope = None, disp0: Dispersion = None) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Transfer the coordinate, envelope, and dispersion through a drift space of the specified length.
        This method is used internally for longitudinal offset (ds) of an element.

        Args:
            length float: Length of the drift space [m].
            cood0 Coordinate: Initial coordinate (optional).
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).

        Returns:
            Coordinate: Coordinate after the drift space (if cood0 is provided).
            Envelope: Beam envelope after the drift space (if evlp0 is provided).
            Dispersion: Dispersion after the drift space (if disp0 is provided).
        '''
        if length == 0.:
            return cood0, evlp0, disp0
        tmat = np.eye(4)
        tmat[0, 1] = length
        tmat[2, 3] = length
        if cood0 is None:
            cood1 = None
        else:
            cood1vec = np.dot(tmat, cood0.vector)
            cood1 = Coordinate(cood1vec, cood0.s, cood0.z, cood0.delta)
        if evlp0 is None:
            evlp1 = None
        else:
            cov = tmat @ evlp0.cov @ tmat.T
            M = np.array([[1., length], [0., 1.]])
            M_ = np.array([[1., -length], [0., 1.]])
            T = M @ evlp0.T @ M_
            dpsix = np.arctan2(length, evlp0.bu-evlp0.au*length)
            dpsiy = np.arctan2(length, evlp0.bv-evlp0.av*length)
            psix = evlp0.psix + dpsix
            psiy = evlp0.psiy + dpsiy
            evlp1 = Envelope(cov, evlp0.s, T, psix, psiy)
        if disp0 is None:
            disp1 = None
        else:
            disp1 = Dispersion(tmat @ disp0.vector, disp0.s)
        return cood1, evlp1, disp1

    @classmethod
    def drift_transfer_array(cls, length: float, cood0: CoordinateArray = None, evlp0: EnvelopeArray = None, disp0: DispersionArray = None) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Transfer the coordinate array, envelope array, and dispersion array through a drift space of the specified length.
        This method is used internally for longitudinal offset (ds) of an element.

        Args:
            length float: Length of the drift space [m].
            cood0 CoordinateArray: Initial coordinate array (optional).
            evlp0 EnvelopeArray: Initial beam envelope array (optional).
            disp0 DispersionArray: Initial dispersion array (optional).

        Returns:
            CoordinateArray: Coordinate array after the drift space (if cood0 is provided).
            EnvelopeArray: Beam envelope array after the drift space (if evlp0 is provided).
            DispersionArray: Dispersion array after the drift space (if disp0 is provided).
        '''
        if length == 0.:
            return cood0, evlp0, disp0
        tmat = np.eye(4)
        tmat[0, 1] = length
        tmat[2, 3] = length
        if cood0 is None:
            cood1 = None
        else:
            coodvec = np.matmul(tmat, cood0.vector)
            cood1 = CoordinateArray(coodvec, cood0.s, cood0.z, cood0.delta)
        if evlp0 is None:
            evlp1 = None
        else:
            cov = np.einsum('ij,njk,kl->nil', tmat, evlp0.cov, tmat.T)
            M = np.array([[1., length], [0., 1.]])
            M_ = np.array([[1., -length], [0., 1.]])
            T = np.einsum('ij,njk,kl->nil', M, evlp0.T, M_)
            dpsix = np.arctan2(length, evlp0.bu+evlp0.au*length)
            dpsiy = np.arctan2(length, evlp0.bv+evlp0.av*length)
            psix = evlp0.psix + dpsix
            psiy = evlp0.psiy + dpsiy
            evlp1 = EnvelopeArray(cov, evlp0.s, T, psix, psiy)
        if disp0 is None:
            disp1 = None
        else:
            dispvec = np.matmul(tmat, disp0.vector)
            disp1 = DispersionArray(dispvec, disp0.s)
        return cood1, evlp1, disp1

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1, method: str = 'symplectic4') \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.
        The start point is always the beginning of the element, and the end point is always the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood = cood0.copy()
        evlp = evlp0.copy() if evlp0 is not None else None
        disp = disp0.copy() if disp0 is not None else None
        cood, evlp, disp = self.drift_transfer(self._ds, cood, evlp, disp)
        cood.x -= self._dx
        cood.y -= self._dy
        if self._elements is not None:
            for elem in self._elements:
                cood, evlp, disp = elem.transfer(cood, evlp, disp, ds, method)
            cood1, evlp1, disp1 = cood, evlp, disp
        else:
            tmat = self.transfer_matrix(cood0, ds, method)
            cood1vec = np.dot(tmat, cood.vector)
            cood1 = Coordinate(cood1vec, cood.s + self._length, cood.z, cood.delta)
            evlp1 = evlp
            if evlp1 is not None:
                evlp1.transfer(tmat, self._length)
            disp1 = disp
            if disp1 is not None:
                disp1vec = np.dot(tmat, disp1.vector) + self.dispersion(cood0, ds, method)
                disp1 = Dispersion(disp1vec, disp1.s + self._length)
        cood1.x += self._dx
        cood1.y += self._dy
        cood1, evlp1, disp1 = self.drift_transfer(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True, method: str = 'symplectic4') \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.
        The start point is always the beginning of the element, and the end point is always the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        cood = cood0.copy()
        evlp = evlp0.copy() if evlp0 is not None else None
        disp = disp0.copy() if disp0 is not None else None
        cood, evlp, disp = self.drift_transfer(self._ds, cood, evlp, disp)
        cood.x -= self._dx
        cood.y -= self._dy
        if self._elements is not None:
            cood1, evlp1, disp1 = None, None, None
            for elem in self._elements:
                coodarray, evlparray, disparray = elem.transfer_array(cood, evlp, disp, ds, False, method)
                if cood1 is None:
                    cood1 = coodarray
                else:
                    cood1.append(coodarray)
                    if cood1.vector.shape[1] != cood1.s.shape[0]:
                        raise ValueError(f'transfer_array: shape mismatch name={elem.name}, len(cood)={cood1.vector.shape[1]}, len(s)={cood1.s.shape[0]}')
                if evlp is not None:
                    if evlp1 is None:
                        evlp1 = evlparray
                    else:
                        evlp1.append(evlparray)
                        if evlp1.cov.shape[0] != evlp1.s.shape[0]:
                            raise ValueError(f'transfer_array: shape mismatch name={elem.name}, len(evlp)={evlp1.cov.shape[0]}, len(s)={evlp1.s.shape[0]}')
                if disp is not None:
                    if disp1 is None:
                        disp1 = disparray
                    else:
                        disp1.append(disparray)
                        if disp1.vector.shape[1] != disp1.s.shape[0]:
                            raise ValueError(f'transfer_array: shape mismatch name={elem.name}, len(disp)={disp1.vector.shape[1]}, len(s)={disp1.s.shape[0]}')
                cood, evlp, disp = elem.transfer(cood, evlp, disp, ds, method)
            if endpoint:
                cood1.append(CoordinateArray(cood.vector[:, np.newaxis], np.array([cood.s])))
                if evlp is not None:
                    evlp1.append(EnvelopeArray(evlp.cov[np.newaxis, :, :], np.array([evlp.s]), evlp.T[np.newaxis, :, :],
                                               np.array([evlp.psix]), np.array([evlp.psiy])))
                if disp is not None:
                    disp1.append(DispersionArray(disp.vector[:, np.newaxis], np.array([disp.s])))
        else:
            tmat, s = self.transfer_matrix_array(cood0, ds, endpoint, method)
            coodvec = np.matmul(tmat, cood.vector).T
            cood1 = CoordinateArray(coodvec, s + cood0.s, np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
            if evlp is not None:
                evlp1 = EnvelopeArray.transport(evlp, tmat, s)
            else:
                evlp1 = None
            if disp is not None:
                disp_add, _ = self.dispersion_array(cood0, ds, endpoint, method)
                dispvec = np.matmul(tmat, disp.vector).T + disp_add
                disp1 = DispersionArray(dispvec, s + disp0.s)
            else:
                disp1 = None
        cood1.x += self._dx
        cood1.y += self._dy
        cood1, evlp1, disp1 = self.drift_transfer_array(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    def radiation_integrals(self, cood0: Coordinate, evlp0: Envelope, disp0: Dispersion, ds: float = 0.1, method: str = 'symplectic4') \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial envelope.
            disp0 Dispersion: Initial dispersion.
            ds float: Maximum step size [m].
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            Tuple[float, float, float, float, float, float]: Radiation integrals I2, I4, I5u, I5v, I4u, and I4v.
        '''
        if self._elements is None:
            return 0., 0., 0., 0., 0., 0.
        I2, I4, I5u, I5v, I4u, I4v = 0., 0., 0., 0., 0., 0.
        cood = cood0.copy()
        evlp = evlp0.copy()
        disp = disp0.copy()
        for elem in self._elements:
            if elem.length == 0.:
                continue
            i2, i4, i5u, i5v, i4u, i4v = elem.radiation_integrals(cood, evlp, disp, ds, method)
            I2 += i2
            I4 += i4
            I5u += i5u
            I5v += i5v
            I4u += i4u
            I4v += i4v
            cood, evlp, disp = elem.transfer(cood, evlp, disp, ds, method)
        return I2, I4, I5u, I5v, I4u, I4v

    def get_element_from_s(self, s: float) -> Tuple[Element, float]:
        '''
        Get element and local longitudinal position by longitudinal position.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Element: Element at the specified longitudinal position.
            float: Local longitudinal position in the element [m].
        '''
        if s < 0. or s > self._length or (self._length > 0. and s == self._length):
            raise ValueError(f'Longitudinal position out of range. s={s}, length={self._length}')
        if self._elements is not None:
            s0 = 0.
            for elem in self._elements:
                if s < s0 + elem._length:
                    return elem.get_element_from_s(s - s0)
                s0 += elem._length
            raise ValueError(f'Longitudinal position out of range. s={s}, length={self._length}')
        else:
            return self, s

    def transfer_matrix_from_s(self, s: float, cood0: Coordinate = Coordinate(), ds: float = 0.1, method: str = 'symplectic4') \
        -> npt.NDArray[np.floating]:
        '''
        Transfer matrices from the given longitudinal position to the end of the element.

        Args:
            s float: Longitudinal position [m].
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}') (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix from s to the end of the element.
        '''
        if s < 0. or s > self._length:
            raise ValueError(f'Longitudinal position out of range. s={s}, length={self._length}')
        cood = cood0.copy()
        cood, _, _ = self.drift_transfer(self._ds, cood, None, None)
        cood.x -= self._dx
        cood.y -= self._dy
        if self._elements is not None:
            s0 = 0.
            for elem in self._elements:
                if s >= s0 and s < s0 + elem._length:
                    tmat = elem.transfer_matrix_from_s(s - s0, cood, ds, method)
                    coodvec = np.dot(tmat, cood.vector)
                    cood = Coordinate(coodvec, cood.s + elem._length - (s - s0), cood.z, cood.delta)
                elif s < s0:
                    tmat_elem = elem.transfer_matrix(cood, ds, method)
                    tmat = np.dot(tmat_elem, tmat)
                    cood, _, _ = elem.transfer(cood, ds=ds, method=method)
                s0 += elem._length
            return tmat
        else:
            elem = self.copy()
            elem._length -= s
            return elem.transfer_matrix(cood0, ds, method)

    # To Do
    def transfer_from_s(self, s: float, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1, method: str = 'symplectic4') \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion from the specified longitudinal position to the end of the element.

        Args:
            s float: Longitudinal start position [m].
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        if s < 0. or s > self._length:
            raise ValueError(f'Initial coordinate s out of range. s={s}, length={self._length}')
        cood = cood0.copy()
        evlp = evlp0.copy() if evlp0 is not None else None
        disp = disp0.copy() if disp0 is not None else None
        cood, evlp, disp = self.drift_transfer(self._ds, cood, evlp, disp)
        cood.x -= self._dx
        cood.y -= self._dy
        if self._elements is not None:
            s0 = 0.
            for elem0 in self._elements:
                if s0 < cood.s:
                    continue
                elif cood.s > s0 and cood.s < s0 + elem0._length:
                    elem = elem0.copy()
                    elem._length -= cood.s - s0
                else:
                    elem = elem0
                cood, evlp, disp = elem.transfer(cood, evlp, disp, ds, method)
                s0 += elem0._length
            cood1, evlp1, disp1 = cood, evlp, disp
        else:
            if cood0.s == 0.:
                elem = self
            else:
                elem = self.copy()
                elem._length -= cood0.s
            tmat = elem.transfer_matrix(cood, ds, method)
            cood = np.dot(tmat, cood.vector)
            cood1 = Coordinate(cood, cood.s + elem._length, cood.z, cood.delta)
            if evlp is not None:
                evlp1 = evlp
                evlp1.transfer(tmat, elem._length)
            else:
                evlp1 = None
            if disp is not None:
                disp = np.dot(tmat, disp.vector) + elem.dispersion(cood, ds, method)
                disp1 = Dispersion(disp, disp.s + elem._length)
            else:
                disp1 = None
        cood1.x += self._dx
        cood1.y += self._dy
        cood1, evlp1, disp1 = self.drift_transfer(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    # To Do
    def transfer_array_from_s(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                              ds: float = 0.1, endpoint: bool = True, method: str = 'symplectic4') \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.
        The start point is cood0.s and the end point is the end of the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        if cood0.s < 0. or cood0.s > self._length:
            raise ValueError(f'Initial coordinate s out of range. s={cood0.s}, length={self._length}')
        cood = cood0.copy()
        evlp = evlp0.copy() if evlp0 is not None else None
        disp = disp0.copy() if disp0 is not None else None
        cood, evlp, disp = self.drift_transfer(self._ds, cood, evlp, disp)
        cood.x -= self._dx
        cood.y -= self._dy
        if self._elements is not None:
            element0, s0 = self.get_element_from_s(cood0.s) # To Do
            cood1, evlp1, disp1 = None, None, None
            for elem in self._elements:
                coodarray, evlparray, disparray = elem.transfer_array(cood, evlp, disp, ds, False, method)
                if cood1 is None:
                    cood1 = coodarray
                else:
                    cood1.append(coodarray)
                    if cood1.vector.shape[1] != cood1.s.shape[0]:
                        raise ValueError(f'transfer_array: shape mismatch name={elem.name}, len(cood)={cood1.vector.shape[1]}, len(s)={cood1.s.shape[0]}')
                if evlp is not None:
                    if evlp1 is None:
                        evlp1 = evlparray
                    else:
                        evlp1.append(evlparray)
                        if evlp1.cov.shape[0] != evlp1.s.shape[0]:
                            raise ValueError(f'transfer_array: shape mismatch name={elem.name}, len(evlp)={evlp1.cov.shape[0]}, len(s)={evlp1.s.shape[0]}')
                if disp is not None:
                    if disp1 is None:
                        disp1 = disparray
                    else:
                        disp1.append(disparray)
                        if disp1.vector.shape[1] != disp1.s.shape[0]:
                            raise ValueError(f'transfer_array: shape mismatch name={elem.name}, len(disp)={disp1.vector.shape[1]}, len(s)={disp1.s.shape[0]}')
                cood, evlp, disp = elem.transfer(cood, evlp, disp, ds, method)
            if endpoint:
                cood1.append(CoordinateArray(cood.vector[:, np.newaxis], np.array([cood.s])))
                if evlp is not None:
                    evlp1.append(EnvelopeArray(evlp.cov[np.newaxis, :, :], np.array([evlp.s]), evlp.T[np.newaxis, :, :],
                                               np.array([evlp.psix]), np.array([evlp.psiy])))
                if disp is not None:
                    disp1.append(DispersionArray(disp.vector[:, np.newaxis], np.array([disp.s])))
            cood1.x += self._dx
            cood1.y += self._dy
            if self._ds != 0.:
                cood1, evlp1, disp1 = self.drift_transfer_array(-self._ds, cood1, evlp1, disp1)
        else:
            tmat, s = self.transfer_matrix_array(cood, ds, endpoint, method)
            cood = np.matmul(tmat, cood.vector).T
            cood[0] += self._dx
            cood[2] += self._dy
            cood1 = CoordinateArray(cood, s + cood0.s, np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
            if evlp is not None:
                evlp1 = EnvelopeArray.transport(evlp, tmat, s - self._ds)
            else:
                evlp1 = None
            if disp is not None:
                disp_add, _ = self.dispersion_array(cood, ds, endpoint, method)
                disp = np.matmul(tmat, disp.vector).T + disp_add
                disp1 = DispersionArray(disp, s + disp0.s)
            else:
                disp1 = None
        return cood1, evlp1, disp1

    def get_element(self, indices: int | Tuple[int, ...]) -> Element:
        '''
        Get element by index or tuple of indices.

        Args:
            indices int or tuple of int: Index or tuple of indices.

        Returns:
            Element: Element at the specified index.
        '''
        if isinstance(indices, int):
            return self._elements[indices]
        elif isinstance(indices, tuple):
            if not isinstance(indices[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if len(indices) > 1 and self._elements[indices[0]]._elements is not None:
                return self._elements[indices[0]].get_element(indices[1:])
            else:
                return self._elements[indices[0]]
        else:
            raise TypeError('Index must be int or tuple of int.')

    def set_element(self, indices: int | Tuple[int, ...], element: Element) -> None:
        '''
        Set element by index or tuple of indices.

        Args:
            indices int or tuple of int: Index or tuple of indices.
            element Element: Element to set.
        '''
        if isinstance(indices, int):
            if element.indices is None or len(element.indices) <= 1:
                element.set_indices((indices,))
            self._elements[indices] = element
        elif isinstance(indices, tuple):
            if not isinstance(indices[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if element.indices is None or len(element.indices) <= len(indices):
                element.set_indices(indices)
            if len(indices) > 1 and self._elements[indices[0]]._elements is not None:
                self._elements[indices[0]].set_element(indices[1:], element)
            else:
                self._elements[indices[0]] = element
        else:
            raise TypeError('Index must be int or tuple of int.')

    def get_s(self, element: Element | int | Tuple[int, ...]) -> float:
        '''
        Get longitudinal position by Element, index or tuple of indices.

        Args:
            element Element, int or tuple of int: Element, index or tuple of indices.

        Returns:
            float: Longitudinal position [m].
        '''
        if isinstance(element, Element):
            return self.get_s(element.indices)
        elif isinstance(element, int):
            indices = element
            if indices < 0 or indices >= len(self._elements):
                raise IndexError('Index out of range.')
            s = 0.
            for i in range(indices):
                s += self._elements[i].length
            return s
        elif isinstance(element, tuple):
            indices = element
            if not isinstance(indices[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if indices[0] < 0 or indices[0] >= len(self._elements):
                raise IndexError('Index out of range.')
            s = 0.
            for i in range(indices[0]):
                s += self._elements[i].length
            if len(indices) > 1 and self._elements[indices[0]]._elements is not None:
                s += self._elements[indices[0]].get_s(indices[1:])
            elif len(indices) > 1:
                raise IndexError(f'Dimension of index is out of range. indices={indices}, element={self._elements[indices[0]]}')
            return s
        else:
            raise TypeError('Argument must be Element, int or tuple of int.')

    def find_index(self, name: str | Tuple[str, ...]) -> List[Tuple[int, ...]]:
        '''
        Find indices of elements starting with a given name.

        Args:
            name str | tuple of str: Name of the element.

        Returns:
            List[Tuple[int, ...]]: List of index tuples of matching elements.
        '''
        index_list = []
        for i,elem in enumerate(self._elements):
            if elem._elements is not None:
                try:
                    sub_index_list = elem.find_index(name)
                    index_list += [((i,) + idx) for idx in sub_index_list]
                except KeyError:
                    continue
            elif isinstance(name, str) and elem.name.startswith(name):
                index_list.append((i,))
            elif isinstance(name, tuple):
                for n in name:
                    if elem.name.startswith(n):
                        index_list.append((i,))
                        break
        if len(index_list) == 0:
            raise KeyError(f'Element starting with name {name} not found.')
        return index_list

