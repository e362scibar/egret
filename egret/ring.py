# ring.py
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

from .element import Element
from .coordinate import Coordinate
from .envelope import Envelope
from .dispersion import Dispersion

import copy
import numpy as np
import numpy.typing as npt
from typing import Tuple, List

class Ring(Element):
    '''
    Ring accelerator.
    '''
    C_q = 3.83193864e-13  # Factor for equilibrium emittance
    m_e_eV = 510998.95  # Electron rest mass in eV

    def __init__(self, name: str, elements: List[Element], energy: float, info: str = ''):
        '''
        Args:
            name str: Name of the lattice.
            elements list of Element: List of elements in the lattice.
            energy float: Beam energy [eV].
            info str: Additional information.
        '''
        length = 0.
        for elem in elements:
            length += elem.length
        super().__init__(name, length, 0., 0., 0., 0., info)
        self.angle = 0.
        self.tune = np.zeros(2)
        self.elements = copy.deepcopy(elements)
        self.energy = energy
        self.update()

    def update(self):
        '''
        Update transfer matrix, dispersion, and emittance.
        '''
        for elem in self.elements:
            try:
                self.angle += elem.angle
            except AttributeError:
                pass
        cood = Coordinate()
        tmat = self.transfer_matrix(cood)
        # initial dispersion
        disp = self.dispersion(cood)
        disp0 = np.dot(np.linalg.inv(np.eye(4) - tmat), disp)
        self.disp0 = Dispersion(disp0[0], disp0[1], disp0[2], disp0[3], 0.)
        # initial beta function and tune
        cospsix = 0.5 * (np.trace(tmat[0:2, 0:2]))
        cospsiy = 0.5 * (np.trace(tmat[2:4, 2:4]))
        sin2psix = np.linalg.det(tmat[0:2, 0:2] - np.eye(2) * cospsix)
        sin2psiy = np.linalg.det(tmat[2:4, 2:4] - np.eye(2) * cospsiy)
        sinpsix = np.sign(tmat[0, 1]-tmat[1, 0]) * np.sqrt(abs(sin2psix))
        sinpsiy = np.sign(tmat[2, 3]-tmat[3, 2]) * np.sqrt(abs(sin2psiy))
        psix = np.arctan2(sinpsix, cospsix)
        psiy = np.arctan2(sinpsiy, cospsiy)
        betax = tmat[0, 1] / sinpsix
        betay = tmat[2, 3] / sinpsiy
        alphax = (tmat[0, 0] - tmat[1, 1]) / (2.*sinpsix)
        alphay = (tmat[2, 2] - tmat[3, 3]) / (2.*sinpsiy)
        self.evlp0 = Envelope(betax, alphax, betay, alphay, 0.)
        self.tune[0] = psix / (2.*np.pi)
        self.tune[1] = psiy / (2.*np.pi)
        for i in range(2):
            if self.tune[i] < 0.:
                self.tune[i] += 1.
        self.I2, self.I4, self.I5 = self.radiation_integrals(cood, self.evlp0, self.disp0)
        self.emittance = self.C_q * (self.energy / self.m_e_eV)**2 * self.I5 / (self.I2 - self.I4)
        self.Jx = 1. - self.I4 / self.I2
        self.Jy = 1.
        self.Jz = 2. + self.I4 / self.I2

    def get_element(self, key: int | Tuple[int, ...]) -> Element:
        '''
        Get element by index or tuple of indices.

        Args:
            key int or tuple of int: Index or tuple of indices.

        Returns:
            Element: Element at the specified index.
        '''
        if isinstance(key, int):
            return self.elements[key]
        elif isinstance(key, tuple):
            if not isinstance(key[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if len(key) > 1 and hasattr(self.elements[key[0]], 'elements'):
                return self.elements[key[0]].get_element(key[1:])
            else:
                return self.elements[key[0]]
        else:
            raise TypeError('Index must be int or tuple of int.')

    def get_s(self, key: int | Tuple[int, ...]) -> float:
        '''
        Get longitudinal position by index or tuple of indices.

        Args:
            key int or tuple of int: Index or tuple of indices.

        Returns:
            float: Longitudinal position [m].
        '''
        if isinstance(key, int):
            if key < 0 or key >= len(self.elements):
                raise IndexError('Index out of range.')
            s = 0.
            for i in range(key):
                s += self.elements[i].length
            return s
        elif isinstance(key, tuple):
            if not isinstance(key[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if key[0] < 0 or key[0] >= len(self.elements):
                raise IndexError('Index out of range.')
            s = 0.
            for i in range(key[0]):
                s += self.elements[i].length
            if len(key) > 1 and hasattr(self.elements[key[0]], 'elements'):
                s += self.elements[key[0]].get_s(key[1:])
            return s
        else:
            raise TypeError('Index must be int or tuple of int.')

    def find_index(self, name: str | Tuple[str, ...]) -> Tuple[int, ...]:
        '''
        Find indices of elements starting with a given name.

        Args:
            name str | tuple of str: Name of the element.

        Returns:
            tuple of int: Tuple of indices of the element.
        '''
        index_list = []
        for i,elem in enumerate(self.elements):
            if hasattr(elem, 'elements'):
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

    def transfer_matrix(self, cood0: Coordinate) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the ring.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        cood = cood0.copy()
        tmat = np.eye(4)
        for elem in self.elements:
            tmat = np.dot(elem.transfer_matrix(cood), tmat)
            cood = elem.transfer(cood)[0]
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix along the ring.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        cood = cood0.copy()
        s0 = 0.
        tmat = np.eye(4)
        sarray = []
        tmatarray = []
        for elem in self.elements:
            tmat_elem, s_elem = elem.transfer_matrix_array(cood, ds, False)
            tmatarray.append(np.moveaxis(np.matmul(tmat_elem, tmat), 0, 2))
            sarray.append(s_elem + s0)
            s0 += elem.length
            tmat = np.dot(elem.transfer_matrix(cood), tmat)
            cood = elem.transfer(cood)[0]
        if endpoint:
            tmatarray.append(tmat[:,:,np.newaxis])
            sarray.append(np.array([s0]))
        return np.moveaxis(np.dstack(tmatarray), 2, 0), np.hstack(sarray)

    def dispersion(self, cood0: Coordinate) -> Dispersion:
        '''
        Additive dispersion of the ring.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            Dispersion: Additive dispersion of the lattice.
        '''
        cood = cood0.copy()
        disp = Dispersion()
        for elem in self.elements:
            cood, _, disp = elem.transfer(cood, None, disp)
        return disp.vector

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the ring.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        cood = cood0.copy()
        disparray = []
        sarray = []
        s0 = 0.
        disp = Dispersion()
        for elem in self.elements:
            disp_elem, s_elem = elem.dispersion_array(cood, ds, False)
            tmat, _ = elem.transfer_matrix_array(cood, ds, False)
            disparray.append(disp_elem + np.dot(tmat, disp.vector).T)
            sarray.append(s_elem + s0)
            s0 += elem.length
            cood, _, disp = elem.transfer(cood, None, disp)
        if endpoint:
            disparray.append(disp.vector[:, np.newaxis])
            sarray.append(np.array([s0]))
        return np.hstack(disparray), np.hstack(sarray)

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
            I2 float: Second radiation integral.
            I4 float: Fourth radiation integral.
            I5 float: Fifth radiation integral.
        '''
        I2 = 0.
        I4 = 0.
        I5 = 0.
        cood = cood0.copy()
        evlp = evlp0.copy()
        disp = disp0.copy()
        for elem in self.elements:
            if elem.length == 0.:
                continue
            i2, i4, i5 = elem.radiation_integrals(cood, evlp, disp, ds)
            I2 += i2
            I4 += i4
            I5 += i5
            cood, evlp, disp = elem.transfer(cood, evlp, disp)
        return I2, I4, I5
