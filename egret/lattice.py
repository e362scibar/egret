# lattice.py
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
from .envelope import Envelope
from .dispersion import Dispersion

import copy
import numpy as np
import numpy.typing as npt
from typing import Tuple, List

# Optional Numba flag for potential future use in-file
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

class Lattice(Element):
    '''
    Lattice element.
    '''

    def __init__(self, name: str, elements: List[Element],
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the lattice.
            elements list of Element: List of elements in the lattice.
            dx float: Horizontal offset of the lattice [m].
            dy float: Vertical offset of the lattice [m].
            ds float: Longitudinal offset of the lattice [m].
            tilt float: Tilt angle of the lattice [rad].
            info str: Additional information.
        '''
        length = 0.
        for elem in elements:
            length += elem.length
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.angle = 0.
        self.elements = copy.deepcopy(elements)
        self.update()

    def copy(self) -> Lattice:
        '''
        Return a copy of the lattice.

        Returns:
            Lattice: Copy of the lattice.
        '''
        return Lattice(self.name, self.elements,
                       self.dx, self.dy, self.ds,
                       self.tilt, self.info)

    def update(self):
        '''
        Update bending angle of the lattice.
        '''
        for elem in self.elements:
            try:
                self.angle += elem.angle
            except AttributeError:
                pass

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
            elif len(key) > 1:
                raise IndexError('Dimension of index is out of range.')
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
        Transfer matrix of the lattice.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: 6x6 transfer matrix of the lattice.
        '''
        tmat = np.eye(4)
        cood = cood0.copy()
        for elem in self.elements:
            tmat = np.dot(elem.transfer_matrix(cood), tmat)
            cood = elem.transfer(cood)[0]
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix along the lattice.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: 6x6xN array of transfer matrices along the lattice.
            npt.NDArray[np.floating]: Longitudinal positions along the lattice [m].
        '''
        # try optimized preallocated assembly of transfer-matrix array
        try:
            return _lattice_transfer_matrix_array(self, cood0, ds, endpoint)
        except Exception:
            # fallback: original implementation
            s0 = 0.
            tmat = np.eye(4)
            cood = cood0.copy()
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
        Additive dispersion of the lattice.

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
        Additive dispersion array along the lattice.

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
            disparray.append(disp_elem + np.matmul(tmat, disp.vector).T)
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
        Calculate radiation integrals along the lattice.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial envelope.
            disp0 Dispersion: Initial dispersion.
            ds float: Maximum step size [m].

        Returns:
            Tuple[float, float, float]: Radiation integrals (I2, I4, I5).
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

def _lattice_transfer_matrix_array_py(self: 'Lattice', cood0: Coordinate, ds: float, endpoint: bool):
    # Precompute sizes to preallocate arrays and avoid repeated concatenation
    # For each element, number of samples n_i = int(length//ds) + 1 (endpoint False)
    n_list = []
    for elem in self.elements:
        n_i = int(elem.length // ds) + 1
        n_list.append(n_i)
    total_n = sum(n_list) + (1 if endpoint else 0)

    tmat_array = np.empty((total_n, 4, 4), dtype=np.float64)
    s_array = np.empty((total_n,), dtype=np.float64)

    cood = cood0.copy()
    pos = 0
    s0 = 0.0
    tmat_running = np.eye(4)
    for idx, elem in enumerate(self.elements):
        # request element transfer-matrix array with endpoint=False
        tmat_elem, s_elem = elem.transfer_matrix_array(cood, ds, False)
        n_i = tmat_elem.shape[0]
        # multiply each tmat_elem by running tmat and store
        for j in range(n_i):
            tmat_array[pos + j] = np.matmul(tmat_elem[j], tmat_running)
            s_array[pos + j] = s_elem[j] + s0
        pos += n_i
        s0 += elem.length
        # update running transfer matrix and coordinate
        tmat_running = np.dot(elem.transfer_matrix(cood), tmat_running)
        cood = elem.transfer(cood)[0]

    if endpoint:
        tmat_array[pos] = tmat_running
        s_array[pos] = s0

    return tmat_array, s_array

# bind and optionally njit (keep pure-Python for safety)
_lattice_transfer_matrix_array = _lattice_transfer_matrix_array_py
if _NUMBA_AVAILABLE:
    try:
        _lattice_transfer_matrix_array = njit(_lattice_transfer_matrix_array_py, cache=True)
    except Exception:
        _lattice_transfer_matrix_array = _lattice_transfer_matrix_array_py
