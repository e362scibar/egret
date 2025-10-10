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

from .element import Element
from .coordinate import Coordinate
from .envelope import Envelope
from .dispersion import Dispersion

import copy
import numpy as np
import numpy.typing as npt
from typing import Tuple, List

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

    def update(self):
        '''
        Update angle of the lattice.
        '''
        for elem in self.elements:
            try:
                self.angle += elem.angle
            except AttributeError:
                pass

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
