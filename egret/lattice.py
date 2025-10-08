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
from .betafunc import BetaFunc

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
        for e in elements:
            length += e.length
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.angle = 0.
        self.elements = copy.deepcopy(elements)
        self.update()

    def update(self):
        '''
        Update transfer matrix and dispersion.
        '''
        for e in self.elements:
            self.tmat = np.dot(e.tmat, self.tmat)
            self.disp = np.dot(e.tmat, self.disp.T).T + e.disp
            try:
                self.angle += e.angle
            except AttributeError:
                pass

    def betafunc(self, b0: BetaFunc, ds: float = 0.01, endpoint: bool = False) -> BetaFunc:
        '''
        Beta function along the lattice.
        
        Args:
            b0 BetaFunc: Initial beta function.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            BetaFunc: Beta function along the lattice.
        '''
        b0 = copy.deepcopy(b0)
        beta = copy.deepcopy(b0)
        for elem in self.elements:
            if elem.length == 0.:
                continue
            beta.append(elem.betafunc(b0, ds, False))
            b0 = b0.transfer(elem.tmat, elem.length)
        if endpoint:
            beta.append(b0)
        return beta

    def dispersion(self, ds:float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Dispersion function along the lattice.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s0 = 0.
        s = np.array([0.])
        disp = np.zeros((4,1))
        for elem in self.elements:
            if elem.length == 0.:
                continue
            dispelem, ss = elem.dispersion(ds, False)
            disp = np.hstack((disp, dispelem))
            s = np.hstack((s, ss+s0))
            s0 += elem.length
        return disp, s

    def etafunc(self, eta0: npt.NDArray[np.floating], ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Calculate the dispersion function along the lattice.
        
        Args:
            eta0 npt.NDArray[np.floating]: Initial dispersion function (4,).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s0 = 0.
        s = np.array([0.])
        eta0 = copy.copy(eta0)
        eta = np.array([copy.copy(eta0)]).T
        for elem in self.elements:
            if elem.length == 0.:
                continue
            etaelem, ss = elem.etafunc(eta0, ds, False)
            eta = np.hstack((eta, etaelem))
            s = np.hstack((s, ss+s0))
            s0 += elem.length
            eta0 = np.matmul(elem.tmat, eta0) + elem.disp
        return eta, s

    def radiation_integrals(self, beta0: BetaFunc, eta0: npt.NDArray[np.floating], ds: float = 0.1) \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals along the lattice.
        
        Args:
            beta0 BetaFunc: Initial Twiss parameters.
            eta0 npt.NDArray[np.floating]: Initial dispersion [eta_x, eta_x', eta_y, eta_y'].
            ds float: Step size for numerical integration.
        
        Returns:
            Tuple[float, float, float]: Radiation integrals (I2, I4, I5).
        '''
        I2 = 0.
        I4 = 0.
        I5 = 0.
        beta = copy.deepcopy(beta0)
        eta = copy.copy(eta0)
        for elem in self.elements:
            if elem.length == 0.:
                continue
            i2, i4, i5 = elem.radiation_integrals(beta, eta, ds)
            I2 += i2
            I4 += i4
            I5 += i5
            beta = beta.transfer(elem.tmat, elem.length)
            eta = np.matmul(elem.tmat, eta) + elem.disp
        return I2, I4, I5
