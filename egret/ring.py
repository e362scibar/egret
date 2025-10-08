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
from .betafunc import BetaFunc

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
        for e in elements:
            length += e.length
        super().__init__(name, length, 0., 0., 0., 0., info)
        self.angle = 0.
        self.disp0 = np.zeros(4)  # initial dispersion
        self.tune = np.zeros(2)
        self.beta0 = BetaFunc()  # initial beta function
        self.elements = copy.deepcopy(elements)
        self.energy = energy
        self.update()

    def update(self):
        '''
        Update transfer matrix, dispersion, and emittance.
        '''
        for e in self.elements:
            self.tmat = np.dot(e.tmat, self.tmat)
            self.disp = np.dot(e.tmat, self.disp.T).T + e.disp
            try:
                self.angle += e.angle
            except AttributeError:
                pass
        # initial dispersion
        self.disp0 = np.linalg.inv(np.eye(4) - self.tmat) @ self.disp
        # initial beta function and tune
        cospsix = 0.5 * (np.trace(self.tmat[0:2, 0:2]))
        cospsiy = 0.5 * (np.trace(self.tmat[2:4, 2:4]))
        sin2psix = np.linalg.det(self.tmat[0:2, 0:2] - np.eye(2) * cospsix)
        sin2psiy = np.linalg.det(self.tmat[2:4, 2:4] - np.eye(2) * cospsiy)
        sinpsix = np.sign(self.tmat[0, 1]-self.tmat[1, 0]) * np.sqrt(abs(sin2psix))
        sinpsiy = np.sign(self.tmat[2, 3]-self.tmat[3, 2]) * np.sqrt(abs(sin2psiy))
        psix = np.arctan2(sinpsix, cospsix)
        psiy = np.arctan2(sinpsiy, cospsiy)
        betax = self.tmat[0, 1] / sinpsix
        betay = self.tmat[2, 3] / sinpsiy
        alphax = (self.tmat[0, 0] - self.tmat[1, 1]) / (2.*sinpsix)
        alphay = (self.tmat[2, 2] - self.tmat[3, 3]) / (2.*sinpsiy)
        self.beta0 = BetaFunc(betax, alphax, betay, alphay, 0.)
        self.tune[0] = psix / (2.*np.pi)
        self.tune[1] = psiy / (2.*np.pi)
        for i in range(2):
            if self.tune[i] < 0.:
                self.tune[i] += 1.
        self.I2, self.I4, self.I5 = self.radiation_integrals()
        self.emittance = self.C_q * (self.energy / self.m_e_eV)**2 * self.I5 / (self.I2 - self.I4)
        self.Jx = 1. - self.I4 / self.I2
        self.Jy = 1.
        self.Jz = 2. + self.I4 / self.I2

    def betafunc(self, ds: float = 0.01, endpoint: bool = True) -> BetaFunc:
        '''
        Beta function along the ring.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            BetaFunc: Beta function along the ring.
        '''
        b0 = copy.deepcopy(self.beta0)
        beta = copy.deepcopy(b0)
        for elem in self.elements:
            if elem.length == 0.:
                continue
            beta.append(elem.betafunc(b0, ds, False))
            b0 = b0.transfer(elem.tmat, elem.length)
        if endpoint:
            beta.append(b0)
        return beta

    def dispersion(self, ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Dispersion along the ring.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            NDArray[np.floating]: Dispersion along the ring.
            NDArray[np.floating]: Longitudinal position along the ring [m].
        '''
        s0 = 0.
        s = np.array([0.])
        disp = np.zeros((4,1))
        for elem in self.elements:
            if elem.length == 0.:
                continue
            _, ss = elem.tmatarray(ds, False)
            disp = np.hstack((disp, elem.dispersion(ds, False)))
            s = np.hstack((s, ss+s0))
            s0 += elem.length
        if endpoint:
            disp = np.hstack((disp, np.zeros((4,1))))
            s = np.hstack((s, np.array([s0])))
        return disp, s

    def etafunc(self, ds: float = 0.01, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Eta function along the ring.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            NDArray[np.floating]: Eta function along the ring.
            NDArray[np.floating]: Longitudinal position along the ring [m].
        '''
        s0 = 0.
        s = np.array([0.])
        eta0 = copy.copy(self.disp0)
        eta = np.array([copy.copy(self.disp0)]).T
        for elem in self.elements:
            if elem.length == 0.:
                continue
            etaelem, ss = elem.etafunc(eta0, ds, False)
            eta = np.hstack((eta, etaelem))
            s = np.hstack((s, ss + s0))
            s0 += elem.length
            eta0 = np.matmul(elem.tmat, eta0) + elem.disp
        if endpoint:
            eta = np.hstack((eta, eta0[:,np.newaxis]))
            s = np.hstack((s, np.array([s0])))
        return eta, s

    def radiation_integrals(self, ds: float = 0.1) -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            ds float: Step size for numerical integration [m].
        
        Returns:
            I2 float: Second radiation integral.
            I4 float: Fourth radiation integral.
            I5 float: Fifth radiation integral.
        '''
        I2 = 0.
        I4 = 0.
        I5 = 0.
        beta = copy.deepcopy(self.beta0)
        eta = copy.deepcopy(self.disp0)
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
