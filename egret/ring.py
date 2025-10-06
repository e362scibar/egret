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
from typing import Tuple

class Ring(Element):
    """
    Ring accelerator.
    """
    C_q = 3.83193864e-13  # Factor for equilibrium emittance
    m_e_eV = 510998.95  # Electron rest mass in eV
    
    def __init__(self, name, elements, energy, info=''):
        length = 0.
        for e in elements:
            length += e.length
        super().__init__(name, length, 0., 0., 0., 0., info)
        self.angle = 0.
        self.disp0 = np.zeros(6)  # initial dispersion
        self.tune = np.zeros(2)
        self.beta0 = BetaFunc()  # initial beta function
        self.elements = copy.deepcopy(elements)
        self.energy = energy
        self.update()

    def update(self):
        for e in self.elements:
            self.tmat = np.dot(e.tmat, self.tmat)
            self.disp = np.dot(e.tmat, self.disp.T).T + e.disp
            try:
                self.angle += e.angle
            except AttributeError:
                pass
        # initial dispersion
        self.disp0[0:4] = np.linalg.inv(np.eye(4) - self.tmat[0:4,0:4]) @ self.disp[0:4]
        # initial beta function and tune
        cospsix = 0.5 * (np.trace(self.tmat[0:2,0:2]))
        cospsiy = 0.5 * (np.trace(self.tmat[2:4,2:4]))
        sin2psix = np.linalg.det(self.tmat[0:2,0:2] - np.eye(2) * cospsix)
        sin2psiy = np.linalg.det(self.tmat[2:4,2:4] - np.eye(2) * cospsiy)
        sinpsix = np.sign(self.tmat[0,1]-self.tmat[1,0]) * np.sqrt(abs(sin2psix))
        sinpsiy = np.sign(self.tmat[2,3]-self.tmat[3,2]) * np.sqrt(abs(sin2psiy))
        psix = np.arctan2(sinpsix, cospsix)
        psiy = np.arctan2(sinpsiy, cospsiy)
        betax = self.tmat[0,1] / sinpsix
        betay = self.tmat[2,3] / sinpsiy
        alphax = (self.tmat[0,0]-self.tmat[1,1]) / (2.*sinpsix)
        alphay = (self.tmat[2,2]-self.tmat[3,3]) / (2.*sinpsiy)
        self.beta0 = BetaFunc(betax, alphax, betay, alphay, 0.)
        self.tune[0] = psix / (2.*np.pi)
        self.tune[1] = psiy / (2.*np.pi)
        for i in range(2):
            if self.tune[i] < 0.:
                self.tune[i] += 1.

    def betafunc(self, ds:float=0.01, endpoint:bool=True)->BetaFunc:
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

    def dispersion(self, ds:float=0.01, endpoint:bool=True)->Tuple[npt.NDArray[np.floating],npt.NDArray[np.floating]]:
        s0 = 0.
        s = np.array([0.])
        disp = np.zeros((6,1))
        for elem in self.elements:
            if elem.length == 0.:
                continue
            _, ss = elem.tmatarray(ds, False)
            disp = np.hstack((disp, elem.dispersion(ds, False)))
            s = np.hstack((s, ss+s0))
            s0 += elem.length
        if endpoint:
            disp = np.hstack((disp, np.zeros((6,1))))
            s = np.hstack((s, np.array([s0])))
        return disp, s

    def etafunc(self, ds:float=0.01, endpoint:bool=True)->Tuple[npt.NDArray[np.floating],npt.NDArray[np.floating]]:
        s0 = 0.
        s = np.array([0.])
        eta0 = copy.copy(self.disp0)
        eta = np.array([copy.copy(self.disp0)]).T
        for elem in self.elements:
            if elem.length == 0.:
                continue
            etaelem, ss = elem.etafunc(eta0, ds, False)
            eta = np.hstack((eta, etaelem))
            s = np.hstack((s, ss+s0))
            s0 += elem.length
            eta0 = np.matmul(elem.tmat, eta0) + elem.disp
        if endpoint:
            eta = np.hstack((eta, eta0[:,np.newaxis]))
            s = np.hstack((s, np.array([s0])))
        return eta, s
