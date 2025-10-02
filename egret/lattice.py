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
from typing import Tuple

class Lattice(Element):
    """
    Lattice element.
    """
    def __init__(self, name, elements, dx=0., dy=0., ds=0., tilt=0., info=''):
        length = 0.
        for e in elements:
            length += e.length
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.angle = 0.
        self.elements = copy.deepcopy(elements)
        self.update()

    def update(self):
        for e in self.elements:
            self.tmat = np.dot(e.tmat, self.tmat)
            self.disp = np.dot(e.tmat, self.disp.T).T + e.disp
            try:
                self.angle += e.angle
            except AttributeError:
                pass

    def betafunc(self, b0:BetaFunc, ds:float=0.01, endpoint:bool=False)->BetaFunc:
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

    def dispersion(self, ds:float=0.01, endpoint:bool=True)->Tuple[npt.NDArray[np.floating],npt.NDArray[np.floating]]:
        s0 = 0.
        s = np.array([0.])
        disp = np.zeros((6,1))
        for elem in self.elements:
            if elem.length == 0.:
                continue
            dispelem, ss = elem.dispersion(ds, False)
            print('Lattice.dispersion(): disp.shape', dispelem.shape)
            disp = np.hstack((disp, dispelem))
            s = np.hstack((s, ss+s0))
            s0 += elem.length
        return disp, s

    def etafunc(self, eta0:npt.NDArray[np.floating], ds:float=0.01, endpoint:bool=True)->Tuple[npt.NDArray[np.floating],npt.NDArray[np.floating]]:
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
