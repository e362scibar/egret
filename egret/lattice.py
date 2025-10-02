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
        b0 = copy.copy(b0)
        beta = copy.copy(b0)
        for elem in self.elements:
            if elem.length == 0.:
                continue
            beta.append(elem.betafunc(b0, ds, False))
            b0.transfer(elem.tmat)
        if endpoint:
            beta.append(b0)
        return beta
