# dipole.py
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

import numpy as np

class Dipole(Element):
    """
    Dipole magnet.
    """
    def __init__(self, name, length, angle, radius, k1=0., e1=0., e2=0., h1=0., h2=0., dx=0., dy=0., ds=0., tilt=0., info=''):
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.angle = angle
        self.radius = radius
        self.k1 = k1
        self.e1 = e1  # edge angle at the entrance
        self.e2 = e2  # edge angle at the exit
        self.h1 = h1  # ??
        self.h2 = h2  # ??
        self.update()

    def update(self):
        phi = self.angle
        rho = self.radius
        self.tmat[:2,:2] = np.array([[np.cos(phi), rho*np.sin(phi)], [-np.sin(phi)/rho, np.cos(phi)]])
        self.tmat[3,4] = rho*phi
        self.disp[:2] = np.array([rho*(1.-np.cos(phi)), np.sin(phi)])
