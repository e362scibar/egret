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
import numpy.typing as npt

class Dipole(Element):
    """
    Dipole magnet.
    """
    def __init__(self, name:str, length:float, angle:float, k1:float=0.,
                 e1:float=0., e2:float=0., h1:float=0., h2:float=0.,
                 dx:float=0., dy:float=0., ds:float=0., tilt:float=0., info:str=''):
        if angle == 0.:
            raise ValueError(f'Angle is zero.')
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.angle = angle
        self.radius = length / angle
        self.k1 = k1
        self.e1 = e1  # edge angle at the entrance
        self.e2 = e2  # edge angle at the exit
        self.h1 = h1  # ??
        self.h2 = h2  # ??
        self.update()

    def update(self):
        phi = self.angle
        rho = self.radius
        if self.k1 == 0.: # simple dipole
            self.tmat[0:2,0:2] = np.array([[np.cos(phi), rho*np.sin(phi)],
                                           [-np.sin(phi)/rho, np.cos(phi)]])
            self.tmat[2,3] = rho*phi
            self.disp[0:2] = np.array([rho*(1.-np.cos(phi)), np.sin(phi)])
        else:
            kx = np.abs(self.k1 + 1./rho**2)
            psix = np.sqrt(kx) * rho * phi
            ky = np.abs(self.k1)
            psiy = np.sqrt(ky) * rho * phi
            if self.k1 < 0.: # defocusing dipole
                self.tmat[0:2,0:2] = np.array([[np.cosh(psix), np.sinh(psix)/np.sqrt(kx)],
                                               [np.sqrt(kx)*np.sinh(psix), np.cosh(psix)]])
                self.tmat[2:4,2:4] = np.array([[np.cos(psiy), np.sin(psiy)/np.sqrt(ky)],
                                               [-np.sqrt(ky)*np.sin(psiy), np.cos(psiy)]])
                self.disp[0:2] = np.array([(np.cosh(psix)-1.)/(kx*rho), np.sinh(psix)/(np.sqrt(kx)*rho)])
            else: # focusing dipole
                self.tmat[0:2,0:2] = np.array([[np.cos(psix), np.sin(psix)/np.sqrt(kx)],
                                               [-np.sqrt(kx)*np.sin(psix), np.cos(psix)]])
                self.tmat[2:4,2:4] = np.array([[np.cosh(psiy), np.sinh(psiy)/np.sqrt(ky)],
                                               [np.sqrt(ky)*np.sinh(psiy), np.cosh(psiy)]])
                self.disp[0:2] = np.array([(1.-np.cos(psix))/(kx*rho), np.sin(psix)/(np.sqrt(kx)*rho)])

    def tmatarray(self, ds:float=0.01, endpoint:bool=False)->npt.NDArray[np.floating]:
        phi = np.linspace(0., self.angle, int(self.length//ds)+int(endpoint)+1, endpoint)
        rho = self.radius
        tmat = np.repeat(np.eye(6)[np.newaxis,:,:], len(phi), axis=0)
        if self.k1 == 0.: # simple dipole
            tmat[:,0:2,0:2] = np.moveaxis(np.array([[np.cos(phi), rho*np.sin(phi)],
                                                    [-np.sin(phi)/rho, np.cos(phi)]]), -1, 0)
            tmat[:,2,3] = rho*phi
        else:
            kx = np.abs(self.k1 + 1./rho**2)
            psix = np.sqrt(kx) * rho * phi
            ky = np.abs(self.k1)
            psiy = np.sqrt(ky) * rho * phi
            if self.k1 < 0.: # defocusing dipole
                tmat[:,0:2,0:2] = np.moveaxis(np.array([[np.cosh(psix), np.sinh(psix)/np.sqrt(kx)],
                                                        [np.sqrt(kx)*np.sinh(psix), np.cosh(psix)]]), -1, 0)
                tmat[:,2:4,2:4] = np.moveaxis(np.array([[np.cos(psiy), np.sin(psiy)/np.sqrt(ky)],
                                                        [-np.sqrt(ky)*np.sin(psiy), np.cos(psiy)]]), -1, 0)
            else: # focusing dipole
                tmat[:,0:2,0:2] = np.moveaxis(np.array([[np.cos(psix), np.sin(psix)/np.sqrt(kx)],
                                                        [-np.sqrt(kx)*np.sin(psix), np.cos(psix)]]), -1, 0)
                tmat[:,2:4,2:4] = np.moveaxis(np.array([[np.cosh(psiy), np.sinh(psiy)/np.sqrt(ky)],
                                                        [np.sqrt(ky)*np.sinh(psiy), np.cosh(psiy)]]), -1, 0)
        return tmat
