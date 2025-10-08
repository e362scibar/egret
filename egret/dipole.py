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
from .betafunc import BetaFunc

import numpy as np
import numpy.typing as npt
from typing import Tuple
import scipy

class Dipole(Element):
    '''
    Dipole magnet.
    '''

    def __init__(self, name: str, length: float, angle: float, k1: float = 0.,
                 e1: float = 0., e2: float = 0., h1: float = 0., h2: float = 0.,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = ''):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            angle float: Bending angle of the dipole [rad].
            k1 float: Quadrupole component [1/m^2].
            e1 float: Entrance edge angle [rad].
            e2 float: Exit edge angle [rad].
            h1 float: Entrance pole-face curvature [1/m].
            h2 float: Exit pole-face curvature [1/m].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        if angle == 0.:
            raise ValueError(f'Angle is zero.')
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.angle = angle
        self.radius = length / angle
        self.k1 = k1
        self.e1 = e1
        self.e2 = e2
        self.h1 = h1
        self.h2 = h2
        self.update()

    def update(self):
        '''
        Update transfer matrix and dispersion.
        '''
        phi = self.angle
        rho = self.radius
        if self.k1 == 0.: # simple dipole
            self.tmat[0:2, 0:2] = np.array([[np.cos(phi), rho*np.sin(phi)],
                                            [-np.sin(phi)/rho, np.cos(phi)]])
            self.tmat[2, 3] = rho*phi
            self.disp[0:2] = np.array([rho*(1.-np.cos(phi)), np.sin(phi)])
        else: # combined-function dipole
            kx = np.abs(self.k1 + 1./rho**2)
            psix = np.sqrt(kx) * rho * phi
            ky = np.abs(self.k1)
            psiy = np.sqrt(ky) * rho * phi
            if self.k1 < 0.: # defocusing dipole
                self.tmat[0:2, 0:2] = np.array([[np.cosh(psix), np.sinh(psix)/np.sqrt(kx)],
                                                [np.sqrt(kx)*np.sinh(psix), np.cosh(psix)]])
                self.tmat[2:4, 2:4] = np.array([[np.cos(psiy), np.sin(psiy)/np.sqrt(ky)],
                                                [-np.sqrt(ky)*np.sin(psiy), np.cos(psiy)]])
                self.disp[0:2] = np.array([(np.cosh(psix)-1.)/(kx*rho), np.sinh(psix)/(np.sqrt(kx)*rho)])
            else: # focusing dipole
                self.tmat[0:2, 0:2] = np.array([[np.cos(psix), np.sin(psix)/np.sqrt(kx)],
                                                [-np.sqrt(kx)*np.sin(psix), np.cos(psix)]])
                self.tmat[2:4, 2:4] = np.array([[np.cosh(psiy), np.sinh(psiy)/np.sqrt(ky)],
                                                [np.sqrt(ky)*np.sinh(psiy), np.cosh(psiy)]])
                self.disp[0:2] = np.array([(1.-np.cos(psix))/(kx*rho), np.sin(psix)/(np.sqrt(kx)*rho)])

    def tmatarray(self, ds: float = 0.01, endpoint: bool = False) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix array along the dipole.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            
        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        phi = np.linspace(0., self.angle, int(self.length//ds) + int(endpoint) + 1, endpoint)
        rho = self.radius
        tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(phi), axis=0)
        if self.k1 == 0.: # simple dipole
            tmat[:, 0:2, 0:2] = np.moveaxis(np.array([[np.cos(phi), rho*np.sin(phi)],
                                                      [-np.sin(phi)/rho, np.cos(phi)]]), 2, 0)
            tmat[:, 2, 3] = rho * phi
        else:
            kx = np.abs(self.k1 + 1./rho**2)
            psix = np.sqrt(kx) * rho * phi
            ky = np.abs(self.k1)
            psiy = np.sqrt(ky) * rho * phi
            if self.k1 < 0.: # defocusing dipole
                tmat[:, 0:2, 0:2] = np.moveaxis(np.array([[np.cosh(psix), np.sinh(psix)/np.sqrt(kx)],
                                                          [np.sqrt(kx)*np.sinh(psix), np.cosh(psix)]]), 2, 0)
                tmat[:, 2:4, 2:4] = np.moveaxis(np.array([[np.cos(psiy), np.sin(psiy)/np.sqrt(ky)],
                                                          [-np.sqrt(ky)*np.sin(psiy), np.cos(psiy)]]), 2, 0)
            else: # focusing dipole
                tmat[:, 0:2, 0:2] = np.moveaxis(np.array([[np.cos(psix), np.sin(psix)/np.sqrt(kx)],
                                                          [-np.sqrt(kx)*np.sin(psix), np.cos(psix)]]), 2, 0)
                tmat[:, 2:4, 2:4] = np.moveaxis(np.array([[np.cosh(psiy), np.sinh(psiy)/np.sqrt(ky)],
                                                          [np.sqrt(ky)*np.sinh(psiy), np.cosh(psiy)]]), 2, 0)
        return tmat, rho * phi

    def dispersion(self, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Dispersion function along the dipole.
        
        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
        
        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        phi = np.linspace(0., self.angle, int(self.length//ds) + int(endpoint) + 1, endpoint)
        rho = self.radius
        s = rho * phi
        disp = np.zeros((4, len(s)))
        if self.k1 == 0.: # simple dipole
            disp[0:2, :] = np.array([rho*(1.-np.cos(phi)), np.sin(phi)])
        else:
            kx = np.abs(self.k1 + 1./rho**2)
            psix = np.sqrt(kx) * rho * phi
            ky = np.abs(self.k1)
            psiy = np.sqrt(ky) * rho * phi
            if self.k1 < 0.: # defocusing dipole
                disp[0:2, :] = np.array([(np.cosh(psix)-1.)/(kx*rho), np.sinh(psix)/(np.sqrt(kx)*rho)])
            else: # focusing dipole
                disp[0:2, :] = np.array([(1.-np.cos(psix))/(kx*rho), np.sin(psix)/(np.sqrt(kx)*rho)])
        return disp, s

    def radiation_integrals(self, beta0: BetaFunc, eta0: npt.NDArray[np.floating], ds: float = 0.1) \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            beta0 BetaFunc: Initial Twiss parameters.
            eta0 npt.NDArray[np.floating]: Initial dispersion [eta_x, eta_x', eta_y, eta_y'].
            ds float: Step size for numerical integration.

        Returns:
            float, float, float: Radiation integrals I2, I4, and I5.
        '''
        kappa = 1./self.radius
        k = self.k1
        beta = self.betafunc(beta0, ds, endpoint=True)
        eta, s = self.etafunc(eta0, ds, endpoint=True)
        I2 = self.length * kappa**2
        I4 = scipy.integrate.simpson(eta[0, :] * kappa * (kappa**2 + 2. * k), x=s)
        I5 = scipy.integrate.simpson(kappa**3 * (beta['bx'] * eta[1, :]**2 + 2. * beta['ax'] * eta[0, :] * eta[1, :] + beta['gx'] * eta[0, :]**2), x=s)
        return I2, I4, I5
