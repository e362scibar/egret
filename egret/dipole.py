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

from __future__ import annotations

from .element import Element
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray
from .drift import Drift

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

    def copy(self) -> Dipole:
        '''
        Return a copy of the dipole element.

        Returns:
            Dipole: Copied dipole element.
        '''
        return Dipole(self.name, self.length, self.angle, self.k1,
                      self.e1, self.e2, self.h1, self.h2,
                      self.dx, self.dy, self.ds, self.tilt, self.info)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the dipole element.

        Args:
            cood0 Coordinate: Initial coordinate (only delta is used in the dipole class).
            ds float: Maximum step size [m] for integration. (not used in the dipole class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        delta = 0. if cood0 is None else cood0.delta
        rho = self.radius * (1. + delta)
        tmat = np.eye(4)
        if self.k1 == 0.: # simple dipole
            phi = self.angle / (1. + delta)
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            tmat[0:2, 0:2] = np.array([[cosphi, rho*sinphi], [-sinphi/rho, cosphi]])
            tmat[2, 3] = self.length
        else: # combined-function dipole
            k1 = self.k1 / (1. + delta)
            kx = k1 + 1./rho**2
            sqrtkx = np.sqrt(np.abs(kx))
            psix = sqrtkx * self.length
            ky = -k1
            sqrtky = np.sqrt(np.abs(ky))
            psiy = sqrtky * self.length
            if kx < 0.: # defocusing dipole
                coshx, sinhx = np.cosh(psix), np.sinh(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                tmat[0:2, 0:2] = np.array([[coshx, sinhx/sqrtkx], [sqrtkx*sinhx, coshx]])
                tmat[2:4, 2:4] = np.array([[cosy, siny/sqrtky], [-sqrtky*siny, cosy]])
            elif ky < 0.: # focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                coshy, sinhy = np.cosh(psiy), np.sinh(psiy)
                tmat[0:2, 0:2] = np.array([[cosx, sinx/sqrtkx], [-sqrtkx*sinx, cosx]])
                tmat[2:4, 2:4] = np.array([[coshy, sinhy/sqrtky], [sqrtky*sinhy, coshy]])
            else: # both focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                tmat[0:2, 0:2] = np.array([[cosx, sinx/sqrtkx], [-sqrtkx*sinx, cosx]])
                tmat[2:4, 2:4] = np.array([[cosy, siny/sqrtky], [-sqrtky*siny, cosy]])
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the dipole element.

        Args:
            cood0 Coordinate: Initial coordinate (only delta is used in the dipole class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        delta = 0. if cood0 is None else cood0.delta
        rho = self.radius * (1. + delta)
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
        tmat = np.repeat(np.eye(4)[:,:,np.newaxis], len(s), axis=2)
        if self.k1 == 0.: # simple dipole
            phi = s / rho
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            tmat[0:2,0:2] = np.array([[cosphi, rho*sinphi], [-sinphi/rho, cosphi]])
            tmat[2,3] = s
        else: # combined-function dipole
            k1 = self.k1 / (1. + delta)
            kx = k1 + 1./rho**2
            sqrtkx = np.sqrt(np.abs(kx))
            psix = sqrtkx * s
            ky = -k1
            sqrtky = np.sqrt(np.abs(ky))
            psiy = sqrtky * s
            if kx < 0.: # defocusing dipole
                coshx, sinhx = np.cosh(psix), np.sinh(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                tmat[0:2,0:2] = np.array([[coshx, sinhx/sqrtkx], [sqrtkx*sinhx, coshx]])
                tmat[2:4,2:4] = np.array([[cosy, siny/sqrtky], [-sqrtky*siny, cosy]])
            elif ky < 0.: # focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                coshy, sinhy = np.cosh(psiy), np.sinh(psiy)
                tmat[0:2,0:2] = np.array([[cosx, sinx/sqrtkx], [-sqrtkx*sinx, cosx]])
                tmat[2:4,2:4] = np.array([[coshy, sinhy/sqrtky], [sqrtky*sinhy, coshy]])
            else: # both focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                tmat[0:2, 0:2] = np.array([[cosx, sinx/sqrtkx], [-sqrtkx*sinx, cosx]])
                tmat[2:4, 2:4] = np.array([[cosy, siny/sqrtky], [-sqrtky*siny, cosy]])
        return tmat, s

    def dispersion(self, cood0: Coordinate) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion function at the end of the dipole.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            npt.NDArray[np.floating]: Additive dispersion function [eta_x, eta_x', eta_y, eta_y'].
        '''
        rho = self.radius * (1. + cood0.delta)
        cood0vec = cood0.vector.copy()
        if self.k1 == 0.: # simple dipole
            phi = self.length / rho
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            disp = np.array([rho*(1.-cosphi), sinphi, 0., 0.])
            Mx1 = np.array([[sinphi, -rho*cosphi], [cosphi/rho, sinphi]]) * 0.5 * self.length / rho
            Mx2 = np.array([[0., rho*sinphi], [sinphi/rho, 0.]]) * 0.5
            disp[0:2] += np.dot(Mx1 + Mx2, cood0vec[0:2])
        else: # combined-function dipole
            k1 = self.k1 / (1. + cood0.delta)
            kx = k1 + 1./rho**2
            sqrtkx = np.sqrt(np.abs(kx))
            psix = sqrtkx * self.length
            ky = -k1
            sqrtky = np.sqrt(np.abs(ky))
            psiy = sqrtky * self.length
            if kx < 0.: # defocusing dipole
                coshx, sinhx = np.cosh(psix), np.sinh(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                disp = np.array([(1.-coshx)/(kx*rho), sinhx/(sqrtkx*rho), 0., 0.])
                Mx1 = np.array([[-sinhx, -coshx/sqrtkx], [-sqrtkx*coshx, -sinhx]]) * 0.5 * self.length * sqrtkx
                Mx2 = np.array([[0., sinhx/sqrtkx], [-sqrtkx*sinhx, 0.]]) * 0.5
                disp[0:2] += np.dot(Mx1 + Mx2, cood0vec[0:2])
                My1 = np.array([[siny, -cosy/sqrtky], [sqrtky*cosy, siny]]) * 0.5 * self.length * sqrtky
                My2 = np.array([[0., siny/sqrtky], [sqrtky*siny, 0.]]) * 0.5
                disp[2:4] += np.dot(My1 + My2, cood0vec[2:4])
            elif ky < 0.: # focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                coshy, sinhy = np.cosh(psiy), np.sinh(psiy)
                disp = np.array([(1.-cosx)/(kx*rho), sinx/(sqrtkx*rho), 0., 0.])
                Mx1 = np.array([[sinx, -cosx/sqrtkx], [sqrtkx*cosx, sinx]]) * 0.5 * self.length * sqrtkx
                Mx2 = np.array([[0., sinx/sqrtkx], [sqrtkx*sinx, 0.]]) * 0.5
                disp[0:2] += np.dot(Mx1 + Mx2, cood0vec[0:2])
                My1 = np.array([[-sinhy, -coshy/sqrtky], [-sqrtky*coshy, -sinhy]]) * 0.5 * self.length * sqrtky
                My2 = np.array([[0., sinhy/sqrtky], [-sqrtky*sinhy, 0.]]) * 0.5
                disp[2:4] += np.dot(My1 + My2, cood0vec[2:4])
            else: # both focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                disp = np.array([(1.-cosx)/(kx*rho), sinx/(sqrtkx*rho), 0., 0.])
                Mx1 = np.array([[sinx, -cosx/sqrtkx], [sqrtkx*cosx, sinx]]) * 0.5 * self.length * sqrtkx
                Mx2 = np.array([[0., sinx/sqrtkx], [sqrtkx*sinx, 0.]]) * 0.5
                disp[0:2] += np.dot(Mx1 + Mx2, cood0vec[0:2])
                My1 = np.array([[siny, -cosy/sqrtky], [sqrtky*cosy, siny]]) * 0.5 * self.length * sqrtky
                My2 = np.array([[0., siny/sqrtky], [sqrtky*siny, 0.]]) * 0.5
                disp[2:4] += np.dot(My1 + My2, cood0vec[2:4])
        return disp

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion function along the dipole.

        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        rho = self.radius * (1. + cood0.delta)
        cood0vec = cood0.vector.copy()
        s = np.linspace(0., self.length, int(self.length//ds) + int(endpoint) + 1, endpoint)
        if self.k1 == 0.: # simple dipole
            phi = s / rho
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            disp = np.array([rho*(1.-cosphi), sinphi, np.zeros_like(s), np.zeros_like(s)])
            Mx1 = np.array([[sinphi, -rho*cosphi], [cosphi/rho, sinphi]]) * 0.5 * s[np.newaxis,np.newaxis,:] / rho
            Mx2 = np.array([[np.zeros_like(s), rho*sinphi], [sinphi/rho, np.zeros_like(s)]]) * 0.5
            disp[0:2,:] += np.matmul((Mx1 + Mx2).transpose(2,0,1), cood0vec[0:2]).T
        else:
            k1 = self.k1 / (1. + cood0.delta)
            kx = k1 + 1./rho**2
            sqrtkx = np.sqrt(np.abs(kx))
            psix = sqrtkx * s
            ky = -k1
            sqrtky = np.sqrt(np.abs(ky))
            psiy = sqrtky * s
            if kx < 0.: # defocusing dipole
                coshx, sinhx = np.cosh(psix), np.sinh(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                disp = np.array([(1.-coshx)/(kx*rho), sinhx/(sqrtkx*rho), np.zeros_like(s), np.zeros_like(s)])
                Mx1 = np.array([[-sinhx, -coshx/sqrtkx], [-sqrtkx*coshx, -sinhx]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtkx
                Mx2 = np.array([[np.zeros_like(s), sinhx/sqrtkx], [-sqrtkx*sinhx, np.zeros_like(s)]]) * 0.5
                disp[0:2,:] += np.matmul((Mx1 + Mx2).transpose(2,0,1), cood0vec[0:2]).T
                My1 = np.array([[siny, -cosy/sqrtky], [sqrtky*cosy, siny]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtky
                My2 = np.array([[np.zeros_like(s), siny/sqrtky], [sqrtky*siny, np.zeros_like(s)]]) * 0.5
                disp[2:4,:] += np.matmul((My1 + My2).transpose(2,0,1), cood0vec[2:4]).T
            elif ky < 0.: # focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                coshy, sinhy = np.cosh(psiy), np.sinh(psiy)
                disp = np.array([(1.-cosx)/(kx*rho), sinx/(sqrtkx*rho), np.zeros_like(s), np.zeros_like(s)])
                Mx1 = np.array([[sinx, -cosx/sqrtkx], [sqrtkx*cosx, sinx]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtkx
                Mx2 = np.array([[np.zeros_like(s), sinx/sqrtkx], [sqrtkx*sinx, np.zeros_like(s)]]) * 0.5
                disp[0:2,:] += np.matmul((Mx1 + Mx2).transpose(2,0,1), cood0vec[0:2]).T
                My1 = np.array([[-sinhy, -coshy/sqrtky], [-sqrtky*coshy, -sinhy]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtky
                My2 = np.array([[np.zeros_like(s), sinhy/sqrtky], [-sqrtky*sinhy, np.zeros_like(s)]]) * 0.5
                disp[2:4,:] += np.matmul((My1 + My2).transpose(2,0,1), cood0vec[2:4]).T
            else: # both focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                disp = np.array([(1.-cosx)/(kx*rho), sinx/(sqrtkx*rho), np.zeros_like(s), np.zeros_like(s)])
                Mx1 = np.array([[sinx, -cosx/sqrtkx], [sqrtkx*cosx, sinx]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtkx
                Mx2 = np.array([[np.zeros_like(s), sinx/sqrtkx], [sqrtkx*sinx, np.zeros_like(s)]]) * 0.5
                disp[0:2,:] += np.matmul((Mx1 + Mx2).transpose(2,0,1), cood0vec[0:2]).T
                My1 = np.array([[siny, -cosy/sqrtky], [sqrtky*cosy, siny]]) * 0.5 * s[np.newaxis,np.newaxis,:] * sqrtky
                My2 = np.array([[np.zeros_like(s), siny/sqrtky], [sqrtky*siny, np.zeros_like(s)]]) * 0.5
                disp[2:4,:] += np.matmul((My1 + My2).transpose(2,0,1), cood0vec[2:4]).T
        return disp, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the Dipole class).

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood0err = cood0.copy()
        cood0err.vector[0] -= self.dx
        cood0err.vector[2] -= self.dy
        cood0err.s -= self.ds
        tmat = self.transfer_matrix(cood0err)
        disp = self.dispersion(Coordinate())
        cood = np.dot(tmat, cood0err.vector) + disp * cood0.delta
        cood[0] += self.dx
        cood[2] += self.dy
        cood1 = Coordinate(cood, cood0err.s + self.length + self.ds, cood0err.z, cood0err.delta)
        if evlp0 is not None:
            evlp1 = evlp0.copy()
            evlp1.transfer(tmat, self.length)
        else:
            evlp1 = None
        if disp0 is not None:
            disp = np.dot(tmat, disp0.vector) + self.dispersion(cood0err)
            disp1 = Dispersion(disp, disp0.s + self.length)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        cood0err = cood0.copy()
        cood0err.vector[0] -= self.dx
        cood0err.vector[2] -= self.dy
        cood0err.s -= self.ds
        tmat, s = self.transfer_matrix_array(cood0err, ds, endpoint)
        disp, _ = self.dispersion_array(Coordinate(), ds, endpoint)
        cood = np.matmul(tmat.transpose(2,0,1), cood0err.vector).T + disp * cood0.delta
        cood[0] += self.dx
        cood[2] += self.dy
        cood1 = CoordinateArray(cood, s + cood0.s + self.ds,
                                np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
        if evlp0 is not None:
            evlp1 = EnvelopeArray.transport(evlp0, tmat, s)
        else:
            evlp1 = None
        if disp0 is not None:
            disp_add, _ = self.dispersion_array(cood0err, ds, endpoint)
            disp = np.matmul(tmat.transpose(2,0,1), disp0.vector).T + disp_add
            disp1 = DispersionArray(disp, s + disp0.s)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def radiation_integrals(self, cood0: Coordinate, evlp0: Envelope, disp0: Dispersion, ds: float = 0.1) \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            beta0 BetaFunc: Initial Twiss parameters.
            eta0 npt.NDArray[np.floating]: Initial dispersion [eta_x, eta_x', eta_y, eta_y'].
            ds float: Step size for numerical integration.

        Returns:
            Tuple[float, float, float, float, float, float]: Radiation integrals I2, I4, I5u, I5v, I4u, and I4v.
        '''
        kappa = 1./self.radius
        k = self.k1
        _, evlp, disp = self.transfer_array(cood0, evlp0, disp0, ds, endpoint=True)
        dispuv = np.matvec(evlp.T_matrix().transpose(2, 0, 1), disp.vector.T).T
        I2 = self.length * kappa**2
        I4 = scipy.integrate.simpson(disp['x'] * kappa * (kappa**2 + 2. * k), x=disp.s)
        I4u = scipy.integrate.simpson(evlp.tau * dispuv[0] * kappa * (kappa**2 + 2. * k), x=disp.s)
        I4v = I4 - I4u
        I5u = scipy.integrate.simpson(kappa**3 * (evlp['bu'] * dispuv[1]**2 + 2. * evlp['au'] * dispuv[0] * dispuv[1] + evlp['gu'] * dispuv[0]**2), x=disp.s)
        I5v = scipy.integrate.simpson(kappa**3 * (evlp['bv'] * dispuv[3]**2 + 2. * evlp['av'] * dispuv[2] * dispuv[3] + evlp['gv'] * dispuv[2]**2), x=disp.s)
        return I2, I4, I5u, I5v, I4u, I4v