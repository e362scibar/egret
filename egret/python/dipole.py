# python/dipole.py
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
from ..base.dipole import Dipole as DipoleABC
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
from typing import Final, Tuple
import scipy

# Focusing strengths below this magnitude are treated as a drift in the transfer block
FOCUSING_IGNORE_THRESHOLD: Final = 1.0e-15


def _thick_block(strength: float, length: float) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    '''
    Transfer block of x'' + strength * x = 0 over a section, and its derivative with respect to the strength.

    Args:
        strength float: Focusing strength K [1/m^2]. Negative values are defocusing.
        length float: Length of the section [m].

    Returns:
        npt.NDArray[np.floating]: 2x2 transfer block [[c, s], [-K s, c]].
        npt.NDArray[np.floating]: 2x2 derivative of the transfer block with respect to K.
    '''
    if abs(strength) < FOCUSING_IGNORE_THRESHOLD: # drift limit; O(K) corrections are negligible
        c, s = 1., length
        ds_dk = -length**3 / 6.
    else:
        root = np.sqrt(np.abs(strength))
        psi = root * length
        if strength > 0.: # focusing
            c, s = np.cos(psi), np.sin(psi) / root
        else: # defocusing
            c, s = np.cosh(psi), np.sinh(psi) / root
        ds_dk = (length * c - s) / (2. * strength)
    block = np.array([[c, s], [-strength * s, c]])
    block_derivative = np.array([[-0.5 * length * s, ds_dk], [-0.5 * (s + length * c), -0.5 * length * s]])
    return block, block_derivative


class Dipole(DipoleABC, Element):
    '''
    Dipole magnet class.
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
        super().__init__(name, length, angle, dx, dy, ds, tilt, info)
        self._k1 = k1
        self._e1 = e1
        self._e2 = e2
        self._h1 = h1
        self._h2 = h2

    @property
    def rho(self) -> float:
        '''
        Bending radius of the dipole [m], computed from the current length and angle.
        '''
        return self._length / self._angle

    @Element.angle.setter
    def angle(self, angle: float) -> None:
        '''
        Set the bending angle of the dipole. The bending radius follows from the length and this angle.

        Args:
            angle float: Bending angle of the dipole [rad].

        Raises:
            ValueError: If angle is zero.
        '''
        if angle == 0.:
            raise ValueError(f'Angle is zero.')
        Element.angle.fset(self, angle)

    @property
    def k1(self) -> float:
        '''
        Quadrupole component [1/m^2].
        '''
        return self._k1

    @property
    def e1(self) -> float:
        '''
        Entrance edge angle [rad].
        '''
        return self._e1

    @property
    def e2(self) -> float:
        '''
        Exit edge angle [rad].
        '''
        return self._e2

    @property
    def h1(self) -> float:
        '''
        Entrance pole-face curvature [1/m].
        '''
        return self._h1

    @property
    def h2(self) -> float:
        '''
        Exit pole-face curvature [1/m].
        '''
        return self._h2

    @k1.setter
    def k1(self, k1: float) -> None:
        '''
        Set quadrupole component.

        Args:
            k1 float: Quadrupole component [1/m^2].
        '''
        self._k1 = k1

    @e1.setter
    def e1(self, e1: float) -> None:
        '''
        Set entrance edge angle.

        Args:
            e1 float: Entrance edge angle [rad].
        '''
        self._e1 = e1

    @e2.setter
    def e2(self, e2: float) -> None:
        '''
        Set exit edge angle.

        Args:
            e2 float: Exit edge angle [rad].
        '''
        self._e2 = e2

    @h1.setter
    def h1(self, h1: float) -> None:
        '''
        Set entrance pole-face curvature.

        Args:
            h1 float: Entrance pole-face curvature [1/m].
        '''
        self._h1 = h1

    @h2.setter
    def h2(self, h2: float) -> None:
        '''
        Set exit pole-face curvature.

        Args:
            h2 float: Exit pole-face curvature [1/m].
        '''
        self._h2 = h2

    def copy(self) -> Dipole:
        '''
        Return a copy of the dipole element.

        Returns:
            Dipole: Copied dipole element.
        '''
        return Dipole(self._name, self._length, self._angle, self._k1,
                      self._e1, self._e2, self._h1, self._h2,
                      self._dx, self._dy, self._ds, self._tilt, self._info)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the dipole element.

        Args:
            cood0 Coordinate: Initial coordinate (only delta is used in the dipole class).
            ds float: Maximum step size [m] for integration. (not used in the dipole class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}'). (not used in the dipole class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        delta = 0. if cood0 is None else cood0.delta
        rho = self.rho * (1. + delta)
        tmat = np.eye(4)
        if self._k1 == 0.: # simple dipole
            phi = self._angle / (1. + delta)
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            tmat[0:2, 0:2] = np.array([[cosphi, rho*sinphi], [-sinphi/rho, cosphi]])
            tmat[2, 3] = self._length
        else: # combined-function dipole
            k1 = self._k1 / (1. + delta)
            kx = k1 + 1./rho**2
            sqrtkx = np.sqrt(np.abs(kx))
            psix = sqrtkx * self._length
            ky = -k1
            sqrtky = np.sqrt(np.abs(ky))
            psiy = sqrtky * self._length
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

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the dipole element.

        Args:
            cood0 Coordinate: Initial coordinate (only delta is used in the dipole class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}'). (not used in the dipole class).

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        delta = 0. if cood0 is None else cood0.delta
        rho = self.rho * (1. + delta)
        s = self.s_array(ds, endpoint)
        tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
        if self._k1 == 0.: # simple dipole
            phi = s / rho
            cosphi, sinphi = np.cos(phi), np.sin(phi)
            tmat[:,0:2,0:2] = np.array([[cosphi, rho*sinphi], [-sinphi/rho, cosphi]]).transpose(2,0,1)
            tmat[:,2,3] = s
        else: # combined-function dipole
            k1 = self._k1 / (1. + delta)
            kx = k1 + 1./rho**2
            sqrtkx = np.sqrt(np.abs(kx))
            psix = sqrtkx * s
            ky = -k1
            sqrtky = np.sqrt(np.abs(ky))
            psiy = sqrtky * s
            if kx < 0.: # defocusing dipole
                coshx, sinhx = np.cosh(psix), np.sinh(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                tmat[:,0:2,0:2] = np.array([[coshx, sinhx/sqrtkx], [sqrtkx*sinhx, coshx]]).transpose(2,0,1)
                tmat[:,2:4,2:4] = np.array([[cosy, siny/sqrtky], [-sqrtky*siny, cosy]]).transpose(2,0,1)
            elif ky < 0.: # focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                coshy, sinhy = np.cosh(psiy), np.sinh(psiy)
                tmat[:,0:2,0:2] = np.array([[cosx, sinx/sqrtkx], [-sqrtkx*sinx, cosx]]).transpose(2,0,1)
                tmat[:,2:4,2:4] = np.array([[coshy, sinhy/sqrtky], [sqrtky*sinhy, coshy]]).transpose(2,0,1)
            else: # both focusing dipole
                cosx, sinx = np.cos(psix), np.sin(psix)
                cosy, siny = np.cos(psiy), np.sin(psiy)
                tmat[:,0:2, 0:2] = np.array([[cosx, sinx/sqrtkx], [-sqrtkx*sinx, cosx]]).transpose(2,0,1)
                tmat[:,2:4, 2:4] = np.array([[cosy, siny/sqrtky], [-sqrtky*siny, cosy]]).transpose(2,0,1)
        return tmat, s

    def _matrix_derivative(self, delta: float, length: float) -> npt.NDArray[np.floating]:
        '''
        Derivative of the transfer matrix of a section with respect to the relative momentum deviation.

        The bending radius rho (1 + delta) and the gradient k1 / (1 + delta) both depend on delta. The horizontal
        focusing strength is K_x = k1 / (1 + delta) + 1 / rho^2 and the vertical one is K_y = -k1 / (1 + delta).

        Args:
            delta float: Relative momentum deviation.
            length float: Length of the section [m].

        Returns:
            npt.NDArray[np.floating]: 4x4 matrix dM / d(delta).
        '''
        k = self._k1 / (1. + delta)
        rho = self.rho * (1. + delta)
        dk_x = -(k + 2. / rho**2) / (1. + delta)
        dk_y = k / (1. + delta)
        tmat = np.zeros((4, 4))
        tmat[0:2, 0:2] = _thick_block(k + 1. / rho**2, length)[1] * dk_x
        tmat[2:4, 2:4] = _thick_block(-k, length)[1] * dk_y
        return tmat

    def _inhomogeneous_dispersion(self, delta: float, length: float) -> npt.NDArray[np.floating]:
        '''
        Dispersion generated by the bending field at a given relative momentum deviation, for a zero orbit.

        Args:
            delta float: Relative momentum deviation.
            length float: Length of the section [m].

        Returns:
            npt.NDArray[np.floating]: Dispersion [eta_x, eta_x', eta_y, eta_y'].
        '''
        rho = self.rho * (1. + delta)
        kx = self._k1 / (1. + delta) + 1. / rho**2
        block, _ = _thick_block(kx, length)
        return np.array([(1. - block[0, 0]) / (kx * rho), block[0, 1] / rho, 0., 0.])

    def _inhomogeneous_dispersion_array(self, delta: float, s: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        '''
        Array of the zero-orbit dispersion along the dipole.

        Args:
            delta float: Relative momentum deviation.
            s npt.NDArray[np.floating]: Longitudinal positions inside the dipole [m].

        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
        '''
        return np.array([self._inhomogeneous_dispersion(delta, length) for length in s]).T

    def _orbit_dispersion(self, cood: Coordinate, length: float) -> npt.NDArray[np.floating]:
        '''
        Dispersion term proportional to the orbit: M'(delta) v, where v is relative to the element offset.

        Args:
            cood Coordinate: Coordinate at the entrance of the element, after the entrance drift and offset removal.
            length float: Length of the section [m].

        Returns:
            npt.NDArray[np.floating]: Dispersion [eta_x, eta_x', eta_y, eta_y'].
        '''
        return self._matrix_derivative(cood.delta, length) @ cood.vector

    def dispersion(self, cood0: Coordinate, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Additive dispersion function at the end of the dipole.

        The dispersion is the momentum derivative of the transfer map at fixed input: the zero-orbit part plus M'(delta) v.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m] for integration. (not used in the dipole class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}'). (not used in the dipole class).

        Returns:
            npt.NDArray[np.floating]: Additive dispersion function [eta_x, eta_x', eta_y, eta_y'].
        '''
        cood, _, _ = self.drift_transfer(self._ds, cood0, None, None)
        cood.x -= self._dx
        cood.y -= self._dy
        return self._inhomogeneous_dispersion(cood0.delta, self._length) + self._orbit_dispersion(cood, self._length)

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion function along the dipole.

        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}'). (not used in the dipole class).

        Returns:
            npt.NDArray[np.floating]: Dispersion function array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        cood, _, _ = self.drift_transfer(self._ds, cood0, None, None)
        cood.x -= self._dx
        cood.y -= self._dy
        s = self.s_array(ds, endpoint)
        disp = self._inhomogeneous_dispersion_array(cood0.delta, s)
        orbit = np.array([self._orbit_dispersion(cood, length) for length in s]).T
        return disp + orbit, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1, method: str = 'symplectic4') \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the Dipole class).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}'). (not used in the Dipole class).

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood, evlp, disp = self.drift_transfer(self._ds, cood0, evlp0, disp0)
        cood.x -= self._dx
        cood.y -= self._dy
        tmat = self.transfer_matrix(cood0, ds)
        disp_add = self._inhomogeneous_dispersion(cood0.delta, self._length)
        coodvec = np.dot(tmat, cood.vector) + disp_add * cood.delta
        cood1 = Coordinate(coodvec, cood0.s + self._length, cood0.z, cood0.delta)
        if evlp is not None:
            evlp1 = evlp
            evlp1.transfer(tmat, self._length)
        else:
            evlp1 = None
        if disp is not None:
            disp1vec = np.dot(tmat, disp.vector) + self.dispersion(cood0, ds)
            disp1 = Dispersion(disp1vec, disp.s + self._length)
        else:
            disp1 = None
        cood1.x += self._dx
        cood1.y += self._dy
        cood1, evlp1, disp1 = self.drift_transfer(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True, method: str = 'symplectic4') \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}'). (not used in the dipole class).

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        cood, evlp, disp = self.drift_transfer(self._ds, cood0, evlp0, disp0)
        cood.x -= self._dx
        cood.y -= self._dy
        tmat, s = self.transfer_matrix_array(cood0, ds, endpoint)
        dispvec = self._inhomogeneous_dispersion_array(cood0.delta, s)
        coodvec = np.matmul(tmat, cood.vector).T + dispvec * cood0.delta
        cood1 = CoordinateArray(coodvec, s + cood0.s,
                                np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
        if evlp is not None:
            evlp1 = EnvelopeArray.transport(evlp, tmat, s)
        else:
            evlp1 = None
        if disp is not None:
            disp_add, _ = self.dispersion_array(cood0, ds, endpoint)
            disp1vec = np.matmul(tmat, disp.vector).T + disp_add
            disp1 = DispersionArray(disp1vec, s + disp.s)
        else:
            disp1 = None
        cood1.x += self._dx
        cood1.y += self._dy
        cood1, evlp1, disp1 = self.drift_transfer_array(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    def radiation_integrals(self, cood0: Coordinate, evlp0: Envelope, disp0: Dispersion, ds: float = 0.1, method: str = 'symplectic4') \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            beta0 BetaFunc: Initial Twiss parameters.
            eta0 npt.NDArray[np.floating]: Initial dispersion [eta_x, eta_x', eta_y, eta_y'].
            ds float: Step size for numerical integration.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            Tuple[float, float, float, float, float, float]: Radiation integrals I2, I4, I5u, I5v, I4u, and I4v.
        '''
        kappa = 1./self.rho
        k = self._k1
        _, evlp, disp = self.transfer_array(cood0, evlp0, disp0, ds, endpoint=True, method=method)
        dispuv = np.matvec(evlp.T_matrix(), disp.vector.T).T
        I2 = self._length * kappa**2
        I4 = scipy.integrate.simpson(disp.x * kappa * (kappa**2 + 2. * k), x=disp.s)
        I4u = scipy.integrate.simpson(evlp.tau * dispuv[0] * kappa * (kappa**2 + 2. * k), x=disp.s)
        I4v = I4 - I4u
        I5u = scipy.integrate.simpson(kappa**3 * (evlp.bu * dispuv[1]**2 + 2. * evlp.au * dispuv[0] * dispuv[1] + evlp.gu * dispuv[0]**2), x=disp.s)
        I5v = scipy.integrate.simpson(kappa**3 * (evlp.bv * dispuv[3]**2 + 2. * evlp.av * dispuv[2] * dispuv[3] + evlp.gv * dispuv[2]**2), x=disp.s)
        return I2, I4, I5u, I5v, I4u, I4v

    def partial_element_from_s(self, s: float) -> Dipole:
        '''
        Return a partial element from the given longitudinal position.

        Args:
            s float: Longitudinal position [m] from the entrance of the element.

        Returns:
            Dipole: Partial dipole element starting from the specified longitudinal position.
        '''
        if s < 0. or s > self._length:
            raise ValueError(f'Longitudinal position out of range. s={s}, length={self._length}')
        length = self._length - s
        angle = self._angle * length / self._length
        return Dipole(self._name + f'_part_from_{s:.3f}', length, angle, self._k1, self._e1, self._e2, self._h1, self._h2,
                      self._dx, self._dy, self._ds + s, self._tilt, self._info)
