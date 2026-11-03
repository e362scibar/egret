# python/quadrupole.py
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
from ..base.quadrupole import Quadrupole as QuadruopoleABC
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

# Quadrupole gradients below this magnitude [1/m^2] are treated as zero (same threshold as the C++ backend)
K1_IGNORE_THRESHOLD: Final = 1.0e-15

class Quadrupole(QuadruopoleABC, Element):
    '''
    Quadrupole magnet class.
    '''

    def __init__(self, name: str, length: float, k1: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: float = '', kick_x: float = 0.0, kick_y: float = 0.0):
        '''
        Args:
            name str: Name of the quadrupole.
            length float: Length of the quadrupole [m].
            k1 float: Normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
            dx float: Horizontal offset of the quadrupole [m].
            dy float: Vertical offset of the quadrupole [m].
            ds float: Longitudinal offset of the quadrupole [m].
            tilt float: Tilt angle of the quadrupole [rad]. (+pi/4: skew quadrupole)
            info str: Additional information.
            kick_x float: Horizontal kick angle of the steering coil [rad].
            kick_y float: Vertical kick angle of the steering coil [rad].
        '''
        super().__init__(name, length, 0.0, dx, dy, ds, tilt, info)
        self._k1 = k1
        self._kick_x = kick_x
        self._kick_y = kick_y

    @property
    def k1(self) -> float:
        '''
        Normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)
        '''
        return self._k1

    @k1.setter
    def k1(self, k1: float) -> None:
        '''
        Set normalized gradient [1/m^2]. (k1 > 0: focusing in horizontal plane)

        Args:
            k1 (float): Normalized gradient [1/m^2].
        '''
        self._k1 = k1

    @property
    def k0x(self) -> float:
        '''
        Horizontal steering strength [1/m].
        '''
        return -self._kick_x / self._length

    @property
    def k0y(self) -> float:
        '''
        Vertical steering strength [1/m].
        '''
        return -self._kick_y / self._length

    @property
    def kick_x(self) -> float:
        '''
        Horizontal kick angle of the steering coil [rad].
        '''
        return self._kick_x

    @property
    def kick_y(self) -> float:
        '''
        Vertical kick angle of the steering coil [rad].
        '''
        return self._kick_y

    @k0x.setter
    def k0x(self, k0x: float) -> None:
        '''
        Set horizontal steering strength.

        Args:
            k0x float: Horizontal steering strength [1/m].
        '''
        self._kick_x = -k0x * self._length

    @k0y.setter
    def k0y(self, k0y: float) -> None:
        '''
        Set vertical steering strength.

        Args:
            k0y float: Vertical steering strength [1/m].
        '''
        self._kick_y = -k0y * self._length

    @kick_x.setter
    def kick_x(self, kick_x: float) -> None:
        '''
        Set horizontal kick angle of the steering coil.

        Args:
            kick_x float: Horizontal kick angle [rad].
        '''
        self._kick_x = kick_x

    @kick_y.setter
    def kick_y(self, kick_y: float) -> None:
        '''
        Set vertical kick angle of the steering coil.

        Args:
            kick_y float: Vertical kick angle [rad].
        '''
        self._kick_y = kick_y

    def copy(self) -> Quadrupole:
        '''
        Return a copy of the quadrupole element.

        Returns:
            Quadrupole: Copied quadrupole element.
        '''
        return Quadrupole(self._name, self._length, self._k1,
                          self._dx, self._dy, self._ds, self._tilt, self._info, self._kick_x, self._kick_y)

    def _is_zero_gradient(self) -> bool:
        '''
        Whether the quadrupole gradient is negligible, so that the steering kick is a distributed dipole kick.

        Returns:
            bool: True if |k1| is below the threshold.
        '''
        return abs(self._k1) < K1_IGNORE_THRESHOLD

    def _kick_at(self, delta: float) -> Tuple[float, float]:
        '''
        Steering kick angles at a relative momentum deviation [rad]. The kick scales as 1 / (1 + delta).

        Args:
            delta float: Relative momentum deviation of the particle.

        Returns:
            Tuple[float, float]: Horizontal and vertical kick angles [rad].
        '''
        scale = 1. / (1. + delta)
        return self._kick_x * scale, self._kick_y * scale

    @staticmethod
    def _block(tmat: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        '''
        The (x', y') x (x, y) block of a 4x4 transfer matrix.

        Args:
            tmat npt.NDArray[np.floating]: 4x4 transfer matrix.

        Returns:
            npt.NDArray[np.floating]: 2x2 block.
        '''
        return np.array([[tmat[1, 0], tmat[1, 2]], [tmat[3, 0], tmat[3, 2]]])

    def _steering_offset(self) -> npt.NDArray[np.floating]:
        '''
        Transverse offset of the magnetic center that realizes the steering kick [m], as a vector (x, y).

        The offset is a mechanical displacement and does not depend on the momentum: it is evaluated once at the
        reference momentum (delta = 0). A particle entering on axis then leaves with an angle change of +kick at
        delta = 0, and with the change of k1 at other momenta. The offset is c = -B^-1 kick, where B is the
        (x', y') x (x, y) block of the transfer matrix. Returns zeros for a negligible gradient or no kick.

        Returns:
            npt.NDArray[np.floating]: Offset (x, y) [m].

        Raises:
            ValueError: If the transfer block is singular and a kick is requested.
        '''
        kick = np.array([self._kick_x, self._kick_y])
        if self._is_zero_gradient() or not np.any(kick) or self._length == 0.:
            return np.zeros(2)
        block = self._block(self._matrix(self._length, 0.))
        if abs(np.linalg.det(block)) < K1_IGNORE_THRESHOLD:
            raise ValueError('Steering kick cannot be realized: the quadrupole transfer block is singular.')
        return -np.linalg.solve(block, kick)

    def _distributed_kick(self, s: npt.NDArray[np.floating], delta: float) -> npt.NDArray[np.floating]:
        '''
        Distributed dipole kick accumulated over the first s meters, for a negligible gradient.

        The angle changes by kick * s / L and the position by kick * s^2 / (2 L), at the given momentum.
        A zero-length quadrupole has no kick.

        Args:
            s npt.NDArray[np.floating]: Longitudinal positions inside the element [m].
            delta float: Relative momentum deviation of the particle.

        Returns:
            npt.NDArray[np.floating]: Array of shape (N, 4) [x, x', y, y'] added to the drifted coordinates.
        '''
        if self._length == 0.:
            return np.zeros((len(s), 4))
        kick_x, kick_y = self._kick_at(delta)
        return np.column_stack([0.5 * kick_x * s**2 / self._length, kick_x * s / self._length,
                                0.5 * kick_y * s**2 / self._length, kick_y * s / self._length])

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True, method='symplectic4') \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate, envelope, and dispersion arrays along the quadrupole magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method. (Not used in the Quadrupole class.)

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        cood, evlp, disp = self.drift_transfer(self._ds, cood0, evlp0, disp0)
        steer_x, steer_y = self._steering_offset()
        cood.x -= self._dx + steer_x
        cood.y -= self._dy + steer_y
        relative = cood.vector.copy()
        tmat_array, s = self.transfer_matrix_array(cood, ds, endpoint, method)
        vector = np.einsum('nij,j->ni', tmat_array, relative)
        if self._is_zero_gradient():
            vector = vector + self._distributed_kick(s, cood.delta)
        vector[:, 0] += self._dx + steer_x
        vector[:, 2] += self._dy + steer_y
        cood1 = CoordinateArray(vector.T, s + cood0.s, np.full_like(s, cood0.z), np.full_like(s, cood0.delta))
        if evlp is not None:
            evlp1 = EnvelopeArray.transport(evlp, tmat_array, s)
        else:
            evlp1 = None
        if disp is not None:
            disp_array = np.zeros((4, len(s)))
            for i, s_i in enumerate(s):
                disp_array[:, i] = tmat_array[i] @ disp.vector \
                    + self._dispersion_at(relative, s_i, cood.delta)
            disp1 = DispersionArray(disp_array, s + disp.s)
        else:
            disp1 = None
        cood1, evlp1, disp1 = self.drift_transfer_array(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1, method='symplectic4') \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the quadrupole magnet.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood, evlp, disp = self.drift_transfer(self._ds, cood0, evlp0, disp0)
        steer_x, steer_y = self._steering_offset()
        cood.x -= self._dx + steer_x
        cood.y -= self._dy + steer_y
        tmat = self._matrix(self._length, cood.delta)
        cood1vec = np.dot(tmat, cood.vector)
        if self._is_zero_gradient():
            cood1vec = cood1vec + self._distributed_kick(np.array([self._length]), cood.delta)[0]
        cood1 = Coordinate(cood1vec, cood.s + self._length, cood.z, cood.delta)
        if evlp is not None:
            evlp1 = evlp.copy()
            evlp1.transfer(tmat, self._length)
        else:
            evlp1 = None
        if disp is not None:
            disp1 = disp.copy()
            disp1vec = np.dot(tmat, disp1.vector) + self._dispersion_at(cood.vector, self._length, cood.delta)
            disp1 = Dispersion(disp1vec, disp1.s + self._length)
        else:
            disp1 = None
        # Restore the element and steering offsets, then apply the drift for the longitudinal offset
        cood1.x += self._dx + steer_x
        cood1.y += self._dy + steer_y
        cood1, evlp1, disp1 = self.drift_transfer(-self._ds, cood1, evlp1, disp1)
        return cood1, evlp1, disp1

    def rotation_matrix(self) -> npt.NDArray[np.floating]:
        '''
        Rotation matrix of the quadrupole.

        Returns:
            npt.NDArray[np.floating]: 4x4 rotation matrix.
        '''
        if self._tilt == 0.:
            return np.eye(4)
        ct = np.cos(self._tilt)
        st = np.sin(self._tilt)
        rmat = np.array([[ct, 0., st, 0.],
                         [0., ct, 0., st],
                         [-st, 0., ct, 0.],
                         [0., -st, 0., ct]])
        return rmat

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m] for integration. (Not used in the Quadrupole class.)
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Quadrupole class.)

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        delta = 0. if cood0 is None else cood0.delta
        return self._matrix(self._length, delta)
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * self._length
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        mf = np.array([[cospsi, sinpsi/sqrtk], [-sqrtk*sinpsi, cospsi]])
        md = np.array([[coshpsi, sinhpsi/sqrtk], [sqrtk*sinhpsi, coshpsi]])
        if k < 0.: # defocusing quadrupole
            tmat[0:2, 0:2] = md
            tmat[2:4, 2:4] = mf
        else: # focusing quadrupole
            tmat[0:2, 0:2] = mf
            tmat[2:4, 2:4] = md
        if self._tilt != 0.:
            rmat = self.rotation_matrix()
            tmat = rmat.T @ tmat @ rmat
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate. (Not used in the Quadrupole class.)
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Quadrupole class.)
        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        delta = 0. if cood0 is None else cood0.delta
        k = self._k1 / (1. + delta)
        s = self.s_array(ds, endpoint)
        tmat = np.repeat(np.eye(4)[np.newaxis,:,:], len(s), axis=0)
        if k == 0.: # drift
            tmat[:,0,1] = s
            tmat[:,2,3] = s
            return tmat, s
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * s
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        mf = np.array([[cospsi, sinpsi/sqrtk], [-sqrtk*sinpsi, cospsi]]).transpose(2,0,1)
        md = np.array([[coshpsi, sinhpsi/sqrtk], [sqrtk*sinhpsi, coshpsi]]).transpose(2,0,1)
        if k < 0.: # defocusing quadrupole
            tmat[:,0:2,0:2] = md
            tmat[:,2:4,2:4] = mf
        else: # focusing quadrupole
            tmat[:,0:2,0:2] = mf
            tmat[:,2:4,2:4] = md
        if self._tilt != 0.:
            rmat = self.rotation_matrix()
            tmat = np.einsum('ji,njk,kl->nil', rmat, tmat, rmat)
        return tmat, s

    def dispersion(self, cood0: Coordinate, ds: float = 0.1, method: str = 'symplectic4') -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size for integration [m]. (Not used in this method.)
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Quadrupole class.)

        Returns:
            npt.NDArray[np.floating]: 4-element dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        cood, _, _ = self.drift_transfer(self._ds, cood0)
        steer = self._steering_offset()
        return self._dispersion_at(self._relative_vector(cood, steer), self._length, cood0.delta)

    def _relative_vector(self, cood: Coordinate, steer: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        '''
        Coordinate vector relative to the magnetic center: the element offset and the steering offset are removed.

        Args:
            cood Coordinate: Coordinate at the entrance of the quadrupole (after the longitudinal offset).
            steer npt.NDArray[np.floating]: Steering offset (x, y) [m].

        Returns:
            npt.NDArray[np.floating]: Vector [x, x', y, y'] relative to the magnetic center.
        '''
        vector = cood.vector.copy()
        vector[0] -= self._dx + steer[0]
        vector[2] -= self._dy + steer[1]
        return vector

    def _matrix(self, length: float, delta: float) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of a quadrupole section of the given length at a relative momentum deviation.

        Args:
            length float: Length of the section [m].
            delta float: Relative momentum deviation.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        k = self._k1 / (1. + delta)
        tmat = np.eye(4)
        if np.abs(k) == 0.: # drift
            tmat[0, 1] = length
            tmat[2, 3] = length
            return tmat
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * length
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        mf = np.array([[cospsi, sinpsi/sqrtk], [-sqrtk*sinpsi, cospsi]])
        md = np.array([[coshpsi, sinhpsi/sqrtk], [sqrtk*sinhpsi, coshpsi]])
        if k < 0.: # defocusing quadrupole
            tmat[0:2, 0:2] = md
            tmat[2:4, 2:4] = mf
        else: # focusing quadrupole
            tmat[0:2, 0:2] = mf
            tmat[2:4, 2:4] = md
        if self._tilt != 0.:
            rmat = self.rotation_matrix()
            tmat = rmat.T @ tmat @ rmat
        return tmat

    def _matrix_derivative(self, length: float, delta: float) -> npt.NDArray[np.floating]:
        '''
        Derivative of the transfer matrix with respect to the relative momentum deviation.

        Args:
            length float: Length of the section [m].
            delta float: Relative momentum deviation.

        Returns:
            npt.NDArray[np.floating]: 4x4 matrix dM / d(delta).
        '''
        return np.column_stack([self._dispersion_vector(unit, length, delta) for unit in np.eye(4)])

    def _dispersion_at(self, vector: npt.NDArray[np.floating], length: float, delta: float) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion of a section of the given length, for a vector relative to the magnetic center.

        The steering offset is fixed at its reference value, so the map is M(delta) (v - c) + c and its derivative
        with respect to delta is M'(delta) (v - c). For a negligible gradient the distributed kick scales as
        1 / (1 + delta), which adds -kick / (1 + delta) to the derivative.

        Args:
            vector npt.NDArray[np.floating]: Vector [x, x', y, y'] relative to the element and steering offsets.
            length float: Length of the section [m].
            delta float: Relative momentum deviation.

        Returns:
            npt.NDArray[np.floating]: 4-element dispersion vector.
        '''
        disp = self._dispersion_vector(vector, length, delta)
        if self._is_zero_gradient():
            disp = disp - self._distributed_kick(np.array([length]), delta)[0] / (1. + delta)
        return disp

    def _dispersion_vector(self, vector: npt.NDArray[np.floating], length: float, delta: float) \
        -> npt.NDArray[np.floating]:
        '''
        Dispersion of a quadrupole section of the given length, for a vector relative to the magnetic center.

        Args:
            vector npt.NDArray[np.floating]: Vector [x, x', y, y'] relative to the magnetic center.
            length float: Length of the quadrupole section [m].
            delta float: Relative momentum deviation of the particle.

        Returns:
            npt.NDArray[np.floating]: 4-element dispersion vector [eta_x, eta'_x, eta_y, eta'_y].
        '''
        k = self._k1 / (1. + delta)
        disp = np.zeros(4)
        if abs(k) < K1_IGNORE_THRESHOLD:
            return disp
        sqrtk = np.sqrt(np.abs(k))
        psi = sqrtk * length
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
        Mf1 = np.array([[sinpsi, -cospsi/sqrtk], [sqrtk*cospsi, sinpsi]]) * 0.5 * length * sqrtk
        Mf2 = np.array([[0., sinpsi/sqrtk], [sqrtk*sinpsi, 0.]]) * 0.5
        Md1 = np.array([[-sinhpsi, -coshpsi/sqrtk], [-sqrtk*coshpsi, -sinhpsi]]) * 0.5 * length * sqrtk
        Md2 = np.array([[0., sinhpsi/sqrtk], [-sqrtk*sinhpsi, 0.]]) * 0.5
        if self._tilt == 0.:
            if k < 0.: # defocusing quadrupole
                disp[0:2] = np.dot(Md1 + Md2, vector[0:2])
                disp[2:4] = np.dot(Mf1 + Mf2, vector[2:4])
            else: # focusing quadrupole
                disp[0:2] = np.dot(Mf1 + Mf2, vector[0:2])
                disp[2:4] = np.dot(Md1 + Md2, vector[2:4])
        else:
            rmat = self.rotation_matrix()
            M = np.zeros((4,4))
            if k < 0.: # defocusing quadrupole
                M[0:2,0:2] = Md1 + Md2
                M[2:4,2:4] = Mf1 + Mf2
            else: # focusing quadrupole
                M[0:2,0:2] = Mf1 + Mf2
                M[2:4,2:4] = Md1 + Md2
            M_rot = rmat.T @ M @ rmat
            disp = M_rot @ vector
        return disp

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False, method: str = 'symplectic4') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the quadrupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method. ('midpoint', 'rk4', 'symplectic{1,2,4}') (Not used in the Quadrupole class.)

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        s = self.s_array(ds, endpoint)
        cood, _, _ = self.drift_transfer(self._ds, cood0)
        steer = self._steering_offset()
        vector = self._relative_vector(cood, steer)
        disp = np.zeros((4, len(s)))
        for i, s_i in enumerate(s):
            disp[:, i] = self._dispersion_at(vector, s_i, cood0.delta)
        return disp, s

    def set_steering(self, kick_x: float = None, kick_y: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            kick_x float: Horizontal kick angle of the steering coil [rad] or None to leave unchanged.
            kick_y float: Vertical kick angle of the steering coil [rad] or None to leave unchanged.
        '''
        if kick_x is not None:
            self._kick_x = kick_x
        if kick_y is not None:
            self._kick_y = kick_y
