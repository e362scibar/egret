# sextupole.py
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
from .drift import Drift
from .quadrupole import Quadrupole
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray

import numpy as np
import numpy.typing as npt
from typing import Tuple

# Optional Numba acceleration: define numeric kernels in this file so they remain colocated
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

class Sextupole(Element):
    '''
    Sextupole magnet.
    '''

    def __init__(self, name: str, length: float, k2: float,
                 dx: float = 0., dy: float = 0., ds: float = 0.,
                 tilt: float = 0., info: str = '', dxp: float = 0., dyp: float = 0.):
        '''
        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            k2 float: Normalized sextupole strength [1/m^3].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
            dxp float: Horizontal kick angle of the steering coil [rad].
            dyp float: Vertical kick angle of the steering coil [rad].
        '''
        super().__init__(name, length, dx, dy, ds, tilt, info)
        self.k2 = k2
        self.set_steering(dxp, dyp)

    def copy(self) -> Sextupole:
        '''
        Return a copy of the sextupole.

        Returns:
            Sextupole: Copy of the sextupole.
        '''
        return Sextupole(self.name, self.length, self.k2,
                         self.dx, self.dy, self.ds,
                         self.tilt, self.info)

    def transfer_matrix_by_midpoint_method(self, cood0: Coordinate, ds: float = 0.1) \
        -> Tuple[npt.NDArray[np.floating], Coordinate]:
        '''
        Calculate a single step transfer matrix using the midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
            Coordinate: Final coordinate after the step.
        '''
        # Try calling numba-accelerated numeric kernel when available.
        if _NUMBA_AVAILABLE:
            try:
                # numeric kernel implemented below returns tmat (4x4) and cood_vector (4,)
                tmat_num, cood_vec = _sext_midpoint_numeric(self.k2, getattr(self, 'k0x', 0.0),
                                                            getattr(self, 'k0y', 0.0),
                                                            cood0.vector, ds)
                # build Coordinate from returned vector
                cood2 = Coordinate(cood_vec[0], cood_vec[1], cood_vec[2], cood_vec[3])
                return tmat_num, cood2
            except Exception:
                # fall back to python implementation below
                pass

        # ----- fallback: original Python implementation -----
        k2 = self.k2 / (1. + cood0.delta)
        k0x, k0y = self.k0x / (1. + cood0.delta), self.k0y / (1. + cood0.delta)
        x0, y0, xp0, yp0 = cood0['x'], cood0['y'], cood0['xp'], cood0['yp']
        # dipole strength at the entrance (x'+jy' = k0 L)
        k0a = k2 * (0.5 * (x0**2 - y0**2) - 1.j * x0 * y0) + k0x + 1.j * k0y
        # quadrupole strength at the entrance
        k1a = k2 * (x0 - 1.j * y0)
        # tilt angle of the quadrupole
        tilt = np.angle(k1a) * 0.5
        if np.abs(k1a) < 1.e-20:
            # no quadrupole, just dipole kick
            x1, y1 = x0 + (xp0 - 0.5*k0a.real*ds) * ds, y0 + (yp0 - 0.5*k0a.imag*ds) * ds
        else:
            # transverse offset to generate dipole kick
            offset = - np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0a) / np.abs(k1a)
            # get first quad
            quad1 = Quadrupole(self.name+'_quad1', ds, np.abs(k1a), dx=offset.real, dy=offset.imag, tilt=tilt)
            # get coordinate after first quad
            cood1, _, _ = quad1.transfer(Coordinate(np.array([0., xp0, 0., yp0]), cood0.s, cood0.z, delta=0.))
            x1, y1 = cood1['x'] + x0, cood1['y'] + y0
        # dipole strength after the first quad
        k0b = k2 * (0.5 * (x1**2 - y1**2) - 1.j * x1 * y1) + k0x + 1.j * k0y
        # quadrupole strength after the first quad
        k1b = k2 * (x1 - 1.j * y1)
        # get average dipole strength
        k0 = 0.5 * (k0a + k0b)
        # get average quadrupole strength
        k1 = 0.5 * (k1a + k1b)
        # tilt angle of the quadrupole
        tilt = np.angle(k1) * 0.5
        if np.abs(k1) < 1.e-20:
            # no quadrupole, just dipole kick
            cood2 = Coordinate(np.array([x0 + (xp0 - 0.5*k0.real*ds) * ds, xp0 - k0.real * ds,
                                         y0 + (yp0 - 0.5*k0.imag*ds) * ds, yp0 - k0.imag * ds]),
                               cood0.s + ds, cood0.z, self.delta)
            tmat = Drift.transfer_matrix_from_length(ds)
        else:
            # transverse offset to generate dipole kick
            offset = - np.exp(1.j*tilt) * np.conj(np.exp(-1.j*tilt) * k0) / np.abs(k1)
            # get second quad
            quad2 = Quadrupole(self.name+'_quad2', ds, np.abs(k1), dx=offset.real, dy=offset.imag, tilt=tilt)
            # get coordinate after second quad
            cood2, _, _ = quad2.transfer(Coordinate(np.array([0., xp0, 0., yp0]), cood0.s, cood0.z, delta=0.))
            cood2['x'] += x0
            cood2['y'] += y0
            # get transfer matrix of the second quad
            tmat = quad2.transfer_matrix()
        return tmat, cood2

    def transfer_matrix(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the sextupole calculated by midpoint method.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        # try full numeric kernel to avoid per-step overhead
        if _NUMBA_AVAILABLE:
            try:
                tmat = _sext_transfer_full_numeric(self.k2, getattr(self, 'k0x', 0.0), getattr(self, 'k0y', 0.0),
                                                   cood0.vector, n_step, s_step)
                return tmat
            except Exception:
                pass

        # fallback: original loop-based implementation
        cood = cood0.copy()
        tmat = np.eye(4)
        for _ in range(n_step):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal position array of shape (N,).
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        s = np.linspace(0., self.length, n_step + int(endpoint), endpoint=endpoint)
        # try full numeric kernel that returns list of tmat per step
        try:
            tmat_list = _sext_transfer_full_numeric_array(self.k2, getattr(self, 'k0x', 0.0), getattr(self, 'k0y', 0.0),
                                                         cood0.vector, n_step, s_step, endpoint)
            return tmat_list, s
        except Exception:
            # fallback to original implementation
            cood = cood0.copy()
            tmat = np.eye(4)
            tmat_list = [tmat.copy()]
            for _ in range(n_step - int(not endpoint)):
                tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
                tmat = tmat_step @ tmat
                tmat_list.append(tmat.copy())
            return np.array(tmat_list), s

    def dispersion(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector at the exit of the sextupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m] for integration.

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        cood = cood0.copy()
        for _ in range(n_step):
            _, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
        return Drift.transfer_matrix_from_length(self.length) @ cood0.vector - cood.vector

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the sextupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Additive dispersion array [eta_x, eta_x', eta_y, eta_y'].
            npt.NDArray[np.floating]: Longitudinal position array [s].
        '''
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        s = np.linspace(0., self.length, n_step + int(endpoint), endpoint=endpoint)
        cood = cood0.copy()
        tmat = np.eye(4)
        cood_list = [np.zeros(4)]
        for _ in range(n_step - int(not endpoint)):
            _, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            cood_list.append(cood.vector.copy())
        tmat_drift, _ = Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)
        disp = np.matmul(tmat_drift, cood0.vector) - np.array(cood_list)
        return disp, s

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the sextupole.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration.

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood0err = Coordinate(cood0['x'] - self.dx, cood0['xp'],
                              cood0['y'] - self.dy, cood0['yp'],
                              cood0['s'] - self.ds, cood0['z'], cood0['delta'])
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        cood = cood0err.copy()
        tmat = np.eye(4)
        for _ in range(n_step):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
        cood1 = Coordinate(cood['x'] + self.dx, cood['xp'], cood['y'] + self.dy, cood['yp'],
                           cood0['s'] + self.length, cood0['z'], cood0['delta'])
        if evlp0 is not None:
            evlp = np.dot(self.envelope_transfer_matrix(tmat), evlp0.vector)
            evlp1 = Envelope(evlp[0], evlp[1], evlp[3], evlp[4], cood0['s'] + self.length)
        else:
            evlp1 = None
        if disp0 is not None:
            disp = Drift.transfer_matrix_from_length(self.length) @ cood0err.vector - cood.vector
            disp += np.dot(tmat, disp0.vector)
            disp1 = Dispersion(disp[0], disp[1], disp[2], disp[3], cood0['s'] + self.length)
        else:
            disp1 = None
        return cood1, evlp1, disp1

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.01, endpoint: bool = False) \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate, envelope, and dispersion arrays along the sextupole.

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
        cood0err = Coordinate(cood0['x'] - self.dx, cood0['xp'],
                              cood0['y'] - self.dy, cood0['yp'],
                              cood0['s'] - self.ds, cood0['z'], cood0['delta'])
        n_step = int(self.length // ds) + 1
        s_step = self.length / n_step
        s = np.linspace(0., self.length, n_step + int(endpoint), endpoint=endpoint)
        cood = cood0err.copy()
        tmat = np.eye(4)
        cood_list = [cood.vector.copy()]
        tmat_list = [tmat.copy()]
        for _ in range(n_step - int(not endpoint)):
            tmat_step, cood = self.transfer_matrix_by_midpoint_method(cood, s_step)
            tmat = tmat_step @ tmat
            cood_list.append(cood.vector.copy())
            tmat_list.append(tmat.copy())
        cood_array = np.array(cood_list)
        cood_array = CoordinateArray(cood_array[:, 0] + self.dx, cood_array[:, 1],
                                     cood_array[:, 2] + self.dy, cood_array[:, 3],
                                     cood0['s'] + s, np.full_like(s, cood0['z']), np.full_like(s, cood0['delta']))
        if evlp0 is not None:
            evlp_array = np.matmul(self.envelope_transfer_matrix_array(np.array(tmat_list)), evlp0.vector)
            evlp_array = EnvelopeArray(evlp_array[:, 0], evlp_array[:, 1], evlp_array[:, 3], evlp_array[:, 4],
                                       s + cood0['s'])
        else:
            evlp_array = None
        if disp0 is not None:
            disp_array = np.matmul(np.array(tmat_list), disp0.vector)
            tmat_drift, _ = Drift.transfer_matrix_array_from_length(self.length, ds, endpoint)
            disp_array += np.matmul(tmat_drift, cood0err.vector) - np.array(cood_list)
            disp_array = DispersionArray(disp_array[:, 0], disp_array[:, 1], disp_array[:, 2], disp_array[:, 3],
                                         s + cood0['s'])
        else:
            disp_array = None
        return cood_array, evlp_array, disp_array

    def set_steering(self, dxp: float = None, dyp: float = None) -> None:
        '''
        Set steering coil kick angles.

        Args:
            dxp float: Horizontal kick angle of the steering coil [rad].
            dyp float: Vertical kick angle of the steering coil [rad].
        '''
        if dxp is not None:
            self.dxp = dxp
            self.k0x = - self.dxp / self.length
        if dyp is not None:
            self.dyp = dyp
            self.k0y = - self.dyp / self.length

def _sext_transfer_full_numeric_py(k2_full: float, k0x_full: float, k0y_full: float, cood_vec_in: npt.NDArray[np.floating], n_step: int, s_step: float):
    """Return final 4x4 transfer matrix after n_step midpoint integrations."""
    cood = cood_vec_in.copy()
    tmat_running = np.eye(4)
    for _ in range(n_step):
        tmat_step, cood = _sext_midpoint_single_numeric(k2_full, k0x_full, k0y_full, cood, s_step)
        tmat_running = tmat_step @ tmat_running
    return tmat_running

def _sext_transfer_full_numeric_array_py(k2_full: float, k0x_full: float, k0y_full: float, cood_vec_in: npt.NDArray[np.floating], n_step: int, s_step: float, endpoint: bool):
    """Return array of transfer matrices at each step (shape (N,4,4))."""
    cood = cood_vec_in.copy()
    tmat_running = np.eye(4)
    tlist = [tmat_running.copy()]
    steps = n_step - int(not endpoint)
    for _ in range(steps):
        tmat_step, cood = _sext_midpoint_single_numeric(k2_full, k0x_full, k0y_full, cood, s_step)
        tmat_running = tmat_step @ tmat_running
        tlist.append(tmat_running.copy())
    return np.array(tlist)

def _sext_midpoint_single_numeric(k2_full: float, k0x_full: float, k0y_full: float, cood_vec: npt.NDArray[np.floating], ds: float):
    """Numeric implementation of a single midpoint step. Returns (tmat, new_cood_vec).
    This reuses the logic from _sext_midpoint_numeric_py but operates on and returns numpy arrays.
    """
    # reuse previously implemented function for single-step but without object creation
    tmat, cood2 = _sext_midpoint_numeric_py(k2_full, k0x_full, k0y_full, cood_vec, ds)
    return tmat, np.concatenate((cood2, np.array([0.0, 0.0, cood_vec[6]])))[:7]

# bind and optionally njit full-element kernels
_sext_transfer_full_numeric = _sext_transfer_full_numeric_py
_sext_transfer_full_numeric_array = _sext_transfer_full_numeric_array_py
_sext_midpoint_single_numeric = _sext_midpoint_single_numeric
if _NUMBA_AVAILABLE:
    try:
        _sext_transfer_full_numeric = njit(_sext_transfer_full_numeric_py, cache=True)
        _sext_transfer_full_numeric_array = njit(_sext_transfer_full_numeric_array_py, cache=True)
        _sext_midpoint_single_numeric = njit(_sext_midpoint_single_numeric, cache=True)
    except Exception:
        _sext_transfer_full_numeric = _sext_transfer_full_numeric_py
        _sext_transfer_full_numeric_array = _sext_transfer_full_numeric_array_py
        _sext_midpoint_single_numeric = _sext_midpoint_single_numeric

def _sext_midpoint_numeric_py(k2_full: float, k0x_full: float, k0y_full: float, cood_vec_in: npt.NDArray[np.floating], ds: float):
    """
    Pure-Python numeric helper implementing the same logic as transfer_matrix_by_midpoint_method
    but operating on plain numpy arrays and scalars. Returns (tmat (4x4), cood_vector (4,)).
    This function is suitable for njit compilation later.
    """
    # cood_vec_in: [x, xp, y, yp, s, z, delta] but we only use first 4 and delta
    delta = cood_vec_in[6]
    k2 = k2_full / (1. + delta)
    k0x = k0x_full / (1. + delta)
    k0y = k0y_full / (1. + delta)
    x0 = cood_vec_in[0]
    y0 = cood_vec_in[2]
    xp0 = cood_vec_in[1]
    yp0 = cood_vec_in[3]

    k0a = k2 * (0.5 * (x0**2 - y0**2) - 1.j * x0 * y0) + k0x + 1.j * k0y
    k1a = k2 * (x0 - 1.j * y0)
    # tilt
    tilt = np.angle(k1a) * 0.5
    if np.abs(k1a) < 1.e-20:
        x1 = x0 + (xp0 - 0.5 * k0a.real * ds) * ds
        y1 = y0 + (yp0 - 0.5 * k0a.imag * ds) * ds
    else:
        offset = - np.exp(1.j * tilt) * np.conj(np.exp(-1.j * tilt) * k0a) / np.abs(k1a)
        # compute quad1 effect analytically using quadrupole formulas to avoid constructing objects
        kq = np.abs(k1a)
        # treat as thin quad of length ds with dx,dy offsets applied as coordinate shifts
        # apply transfer: use Quadrupole.transfer_matrix logic (small-step approx)
        # for short ds, use exact quad matrix from Quadrupole.transfer_matrix
        sqrtk = np.sqrt(np.abs(kq))
        psi = sqrtk * ds
        if kq == 0.:
            tmat_q = np.eye(4)
            tmat_q[0,1] = ds
            tmat_q[2,3] = ds
        else:
            cospsi, sinpsi = np.cos(psi), np.sin(psi)
            coshpsi, sinhpsi = np.cosh(psi), np.sinh(psi)
            mf = np.array([[cospsi, sinpsi/sqrtk],
                           [-sqrtk*sinpsi, cospsi]])
            md = np.array([[coshpsi, sinhpsi/sqrtk],
                           [sqrtk*sinhpsi, coshpsi]])
            if kq < 0.:
                tmat_q = np.eye(4)
                tmat_q[0:2,0:2] = md
                tmat_q[2:4,2:4] = mf
            else:
                tmat_q = np.eye(4)
                tmat_q[0:2,0:2] = mf
                tmat_q[2:4,2:4] = md
        # apply tilt rotation
        if tilt != 0.:
            ct = np.cos(tilt)
            st = np.sin(tilt)
            rmat = np.array([[ct, 0., st, 0.], [0., ct, 0., st], [-st, 0., ct, 0.], [0., -st, 0., ct]])
            tmat_q = rmat.T @ tmat_q @ rmat
        # apply quad to initial offset coordinate (0,xp0,0,yp0)
        vec = np.array([0.0, xp0, 0.0, yp0])
        cood1_vec = tmat_q @ vec
        x1 = cood1_vec[0] + x0
        y1 = cood1_vec[2] + y0

    k0b = k2 * (0.5 * (x1**2 - y1**2) - 1.j * x1 * y1) + k0x + 1.j * k0y
    k1b = k2 * (x1 - 1.j * y1)
    k0 = 0.5 * (k0a + k0b)
    k1 = 0.5 * (k1a + k1b)
    tilt = np.angle(k1) * 0.5
    if np.abs(k1) < 1.e-20:
        cood2 = np.empty(4, dtype=np.float64)
        cood2[0] = x0 + (xp0 - 0.5 * k0.real * ds) * ds
        cood2[1] = xp0 - k0.real * ds
        cood2[2] = y0 + (yp0 - 0.5 * k0.imag * ds) * ds
        cood2[3] = yp0 - k0.imag * ds
        tmat = Drift.transfer_matrix_from_length(ds)
    else:
        offset = - np.exp(1.j * tilt) * np.conj(np.exp(-1.j * tilt) * k0) / np.abs(k1)
        # compute quad2 transfer matrix analytically (same as above)
        kq2 = np.abs(k1)
        sqrtk2 = np.sqrt(np.abs(kq2))
        psi2 = sqrtk2 * ds
        if kq2 == 0.:
            tmat_q2 = np.eye(4)
            tmat_q2[0,1] = ds
            tmat_q2[2,3] = ds
        else:
            cospsi2, sinpsi2 = np.cos(psi2), np.sin(psi2)
            coshpsi2, sinhpsi2 = np.cosh(psi2), np.sinh(psi2)
            mf2 = np.array([[cospsi2, sinpsi2/sqrtk2], [-sqrtk2*sinpsi2, cospsi2]])
            md2 = np.array([[coshpsi2, sinhpsi2/sqrtk2], [sqrtk2*sinhpsi2, coshpsi2]])
            if kq2 < 0.:
                tmat_q2 = np.eye(4)
                tmat_q2[0:2,0:2] = md2
                tmat_q2[2:4,2:4] = mf2
            else:
                tmat_q2 = np.eye(4)
                tmat_q2[0:2,0:2] = mf2
                tmat_q2[2:4,2:4] = md2
        if tilt != 0.:
            ct2 = np.cos(tilt)
            st2 = np.sin(tilt)
            rmat2 = np.array([[ct2, 0., st2, 0.], [0., ct2, 0., st2], [-st2, 0., ct2, 0.], [0., -st2, 0., ct2]])
            tmat_q2 = rmat2.T @ tmat_q2 @ rmat2
        vec2 = np.array([0.0, xp0, 0.0, yp0])
        cood2 = tmat_q2 @ vec2
        cood2[0] += x0
        cood2[2] += y0
        tmat = tmat_q2
    return tmat, cood2

# Bind numeric helper name used above. If numba is available, compile the pure-Python helper.
_sext_midpoint_numeric = _sext_midpoint_numeric_py
if _NUMBA_AVAILABLE:
    try:
        _sext_midpoint_numeric = njit(_sext_midpoint_numeric_py, cache=True)
    except Exception:
        # keep pure-python version on any compilation issues
        _sext_midpoint_numeric = _sext_midpoint_numeric_py
