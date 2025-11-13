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

from __future__ import annotations

from modules_for_beta_func import chi, tau

from .element import Element
from .coordinate import Coordinate
from .envelope import Envelope
from .dispersion import Dispersion

import copy
import numpy as np
import numpy.typing as npt
import scipy
from typing import Tuple, List

class Ring(Element):
    '''
    Ring accelerator.
    '''
    C_q = 3.83193864e-13  # Factor for equilibrium emittance
    m_e_eV = 510998.95  # Electron rest mass in eV

    def __init__(self, name: str, elements: List[Element], energy: float, info: str = ''):
        '''
        Args:
            name str: Name of the lattice.
            elements list of Element: List of elements in the lattice.
            energy float: Beam energy [eV].
            info str: Additional information.
        '''
        length = 0.
        for elem in elements:
            length += elem.length
        super().__init__(name, length, 0., 0., 0., 0., info)
        self.angle = 0.
        self.tune = np.zeros(2)
        self.elements = copy.deepcopy(elements)
        self.energy = energy
        self.set_index()
        self.update()

    def copy(self) -> Ring:
        '''
        Return a copy of the ring.

        Returns:
            Ring: Copy of the ring.
        '''
        return Ring(self.name, self.elements, self.energy, self.info)

    def update(self, delta: float = 0.):
        '''
        Update transfer matrix, dispersion, and emittance.

        Args:
            delta float: Relative momentum deviation (default: 0.).
        '''
        for elem in self.elements:
            try:
                self.angle += elem.angle
            except AttributeError:
                pass
        # initial coordinate of closed orbit
        try:
            cood_guess = Coordinate(delta=delta)
            self.cood0 = self.find_initial_coordinate_of_closed_orbit(guess=cood_guess, tol=1.e-7)
        except RuntimeError as e:
            print(f'Warning: Failed to find closed orbit. Using zero coordinate. {e}')
            self.cood0 = Coordinate(delta=delta)
        M = self.transfer_matrix(self.cood0)
        # initial dispersion
        disp = self.dispersion(self.cood0)
        disp0 = np.dot(np.linalg.inv(np.eye(4) - M), disp)
        self.disp0 = Dispersion(disp0, 0.)
        # initial beta function and tune
        Mxx, Mxy, Myx, Myy = M[0:2,0:2], M[0:2,2:4], M[2:4,0:2], M[2:4,2:4]
        Mxy_ = np.array([[Mxy[1,1], -Mxy[0,1]], [-Mxy[1,0], Mxy[0,0]]])
        chi = 1. + 4. * np.linalg.det(Myx + Mxy_) / np.linalg.trace(Mxx - Myy)**2
        sqrtchi = np.sqrt(chi)
        tau = np.sqrt(0.5 * (1. + 1./sqrtchi))
        T = -(Myx + Mxy_) / (sqrtchi * tau * np.trace(Mxx - Myy))
        T_ = np.array([[T[1,1], -T[0,1]], [-T[1,0], T[0,0]]])
        U = np.sqrt(chi) * (tau**2 * Mxx - T_ @ Myy @ T)
        V = np.sqrt(chi) * (tau**2 * Myy - T @ Mxx @ T_)
        cos_u = 0.5 * np.trace(U)
        sin_u = np.sign(U[0,1]-U[1,0]) * np.sqrt(np.linalg.det(U - cos_u * np.eye(2)))
        cos_v = 0.5 * np.trace(V)
        sin_v = np.sign(V[0,1]-V[1,0]) * np.sqrt(np.linalg.det(V - cos_v * np.eye(2)))
        mu_u = np.arctan2(sin_u, cos_u)
        mu_v = np.arctan2(sin_v, cos_v)
        bu = U[0,1] / sin_u
        au = (U[0,0] - U[1,1]) * 0.5 / sin_u
        gu = -U[1,0] / sin_u
        bv = V[0,1] / sin_v
        av = (V[0,0] - V[1,1]) * 0.5 / sin_v
        gv = -V[1,0] / sin_v
        # TT = np.block([[tau * np.eye(2), -T_], [T, tau * np.eye(2)]])
        TT_inv = np.block([[tau * np.eye(2), T_], [-T, tau * np.eye(2)]])
        Su = np.array([[bu, -au], [-au, gu]])
        Sv = np.array([[bv, -av], [-av, gv]])
        SSuv = np.block([[Su, np.zeros((2,2))], [np.zeros((2,2)), Sv]])
        SSxy = TT_inv @ SSuv @ TT_inv.T
        self.evlp0 = Envelope(SSxy, 0., T)
        self.tune[0] = mu_u / (2.*np.pi)
        self.tune[1] = mu_v / (2.*np.pi)
        for i in range(2):
            if self.tune[i] < 0.:
                self.tune[i] += 1.
        self.I2, self.I4, self.I5u, self.I5v, self.I4u, self.I4v = self.radiation_integrals(self.cood0, self.evlp0, self.disp0)
        self.emittance = self.C_q * (self.energy / self.m_e_eV)**2 * np.array([self.I5u / (self.I2 - self.I4u), self.I5v / (self.I2 - self.I4v)])
        self.Jx = 1. - self.I4u / self.I2
        self.Jy = 1. - self.I4v / self.I2
        self.Jz = 2. + self.I4 / self.I2

    def get_element(self, key: int | Tuple[int, ...]) -> Element:
        '''
        Get element by index or tuple of indices.

        Args:
            key int or tuple of int: Index or tuple of indices.

        Returns:
            Element: Element at the specified index.
        '''
        if isinstance(key, int):
            return self.elements[key]
        elif isinstance(key, tuple):
            if not isinstance(key[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if len(key) > 1 and hasattr(self.elements[key[0]], 'elements'):
                return self.elements[key[0]].get_element(key[1:])
            else:
                return self.elements[key[0]]
        else:
            raise TypeError('Index must be int or tuple of int.')

    def get_s(self, key: int | Tuple[int, ...] | Element) -> float:
        '''
        Get longitudinal position by index or tuple of indices.

        Args:
            key int or tuple of int or Element: Index, tuple of indices, or Element.

        Returns:
            float: Longitudinal position [m].
        '''
        if isinstance(key, int):
            if key < 0 or key >= len(self.elements):
                raise IndexError('Index out of range.')
            s = 0.
            for i in range(key):
                s += self.elements[i].length
            return s
        elif isinstance(key, tuple):
            if not isinstance(key[0], int):
                raise TypeError('Index must be int or tuple of int.')
            if key[0] < 0 or key[0] >= len(self.elements):
                raise IndexError('Index out of range.')
            s = 0.
            for i in range(key[0]):
                s += self.elements[i].length
            if len(key) > 1 and hasattr(self.elements[key[0]], 'elements'):
                s += self.elements[key[0]].get_s(key[1:])
            return s
        elif isinstance(key, Element):
            return self.get_s(key.index)
        else:
            raise TypeError('Index must be int, tuple of int, or Element.')

    def find_index(self, name: str | Tuple[str, ...]) -> Tuple[int, ...]:
        '''
        Find indices of elements starting with a given name.

        Args:
            name str | tuple of str: Name of the element.

        Returns:
            tuple of int: Tuple of indices of the element.
        '''
        index_list = []
        for i,elem in enumerate(self.elements):
            if hasattr(elem, 'elements'):
                try:
                    sub_index_list = elem.find_index(name)
                    index_list += [((i,) + idx) for idx in sub_index_list]
                except KeyError:
                    continue
            elif isinstance(name, str) and elem.name.startswith(name):
                index_list.append((i,))
            elif isinstance(name, tuple):
                for n in name:
                    if elem.name.startswith(n):
                        index_list.append((i,))
                        break
        if len(index_list) == 0:
            raise KeyError(f'Element starting with name {name} not found.')
        return index_list

    def transfer_matrix(self, cood0: Coordinate, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the ring.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        cood = cood0.copy()
        tmat = np.eye(4)
        for elem in self.elements:
            tmat = np.dot(elem.transfer_matrix(cood, ds), tmat)
            cood = elem.transfer(cood, ds=ds)[0]
        return tmat

    def transfer_matrix_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix along the ring.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        cood = cood0.copy()
        s0 = 0.
        tmat = np.eye(4)
        sarray = []
        tmatarray = []
        for elem in self.elements:
            tmat_elem, s_elem = elem.transfer_matrix_array(cood, ds, False)
            tmatarray.append(np.matmul(tmat_elem.transpose(2,0,1), tmat).transpose(1,2,0))
            sarray.append(s_elem + s0)
            s0 += elem.length
            tmat = np.dot(elem.transfer_matrix(cood), tmat)
            cood, _, _ = elem.transfer(cood)
        if endpoint:
            tmatarray.append(tmat[:,:,np.newaxis])
            sarray.append(np.array([s0]))
        return np.dstack(tmatarray), np.hstack(sarray)

    def dispersion(self, cood0: Coordinate) -> Dispersion:
        '''
        Additive dispersion of the ring.

        Args:
            cood0 Coordinate: Initial coordinate.

        Returns:
            Dispersion: Additive dispersion of the lattice.
        '''
        cood = cood0.copy()
        disp = Dispersion()
        for elem in self.elements:
            cood, _, disp = elem.transfer(cood, None, disp)
        return disp.vector

    def dispersion_array(self, cood0: Coordinate, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the ring.

        Args:
            cood0 Coordinate: Initial coordinate.
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        cood = cood0.copy()
        disparray = []
        sarray = []
        s0 = 0.
        disp = Dispersion()
        for elem in self.elements:
            disp_elem, s_elem = elem.dispersion_array(cood, ds, False)
            tmat, _ = elem.transfer_matrix_array(cood, ds, False)
            disparray.append(disp_elem + np.dot(tmat, disp.vector).T)
            sarray.append(s_elem + s0)
            s0 += elem.length
            cood, _, disp = elem.transfer(cood, None, disp)
        if endpoint:
            disparray.append(disp.vector[:, np.newaxis])
            sarray.append(np.array([s0]))
        return np.hstack(disparray), np.hstack(sarray)

    def radiation_integrals(self, cood0: Coordinate, evlp0: Envelope, disp0: Dispersion, ds: float = 0.1) \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial envelope.
            disp0 Dispersion: Initial dispersion.
            ds float: Maximum step size [m].

        Returns:
            I2 float: Second radiation integral.
            I4 float: Fourth radiation integral.
            I5 float: Fifth radiation integral.
        '''
        I2, I4, I5u, I5v, I4u, I4v = 0., 0., 0., 0., 0., 0.
        cood = cood0.copy()
        evlp = evlp0.copy()
        disp = disp0.copy()
        for elem in self.elements:
            if elem.length == 0.:
                continue
            i2, i4, i5u, i5v, i4u, i4v = elem.radiation_integrals(cood, evlp, disp, ds)
            I2 += i2
            I4 += i4
            I5u += i5u
            I5v += i5v
            I4u += i4u
            I4v += i4v
            cood, evlp, disp = elem.transfer(cood, evlp, disp)
        return I2, I4, I5u, I5v, I4u, I4v

    def find_initial_coordinate_of_closed_orbit(self, guess: Coordinate = Coordinate(),
        tol: float = None, maxiter: int = 500) -> Coordinate:
        '''
        Find initial coordinate of the closed orbit using Newton-Raphson method.

        Args:
            guess Coordinate: Initial guess of the coordinate.
            tol float: Tolerance for convergence.
            maxiter int: Maximum number of iterations.

        Returns:
            Coordinate: Initial coordinate of the closed orbit.
        '''
        cood = guess.copy()
        eval_func = lambda x: np.linalg.norm(self.transfer(Coordinate(x, delta=guess.delta))[0].vector - x)
        result = scipy.optimize.minimize(eval_func, cood.vector, method='Nelder-Mead', tol=tol, options={'maxiter': maxiter})
        if not result.success:
            raise RuntimeError('Failed to find closed orbit: ' + result.message)
        cood = Coordinate(result.x, delta=guess.delta)
        if result.nit == maxiter:
            raise RuntimeError('Failed to find closed orbit: Maximum number of iterations reached.')
        return cood
