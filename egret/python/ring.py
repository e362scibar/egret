# python/ring.py
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
from ..base.ring import Ring as RingABC
from .element import Element
from .lattice import Lattice
from .drift import Drift
from .dipole import Dipole
from .quadrupole import Quadrupole
from .sextupole import Sextupole
from .octupole import Octupole
from .coordinate import Coordinate
from .envelope import Envelope
from .dispersion import Dispersion
import copy
import numpy as np
import numpy.typing as npt
import scipy
from typing import Dict, Tuple, List, Any
import latticejson
from pathlib import Path

class Ring(RingABC, Element):
    '''
    Ring accelerator class.
    '''

    def __init__(self, name: str, elements: List[Element], energy: float, info: str = ''):
        '''
        Args:
            name str: Name of the lattice.
            elements list of Element: List of elements in the lattice.
            energy float: Beam energy [eV].
            info str: Additional information.
        '''
        length = Lattice.length_of(elements)
        angle = Lattice.angle_of(elements)
        super().__init__(name, length, angle, 0., 0., 0., info)
        self._elements = copy.deepcopy(elements)
        self._energy = energy
        self._tune_x = 0.
        self._tune_y = 0.
        self._cood0 = Coordinate()
        self._evlp0 = Envelope()
        self._disp0 = Dispersion()
        self._emittance_x = 0.
        self._emittance_y = 0.
        self._Jx = 0.
        self._Jy = 0.
        self._Jz = 0.
        self.set_indices()

    @property
    def energy(self) -> float:
        '''
        Beam energy [eV].
        '''
        return self._energy

    @property
    def tune_x(self) -> float:
        '''
        Horizontal tune of the ring.
        '''
        return self._tune_x

    @property
    def tune_y(self) -> float:
        '''
        Vertical tune of the ring.
        '''
        return self._tune_y

    @property
    def cood0(self) -> Coordinate:
        '''
        Initial coordinate of the closed orbit.
        '''
        return self._cood0

    @property
    def evlp0(self) -> Envelope:
        '''
        Initial beam envelope of the closed orbit.
        '''
        return self._evlp0

    @property
    def disp0(self) -> Dispersion:
        '''
        Initial dispersion of the closed orbit.
        '''
        return self._disp0

    @property
    def emittance_x(self) -> float:
        '''
        Horizontal equilibrium emittance [m.rad].
        '''
        return self._emittance_x

    @property
    def emittance_y(self) -> float:
        '''
        Vertical equilibrium emittance [m.rad].
        '''
        return self._emittance_y

    @property
    def Jx(self) -> float:
        '''
        Horizontal damping partition number.
        '''
        return self._Jx

    @property
    def Jy(self) -> float:
        '''
        Vertical damping partition number.
        '''
        return self._Jy

    @property
    def Jz(self) -> float:
        '''
        Longitudinal damping partition number.
        '''
        return self._Jz

    def copy(self) -> Ring:
        '''
        Return a copy of the ring.

        Returns:
            Ring: Copy of the ring.
        '''
        return Ring(self._name, self._elements, self._energy, self._info)

    def update(self, delta: float = 0., method: str = 'symplectic4') -> None:
        '''
        Update transfer matrix, dispersion, and emittance.

        Args:
            delta float: Relative momentum deviation (default: 0.).
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').
        '''
        # initial coordinate of closed orbit
        try:
            cood_guess = Coordinate(delta=delta)
            self._cood0 = self.find_initial_coordinate_of_closed_orbit(guess=cood_guess, tol=self.tol_cod)
        except RuntimeError as e:
            print(f'Warning: Failed to find closed orbit. Using zero coordinate. {e}')
            self._cood0 = Coordinate(delta=delta)
        M = self.transfer_matrix(self._cood0)
        # initial dispersion
        disp = self.dispersion(self._cood0)
        disp0 = np.dot(np.linalg.inv(np.eye(4) - M), disp)
        self._disp0 = Dispersion(disp0, 0.)
        # initial beta function and tune
        Mxx, Mxy, Myx, Myy = M[0:2,0:2], M[0:2,2:4], M[2:4,0:2], M[2:4,2:4]
        Mxy_ = np.array([[Mxy[1,1], -Mxy[0,1]], [-Mxy[1,0], Mxy[0,0]]])
        chi = 1. + 4. * np.linalg.det(Myx + Mxy_) / np.linalg.trace(Mxx - Myy)**2
        sqrtchi = np.sqrt(chi)
        tau = np.sqrt(0.5 * (1. + 1./sqrtchi))
        T = -(Myx + Mxy_) / (sqrtchi * tau * np.trace(Mxx - Myy))
        T_ = np.array([[T[1,1], -T[0,1]], [-T[1,0], T[0,0]]])
        U = sqrtchi * (tau**2 * Mxx - T_ @ Myy @ T)
        V = sqrtchi * (tau**2 * Myy - T @ Mxx @ T_)
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
        self._evlp0 = Envelope(SSxy, 0., T)
        self._tune_x = mu_u / (2.*np.pi)
        self._tune_y = mu_v / (2.*np.pi)
        if self._tune_x < 0.:
            self._tune_x += 1.
        if self._tune_y < 0.:
            self._tune_y += 1.
        self.I2, self.I4, self.I5u, self.I5v, self.I4u, self.I4v = self.radiation_integrals(self._cood0, self._evlp0, self._disp0)
        self._emittance_x = self.C_q * (self._energy / self.m_e_eV)**2 * (self.I5u / (self.I2 - self.I4u))
        self._emittance_y = self.C_q * (self._energy / self.m_e_eV)**2 * (self.I5v / (self.I2 - self.I4v))
        self._Jx = 1. - self.I4u / self.I2
        self._Jy = 1. - self.I4v / self.I2
        self._Jz = 2. + self.I4 / self.I2

    def find_initial_coordinate_of_closed_orbit(self, guess: Coordinate = Coordinate(),
        tol: float = None, maxiter: int = 500, method: str = 'symplectic4') -> Coordinate:
        '''
        Find initial coordinate of the closed orbit using Nelder-Mead method.

        Args:
            guess Coordinate: Initial guess of the coordinate.
            tol float: Tolerance for convergence.
            maxiter int: Maximum number of iterations.
            method str: Integration method ('midpoint', 'rk4', 'symplectic{1,2,4}').

        Returns:
            Coordinate: Initial coordinate of the closed orbit.
        '''
        cood = guess.copy()
        eval_func = lambda x: np.linalg.norm(self.transfer(Coordinate(x, delta=guess.delta), method=method)[0].vector - x)
        result = scipy.optimize.minimize(eval_func, cood.vector, method='Nelder-Mead', tol=tol, options={'maxiter': maxiter})
        if not result.success:
            raise RuntimeError('Failed to find closed orbit: ' + result.message)
        cood = Coordinate(result.x, delta=guess.delta)
        if result.nit == maxiter:
            raise RuntimeError('Failed to find closed orbit: Maximum number of iterations reached.')
        return cood

    @classmethod
    def read_json(cls, path: str | Path) -> Ring:
        '''
        Read a ring configuration from a Lattice JSON file and return a Ring object.

        Args:
            path (str or Path): Path to the Lattice JSON file.

        Returns:
            Ring: The constructed Ring object.
        '''
        config = latticejson.load(path)
        rootname = config['root']
        elements = cls._make_elements(config)
        lattices = [cls._make_lattice(config, name, elements) for name in config['lattices'][rootname]]
        return Ring(rootname, lattices, config['energy']*1.e9, config['info'])

    @classmethod
    def _make_lattice(cls, config: Dict[str, Any], name: str, elems: List[Lattice | Element]) \
        -> Lattice | Element:
        '''
        Recursively construct a Lattice or Element based on the lattice configuration.

        Args:
            config (dict): The lattice configuration dictionary.
            name (str): The name of the lattice or element to construct.
            elems (dict): A dictionary of Element objects.

        Returns:
            Lattice or Element: The constructed Lattice if 'name' is in lattices, otherwise the Element.
        '''
        if name in config['lattices']:
            lattices = [cls._make_lattice(config, k, elems) for k in config['lattices'][name]]
            return Lattice(name, lattices)
        return elems[name]

    @classmethod
    def _make_elements(cls, config: Dict[str, Any]) -> Dict[str, Element]:
        '''
        Create a dictionary of Element objects from the lattice configuration.

        Args:
            config (dict): The lattice configuration dictionary.

        Returns:
            dict: A dictionary mapping element names to Element objects.

        Raises:
            KeyError: If an unknown element type is encountered.
        '''
        elements = {}
        for name,val in config['elements'].items():
            key = val[0]
            dat = val[1]
            match key:
                case 'Drift':
                    elements[name] = Drift(name, dat['length'])
                case 'Dipole':
                    elements[name] = Dipole(name, dat['length'], dat['angle'], dat['k1'])
                case 'Quadrupole':
                    elements[name] = Quadrupole(name, dat['length'], dat['k1'])
                case 'Sextupole':
                    elements[name] = Sextupole(name, dat['length'], dat['k2'])
                case 'Octupole':
                    elements[name] = Octupole(name, dat['length'], dat['k3'])
                case _:
                    raise KeyError(key)
        return elements
