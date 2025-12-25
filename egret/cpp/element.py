# cpp/element.py
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
from ..base.element import Element as ElementABC
from egret.cppegret import Element as ElementCPP
from egret.cppegret import IntegrationMethod
from .object import Object
from .coordinate import Coordinate
from .coordinatearray import CoordinateArray
from .envelope import Envelope
from .envelopearray import EnvelopeArray
from .dispersion import Dispersion
from .dispersionarray import DispersionArray
import numpy as np
import numpy.typing as npt
from typing import Tuple, List

class Element(ElementABC, Object):
    '''
    Base class of an accelerator element.
    '''
    # Integration methods
    INTEGRATION_METHODS = {'midpoint': IntegrationMethod.MIDPOINT,
                           'rk4': IntegrationMethod.RK4}

    def __init__(self, name: str, length: float, angle: float,
                 dx: float = 0.0, dy: float = 0.0, ds: float = 0.0,
                 tilt: float = 0.0, info: str = "", **kwargs) -> None:
        '''
        Initialize the Element.

        Args:
            name str: Name of the element.
            length float: Length of the element [m].
            angle float: Bending angle of the element [rad].
            dx float: Horizontal offset of the element [m].
            dy float: Vertical offset of the element [m].
            ds float: Longitudinal offset of the element [m].
            tilt float: Tilt angle of the element [rad].
            info str: Additional information.
        '''
        if 'instance' in kwargs:
            self.instance = kwargs['instance']
        else:
            self.instance = ElementCPP(name, length, angle, dx, dy, ds, tilt, info)
        super().__init__(None, instance=self.instance)

    @classmethod
    def actual_element(cls, instance) -> Element:
        '''
        Create an Element subclass instance from a C++ Element instance.

        Args:
            instance: C++ Element instance.

        Returns:
            Element: Corresponding Element subclass instance.
        '''
        from .drift import Drift
        from .steering import Steering
        from .dipole import Dipole
        from .quadrupole import Quadrupole
        from .sextupole import Sextupole
        from .octupole import Octupole
        from .lattice import Lattice
        from egret.cppegret import Drift as DriftCPP
        from egret.cppegret import Steering as SteeringCPP
        from egret.cppegret import Quadrupole as QuadrupoleCPP
        from egret.cppegret import Dipole as DipoleCPP
        from egret.cppegret import Sextupole as SextupoleCPP
        from egret.cppegret import Octupole as OctupoleCPP
        from egret.cppegret import Lattice as LatticeCPP
        if isinstance(instance, DriftCPP):
            return Drift(None, None, instance=instance)
        elif isinstance(instance, SteeringCPP):
            return Steering(None, None, instance=instance)
        elif isinstance(instance, QuadrupoleCPP):
            return Quadrupole(None, None, None, instance=instance)
        elif isinstance(instance, DipoleCPP):
            return Dipole(None, None, None, instance=instance)
        elif isinstance(instance, SextupoleCPP):
            return Sextupole(None, None, instance=instance)
        elif isinstance(instance, OctupoleCPP):
            return Octupole(None, None, instance=instance)
        elif isinstance(instance, LatticeCPP):
            return Lattice(None, None, instance=instance)
        else:
            return Element(instance=instance)

    @property
    def length(self) -> float:
        '''
        Length of the element [m].
        '''
        return self.instance.length

    @property
    def angle(self) -> float:
        '''
        Bending angle of the element [rad].
        '''
        return self.instance.angle

    @property
    def dx(self) -> float:
        '''
        Horizontal offset of the element [m].
        '''
        return self.instance.dx

    @property
    def dy(self) -> float:
        '''
        Vertical offset of the element [m].
        '''
        return self.instance.dy

    @property
    def ds(self) -> float:
        '''
        Longitudinal offset of the element [m].
        '''
        return self.instance.ds

    @property
    def tilt(self) -> float:
        '''
        Tilt angle of the element [rad].
        '''
        return self.instance.tilt

    @property
    def info(self) -> str:
        '''
        Additional information.
        '''
        return self.instance.info

    @property
    def elements(self) -> List[Element] | None:
        '''
        List of elements in the lattice (None for non-lattice elements).
        '''
        if self.instance.elements is None:
            return None
        else:
            return [self.actual_element(elem) for elem in self.instance.elements]

    @property
    def indices(self) -> Tuple[int, ...]:
        '''
        Indices of this element in the lattice.

        Returns:
            Tuple[int, ...]: Indices of this element.
        '''
        return tuple(self.instance.indices)

    @length.setter
    def length(self, length: float) -> None:
        '''
        Set length of the element [m].

        Args:
            length float: Length of the element
        '''
        self.instance.length = length

    @dx.setter
    def dx(self, dx: float) -> None:
        '''
        Set horizontal offset of the element [m].

        Args:
            dx float: Horizontal offset of the element.
        '''
        self.instance.dx = dx

    @dy.setter
    def dy(self, dy: float) -> None:
        '''
        Set vertical offset of the element [m].

        Args:
            dy float: Vertical offset of the element.
        '''
        self.instance.dy = dy

    @ds.setter
    def ds(self, ds: float) -> None:
        '''
        Set longitudinal offset of the element [m].

        Args:
            ds float: Longitudinal offset of the element.
        '''
        self.instance.ds = ds

    @tilt.setter
    def tilt(self, tilt: float) -> None:
        '''
        Set tilt angle of the element [rad].

        Args:
            tilt float: Tilt angle of the element.
        '''
        self.instance.tilt = tilt

    @info.setter
    def info(self, info: str) -> None:
        '''
        Set additional information.

        Args:
            info str: Additional information
        '''
        self.instance.info = info

    def copy(self) -> Element:
        '''
        Create a copy of the element.

        Returns:
            Element: A copy of the element.
        '''
        return Element(self.instance.name, self.instance.length, self.instance.angle,
                       self.instance.dx, self.instance.dy, self.instance.ds,
                       self.instance.tilt, self.instance.info)

    def set_indices(self, indices: Tuple[int, ...] | None = None) -> None:
        '''
        Set the indices of the element in the lattice.

        Args:
            indices Tuple[int, ...] | None: Tuple of indices representing the position of the element in the lattice.
        '''
        self.instance.set_indices(indices)

    def s_array(self, ds: float = 0.1, endpoint: bool = True) -> npt.NDArray[np.floating]:
        '''
        Longitudinal position array along the element.

        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Longitudinal position array [m].
        '''
        return self.instance.s_array(ds, endpoint)

    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'midpoint') -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method.

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        return self.instance.transfer_matrix(cood0.instance if cood0 is not None else None, ds,
                                             self.INTEGRATION_METHODS[method])

    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = True, method: str = 'midpoint') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (N, 4, 4).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        tmat, s = self.instance.transfer_matrix_array(cood0.instance if cood0 is not None else None, ds, endpoint,
                                                     self.INTEGRATION_METHODS[method])
        return np.array(tmat), s

    def dispersion(self, cood0: Coordinate = None, ds: float = 0.1, method: str = 'midpoint') -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method.

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        return self.instance.dispersion(cood0.instance if cood0 is not None else None, ds,
                                        self.INTEGRATION_METHODS[method])

    def dispersion_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False, method: str = 'midpoint') \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method.

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        return self.instance.dispersion_array(cood0.instance if cood0 is not None else None, ds, endpoint,
                                              self.INTEGRATION_METHODS[method])

    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1, method: str = 'midpoint') \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method.

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        cood, evlp, disp =  self.instance.transfer(cood0.instance,
                                                   evlp0.instance if evlp0 is not None else None,
                                                   disp0.instance if disp0 is not None else None,
                                                   ds, self.INTEGRATION_METHODS[method])
        return Coordinate(instance=cood), \
            Envelope(instance=evlp) if evlp is not None else None, \
            Dispersion(instance=disp) if disp is not None else None

    def transfer_array(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None,
                       ds: float = 0.1, endpoint: bool = True, method: str = 'midpoint') \
        -> Tuple[CoordinateArray, EnvelopeArray, DispersionArray]:
        '''
        Calculate the coordinate array along the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.
            method str: Integration method.

        Returns:
            CoordinateArray: Coordinate array along the element.
            EnvelopeArray: Beam envelope array along the element (if evlp0 is provided).
            DispersionArray: Dispersion array along the element (if disp0 is provided).
        '''
        cood, evlp, disp = self.instance.transfer_array(cood0.instance,
                                                        evlp0.instance if evlp0 is not None else None,
                                                        disp0.instance if disp0 is not None else None,
                                                        ds, endpoint, self.INTEGRATION_METHODS[method])
        return CoordinateArray(None, None, instance=cood), \
            EnvelopeArray(None, None, instance=evlp) if evlp is not None else None, \
            DispersionArray(None, None, instance=disp) if disp is not None else None

    def radiation_integrals(self, cood0: Coordinate, evlp0: Envelope, disp0: Dispersion, ds: float = 0.1, method: str = 'midpoint') \
        -> Tuple[float, float, float]:
        '''
        Calculate radiation integrals.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial envelope.
            disp0 Dispersion: Initial dispersion.
            ds float: Maximum step size [m].
            method str: Integration method.

        Returns:
            Tuple[float, float, float, float, float, float]: Radiation integrals I2, I4, I5u, I5v, I4u, and I4v.
        '''
        return self.instance.radiation_integrals(cood0.instance, evlp0.instance, disp0.instance, ds, self.INTEGRATION_METHODS[method])

    def get_element_from_s(self, s: float) -> Tuple[Element, float]:
        '''
        Get element and local longitudinal position by longitudinal position.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Element: Element at the specified longitudinal position.
            float: Local longitudinal position in the element [m].
        '''
        elem_cpp, s_local = self.instance.get_element_from_s(s)
        return self.actual_element(elem_cpp), s_local

    def transfer_matrix_from_s(self, s: float, cood0: Coordinate = Coordinate(), ds: float = 0.1, method: str = 'midpoint') \
        -> npt.NDArray[np.floating]:
        '''
        Transfer matrices from the given longitudinal position to the end of the element.

        Args:
            s float: Longitudinal position [m].
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).
            method str: Integration method (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix from s to the end of the element.
        '''
        return self.instance.transfer_matrix_from_s(s, cood0.instance, ds, self.INTEGRATION_METHODS[method])

    def get_element(self, indices: int | Tuple[int, ...]) -> Element:
        '''
        Get element by index or tuple of indices.

        Args:
            indices int | Tuple[int, ...]: Index or tuple of indices.

        Returns:
            Element: Element at the specified index or indices.
        '''
        elem_cpp = self.instance.get_element(indices)
        return self.actual_element(elem_cpp)

    def set_element(self, indices: int | Tuple[int, ...], element: Element) -> None:
        '''
        Set element by index or tuple of indices.

        Args:
            indices int | Tuple[int, ...]: Index or tuple of indices.
            element Element: Element to set.
        '''
        self.instance.set_element(indices, element.instance)

    def get_s(self, element: Element | int | Tuple[int, ...]) -> float:
        '''
        Get longitudinal position by Element, index or tuple of indices.

        Args:
            element Element | int | Tuple[int, ...]: Element, index or tuple of indices.

        Returns:
            float: Longitudinal position [m].
        '''
        if isinstance(element, Element):
            indices = element.indices
        else:
            indices = element
        return self.instance.get_s(indices)

    def find_index(self, names: str | List[str]) -> List[Tuple[int, ...]]:
        '''
        Find indices of elements by their names.

        Args:
            names str | List[str]: Name or list of names of the elements.

        Returns:
            List[Tuple[int, ...]]: List of tuples of indices of the elements.
        '''
        if isinstance(names, str):
            names = [names]
        return [tuple(idx) for idx in self.instance.find_index(names)]
