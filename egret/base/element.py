# base/element.py
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
from abc import abstractmethod
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

class Element(Object):
    '''
    Base class of an accelerator element.
    '''

    @property
    @abstractmethod
    def length(self) -> float:
        '''
        Length of the element [m].
        '''
        pass

    @property
    @abstractmethod
    def angle(self) -> float:
        '''
        Bending angle of the element [rad].
        '''
        pass

    @property
    @abstractmethod
    def dx(self) -> float:
        '''
        Horizontal offset of the element [m].
        '''
        pass

    @property
    @abstractmethod
    def dy(self) -> float:
        '''
        Vertical offset of the element [m].
        '''
        pass

    @property
    @abstractmethod
    def ds(self) -> float:
        '''
        Longitudinal offset of the element [m].
        '''
        pass

    @property
    @abstractmethod
    def tilt(self) -> float:
        '''
        Tilt angle of the element [rad].
        '''
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        '''
        Additional information.
        '''
        pass

    @property
    @abstractmethod
    def elements(self) -> List[Element] | None:
        '''
        List of elements (None for single element).
        '''
        pass

    @length.setter
    @abstractmethod
    def length(self, length: float) -> None:
        '''
        Set length of the element [m].

        Args:
            length float: Length of the element [m].
        '''
        pass

    @angle.setter
    @abstractmethod
    def angle(self, angle: float) -> None:
        '''
        Set bending angle of the element [rad].

        Args:
            angle float: Bending angle of the element [rad].
        '''
        pass

    @dx.setter
    @abstractmethod
    def dx(self, dx: float) -> None:
        '''
        Set horizontal offset of the element [m].

        Args:
            dx float: Horizontal offset of the element [m].
        '''
        pass

    @dy.setter
    @abstractmethod
    def dy(self, dy: float) -> None:
        '''
        Set vertical offset of the element [m].

        Args:
            dy float: Vertical offset of the element [m].
        '''
        pass

    @ds.setter
    @abstractmethod
    def ds(self, ds: float) -> None:
        '''
        Set longitudinal offset of the element [m].

        Args:
            ds float: Longitudinal offset of the element [m].
        '''
        pass

    @tilt.setter
    @abstractmethod
    def tilt(self, tilt: float) -> None:
        '''
        Set tilt angle of the element [rad].

        Args:
            tilt float: Tilt angle of the element [rad].
        '''
        pass

    @info.setter
    @abstractmethod
    def info(self, info: str) -> None:
        '''
        Set additional information.

        Args:
            info str: Additional information.
        '''
        pass

    @abstractmethod
    def copy(self) -> Element:
        '''
        Create a copy of the element.

        Returns:
            Element: A copy of the element.
        '''
        pass

    @abstractmethod
    def set_indices(self, indices: Tuple[int, ...] | None = None) -> None:
        '''
        Set the indices of the element in the lattice.

        Args:
            indices Tuple[int, ...] | None: Tuple of indices representing the position of the element in the lattice.
        '''
        pass

    @abstractmethod
    def s_array(self, ds: float = 0.1, endpoint: bool = True) -> npt.NDArray[np.floating]:
        '''
        Longitudinal position array along the element.

        Args:
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Longitudinal position array [m].
        '''
        pass

    @abstractmethod
    def transfer_matrix(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Transfer matrix of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix.
        '''
        pass

    @abstractmethod
    def transfer_matrix_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = True) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Transfer matrix array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Transfer matrix array of shape (4, 4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        pass

    @abstractmethod
    def dispersion(self, cood0: Coordinate = None, ds: float = 0.1) -> npt.NDArray[np.floating]:
        '''
        Additive dispersion vector of the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: Dispersion vector [eta_x, eta_x', eta_y, eta_y'].
        '''
        pass

    @abstractmethod
    def dispersion_array(self, cood0: Coordinate = None, ds: float = 0.1, endpoint: bool = False) \
        -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        '''
        Additive dispersion array along the element.

        Args:
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m].
            endpoint bool: If True, include the endpoint.

        Returns:
            npt.NDArray[np.floating]: Dispersion array of shape (4, N).
            npt.NDArray[np.floating]: Longitudinal positions [m].
        '''
        pass

    @abstractmethod
    def transfer(self, cood0: Coordinate, evlp0: Envelope = None, disp0: Dispersion = None, ds: float = 0.1) \
        -> Tuple[Coordinate, Envelope, Dispersion]:
        '''
        Calculate the coordinate, envelope, and dispersion after the element.

        Args:
            cood0 Coordinate: Initial coordinate.
            evlp0 Envelope: Initial beam envelope (optional).
            disp0 Dispersion: Initial dispersion (optional).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            Coordinate: Coordinate after the element.
            Envelope: Beam envelope after the element (if evlp0 is provided).
            Dispersion: Dispersion after the element (if disp0 is provided).
        '''
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
            Tuple[float, float, float, float, float, float]: Radiation integrals I2, I4, I5u, I5v, I4u, and I4v.
        '''
        pass

    @abstractmethod
    def get_element_from_s(self, s: float) -> Tuple[Element, float]:
        '''
        Get element and local longitudinal position by longitudinal position.

        Args:
            s float: Longitudinal position [m].

        Returns:
            Element: Element at the specified longitudinal position.
            float: Local longitudinal position in the element [m].
        '''
        pass

    @abstractmethod
    def transfer_matrix_from_s(self, s: float, cood0: Coordinate, ds: float = 0.1) \
        -> npt.NDArray[np.floating]:
        '''
        Transfer matrices from the given longitudinal position to the end of the element.

        Args:
            s float: Longitudinal position [m].
            cood0 Coordinate: Initial coordinate (not used in the base class).
            ds float: Maximum step size [m] for integration (not used in the base class).

        Returns:
            npt.NDArray[np.floating]: 4x4 transfer matrix from s to the end of the element.
        '''
        pass

    @abstractmethod
    def get_element(self, indices: int | Tuple[int, ...]) -> Element:
        '''
        Get element by index or tuple of indices.

        Args:
            indices int | Tuple[int, ...]: Index or tuple of indices.

        Returns:
            Element: Element at the specified index or indices.
        '''
        pass

    @abstractmethod
    def get_s(self, indices: int | Tuple[int, ...]) -> float:
        '''
        Get longitudinal position by index or tuple of indices.

        Args:
            indices int | Tuple[int, ...]: Index or tuple of indices.

        Returns:
            float: Longitudinal position [m].
        '''
        pass

    @abstractmethod
    def find_index(self, names: str | List[str]) -> List[Tuple[int, ...]]:
        '''
        Find indices of elements by their names.

        Args:
            names str | List[str]: Element name or list of element names.

        Returns:
            List[Tuple[int, ...]]: List of tuples of indices corresponding to the element names.
        '''
        pass
