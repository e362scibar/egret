# tests/test_coordinate_ci.py
#
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
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

"""CI checks for the Coordinate class (pytest counterpart of test_coordinate_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

VECTOR = np.array([1.0e-3, -2.0e-4, 3.0e-3, -4.0e-4])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def test_default_constructor_is_zero(backend):
    cood = backend.Coordinate()
    assert_close(cood.vector, np.zeros(4))
    assert_close([cood.s, cood.z, cood.delta], [0.0, 0.0, 0.0])


def test_constructor_components_and_accessors(backend):
    cood = backend.Coordinate(VECTOR.copy(), 12.5, -0.03, 2.5e-4)
    assert_close([cood.x, cood.xp, cood.y, cood.yp], VECTOR)
    assert_close([cood.s, cood.z, cood.delta], [12.5, -0.03, 2.5e-4])


def test_constructor_copies_its_input(backend):
    vector = VECTOR.copy()
    cood = backend.Coordinate(vector, 0.0, 0.0, 0.0)
    vector[:] = 999.0
    assert_close(cood.vector, VECTOR, err_msg="coordinate follows the input array")


def test_copy_is_independent(backend):
    cood = backend.Coordinate(VECTOR.copy(), 1.0, 0.0, 0.0)
    copied = cood.copy()
    assert copied is not cood
    copied.x = copied.x + 1.0
    assert_close(cood.x, VECTOR[0], err_msg="original changed by copy")
    assert_close(copied.x, VECTOR[0] + 1.0)


def test_vector_setter_replaces_components(backend):
    cood = backend.Coordinate(VECTOR.copy())
    cood.vector = np.array([1.0, 2.0, 3.0, 4.0])
    assert_close([cood.x, cood.xp, cood.y, cood.yp], [1.0, 2.0, 3.0, 4.0])


def test_backends_agree_on_components_and_copy():
    py = pyegret.Coordinate(VECTOR.copy(), 12.5, -0.03, 2.5e-4)
    cpp = cppegret.Coordinate(VECTOR.copy(), 12.5, -0.03, 2.5e-4)
    assert_close(cpp.vector, py.vector)
    assert_close([cpp.s, cpp.z, cpp.delta], [py.s, py.z, py.delta])
    assert_close(cpp.copy().vector, py.copy().vector)
