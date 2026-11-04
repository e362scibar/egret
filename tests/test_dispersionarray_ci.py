# tests/test_dispersionarray_ci.py
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

"""CI checks for the DispersionArray class (pytest counterpart of test_dispersionarray_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

N = 6
S_GRID = np.arange(N, dtype=float)
RNG = np.random.default_rng(20260701)
VECTOR = RNG.normal(0.0, 1.0e-4, size=(4, N))


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def test_constructor_components_and_copy(backend):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    assert_close(arr.vector, VECTOR)
    assert_close([arr.x, arr.xp, arr.y, arr.yp], VECTOR)
    copied = arr.copy()
    copied.x = copied.x * 0.0 + 9.0
    assert_close(arr.x, VECTOR[0], err_msg="original changed by copy")


def test_constructor_copies_its_inputs(backend):
    vector, s = VECTOR.copy(), S_GRID.copy()
    arr = backend.DispersionArray(vector, s)
    vector[:] = 999.0
    s[:] = 999.0
    assert_close(arr.vector, VECTOR)
    assert_close(arr.s, S_GRID)


@pytest.mark.parametrize("name, row", [("x", 0), ("xp", 1), ("y", 2), ("yp", 3)])
def test_component_setter_updates_its_row(backend, name, row):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    new = np.linspace(0.0, 1.0, N) * 1.0e-4
    setattr(arr, name, new.copy())
    expected = VECTOR.copy()
    expected[row, :] = new
    assert_close(arr.vector, expected)


def test_append(backend):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    arr.append(backend.DispersionArray(2.0 * VECTOR, S_GRID + N))
    assert len(arr) == 2 * N
    assert_close(arr.s, np.concatenate([S_GRID, S_GRID + N]))
    assert_close(arr.vector, np.hstack([VECTOR, 2.0 * VECTOR]))


def test_from_s_is_linear_between_points(backend):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    for s_val in (0.25, 2.5, 4.0, 5.0):
        i = min(int(s_val), N - 2)
        expected = VECTOR[:, i] + (s_val - S_GRID[i]) * (VECTOR[:, i + 1] - VECTOR[:, i])
        assert_close(arr.from_s(s_val).vector, expected, atol=1.0e-12, err_msg=f"s={s_val}")


def test_from_s_out_of_range_raises_value_error(backend):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    with pytest.raises(ValueError):
        arr.from_s(S_GRID[0] - 0.1)
    with pytest.raises(ValueError):
        arr.from_s(S_GRID[-1] + 0.1)


@pytest.mark.parametrize("build", [
    lambda m: m.DispersionArray(VECTOR.copy(), S_GRID[:-1].copy()),
    lambda m: m.DispersionArray(VECTOR.copy(), S_GRID[::-1].copy()),
    lambda m: m.DispersionArray(VECTOR[:, :-1].copy(), S_GRID.copy()),
])
def test_invalid_constructor_inputs_raise_value_error(backend, build):
    with pytest.raises(ValueError):
        build(backend)


@pytest.mark.parametrize("use", [
    lambda arr: setattr(arr, "s", S_GRID[:-1].copy()),
    lambda arr: setattr(arr, "s", S_GRID[::-1].copy()),
    lambda arr: setattr(arr, "vector", VECTOR[:, :-1].copy()),
    lambda arr: setattr(arr, "x", np.array([5.0])),
])
def test_invalid_setters_raise_value_error(backend, use):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    with pytest.raises(ValueError):
        use(arr)


def test_vector_getter_cannot_change_the_object(backend):
    arr = backend.DispersionArray(VECTOR.copy(), S_GRID.copy())
    view = arr.vector
    try:
        view[0, 0] = 99.0
    except ValueError:
        return
    assert_close(arr.vector, VECTOR, err_msg="writing to the returned vector changed the object")


def test_backends_agree():
    py = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
    cpp = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
    assert_close(cpp.from_s(2.5).vector, py.from_s(2.5).vector, atol=1.0e-12)
    assert_close(cpp.vector, py.vector, atol=1.0e-12)
