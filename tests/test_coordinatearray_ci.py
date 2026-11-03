"""CI checks for the CoordinateArray class (pytest counterpart of test_coordinatearray_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

N = 6
S_GRID = np.arange(N, dtype=float)
RNG = np.random.default_rng(20260615)
VECTOR = RNG.normal(0.0, 1.0e-3, size=(4, N))
Z_VALUES = np.linspace(-3.0e-2, 3.0e-2, N)
DELTA_VALUES = RNG.normal(0.0, 2.5e-4, size=N)


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def test_constructor_defaults_z_and_delta_to_zero(backend):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy())
    assert_close(arr.z, np.zeros(N))
    assert_close(arr.delta, np.zeros(N))
    assert_close(arr.vector, VECTOR)
    assert_close(arr.s, S_GRID)
    assert len(arr) == N


def test_constructor_copies_its_inputs(backend):
    vector, s = VECTOR.copy(), S_GRID.copy()
    arr = backend.CoordinateArray(vector, s, Z_VALUES.copy(), DELTA_VALUES.copy())
    vector[:] = 999.0
    s[:] = 999.0
    assert_close(arr.vector, VECTOR)
    assert_close(arr.s, S_GRID)


def test_copy_and_append(backend):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
    copied = arr.copy()
    copied.x = copied.x + 1.0
    assert_close(arr.x, VECTOR[0], err_msg="original changed by copy")
    other = backend.CoordinateArray(2.0 * VECTOR, S_GRID + N, Z_VALUES.copy(), DELTA_VALUES.copy())
    arr.append(other)
    assert len(arr) == 2 * N
    assert_close(arr.s, np.concatenate([S_GRID, S_GRID + N]))
    assert_close(arr.vector, np.hstack([VECTOR, 2.0 * VECTOR]))


@pytest.mark.parametrize("name, row", [("x", 0), ("xp", 1), ("y", 2), ("yp", 3)])
def test_component_setter_updates_its_row(backend, name, row):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy())
    new = np.linspace(0.0, 1.0, N)
    setattr(arr, name, new.copy())
    expected = VECTOR.copy()
    expected[row, :] = new
    assert_close(arr.vector, expected)


def test_from_s_interpolates_linearly_inside_and_at_endpoints(backend):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
    for s_val in (0.0, 0.5, 2.25, 4.0, 5.0):
        coord = arr.from_s(s_val)
        i = min(int(s_val), N - 2)
        t = (s_val - S_GRID[i])
        expected = VECTOR[:, i] + t * (VECTOR[:, i + 1] - VECTOR[:, i])
        assert_close(coord.vector, expected, atol=1.0e-12, err_msg=f"s={s_val}")


def test_from_s_out_of_range_and_too_short_raise_value_error(backend):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy())
    with pytest.raises(ValueError):
        arr.from_s(S_GRID[0] - 0.1)
    with pytest.raises(ValueError):
        arr.from_s(S_GRID[-1] + 0.1)
    short = backend.CoordinateArray(VECTOR[:, :1].copy(), S_GRID[:1].copy())
    with pytest.raises(ValueError):
        short.from_s(S_GRID[0])


@pytest.mark.parametrize("build", [
    lambda m: m.CoordinateArray(VECTOR.copy(), S_GRID[:-1].copy()),
    lambda m: m.CoordinateArray(VECTOR.copy(), S_GRID[::-1].copy()),
    lambda m: m.CoordinateArray(VECTOR[:, :-1].copy(), S_GRID.copy()),
    lambda m: m.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES[:-1].copy()),
    lambda m: m.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES[:-1].copy()),
])
def test_invalid_constructor_inputs_raise_value_error(backend, build):
    with pytest.raises(ValueError):
        build(backend)


@pytest.mark.parametrize("use", [
    lambda arr: setattr(arr, "s", S_GRID[:-1].copy()),
    lambda arr: setattr(arr, "s", S_GRID[::-1].copy()),
    lambda arr: setattr(arr, "vector", VECTOR[:, :-1].copy()),
    lambda arr: setattr(arr, "z", Z_VALUES[:-1].copy()),
    lambda arr: setattr(arr, "delta", DELTA_VALUES[:-1].copy()),
    lambda arr: setattr(arr, "x", np.array([5.0])),
])
def test_invalid_setters_raise_value_error(backend, use):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
    with pytest.raises(ValueError):
        use(arr)


def test_vector_getter_cannot_change_the_object(backend):
    arr = backend.CoordinateArray(VECTOR.copy(), S_GRID.copy())
    view = arr.vector
    try:
        view[0, 0] = 99.0
    except ValueError:
        return  # read-only view is acceptable
    assert_close(arr.vector, VECTOR, err_msg="writing to the returned vector changed the object")


def test_backends_agree_on_from_s_and_append():
    py = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
    cpp = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
    assert_close(cpp.from_s(2.5).vector, py.from_s(2.5).vector, atol=1.0e-12)
    py.append(pyegret.CoordinateArray(VECTOR.copy(), S_GRID + N, Z_VALUES.copy(), DELTA_VALUES.copy()))
    cpp.append(cppegret.CoordinateArray(VECTOR.copy(), S_GRID + N, Z_VALUES.copy(), DELTA_VALUES.copy()))
    assert_close(cpp.vector, py.vector, atol=1.0e-12)
