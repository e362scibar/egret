# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: egret
#     language: python
#     name: python3
# ---

# %% [markdown]
# tests/test_coordinatearray_parity.py
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


# %% [markdown]
# # CoordinateArray parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `CoordinateArray` behavior parity between:
# - `egret.python.CoordinateArray` (pure-Python reference)
# - `egret.cpp.CoordinateArray` (C++ backend wrapper)
#
# Synthetic, deterministic fixtures are used (no SPring-8-II lattice).
# Trivial setter/getter-only checks are limited to the cases where the setter has side effects.

# %%
from __future__ import annotations

import sys

import matplotlib

# Plain `python` runs must not open plot windows; Jupyter (IPython loaded) keeps its inline backend.
if "IPython" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import egret.python as pyegret
import egret.cpp as cppegret

try:
    from IPython.display import display
except Exception:  # pragma: no cover - fallback for non-IPython execution
    def display(obj):
        print(obj)


# %%
RNG_SEED = 20260615
N_POINTS = 6
S_STEP = 1.0
VECTOR_SCALE = 1.0e-3
DELTA_SCALE = 2.5e-4
ATOL = 1.0e-14

COMPONENT_ROWS = {"x": 0, "xp": 1, "y": 2, "yp": 3}

rng = np.random.default_rng(RNG_SEED)
S_GRID = np.arange(N_POINTS, dtype=float) * S_STEP
VECTOR = rng.normal(0.0, VECTOR_SCALE, size=(4, N_POINTS))
Z_VALUES = np.linspace(-3.0e-2, 3.0e-2, N_POINTS)
DELTA_VALUES = rng.normal(0.0, DELTA_SCALE, size=N_POINTS)


# %%
def array_snapshot(arr, include_z_delta: bool = False) -> dict[str, np.ndarray]:
    snap = {
        "vector": np.array(arr.vector, dtype=float).copy(),
        "s": np.array(arr.s, dtype=float).copy(),
        "x": np.array(arr.x, dtype=float).copy(),
        "xp": np.array(arr.xp, dtype=float).copy(),
        "y": np.array(arr.y, dtype=float).copy(),
        "yp": np.array(arr.yp, dtype=float).copy(),
    }
    if include_z_delta:
        snap["z"] = np.array(arr.z, dtype=float).copy()
        snap["delta"] = np.array(arr.delta, dtype=float).copy()
    return snap


def assert_snapshot_equal(left: dict, right: dict, atol: float = ATOL) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key], right[key], atol=atol, rtol=0.0, err_msg=key)


def assert_raises(fn, expected: type[Exception]) -> None:
    try:
        fn()
    except expected:
        return
    except Exception as exc:
        raise AssertionError(f"expected {expected.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {expected.__name__} but none was raised")


def returned_vector_is_isolated(arr) -> bool:
    # A read-only view is acceptable; a writable view that changes the object is not.
    before = np.array(arr.vector, copy=True)
    view = arr.vector
    try:
        view[0, 0] = before[0, 0] + 1.0
    except ValueError:
        return True
    return np.array_equal(arr.vector, before)


def plot_profile_parity(s_query: np.ndarray, py_x: np.ndarray, cpp_x: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(s_query, py_x, "o-", label="pyegret")
    ax.plot(s_query, cpp_x, "x--", label="cppegret")
    ax.set_title(title)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("x [m]")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity (vector, s, and derived components)

# %%
py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())

py_snap = array_snapshot(py_arr)
cpp_snap = array_snapshot(cpp_arr)
assert_snapshot_equal(py_snap, cpp_snap)
np.testing.assert_allclose(py_snap["vector"], VECTOR, atol=ATOL, rtol=0.0)
assert len(py_arr) == len(cpp_arr) == N_POINTS
display({"case1_py_snapshot": py_snap, "case1_cpp_snapshot": cpp_snap})
print("Case 1 passed: constructor parity")


# %% [markdown]
# ## Case 2: constructor copies its inputs (defensive copy)

# %%
vector_in = VECTOR.copy()
s_in = S_GRID.copy()
py_arr = pyegret.CoordinateArray(vector_in, s_in)
cpp_arr = cppegret.CoordinateArray(vector_in, s_in)

vector_in[:] = 999.0
s_in[:] = 999.0

py_snap = array_snapshot(py_arr)
cpp_snap = array_snapshot(cpp_arr)
assert_snapshot_equal(py_snap, cpp_snap)
np.testing.assert_allclose(py_snap["vector"], VECTOR, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(py_snap["s"], S_GRID, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(cpp_snap["vector"], VECTOR, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(cpp_snap["s"], S_GRID, atol=ATOL, rtol=0.0)
print("Case 2 passed: defensive copy of inputs")


# %% [markdown]
# ## Case 3: explicit `z` and `delta` parity

# %%
py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())

py_snap = array_snapshot(py_arr, include_z_delta=True)
cpp_snap = array_snapshot(cpp_arr, include_z_delta=True)
assert_snapshot_equal(py_snap, cpp_snap)
np.testing.assert_allclose(py_snap["z"], Z_VALUES, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(py_snap["delta"], DELTA_VALUES, atol=ATOL, rtol=0.0)
display({"case3_py_snapshot": py_snap, "case3_cpp_snapshot": cpp_snap})
print("Case 3 passed: explicit z and delta parity")


# %% [markdown]
# ## Case 4: omitted `z` and `delta` default to zeros in both backends

# %%
py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())

np.testing.assert_allclose(np.asarray(py_arr.z), np.zeros(N_POINTS), atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(py_arr.delta), np.zeros(N_POINTS), atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.z), np.zeros(N_POINTS), atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.delta), np.zeros(N_POINTS), atol=ATOL, rtol=0.0)
print("Case 4 passed: default z and delta are zeros")


# %% [markdown]
# ## Case 5: `copy()` parity and independence of the copy

# %%
py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())

py_copy = py_arr.copy()
cpp_copy = cpp_arr.copy()
assert py_copy is not py_arr
assert cpp_copy is not cpp_arr

assert_snapshot_equal(array_snapshot(py_copy), array_snapshot(cpp_copy))

py_copy.x = py_copy.x + 1.0
cpp_copy.x = cpp_copy.x + 1.0

np.testing.assert_allclose(py_arr.x, VECTOR[0], atol=ATOL, rtol=0.0)
np.testing.assert_allclose(cpp_arr.x, VECTOR[0], atol=ATOL, rtol=0.0)
np.testing.assert_allclose(py_copy.x, VECTOR[0] + 1.0, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(cpp_copy.x, VECTOR[0] + 1.0, atol=ATOL, rtol=0.0)
print("Case 5 passed: copy parity and independence")


# %% [markdown]
# ## Case 6: component setters update the corresponding row of `vector`

# %%
for name, row in COMPONENT_ROWS.items():
    new_values = rng.normal(0.0, VECTOR_SCALE, size=N_POINTS)

    py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
    cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
    setattr(py_arr, name, new_values.copy())
    setattr(cpp_arr, name, new_values.copy())

    expected = VECTOR.copy()
    expected[row, :] = new_values

    assert_snapshot_equal(array_snapshot(py_arr), array_snapshot(cpp_arr))
    np.testing.assert_allclose(np.asarray(py_arr.vector), expected, atol=ATOL, rtol=0.0, err_msg=name)
    np.testing.assert_allclose(np.asarray(cpp_arr.vector), expected, atol=ATOL, rtol=0.0, err_msg=name)
    print(f"Setter {name!r} parity passed")
print("Case 6 passed: component setters")


# %% [markdown]
# ## Case 7: `append()` parity

# %%
s_second = S_GRID + (S_GRID[-1] + S_STEP)
vector_second = VECTOR * 2.0

py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
py_second = pyegret.CoordinateArray(vector_second.copy(), s_second.copy())
cpp_second = cppegret.CoordinateArray(vector_second.copy(), s_second.copy())

py_arr.append(py_second)
cpp_arr.append(cpp_second)

assert len(py_arr) == len(cpp_arr) == 2 * N_POINTS
assert_snapshot_equal(array_snapshot(py_arr), array_snapshot(cpp_arr))
np.testing.assert_allclose(py_arr.s, np.concatenate([S_GRID, s_second]), atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(py_arr.vector), np.hstack([VECTOR, vector_second]), atol=ATOL, rtol=0.0)
print("Case 7 passed: append parity")


# %% [markdown]
# ## Case 8: `from_s` interpolation parity inside the range (excluding the last interval)

# %%
S_QUERY_INTERIOR = np.array([0.0, 0.25, 1.0, 1.5, 2.0, 3.0, 3.75])

py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES.copy())

py_coords = [py_arr.from_s(s) for s in S_QUERY_INTERIOR]
cpp_coords = [cpp_arr.from_s(s) for s in S_QUERY_INTERIOR]

for s_val, py_c, cpp_c in zip(S_QUERY_INTERIOR, py_coords, cpp_coords):
    np.testing.assert_allclose(np.asarray(py_c.vector), np.asarray(cpp_c.vector), atol=ATOL, rtol=0.0, err_msg=f"s={s_val}")
    assert py_c.s == cpp_c.s == s_val

py_x = np.array([np.asarray(c.vector)[0] for c in py_coords])
cpp_x = np.array([np.asarray(c.vector)[0] for c in cpp_coords])
display({"case8_s": S_QUERY_INTERIOR, "case8_py_x": py_x, "case8_cpp_x": cpp_x})
plot_profile_parity(S_QUERY_INTERIOR, py_x, cpp_x, "Case 8 interpolated x parity (interior)")
print("Case 8 passed: interior from_s parity")


# %% [markdown]
# ## Case 9: `from_s` at the left endpoint

# %%
py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())

np.testing.assert_allclose(np.asarray(py_arr.from_s(S_GRID[0]).vector), VECTOR[:, 0], atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.from_s(S_GRID[0]).vector), VECTOR[:, 0], atol=ATOL, rtol=0.0)
print("Case 9 passed: left endpoint parity")


# %% [markdown]
# ## Case 10: `from_s` in the last interval, including the right endpoint
#
# The last interval is interpolated by both backends through the same interval selection.

# %%
S_QUERY_LAST = np.array([S_GRID[-2], 0.5 * (S_GRID[-2] + S_GRID[-1]), S_GRID[-1]])

py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())

for s_val in S_QUERY_LAST:
    py_vec = np.asarray(py_arr.from_s(s_val).vector)
    cpp_vec = np.asarray(cpp_arr.from_s(s_val).vector)
    np.testing.assert_allclose(py_vec, cpp_vec, atol=ATOL, rtol=0.0, err_msg=f"s={s_val}")
print("Case 10 passed: last-interval and right-endpoint from_s parity")


# %% [markdown]
# ## Case 11: invalid `from_s` requests raise in both backends
#
# Both backends raise `ValueError`, matching the Python reference.

# %%
S_QUERY_OUT_OF_RANGE = [S_GRID[0] - 0.1, S_GRID[-1] + 0.1]

py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())

for s_val in S_QUERY_OUT_OF_RANGE:
    assert_raises(lambda: py_arr.from_s(s_val), ValueError)
    assert_raises(lambda: cpp_arr.from_s(s_val), ValueError)

single_point_py = pyegret.CoordinateArray(VECTOR[:, :1].copy(), S_GRID[:1].copy())
single_point_cpp = cppegret.CoordinateArray(VECTOR[:, :1].copy(), S_GRID[:1].copy())
assert_raises(lambda: single_point_py.from_s(S_GRID[0]), ValueError)
assert_raises(lambda: single_point_cpp.from_s(S_GRID[0]), ValueError)

print("Case 11 passed: invalid from_s requests raise ValueError in both backends")


# %% [markdown]
# ## Case 12: constructor rejects a size mismatch between `vector` and `s` with `ValueError`

# %%
assert_raises(lambda: pyegret.CoordinateArray(VECTOR.copy(), S_GRID[:-1].copy()), ValueError)
assert_raises(lambda: cppegret.CoordinateArray(VECTOR.copy(), S_GRID[:-1].copy()), ValueError)
print("Case 12 passed: size mismatch rejected in both backends")


# %% [markdown]
# ## Case 13: constructor rejects a decreasing `s` with `ValueError`

# %%
assert_raises(lambda: pyegret.CoordinateArray(VECTOR.copy(), S_GRID[::-1].copy()), ValueError)
assert_raises(lambda: cppegret.CoordinateArray(VECTOR.copy(), S_GRID[::-1].copy()), ValueError)
print("Case 13 passed: decreasing s rejected in both backends")


# %% [markdown]
# ## Case 14: the `vector` getter does not expose the internal storage

# %%
py_arr = pyegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.CoordinateArray(VECTOR.copy(), S_GRID.copy())

assert returned_vector_is_isolated(py_arr), "pyegret vector getter exposes internal storage"
assert returned_vector_is_isolated(cpp_arr), "cppegret vector getter exposes internal storage"
print("Case 14 passed: vector getter is isolated from internal storage")


# %% [markdown]
# ## Case 15: wrong lengths and unsorted `s` raise `ValueError` in the constructor and setters

# %%
INVALID_COORDINATE_USES = {
    "constructor: z length": lambda m: m.CoordinateArray(VECTOR.copy(), S_GRID.copy(), Z_VALUES[:-1].copy()),
    "constructor: delta length": lambda m: m.CoordinateArray(
        VECTOR.copy(), S_GRID.copy(), Z_VALUES.copy(), DELTA_VALUES[:-1].copy()
    ),
    "constructor: vector columns": lambda m: m.CoordinateArray(VECTOR[:, :-1].copy(), S_GRID.copy()),
    "s setter: length": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "s", S_GRID[:-1].copy()),
    "s setter: decreasing": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "s", S_GRID[::-1].copy()),
    "vector setter: columns": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "vector", VECTOR[:, :-1].copy()),
    "z setter: length": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "z", Z_VALUES[:-1].copy()),
    "delta setter: length": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "delta", DELTA_VALUES[:-1].copy()),
    "x setter: length 1 (no broadcasting)": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "x", np.array([5.0])),
    "yp setter: length": lambda m: setattr(m.CoordinateArray(VECTOR.copy(), S_GRID.copy()), "yp", np.zeros(N_POINTS - 1)),
}

for name, use in INVALID_COORDINATE_USES.items():
    for module in (pyegret, cppegret):
        try:
            assert_raises(lambda: use(module), ValueError)
        except AssertionError as exc:
            raise AssertionError(f"{name} [{module.__name__}]: {exc}") from exc
print(f"Case 15 passed: {len(INVALID_COORDINATE_USES)} invalid uses rejected in both backends")


# %% [markdown]
# ## Summary

# %%
print("CoordinateArray parity cases finished.")
