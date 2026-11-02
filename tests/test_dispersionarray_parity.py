# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: egret
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DispersionArray parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `DispersionArray` behavior parity between:
# - `egret.python.DispersionArray` (pure-Python reference)
# - `egret.cpp.DispersionArray` (C++ backend wrapper)
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
RNG_SEED = 20260701
N_POINTS = 6
S_STEP = 1.0
DISP_SCALE = 1.0e-4
ATOL = 1.0e-14

COMPONENT_ROWS = {"x": 0, "xp": 1, "y": 2, "yp": 3}

rng = np.random.default_rng(RNG_SEED)
S_GRID = np.arange(N_POINTS, dtype=float) * S_STEP
VECTOR = rng.normal(0.0, DISP_SCALE, size=(4, N_POINTS))


# %%
def array_snapshot(arr) -> dict[str, np.ndarray]:
    return {
        "vector": np.array(arr.vector, dtype=float).copy(),
        "s": np.array(arr.s, dtype=float).copy(),
        "x": np.array(arr.x, dtype=float).copy(),
        "xp": np.array(arr.xp, dtype=float).copy(),
        "y": np.array(arr.y, dtype=float).copy(),
        "yp": np.array(arr.yp, dtype=float).copy(),
    }


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


def plot_profile_parity(s_query: np.ndarray, py_eta_x: np.ndarray, cpp_eta_x: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(s_query, py_eta_x, "o-", label="pyegret")
    ax.plot(s_query, cpp_eta_x, "x--", label="cppegret")
    ax.set_title(title)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("eta_x [m]")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity (vector, s, and components)

# %%
py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())

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
py_arr = pyegret.DispersionArray(vector_in, s_in)
cpp_arr = cppegret.DispersionArray(vector_in, s_in)

vector_in[:] = 999.0
s_in[:] = 999.0

assert_snapshot_equal(array_snapshot(py_arr), array_snapshot(cpp_arr))
np.testing.assert_allclose(array_snapshot(py_arr)["vector"], VECTOR, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(array_snapshot(py_arr)["s"], S_GRID, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(array_snapshot(cpp_arr)["vector"], VECTOR, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(array_snapshot(cpp_arr)["s"], S_GRID, atol=ATOL, rtol=0.0)
print("Case 2 passed: defensive copy of inputs")


# %% [markdown]
# ## Case 3: `copy()` parity and independence of the copy

# %%
py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())

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
print("Case 3 passed: copy parity and independence")


# %% [markdown]
# ## Case 4: component setters update the corresponding row of `vector`

# %%
for name, row in COMPONENT_ROWS.items():
    new_values = rng.normal(0.0, DISP_SCALE, size=N_POINTS)

    py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
    cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
    setattr(py_arr, name, new_values.copy())
    setattr(cpp_arr, name, new_values.copy())

    expected = VECTOR.copy()
    expected[row, :] = new_values

    assert_snapshot_equal(array_snapshot(py_arr), array_snapshot(cpp_arr))
    np.testing.assert_allclose(np.asarray(py_arr.vector), expected, atol=ATOL, rtol=0.0, err_msg=name)
    np.testing.assert_allclose(np.asarray(cpp_arr.vector), expected, atol=ATOL, rtol=0.0, err_msg=name)
    print(f"Setter {name!r} parity passed")
print("Case 4 passed: component setters")


# %% [markdown]
# ## Case 5: `append()` parity

# %%
s_second = S_GRID + (S_GRID[-1] + S_STEP)
vector_second = VECTOR * 2.0

py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
py_second = pyegret.DispersionArray(vector_second.copy(), s_second.copy())
cpp_second = cppegret.DispersionArray(vector_second.copy(), s_second.copy())

py_arr.append(py_second)
cpp_arr.append(cpp_second)

assert len(py_arr) == len(cpp_arr) == 2 * N_POINTS
assert_snapshot_equal(array_snapshot(py_arr), array_snapshot(cpp_arr))
np.testing.assert_allclose(py_arr.s, np.concatenate([S_GRID, s_second]), atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(py_arr.vector), np.hstack([VECTOR, vector_second]), atol=ATOL, rtol=0.0)
print("Case 5 passed: append parity")


# %% [markdown]
# ## Case 6: `from_s` interpolation parity over the full range (including the last interval)

# %%
S_QUERY = np.array([0.0, 0.25, 1.0, 1.5, 2.0, 3.75, 4.0, 4.5, 5.0])

py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())

py_disp = [py_arr.from_s(s) for s in S_QUERY]
cpp_disp = [cpp_arr.from_s(s) for s in S_QUERY]

for s_val, py_d, cpp_d in zip(S_QUERY, py_disp, cpp_disp):
    np.testing.assert_allclose(np.asarray(py_d.vector), np.asarray(cpp_d.vector), atol=ATOL, rtol=0.0,
                               err_msg=f"s={s_val}")
    assert py_d.s == cpp_d.s == s_val

py_eta_x = np.array([np.asarray(d.vector)[0] for d in py_disp])
cpp_eta_x = np.array([np.asarray(d.vector)[0] for d in cpp_disp])
display({"case6_s": S_QUERY, "case6_py_eta_x": py_eta_x, "case6_cpp_eta_x": cpp_eta_x})
plot_profile_parity(S_QUERY, py_eta_x, cpp_eta_x, "Case 6 interpolated eta_x parity")
print("Case 6 passed: from_s parity over the full range")


# %% [markdown]
# ## Case 7: `from_s` at the left endpoint

# %%
py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())

np.testing.assert_allclose(np.asarray(py_arr.from_s(S_GRID[0]).vector), VECTOR[:, 0], atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.from_s(S_GRID[0]).vector), VECTOR[:, 0], atol=ATOL, rtol=0.0)
print("Case 7 passed: left endpoint parity")


# %% [markdown]
# ## Case 8: invalid `from_s` requests raise `ValueError` in both backends

# %%
S_QUERY_OUT_OF_RANGE = [S_GRID[0] - 0.1, S_GRID[-1] + 0.1]

py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())

for s_val in S_QUERY_OUT_OF_RANGE:
    assert_raises(lambda: py_arr.from_s(s_val), ValueError)
    assert_raises(lambda: cpp_arr.from_s(s_val), ValueError)

single_point_py = pyegret.DispersionArray(VECTOR[:, :1].copy(), S_GRID[:1].copy())
single_point_cpp = cppegret.DispersionArray(VECTOR[:, :1].copy(), S_GRID[:1].copy())
assert_raises(lambda: single_point_py.from_s(S_GRID[0]), ValueError)
assert_raises(lambda: single_point_cpp.from_s(S_GRID[0]), ValueError)
print("Case 8 passed: invalid from_s requests raise ValueError in both backends")


# %% [markdown]
# ## Case 9: constructor rejects a size mismatch between `vector` and `s` with `ValueError`

# %%
assert_raises(lambda: pyegret.DispersionArray(VECTOR.copy(), S_GRID[:-1].copy()), ValueError)
assert_raises(lambda: cppegret.DispersionArray(VECTOR.copy(), S_GRID[:-1].copy()), ValueError)
print("Case 9 passed: size mismatch rejected in both backends")


# %% [markdown]
# ## Case 10: constructor rejects a decreasing `s` with `ValueError`

# %%
assert_raises(lambda: pyegret.DispersionArray(VECTOR.copy(), S_GRID[::-1].copy()), ValueError)
assert_raises(lambda: cppegret.DispersionArray(VECTOR.copy(), S_GRID[::-1].copy()), ValueError)
print("Case 10 passed: decreasing s rejected in both backends")


# %% [markdown]
# ## Case 11: the `vector` getter does not expose the internal storage

# %%
py_arr = pyegret.DispersionArray(VECTOR.copy(), S_GRID.copy())
cpp_arr = cppegret.DispersionArray(VECTOR.copy(), S_GRID.copy())

assert returned_vector_is_isolated(py_arr), "pyegret vector getter exposes internal storage"
assert returned_vector_is_isolated(cpp_arr), "cppegret vector getter exposes internal storage"
print("Case 11 passed: vector getter is isolated from internal storage")


# %% [markdown]
# ## Case 12: wrong lengths and unsorted `s` raise `ValueError` in the constructor and setters

# %%
INVALID_DISPERSION_USES = {
    "constructor: vector columns": lambda m: m.DispersionArray(VECTOR[:, :-1].copy(), S_GRID.copy()),
    "s setter: length": lambda m: setattr(m.DispersionArray(VECTOR.copy(), S_GRID.copy()), "s", S_GRID[:-1].copy()),
    "s setter: decreasing": lambda m: setattr(m.DispersionArray(VECTOR.copy(), S_GRID.copy()), "s", S_GRID[::-1].copy()),
    "vector setter: columns": lambda m: setattr(m.DispersionArray(VECTOR.copy(), S_GRID.copy()), "vector", VECTOR[:, :-1].copy()),
    "x setter: length 1 (no broadcasting)": lambda m: setattr(m.DispersionArray(VECTOR.copy(), S_GRID.copy()), "x", np.array([5.0])),
    "xp setter: length": lambda m: setattr(m.DispersionArray(VECTOR.copy(), S_GRID.copy()), "xp", np.zeros(N_POINTS - 1)),
}

for name, use in INVALID_DISPERSION_USES.items():
    for module in (pyegret, cppegret):
        try:
            assert_raises(lambda: use(module), ValueError)
        except AssertionError as exc:
            raise AssertionError(f"{name} [{module.__name__}]: {exc}") from exc
print(f"Case 12 passed: {len(INVALID_DISPERSION_USES)} invalid uses rejected in both backends")


# %% [markdown]
# ## Summary

# %%
print("DispersionArray parity cases finished.")
