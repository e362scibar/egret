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
# tests/test_steering_parity.py
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
# # Steering parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Steering` behavior parity between:
# - `egret.python.Steering` (pure-Python reference)
# - `egret.cpp.Steering` (C++ backend wrapper)
#
# The steering magnet is a drift with a kick distributed uniformly along its length. The kick is rotated by
# the tilt and divided by `1 + delta`. Results are checked against that analytic map: the exit coordinate
# becomes `x -> x + L x' + (L/2) kick_x`, `x' -> x' + kick_x` (same for y), and inside the magnet
# `x'(s) = x' + kick_x s / L` and `x(s) = x + x' s + kick_x s^2 / (2 L)`.
# Synthetic fixtures are used (no SPring-8-II lattice).

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
LENGTH = 1.0
KICK_X = 1.0e-4
KICK_Y = -2.0e-4
ATOL = 1.0e-12
METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")
COORD_VECTOR = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])  # x, x', y, y'
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])


def drift_matrix(length: float) -> np.ndarray:
    tmat = np.eye(4)
    tmat[0, 1] = length
    tmat[2, 3] = length
    return tmat


def tilted_kick(kick_x: float, kick_y: float, tilt: float = 0.0, delta: float = 0.0) -> np.ndarray:
    # Kick rotated by the tilt and scaled by the relative energy deviation, as in the reference.
    cosphi, sinphi = np.cos(tilt), np.sin(tilt)
    rotated = np.array([[cosphi, -sinphi], [sinphi, cosphi]]) @ np.array([kick_x, kick_y]) / (1.0 + delta)
    return rotated


def analytic_steering_map(length: float, kick_x: float, kick_y: float, tilt: float = 0.0,
                          delta: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # Returns the drift matrix and the kick vector [L/2 kx, kx, L/2 ky, ky] added after it.
    kx, ky = tilted_kick(kick_x, kick_y, tilt, delta)
    kick_vector = np.array([0.5 * length * kx, kx, 0.5 * length * ky, ky])
    return drift_matrix(length), kick_vector


def analytic_steering_dispersion(length: float, kick_x: float, kick_y: float, tilt: float = 0.0,
                                 delta: float = 0.0) -> np.ndarray:
    kx, ky = tilted_kick(kick_x, kick_y, tilt, delta)
    return -np.array([0.5 * length * kx, kx, 0.5 * length * ky, ky])


def make_steering(module, kick_x: float = KICK_X, kick_y: float = KICK_Y, tilt: float = 0.0,
                  length: float = LENGTH, **kwargs):
    return module.Steering("S", length, kick_x, kick_y, tilt=tilt, **kwargs)


def make_coordinate(module, vector: np.ndarray = COORD_VECTOR, delta: float = DELTA0):
    return module.Coordinate(vector.copy(), S0, Z0, delta)


def assert_close(left, right, atol: float = ATOL, err_msg: str = "") -> None:
    np.testing.assert_allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float),
                               atol=atol, rtol=0.0, err_msg=err_msg)


def assert_raises(fn, expected: type[Exception]) -> None:
    try:
        fn()
    except expected:
        return
    except Exception as exc:
        raise AssertionError(f"expected {expected.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {expected.__name__} but none was raised")


def plot_profile_parity(s_grid: np.ndarray, py_x: np.ndarray, cpp_x: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(s_grid, py_x, "o-", label="pyegret")
    ax.plot(s_grid, cpp_x, "x--", label="cppegret")
    ax.set_title(title)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("x [m]")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity, attributes, and `copy()` independence

# %%
py_st = make_steering(pyegret, tilt=0.2, dx=1.0e-3, dy=-2.0e-3, ds=0.1, info="corrector")
cpp_st = make_steering(cppegret, tilt=0.2, dx=1.0e-3, dy=-2.0e-3, ds=0.1, info="corrector")

for name, expected in {"length": LENGTH, "kick_x": KICK_X, "kick_y": KICK_Y, "tilt": 0.2,
                       "dx": 1.0e-3, "dy": -2.0e-3, "ds": 0.1}.items():
    assert_close(getattr(py_st, name), expected, err_msg=f"py {name}")
    assert_close(getattr(cpp_st, name), expected, err_msg=f"cpp {name}")
assert_close(py_st.kick, (KICK_X, KICK_Y), err_msg="py kick tuple")
assert_close(cpp_st.kick, (KICK_X, KICK_Y), err_msg="cpp kick tuple")
assert py_st.info == cpp_st.info == "corrector"

py_copy = py_st.copy()
cpp_copy = cpp_st.copy()
assert_close(py_copy.kick, py_st.kick, err_msg="py copy kick")
assert_close(cpp_copy.kick, cpp_st.kick, err_msg="cpp copy kick")

py_copy.kick_x = 5.0e-4
cpp_copy.kick_x = 5.0e-4
assert_close(py_st.kick_x, KICK_X, err_msg="py original changed by copy")
assert_close(cpp_st.kick_x, KICK_X, err_msg="cpp original changed by copy")
assert_close(py_copy.kick_x, 5.0e-4)
assert_close(cpp_copy.kick_x, 5.0e-4)
print("Case 1 passed: constructor parity, attributes, and copy independence")


# %% [markdown]
# ## Case 2: transfer matrix parity (drift form), method independence, and array layout

# %%
expected_tmat = drift_matrix(LENGTH)
assert_close(make_steering(pyegret).transfer_matrix(), expected_tmat, err_msg="py tmat")
assert_close(make_steering(cppegret).transfer_matrix(), expected_tmat, err_msg="cpp tmat")

for module in (pyegret, cppegret):
    for method in METHODS:
        tmat = make_steering(module).transfer_matrix(make_coordinate(module), 0.1, method)
        assert_close(tmat, expected_tmat, err_msg=f"{module.__name__} method={method}")

for endpoint in (True, False):
    py_arr, py_s = make_steering(pyegret).transfer_matrix_array(ds=0.25, endpoint=endpoint)
    cpp_arr, cpp_s = make_steering(cppegret).transfer_matrix_array(ds=0.25, endpoint=endpoint)
    assert np.asarray(py_arr).shape == np.asarray(cpp_arr).shape == (len(py_s), 4, 4)
    assert_close(py_s, cpp_s, err_msg=f"s endpoint={endpoint}")
    assert_close(py_arr, cpp_arr, err_msg=f"tmat array endpoint={endpoint}")
    for k, s_val in enumerate(py_s):
        assert_close(np.asarray(py_arr)[k], drift_matrix(s_val), err_msg=f"analytic at s={s_val}")
print("Case 2 passed: drift-form transfer matrix, method independence, and N-first array layout")


# %% [markdown]
# ## Case 3: coordinate transfer parity and the analytic kick map

# %%
expected_out = analytic_steering_map(LENGTH, KICK_X, KICK_Y, delta=0.0)
tmat0, kick0 = expected_out
expected_vector = tmat0 @ COORD_VECTOR + kick0

py_out, _, _ = make_steering(pyegret).transfer(make_coordinate(pyegret, delta=0.0))
cpp_out, _, _ = make_steering(cppegret).transfer(make_coordinate(cppegret, delta=0.0))
assert_close(py_out.vector, expected_vector, err_msg="py coordinate analytic")
assert_close(cpp_out.vector, expected_vector, err_msg="cpp coordinate analytic")
assert_close(cpp_out.vector, py_out.vector, err_msg="parity coordinate")
assert_close(cpp_out.s, S0 + LENGTH)

zero_py, _, _ = make_steering(pyegret).transfer(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0))
zero_cpp, _, _ = make_steering(cppegret).transfer(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0))
assert_close(zero_py.vector, [0.5 * LENGTH * KICK_X, KICK_X, 0.5 * LENGTH * KICK_Y, KICK_Y], err_msg="py zero input")
assert_close(zero_cpp.vector, zero_py.vector, err_msg="parity zero input")

# Element offsets do not change the kick map
for module in (pyegret, cppegret):
    shifted, _, _ = make_steering(module, dx=1.0e-3, dy=-2.0e-3).transfer(make_coordinate(module, delta=0.0))
    assert_close(shifted.vector, expected_vector, err_msg=f"{module.__name__} offset invariance")
print("Case 3 passed: coordinate transfer parity and analytic kick map")


# %% [markdown]
# ## Case 4: tilted kick parity and analytic rotation

# %%
for tilt in (0.0, 0.3, np.pi / 2):
    _, expected_t = analytic_steering_map(LENGTH, KICK_X, KICK_Y, tilt=tilt, delta=0.0)  # zero input isolates the kick
    py_t, _, _ = make_steering(pyegret, tilt=tilt).transfer(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0))
    cpp_t, _, _ = make_steering(cppegret, tilt=tilt).transfer(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0))
    assert_close(py_t.vector, expected_t, err_msg=f"py tilt={tilt}")
    assert_close(cpp_t.vector, expected_t, err_msg=f"cpp tilt={tilt}")
print("Case 4 passed: tilted kick parity and analytic rotation")


# %% [markdown]
# ## Case 5: momentum dependence of the kick (division by 1 + delta)

# %%
for delta in (1.0e-2, -2.0e-3):
    tmat_d, kick_d = analytic_steering_map(LENGTH, KICK_X, KICK_Y, delta=delta)
    py_d, _, _ = make_steering(pyegret).transfer(make_coordinate(pyegret, vector=np.zeros(4), delta=delta))
    cpp_d, _, _ = make_steering(cppegret).transfer(make_coordinate(cppegret, vector=np.zeros(4), delta=delta))
    assert_close(py_d.vector, kick_d, err_msg=f"py delta={delta}")
    assert_close(cpp_d.vector, kick_d, err_msg=f"cpp delta={delta}")
    assert_close(cpp_d.vector, py_d.vector, err_msg=f"parity delta={delta}")
print("Case 5 passed: kick scales with 1 / (1 + delta) in both backends")


# %% [markdown]
# ## Case 6: dispersion parity and the analytic dispersion of the kick

# %%
expected_disp = analytic_steering_dispersion(LENGTH, KICK_X, KICK_Y, delta=0.0)
py_disp = make_steering(pyegret).dispersion(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0))
cpp_disp = make_steering(cppegret).dispersion(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0))
assert_close(py_disp, expected_disp, err_msg="py dispersion analytic")
assert_close(cpp_disp, expected_disp, err_msg="cpp dispersion analytic")
assert_close(cpp_disp, py_disp, err_msg="parity dispersion")

py_disp_in = pyegret.Dispersion(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4]), S0)
cpp_disp_in = cppegret.Dispersion(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4]), S0)
_, _, py_disp_out = make_steering(pyegret).transfer(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0), None, py_disp_in)
_, _, cpp_disp_out = make_steering(cppegret).transfer(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0), None, cpp_disp_in)
expected_disp_out = drift_matrix(LENGTH) @ np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4]) + expected_disp
assert_close(py_disp_out.vector, expected_disp_out, err_msg="py dispersion transfer analytic")
assert_close(cpp_disp_out.vector, py_disp_out.vector, err_msg="parity dispersion transfer")
print("Case 6 passed: dispersion parity and analytic dispersion of the kick")


# %% [markdown]
# ## Case 7: envelope transfer parity and the analytic covariance map

# %%
expected_cov = drift_matrix(LENGTH) @ COV @ drift_matrix(LENGTH).T
py_env = pyegret.Envelope(cov=COV.copy(), s=S0)
cpp_env = cppegret.Envelope(cov=COV.copy(), s=S0)
_, py_env_out, _ = make_steering(pyegret).transfer(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0), py_env)
_, cpp_env_out, _ = make_steering(cppegret).transfer(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0), cpp_env)
assert_close(py_env_out.cov, expected_cov, atol=1.0e-10, err_msg="py envelope analytic")
assert_close(cpp_env_out.cov, expected_cov, atol=1.0e-10, err_msg="cpp envelope analytic")
assert_close(cpp_env_out.cov, py_env_out.cov, atol=1.0e-10, err_msg="parity envelope")
print("Case 7 passed: envelope transfer parity and analytic covariance map")


# %% [markdown]
# ## Case 8: coordinate array transfer parity along the element

# %%
py_arr, _, _ = make_steering(pyegret).transfer_array(make_coordinate(pyegret, delta=0.0), ds=0.25, endpoint=True)
cpp_arr, _, _ = make_steering(cppegret).transfer_array(make_coordinate(cppegret, delta=0.0), ds=0.25, endpoint=True)
assert len(py_arr) == len(cpp_arr)
assert_close(py_arr.s, cpp_arr.s, err_msg="array s")

# The kick is distributed uniformly along the magnet: x' = x'_0 + kick * s / L and x = x_0 + x'_0 s + kick s^2 / (2 L).
# Check each backend against this map before comparing them.
s_rel = np.asarray(py_arr.s) - S0
expected_arr_x = COORD_VECTOR[0] + COORD_VECTOR[1] * s_rel + 0.5 * KICK_X * s_rel**2 / LENGTH
expected_arr_xp = COORD_VECTOR[1] + KICK_X * s_rel / LENGTH
assert_close(py_arr.x, expected_arr_x, err_msg="py array x analytic")
assert_close(cpp_arr.x, expected_arr_x, err_msg="cpp array x analytic")
assert_close(py_arr.xp, expected_arr_xp, err_msg="py array xp analytic")
assert_close(cpp_arr.xp, expected_arr_xp, err_msg="cpp array xp analytic")
assert_close(cpp_arr.vector, py_arr.vector, err_msg="array vector parity")

# The last point of the array is the element exit, which must equal the single-coordinate transfer
exit_py, _, _ = make_steering(pyegret).transfer(make_coordinate(pyegret, delta=0.0))
assert_close(np.asarray(py_arr.vector)[:, -1], exit_py.vector, err_msg="array exit equals transfer")

plot_profile_parity(np.asarray(py_arr.s), np.asarray(py_arr.x), np.asarray(cpp_arr.x),
                    "Case 8 x along the steering magnet")
print("Case 8 passed: coordinate array transfer parity and exit consistency")


# %% [markdown]
# ## Case 9: `set_steering` updates only the given components

# %%
py_set = make_steering(pyegret)
cpp_set = make_steering(cppegret)
py_set.set_steering(kick_y=5.0e-4)
cpp_set.set_steering(kick_y=5.0e-4)
assert_close(py_set.kick, (KICK_X, 5.0e-4), err_msg="py set_steering kick")
assert_close(cpp_set.kick, (KICK_X, 5.0e-4), err_msg="cpp set_steering kick")

expected_after = analytic_steering_map(LENGTH, KICK_X, 5.0e-4, delta=0.0)
py_after, _, _ = py_set.transfer(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0))
cpp_after, _, _ = cpp_set.transfer(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0))
assert_close(py_after.vector, expected_after[1], err_msg="py transfer after set_steering")
assert_close(cpp_after.vector, expected_after[1], err_msg="cpp transfer after set_steering")
print("Case 9 passed: set_steering updates only the given components")


# %% [markdown]
# ## Case 10: the `kick` property accepts a `(kick_x, kick_y)` tuple on assignment
#
# The `kick` setter in both backends takes two arguments, so `element.kick = (kx, ky)` raises `TypeError`.
# The getter returns a tuple, so assignment should accept one.

# %%
KICK_TUPLE = (4.0e-4, -3.0e-4)
failures = []
for module in (pyegret, cppegret):
    element = make_steering(module)
    try:
        element.kick = KICK_TUPLE
        assert_close(element.kick, KICK_TUPLE, err_msg="kick after tuple assignment")
    except Exception as exc:  # noqa: BLE001 - collect every backend's failure
        failures.append(f"kick tuple assignment [{module.__name__}]: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 10 passed: kick tuple assignment works in both backends")


# %% [markdown]
# ## Case 11: zero length is a pure kick

# %%
py_zero, _, _ = make_steering(pyegret, length=0.0).transfer(make_coordinate(pyegret, delta=0.0))
cpp_zero, _, _ = make_steering(cppegret, length=0.0).transfer(make_coordinate(cppegret, delta=0.0))
expected_zero = COORD_VECTOR + np.array([0.0, KICK_X, 0.0, KICK_Y])
assert_close(py_zero.vector, expected_zero, err_msg="py zero length")
assert_close(cpp_zero.vector, expected_zero, err_msg="cpp zero length")
print("Case 11 passed: zero-length steering magnet applies only the kick")


# %% [markdown]
# ## Summary

# %%
print("Steering parity cases finished.")
