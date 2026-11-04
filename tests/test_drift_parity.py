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
# tests/test_drift_parity.py
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
# # Drift parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Drift` behavior parity between:
# - `egret.python.Drift` (pure-Python reference)
# - `egret.cpp.Drift` (C++ backend wrapper)
#
# Each result is also checked against the analytic drift map, `x -> x + L x'`, `s -> s + L`,
# and `cov -> M cov M^T`. Synthetic fixtures are used (no SPring-8-II lattice).

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
LENGTH = 1.5
ATOL = 1.0e-12
METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")
COORD_VECTOR = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])  # x, x', y, y'
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
# Uncoupled Twiss covariance with unit determinant in each plane: [[beta, -alpha], [-alpha, gamma]]
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])
DISP_VECTOR = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])  # eta_x, eta_x', eta_y, eta_y'


def analytic_drift_matrix(length: float) -> np.ndarray:
    tmat = np.eye(4)
    tmat[0, 1] = length
    tmat[2, 3] = length
    return tmat


def make_coordinate(module, vector: np.ndarray = COORD_VECTOR, s: float = S0):
    return module.Coordinate(vector.copy(), s, Z0, DELTA0)


def make_drift(module, **kwargs):
    return module.Drift("D", LENGTH, **kwargs)


def assert_close(left, right, atol: float = ATOL, err_msg: str = "") -> None:
    np.testing.assert_allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float),
                               atol=atol, rtol=0.0, err_msg=err_msg)


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
py_drift = make_drift(pyegret, dx=1.0e-3, dy=-2.0e-3, ds=0.1, tilt=0.2, info="straight")
cpp_drift = make_drift(cppegret, dx=1.0e-3, dy=-2.0e-3, ds=0.1, tilt=0.2, info="straight")

attrs = {"length": LENGTH, "dx": 1.0e-3, "dy": -2.0e-3, "ds": 0.1, "tilt": 0.2}
for name, expected in attrs.items():
    assert_close(getattr(py_drift, name), expected, err_msg=f"py {name}")
    assert_close(getattr(cpp_drift, name), expected, err_msg=f"cpp {name}")
assert py_drift.info == cpp_drift.info == "straight"
assert py_drift.name == cpp_drift.name == "D"

py_copy = py_drift.copy()
cpp_copy = cpp_drift.copy()
assert py_copy is not py_drift and cpp_copy is not cpp_drift
assert_close(py_copy.length, LENGTH)
assert_close(cpp_copy.length, LENGTH)

py_copy.length = 3.0
cpp_copy.length = 3.0
assert_close(py_drift.length, LENGTH, err_msg="py original changed by copy")
assert_close(cpp_drift.length, LENGTH, err_msg="cpp original changed by copy")
assert_close(py_copy.length, 3.0)
assert_close(cpp_copy.length, 3.0)
print("Case 1 passed: constructor parity, attributes, and copy independence")


# %% [markdown]
# ## Case 2: transfer matrix parity, analytic form, and independence from tilt, offsets, and method

# %%
expected = analytic_drift_matrix(LENGTH)
py_tmat = make_drift(pyegret).transfer_matrix()
cpp_tmat = make_drift(cppegret).transfer_matrix()
assert_close(py_tmat, expected, err_msg="py transfer_matrix")
assert_close(cpp_tmat, expected, err_msg="cpp transfer_matrix")
assert_close(cpp_tmat, py_tmat, err_msg="parity transfer_matrix")

for module in (pyegret, cppegret):
    shifted = make_drift(module, dx=1.0e-3, dy=-2.0e-3, tilt=0.4).transfer_matrix()
    assert_close(shifted, expected, err_msg=f"{module.__name__} offsets/tilt change matrix")
    for method in METHODS:
        assert_close(make_drift(module).transfer_matrix(make_coordinate(module), 0.1, method), expected,
                     err_msg=f"{module.__name__} method {method}")
print("Case 2 passed: transfer matrix parity; tilt, offsets, and method do not change it")


# %% [markdown]
# ## Case 3: transfer matrix array parity along the element

# %%
# The array is indexed N-first: shape (N, 4, 4), as used by EnvelopeArray.transport.
# The Python docstring says (4, 4, N); this is a documentation mismatch.
# A step of 0.4 with the endpoint gives five points, so N is distinguishable from 4 in both layouts.
for endpoint in (True, False):
    py_arr, py_s = make_drift(pyegret).transfer_matrix_array(ds=0.4, endpoint=endpoint)
    cpp_arr, cpp_s = make_drift(cppegret).transfer_matrix_array(ds=0.4, endpoint=endpoint)
    assert np.asarray(py_arr).shape == np.asarray(cpp_arr).shape == (len(py_s), 4, 4)
    assert_close(py_s, cpp_s, err_msg=f"s endpoint={endpoint}")
    assert_close(py_arr, cpp_arr, err_msg=f"tmat array endpoint={endpoint}")
    for k, s_val in enumerate(py_s):
        assert_close(np.asarray(py_arr)[k], analytic_drift_matrix(s_val), err_msg=f"analytic at s={s_val}")
print("Case 3 passed: transfer matrix array parity and analytic form")


# %% [markdown]
# ## Case 4: coordinate transfer parity, analytic map, and offset invariance

# %%
expected_vector = np.array([
    COORD_VECTOR[0] + LENGTH * COORD_VECTOR[1],
    COORD_VECTOR[1],
    COORD_VECTOR[2] + LENGTH * COORD_VECTOR[3],
    COORD_VECTOR[3],
])

py_out, _, _ = make_drift(pyegret).transfer(make_coordinate(pyegret))
cpp_out, _, _ = make_drift(cppegret).transfer(make_coordinate(cppegret))

assert_close(py_out.vector, expected_vector, err_msg="py vector")
assert_close(cpp_out.vector, expected_vector, err_msg="cpp vector")
assert_close(cpp_out.vector, py_out.vector, err_msg="parity vector")
assert_close(py_out.s, S0 + LENGTH)
assert_close(cpp_out.s, S0 + LENGTH)
assert_close(py_out.z, Z0)
assert_close(cpp_out.z, Z0)
assert_close(py_out.delta, DELTA0)
assert_close(cpp_out.delta, DELTA0)

for module in (pyegret, cppegret):
    with_offset, _, _ = make_drift(module, dx=1.0e-3, dy=-2.0e-3, tilt=0.4).transfer(make_coordinate(module))
    without_offset, _, _ = make_drift(module).transfer(make_coordinate(module))
    assert_close(with_offset.vector, without_offset.vector, err_msg=f"{module.__name__} offset invariance")
print("Case 4 passed: coordinate transfer parity, analytic map, and offset invariance")


# %% [markdown]
# ## Case 5: envelope transfer parity and the analytic covariance map

# %%
expected_cov = analytic_drift_matrix(LENGTH) @ COV @ analytic_drift_matrix(LENGTH).T

py_env = pyegret.Envelope(cov=COV.copy(), s=S0)
cpp_env = cppegret.Envelope(cov=COV.copy(), s=S0)
_, py_out_env, _ = make_drift(pyegret).transfer(make_coordinate(pyegret), py_env)
_, cpp_out_env, _ = make_drift(cppegret).transfer(make_coordinate(cppegret), cpp_env)

assert_close(py_out_env.cov, expected_cov, err_msg="py envelope cov")
assert_close(cpp_out_env.cov, expected_cov, err_msg="cpp envelope cov")
assert_close(cpp_out_env.cov, py_out_env.cov, err_msg="parity envelope cov")
assert_close(py_out_env.s, S0 + LENGTH)
assert_close(cpp_out_env.s, S0 + LENGTH)
print("Case 5 passed: envelope transfer parity and analytic covariance map")


# %% [markdown]
# ## Case 6: dispersion transfer parity and analytic map

# %%
expected_disp = np.array([
    DISP_VECTOR[0] + LENGTH * DISP_VECTOR[1],
    DISP_VECTOR[1],
    DISP_VECTOR[2] + LENGTH * DISP_VECTOR[3],
    DISP_VECTOR[3],
])
py_disp = pyegret.Dispersion(DISP_VECTOR.copy(), S0)
cpp_disp = cppegret.Dispersion(DISP_VECTOR.copy(), S0)
_, _, py_out_disp = make_drift(pyegret).transfer(make_coordinate(pyegret), None, py_disp)
_, _, cpp_out_disp = make_drift(cppegret).transfer(make_coordinate(cppegret), None, cpp_disp)

assert_close(py_out_disp.vector, expected_disp, err_msg="py dispersion")
assert_close(cpp_out_disp.vector, expected_disp, err_msg="cpp dispersion")
assert_close(cpp_out_disp.vector, py_out_disp.vector, err_msg="parity dispersion")
assert_close(cpp_out_disp.s, S0 + LENGTH)
print("Case 6 passed: dispersion transfer parity and analytic map")


# %% [markdown]
# ## Case 7: coordinate array transfer parity along the element

# %%
py_arr, _, _ = make_drift(pyegret).transfer_array(make_coordinate(pyegret), ds=0.5, endpoint=True)
cpp_arr, _, _ = make_drift(cppegret).transfer_array(make_coordinate(cppegret), ds=0.5, endpoint=True)

assert len(py_arr) == len(cpp_arr)
assert_close(py_arr.s, cpp_arr.s, err_msg="array s")
assert_close(py_arr.vector, cpp_arr.vector, err_msg="array vector")
expected_x = COORD_VECTOR[0] + (np.asarray(py_arr.s) - S0) * COORD_VECTOR[1]
assert_close(py_arr.x, expected_x, err_msg="array x analytic")

plot_profile_parity(np.asarray(py_arr.s), np.asarray(py_arr.x), np.asarray(cpp_arr.x),
                    "Case 7 x along the drift")
print("Case 7 passed: coordinate array transfer parity and analytic x(s)")


# %% [markdown]
# ## Case 8: transfer starting inside the drift uses the remaining length

# %%
S_INSIDE = 0.5
remaining = LENGTH - S_INSIDE
expected_mid = analytic_drift_matrix(remaining)

py_mid_tmat = make_drift(pyegret).transfer_matrix_from_s(S_INSIDE, make_coordinate(pyegret, s=0.0))
cpp_mid_tmat = make_drift(cppegret).transfer_matrix_from_s(S_INSIDE, make_coordinate(cppegret, s=0.0))
assert_close(py_mid_tmat, expected_mid, err_msg="py transfer_matrix_from_s")
assert_close(cpp_mid_tmat, expected_mid, err_msg="cpp transfer_matrix_from_s")

py_mid, _, _ = make_drift(pyegret).transfer_from_s(S_INSIDE, make_coordinate(pyegret, s=0.0))
cpp_mid, _, _ = make_drift(cppegret).transfer_from_s(S_INSIDE, make_coordinate(cppegret, s=0.0))
assert_close(py_mid.vector, cpp_mid.vector, err_msg="transfer_from_s parity")
assert_close(py_mid.vector, expected_mid @ COORD_VECTOR, err_msg="transfer_from_s analytic")
print("Case 8 passed: transfer from a point inside the drift")


# %% [markdown]
# ## Case 9: a zero-length drift is the identity map

# %%
py_zero, _, _ = pyegret.Drift("Z", 0.0).transfer(make_coordinate(pyegret))
cpp_zero, _, _ = cppegret.Drift("Z", 0.0).transfer(make_coordinate(cppegret))
assert_close(py_zero.vector, COORD_VECTOR, err_msg="py zero length")
assert_close(cpp_zero.vector, COORD_VECTOR, err_msg="cpp zero length")
assert_close(cpp_zero.s, S0)
print("Case 9 passed: zero-length drift is the identity")


# %% [markdown]
# ## Summary

# %%
print("Drift parity cases finished.")
