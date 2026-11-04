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
# tests/test_coordinate_parity.py
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
# # Coordinate parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Coordinate` behavior parity between:
# - `egret.python.Coordinate` (pure-Python reference)
# - `egret.cpp.Coordinate` (C++ backend wrapper)
#
# Setter/getter-focused tests are intentionally excluded.

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
def snapshot(coord) -> dict[str, np.ndarray | float]:
    return {
        "vector": np.array(coord.vector, dtype=float).copy(),
        "s": float(coord.s),
        "z": float(coord.z),
        "delta": float(coord.delta),
    }


def assert_snapshot_equal(left: dict, right: dict, atol: float = 1.0e-14) -> None:
    np.testing.assert_allclose(left["vector"], right["vector"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["s"], right["s"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["z"], right["z"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["delta"], right["delta"], atol=atol, rtol=0.0)


def plot_vector_parity(py_vector: np.ndarray, cpp_vector: np.ndarray, title: str) -> None:
    labels = ["x", "xp", "y", "yp"]
    indices = np.arange(len(labels), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(indices - width / 2.0, py_vector, width=width, label="pyegret")
    ax.bar(indices + width / 2.0, cpp_vector, width=width, label="cppegret")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: default constructor parity

# %%
py_default = pyegret.Coordinate()
cpp_default = cppegret.Coordinate()
assert_snapshot_equal(snapshot(py_default), snapshot(cpp_default))
display({"case1_py_snapshot": snapshot(py_default), "case1_cpp_snapshot": snapshot(cpp_default)})
plot_vector_parity(snapshot(py_default)["vector"], snapshot(cpp_default)["vector"], "Case 1 vector parity")
print("Case 1 passed: default constructor parity")


# %% [markdown]
# ## Case 2: constructor parity + defensive copy from input vector

# %%
input_vector_py = np.array([1.0e-3, -2.0e-4, 3.0e-3, -4.0e-4], dtype=float)
input_vector_cpp = input_vector_py.copy()

py_coord = pyegret.Coordinate(input_vector_py, s=12.5, z=-0.03, delta=2.5e-4)
cpp_coord = cppegret.Coordinate(input_vector_cpp, s=12.5, z=-0.03, delta=2.5e-4)

input_vector_py[0] = 999.0
input_vector_cpp[0] = 999.0

assert_snapshot_equal(snapshot(py_coord), snapshot(cpp_coord))
assert not np.isclose(py_coord.vector[0], 999.0)
assert not np.isclose(cpp_coord.vector[0], 999.0)
display({"case2_py_snapshot": snapshot(py_coord), "case2_cpp_snapshot": snapshot(cpp_coord)})
plot_vector_parity(snapshot(py_coord)["vector"], snapshot(cpp_coord)["vector"], "Case 2 vector parity")
print("Case 2 passed: constructor parity and defensive copy")


# %% [markdown]
# ## Case 3: `copy()` parity + distinct instance

# %%
py_copy = py_coord.copy()
cpp_copy = cpp_coord.copy()

assert_snapshot_equal(snapshot(py_copy), snapshot(cpp_copy))
assert py_copy is not py_coord
assert cpp_copy is not cpp_coord
display({"case3_py_copy_snapshot": snapshot(py_copy), "case3_cpp_copy_snapshot": snapshot(cpp_copy)})
plot_vector_parity(snapshot(py_copy)["vector"], snapshot(cpp_copy)["vector"], "Case 3 copy vector parity")
print("Case 3 passed: copy parity and distinct instance")


# %% [markdown]
# All `Coordinate` parity checks passed.
