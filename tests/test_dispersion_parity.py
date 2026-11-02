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
# # Dispersion parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Dispersion` behavior parity between:
# - `egret.python.Dispersion` (pure-Python reference)
# - `egret.cpp.Dispersion` (C++ backend wrapper)
#
# Trivial setter/getter-only tests are intentionally excluded.

# %%
from __future__ import annotations

import sys

import matplotlib

# Plain `python` runs must not open plot windows; Jupyter (IPython loaded) keeps its inline backend.
if "IPython" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import egret.cpp as cppegret
import egret.python as pyegret

try:
    from IPython.display import display
except Exception:  # pragma: no cover - fallback for non-IPython execution
    def display(obj):
        print(obj)


# %%
def snapshot(disp) -> dict[str, np.ndarray | float]:
    return {
        "vector": np.array(disp.vector, dtype=float).copy(),
        "s": float(disp.s),
    }


def assert_snapshot_equal(left: dict, right: dict, atol: float = 1.0e-14) -> None:
    np.testing.assert_allclose(left["vector"], right["vector"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["s"], right["s"], atol=atol, rtol=0.0)


def plot_vector_parity(py_vector: np.ndarray, cpp_vector: np.ndarray, title: str) -> None:
    labels = ["eta_x", "eta_xp", "eta_y", "eta_yp"]
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
py_default = pyegret.Dispersion()
cpp_default = cppegret.Dispersion()

assert_snapshot_equal(snapshot(py_default), snapshot(cpp_default))
display({"case1_py_snapshot": snapshot(py_default), "case1_cpp_snapshot": snapshot(cpp_default)})
plot_vector_parity(snapshot(py_default)["vector"], snapshot(cpp_default)["vector"], "Case 1 vector parity")
print("Case 1 passed: default constructor parity")


# %% [markdown]
# ## Case 2: constructor parity + defensive copy from input vector

# %%
input_vector_py = np.array([2.0e-3, -1.0e-4, 7.0e-5, -3.0e-5], dtype=float)
input_vector_cpp = input_vector_py.copy()

py_disp = pyegret.Dispersion(input_vector_py, s=42.0)
cpp_disp = cppegret.Dispersion(input_vector_cpp, s=42.0)

input_vector_py[0] = 999.0
input_vector_cpp[0] = 999.0

assert_snapshot_equal(snapshot(py_disp), snapshot(cpp_disp))
assert not np.isclose(py_disp.vector[0], 999.0)
assert not np.isclose(cpp_disp.vector[0], 999.0)
display({"case2_py_snapshot": snapshot(py_disp), "case2_cpp_snapshot": snapshot(cpp_disp)})
plot_vector_parity(snapshot(py_disp)["vector"], snapshot(cpp_disp)["vector"], "Case 2 vector parity")
print("Case 2 passed: constructor parity and defensive copy")


# %% [markdown]
# ## Case 3: `copy()` parity + distinct instance

# %%
py_copy = py_disp.copy()
cpp_copy = cpp_disp.copy()

assert_snapshot_equal(snapshot(py_copy), snapshot(cpp_copy))
assert py_copy is not py_disp
assert cpp_copy is not cpp_disp
display({"case3_py_copy_snapshot": snapshot(py_copy), "case3_cpp_copy_snapshot": snapshot(cpp_copy)})
plot_vector_parity(snapshot(py_copy)["vector"], snapshot(cpp_copy)["vector"], "Case 3 copy vector parity")
print("Case 3 passed: copy parity and distinct instance")


# %% [markdown]
# All `Dispersion` parity checks passed.
