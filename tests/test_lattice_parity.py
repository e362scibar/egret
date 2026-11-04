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
# tests/test_lattice_parity.py
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
# # Lattice and Ring parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Lattice` and `Ring` behavior parity between:
# - `egret.python.Lattice` / `egret.python.Ring` (pure-Python reference)
# - `egret.cpp.Lattice` / `egret.cpp.Ring` (C++ backend wrapper)
#
# The lattice is a synthetic double-bend cell: two 1 m bends (each bending `pi/8`) between a horizontally
# focusing centre quadrupole and two horizontally defocusing outer quadrupoles, with drift spaces. It is **not** an achromat (the periodic dispersion is nonzero),
# so the checks are the transfer composition, the tune-trace relation, and the periodicity of the dispersion.
# No SPring-8-II lattice is used.

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
BEND_ANGLE = np.pi / 8
K_OUTER = -1.8  # outer quadrupoles: horizontally defocusing
K_CENTER = 2.1  # centre quadrupole: horizontally focusing
L_BEND_GAP = 0.1
L_QUAD = 0.3
L_DRIFT_OUTER = 0.4
L_DRIFT_INNER = 0.3
ENERGY = 1.0
ATOL = 1.0e-12
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])


def cell_elements(module) -> list:
    """Synthetic double-bend cell: QO1 - drift - B - drift - QC - drift - B - drift - QO2."""
    return [
        module.Drift("d0", L_DRIFT_OUTER),
        module.Quadrupole("QO1", L_QUAD, K_OUTER),
        module.Drift("d1", L_DRIFT_INNER),
        module.Dipole("B1", 1.0, BEND_ANGLE),
        module.Drift("d2", L_BEND_GAP),
        module.Quadrupole("QC", L_QUAD, K_CENTER),
        module.Drift("d3", L_BEND_GAP),
        module.Dipole("B2", 1.0, BEND_ANGLE),
        module.Drift("d4", L_DRIFT_INNER),
        module.Quadrupole("QO2", L_QUAD, K_OUTER),
        module.Drift("d5", L_DRIFT_OUTER),
    ]


def zero_coordinate(module):
    return module.Coordinate(np.zeros(4), 0.0, 0.0, 0.0)


def tune_from_trace(tmat2x2: np.ndarray) -> float:
    return float(np.arccos(np.trace(tmat2x2) / 2.0) / (2.0 * np.pi))


def assert_close(left, right, atol: float = ATOL, err_msg: str = "") -> None:
    np.testing.assert_allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float),
                               atol=atol, rtol=0.0, err_msg=err_msg)


# %% [markdown]
# ## Case 1: element sequence, length and angle of the lattice, and `copy()`

# %%
py_lat = pyegret.Lattice("cell", cell_elements(pyegret))
cpp_lat = cppegret.Lattice("cell", cell_elements(cppegret))
# Sum of the element lengths: 0.4 + 0.3 + 0.3 + 1.0 + 0.1 + 0.3 + 0.1 + 1.0 + 0.3 + 0.3 + 0.4 = 4.5 m.
# Two bends of BEND_ANGLE each.
for name, expected in {"length": 4.5, "angle": 2.0 * BEND_ANGLE}.items():
    assert_close(getattr(py_lat, name), expected, err_msg=f"py {name}")
    assert_close(getattr(cpp_lat, name), expected, err_msg=f"cpp {name}")
display({"elements": [e.name for e in cell_elements(pyegret)]})

py_copy = py_lat.copy()
cpp_copy = cpp_lat.copy()
assert_close(py_copy.length, py_lat.length)
assert_close(cpp_copy.length, cpp_lat.length)
print("Case 1 passed: length, bending angle, and copy in both backends")


# %% [markdown]
# ## Case 2: the lattice transfer matrix is the ordered product of the element matrices

# %%
def product_of_elements(module) -> np.ndarray:
    tmat = np.eye(4)
    for elem in cell_elements(module):
        tmat = elem.transfer_matrix(zero_coordinate(module), 0.05) @ tmat
    return tmat


py_tmat = np.asarray(py_lat.transfer_matrix(zero_coordinate(pyegret), 0.05))
cpp_tmat = np.asarray(cpp_lat.transfer_matrix(zero_coordinate(cppegret), 0.05))
assert_close(py_tmat, product_of_elements(pyegret), err_msg="py composition")
assert_close(cpp_tmat, product_of_elements(cppegret), err_msg="cpp composition")
assert_close(cpp_tmat, py_tmat, err_msg="parity transfer matrix")
display({"case2_py_R12x": py_tmat[0, 1], "case2_cpp_R12x": cpp_tmat[0, 1]})
print("Case 2 passed: lattice matrix equals the ordered product of element matrices, in both backends")


# %% [markdown]
# ## Case 3: nested sub-lattices compose to the flat lattice

# %%
def nested(module) -> np.ndarray:
    elements = cell_elements(module)
    first = module.Lattice("first", elements[:6])
    second = module.Lattice("second", elements[6:])
    return np.asarray(module.Lattice("cell", [first, second]).transfer_matrix(zero_coordinate(module), 0.05))


assert_close(nested(pyegret), py_tmat, err_msg="py nested composition")
assert_close(nested(cppegret), cpp_tmat, err_msg="cpp nested composition")
print("Case 3 passed: nested sub-lattices give the same matrix as the flat lattice")


# %% [markdown]
# ## Case 4: the array transfer ends at the single-coordinate transfer

# %%
py_arr, _, _ = py_lat.transfer_array(pyegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), ds=0.1, endpoint=True)
cpp_arr, _, _ = cpp_lat.transfer_array(cppegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), ds=0.1, endpoint=True)
py_single, _, _ = py_lat.transfer(pyegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0))
cpp_single, _, _ = cpp_lat.transfer(cppegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0))
assert_close(np.asarray(py_arr.vector)[:, -1], py_single.vector, err_msg="py array end equals transfer")
assert_close(np.asarray(cpp_arr.vector)[:, -1], cpp_single.vector, err_msg="cpp array end equals transfer")
assert_close(cpp_arr.vector, py_arr.vector, err_msg="parity array")

plot_s = np.asarray(py_arr.s, dtype=float)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(plot_s, np.asarray(py_arr.x), label="x (pyegret)")
ax.plot(plot_s, np.asarray(cpp_arr.x), "x", label="x (cppegret)")
ax.set_xlabel("s [m]")
ax.set_ylabel("x [m]")
ax.set_title("Orbit through the synthetic double-bend cell")
ax.grid(alpha=0.3)
ax.legend()
print("Case 4 passed: array end equals the single transfer, and the backends agree along the cell")


# %% [markdown]
# ## Case 5: ring tunes equal the tunes from the trace of the one-turn matrix

# %%
py_ring = pyegret.Ring("R", cell_elements(pyegret), ENERGY)
cpp_ring = cppegret.Ring("R", cell_elements(cppegret), ENERGY)
assert py_ring.update()
assert cpp_ring.update()
trace_tune_x = tune_from_trace(py_tmat[0:2, 0:2])
trace_tune_y = tune_from_trace(py_tmat[2:4, 2:4])
assert_close([py_ring.tune_x, py_ring.tune_y], [trace_tune_x, trace_tune_y], atol=1.0e-9, err_msg="py tunes")
assert_close([cpp_ring.tune_x, cpp_ring.tune_y], [trace_tune_x, trace_tune_y], atol=1.0e-9, err_msg="cpp tunes")
display({"tune_x": cpp_ring.tune_x, "tune_y": cpp_ring.tune_y})
print("Case 5 passed: ring tunes equal the traces of the one-turn matrix in both backends")


# %% [markdown]
# ## Case 6: the periodic dispersion reproduces itself after one turn; the closed orbit is zero

# %%
for label, ring, module, lat in (("py", py_ring, pyegret, py_lat), ("cpp", cpp_ring, cppegret, cpp_lat)):
    disp0 = module.Dispersion(np.asarray(ring.disp0.vector, dtype=float).copy(), 0.0)
    out = lat.transfer(module.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), None, disp0, 0.05)[2]
    assert_close(out.vector, disp0.vector, atol=1.0e-12, err_msg=f"{label} dispersion periodicity")
    assert_close(np.asarray(ring.cood0.vector), np.zeros(4), atol=1.0e-12, err_msg=f"{label} closed orbit")
assert_close(cpp_ring.disp0.vector, py_ring.disp0.vector, atol=1.0e-9, err_msg="parity periodic dispersion")
print("Case 6 passed: periodic dispersion is reproduced after one turn; closed orbit is zero; backends agree")


# %% [markdown]
# ## Case 7: beta functions and dispersion along the cell
#
# The periodic envelope and dispersion of the ring are propagated through the cell. The optics must close
# after one turn, and both backends must agree at every point along the cell.

# %%
def optics_along_cell(module, ring):
    lattice = module.Lattice("cell", cell_elements(module))
    disp0 = module.Dispersion(np.asarray(ring.disp0.vector, dtype=float).copy(), 0.0)
    _, evlp, disp = lattice.transfer_array(module.Coordinate(np.zeros(4), 0.0, 0.0, 0.0),
                                           ring.evlp0, disp0, 0.05, True)
    return {
        "s": np.asarray(evlp.s, dtype=float),
        "bx": np.asarray(evlp.bx, dtype=float),
        "by": np.asarray(evlp.by, dtype=float),
        "eta_x": np.asarray(disp.x, dtype=float),
    }


py_optics = optics_along_cell(pyegret, py_ring)
cpp_optics = optics_along_cell(cppegret, cpp_ring)
for key in ("bx", "by", "eta_x"):
    assert_close(py_optics[key][-1], py_optics[key][0], atol=1.0e-9, err_msg=f"py {key} periodicity")
    assert_close(cpp_optics[key][-1], cpp_optics[key][0], atol=1.0e-9, err_msg=f"cpp {key} periodicity")
    assert_close(cpp_optics[key], py_optics[key], atol=1.0e-9, err_msg=f"parity {key}")
display({"case7_bx_start_end": (py_optics["bx"][0], py_optics["bx"][-1]),
         "case7_eta_start_end": (py_optics["eta_x"][0], py_optics["eta_x"][-1])})

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, key, label in zip(axes, ("bx", "by", "eta_x"), ("beta_x [m]", "beta_y [m]", "eta_x [m]")):
    ax.plot(py_optics["s"], py_optics[key], "-", label="pyegret")
    ax.plot(cpp_optics["s"], cpp_optics[key], "x", label="cppegret")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("s [m]")
axes[0].set_title("Periodic optics of the synthetic double-bend cell")
print("Case 7 passed: optics close after one turn and agree between backends")


# %% [markdown]
# ## Summary

# %%
print("Lattice and Ring parity cases finished.")
