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
# tests/test_envelopearray_parity.py
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
# # EnvelopeArray parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `EnvelopeArray` behavior parity between:
# - `egret.python.EnvelopeArray` (pure-Python reference)
# - `egret.cpp.EnvelopeArray` (C++ backend wrapper)
#
# Synthetic, deterministic fixtures are used (no SPring-8-II lattice).
# The covariance matrices are built from uncoupled Twiss parameters with unit determinant,
# which is the convention of the `Envelope` class.

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
N_POINTS = 5
S_GRID = np.linspace(0.0, 4.0, N_POINTS)
ATOL = 1.0e-12
PSI_ATOL = 1.0e-6

BX0 = 5.0 + 0.5 * np.cos(S_GRID)
AX0 = 0.3 * np.sin(S_GRID)
BY0 = 3.0 + 0.2 * S_GRID
AY0 = -0.1 * np.ones(N_POINTS)


def twiss_cov(bx: float, ax: float, by: float, ay: float) -> np.ndarray:
    # Uncoupled 4x4 covariance with unit determinant in each plane: [[beta, -alpha], [-alpha, gamma]].
    cov = np.zeros((4, 4))
    cov[0:2, 0:2] = np.array([[bx, -ax], [-ax, (1.0 + ax * ax) / bx]])
    cov[2:4, 2:4] = np.array([[by, -ay], [-ay, (1.0 + ay * ay) / by]])
    return cov


COV = np.stack([twiss_cov(BX0[i], AX0[i], BY0[i], AY0[i]) for i in range(N_POINTS)])
PSIX_GIVEN = np.linspace(0.0, 0.4, N_POINTS)
PSIY_GIVEN = np.linspace(0.0, -0.3, N_POINTS)


# %%
ARRAY_ATTRS = ("bx", "ax", "gx", "by", "ay", "gy", "tau", "psix", "psiy")


def array_snapshot(arr) -> dict[str, np.ndarray]:
    snap = {
        "cov": np.array(arr.cov, dtype=float).copy(),
        "s": np.array(arr.s, dtype=float).copy(),
    }
    for name in ARRAY_ATTRS:
        snap[name] = np.array(getattr(arr, name), dtype=float).copy()
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


def returned_array_is_isolated(arr, attr: str) -> bool:
    # A read-only view is acceptable; a writable view that changes the object is not.
    before = np.array(getattr(arr, attr), copy=True)
    view = getattr(arr, attr)
    try:
        view.flat[0] = before.flat[0] + 1.0
    except ValueError:
        return True
    return np.array_equal(np.asarray(getattr(arr, attr)), before)


def plot_profile_parity(s_query: np.ndarray, py_vals: np.ndarray, cpp_vals: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(s_query, py_vals, "o-", label="pyegret")
    ax.plot(s_query, cpp_vals, "x--", label="cppegret")
    ax.set_title(title)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("beta_x [m]")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity (covariance, s, and Twiss-derived arrays)

# %%
py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

py_snap = array_snapshot(py_arr)
cpp_snap = array_snapshot(cpp_arr)
assert_snapshot_equal(py_snap, cpp_snap)
np.testing.assert_allclose(py_snap["cov"], COV, atol=ATOL, rtol=0.0)
assert len(py_arr) == len(cpp_arr) == N_POINTS
display({"case1_py_bx": py_snap["bx"], "case1_cpp_bx": cpp_snap["bx"]})
print("Case 1 passed: constructor parity")


# %% [markdown]
# ## Case 2: constructor copies its inputs (defensive copy)

# %%
cov_in = COV.copy()
s_in = S_GRID.copy()
py_arr = pyegret.EnvelopeArray(cov_in, s_in, psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(cov_in, s_in, psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

cov_in[:] = 999.0
s_in[:] = 999.0

np.testing.assert_allclose(np.asarray(py_arr.cov), COV, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.cov), COV, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(py_arr.s), S_GRID, atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.s), S_GRID, atol=ATOL, rtol=0.0)
print("Case 2 passed: defensive copy of inputs")


# %% [markdown]
# ## Case 3: eigenmode parity for uncoupled optics
#
# Uncoupled optics have no x-y coupling, so the coordinate transformation `T` must be zero,
# and `U` and `V` must equal the horizontal and vertical covariance blocks.

# %%
py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

np.testing.assert_allclose(np.asarray(py_arr.T), np.zeros((N_POINTS, 2, 2)), atol=ATOL, rtol=0.0)
np.testing.assert_allclose(np.asarray(cpp_arr.T), np.asarray(py_arr.T), atol=ATOL, rtol=0.0, err_msg="T")
np.testing.assert_allclose(np.asarray(cpp_arr.U), np.asarray(py_arr.U), atol=ATOL, rtol=0.0, err_msg="U")
np.testing.assert_allclose(np.asarray(cpp_arr.V), np.asarray(py_arr.V), atol=ATOL, rtol=0.0, err_msg="V")
display({"case3_py_U0": np.asarray(py_arr.U)[0], "case3_cpp_U0": np.asarray(cpp_arr.U)[0]})
print("Case 3 passed: eigenmode parity for uncoupled optics")


# %% [markdown]
# ## Case 4: computed betatron phases `psix` and `psiy` parity
#
# The phases are integrated from the Twiss functions when they are not given.

# %%
py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy())

np.testing.assert_allclose(np.asarray(cpp_arr.psix), np.asarray(py_arr.psix), atol=PSI_ATOL, rtol=0.0, err_msg="psix")
np.testing.assert_allclose(np.asarray(cpp_arr.psiy), np.asarray(py_arr.psiy), atol=PSI_ATOL, rtol=0.0, err_msg="psiy")
print("Case 4 passed: computed betatron phases parity")


# %% [markdown]
# ## Case 5: `copy()` parity and independence of the copy

# %%
py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

py_copy = py_arr.copy()
cpp_copy = cpp_arr.copy()
assert py_copy is not py_arr
assert cpp_copy is not cpp_arr

assert_snapshot_equal(array_snapshot(py_copy), array_snapshot(cpp_copy))

py_copy.append(py_copy.copy())
cpp_copy.append(cpp_copy.copy())

assert len(py_arr) == len(cpp_arr) == N_POINTS
assert len(py_copy) == len(cpp_copy) == 2 * N_POINTS
print("Case 5 passed: copy parity and independence of the copy")


# %% [markdown]
# ## Case 6: `append()` parity

# %%
s_second = S_GRID + (S_GRID[-1] + 1.0)
cov_second = COV * 1.0
psix_second = PSIX_GIVEN + 1.0
psiy_second = PSIY_GIVEN - 1.0

py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
py_second = pyegret.EnvelopeArray(cov_second.copy(), s_second.copy(),
                                  psix=psix_second.copy(), psiy=psiy_second.copy())
cpp_second = cppegret.EnvelopeArray(cov_second.copy(), s_second.copy(),
                                    psix=psix_second.copy(), psiy=psiy_second.copy())

py_arr.append(py_second)
cpp_arr.append(cpp_second)

assert len(py_arr) == len(cpp_arr) == 2 * N_POINTS
assert_snapshot_equal(array_snapshot(py_arr), array_snapshot(cpp_arr))
np.testing.assert_allclose(np.asarray(py_arr.s), np.concatenate([S_GRID, s_second]), atol=ATOL, rtol=0.0)
print("Case 6 passed: append parity")


# %% [markdown]
# ## Case 7: `transport()` parity through a drift-like transfer matrix

# %%
DRIFT_LENGTH = 1.5
tmat = np.stack([np.eye(4) for _ in range(N_POINTS)])
tmat[:, 0, 1] = DRIFT_LENGTH
tmat[:, 2, 3] = DRIFT_LENGTH

py_e0 = pyegret.Envelope(cov=twiss_cov(BX0[0], AX0[0], BY0[0], AY0[0]), s=0.0)
cpp_e0 = cppegret.Envelope(cov=twiss_cov(BX0[0], AX0[0], BY0[0], AY0[0]), s=0.0)

py_out = pyegret.EnvelopeArray.transport(py_e0, tmat.copy(), S_GRID.copy())
cpp_out = cppegret.EnvelopeArray.transport(cpp_e0, tmat.copy(), S_GRID.copy())

# The phases are compared in Case 4; here only the covariance-derived arrays are compared.
TRANSPORT_KEYS = ("cov", "s", "bx", "ax", "gx", "by", "ay", "gy", "tau")
py_out_snap = array_snapshot(py_out)
cpp_out_snap = array_snapshot(cpp_out)
assert_snapshot_equal({k: py_out_snap[k] for k in TRANSPORT_KEYS}, {k: cpp_out_snap[k] for k in TRANSPORT_KEYS})
print("Case 7 passed: transport parity")


# %% [markdown]
# ## Case 8: `from_s` interpolation parity over the full range

# %%
S_QUERY = np.array([0.0, 0.5, 1.5, 3.0, 4.0])

py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

py_env = [py_arr.from_s(s) for s in S_QUERY]
cpp_env = [cpp_arr.from_s(s) for s in S_QUERY]

for s_val, py_e, cpp_e in zip(S_QUERY, py_env, cpp_env):
    np.testing.assert_allclose(np.asarray(py_e.cov), np.asarray(cpp_e.cov), atol=ATOL, rtol=0.0,
                               err_msg=f"s={s_val}")
    assert py_e.s == cpp_e.s == s_val

py_bx = np.array([np.asarray(e.cov)[0, 0] for e in py_env])
cpp_bx = np.array([np.asarray(e.cov)[0, 0] for e in cpp_env])
display({"case8_s": S_QUERY, "case8_py_cov00": py_bx, "case8_cpp_cov00": cpp_bx})
plot_profile_parity(S_QUERY, py_bx, cpp_bx, "Case 8 interpolated cov[0,0] parity")
print("Case 8 passed: from_s parity over the full range")


# %% [markdown]
# ## Case 9: invalid `from_s` requests raise `ValueError` in both backends

# %%
py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

for s_val in (S_GRID[0] - 0.1, S_GRID[-1] + 0.1):
    assert_raises(lambda: py_arr.from_s(s_val), ValueError)
    assert_raises(lambda: cpp_arr.from_s(s_val), ValueError)

single_py = pyegret.EnvelopeArray(COV[:1].copy(), S_GRID[:1].copy())
single_cpp = cppegret.EnvelopeArray(COV[:1].copy(), S_GRID[:1].copy())
assert_raises(lambda: single_py.from_s(S_GRID[0]), ValueError)
assert_raises(lambda: single_cpp.from_s(S_GRID[0]), ValueError)
print("Case 9 passed: invalid from_s requests raise ValueError in both backends")


# %% [markdown]
# ## Case 10: invalid constructor inputs raise `ValueError` in both backends

# %%
INVALID_ENVELOPE_INPUTS = {
    "cov count != s length": lambda m: m.EnvelopeArray(COV.copy(), S_GRID[:-1].copy()),
    "s decreasing": lambda m: m.EnvelopeArray(COV.copy(), S_GRID[::-1].copy()),
    "cov not 4x4": lambda m: m.EnvelopeArray(COV[:, :3, :3].copy(), S_GRID.copy()),
    "psix length": lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN[:-1].copy()),
    "psiy length": lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy(), psiy=PSIY_GIVEN[:-1].copy()),
}

failures = []
for name, make in INVALID_ENVELOPE_INPUTS.items():
    for module in (pyegret, cppegret):
        try:
            assert_raises(lambda: make(module), ValueError)
        except AssertionError as exc:
            failures.append(f"{name} [{module.__name__}]: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print(f"Case 10 passed: {len(INVALID_ENVELOPE_INPUTS)} invalid inputs rejected in both backends")


# %% [markdown]
# ## Case 11: the array getters do not expose internal storage

# %%
py_arr = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())
cpp_arr = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX_GIVEN.copy(), psiy=PSIY_GIVEN.copy())

failures = []
for attr in ("cov", "bx", "tau", "T", "s"):
    for module, arr in ((pyegret, py_arr), (cppegret, cpp_arr)):
        if not returned_array_is_isolated(arr, attr):
            failures.append(f"{attr} getter exposes internal storage [{module.__name__}]")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 11 passed: getters are isolated from internal storage")


# %% [markdown]
# ## Case 12: singular eigenmode systems with a consistent right-hand side
#
# The eigenmode system is singular when `Sxx == Syy`, e.g. for a round beam.
# When the system is consistent, the minimum-norm solution is returned; for uncoupled optics this is `T = 0`.

# %%
def block_cov(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    cov = np.zeros((4, 4))
    cov[0:2, 0:2] = sxx
    cov[2:4, 2:4] = syy
    cov[0:2, 2:4] = sxy
    cov[2:4, 0:2] = sxy.T
    return cov


def eigenmode_matrix(cov: np.ndarray) -> np.ndarray:
    # Coefficient matrix of the eigenmode system; it depends only on Sxx and Syy.
    sxx, syy = cov[0:2, 0:2], cov[2:4, 2:4]
    return np.array([
        [-sxx[0, 0], -sxx[0, 1] - syy[0, 1], 0.0, syy[0, 0]],
        [0.0, -syy[1, 1], -sxx[0, 0], -sxx[0, 1] + syy[0, 1]],
        [-sxx[0, 1] + syy[0, 1], -sxx[1, 1], -syy[0, 0], 0.0],
        [syy[1, 1], 0.0, -sxx[0, 1] - syy[0, 1], -sxx[1, 1]],
    ])


EYE2 = np.eye(2)
X_CONSISTENT = np.array([0.05, 0.02, -0.03, 0.04])
MAT_ROUND = eigenmode_matrix(block_cov(EYE2, EYE2, np.zeros((2, 2))))
B_CONSISTENT = MAT_ROUND @ X_CONSISTENT
COV_CONSISTENT = block_cov(EYE2, EYE2, B_CONSISTENT.reshape(2, 2))
assert np.all(np.linalg.eigvalsh(COV_CONSISTENT) > 0.0)
T_MIN_NORM = (np.linalg.pinv(MAT_ROUND) @ B_CONSISTENT).reshape(2, 2)

SINGULAR_CONSISTENT = {
    "round uncoupled (T = 0)": (block_cov(EYE2, EYE2, np.zeros((2, 2))), np.zeros((2, 2))),
    "round coupled, minimum-norm T": (COV_CONSISTENT, T_MIN_NORM),
}
for name, (cov_one, expected_T) in SINGULAR_CONSISTENT.items():
    cov_pair = np.stack([cov_one, cov_one])
    s_pair = np.array([0.0, 1.0])
    py_T = np.asarray(pyegret.EnvelopeArray(cov_pair.copy(), s_pair).T)[0]
    cpp_T = np.asarray(cppegret.EnvelopeArray(cov_pair.copy(), s_pair).T)[0]
    np.testing.assert_allclose(py_T, expected_T, atol=1.0e-10, rtol=0.0, err_msg=f"{name} [pyegret]")
    np.testing.assert_allclose(cpp_T, expected_T, atol=1.0e-10, rtol=0.0, err_msg=f"{name} [cppegret]")
print(f"Case 12 passed: {len(SINGULAR_CONSISTENT)} singular but consistent systems in both backends")


# %% [markdown]
# ## Case 13: singular eigenmode system with no solution raises `ValueError` in both backends

# %%
COV_INCONSISTENT = block_cov(EYE2, EYE2, np.diag([0.3, -0.1]))
pair_inconsistent = np.stack([COV_INCONSISTENT, COV_INCONSISTENT])
s_pair = np.array([0.0, 1.0])
assert_raises(lambda: pyegret.EnvelopeArray(pair_inconsistent.copy(), s_pair), ValueError)
assert_raises(lambda: cppegret.EnvelopeArray(pair_inconsistent.copy(), s_pair), ValueError)
print("Case 13 passed: inconsistent singular system rejected in both backends")


# %% [markdown]
# ## Case 14: `calc_eigenmode` rejects a `T` with the wrong shape or length in both backends

# %%
INVALID_T_USES = {
    "T shape (N, 3, 3)": lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy()).calc_eigenmode(
        T=np.zeros((N_POINTS, 3, 3))
    ),
    "T length (N - 1)": lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy()).calc_eigenmode(
        T=np.zeros((N_POINTS - 1, 2, 2))
    ),
    "T not 3-D": lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy()).calc_eigenmode(T=np.zeros((2, 2))),
}

failures = []
for name, use in INVALID_T_USES.items():
    for module in (pyegret, cppegret):
        try:
            assert_raises(lambda: use(module), ValueError)
        except AssertionError as exc:
            failures.append(f"{name} [{module.__name__}]: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print(f"Case 14 passed: {len(INVALID_T_USES)} invalid T arguments rejected in both backends")


# %% [markdown]
# ## Summary

# %%
print("EnvelopeArray parity cases finished.")
