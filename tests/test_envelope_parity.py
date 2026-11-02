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
# # Envelope parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Envelope` behavior parity between:
# - `egret.python.Envelope` (pure-Python reference)
# - `egret.cpp.Envelope` (C++ backend wrapper)
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
def snapshot(env) -> dict[str, np.ndarray | float]:
    return {
        "cov": np.array(env.cov, dtype=float).copy(),
        "T": np.array(env.T, dtype=float).copy(),
        "U": np.array(env.U, dtype=float).copy(),
        "V": np.array(env.V, dtype=float).copy(),
        "tau": float(env.tau),
        "s": float(env.s),
        "psix": float(env.psix),
        "psiy": float(env.psiy),
        "twiss": np.array(
            [env.bx, env.ax, env.gx, env.by, env.ay, env.gy, env.bu, env.au, env.gu, env.bv, env.av, env.gv],
            dtype=float,
        ),
    }


def assert_snapshot_equal(left: dict, right: dict, atol: float = 1.0e-12) -> None:
    np.testing.assert_allclose(left["cov"], right["cov"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["tau"], right["tau"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["s"], right["s"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["psix"], right["psix"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["psiy"], right["psiy"], atol=atol, rtol=0.0)


def assert_eigenmode_snapshot_equal(left: dict, right: dict, atol: float = 1.0e-12) -> None:
    np.testing.assert_allclose(left["T"], right["T"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["U"], right["U"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["V"], right["V"], atol=atol, rtol=0.0)
    np.testing.assert_allclose(left["twiss"], right["twiss"], atol=atol, rtol=0.0)


def plot_cov_diagonal_parity(py_cov: np.ndarray, cpp_cov: np.ndarray, title: str) -> None:
    labels = ["xx", "xpxp", "yy", "ypyp"]
    py_diag = np.array([py_cov[0, 0], py_cov[1, 1], py_cov[2, 2], py_cov[3, 3]], dtype=float)
    cpp_diag = np.array([cpp_cov[0, 0], cpp_cov[1, 1], cpp_cov[2, 2], cpp_cov[3, 3]], dtype=float)
    idx = np.arange(len(labels), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(idx - width / 2.0, py_diag, width=width, label="pyegret")
    ax.bar(idx + width / 2.0, cpp_diag, width=width, label="cppegret")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cov diagonal")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def get_t_matrix(env) -> np.ndarray:
    t_matrix_attr = getattr(env, "T_matrix")
    if callable(t_matrix_attr):
        try:
            return np.array(t_matrix_attr(), dtype=float)
        except TypeError:
            pass
    elif t_matrix_attr is not None:
        return np.array(t_matrix_attr, dtype=float)

    instance = getattr(env, "instance", None)
    if instance is None:
        raise RuntimeError("Failed to resolve T_matrix from envelope object.")

    instance_t_matrix_attr = getattr(instance, "T_matrix", None)
    if callable(instance_t_matrix_attr):
        return np.array(instance_t_matrix_attr(), dtype=float)
    if instance_t_matrix_attr is not None:
        return np.array(instance_t_matrix_attr, dtype=float)

    raise RuntimeError("Failed to resolve T_matrix from envelope instance.")


# %% [markdown]
# ## Case 1: default constructor parity

# %%
py_default = pyegret.Envelope()
cpp_default = cppegret.Envelope()

snap_py_default = snapshot(py_default)
snap_cpp_default = snapshot(cpp_default)
assert_snapshot_equal(snap_py_default, snap_cpp_default)
display({"case1_tau_py_cpp": (snap_py_default["tau"], snap_cpp_default["tau"]), "case1_s_py_cpp": (snap_py_default["s"], snap_cpp_default["s"])})
plot_cov_diagonal_parity(snap_py_default["cov"], snap_cpp_default["cov"], "Case 1 covariance diagonal parity")
print("Case 1 passed: default constructor parity")


# %% [markdown]
# ## Case 2: constructor parity with custom covariance + provided T

# %%
cov_in = np.array(
    [
        [5.0, -1.2, 0.35, 0.08],
        [-1.2, 0.68, -0.11, 0.05],
        [0.35, -0.11, 4.3, 0.9],
        [0.08, 0.05, 0.9, 0.55],
    ],
    dtype=float,
)
T_in = np.array([[0.02, -0.01], [0.015, 0.018]], dtype=float)

py_env = pyegret.Envelope(cov=cov_in.copy(), s=10.0, T=T_in.copy(), psix=0.2, psiy=-0.3)
cpp_env = cppegret.Envelope(cov=cov_in.copy(), s=10.0, T=T_in.copy(), psix=0.2, psiy=-0.3)

cov_in[0, 0] = 999.0
T_in[0, 0] = 999.0

snap_py_case2 = snapshot(py_env)
snap_cpp_case2 = snapshot(cpp_env)
assert_snapshot_equal(snap_py_case2, snap_cpp_case2)
assert_eigenmode_snapshot_equal(snap_py_case2, snap_cpp_case2)
assert not np.isclose(py_env.cov[0, 0], 999.0)
assert not np.isclose(cpp_env.cov[0, 0], 999.0)
display(
    {
        "case2_tau_py_cpp": (snap_py_case2["tau"], snap_cpp_case2["tau"]),
        "case2_phase_py_cpp": (snap_py_case2["psix"], snap_cpp_case2["psix"], snap_py_case2["psiy"], snap_cpp_case2["psiy"]),
    }
)
plot_cov_diagonal_parity(snap_py_case2["cov"], snap_cpp_case2["cov"], "Case 2 covariance diagonal parity")
print("Case 2 passed: constructor parity and defensive copy")


# %% [markdown]
# ## Case 3: transfer parity + `T_matrix()` parity

# %%
length = 1.5
tmat = np.array(
    [
        [1.0, length, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, length],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

py_env.transfer(tmat, length)
cpp_env.transfer(tmat, length)

snap_py_case3 = snapshot(py_env)
snap_cpp_case3 = snapshot(cpp_env)
assert_snapshot_equal(snap_py_case3, snap_cpp_case3, atol=2.0e-12)
assert_eigenmode_snapshot_equal(snap_py_case3, snap_cpp_case3, atol=2.0e-12)
np.testing.assert_allclose(get_t_matrix(py_env), get_t_matrix(cpp_env), atol=2.0e-12, rtol=0.0)
display({"case3_s_py_cpp": (snap_py_case3["s"], snap_cpp_case3["s"]), "case3_tau_py_cpp": (snap_py_case3["tau"], snap_cpp_case3["tau"])})
plot_cov_diagonal_parity(snap_py_case3["cov"], snap_cpp_case3["cov"], "Case 3 covariance diagonal parity after transfer")
print("Case 3 passed: transfer and T_matrix parity")


# %% [markdown]
# ## Case 4: `copy()` parity + distinct instance

# %%
py_copy = py_env.copy()
cpp_copy = cpp_env.copy()

snap_py_copy = snapshot(py_copy)
snap_cpp_copy = snapshot(cpp_copy)
assert_snapshot_equal(snap_py_copy, snap_cpp_copy, atol=2.0e-12)
assert_eigenmode_snapshot_equal(snap_py_copy, snap_cpp_copy, atol=2.0e-12)
assert py_copy is not py_env
assert cpp_copy is not cpp_env
display({"case4_copy_tau_py_cpp": (snap_py_copy["tau"], snap_cpp_copy["tau"]), "case4_copy_s_py_cpp": (snap_py_copy["s"], snap_cpp_copy["s"])})
plot_cov_diagonal_parity(snap_py_copy["cov"], snap_cpp_copy["cov"], "Case 4 covariance diagonal parity for copy")
print("Case 4 passed: copy parity and distinct instance")


# %% [markdown]
# All `Envelope` parity checks passed.
