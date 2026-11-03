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
# # Sextupole parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Sextupole` behavior parity between:
# - `egret.python.Sextupole` (pure-Python reference)
# - `egret.cpp.Sextupole` (C++ backend wrapper)
#
# The sextupole is a nonlinear multipole integrated numerically. Checks are:
# - the strengths at a coordinate against the analytic expressions `k0 = k2 (x^2 - y^2)/2 - i k2 x y` and
#   `k1 = k2 (x - i y)`;
# - symplecticity of the transfer matrices (`M^T J M = J`) for the symplectic integrators;
# - the zero-amplitude limit, which is a drift;
# - convergence of the midpoint integrator with the step size.
#
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
LENGTH = 0.2
K2 = 10.0
ATOL = 1.0e-12
SYMPLECTIC_METHODS = ("symplectic1", "symplectic2", "symplectic4")
METHODS = ("midpoint", "rk4") + SYMPLECTIC_METHODS
DS = 0.05
COORD_VECTOR = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])  # x, x', y, y'
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
# Symplectic form for the ordering (x, x', y, y')
J = np.zeros((4, 4))
J[0, 1], J[1, 0], J[2, 3], J[3, 2] = 1.0, -1.0, 1.0, -1.0


def drift_matrix(length: float) -> np.ndarray:
    tmat = np.eye(4)
    tmat[0, 1] = length
    tmat[2, 3] = length
    return tmat


def make_sextupole(module, k2: float = K2, **kwargs):
    return module.Sextupole("S", LENGTH, k2, **kwargs)


def make_coordinate(module, vector: np.ndarray = COORD_VECTOR, delta: float = 0.0):
    return module.Coordinate(vector.copy(), S0, Z0, delta)


def analytic_strengths(x0: float, y0: float, k2: float = K2) -> tuple[complex, complex]:
    k0 = k2 * (0.5 * (x0**2 - y0**2) - 1.0j * x0 * y0)
    k1 = k2 * (x0 - 1.0j * y0)
    return k0, k1


def assert_close(left, right, atol: float = ATOL, err_msg: str = "") -> None:
    np.testing.assert_allclose(np.asarray(left, dtype=complex if np.iscomplexobj(left) else float),
                               np.asarray(right, dtype=complex if np.iscomplexobj(right) else float),
                               atol=atol, rtol=0.0, err_msg=err_msg)


def assert_raises(fn, expected: type[Exception]) -> None:
    try:
        fn()
    except expected:
        return
    except Exception as exc:
        raise AssertionError(f"expected {expected.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {expected.__name__} but none was raised")


def symplectic_residual(tmat: np.ndarray) -> float:
    return float(np.max(np.abs(tmat.T @ J @ tmat - J)))


def plot_convergence(ds_values: np.ndarray, py_err: np.ndarray, cpp_err: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.loglog(ds_values, py_err, "o-", label="pyegret midpoint")
    ax.loglog(ds_values, cpp_err, "x--", label="cppegret midpoint")
    ax.loglog(ds_values, ds_values**2 * (cpp_err[0] / ds_values[0]**2), ":", color="gray", label="O(ds^2)")
    ax.set_title("Midpoint error against a fine-step reference")
    ax.set_xlabel("ds [m]")
    ax.set_ylabel("max |M - M_ref|")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity, attributes, `copy()`, and `k2` setter

# %%
py_sex = make_sextupole(pyegret, kick_x=1.0e-4, kick_y=-2.0e-4, tilt=0.1, dx=1.0e-3)
cpp_sex = make_sextupole(cppegret, kick_x=1.0e-4, kick_y=-2.0e-4, tilt=0.1, dx=1.0e-3)
for name, expected in {"length": LENGTH, "k2": K2, "kick_x": 1.0e-4, "kick_y": -2.0e-4, "tilt": 0.1,
                       "dx": 1.0e-3}.items():
    assert_close(getattr(py_sex, name), expected, err_msg=f"py {name}")
    assert_close(getattr(cpp_sex, name), expected, err_msg=f"cpp {name}")

py_copy = py_sex.copy()
cpp_copy = cpp_sex.copy()
py_copy.k2 = 3.0
cpp_copy.k2 = 3.0
assert_close(py_sex.k2, K2, err_msg="py original k2 changed by copy")
assert_close(cpp_sex.k2, K2, err_msg="cpp original k2 changed by copy")
assert_close(py_copy.k2, 3.0)
assert_close(cpp_copy.k2, 3.0)
print("Case 1 passed: constructor parity, attributes, copy independence, and k2 setter")


# %% [markdown]
# ## Case 2: strengths at a coordinate (`get_k`) against the analytic expressions

# %%
for x0, y0 in ((1.0e-3, -5.0e-4), (-2.0e-3, 3.0e-3), (0.0, 0.0)):
    vector = np.array([x0, 0.0, y0, 0.0])
    expected_k0, expected_k1 = analytic_strengths(x0, y0)
    py_k0, py_k1 = make_sextupole(pyegret).get_k(make_coordinate(pyegret, vector=vector))
    cpp_k0, cpp_k1 = make_sextupole(cppegret).get_k(make_coordinate(cppegret, vector=vector))
    assert_close(py_k0, expected_k0, err_msg=f"py k0 at ({x0}, {y0})")
    assert_close(py_k1, expected_k1, err_msg=f"py k1 at ({x0}, {y0})")
    assert_close(cpp_k0, expected_k0, err_msg=f"cpp k0 at ({x0}, {y0})")
    assert_close(cpp_k1, expected_k1, err_msg=f"cpp k1 at ({x0}, {y0})")
print("Case 2 passed: strengths match the analytic expressions in both backends")


# %% [markdown]
# ## Case 3: zero amplitude is a drift, and `k2 = 0` is a drift at any amplitude

# %%
for module in (pyegret, cppegret):
    zero_amp = make_sextupole(module).transfer_matrix(make_coordinate(module, vector=np.zeros(4)), DS)
    assert_close(zero_amp, drift_matrix(LENGTH), err_msg=f"{module.__name__} zero amplitude")
    no_sextupole = make_sextupole(module, k2=0.0).transfer_matrix(make_coordinate(module), DS)
    assert_close(no_sextupole, drift_matrix(LENGTH), err_msg=f"{module.__name__} k2 = 0")
print("Case 3 passed: zero amplitude and k2 = 0 reduce to a drift")


# %% [markdown]
# ## Case 4: transfer matrix parity for every integration method

# %%
for method in METHODS:
    py_m = make_sextupole(pyegret).transfer_matrix(make_coordinate(pyegret), DS, method)
    cpp_m = make_sextupole(cppegret).transfer_matrix(make_coordinate(cppegret), DS, method)
    assert_close(cpp_m, py_m, err_msg=f"parity {method}")
print("Case 4 passed: transfer matrix parity for midpoint, rk4, and symplectic methods")


# %% [markdown]
# ## Case 5: symplecticity of the symplectic integrators (`M^T J M = J`)

# %%
for module in (pyegret, cppegret):
    for method in SYMPLECTIC_METHODS:
        tmat = make_sextupole(module).transfer_matrix(make_coordinate(module), DS, method)
        residual = symplectic_residual(tmat)
        assert residual < 1.0e-12, f"{module.__name__} {method} residual {residual:.3e}"
print("Case 5 passed: symplectic integrators produce symplectic transfer matrices in both backends")


# %% [markdown]
# ## Case 6: step-size convergence of the symplectic-4 integrator

# %%
STEP_SIZES = np.array([0.1, 0.05, 0.025])
reference = make_sextupole(pyegret).transfer_matrix(make_coordinate(pyegret), 1.0e-3, "symplectic4")
for module in (pyegret, cppegret):
    errors = np.array([np.max(np.abs(make_sextupole(module).transfer_matrix(make_coordinate(module), ds,
                                                                            "symplectic4") - reference))
                       for ds in STEP_SIZES])
    assert np.all(np.diff(errors) <= 0.0), f"{module.__name__} error does not decrease with ds"
print("Case 6 passed: symplectic-4 error decreases with the step size in both backends")


# %% [markdown]
# ## Case 7: midpoint integrator convergence and parity
#
# The midpoint integrator is second order, so its error should fall by about 4 when `ds` is halved.

# %%
midpoint_ref = make_sextupole(pyegret).transfer_matrix(make_coordinate(pyegret), 1.0e-3, "symplectic4")
py_err = np.array([np.max(np.abs(make_sextupole(pyegret).transfer_matrix(make_coordinate(pyegret), ds, "midpoint")
                                 - midpoint_ref)) for ds in STEP_SIZES])
cpp_err = np.array([np.max(np.abs(make_sextupole(cppegret).transfer_matrix(make_coordinate(cppegret), ds, "midpoint")
                                  - midpoint_ref)) for ds in STEP_SIZES])
display({"case7_ds": STEP_SIZES, "case7_py_error": py_err, "case7_cpp_error": cpp_err})
plot_convergence(STEP_SIZES, py_err, cpp_err)

for label, err in (("pyegret", py_err), ("cppegret", cpp_err)):
    ratios = err[:-1] / err[1:]
    assert np.all(ratios > 3.0), f"{label} midpoint is not second order: error ratios {ratios}"
print("Case 7 passed: midpoint integrator is second order in both backends")


# %% [markdown]
# ## Case 8: coordinate transfer parity with offsets, kicks, and tilt

# %%
for kwargs in ({}, {"kick_x": 1.0e-4, "kick_y": -2.0e-4}, {"tilt": 0.3}, {"dx": 1.0e-3, "dy": -2.0e-3}):
    py_out, _, _ = make_sextupole(pyegret, **kwargs).transfer(make_coordinate(pyegret, delta=DELTA0), None, None, DS)
    cpp_out, _, _ = make_sextupole(cppegret, **kwargs).transfer(make_coordinate(cppegret, delta=DELTA0), None, None, DS)
    assert_close(cpp_out.vector, py_out.vector, err_msg=f"parity transfer {kwargs}")
    assert_close(cpp_out.s, S0 + LENGTH, err_msg=f"exit s {kwargs}")
print("Case 8 passed: coordinate transfer parity with offsets, kicks, and tilt")


# %% [markdown]
# ## Case 9: transfer matrix array parity (N-first layout) and longitudinal positions

# %%
py_arr, py_s = make_sextupole(pyegret).transfer_matrix_array(make_coordinate(pyegret), 0.04, True)
cpp_arr, cpp_s = make_sextupole(cppegret).transfer_matrix_array(make_coordinate(cppegret), 0.04, True)
assert np.asarray(py_arr).shape == np.asarray(cpp_arr).shape == (len(py_s), 4, 4)
assert_close(py_s, cpp_s, err_msg="array s")
assert_close(cpp_arr, py_arr, err_msg="tmat array parity")
assert_close(py_arr[0], np.eye(4), err_msg="first element is the identity")
print("Case 9 passed: transfer matrix array parity and N-first layout")


# %% [markdown]
# ## Case 10: envelope and dispersion transfer parity

# %%
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])
py_env = pyegret.Envelope(cov=COV.copy(), s=S0)
cpp_env = cppegret.Envelope(cov=COV.copy(), s=S0)
_, py_env_out, _ = make_sextupole(pyegret).transfer(make_coordinate(pyegret), py_env, None, DS)
_, cpp_env_out, _ = make_sextupole(cppegret).transfer(make_coordinate(cppegret), cpp_env, None, DS)
assert_close(cpp_env_out.cov, py_env_out.cov, atol=1.0e-10, err_msg="parity envelope")

DISP_VECTOR = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])
py_disp = pyegret.Dispersion(DISP_VECTOR.copy(), S0)
cpp_disp = cppegret.Dispersion(DISP_VECTOR.copy(), S0)
_, _, py_disp_out = make_sextupole(pyegret).transfer(make_coordinate(pyegret), None, py_disp, DS)
_, _, cpp_disp_out = make_sextupole(cppegret).transfer(make_coordinate(cppegret), None, cpp_disp, DS)
assert_close(cpp_disp_out.vector, py_disp_out.vector, err_msg="parity dispersion transfer")
print("Case 10 passed: envelope and dispersion transfer parity")


# %% [markdown]
# ## Case 11: `transfer_matrix(None)` means zero amplitude in both backends
#
# The C++ backend accepts `cood0=None`. The Python reference currently requires a coordinate.

# %%
failures = []
for module in (pyegret, cppegret):
    try:
        tmat_none = make_sextupole(module).transfer_matrix(None, DS)
        assert_close(tmat_none, drift_matrix(LENGTH), err_msg="cood0 = None")
    except Exception as exc:  # noqa: BLE001 - collect every backend's failure
        failures.append(f"transfer_matrix(None) [{module.__name__}]: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 11 passed: transfer_matrix(None) equals the zero-amplitude drift in both backends")


# %% [markdown]
# ## Case 12: zero-length sextupole leaves the coordinate unchanged

# %%
failures = []
for module in (pyegret, cppegret):
    try:
        zero_len, _, _ = module.Sextupole("S", 0.0, K2).transfer(make_coordinate(module, delta=DELTA0), None, None, DS)
        assert_close(zero_len.vector, COORD_VECTOR, err_msg="zero length")
    except Exception as exc:  # noqa: BLE001 - collect every backend's failure
        failures.append(f"zero-length transfer [{module.__name__}]: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 12 passed: zero-length sextupole is the identity in both backends")


# %% [markdown]
# ## Case 13: trajectories for every integration method, with and without a dipole kick
#
# A particle with an initial offset is transferred along the magnet as an array, for each integration method,
# focusing and defocusing sextupoles, and with and without a dipole kick. Checks are:
# - the array end equals the single-coordinate transfer;
# - the finite difference `(x(delta) - x(0)) / delta` equals the output dispersion, because the orbit is
#   linear in the momentum deviation;
# - with a dipole kick, the zero-amplitude response is `[L kx / 2, kx, L ky / 2, ky]`;
# - pyegret and cppegret agree along the whole trajectory.
#
# Failures are collected and reported together at the end of the cell. Tolerances depend on the order of accuracy
# of the integrator (first order: looser; second and fourth order: tight), set at about 10x the largest measured error.
# Two figures are shown: the raw residuals of every integrator, and each integrator's deviation from
# symplectic4 along the magnet on a log scale.

# %%
TRAJ_OFFSET = np.array([0.002, 0.0, -0.001, 0.0])
TRAJ_DISP0 = np.array([0.001, 0.0, 0.001, 0.0])
TRAJ_DELTA = 1.0e-6
TRAJ_DS = 0.05
TRAJ_PARITY_ATOL = 1.0e-9
TRAJ_METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")
# Order of accuracy of each integrator. Tolerances are set per order (about 10x the largest error measured for
# that order in both backends): first order keeps a looser bound, second and fourth order share a tight one.
TRAJ_ORDER = {"midpoint": 2, "rk4": 4, "symplectic1": 1, "symplectic2": 2, "symplectic4": 4}
TRAJ_FD_ATOL_BY_ORDER = {1: 2.0e-6, 2: 1.0e-8, 4: 1.0e-8}
TRAJ_KICK_ATOL_BY_ORDER = {1: 5.0e-5, 2: 1.0e-9, 4: 1.0e-9}
TRAJ_STRENGTHS = {"focusing": 10.0, "defocusing": -10.0}
TRAJ_KICKS = {"no kick": (0.0, 0.0), "dipole kick": (1.0e-4, -2.0e-4)}


def trajectory_case(module, k2: float, kick: tuple[float, float], method: str) -> dict[str, np.ndarray]:
    sex = module.Sextupole("S", LENGTH, k2, kick_x=kick[0], kick_y=kick[1])
    disp0 = module.Dispersion(TRAJ_DISP0.copy())
    cood0 = module.Coordinate(TRAJ_OFFSET.copy())
    cood_delta = module.Coordinate(TRAJ_OFFSET + TRAJ_DISP0 * TRAJ_DELTA, 0.0, 0.0, TRAJ_DELTA)
    arr, _, disp_arr = sex.transfer_array(cood0, ds=TRAJ_DS, disp0=disp0, method=method)
    arr_delta, _, _ = sex.transfer_array(cood_delta, ds=TRAJ_DS, method=method)
    single, _, _ = sex.transfer(module.Coordinate(TRAJ_OFFSET.copy()), disp0=disp0, ds=TRAJ_DS, method=method)
    zero_amp, _, _ = sex.transfer(module.Coordinate(np.zeros(4)), ds=TRAJ_DS, method=method)
    return {
        "s": np.asarray(arr.s, dtype=float),
        "vector": np.asarray(arr.vector, dtype=float),
        "x": np.asarray(arr.x, dtype=float),
        "fd_x": (np.asarray(arr_delta.x) - np.asarray(arr.x)) / TRAJ_DELTA,
        "fd_y": (np.asarray(arr_delta.y) - np.asarray(arr.y)) / TRAJ_DELTA,
        "disp_x": np.asarray(disp_arr.x, dtype=float),
        "disp_y": np.asarray(disp_arr.y, dtype=float),
        "end_vector": np.asarray(arr.vector, dtype=float)[:, -1],
        "single_vector": np.asarray(single.vector, dtype=float),
        "zero_amp_vector": np.asarray(zero_amp.vector, dtype=float),
    }


def trajectory_failures(label: str, traj: dict[str, np.ndarray], kick: tuple[float, float], method: str) -> list[str]:
    order = TRAJ_ORDER[method]
    fd_atol = TRAJ_FD_ATOL_BY_ORDER[order]
    kick_atol = TRAJ_KICK_ATOL_BY_ORDER[order]
    checks = [
        ("array end vs transfer", lambda: assert_close(traj["end_vector"], traj["single_vector"], atol=TRAJ_PARITY_ATOL)),
        ("fd x vs dispersion x", lambda: assert_close(traj["fd_x"], traj["disp_x"], atol=fd_atol)),
        ("fd y vs dispersion y", lambda: assert_close(traj["fd_y"], traj["disp_y"], atol=fd_atol)),
    ]
    if kick != (0.0, 0.0):
        kick_expected = np.array([0.5 * LENGTH * kick[0], kick[0], 0.5 * LENGTH * kick[1], kick[1]])
        checks.append(("zero-amplitude kick response",
                       lambda: assert_close(traj["zero_amp_vector"], kick_expected, atol=kick_atol)))
    failures = []
    for name, check in checks:
        try:
            check()
        except AssertionError as exc:
            failures.append(f"{label} {name}: {' '.join(str(exc).split())[:220]}")
    return failures


failures = []
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
TRAJ_CPP = {}
for kick_name, kick in TRAJ_KICKS.items():
    for strength_name, k2 in TRAJ_STRENGTHS.items():
        for method in TRAJ_METHODS:
            py_traj = trajectory_case(pyegret, k2, kick, method)
            cpp_traj = trajectory_case(cppegret, k2, kick, method)
            label = f"[{kick_name}, {strength_name}, {method}]"
            failures += trajectory_failures(f"py {label}", py_traj, kick, method)
            failures += trajectory_failures(f"cpp {label}", cpp_traj, kick, method)
            try:
                assert_close(cpp_traj["vector"], py_traj["vector"], atol=TRAJ_PARITY_ATOL)
            except AssertionError as exc:
                failures.append(f"parity {label}: {' '.join(str(exc).split())[:220]}")
            TRAJ_CPP[(kick_name, strength_name, method)] = cpp_traj
# Figure 1: raw residuals x - x0 - x'0 s of every integrator (cppegret).
fig_res, axes_res = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for (row, kick_name), ax_row in zip(enumerate(TRAJ_KICKS), axes_res):
    for (col, strength_name), ax in zip(enumerate(TRAJ_STRENGTHS), ax_row):
        for method in TRAJ_METHODS:
            traj = TRAJ_CPP[(kick_name, strength_name, method)]
            s_rel = traj["s"] - S0
            residual_x = traj["x"] - TRAJ_OFFSET[0] - TRAJ_OFFSET[1] * s_rel
            ax.plot(traj["s"], residual_x, label=method)
        ax.set_title(f"{strength_name} sextupole, {kick_name} (cppegret)")
        ax.set_ylabel("x - x0 - x'0 s [m]")
        ax.grid(alpha=0.3)
        ax.legend()
for ax in axes_res[-1]:
    ax.set_xlabel("s [m]")

# Figure 2: deviation of each integrator from the most accurate one (symplectic4), on a log scale. The raw residuals
# of all methods nearly coincide, so only the deviations are plotted.
REFERENCE_METHOD = "symplectic4"
for (row, kick_name), ax_row in zip(enumerate(TRAJ_KICKS), axes):
    for (col, strength_name), ax in zip(enumerate(TRAJ_STRENGTHS), ax_row):
        reference = TRAJ_CPP[(kick_name, strength_name, REFERENCE_METHOD)]
        for method in TRAJ_METHODS:
            if method == REFERENCE_METHOD:
                continue
            traj = TRAJ_CPP[(kick_name, strength_name, method)]
            ax.semilogy(traj["s"][1:], np.abs(traj["x"] - reference["x"])[1:], label=f"{method} (order {TRAJ_ORDER[method]})")
        ax.set_title(f"{strength_name} sextupole, {kick_name}")
        ax.set_ylabel(f"|x - x({REFERENCE_METHOD})| [m]")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
for ax in axes[-1]:
    ax.set_xlabel("s [m]")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 13 passed: trajectory checks for every method, with and without a dipole kick, in both backends")


# %% [markdown]
# ## Case 14: alignment offsets (dx, dy) and longitudinal offset (ds) in both backends
#
# An element with offsets is a drift of `ds`, then the centred map applied to the coordinate relative to the offset,
# then the offset restored, then a drift of `-ds`. The exit coordinate and the envelope (beta, alpha, emittance) are
# checked against this map. The orbit error is the exit offset of a particle that enters on the axis.

# %%
ALIGN_CASES = [(1.0e-3, 0.0, 0.0), (0.0, -2.0e-3, 0.0), (0.0, 0.0, 3.0e-2), (1.5e-3, -8.0e-4, -2.0e-2)]  # (dx, dy, ds) in metres
ALIGN_METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")


def centred_map(module, centred, vector: np.ndarray, method: str) -> np.ndarray:
    return centred.transfer(module.Coordinate(vector.copy(), 0.0, 0.0, 0.0), None, None, DS, method)[0].vector


def expected_aligned(module, centred, vector: np.ndarray, dx: float, dy: float, ds: float, method: str) -> np.ndarray:
    offset = np.array([dx, 0.0, dy, 0.0])
    drift_in = np.eye(4)
    drift_in[0, 1], drift_in[2, 3] = ds, ds
    drift_out = np.eye(4)
    drift_out[0, 1], drift_out[2, 3] = -ds, -ds
    return drift_out @ (centred_map(module, centred, drift_in @ vector - offset, method) + offset)


for dx, dy, ds in ALIGN_CASES:
    for method in ALIGN_METHODS:
        for module in (pyegret, cppegret):
            centred = make_sextupole(module)
            aligned = make_sextupole(module, dx=dx, dy=dy, ds=ds)
            cood, _, _ = aligned.transfer(module.Coordinate(COORD_VECTOR.copy(), 0.0, 0.0, 0.0), None, None, DS,
                                          method)
            expected = expected_aligned(module, centred, COORD_VECTOR, dx, dy, ds, method)
            assert_close(cood.vector, expected, atol=1.0e-9,
                         err_msg=f"{module.__name__} {method} dx={dx} dy={dy} ds={ds}")
    print(f"dx={dx:+.1e} dy={dy:+.1e} ds={ds:+.1e}: all integration methods agree with the centred element")
print("Case 14 passed: sextupole with offsets and ds equals the centred element on the offset coordinate, in both backends")


# %% [markdown]
# ## Summary

# %%
print("Sextupole parity cases finished.")
