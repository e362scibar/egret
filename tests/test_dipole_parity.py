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
# # Dipole parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Dipole` behavior parity between:
# - `egret.python.Dipole` (pure-Python reference)
# - `egret.cpp.Dipole` (C++ backend wrapper)
#
# Results are also checked against the analytic sector-bend map, `rho = L / theta`,
# and the momentum-dependent radius `rho (1 + delta)` with bending angle `theta / (1 + delta)`.
# Edge angles and pole-face curvatures are not applied by either backend and are not tested here.

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
ANGLE = 0.1
K1_COMBINED = 0.3
ATOL = 1.0e-12
METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")
COORD_VECTOR = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])  # x, x', y, y'
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
DISP_VECTOR = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])  # eta_x, eta_x', eta_y, eta_y'


def sector_matrix(length: float, angle: float, delta: float = 0.0) -> np.ndarray:
    # Sector bend without quadrupole component: x-plane sector map, y-plane drift.
    rho = (length / angle) * (1.0 + delta)
    phi = angle / (1.0 + delta)
    tmat = np.eye(4)
    tmat[0:2, 0:2] = np.array([[np.cos(phi), rho * np.sin(phi)], [-np.sin(phi) / rho, np.cos(phi)]])
    tmat[2, 3] = length
    return tmat


def partial_sector_matrix(s_val: float, length: float, angle: float) -> np.ndarray:
    # Map from the entrance to s_val inside the bend: radius rho = length / angle is fixed, arc angle phi = s_val / rho.
    rho = length / angle
    phi = s_val / rho
    tmat = np.eye(4)
    tmat[0:2, 0:2] = np.array([[np.cos(phi), rho * np.sin(phi)], [-np.sin(phi) / rho, np.cos(phi)]])
    tmat[2, 3] = s_val
    return tmat


def sector_dispersion(length: float, angle: float) -> np.ndarray:
    rho = length / angle
    return np.array([rho * (1.0 - np.cos(angle)), np.sin(angle), 0.0, 0.0])


def make_dipole(module, **kwargs):
    return module.Dipole("B", LENGTH, ANGLE, **kwargs)


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


def plot_profile_parity(x_values: np.ndarray, py_values: np.ndarray, cpp_values: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x_values, py_values, "o-", label="pyegret")
    ax.plot(x_values, cpp_values, "x--", label="cppegret")
    ax.set_title(title)
    ax.set_xlabel("k1 [1/m^2]")
    ax.set_ylabel("R12 [m]")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity, attributes, `rho`, and `copy()`

# %%
py_dip = pyegret.Dipole("B", LENGTH, ANGLE, k1=K1_COMBINED, dx=1.0e-3, dy=-2.0e-3, ds=0.1, tilt=0.2, info="bend")
cpp_dip = cppegret.Dipole("B", LENGTH, ANGLE, k1=K1_COMBINED, dx=1.0e-3, dy=-2.0e-3, ds=0.1, tilt=0.2, info="bend")

for name, expected in {"length": LENGTH, "angle": ANGLE, "k1": K1_COMBINED, "dx": 1.0e-3,
                       "dy": -2.0e-3, "ds": 0.1, "tilt": 0.2}.items():
    assert_close(getattr(py_dip, name), expected, err_msg=f"py {name}")
    assert_close(getattr(cpp_dip, name), expected, err_msg=f"cpp {name}")
assert_close(py_dip.rho, LENGTH / ANGLE, err_msg="py rho")
assert_close(cpp_dip.rho, LENGTH / ANGLE, err_msg="cpp rho")
assert py_dip.info == cpp_dip.info == "bend"

py_copy = py_dip.copy()
cpp_copy = cpp_dip.copy()
assert_close(py_copy.rho, LENGTH / ANGLE, err_msg="py copy rho")
assert_close(cpp_copy.rho, LENGTH / ANGLE, err_msg="cpp copy rho")
assert_close(py_copy.k1, K1_COMBINED)
assert_close(cpp_copy.k1, K1_COMBINED)
print("Case 1 passed: constructor parity, attributes, rho, and copy")


# %% [markdown]
# ## Case 2: sector-bend transfer matrix parity and analytic form (k1 = 0), including momentum dependence

# %%
for delta in (0.0, 1.0e-3, -2.0e-3):
    coord = make_coordinate(pyegret, delta=delta)
    py_tmat = make_dipole(pyegret).transfer_matrix(coord)
    cpp_tmat = make_dipole(cppegret).transfer_matrix(make_coordinate(cppegret, delta=delta))
    assert_close(py_tmat, sector_matrix(LENGTH, ANGLE, delta), err_msg=f"py analytic delta={delta}")
    assert_close(cpp_tmat, sector_matrix(LENGTH, ANGLE, delta), err_msg=f"cpp analytic delta={delta}")
    assert_close(cpp_tmat, py_tmat, err_msg=f"parity delta={delta}")

for module in (pyegret, cppegret):
    for method in METHODS:
        tmat = make_dipole(module).transfer_matrix(make_coordinate(module, delta=0.0), 0.1, method)
        assert_close(tmat, sector_matrix(LENGTH, ANGLE), err_msg=f"{module.__name__} method={method}")
print("Case 2 passed: sector-bend transfer matrix parity, analytic form, and momentum dependence")


# %% [markdown]
# ## Case 3: combined-function transfer matrix parity; the k1 -> 0 limit reduces to the sector bend

# %%
K1_VALUES = np.linspace(-0.4, 0.4, 9)
py_r12 = np.array([make_dipole(pyegret, k1=k1).transfer_matrix()[0, 1] for k1 in K1_VALUES])
cpp_r12 = np.array([make_dipole(cppegret, k1=k1).transfer_matrix()[0, 1] for k1 in K1_VALUES])
assert_close(cpp_r12, py_r12, err_msg="R12 over k1 range")

for module in (pyegret, cppegret):
    tiny = make_dipole(module, k1=1.0e-9).transfer_matrix()
    assert_close(tiny, sector_matrix(LENGTH, ANGLE), atol=1.0e-7, err_msg=f"{module.__name__} k1 -> 0")

plot_profile_parity(K1_VALUES, py_r12, cpp_r12, "Case 3 R12 parity over k1")
print("Case 3 passed: combined-function parity and k1 -> 0 limit")


# %% [markdown]
# ## Case 4: dispersion parity and the analytic sector-bend dispersion

# %%
expected_disp = sector_dispersion(LENGTH, ANGLE)
py_disp = make_dipole(pyegret).dispersion(make_coordinate(pyegret, vector=np.zeros(4), delta=0.0))
cpp_disp = make_dipole(cppegret).dispersion(make_coordinate(cppegret, vector=np.zeros(4), delta=0.0))
assert_close(py_disp, expected_disp, err_msg="py dispersion analytic")
assert_close(cpp_disp, expected_disp, err_msg="cpp dispersion analytic")
assert_close(cpp_disp, py_disp, err_msg="parity dispersion")
# The dispersion also depends on the coordinate vector; both backends must agree for a non-zero orbit.
py_disp_v = make_dipole(pyegret).dispersion(make_coordinate(pyegret, delta=0.0))
cpp_disp_v = make_dipole(cppegret).dispersion(make_coordinate(cppegret, delta=0.0))
assert_close(cpp_disp_v, py_disp_v, err_msg="parity dispersion with non-zero orbit")
print("Case 4 passed: dispersion parity and analytic sector-bend dispersion")


# %% [markdown]
# ## Case 5: coordinate transfer parity and offset behavior

# %%
expected_vector = sector_matrix(LENGTH, ANGLE) @ COORD_VECTOR
py_out, _, _ = make_dipole(pyegret).transfer(make_coordinate(pyegret, delta=0.0))
cpp_out, _, _ = make_dipole(cppegret).transfer(make_coordinate(cppegret, delta=0.0))
assert_close(py_out.vector, expected_vector, err_msg="py coordinate analytic")
assert_close(cpp_out.vector, expected_vector, err_msg="cpp coordinate analytic")
assert_close(cpp_out.vector, py_out.vector, err_msg="parity coordinate")
assert_close(cpp_out.s, S0 + LENGTH)

# Momentum offset: transfer includes the dispersion term disp * delta
py_off, _, _ = make_dipole(pyegret).transfer(make_coordinate(pyegret, delta=DELTA0))
cpp_off, _, _ = make_dipole(cppegret).transfer(make_coordinate(cppegret, delta=DELTA0))
assert_close(cpp_off.vector, py_off.vector, err_msg="parity with momentum offset")

# Element offsets: parity between backends
py_shift, _, _ = make_dipole(pyegret, dx=1.0e-3, dy=-2.0e-3).transfer(make_coordinate(pyegret, delta=0.0))
cpp_shift, _, _ = make_dipole(cppegret, dx=1.0e-3, dy=-2.0e-3).transfer(make_coordinate(cppegret, delta=0.0))
assert_close(cpp_shift.vector, py_shift.vector, err_msg="parity with element offsets")
print("Case 5 passed: coordinate transfer parity, momentum offset, and element offsets")


# %% [markdown]
# ## Case 6: envelope and dispersion transfer parity

# %%
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])
tmat0 = sector_matrix(LENGTH, ANGLE)
expected_cov = tmat0 @ COV @ tmat0.T

py_env = pyegret.Envelope(cov=COV.copy(), s=S0)
cpp_env = cppegret.Envelope(cov=COV.copy(), s=S0)
_, py_env_out, _ = make_dipole(pyegret).transfer(make_coordinate(pyegret, delta=0.0), py_env)
_, cpp_env_out, _ = make_dipole(cppegret).transfer(make_coordinate(cppegret, delta=0.0), cpp_env)
assert_close(py_env_out.cov, expected_cov, atol=1.0e-10, err_msg="py envelope analytic")
assert_close(cpp_env_out.cov, expected_cov, atol=1.0e-10, err_msg="cpp envelope analytic")
assert_close(cpp_env_out.cov, py_env_out.cov, atol=1.0e-10, err_msg="parity envelope")

py_disp_in = pyegret.Dispersion(DISP_VECTOR.copy(), S0)
cpp_disp_in = cppegret.Dispersion(DISP_VECTOR.copy(), S0)
_, _, py_disp_out = make_dipole(pyegret).transfer(make_coordinate(pyegret, delta=0.0), None, py_disp_in)
_, _, cpp_disp_out = make_dipole(cppegret).transfer(make_coordinate(cppegret, delta=0.0), None, cpp_disp_in)
assert_close(cpp_disp_out.vector, py_disp_out.vector, err_msg="parity dispersion transfer")
print("Case 6 passed: envelope and dispersion transfer parity")


# %% [markdown]
# ## Case 7: transfer matrix array and coordinate array parity (N-first layout)

# %%
py_arr, py_s = make_dipole(pyegret).transfer_matrix_array(ds=0.25, endpoint=True)
cpp_arr, cpp_s = make_dipole(cppegret).transfer_matrix_array(ds=0.25, endpoint=True)
assert np.asarray(py_arr).shape == np.asarray(cpp_arr).shape == (len(py_s), 4, 4)
assert_close(py_s, cpp_s, err_msg="array s")
assert_close(py_arr, cpp_arr, err_msg="tmat array parity")
for k, s_val in enumerate(py_s):
    assert_close(np.asarray(py_arr)[k], partial_sector_matrix(s_val, LENGTH, ANGLE), err_msg=f"analytic at s={s_val}")

py_carr, _, _ = make_dipole(pyegret).transfer_array(make_coordinate(pyegret, delta=0.0), ds=0.25, endpoint=True)
cpp_carr, _, _ = make_dipole(cppegret).transfer_array(make_coordinate(cppegret, delta=0.0), ds=0.25, endpoint=True)
assert_close(py_carr.s, cpp_carr.s, err_msg="coordinate array s")
assert_close(py_carr.vector, cpp_carr.vector, err_msg="coordinate array vector")
print("Case 7 passed: transfer matrix array and coordinate array parity")


# %% [markdown]
# ## Case 8: zero bending angle is rejected with `ValueError` in both backends

# %%
assert_raises(lambda: pyegret.Dipole("B", LENGTH, 0.0), ValueError)
assert_raises(lambda: cppegret.Dipole("B", LENGTH, 0.0), ValueError)
print("Case 8 passed: zero bending angle rejected in both backends")


# %% [markdown]
# ## Case 9: changing `length` updates `rho` and the transfer matrix in both backends

# %%
NEW_LENGTH = 2.0
py_set = make_dipole(pyegret)
cpp_set = make_dipole(cppegret)
py_set.length = NEW_LENGTH
cpp_set.length = NEW_LENGTH

assert_close(py_set.rho, NEW_LENGTH / ANGLE, err_msg="py rho after length setter")
assert_close(cpp_set.rho, NEW_LENGTH / ANGLE, err_msg="cpp rho after length setter")
assert_close(py_set.transfer_matrix(), sector_matrix(NEW_LENGTH, ANGLE), err_msg="py tmat after length setter")
assert_close(cpp_set.transfer_matrix(), sector_matrix(NEW_LENGTH, ANGLE), err_msg="cpp tmat after length setter")
print("Case 9 passed: length setter updates rho and the transfer matrix")


# %% [markdown]
# ## Case 10: changing `angle` updates `rho` and the transfer matrix in both backends

# %%
NEW_ANGLE = 0.05
py_ang = make_dipole(pyegret)
py_ang.angle = NEW_ANGLE
assert_close(py_ang.rho, LENGTH / NEW_ANGLE, err_msg="py rho after angle setter")
assert_close(py_ang.transfer_matrix(), sector_matrix(LENGTH, NEW_ANGLE), err_msg="py tmat after angle setter")

cpp_ang = make_dipole(cppegret)
cpp_ang.angle = NEW_ANGLE
assert_close(cpp_ang.rho, LENGTH / NEW_ANGLE, err_msg="cpp rho after angle setter")
assert_close(cpp_ang.transfer_matrix(), sector_matrix(LENGTH, NEW_ANGLE), err_msg="cpp tmat after angle setter")

# Setting the angle to zero is rejected, as in the constructor
assert_raises(lambda: setattr(make_dipole(pyegret), "angle", 0.0), ValueError)
assert_raises(lambda: setattr(make_dipole(cppegret), "angle", 0.0), ValueError)
print("Case 10 passed: angle setter updates rho and the transfer matrix; zero angle rejected")


# %% [markdown]
# ## Case 11: trajectories and dispersion of a single bend (same checks as the quadrupole trajectory case)
#
# A particle starts at the orbit v0 with momentum delta, and a second particle starts at v0 + eta0 * delta.
# Their trajectories differ by delta * eta(s), so (x - x_delta) / delta = -eta(s).
# For each parameter set, both backends are checked for:
# (a) the end point of transfer_array equals transfer (coordinates and dispersion),
# (b) the finite-difference dispersion matches the dispersion from transfer_array,
# (c) Python and C++ agree.

# %%
DELTA_FD = 1.0e-6
FD_ATOL = 1.0e-8
BEND_TRAJ_CASES = [
    dict(name="sector bend, no quadrupole", length=1.0, angle=0.1, k1=0.0, init=[1.0e-3, 0.0, 1.0e-3, 0.0],
         disp=[3.0e-3, 0.0, 3.0e-3, 0.0], dx=0.0, dy=0.0),
    dict(name="combined function, k1=+0.5", length=0.5, angle=0.2, k1=0.5, init=[-2.0e-3, 5.0e-4, 1.0e-3, -2.0e-4],
         disp=[2.0e-3, 1.0e-4, -1.0e-3, 2.0e-4], dx=0.0, dy=0.0),
    dict(name="offset, combined function, k1=-0.3", length=1.2, angle=0.05, k1=-0.3,
         init=[1.0e-3, 0.0, -1.0e-3, 1.0e-4], disp=[1.0e-3, -2.0e-4, 5.0e-4, 1.0e-4], dx=1.0e-3, dy=-2.0e-3),
]


def bend_trajectory_arrays(module, case: dict):
    bend = module.Dipole("B", case["length"], case["angle"], k1=case["k1"], dx=case["dx"], dy=case["dy"])
    init = np.array(case["init"])
    eta0 = np.array(case["disp"])
    disp0 = module.Dispersion(eta0.copy(), 0.0)
    cood0 = module.Coordinate(init.copy(), 0.0, 0.0, 0.0)
    cood_arr, _, disp_arr = bend.transfer_array(cood0, disp0=disp0, ds=0.05, endpoint=True)
    cood_end, _, disp_end = bend.transfer(cood0, disp0=disp0)
    cood_plus, _, _ = bend.transfer_array(module.Coordinate(init + eta0 * DELTA_FD, 0.0, 0.0, DELTA_FD),
                                          ds=0.05, endpoint=True)
    cood_minus, _, _ = bend.transfer_array(module.Coordinate(init - eta0 * DELTA_FD, 0.0, 0.0, -DELTA_FD),
                                           ds=0.05, endpoint=True)
    # Central difference in delta: (x_minus - x_plus) / (2 delta) = -eta(s) with O(delta^2) truncation error
    fd = (np.asarray(cood_minus.vector) - np.asarray(cood_plus.vector)) / (2.0 * DELTA_FD)
    return cood_arr, disp_arr, cood_end, disp_end, fd


def plot_bend_trajectory(case: dict, py_arr, cpp_arr, py_disp, py_fd: np.ndarray) -> None:
    s = np.asarray(py_arr.s)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, np.asarray(py_arr.x), "r-", label="x (pyegret)")
    ax.plot(s, np.asarray(cpp_arr.x), "ro", markersize=3, label="x (cppegret)")
    ax.plot(s, np.asarray(py_arr.y), "b-", label="y (pyegret)")
    ax.plot(s, np.asarray(cpp_arr.y), "bo", markersize=3, label="y (cppegret)")
    ax.plot(s, -np.asarray(py_disp.vector)[0], "g--", label="-etax (pyegret)")
    ax.plot(s, -np.asarray(py_disp.vector)[2], "m--", label="-etay (pyegret)")
    ax.plot(s, py_fd[0], "c:", linewidth=3, label="x finite difference")
    ax.set_title(f"Case 11: {case['name']}")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("x, y [m], -eta [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


failures = []
for case in BEND_TRAJ_CASES:
    try:
        results = {}
        for module in (pyegret, cppegret):
            results[module.__name__] = bend_trajectory_arrays(module, case)
        py_arr, py_disp, py_end, py_disp_end, py_fd = results["egret.python"]
        cpp_arr, cpp_disp, cpp_end, cpp_disp_end, cpp_fd = results["egret.cpp"]
        for label, arr, disp, end, disp_end, fd in (("python", py_arr, py_disp, py_end, py_disp_end, py_fd),
                                                    ("cpp", cpp_arr, cpp_disp, cpp_end, cpp_disp_end, cpp_fd)):
            try:
                assert_close(np.asarray(arr.vector)[:, -1], np.asarray(end.vector), atol=1.0e-12,
                             err_msg=f"{label} end point equals transfer")
            except AssertionError as exc:
                failures.append(f"{case['name']} [{label}] end point: {exc}")
            try:
                assert_close(np.asarray(disp.vector)[:, -1], np.asarray(disp_end.vector), atol=1.0e-12,
                             err_msg=f"{label} dispersion end point equals transfer")
            except AssertionError as exc:
                failures.append(f"{case['name']} [{label}] dispersion end point: {exc}")
            try:
                assert_close(fd, -np.asarray(disp.vector), atol=FD_ATOL, err_msg=f"{label} finite-difference dispersion")
            except AssertionError as exc:
                failures.append(f"{case['name']} [{label}] finite-difference dispersion: {exc}")
        assert_close(np.asarray(cpp_arr.vector), np.asarray(py_arr.vector), atol=1.0e-12,
                     err_msg="python and cpp trajectories agree")
        assert_close(np.asarray(cpp_disp.vector), np.asarray(py_disp.vector), atol=1.0e-12,
                     err_msg="python and cpp dispersion agree")
        plot_bend_trajectory(case, py_arr, cpp_arr, py_disp, py_fd)
        plt.show()
    except Exception as exc:
        failures.append(f"{case['name']}: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print(f"Case 11 passed: {len(BEND_TRAJ_CASES)} bend trajectory and dispersion parameter sets, both backends")


# %% [markdown]
# ## Summary

# %%
print("Dipole parity cases finished.")
