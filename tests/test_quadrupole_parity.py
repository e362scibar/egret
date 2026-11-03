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
# # Quadrupole parity test (`pyegret` vs `cppegret`)
#
# This notebook verifies `Quadrupole` behavior parity between:
# - `egret.python.Quadrupole` (pure-Python reference)
# - `egret.cpp.Quadrupole` (C++ backend wrapper)
#
# Each result is also checked against the analytic thin-structure map of a focusing (`k1 > 0`)
# or defocusing (`k1 < 0`) quadrupole, with tilt and offset conventions as implemented.
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
LENGTH = 0.5
K1 = 0.8
ATOL = 1.0e-12
METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")
COORD_VECTOR = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])  # x, x', y, y'
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
DX = 1.0e-3
DY = -2.0e-3
KICK_X = 4.0e-4
KICK_Y = -3.0e-4
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])
DISP_VECTOR = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])


def quad_matrix(length: float, k1: float) -> np.ndarray:
    # Thin-structure transfer map of a quadrupole in the untilted frame (k1 > 0 focuses in x).
    tmat = np.eye(4)
    if k1 == 0.0:
        tmat[0, 1] = length
        tmat[2, 3] = length
        return tmat
    sqrtk = np.sqrt(abs(k1))
    psi = sqrtk * length
    mf = np.array([[np.cos(psi), np.sin(psi) / sqrtk], [-sqrtk * np.sin(psi), np.cos(psi)]])
    md = np.array([[np.cosh(psi), np.sinh(psi) / sqrtk], [sqrtk * np.sinh(psi), np.cosh(psi)]])
    if k1 > 0.0:
        tmat[0:2, 0:2], tmat[2:4, 2:4] = mf, md
    else:
        tmat[0:2, 0:2], tmat[2:4, 2:4] = md, mf
    return tmat


def make_quad(module, k1: float = K1, length: float = LENGTH, **kwargs):
    return module.Quadrupole("Q", length, k1, **kwargs)


def make_coordinate(module, vector: np.ndarray = COORD_VECTOR, delta: float = DELTA0):
    return module.Coordinate(vector.copy(), S0, Z0, delta)


def assert_close(left, right, atol: float = ATOL, err_msg: str = "") -> None:
    np.testing.assert_allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float),
                               atol=atol, rtol=0.0, err_msg=err_msg)


def plot_profile_parity(k_values: np.ndarray, py_r12: np.ndarray, cpp_r12: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(k_values, py_r12, "o-", label="pyegret")
    ax.plot(k_values, cpp_r12, "x--", label="cppegret")
    ax.set_title(title)
    ax.set_xlabel("k1 [1/m^2]")
    ax.set_ylabel("R12 [m]")
    ax.legend()
    ax.grid(alpha=0.3)


# %% [markdown]
# ## Case 1: constructor parity, attributes, `k1` setter, and `copy()` independence

# %%
py_q = pyegret.Quadrupole("Q", LENGTH, K1, dx=DX, dy=DY, ds=0.1, tilt=0.2, info="quad")
cpp_q = cppegret.Quadrupole("Q", LENGTH, K1, dx=DX, dy=DY, ds=0.1, tilt=0.2, info="quad")

for name, expected in {"length": LENGTH, "k1": K1, "dx": DX, "dy": DY, "ds": 0.1, "tilt": 0.2}.items():
    assert_close(getattr(py_q, name), expected, err_msg=f"py {name}")
    assert_close(getattr(cpp_q, name), expected, err_msg=f"cpp {name}")
assert py_q.info == cpp_q.info == "quad"

py_copy = py_q.copy()
cpp_copy = cpp_q.copy()
assert_close(py_copy.k1, K1)
assert_close(cpp_copy.k1, K1)

py_copy.k1 = -0.3
cpp_copy.k1 = -0.3
assert_close(py_q.k1, K1, err_msg="py original changed by copy")
assert_close(cpp_q.k1, K1, err_msg="cpp original changed by copy")
# The fixture carries tilt = 0.2, so the reference is a freshly built quadrupole with the same tilt.
assert_close(py_copy.transfer_matrix(), pyegret.Quadrupole("Q", LENGTH, -0.3, tilt=0.2).transfer_matrix(), err_msg="py copy k1 setter")
assert_close(cpp_copy.transfer_matrix(), cppegret.Quadrupole("Q", LENGTH, -0.3, tilt=0.2).transfer_matrix(), err_msg="cpp copy k1 setter")
print("Case 1 passed: constructor parity, attributes, k1 setter, and copy independence")


# %% [markdown]
# ## Case 2: transfer matrix parity and analytic form (focusing, defocusing, and drift)

# %%
for k1 in (K1, -K1, 0.0):
    py_tmat = make_quad(pyegret, k1).transfer_matrix()
    cpp_tmat = make_quad(cppegret, k1).transfer_matrix()
    assert_close(py_tmat, quad_matrix(LENGTH, k1), err_msg=f"py analytic k1={k1}")
    assert_close(cpp_tmat, quad_matrix(LENGTH, k1), err_msg=f"cpp analytic k1={k1}")
    assert_close(cpp_tmat, py_tmat, err_msg=f"parity k1={k1}")

for module in (pyegret, cppegret):
    for method in METHODS:
        assert_close(make_quad(module).transfer_matrix(make_coordinate(module, delta=0.0), 0.1, method),
                     quad_matrix(LENGTH, K1), err_msg=f"{module.__name__} method={method}")
print("Case 2 passed: transfer matrix parity, analytic form, and integration method independence")


# %% [markdown]
# ## Case 3: momentum dependence, `k1 / (1 + delta)`

# %%
for delta in (1.0e-3, -2.0e-3):
    py_tmat = make_quad(pyegret).transfer_matrix(make_coordinate(pyegret, delta=delta))
    cpp_tmat = make_quad(cppegret).transfer_matrix(make_coordinate(cppegret, delta=delta))
    assert_close(py_tmat, quad_matrix(LENGTH, K1 / (1.0 + delta)), err_msg=f"py delta={delta}")
    assert_close(cpp_tmat, quad_matrix(LENGTH, K1 / (1.0 + delta)), err_msg=f"cpp delta={delta}")
    assert_close(cpp_tmat, py_tmat, err_msg=f"parity delta={delta}")
print("Case 3 passed: momentum-dependent focusing matches k1 / (1 + delta)")


# %% [markdown]
# ## Case 4: tilt convention (tilt = pi is the identity; tilt = pi/2 swaps the planes)

# %%
for module in (pyegret, cppegret):
    tilt_pi = make_quad(module, tilt=np.pi).transfer_matrix()
    tilt_half = make_quad(module, tilt=np.pi / 2).transfer_matrix()
    assert_close(tilt_pi, quad_matrix(LENGTH, K1), err_msg=f"{module.__name__} tilt pi")
    assert_close(tilt_half, quad_matrix(LENGTH, -K1), err_msg=f"{module.__name__} tilt pi/2")

py_skew = make_quad(pyegret, tilt=np.pi / 4).transfer_matrix()
cpp_skew = make_quad(cppegret, tilt=np.pi / 4).transfer_matrix()
assert_close(cpp_skew, py_skew, err_msg="parity skew quadrupole (tilt pi/4)")
print("Case 4 passed: tilt pi is the identity, tilt pi/2 swaps the planes, and skew matrices agree")


# %% [markdown]
# ## Case 5: transfer matrix array parity along the element (N-first layout)

# %%
py_arr, py_s = make_quad(pyegret).transfer_matrix_array(ds=0.125, endpoint=True)
cpp_arr, cpp_s = make_quad(cppegret).transfer_matrix_array(ds=0.125, endpoint=True)
assert np.asarray(py_arr).shape == np.asarray(cpp_arr).shape == (len(py_s), 4, 4)
assert_close(py_s, cpp_s, err_msg="array s")
assert_close(py_arr, cpp_arr, err_msg="tmat array parity")
for k, s_val in enumerate(py_s):
    if s_val > 0.0:
        assert_close(np.asarray(py_arr)[k], quad_matrix(s_val, K1), err_msg=f"analytic at s={s_val}")
print("Case 5 passed: transfer matrix array parity and partial-length analytic form")


# %% [markdown]
# ## Case 6: coordinate transfer parity, analytic map, and offsets

# %%
offset = np.array([DX, 0.0, DY, 0.0])
expected_vector = quad_matrix(LENGTH, K1) @ (COORD_VECTOR - offset) + offset

py_out, _, _ = make_quad(pyegret, dx=DX, dy=DY).transfer(make_coordinate(pyegret, delta=0.0))
cpp_out, _, _ = make_quad(cppegret, dx=DX, dy=DY).transfer(make_coordinate(cppegret, delta=0.0))
assert_close(py_out.vector, expected_vector, err_msg="py coordinate with offsets")
assert_close(cpp_out.vector, expected_vector, err_msg="cpp coordinate with offsets")
assert_close(cpp_out.vector, py_out.vector, err_msg="parity coordinate with offsets")
assert_close(cpp_out.s, S0 + LENGTH, err_msg="output s")
assert_close(cpp_out.z, Z0, err_msg="output z")
assert_close(cpp_out.delta, 0.0, err_msg="output delta equals input delta")

py_plain, _, _ = make_quad(pyegret).transfer(make_coordinate(pyegret, delta=0.0))
cpp_plain, _, _ = make_quad(cppegret).transfer(make_coordinate(cppegret, delta=0.0))
assert_close(cpp_plain.vector, quad_matrix(LENGTH, K1) @ COORD_VECTOR, err_msg="parity without offsets")
assert_close(cpp_plain.vector, py_plain.vector, err_msg="parity without offsets (python)")
print("Case 6 passed: coordinate transfer parity and offset map")


# %% [markdown]
# ## Case 7: envelope and dispersion transfer parity

# %%
expected_cov = quad_matrix(LENGTH, K1) @ COV @ quad_matrix(LENGTH, K1).T
py_env = pyegret.Envelope(cov=COV.copy(), s=S0)
cpp_env = cppegret.Envelope(cov=COV.copy(), s=S0)
_, py_env_out, _ = make_quad(pyegret).transfer(make_coordinate(pyegret, delta=0.0), py_env)
_, cpp_env_out, _ = make_quad(cppegret).transfer(make_coordinate(cppegret, delta=0.0), cpp_env)
assert_close(py_env_out.cov, expected_cov, atol=1.0e-10, err_msg="py envelope analytic")
assert_close(cpp_env_out.cov, expected_cov, atol=1.0e-10, err_msg="cpp envelope analytic")
assert_close(cpp_env_out.cov, py_env_out.cov, atol=1.0e-10, err_msg="parity envelope")

py_disp_in = pyegret.Dispersion(DISP_VECTOR.copy(), S0)
cpp_disp_in = cppegret.Dispersion(DISP_VECTOR.copy(), S0)
_, _, py_disp_out = make_quad(pyegret).transfer(make_coordinate(pyegret, delta=0.0), None, py_disp_in)
_, _, cpp_disp_out = make_quad(cppegret).transfer(make_coordinate(cppegret, delta=0.0), None, cpp_disp_in)
# The dispersion term includes the momentum derivative of the map acting on the orbit: dM/d(delta) @ v.
DELTA_H = 1.0e-6
chromatic = (quad_matrix(LENGTH, K1 / (1.0 + DELTA_H)) - quad_matrix(LENGTH, K1 / (1.0 - DELTA_H))) / (2.0 * DELTA_H) @ COORD_VECTOR
assert_close(py_disp_out.vector, quad_matrix(LENGTH, K1) @ DISP_VECTOR + chromatic, atol=1.0e-9, err_msg="py dispersion analytic")
assert_close(cpp_disp_out.vector, py_disp_out.vector, err_msg="parity dispersion")
print("Case 7 passed: envelope and dispersion transfer parity")


# %% [markdown]
# ## Case 8: transfer from inside the quadrupole and the zero-length identity

# %%
S_INSIDE = 0.2
remaining = LENGTH - S_INSIDE
py_mid = make_quad(pyegret).transfer_matrix_from_s(S_INSIDE, make_coordinate(pyegret, delta=0.0))
cpp_mid = make_quad(cppegret).transfer_matrix_from_s(S_INSIDE, make_coordinate(cppegret, delta=0.0))
assert_close(py_mid, quad_matrix(remaining, K1), err_msg="py transfer_matrix_from_s analytic")
assert_close(cpp_mid, quad_matrix(remaining, K1), err_msg="cpp transfer_matrix_from_s analytic")

py_zero, _, _ = pyegret.Quadrupole("Z", 0.0, K1).transfer(make_coordinate(pyegret, delta=0.0))
cpp_zero, _, _ = cppegret.Quadrupole("Z", 0.0, K1).transfer(make_coordinate(cppegret, delta=0.0))
assert_close(py_zero.vector, COORD_VECTOR, err_msg="py zero-length identity")
assert_close(cpp_zero.vector, COORD_VECTOR, err_msg="cpp zero-length identity")
print("Case 8 passed: transfer from inside the element and zero-length identity")


# %% [markdown]
# ## Case 9: steering kick realized by a magnetic-center offset (k1 != 0) or a distributed dipole kick (k1 = 0)
#
# For k1 != 0 the kick is produced by offsetting the magnetic center by c = B^-1 kick, where B is the
# (x', y') x (x, y) block of the transfer matrix. An on-axis particle then leaves with the angle change +kick.
# For k1 = 0 the kick is distributed over the length: x += kick * L / 2 and x' += kick.
# The kick scales with the momentum deviation as 1 / (1 + delta), as for Steering and NonlinearMultipole.
# All failures are collected so that the rest of the notebook runs.

# %%
def expected_steered(k1: float, vector: np.ndarray) -> np.ndarray:
    kick = np.array([KICK_X, KICK_Y])
    if k1 == 0.0:
        return quad_matrix(LENGTH, 0.0) @ vector + np.array([0.5 * KICK_X * LENGTH, KICK_X,
                                                             0.5 * KICK_Y * LENGTH, KICK_Y])
    tmat = quad_matrix(LENGTH, k1)
    block = np.array([[tmat[1, 0], tmat[1, 2]], [tmat[3, 0], tmat[3, 2]]])
    c = -np.linalg.solve(block, kick)
    offset = np.array([c[0], 0.0, c[1], 0.0])
    return tmat @ (vector - offset) + offset


failures = []
for k1 in (K1, -K1, 0.0):
    for module in (pyegret, cppegret):
        try:
            quad = module.Quadrupole("Q", LENGTH, k1, kick_x=KICK_X, kick_y=KICK_Y)
            out, _, _ = quad.transfer(make_coordinate(module, delta=0.0))
            assert_close(out.vector, expected_steered(k1, COORD_VECTOR),
                         err_msg=f"{module.__name__} steered transfer k1={k1}")
        except Exception as exc:
            failures.append(f"steering kick k1={k1} [{module.__name__}]: {type(exc).__name__}: {exc}")

# On-axis particle at a non-zero delta. For k1 != 0 the offset is fixed at its reference value, so the exit kick is
# -B(delta) c(0). For k1 = 0 the distributed kick scales exactly as kick / (1 + delta).
DELTA_TEST = 2.0e-3
for k1 in (K1, -K1, 0.0):
    for module in (pyegret, cppegret):
        try:
            quad = module.Quadrupole("Q", LENGTH, k1, kick_x=KICK_X, kick_y=KICK_Y)
            out, _, _ = quad.transfer(module.Coordinate(np.zeros(4), S0, Z0, DELTA_TEST))
            if k1 == 0.0:
                expected_on_axis = np.array([KICK_X, KICK_Y]) / (1.0 + DELTA_TEST)
            else:
                block0 = quad_matrix(LENGTH, k1)[np.ix_([1, 3], [0, 2])]
                offset0 = -np.linalg.solve(block0, np.array([KICK_X, KICK_Y]))
                block_delta = quad_matrix(LENGTH, k1 / (1.0 + DELTA_TEST))[np.ix_([1, 3], [0, 2])]
                expected_on_axis = -block_delta @ offset0
            assert_close([out.xp, out.yp], expected_on_axis, err_msg=f"{module.__name__} on-axis kick k1={k1}")
        except Exception as exc:
            failures.append(f"on-axis kick k1={k1} [{module.__name__}]: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 9 passed: steering kick by offset (k1 != 0) and distributed kick (k1 = 0) in both backends")


# %% [markdown]
# ## Case 10: k1 sweep plot of R12 for both backends

# %%
K_VALUES = np.linspace(-0.4, 0.4, 9)
py_r12 = np.array([make_quad(pyegret, k1).transfer_matrix()[0, 1] for k1 in K_VALUES])
cpp_r12 = np.array([make_quad(cppegret, k1).transfer_matrix()[0, 1] for k1 in K_VALUES])
assert_close(cpp_r12, py_r12, err_msg="R12 over k1 range")
plot_profile_parity(K_VALUES, py_r12, cpp_r12, "Case 10 R12 parity over k1")
print("Case 10 passed: R12 parity over the k1 sweep")


# %% [markdown]
# ## Case 11: kick properties, steering strength k0 = -kick / length, `copy()`, and unknown keywords

# %%
for module in (pyegret, cppegret):
    quad = module.Quadrupole("Q", LENGTH, K1, kick_x=KICK_X, kick_y=KICK_Y)
    assert_close(quad.kick_x, KICK_X, err_msg=f"{module.__name__} kick_x")
    assert_close(quad.kick_y, KICK_Y, err_msg=f"{module.__name__} kick_y")
    assert_close(quad.k0x, -KICK_X / LENGTH, err_msg=f"{module.__name__} k0x")
    assert_close(quad.k0y, -KICK_Y / LENGTH, err_msg=f"{module.__name__} k0y")

    quad.k0x = 0.02
    assert_close(quad.kick_x, -0.02 * LENGTH, err_msg=f"{module.__name__} kick_x after k0x setter")

    copied = quad.copy()
    assert_close(copied.kick_x, -0.02 * LENGTH, err_msg=f"{module.__name__} copy kick_x")
    assert_close(copied.kick_y, KICK_Y, err_msg=f"{module.__name__} copy kick_y")

    try:
        module.Quadrupole("Q", LENGTH, K1, bogus=1.0)
    except TypeError:
        pass
    else:
        raise AssertionError(f"{module.__name__}: unknown keyword accepted")
print("Case 11 passed: kick properties, k0 relation, copy, and unknown keyword rejection")


# %% [markdown]
# ## Case 12: `set_steering` on both backends (`None` leaves a kick unchanged)

# %%
for module in (pyegret, cppegret):
    steer = module.Quadrupole("Q", LENGTH, K1, kick_x=KICK_X, kick_y=KICK_Y)
    steer.set_steering(kick_x=5.0e-4)
    assert_close(steer.kick_x, 5.0e-4, err_msg=f"{module.__name__} set_steering kick_x only")
    assert_close(steer.kick_y, KICK_Y, err_msg=f"{module.__name__} set_steering leaves kick_y")

    steer.set_steering(kick_y=7.0e-4)
    assert_close(steer.kick_x, 5.0e-4, err_msg=f"{module.__name__} set_steering leaves kick_x")
    assert_close(steer.kick_y, 7.0e-4, err_msg=f"{module.__name__} set_steering kick_y only")

    steer.set_steering()
    assert_close(steer.kick_x, 5.0e-4, err_msg=f"{module.__name__} set_steering with no arguments")
    assert_close(steer.kick_y, 7.0e-4, err_msg=f"{module.__name__} set_steering with no arguments")

    steer.set_steering(-1.0e-4, -2.0e-4)
    assert_close(steer.kick_x, -1.0e-4, err_msg=f"{module.__name__} set_steering both kicks")
    assert_close(steer.kick_y, -2.0e-4, err_msg=f"{module.__name__} set_steering both kicks (y)")
print("Case 12 passed: set_steering updates only the kicks that are given, in both backends")


# %% [markdown]
# ## Case 13: transfer matrix is independent of the kick; the kick enters only the affine part
#
# The transfer matrix of the quadrupole does not depend on the kick: the kick is realized by a constant
# magnetic-center offset (k1 != 0) or by a constant distributed dipole kick (k1 = 0), so it only adds
# an affine term to the map. The transfer matrix is therefore identical with and without the kick.

# %%
for module in (pyegret, cppegret):
    kicked = module.Quadrupole("Q", LENGTH, K1, kick_x=KICK_X, kick_y=KICK_Y)
    plain = module.Quadrupole("Q", LENGTH, K1)
    assert_close(kicked.transfer_matrix(), plain.transfer_matrix(), err_msg=f"{module.__name__} tmat with kick")
    assert_close(kicked.transfer_matrix(), quad_matrix(LENGTH, K1), err_msg=f"{module.__name__} tmat analytic")
    tmat_array, _ = kicked.transfer_matrix_array(ds=0.125, endpoint=True)
    tmat_plain, _ = plain.transfer_matrix_array(ds=0.125, endpoint=True)
    assert_close(tmat_array, tmat_plain, err_msg=f"{module.__name__} tmat array with kick")
print("Case 13 passed: transfer matrices are identical with and without the kick")


# %% [markdown]
# ## Case 14: transfer_array with the kick; the end point equals the single transfer; analytic along s

# %%
def expected_array(k1: float, vector: np.ndarray, s_values: np.ndarray) -> np.ndarray:
    kick = np.array([KICK_X, KICK_Y])
    rows = []
    for s_val in s_values:
        if k1 == 0.0:
            kappa = kick / LENGTH
            rows.append(quad_matrix(s_val, 0.0) @ vector
                        + np.array([0.5 * kappa[0] * s_val**2, kappa[0] * s_val,
                                    0.5 * kappa[1] * s_val**2, kappa[1] * s_val]))
        else:
            tmat_full = quad_matrix(LENGTH, k1)
            block = np.array([[tmat_full[1, 0], tmat_full[1, 2]], [tmat_full[3, 0], tmat_full[3, 2]]])
            c = -np.linalg.solve(block, kick)
            offset = np.array([c[0], 0.0, c[1], 0.0])
            rows.append(quad_matrix(s_val, k1) @ (vector - offset) + offset)
    return np.array(rows).T


failures = []
for k1 in (K1, 0.0):
    for module in (pyegret, cppegret):
        try:
            quad = module.Quadrupole("Q", LENGTH, k1, dx=DX, dy=DY, kick_x=KICK_X, kick_y=KICK_Y)
            cood = make_coordinate(module, delta=0.0)
            arr, _, _ = quad.transfer_array(cood, ds=0.125, endpoint=True)
            single, _, _ = quad.transfer(cood)
            assert_close(np.asarray(arr.s), S0 + np.linspace(0.0, LENGTH, len(arr.s)), err_msg="s grid")
            assert_close(np.asarray(arr.vector)[:, -1], single.vector, err_msg="end point equals single transfer")
            s_rel = np.asarray(arr.s) - S0
            # Analytic check with the element offset included in the map (the offset is applied in the lab frame)
            base = COORD_VECTOR.copy()
            analytic = expected_array(k1, base - np.array([DX, 0.0, DY, 0.0]), s_rel)
            analytic = analytic + np.array([[DX], [0.0], [DY], [0.0]])
            assert_close(np.asarray(arr.vector), analytic, err_msg=f"analytic transfer_array k1={k1}")
        except Exception as exc:
            failures.append(f"transfer_array k1={k1} [{module.__name__}]: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 14 passed: transfer_array with kick and offset, consistent with transfer and the analytic map")


# %% [markdown]
# ## Case 15: dispersion and dispersion_array include the kick (checked against finite differences in delta)

# %%
H_DELTA = 1.0e-6
failures = []
for k1 in (K1, -K1, 0.0):
    for module in (pyegret, cppegret):
        try:
            quad = module.Quadrupole("Q", LENGTH, k1, dx=DX, dy=DY, kick_x=KICK_X, kick_y=KICK_Y)
            cood = make_coordinate(module, delta=0.0)

            def single_out(delta: float) -> np.ndarray:
                out, _, _ = quad.transfer(make_coordinate(module, delta=delta))
                return np.asarray(out.vector)

            def array_out(delta: float) -> np.ndarray:
                out, _, _ = quad.transfer_array(make_coordinate(module, delta=delta), ds=0.125, endpoint=True)
                return np.asarray(out.vector)

            fd_single = (single_out(H_DELTA) - single_out(-H_DELTA)) / (2.0 * H_DELTA)
            assert_close(quad.dispersion(cood, ds=0.125), fd_single, atol=1.0e-7,
                         err_msg=f"dispersion vs finite difference k1={k1}")
            fd_array = (array_out(H_DELTA) - array_out(-H_DELTA)) / (2.0 * H_DELTA)
            disp_array, _ = quad.dispersion_array(cood, ds=0.125, endpoint=True)
            assert_close(np.asarray(disp_array), fd_array, atol=1.0e-7,
                         err_msg=f"dispersion_array vs finite difference k1={k1}")
        except Exception as exc:
            failures.append(f"dispersion k1={k1} [{module.__name__}]: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print("Case 15 passed: dispersion and dispersion_array match finite differences in delta, both backends")


# %% [markdown]
# ## Case 16: trajectories and dispersion of a single quadrupole (ported from egret_test2)
#
# A particle starts at the orbit v0 with momentum delta, and a second particle starts on the dispersion orbit
# v0 + eta0 * delta. Their trajectories differ by delta * eta(s), so (x - x_delta) / delta = -eta(s) (the
# "-etax" and "-etay" curves in egret_test2). For each parameter set, both backends are checked for:
# (a) the end point of transfer_array equals transfer, (b) the finite-difference dispersion matches the
# dispersion from transfer_array, and (c) Python and C++ agree.

# %%
DELTA_FD = 1.0e-6
FD_ATOL = 1.0e-8
TRAJ_CASES = [
    dict(name="focusing k1=+2", length=0.5, k1=2.0, init=[1.0e-3, 0.0, 1.0e-3, 0.0],
         disp=[3.0e-3, 0.0, 3.0e-3, 0.0], dx=0.0, dy=0.0, kick=(0.0, 0.0)),
    dict(name="defocusing k1=-2", length=0.5, k1=-2.0, init=[1.0e-3, 0.0, 1.0e-3, 0.0],
         disp=[3.0e-3, 0.0, 3.0e-3, 0.0], dx=0.0, dy=0.0, kick=(0.0, 0.0)),
    dict(name="long weak k1=+0.5", length=1.0, k1=0.5, init=[0.0, 1.0e-3, 0.0, 1.0e-3],
         disp=[1.0e-3, -2.0e-4, 5.0e-4, 1.0e-4], dx=0.0, dy=0.0, kick=(0.0, 0.0)),
    dict(name="offset, kick, k1=-0.7", length=0.8, k1=-0.7, init=[-2.0e-3, 5.0e-4, 1.0e-3, -2.0e-4],
         disp=[2.0e-3, 1.0e-4, -1.0e-3, 2.0e-4], dx=1.0e-3, dy=-2.0e-3, kick=(4.0e-4, -3.0e-4)),
    dict(name="zero gradient with kick", length=0.5, k1=0.0, init=[1.0e-3, 0.0, -5.0e-4, 1.0e-4],
         disp=[2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dx=0.0, dy=0.0, kick=(4.0e-4, -3.0e-4)),
]


def trajectory_arrays(module, case: dict):
    quad = module.Quadrupole("Q", case["length"], case["k1"], dx=case["dx"], dy=case["dy"],
                             kick_x=case["kick"][0], kick_y=case["kick"][1])
    init = np.array(case["init"])
    eta0 = np.array(case["disp"])
    disp0 = module.Dispersion(eta0.copy(), 0.0)
    cood0 = module.Coordinate(init.copy(), 0.0, 0.0, 0.0)
    cood_arr, _, disp_arr = quad.transfer_array(cood0, disp0=disp0, ds=0.05, endpoint=True)
    cood_end, _, disp_end = quad.transfer(cood0, disp0=disp0)
    cood_d, _, _ = quad.transfer_array(module.Coordinate(init + eta0 * DELTA_FD, 0.0, 0.0, DELTA_FD),
                                       ds=0.05, endpoint=True)
    return cood_arr, disp_arr, cood_end, disp_end, cood_d


def plot_trajectory(case: dict, py_arr, cpp_arr, py_disp, cpp_disp, py_fd: np.ndarray) -> None:
    s = np.asarray(py_arr[0].s)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, np.asarray(py_arr[0].x), "r-", label="x (pyegret)")
    ax.plot(s, np.asarray(cpp_arr[0].x), "ro", markersize=3, label="x (cppegret)")
    ax.plot(s, np.asarray(py_arr[0].y), "b-", label="y (pyegret)")
    ax.plot(s, np.asarray(cpp_arr[0].y), "bo", markersize=3, label="y (cppegret)")
    ax.plot(s, -np.asarray(py_disp.vector)[0], "g--", label="-etax (pyegret)")
    ax.plot(s, -np.asarray(py_disp.vector)[2], "m--", label="-etay (pyegret)")
    ax.plot(s, py_fd[0], "c:", linewidth=3, label="x finite difference")
    ax.set_title(f"Case 16: {case['name']}")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("x, y [m], -eta [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


failures = []
for case in TRAJ_CASES:
    try:
        py_arr, py_disp, py_end, py_disp_end, py_d = trajectory_arrays(pyegret, case)
        cpp_arr, cpp_disp, cpp_end, cpp_disp_end, cpp_d = trajectory_arrays(cppegret, case)
        for label, arr, disp, end, disp_end, dd in (("python", py_arr, py_disp, py_end, py_disp_end, py_d),
                                                    ("cpp", cpp_arr, cpp_disp, cpp_end, cpp_disp_end, cpp_d)):
            module = pyegret if label == "python" else cppegret
            quad_single_vec = np.asarray(end.vector)
            assert_close(np.asarray(arr.vector)[:, -1], quad_single_vec, atol=1.0e-12,
                         err_msg=f"{label} end point equals transfer")
            assert_close(np.asarray(disp.vector)[:, -1], np.asarray(disp_end.vector), atol=1.0e-12,
                         err_msg=f"{label} dispersion end point equals transfer")
            fd = (np.asarray(arr.vector) - np.asarray(dd.vector)) / DELTA_FD
            assert_close(fd, -np.asarray(disp.vector), atol=FD_ATOL,
                         err_msg=f"{label} finite-difference dispersion")
        assert_close(np.asarray(cpp_arr.vector), np.asarray(py_arr.vector), atol=1.0e-12,
                     err_msg="python and cpp trajectories agree")
        assert_close(np.asarray(cpp_disp.vector), np.asarray(py_disp.vector), atol=1.0e-12,
                     err_msg="python and cpp dispersion agree")
        py_fd = (np.asarray(py_arr.vector) - np.asarray(py_d.vector)) / DELTA_FD
        plot_trajectory(case, (py_arr,), (cpp_arr,), py_disp, cpp_disp, py_fd)
        plt.show()
    except Exception as exc:
        failures.append(f"{case['name']}: {type(exc).__name__}: {exc}")
if failures:
    raise AssertionError("\n".join(failures))
print(f"Case 16 passed: {len(TRAJ_CASES)} trajectory and dispersion parameter sets, both backends")


# %% [markdown]
# ## Summary

# %%
print("Quadrupole parity cases finished.")
