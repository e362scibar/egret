"""CI checks for the Sextupole element (pytest counterpart of test_sextupole_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import (
    METHODS, MULTIPOLE_ORDER, TRAJ_FD_ATOL_BY_ORDER, TRAJ_KICK_ATOL_BY_METHOD, TRAJ_KICK_ATOL_BY_ORDER,
    assert_close, drift_matrix, multipole_symplectic_residual, multipole_trajectory,
)

LENGTH = 0.2
K2 = 10.0
SYMPLECTIC = ("symplectic1", "symplectic2", "symplectic4")
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
DS = 0.05
KICKS = {"no kick": (0.0, 0.0), "dipole kick": (1.0e-4, -2.0e-4)}
STRENGTHS = {"focusing": 10.0, "defocusing": -10.0}


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def analytic_strengths(x0: float, y0: float, k2: float):
    k0 = k2 * (0.5 * (x0**2 - y0**2) - 1.0j * x0 * y0)
    k1 = k2 * (x0 - 1.0j * y0)
    return k0, k1


def test_constructor_attributes_copy_and_setter(backend):
    sex = backend.Sextupole("S", LENGTH, K2, kick_x=1.0e-4, kick_y=-2.0e-4, tilt=0.1, dx=1.0e-3)
    assert_close(sex.length, LENGTH)
    assert_close(sex.k2, K2)
    assert_close(sex.kick_x, 1.0e-4)
    assert_close(sex.kick_y, -2.0e-4)
    assert_close(sex.tilt, 0.1)
    copied = sex.copy()
    copied.k2 = 3.0
    assert_close(sex.k2, K2, err_msg="original k2 changed by copy")
    assert_close(copied.k2, 3.0)


@pytest.mark.parametrize("x0, y0", [(1.0e-3, -5.0e-4), (-2.0e-3, 3.0e-3), (0.0, 0.0)])
def test_get_k_matches_analytic_strengths(backend, x0, y0):
    vector = np.array([x0, 0.0, y0, 0.0])
    k0, k1 = backend.Sextupole("S", LENGTH, K2).get_k(backend.Coordinate(vector.copy(), 0.0, 0.0, 0.0))
    expected_k0, expected_k1 = analytic_strengths(x0, y0, K2)
    assert_close(k0, expected_k0, err_msg="k0")
    assert_close(k1, expected_k1, err_msg="k1")


@pytest.mark.parametrize("method", METHODS)
def test_zero_amplitude_is_a_drift(backend, method):
    tmat = backend.Sextupole("S", LENGTH, K2).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS, method)
    assert_close(tmat, drift_matrix(LENGTH), err_msg=method)


def test_zero_strength_is_a_drift_at_any_amplitude(backend):
    tmat = backend.Sextupole("S", LENGTH, 0.0).transfer_matrix(backend.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS)
    assert_close(tmat, drift_matrix(LENGTH))


@pytest.mark.parametrize("method", METHODS)
def test_transfer_matrix_parity_for_every_method(method):
    py = pyegret.Sextupole("S", LENGTH, K2).transfer_matrix(pyegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS, method)
    cpp = cppegret.Sextupole("S", LENGTH, K2).transfer_matrix(cppegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS, method)
    assert_close(cpp, py, err_msg=method)


@pytest.mark.parametrize("method", SYMPLECTIC)
def test_symplectic_integrators_give_symplectic_matrices(backend, method):
    tmat = backend.Sextupole("S", LENGTH, K2).transfer_matrix(backend.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS, method)
    assert multipole_symplectic_residual(tmat) < 1.0e-12


def test_midpoint_is_second_order(backend):
    # Halving the step size reduces the midpoint error by about 4 (second order); amplitude where the error is large.
    vector = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
    reference = backend.Sextupole("S", LENGTH, K2).transfer_matrix(
        backend.Coordinate(vector.copy(), 0.0, 0.0, 0.0), 1.0e-3, "symplectic4")
    errors = [np.max(np.abs(backend.Sextupole("S", LENGTH, K2).transfer_matrix(
        backend.Coordinate(vector.copy(), 0.0, 0.0, 0.0), ds, "midpoint") - reference)) for ds in (0.1, 0.05)]
    assert errors[0] / errors[1] > 3.0


def test_transfer_matrix_none_means_zero_amplitude(backend):
    tmat = backend.Sextupole("S", LENGTH, K2).transfer_matrix(None, DS)
    assert_close(tmat, drift_matrix(LENGTH))


def test_zero_length_is_identity(backend):
    cood, _, _ = backend.Sextupole("S", 0.0, K2).transfer(backend.Coordinate(COORD.copy(), 0.0, 0.0, 3.0e-4), None, None, DS)
    assert_close(cood.vector, COORD)


def test_backends_agree_on_coordinate_transfer():
    py, _, _ = pyegret.Sextupole("S", LENGTH, K2, kick_x=1.0e-4, tilt=0.3).transfer(
        pyegret.Coordinate(COORD.copy(), 0.25, -0.02, 3.0e-4), None, None, DS)
    cpp, _, _ = cppegret.Sextupole("S", LENGTH, K2, kick_x=1.0e-4, tilt=0.3).transfer(
        cppegret.Coordinate(COORD.copy(), 0.25, -0.02, 3.0e-4), None, None, DS)
    assert_close(cpp.vector, py.vector, err_msg="coordinate parity with kick and tilt")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kick_name", list(KICKS))
@pytest.mark.parametrize("strength_name", list(STRENGTHS))
def test_trajectory_checks(method, kick_name, strength_name):
    kick = KICKS[kick_name]
    order = MULTIPOLE_ORDER[method]
    data = {}
    for module in (pyegret, cppegret):
        elem = module.Sextupole("S", LENGTH, STRENGTHS[strength_name], kick_x=kick[0], kick_y=kick[1])
        data[module.__name__] = multipole_trajectory(module, elem, method)
        d = data[module.__name__]
        assert_close(d["end"], d["single"], atol=1.0e-9, err_msg=f"{module.__name__} array end vs transfer")
        assert_close(d["fd_x"], d["disp_x"], atol=TRAJ_FD_ATOL_BY_ORDER[order], err_msg=f"{module.__name__} fd x")
        assert_close(d["fd_y"], d["disp_y"], atol=TRAJ_FD_ATOL_BY_ORDER[order], err_msg=f"{module.__name__} fd y")
        if kick != (0.0, 0.0):
            expected = np.array([0.5 * LENGTH * kick[0], kick[0], 0.5 * LENGTH * kick[1], kick[1]])
            atol = TRAJ_KICK_ATOL_BY_METHOD.get(method, TRAJ_KICK_ATOL_BY_ORDER[order])
            assert_close(d["zero"], expected, atol=atol, err_msg=f"{module.__name__} zero-amplitude kick response")
    assert_close(data["egret.cpp"]["vector"], data["egret.python"]["vector"], atol=1.0e-9, err_msg="trajectory parity")
