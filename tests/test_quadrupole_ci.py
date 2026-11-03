"""CI checks for the Quadrupole element (pytest counterpart of test_quadrupole_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import METHODS, assert_close, drift_matrix

LENGTH = 0.5
K1 = 1.2
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
DS = 0.1


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def plane_matrix(length: float, k: float) -> np.ndarray:
    """2x2 analytic map of a thick quadrupole in one plane: focusing for k > 0, defocusing for k < 0."""
    if k == 0.0:
        return np.array([[1.0, length], [0.0, 1.0]])
    if k > 0.0:
        root = np.sqrt(k)
        return np.array([[np.cos(root * length), np.sin(root * length) / root],
                         [-root * np.sin(root * length), np.cos(root * length)]])
    root = np.sqrt(-k)
    return np.array([[np.cosh(root * length), np.sinh(root * length) / root],
                     [root * np.sinh(root * length), np.cosh(root * length)]])


def quad_matrix(length: float, k: float, tilt: float = 0.0) -> np.ndarray:
    """4x4 analytic map: x focused by k, y by -k, rotated by the tilt."""
    tmat = np.eye(4)
    tmat[0:2, 0:2] = plane_matrix(length, k)
    tmat[2:4, 2:4] = plane_matrix(length, -k)
    if tilt == 0.0:
        return tmat
    rot = np.eye(4)
    c, s = np.cos(tilt), np.sin(tilt)
    rot[0:2, 0:2] = [[c, 0.0], [0.0, c]]
    rot[0:2, 2:4] = [[s, 0.0], [0.0, s]]
    rot[2:4, 0:2] = [[-s, 0.0], [0.0, -s]]
    rot[2:4, 2:4] = [[c, 0.0], [0.0, c]]
    return rot.T @ tmat @ rot


def test_constructor_attributes_copy_and_setter(backend):
    quad = backend.Quadrupole("Q", LENGTH, K1, tilt=0.13)
    assert_close(quad.length, LENGTH)
    assert_close(quad.k1, K1)
    assert_close(quad.tilt, 0.13)
    copied = quad.copy()
    copied.k1 = 3.0
    assert_close(quad.k1, K1, err_msg="original k1 changed by copy")
    assert_close(copied.k1, 3.0)


@pytest.mark.parametrize("k1", [K1, -K1, 0.0])
@pytest.mark.parametrize("method", METHODS)
def test_transfer_matrix_is_analytic(backend, k1, method):
    tmat = backend.Quadrupole("Q", LENGTH, k1).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS, method)
    assert_close(tmat, quad_matrix(LENGTH, k1), atol=1.0e-9, err_msg=f"k1={k1} {method}")


def test_strength_scales_with_momentum(backend):
    delta = 0.02
    tmat = backend.Quadrupole("Q", LENGTH, K1).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, delta), DS)
    assert_close(tmat, quad_matrix(LENGTH, K1 / (1.0 + delta)), atol=1.0e-9)


def test_tilt_pi_is_identity_and_pi_over_two_swaps_planes(backend):
    tmat_pi = backend.Quadrupole("Q", LENGTH, K1, tilt=np.pi).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS)
    assert_close(tmat_pi, quad_matrix(LENGTH, K1), atol=1.0e-9, err_msg="tilt pi")
    tmat_half = backend.Quadrupole("Q", LENGTH, K1, tilt=np.pi / 2).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS)
    assert_close(tmat_half, quad_matrix(LENGTH, K1, np.pi / 2), atol=1.0e-9, err_msg="tilt pi/2")


@pytest.mark.parametrize("tilt", [0.0, 0.3, np.pi / 4])
def test_transfer_matrix_parity_with_tilt(tilt):
    zero = pyegret.Coordinate(np.zeros(4), 0.0, 0.0, 0.0)
    py = pyegret.Quadrupole("Q", LENGTH, K1, tilt=tilt).transfer_matrix(zero, DS)
    cpp = cppegret.Quadrupole("Q", LENGTH, K1, tilt=tilt).transfer_matrix(
        cppegret.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS)
    assert_close(cpp, py, atol=1.0e-9, err_msg=f"tilt={tilt}")


def test_transfer_matrix_array_is_n_first_and_partial_length_is_analytic(backend):
    arr, s = backend.Quadrupole("Q", LENGTH, K1).transfer_matrix_array(
        backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), ds=0.1, endpoint=True)
    arr = np.asarray(arr)
    assert arr.shape == (len(s), 4, 4)
    for k, s_val in enumerate(s):
        assert_close(arr[k][0:2, 0:2], plane_matrix(s_val, K1), atol=1.0e-9, err_msg=f"s={s_val}")


def test_coordinate_transfer_parity_and_offsets(backend):
    cood, _, _ = backend.Quadrupole("Q", LENGTH, K1, dx=1.0e-3).transfer(
        backend.Coordinate(COORD.copy(), S0, Z0, DELTA0), None, None, DS)
    assert_close(cood.s, S0 + LENGTH)
    assert_close(cood.z, Z0)
    assert_close(cood.delta, DELTA0)
    py, _, _ = pyegret.Quadrupole("Q", LENGTH, K1, dx=1.0e-3).transfer(
        pyegret.Coordinate(COORD.copy(), S0, Z0, DELTA0), None, None, DS)
    cpp, _, _ = cppegret.Quadrupole("Q", LENGTH, K1, dx=1.0e-3).transfer(
        cppegret.Coordinate(COORD.copy(), S0, Z0, DELTA0), None, None, DS)
    assert_close(cpp.vector, py.vector, err_msg="coordinate parity")


def test_envelope_and_dispersion_transfer_parity():
    cov = np.diag([5.0, 1.0 / 5.0, 3.0, 1.0 / 3.0])
    disp = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])
    py_env, py_disp = pyegret.Quadrupole("Q", LENGTH, K1).transfer(
        pyegret.Coordinate(COORD.copy(), S0, Z0, DELTA0), pyegret.Envelope(cov=cov.copy(), s=S0),
        pyegret.Dispersion(disp.copy(), S0))[1:]
    cpp_env, cpp_disp = cppegret.Quadrupole("Q", LENGTH, K1).transfer(
        cppegret.Coordinate(COORD.copy(), S0, Z0, DELTA0), cppegret.Envelope(cov=cov.copy(), s=S0),
        cppegret.Dispersion(disp.copy(), S0))[1:]
    assert_close(cpp_env.cov, py_env.cov, atol=1.0e-9, err_msg="envelope parity")
    assert_close(cpp_disp.vector, py_disp.vector, atol=1.0e-9, err_msg="dispersion parity")


def test_zero_length_is_identity(backend):
    cood, _, _ = backend.Quadrupole("Q", 0.0, K1).transfer(backend.Coordinate(COORD.copy(), S0, Z0, DELTA0), None, None, DS)
    assert_close(cood.vector, COORD)


def test_kick_does_not_change_the_transfer_matrix(backend):
    plain = backend.Quadrupole("Q", LENGTH, K1).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS)
    kicked = backend.Quadrupole("Q", LENGTH, K1, kick_x=1.0e-4, kick_y=-2.0e-4).transfer_matrix(
        backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS)
    assert_close(kicked, plain, atol=1.0e-9)


def test_kick_properties_and_set_steering(backend):
    quad = backend.Quadrupole("Q", LENGTH, K1, kick_x=1.0e-4)
    assert_close(quad.kick_x, 1.0e-4)
    quad.set_steering(kick_y=5.0e-4)
    assert_close(quad.kick_x, 1.0e-4, err_msg="set_steering with None left kick_x unchanged")
    assert_close(quad.kick_y, 5.0e-4)


def test_backends_agree_on_kicked_transfer():
    py, _, _ = pyegret.Quadrupole("Q", LENGTH, K1, kick_x=1.0e-4, kick_y=-2.0e-4).transfer(
        pyegret.Coordinate(COORD.copy(), S0, Z0, DELTA0), None, None, DS)
    cpp, _, _ = cppegret.Quadrupole("Q", LENGTH, K1, kick_x=1.0e-4, kick_y=-2.0e-4).transfer(
        cppegret.Coordinate(COORD.copy(), S0, Z0, DELTA0), None, None, DS)
    assert_close(cpp.vector, py.vector, atol=1.0e-9, err_msg="kicked coordinate parity")
