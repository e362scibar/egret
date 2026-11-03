"""CI checks for the Steering element (pytest counterpart of test_steering_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import METHODS, assert_close, drift_matrix

LENGTH = 1.0
KICK_X = 1.0e-4
KICK_Y = -2.0e-4
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def tilted_kick(kick_x: float, kick_y: float, tilt: float = 0.0, delta: float = 0.0) -> np.ndarray:
    c, s = np.cos(tilt), np.sin(tilt)
    return np.array([[c, -s], [s, c]]) @ np.array([kick_x, kick_y]) / (1.0 + delta)


def kick_vector(length: float, kick_x: float, kick_y: float, tilt: float = 0.0, delta: float = 0.0) -> np.ndarray:
    """Thin-kick map of a uniformly distributed kick: exit offset L/2 * kick and slope kick."""
    kx, ky = tilted_kick(kick_x, kick_y, tilt, delta)
    return np.array([0.5 * length * kx, kx, 0.5 * length * ky, ky])


def test_constructor_attributes_copy_and_kick_setters(backend):
    st = backend.Steering("S", LENGTH, KICK_X, KICK_Y, tilt=0.1)
    assert_close(st.kick_x, KICK_X)
    assert_close(st.kick_y, KICK_Y)
    assert_close(np.asarray(st.kick), [KICK_X, KICK_Y])
    copied = st.copy()
    copied.kick_x = 5.0e-4
    assert_close(st.kick_x, KICK_X, err_msg="original kick changed by copy")


def test_kick_tuple_assignment(backend):
    st = backend.Steering("S", LENGTH)
    st.kick = (4.0e-4, -3.0e-4)
    assert_close(np.asarray(st.kick), [4.0e-4, -3.0e-4])


def test_set_steering_updates_only_given_components(backend):
    st = backend.Steering("S", LENGTH, KICK_X, KICK_Y)
    st.set_steering(kick_y=5.0e-4)
    assert_close(np.asarray(st.kick), [KICK_X, 5.0e-4])


@pytest.mark.parametrize("method", METHODS)
def test_transfer_matrix_is_drift_for_every_method(backend, method):
    tmat = backend.Steering("S", LENGTH, KICK_X, KICK_Y).transfer_matrix(
        backend.Coordinate(COORD.copy(), S0, Z0, DELTA0), 0.1, method)
    assert_close(tmat, drift_matrix(LENGTH))


def test_zero_amplitude_response_is_the_thin_kick(backend):
    cood, _, _ = backend.Steering("S", LENGTH, KICK_X, KICK_Y).transfer(
        backend.Coordinate(np.zeros(4), S0, Z0, 0.0))
    assert_close(cood.vector, kick_vector(LENGTH, KICK_X, KICK_Y))


@pytest.mark.parametrize("tilt", [0.0, 0.3, np.pi / 2])
def test_tilted_kick(backend, tilt):
    cood, _, _ = backend.Steering("S", LENGTH, KICK_X, KICK_Y, tilt=tilt).transfer(
        backend.Coordinate(np.zeros(4), S0, Z0, 0.0))
    assert_close(cood.vector, kick_vector(LENGTH, KICK_X, KICK_Y, tilt=tilt), err_msg=f"tilt={tilt}")


@pytest.mark.parametrize("delta", [1.0e-2, -2.0e-3])
def test_kick_scales_with_momentum(backend, delta):
    cood, _, _ = backend.Steering("S", LENGTH, KICK_X, KICK_Y).transfer(
        backend.Coordinate(np.zeros(4), S0, Z0, delta))
    assert_close(cood.vector, kick_vector(LENGTH, KICK_X, KICK_Y, delta=delta))


def test_coordinate_transfer_with_offset_is_affine(backend):
    cood, _, _ = backend.Steering("S", LENGTH, KICK_X, KICK_Y).transfer(
        backend.Coordinate(COORD.copy(), S0, Z0, 0.0))
    expected = drift_matrix(LENGTH) @ COORD + kick_vector(LENGTH, KICK_X, KICK_Y)
    assert_close(cood.vector, expected)


def test_dispersion_is_minus_the_kick(backend):
    disp = backend.Steering("S", LENGTH, KICK_X, KICK_Y).dispersion(
        backend.Coordinate(np.zeros(4), S0, Z0, 0.0), 0.1)
    assert_close(disp, -kick_vector(LENGTH, KICK_X, KICK_Y))


def test_array_kick_is_quadratic_along_the_magnet(backend):
    """The kick is uniformly distributed: x(s) = x0 + x0' s + kick s^2 / (2 L), x'(s) = x0' + kick s / L."""
    arr, _, _ = backend.Steering("S", LENGTH, KICK_X, KICK_Y).transfer_array(
        backend.Coordinate(COORD.copy(), 0.0, Z0, 0.0), ds=0.05, endpoint=True)
    s = np.asarray(arr.s, dtype=float)
    assert_close(arr.x, COORD[0] + COORD[1] * s + 0.5 * KICK_X * s**2 / LENGTH, atol=1.0e-12)
    assert_close(arr.xp, COORD[1] + KICK_X * s / LENGTH, atol=1.0e-12)


def test_envelope_transfer_is_drift_map(backend):
    _, env, _ = backend.Steering("S", LENGTH, KICK_X, KICK_Y).transfer(
        backend.Coordinate(np.zeros(4), S0, Z0, 0.0), backend.Envelope(cov=COV.copy(), s=S0))
    tmat = drift_matrix(LENGTH)
    assert_close(env.cov, tmat @ COV @ tmat.T, atol=1.0e-10)


def test_zero_length_applies_only_the_kick(backend):
    cood, _, _ = backend.Steering("S", 0.0, KICK_X, KICK_Y).transfer(backend.Coordinate(COORD.copy(), S0, Z0, 0.0))
    assert_close(cood.vector, COORD + np.array([0.0, KICK_X, 0.0, KICK_Y]))


def test_backends_agree_on_outputs():
    py_cood, _, _ = pyegret.Steering("S", LENGTH, KICK_X, KICK_Y, tilt=0.2).transfer(
        pyegret.Coordinate(COORD.copy(), S0, Z0, DELTA0))
    cpp_cood, _, _ = cppegret.Steering("S", LENGTH, KICK_X, KICK_Y, tilt=0.2).transfer(
        cppegret.Coordinate(COORD.copy(), S0, Z0, DELTA0))
    assert_close(cpp_cood.vector, py_cood.vector, err_msg="coordinate parity")
    py_arr, _, _ = pyegret.Steering("S", LENGTH, KICK_X, KICK_Y).transfer_array(
        pyegret.Coordinate(COORD.copy(), 0.0, Z0, 0.0), ds=0.05)
    cpp_arr, _, _ = cppegret.Steering("S", LENGTH, KICK_X, KICK_Y).transfer_array(
        cppegret.Coordinate(COORD.copy(), 0.0, Z0, 0.0), ds=0.05)
    assert_close(cpp_arr.vector, py_arr.vector, err_msg="array parity")
