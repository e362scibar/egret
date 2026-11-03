"""CI checks for the Envelope class (pytest counterpart of test_envelope_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

COV = np.array([
    [5.0, -1.2, 0.35, 0.08],
    [-1.2, 0.68, -0.11, 0.05],
    [0.35, -0.11, 4.3, 0.9],
    [0.08, 0.05, 0.9, 0.55],
])
T_IN = np.array([[0.02, -0.01], [0.015, 0.018]])
UNCOUPLED = np.zeros((4, 4))
UNCOUPLED[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
UNCOUPLED[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def t_matrix(env) -> np.ndarray:
    attr = env.T_matrix
    return np.asarray(attr() if callable(attr) else attr, dtype=float)


def test_default_constructor_has_zero_coupling(backend):
    env = backend.Envelope()
    assert_close(env.T, np.zeros((2, 2)))
    assert_close(env.tau, 1.0)


def test_constructor_with_cov_and_t_keeps_its_inputs(backend):
    cov = COV.copy()
    env = backend.Envelope(cov=cov, s=10.0, T=T_IN.copy(), psix=0.2, psiy=-0.3)
    cov[0, 0] = 999.0
    assert_close(env.cov[0, 0], COV[0, 0], err_msg="cov follows the input array")
    assert_close(env.s, 10.0)
    assert_close(env.T, T_IN)
    assert_close(env.psix, 0.2)
    assert_close(env.psiy, -0.3)


def test_transfer_is_symplectic_map_of_the_covariance(backend):
    tmat = np.array([[1.0, 1.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]])
    env = backend.Envelope(cov=COV.copy(), s=10.0, T=T_IN.copy(), psix=0.2, psiy=-0.3)
    expected = tmat @ COV @ tmat.T
    env.transfer(tmat, 1.5)
    assert_close(env.cov, expected, atol=1.0e-10)
    assert_close(env.s, 11.5)


def test_t_matrix_is_identity_for_uncoupled_optics(backend):
    # Uncoupled optics: tau = 1 and T = 0, so the 4x4 transformation is the identity.
    env = backend.Envelope(cov=UNCOUPLED.copy(), s=0.0)
    assert_close(t_matrix(env), np.eye(4), atol=1.0e-12)


def test_copy_is_independent(backend):
    env = backend.Envelope(cov=COV.copy(), s=10.0, T=T_IN.copy(), psix=0.2, psiy=-0.3)
    copied = env.copy()
    assert copied is not env
    copied.transfer(np.eye(4), 1.0)
    assert_close(env.s, 10.0, err_msg="original changed by transfer of the copy")


def test_backends_agree_on_transfer_and_t_matrix():
    py = pyegret.Envelope(cov=COV.copy(), s=10.0, T=T_IN.copy(), psix=0.2, psiy=-0.3)
    cpp = cppegret.Envelope(cov=COV.copy(), s=10.0, T=T_IN.copy(), psix=0.2, psiy=-0.3)
    tmat = np.eye(4)
    tmat[0, 1] = 1.5
    py.transfer(tmat, 1.5)
    cpp.transfer(tmat, 1.5)
    assert_close(cpp.cov, py.cov, atol=1.0e-9)
    assert_close(t_matrix(cpp), t_matrix(py), atol=1.0e-9)
