"""CI checks for the Dipole element (pytest counterpart of test_dipole_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import ALIGNMENT_CASES, METHODS, aligned_matrix, aligned_reference, assert_close, drift_matrix

LENGTH = 1.0
ANGLE = 0.1
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
DS = 0.1
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def sector_matrix(length: float, angle: float, delta: float = 0.0) -> np.ndarray:
    """Sector bend without quadrupole component; x-plane sector map, y-plane drift."""
    rho = (length / angle) * (1.0 + delta)
    phi = angle / (1.0 + delta)
    tmat = np.eye(4)
    tmat[0:2, 0:2] = np.array([[np.cos(phi), rho * np.sin(phi)], [-np.sin(phi) / rho, np.cos(phi)]])
    tmat[2, 3] = length
    return tmat


def sector_dispersion(length: float, angle: float) -> np.ndarray:
    rho = length / angle
    return np.array([rho * (1.0 - np.cos(angle)), np.sin(angle), 0.0, 0.0])


def test_constructor_attributes_rho_and_copy(backend):
    dip = backend.Dipole("B", LENGTH, ANGLE, k1=0.3, dx=1.0e-3, info="bend")
    assert_close(dip.length, LENGTH)
    assert_close(dip.angle, ANGLE)
    assert_close(dip.k1, 0.3)
    assert_close(dip.rho, LENGTH / ANGLE)
    copied = dip.copy()
    assert_close(copied.rho, LENGTH / ANGLE)
    copied.k1 = 1.0
    assert_close(dip.k1, 0.3, err_msg="original k1 changed by copy")


def test_zero_angle_is_rejected(backend):
    with pytest.raises(ValueError):
        backend.Dipole("B", LENGTH, 0.0)


def test_length_setter_updates_rho_and_transfer(backend):
    dip = backend.Dipole("B", LENGTH, ANGLE)
    dip.length = 2.0
    assert_close(dip.rho, 2.0 / ANGLE)
    assert_close(dip.transfer_matrix(), sector_matrix(2.0, ANGLE), err_msg="transfer after length setter")


def test_angle_setter_updates_rho_and_transfer_and_rejects_zero(backend):
    dip = backend.Dipole("B", LENGTH, ANGLE)
    dip.angle = 0.05
    assert_close(dip.rho, LENGTH / 0.05)
    assert_close(dip.transfer_matrix(), sector_matrix(LENGTH, 0.05), err_msg="transfer after angle setter")
    with pytest.raises(ValueError):
        dip.angle = 0.0


@pytest.mark.parametrize("delta", [0.0, 1.0e-3, -2.0e-3])
def test_sector_transfer_matrix_is_analytic_with_momentum(backend, delta):
    tmat = backend.Dipole("B", LENGTH, ANGLE).transfer_matrix(backend.Coordinate(COORD.copy(), S0, Z0, delta), DS)
    assert_close(tmat, sector_matrix(LENGTH, ANGLE, delta), err_msg=f"delta={delta}")


@pytest.mark.parametrize("method", METHODS)
def test_method_does_not_change_the_matrix(backend, method):
    tmat = backend.Dipole("B", LENGTH, ANGLE).transfer_matrix(backend.Coordinate(COORD.copy(), S0, Z0, 0.0), DS, method)
    assert_close(tmat, sector_matrix(LENGTH, ANGLE), err_msg=method)


def test_small_quadrupole_component_recovers_sector_bend(backend):
    tmat = backend.Dipole("B", LENGTH, ANGLE, k1=1.0e-9).transfer_matrix()
    assert_close(tmat, sector_matrix(LENGTH, ANGLE), atol=1.0e-7)


@pytest.mark.parametrize("k1", [0.3, -0.3])
def test_combined_function_parity(k1):
    py = pyegret.Dipole("B", LENGTH, ANGLE, k1=k1).transfer_matrix()
    cpp = cppegret.Dipole("B", LENGTH, ANGLE, k1=k1).transfer_matrix()
    assert_close(cpp, py, atol=1.0e-12, err_msg=f"k1={k1}")


def test_dispersion_is_analytic_and_parity(backend):
    disp = backend.Dipole("B", LENGTH, ANGLE).dispersion(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS)
    assert_close(disp, sector_dispersion(LENGTH, ANGLE))


def test_coordinate_transfer_is_analytic_and_offsets_are_consistent(backend):
    cood, _, _ = backend.Dipole("B", LENGTH, ANGLE).transfer(backend.Coordinate(COORD.copy(), S0, Z0, 0.0))
    assert_close(cood.vector, sector_matrix(LENGTH, ANGLE) @ COORD)
    assert_close(cood.s, S0 + LENGTH)
    py, _, _ = pyegret.Dipole("B", LENGTH, ANGLE, dx=1.0e-3, dy=-2.0e-3, ds=2.0e-3).transfer(
        pyegret.Coordinate(COORD.copy(), S0, Z0, DELTA0))
    cpp, _, _ = cppegret.Dipole("B", LENGTH, ANGLE, dx=1.0e-3, dy=-2.0e-3, ds=2.0e-3).transfer(
        cppegret.Coordinate(COORD.copy(), S0, Z0, DELTA0))
    assert_close(cpp.vector, py.vector, atol=1.0e-12, err_msg="offset coordinate parity")


def test_envelope_transfer_is_analytic(backend):
    _, env, _ = backend.Dipole("B", LENGTH, ANGLE).transfer(
        backend.Coordinate(np.zeros(4), S0, Z0, 0.0), backend.Envelope(cov=COV.copy(), s=S0))
    tmat = sector_matrix(LENGTH, ANGLE)
    assert_close(env.cov, tmat @ COV @ tmat.T, atol=1.0e-10)


@pytest.mark.parametrize("endpoint", [True, False])
def test_transfer_matrix_array_parity_and_layout(endpoint):
    py_arr, py_s = pyegret.Dipole("B", LENGTH, ANGLE).transfer_matrix_array(ds=0.25, endpoint=endpoint)
    cpp_arr, cpp_s = cppegret.Dipole("B", LENGTH, ANGLE).transfer_matrix_array(ds=0.25, endpoint=endpoint)
    assert np.asarray(cpp_arr).shape == (len(cpp_s), 4, 4)
    assert_close(cpp_s, py_s)
    assert_close(cpp_arr, py_arr, atol=1.0e-12)


def test_offset_transfer_from_inside_matches_between_backends():
    """Transfer from a point inside a dipole with element offsets, through the envelope and dispersion."""
    def run(module):
        element = module.Dipole("D", 1.2, 0.25, dx=1.5e-3, dy=-8.0e-4, ds=2.0e-3)
        cood0 = module.Coordinate(np.array([1.0e-3, -2.0e-4, 3.0e-4, 4.0e-4]), 0.2, -0.1, 0.01)
        disp0 = module.Dispersion(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4]), 0.2)
        return element.transfer_from_s(0.35, cood0, module.Envelope(), disp0, ds=0.05)
    py_cood, py_env, py_disp = run(pyegret)
    cpp_cood, cpp_env, cpp_disp = run(cppegret)
    assert_close(cpp_cood.vector, py_cood.vector, atol=1.0e-12)
    assert_close(cpp_env.cov, py_env.cov, atol=1.0e-12)
    assert_close(cpp_disp.vector, py_disp.vector, atol=1.0e-12)


@pytest.mark.parametrize("dx, dy, ds", ALIGNMENT_CASES)
def test_alignment_matches_analytic_map(backend, dx, dy, ds):
    cood, _, _ = backend.Dipole("B", LENGTH, ANGLE, dx=dx, dy=dy, ds=ds).transfer(
        backend.Coordinate(COORD.copy(), S0, Z0, 0.0), None, None, DS)
    expected = aligned_reference(lambda vec: sector_matrix(LENGTH, ANGLE) @ vec, COORD, dx, dy, ds)
    assert_close(cood.vector, expected, err_msg=f"dx={dx} dy={dy} ds={ds}")


@pytest.mark.parametrize("dx, dy, ds", ALIGNMENT_CASES)
def test_alignment_envelope_follows_conjugated_map(backend, dx, dy, ds):
    """The envelope (beta, alpha, emittance) is transported by the matrix with the longitudinal offset."""
    _, env, _ = backend.Dipole("B", LENGTH, ANGLE, dx=dx, dy=dy, ds=ds).transfer(
        backend.Coordinate(np.zeros(4), S0, Z0, 0.0), backend.Envelope(cov=COV.copy(), s=S0), None, DS)
    tmat = aligned_matrix(sector_matrix(LENGTH, ANGLE), ds)
    assert_close(env.cov, tmat @ COV @ tmat.T, atol=1.0e-10, err_msg=f"ds={ds}")

