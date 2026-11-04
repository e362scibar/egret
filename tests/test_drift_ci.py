# tests/test_drift_ci.py
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

"""CI checks for the Drift element (pytest counterpart of test_drift_parity.ipynb).

Every check runs on both backends. Results are compared with the analytic drift map and, where the
notebook compares them, between pyegret (`egret.python`) and cppegret (`egret.cpp`).
"""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import METHODS, assert_close, drift_matrix

LENGTH = 1.5
S0 = 0.25
Z0 = -0.02
DELTA0 = 3.0e-4
COORD_VECTOR = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])  # x, x', y, y'
DISP_VECTOR = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])  # eta_x, eta_x', eta_y, eta_y'
COV = np.zeros((4, 4))
COV[0:2, 0:2] = np.array([[5.0, -0.3], [-0.3, 1.09 / 5.0]])  # unit determinant, uncoupled Twiss
COV[2:4, 2:4] = np.array([[3.0, 0.1], [0.1, 1.01 / 3.0]])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def make_coordinate(module, vector: np.ndarray = COORD_VECTOR, s: float = S0, delta: float = DELTA0):
    return module.Coordinate(vector.copy(), s, Z0, delta)


def test_constructor_and_attributes(backend):
    drift = backend.Drift("D", LENGTH, dx=1.0e-3, dy=-2.0e-3, ds=0.1, tilt=0.2, info="straight")
    assert_close(drift.length, LENGTH)
    assert_close(drift.dx, 1.0e-3)
    assert_close(drift.dy, -2.0e-3)
    assert_close(drift.ds, 0.1)
    assert_close(drift.tilt, 0.2)
    assert drift.name == "D"
    assert drift.info == "straight"


def test_copy_is_independent(backend):
    drift = backend.Drift("D", LENGTH, dx=0.1)
    copied = drift.copy()
    assert copied is not drift
    assert_close(copied.dx, 0.1)
    copied.length = 3.0
    assert_close(drift.length, LENGTH, err_msg="original length changed by copy")
    assert_close(copied.length, 3.0)


@pytest.mark.parametrize("method", METHODS)
def test_transfer_matrix_is_analytic_for_every_method(backend, method):
    drift = backend.Drift("D", LENGTH)
    tmat = drift.transfer_matrix(make_coordinate(backend), 0.1, method)
    assert_close(tmat, drift_matrix(LENGTH), err_msg=f"{backend.__name__} {method}")


def test_transfer_matrix_ignores_tilt_and_offsets(backend):
    tmat = backend.Drift("D", LENGTH, dx=1.0e-3, dy=-2.0e-3, tilt=0.4).transfer_matrix()
    assert_close(tmat, drift_matrix(LENGTH))


@pytest.mark.parametrize("endpoint", [True, False])
def test_transfer_matrix_array_layout_and_positions(backend, endpoint):
    tmat_array, s = backend.Drift("D", LENGTH).transfer_matrix_array(ds=0.4, endpoint=endpoint)
    tmat_array = np.asarray(tmat_array)
    assert tmat_array.shape == (len(s), 4, 4)  # N-first layout
    for k, s_val in enumerate(s):
        assert_close(tmat_array[k], drift_matrix(s_val), err_msg=f"s={s_val}")


def test_coordinate_transfer_is_analytic(backend):
    cood, _, _ = backend.Drift("D", LENGTH).transfer(make_coordinate(backend))
    expected = np.array([
        COORD_VECTOR[0] + LENGTH * COORD_VECTOR[1],
        COORD_VECTOR[1],
        COORD_VECTOR[2] + LENGTH * COORD_VECTOR[3],
        COORD_VECTOR[3],
    ])
    assert_close(cood.vector, expected)
    assert_close(cood.s, S0 + LENGTH)
    assert_close(cood.z, Z0)
    assert_close(cood.delta, DELTA0)


def test_element_offsets_do_not_change_coordinate_transfer(backend):
    plain, _, _ = backend.Drift("D", LENGTH).transfer(make_coordinate(backend))
    shifted, _, _ = backend.Drift("D", LENGTH, dx=1.0e-3, dy=-2.0e-3).transfer(make_coordinate(backend))
    assert_close(shifted.vector, plain.vector)


def test_envelope_transfer_is_analytic(backend):
    envelope = backend.Envelope(cov=COV.copy(), s=S0)
    _, out, _ = backend.Drift("D", LENGTH).transfer(make_coordinate(backend), envelope)
    tmat = drift_matrix(LENGTH)
    assert_close(out.cov, tmat @ COV @ tmat.T, atol=1.0e-10)
    assert_close(out.s, S0 + LENGTH)


def test_dispersion_transfer_is_analytic(backend):
    dispersion = backend.Dispersion(DISP_VECTOR.copy(), S0)
    _, _, out = backend.Drift("D", LENGTH).transfer(make_coordinate(backend), None, dispersion)
    assert_close(out.vector, drift_matrix(LENGTH) @ DISP_VECTOR)
    assert_close(out.s, S0 + LENGTH)


def test_coordinate_array_matches_single_transfer_at_exit(backend):
    arr, _, _ = backend.Drift("D", LENGTH).transfer_array(make_coordinate(backend), ds=0.5, endpoint=True)
    single, _, _ = backend.Drift("D", LENGTH).transfer(make_coordinate(backend))
    assert_close(np.asarray(arr.vector)[:, -1], single.vector)
    assert_close(arr.s, S0 + np.array([0.0, 0.5, 1.0, 1.5]))


def test_transfer_from_inside_uses_remaining_length(backend):
    s_inside = 0.5
    remaining = LENGTH - s_inside
    tmat = backend.Drift("D", LENGTH).transfer_matrix_from_s(s_inside, make_coordinate(backend, s=0.0))
    assert_close(tmat, drift_matrix(remaining))
    cood, _, _ = backend.Drift("D", LENGTH).transfer_from_s(s_inside, make_coordinate(backend, s=0.0))
    assert_close(cood.vector, drift_matrix(remaining) @ COORD_VECTOR)


def test_zero_length_drift_is_identity(backend):
    cood, _, _ = backend.Drift("D", 0.0).transfer(make_coordinate(backend))
    assert_close(cood.vector, COORD_VECTOR)


def test_backends_agree_on_transfer_outputs():
    """Direct parity: both backends give the same coordinate, envelope, and dispersion outputs."""
    py_cood, py_env, py_disp = pyegret.Drift("D", LENGTH).transfer(
        make_coordinate(pyegret), pyegret.Envelope(cov=COV.copy(), s=S0),
        pyegret.Dispersion(DISP_VECTOR.copy(), S0))
    cpp_cood, cpp_env, cpp_disp = cppegret.Drift("D", LENGTH).transfer(
        make_coordinate(cppegret), cppegret.Envelope(cov=COV.copy(), s=S0),
        cppegret.Dispersion(DISP_VECTOR.copy(), S0))
    assert_close(cpp_cood.vector, py_cood.vector, err_msg="coordinate parity")
    assert_close(cpp_env.cov, py_env.cov, atol=1.0e-10, err_msg="envelope parity")
    assert_close(cpp_disp.vector, py_disp.vector, err_msg="dispersion parity")


def test_backends_agree_on_transfer_matrix_array():
    py_arr, py_s = pyegret.Drift("D", LENGTH).transfer_matrix_array(ds=0.4, endpoint=True)
    cpp_arr, cpp_s = cppegret.Drift("D", LENGTH).transfer_matrix_array(ds=0.4, endpoint=True)
    assert_close(cpp_s, py_s, err_msg="array positions parity")
    assert_close(cpp_arr, py_arr, err_msg="array matrices parity")

