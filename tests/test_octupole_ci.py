# tests/test_octupole_ci.py
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

"""CI checks for the Octupole element (pytest counterpart of test_octupole_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import (
    METHODS, MULTIPOLE_ORDER, TRAJ_FD_ATOL_BY_ORDER, TRAJ_KICK_ATOL_BY_METHOD, TRAJ_KICK_ATOL_BY_ORDER,
    ALIGNMENT_CASES, aligned_reference, assert_close, drift_matrix, multipole_symplectic_residual,
    multipole_trajectory,
)

LENGTH = 0.2
K3 = 10.0
SYMPLECTIC = ("symplectic1", "symplectic2", "symplectic4")
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
CONV_VECTOR = np.array([0.02, 0.005, -0.01, 0.003])  # amplitude where the octupole error is well above round-off
DS = 0.05
KICKS = {"no kick": (0.0, 0.0), "dipole kick": (1.0e-4, -2.0e-4)}
STRENGTHS = {"focusing": 10.0, "defocusing": -10.0}


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def analytic_strengths(x0: float, y0: float, k3: float, k1: float = 0.0, tilt_quad: float = 0.0, delta: float = 0.0):
    k3_eff, k1_eff = k3 / (1.0 + delta), k1 / (1.0 + delta)
    quad_phase = np.exp(2.0j * tilt_quad)
    k0 = k3_eff * (x0**3 / 6.0 - 0.5 * x0 * y0**2 + 1.0j * (y0**3 / 6.0 - 0.5 * x0**2 * y0)) \
        + k1_eff * quad_phase * (x0 - 1.0j * y0)
    k1_total = k3_eff * (0.5 * (x0**2 - y0**2) - 1.0j * x0 * y0) + k1_eff * quad_phase
    return k0, k1_total


def test_constructor_attributes_copy_and_setter(backend):
    oct_ = backend.Octupole("O", LENGTH, K3, k1=2.0, tilt_quad=0.3, kick_x=1.0e-4)
    assert_close(oct_.k3, K3)
    assert_close(oct_.k1, 2.0)
    assert_close(oct_.tilt_quad, 0.3)
    assert_close(oct_.kick_x, 1.0e-4)
    copied = oct_.copy()
    copied.k3 = 3.0
    assert_close(oct_.k3, K3, err_msg="original k3 changed by copy")
    assert_close(copied.k3, 3.0)


@pytest.mark.parametrize("x0, y0, k1, tilt_quad, delta", [
    (1.0e-3, -5.0e-4, 0.0, 0.0, 0.0),
    (-2.0e-3, 3.0e-3, 0.0, 0.0, 0.0),
    (1.0e-3, -5.0e-4, 2.0, 0.3, 0.0),
    (1.0e-3, -5.0e-4, 2.0, 0.3, 0.01),
])
def test_get_k_matches_analytic_strengths(backend, x0, y0, k1, tilt_quad, delta):
    elem = backend.Octupole("O", LENGTH, K3, k1=k1, tilt_quad=tilt_quad)
    cood = backend.Coordinate(np.array([x0, 0.0, y0, 0.0]), 0.0, 0.0, delta)
    k0, kq = elem.get_k(cood)
    expected_k0, expected_k1 = analytic_strengths(x0, y0, K3, k1, tilt_quad, delta)
    assert_close(k0, expected_k0, err_msg="k0")
    assert_close(kq, expected_k1, err_msg="k1")


@pytest.mark.parametrize("method", METHODS)
def test_zero_amplitude_is_a_drift(backend, method):
    tmat = backend.Octupole("O", LENGTH, K3).transfer_matrix(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), DS, method)
    assert_close(tmat, drift_matrix(LENGTH), err_msg=method)


@pytest.mark.parametrize("tilt_quad", [0.0, 0.3])
def test_zero_octupole_with_quadrupole_matches_quadrupole_element(backend, tilt_quad):
    octupole = backend.Octupole("O", LENGTH, 0.0, k1=2.0, tilt_quad=tilt_quad)
    quadrupole = backend.Quadrupole("Q", LENGTH, 2.0, tilt=tilt_quad)
    zero = backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0)
    assert_close(octupole.transfer_matrix(zero, DS), quadrupole.transfer_matrix(zero, DS), atol=1.0e-9)


@pytest.mark.parametrize("method", METHODS)
def test_transfer_matrix_parity_for_every_method(method):
    py = pyegret.Octupole("O", LENGTH, K3).transfer_matrix(pyegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS, method)
    cpp = cppegret.Octupole("O", LENGTH, K3).transfer_matrix(cppegret.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS, method)
    assert_close(cpp, py, err_msg=method)


@pytest.mark.parametrize("method", SYMPLECTIC)
def test_symplectic_integrators_give_symplectic_matrices(backend, method):
    tmat = backend.Octupole("O", LENGTH, K3).transfer_matrix(backend.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), DS, method)
    assert multipole_symplectic_residual(tmat) < 1.0e-12


def test_symplectic4_error_decreases_with_step_size(backend):
    reference = backend.Octupole("O", LENGTH, K3).transfer_matrix(
        backend.Coordinate(CONV_VECTOR.copy(), 0.0, 0.0, 0.0), 1.0e-3, "symplectic4")
    errors = [np.max(np.abs(backend.Octupole("O", LENGTH, K3).transfer_matrix(
        backend.Coordinate(CONV_VECTOR.copy(), 0.0, 0.0, 0.0), ds, "symplectic4") - reference)) for ds in (0.1, 0.05, 0.025)]
    assert np.all(np.diff(errors) <= 0.0)


def test_midpoint_is_second_order(backend):
    reference = backend.Octupole("O", LENGTH, K3).transfer_matrix(
        backend.Coordinate(CONV_VECTOR.copy(), 0.0, 0.0, 0.0), 1.0e-3, "symplectic4")
    errors = [np.max(np.abs(backend.Octupole("O", LENGTH, K3).transfer_matrix(
        backend.Coordinate(CONV_VECTOR.copy(), 0.0, 0.0, 0.0), ds, "midpoint") - reference)) for ds in (0.1, 0.05)]
    assert errors[0] / errors[1] > 3.0


def test_transfer_matrix_none_means_zero_amplitude(backend):
    assert_close(backend.Octupole("O", LENGTH, K3).transfer_matrix(None, DS), drift_matrix(LENGTH))


def test_zero_length_is_identity(backend):
    cood, _, _ = backend.Octupole("O", 0.0, K3).transfer(backend.Coordinate(COORD.copy(), 0.0, 0.0, 3.0e-4), None, None, DS)
    assert_close(cood.vector, COORD)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kick_name", list(KICKS))
@pytest.mark.parametrize("strength_name", list(STRENGTHS))
def test_trajectory_checks(method, kick_name, strength_name):
    kick = KICKS[kick_name]
    order = MULTIPOLE_ORDER[method]
    data = {}
    for module in (pyegret, cppegret):
        elem = module.Octupole("O", LENGTH, STRENGTHS[strength_name], kick_x=kick[0], kick_y=kick[1])
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


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dx, dy, ds", ALIGNMENT_CASES)
def test_alignment_matches_centred_element(backend, dx, dy, ds, method):
    """An element with offsets equals the centred element applied to the offset coordinate (same integrator)."""
    centred = backend.Octupole("O", LENGTH, K3)

    def centred_map(vec: np.ndarray) -> np.ndarray:
        return centred.transfer(backend.Coordinate(vec.copy(), 0.0, 0.0, 0.0), None, None, DS, method)[0].vector

    aligned = backend.Octupole("O", LENGTH, K3, dx=dx, dy=dy, ds=ds)
    cood, _, _ = aligned.transfer(backend.Coordinate(COORD.copy(), 0.0, 0.0, 0.0), None, None, DS, method)
    assert_close(cood.vector, aligned_reference(centred_map, COORD, dx, dy, ds), atol=1.0e-9,
                 err_msg=f"{method} dx={dx} dy={dy} ds={ds}")
