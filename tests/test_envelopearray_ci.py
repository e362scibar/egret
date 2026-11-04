# tests/test_envelopearray_ci.py
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

"""CI checks for the EnvelopeArray class (pytest counterpart of test_envelopearray_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

N = 5
S_GRID = np.linspace(0.0, 4.0, N)
PSIX = np.linspace(0.0, 0.4, N)
PSIY = np.linspace(0.0, -0.3, N)


def twiss_cov(bx: float, ax: float, by: float, ay: float) -> np.ndarray:
    cov = np.zeros((4, 4))
    cov[0:2, 0:2] = np.array([[bx, -ax], [-ax, (1.0 + ax * ax) / bx]])
    cov[2:4, 2:4] = np.array([[by, -ay], [-ay, (1.0 + ay * ay) / by]])
    return cov


COV = np.stack([twiss_cov(5.0 + 0.5 * np.cos(s), 0.3 * np.sin(s), 3.0 + 0.2 * s, -0.1) for s in S_GRID])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def make(module, **kwargs):
    return module.EnvelopeArray(COV.copy(), S_GRID.copy(), **kwargs)


def test_uncoupled_optics_have_zero_coupling_and_unit_tau(backend):
    arr = backend.EnvelopeArray(COV.copy(), S_GRID.copy())
    assert_close(arr.T, np.zeros((N, 2, 2)), atol=1.0e-12)
    assert_close(arr.tau, np.ones(N), atol=1.0e-12)
    assert_close(arr.bx, COV[:, 0, 0])
    assert_close(arr.gy, COV[:, 3, 3])


def test_given_phases_are_kept(backend):
    arr = backend.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX.copy(), psiy=PSIY.copy())
    assert_close(arr.psix, PSIX)
    assert_close(arr.psiy, PSIY)


def test_computed_phases_agree_between_backends():
    py = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy())
    cpp = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy())
    assert_close(cpp.psix, py.psix, atol=1.0e-6)
    assert_close(cpp.psiy, py.psiy, atol=1.0e-6)


def test_copy_and_append(backend):
    arr = make(backend, psix=PSIX.copy(), psiy=PSIY.copy())
    copied = arr.copy()
    assert copied is not arr
    arr.append(make(backend, psix=PSIX.copy(), psiy=PSIY.copy()))
    assert len(arr) == 2 * N
    assert len(copied) == N


def test_from_s_returns_the_envelope_at_the_point(backend):
    arr = make(backend, psix=PSIX.copy(), psiy=PSIY.copy())
    env = arr.from_s(0.5)
    assert_close(env.s, 0.5)
    assert_close(env.cov, 0.5 * (COV[0] + COV[1]), atol=1.0e-12)


def test_from_s_out_of_range_raises_value_error(backend):
    arr = make(backend)
    with pytest.raises(ValueError):
        arr.from_s(S_GRID[-1] + 0.1)


def test_transport_matches_matrix_map(backend):
    tmat = np.stack([np.eye(4) for _ in range(N)])
    tmat[:, 0, 1] = 1.5
    tmat[:, 2, 3] = 1.5
    env0 = backend.Envelope(cov=twiss_cov(5.0, 0.3, 3.0, -0.1), s=0.0)
    out = backend.EnvelopeArray.transport(env0, tmat, S_GRID.copy())
    for k in range(N):
        expected = tmat[k] @ env0.cov @ tmat[k].T
        assert_close(out.cov[k], expected, atol=1.0e-9, err_msg=f"point {k}")


@pytest.mark.parametrize("build", [
    lambda m: m.EnvelopeArray(COV.copy(), S_GRID[:-1].copy()),
    lambda m: m.EnvelopeArray(COV.copy(), S_GRID[::-1].copy()),
    lambda m: m.EnvelopeArray(COV[:, :3, :3].copy(), S_GRID.copy()),
    lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX[:-1].copy()),
    lambda m: m.EnvelopeArray(COV.copy(), S_GRID.copy(), psiy=PSIY[:-1].copy()),
])
def test_invalid_inputs_raise_value_error(backend, build):
    with pytest.raises(ValueError):
        build(backend)


def test_wrong_t_shape_or_length_raises_value_error(backend):
    arr = make(backend)
    with pytest.raises(ValueError):
        arr.calc_eigenmode(T=np.zeros((N, 3, 3)))
    with pytest.raises(ValueError):
        arr.calc_eigenmode(T=np.zeros((N - 1, 2, 2)))


def test_singular_but_consistent_system_gives_minimum_norm_coupling(backend):
    # Sxx == Syy makes the eigenmode system singular; a right-hand side in its range has a unique minimum-norm solution.
    sxx = np.eye(2)
    cov_round = np.zeros((4, 4))
    cov_round[0:2, 0:2] = sxx
    cov_round[2:4, 2:4] = sxx
    cov_pair = np.stack([cov_round, cov_round])
    arr = backend.EnvelopeArray(cov_pair.copy(), np.array([0.0, 1.0]))
    assert_close(arr.T[0], np.zeros((2, 2)), atol=1.0e-10)


def test_singular_inconsistent_system_raises_value_error(backend):
    cov = np.zeros((4, 4))
    cov[0:2, 0:2] = np.eye(2)
    cov[2:4, 2:4] = np.eye(2)
    cov[0:2, 2:4] = np.diag([0.3, -0.1])
    cov[2:4, 0:2] = cov[0:2, 2:4].T
    pair = np.stack([cov, cov])
    with pytest.raises(ValueError):
        backend.EnvelopeArray(pair.copy(), np.array([0.0, 1.0]))


def test_array_getters_cannot_change_the_object(backend):
    arr = make(backend, psix=PSIX.copy(), psiy=PSIY.copy())
    before = np.array(arr.cov, copy=True)
    for attr in ("cov", "bx", "tau", "T", "s"):
        view = getattr(arr, attr)
        try:
            np.asarray(view).flat[0] = 99.0
        except ValueError:
            continue
        assert_close(np.asarray(arr.cov), before, err_msg=f"writing to {attr} changed the covariance")


def test_backends_agree_on_transport_and_from_s():
    py = pyegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX.copy(), psiy=PSIY.copy())
    cpp = cppegret.EnvelopeArray(COV.copy(), S_GRID.copy(), psix=PSIX.copy(), psiy=PSIY.copy())
    assert_close(cpp.from_s(1.5).cov, py.from_s(1.5).cov, atol=1.0e-12)
    assert_close(cpp.cov, py.cov, atol=1.0e-12)
