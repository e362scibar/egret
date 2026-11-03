"""CI checks for the Dispersion class (pytest counterpart of test_dispersion_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

VECTOR = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def test_default_constructor_is_zero(backend):
    disp = backend.Dispersion()
    assert_close(disp.vector, np.zeros(4))
    assert_close(disp.s, 0.0)


def test_components_and_position(backend):
    disp = backend.Dispersion(VECTOR.copy(), 0.2)
    assert_close([disp.x, disp.xp, disp.y, disp.yp], VECTOR)
    assert_close(disp.s, 0.2)


def test_constructor_copies_its_input(backend):
    vector = VECTOR.copy()
    disp = backend.Dispersion(vector, 0.2)
    vector[:] = 999.0
    assert_close(disp.vector, VECTOR)


def test_copy_is_independent(backend):
    disp = backend.Dispersion(VECTOR.copy(), 0.2)
    copied = disp.copy()
    assert copied is not disp
    copied.x = 1.0
    assert_close(disp.x, VECTOR[0], err_msg="original changed by copy")
    assert_close(copied.x, 1.0)


def test_backends_agree():
    py = pyegret.Dispersion(VECTOR.copy(), 0.2)
    cpp = cppegret.Dispersion(VECTOR.copy(), 0.2)
    assert_close(cpp.vector, py.vector)
    assert_close(cpp.s, py.s)
    assert_close(cpp.copy().vector, py.copy().vector)
