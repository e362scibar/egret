"""CI checks for Lattice and Ring (pytest counterpart of test_lattice_parity.ipynb)."""

from __future__ import annotations

import numpy as np
import pytest

import egret.cpp as cppegret
import egret.python as pyegret

from _ci_helpers import assert_close

BEND_ANGLE = np.pi / 8
K_OUTER = -1.8  # outer quadrupoles: horizontally defocusing
K_CENTER = 2.1  # centre quadrupole: horizontally focusing
ENERGY = 1.0
COORD = np.array([1.0e-3, 2.0e-4, -5.0e-4, 1.0e-4])
LENGTH = 4.5  # sum of the element lengths
ANGLE = 2.0 * BEND_ANGLE  # two bends


@pytest.fixture(params=[pyegret, cppegret], ids=["pyegret", "cppegret"])
def backend(request):
    return request.param


def cell_elements(module) -> list:
    """Synthetic double-bend cell (not an achromat): QO1 - drift - B - drift - QC - drift - B - drift - QO2."""
    return [
        module.Drift("d0", 0.4),
        module.Quadrupole("QO1", 0.3, K_OUTER),
        module.Drift("d1", 0.3),
        module.Dipole("B1", 1.0, BEND_ANGLE),
        module.Drift("d2", 0.1),
        module.Quadrupole("QC", 0.3, K_CENTER),
        module.Drift("d3", 0.1),
        module.Dipole("B2", 1.0, BEND_ANGLE),
        module.Drift("d4", 0.3),
        module.Quadrupole("QO2", 0.3, K_OUTER),
        module.Drift("d5", 0.4),
    ]


def zero(module):
    return module.Coordinate(np.zeros(4), 0.0, 0.0, 0.0)


def product_of_elements(module, elements, ds: float = 0.05) -> np.ndarray:
    tmat = np.eye(4)
    for elem in elements:
        tmat = elem.transfer_matrix(zero(module), ds) @ tmat
    return tmat


def tune_from_trace(block: np.ndarray) -> float:
    return float(np.arccos(np.trace(block) / 2.0) / (2.0 * np.pi))


@pytest.fixture(scope="module")
def reference_matrix():
    """Cell matrix from the pyegret elements, used as the analytic reference for both backends."""
    return product_of_elements(pyegret, cell_elements(pyegret))


def test_length_angle_and_copy(backend):
    lat = backend.Lattice("cell", cell_elements(backend))
    assert_close(lat.length, LENGTH)
    assert_close(lat.angle, ANGLE)
    copied = lat.copy()
    assert_close(copied.length, LENGTH)
    assert copied is not lat


def test_transfer_matrix_is_ordered_product_of_elements(backend):
    lat = backend.Lattice("cell", cell_elements(backend))
    assert_close(lat.transfer_matrix(zero(backend), 0.05), product_of_elements(backend, cell_elements(backend)),
                 atol=1.0e-12, err_msg="composition")


def test_nested_sub_lattices_compose_to_flat_lattice(backend, reference_matrix):
    elements = cell_elements(backend)
    nested = backend.Lattice("cell", [backend.Lattice("first", elements[:6]), backend.Lattice("second", elements[6:])])
    assert_close(nested.transfer_matrix(zero(backend), 0.05), reference_matrix, atol=1.0e-12)


def test_array_end_equals_single_transfer(backend):
    lat = backend.Lattice("cell", cell_elements(backend))
    start = backend.Coordinate(COORD.copy(), 0.0, 0.0, 0.0)
    arr, _, _ = lat.transfer_array(start, ds=0.1, endpoint=True)
    single, _, _ = lat.transfer(backend.Coordinate(COORD.copy(), 0.0, 0.0, 0.0))
    assert_close(np.asarray(arr.vector)[:, -1], single.vector, atol=1.0e-12)


def test_ring_tunes_equal_trace_tunes(backend, reference_matrix):
    ring = backend.Ring("R", cell_elements(backend), ENERGY)
    assert ring.update()
    assert_close([ring.tune_x, ring.tune_y],
                 [tune_from_trace(reference_matrix[0:2, 0:2]), tune_from_trace(reference_matrix[2:4, 2:4])],
                 atol=1.0e-9)


def test_periodic_dispersion_reproduces_after_one_turn(backend):
    ring = backend.Ring("R", cell_elements(backend), ENERGY)
    assert ring.update()
    disp0 = backend.Dispersion(np.asarray(ring.disp0.vector, dtype=float).copy(), 0.0)
    lat = backend.Lattice("cell", cell_elements(backend))
    out = lat.transfer(backend.Coordinate(np.zeros(4), 0.0, 0.0, 0.0), None, disp0, 0.05)[2]
    assert_close(out.vector, disp0.vector, atol=1.0e-12, err_msg="dispersion periodicity")
    assert_close(np.asarray(ring.cood0.vector), np.zeros(4), atol=1.0e-12, err_msg="closed orbit")


def test_backends_agree_on_matrix_tunes_and_dispersion():
    py_lat = pyegret.Lattice("cell", cell_elements(pyegret))
    cpp_lat = cppegret.Lattice("cell", cell_elements(cppegret))
    assert_close(cpp_lat.transfer_matrix(zero(cppegret), 0.05), py_lat.transfer_matrix(zero(pyegret), 0.05),
                 atol=1.0e-12, err_msg="matrix parity")
    py_ring = pyegret.Ring("R", cell_elements(pyegret), ENERGY)
    cpp_ring = cppegret.Ring("R", cell_elements(cppegret), ENERGY)
    assert py_ring.update() and cpp_ring.update()
    assert_close([cpp_ring.tune_x, cpp_ring.tune_y], [py_ring.tune_x, py_ring.tune_y], atol=1.0e-9)
    assert_close(cpp_ring.disp0.vector, py_ring.disp0.vector, atol=1.0e-9)
