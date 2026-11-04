# tests/_ci_helpers.py
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

"""Shared constants and assertion helpers for the CI pytest suites (`test_*_ci.py`).

The notebooks in this directory keep their own copies of these values for plotting and exploration; the CI
tests import them from here so that tolerances are defined in one place.
"""

from __future__ import annotations

import numpy as np

ATOL = 1.0e-12
METHODS = ("midpoint", "rk4", "symplectic1", "symplectic2", "symplectic4")


def assert_close(actual, expected, atol: float = ATOL, err_msg: str = "") -> None:
    """Assert that two arrays (or scalars) agree within an absolute tolerance."""
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=0.0, err_msg=err_msg)


def drift_matrix(length: float) -> np.ndarray:
    """Analytic transfer matrix of a drift of the given length."""
    tmat = np.eye(4)
    tmat[0, 1] = length
    tmat[2, 3] = length
    return tmat


# ---------------------------------------------------------------------------------------------------------------------
# Alignment helpers: element offsets (dx, dy) and longitudinal offset (ds), shared by the CI suites.
# An element with offsets is a drift of ds, then the centred map applied to the coordinate relative to the offset,
# then the offset restored, then a drift of -ds.
# ---------------------------------------------------------------------------------------------------------------------

ALIGNMENT_CASES = (
    (1.0e-3, 0.0, 0.0),
    (0.0, -2.0e-3, 0.0),
    (0.0, 0.0, 3.0e-2),
    (1.5e-3, -8.0e-4, -2.0e-2),
)  # (dx, dy, ds) in metres


def aligned_reference(centred_map, vector: np.ndarray, dx: float, dy: float, ds: float) -> np.ndarray:
    """Expected exit vector of an element with offsets (dx, dy) and longitudinal offset ds.

    Args:
        centred_map: Callable mapping an entrance vector to the exit vector of the centred element.
        vector: Entrance coordinate (x, x', y, y').
        dx: Horizontal offset [m].
        dy: Vertical offset [m].
        ds: Longitudinal offset [m].
    """
    offset = np.array([dx, 0.0, dy, 0.0])
    return drift_matrix(-ds) @ (centred_map(drift_matrix(ds) @ vector - offset) + offset)


def aligned_matrix(tmat: np.ndarray, ds: float) -> np.ndarray:
    """Transfer matrix of a centred linear element after a longitudinal offset ds (drift, map, drift -ds)."""
    return drift_matrix(-ds) @ tmat @ drift_matrix(ds)


# ---------------------------------------------------------------------------------------------------------------------
# Multipole (sextupole / octupole) helpers, shared by the CI suites.
# ---------------------------------------------------------------------------------------------------------------------

MULTIPOLE_ORDER = {"midpoint": 2, "rk4": 4, "symplectic1": 1, "symplectic2": 2, "symplectic4": 4}
# Tolerances per order of accuracy (about 10x the largest error measured in the notebooks, both backends).
TRAJ_FD_ATOL_BY_ORDER = {1: 2.0e-6, 2: 1.0e-8, 4: 1.0e-8}
TRAJ_KICK_ATOL_BY_ORDER = {1: 5.0e-5, 2: 1.0e-9, 4: 1.0e-9}
# midpoint and rk4 build the quadrupole from an offset, so their zero-amplitude kick response has a floor of ~1e-7.
TRAJ_KICK_ATOL_BY_METHOD = {"midpoint": 2.0e-7, "rk4": 2.0e-7}
TRAJ_DELTA = 1.0e-4  # central finite difference in the momentum deviation
TRAJ_DS = 0.05
TRAJ_OFFSET = (0.002, 0.0, -0.001, 0.0)
TRAJ_DISP0 = (0.001, 0.0, 0.001, 0.0)
SYMPLECTIC_J = np.zeros((4, 4))
SYMPLECTIC_J[0, 1], SYMPLECTIC_J[1, 0], SYMPLECTIC_J[2, 3], SYMPLECTIC_J[3, 2] = 1.0, -1.0, 1.0, -1.0


def multipole_trajectory(module, elem, method: str):
    """Transfer an offset particle with a dispersion through a multipole as an array; return the data for the checks."""
    disp0 = module.Dispersion(np.array(TRAJ_DISP0))
    offset = np.array(TRAJ_OFFSET)
    disp = np.array(TRAJ_DISP0)
    arr, _, disp_arr = elem.transfer_array(module.Coordinate(offset.copy()), ds=TRAJ_DS, disp0=disp0, method=method)
    arr_plus, _, _ = elem.transfer_array(module.Coordinate(offset + disp * TRAJ_DELTA, 0.0, 0.0, TRAJ_DELTA),
                                         ds=TRAJ_DS, method=method)
    arr_minus, _, _ = elem.transfer_array(module.Coordinate(offset - disp * TRAJ_DELTA, 0.0, 0.0, -TRAJ_DELTA),
                                          ds=TRAJ_DS, method=method)
    single, _, _ = elem.transfer(module.Coordinate(offset.copy()), disp0=disp0, ds=TRAJ_DS, method=method)
    zero, _, _ = elem.transfer(module.Coordinate(np.zeros(4)), ds=TRAJ_DS, method=method)
    return {
        "vector": np.asarray(arr.vector, dtype=float),
        "end": np.asarray(arr.vector, dtype=float)[:, -1],
        "single": np.asarray(single.vector, dtype=float),
        "fd_x": (np.asarray(arr_plus.x) - np.asarray(arr_minus.x)) / (2.0 * TRAJ_DELTA),
        "fd_y": (np.asarray(arr_plus.y) - np.asarray(arr_minus.y)) / (2.0 * TRAJ_DELTA),
        "disp_x": np.asarray(disp_arr.x, dtype=float),
        "disp_y": np.asarray(disp_arr.y, dtype=float),
        "zero": np.asarray(zero.vector, dtype=float),
    }


def multipole_symplectic_residual(tmat: np.ndarray) -> float:
    return float(np.max(np.abs(tmat.T @ SYMPLECTIC_J @ tmat - SYMPLECTIC_J)))
