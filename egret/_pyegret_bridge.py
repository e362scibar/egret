"""Bridge helpers to call compiled `pyegret` kernels when available.

This module provides small converters between the Python `egret` data
structures and the `pyegret` extension objects, and safe wrappers that
fall back to Python implementations when the extension isn't present.
"""
from __future__ import annotations

import numpy as np
try:
    # Prefer package-local extension (included in wheel as egret/pyegret*.so)
    from . import pyegret as _pyegret_mod
    pyegret = _pyegret_mod
    _PYEGRET_AVAILABLE = True
except Exception:
    try:
        import pyegret as _pyegret_mod
        pyegret = _pyegret_mod
        _PYEGRET_AVAILABLE = True
    except Exception:
        pyegret = None
        _PYEGRET_AVAILABLE = False


def available() -> bool:
    return _PYEGRET_AVAILABLE


def _to_pyegret_coordinate(py_cood) -> object:
    """Convert a Python `Coordinate`-like object to a `pyegret.Coordinate`."""
    c = pyegret.Coordinate()
    # copy vector elements (expect length-4)
    vec = np.asarray(py_cood.vector, dtype=float)
    c.vector = vec
    c.s = float(getattr(py_cood, 's', 0.0))
    c.z = float(getattr(py_cood, 'z', 0.0))
    c.delta = float(getattr(py_cood, 'delta', 0.0))
    return c


def quadrupole_transfer_matrix(length: float, k1: float, tilt: float = 0.0, delta: float = 0.0):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    return pyegret.quadrupole_transfer_matrix(length, k1, tilt, delta)


def quadrupole_transfer_matrix_array(length: float, k1: float, tilt: float = 0.0, delta: float = 0.0, ds: float = 0.1, endpoint: bool = False):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    # pyegret returns (ndarray, list[double])
    return pyegret.quadrupole_transfer_matrix_array(length, k1, tilt, delta, ds, endpoint)


def dipole_transfer_matrix(length: float, angle: float, k1: float = 0.0, delta: float = 0.0):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    return pyegret.dipole_transfer_matrix(length, angle, k1, delta)


def dipole_transfer_matrix_array(length: float, angle: float, k1: float = 0.0, delta: float = 0.0, ds: float = 0.1, endpoint: bool = False):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    return pyegret.dipole_transfer_matrix_array(length, angle, k1, delta, ds, endpoint)


def drift_transfer_matrix(length: float):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    # pyegret does not yet expose a dedicated drift helper; emulate via quadrupole k1=0
    return pyegret.quadrupole_transfer_matrix(length, 0.0, 0.0, 0.0)


def sextupole_transfer_matrix(py_cood, length: float, k2: float, ds: float = 0.1):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    py_cood_conv = _to_pyegret_coordinate(py_cood)
    return pyegret.sextupole_transfer_matrix(py_cood_conv, length, k2, ds)


def sextupole_dispersion(py_cood, length: float, k2: float, ds: float = 0.1):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    py_cood_conv = _to_pyegret_coordinate(py_cood)
    return pyegret.sextupole_dispersion(py_cood_conv, length, k2, ds)


def sextupole_transfer_matrix_array(py_cood, length: float, k2: float, ds: float = 0.1, endpoint: bool = False):
    """Return (ndarray(4,4,N), s_list) from compiled sextupole kernel.

    Accepts a Python Coordinate-like object and converts it to the
    `pyegret.Coordinate` used by the C++ binding.
    """
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    py_cood_conv = _to_pyegret_coordinate(py_cood)
    return pyegret.sextupole_transfer_matrix_array(py_cood_conv, length, k2, ds, endpoint)


def apply_transfer_matrix_array(mats, vecs):
    """Apply stacked transfer matrices to one or many 4-vectors using the compiled helper.

    mats: ndarray shape (4,4,N)
    vecs: ndarray shape (4,) or (4,M) or (M,4)
    Returns ndarray: (4,N) for single vector or (4,M,N) for many vectors.
    """
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    return pyegret.apply_transfer_matrix_array(mats, vecs)
