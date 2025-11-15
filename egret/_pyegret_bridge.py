"""Bridge helpers to call compiled `pyegret` kernels when available.

This module provides small converters between the Python `egret` data
structures and the `pyegret` extension objects, and safe wrappers that
fall back to Python implementations when the extension isn't present.
"""
from __future__ import annotations

import numpy as np
# Try possible locations for the compiled extension.
# 1) the extension installed inside the `egret` package as `egret.pyegret`
# 2) a top-level module `pyegret` (rare)
# 3) a package-local relative import (development layout)
def _try_load_installed_egret_so():
    """Attempt to locate an installed egret/pyegret*.so in site-packages and
    load it as module `egret.pyegret` via ExtensionFileLoader. Returns the
    loaded module or None.
    """
    import importlib.machinery
    import importlib.util
    import glob
    import sysconfig
    import os
    import sys

    candidates = []
    try:
        candidates.append(sysconfig.get_paths()['purelib'])
    except Exception:
        pass
    # also include any sys.path entries that look like site-packages
    for p in sys.path:
        if p and 'site-packages' in p and p not in candidates:
            candidates.append(p)

    for base in candidates:
        pkg_dir = os.path.join(base, 'egret')
        if not os.path.isdir(pkg_dir):
            continue
        # look for pyegret shared object files
        patterns = [os.path.join(pkg_dir, 'pyegret*.so'), os.path.join(pkg_dir, 'pyegret.*.so')]
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                so = matches[0]
                try:
                    loader = importlib.machinery.ExtensionFileLoader('egret.pyegret', so)
                    spec = importlib.util.spec_from_loader('egret.pyegret', loader)
                    mod = importlib.util.module_from_spec(spec)
                    loader.exec_module(mod)
                    return mod
                except Exception:
                    continue
    return None


# First, try to load an egret-packaged shared object from site-packages
_pyegret_mod = _try_load_installed_egret_so()
if _pyegret_mod is not None:
    pyegret = _pyegret_mod
    _PYEGRET_AVAILABLE = True
else:
    try:
        import pyegret as _pyegret_mod
        pyegret = _pyegret_mod
        _PYEGRET_AVAILABLE = True
    except Exception:
        try:
            from . import pyegret as _pyegret_mod
            pyegret = _pyegret_mod
            _PYEGRET_AVAILABLE = True
        except Exception:
            pyegret = None
            _PYEGRET_AVAILABLE = False


def available() -> bool:
    return _PYEGRET_AVAILABLE


# If the compiled module is available, export commonly-used classes into
# this bridge module namespace and adjust their __module__ so that
# pickling/unpickling works even when the installed `egret` package is
# shadowed by a local development package on `sys.path`.
if _PYEGRET_AVAILABLE:
    for _name in ("Quadrupole", "Dipole", "Sextupole", "Envelope", "Dispersion", "Coordinate", "CoordinateArray"):
        _obj = getattr(pyegret, _name, None)
        if _obj is not None:
            # expose the class/object at module level so `from egret._pyegret_bridge import Quadrupole` works
            globals()[_name] = _obj
            try:
                # set the __module__ to this module so pickle imports this module path
                _obj.__module__ = __name__
            except Exception:
                pass

    # Register simple reducers for element wrapper classes so pickle can
    # reconstruct instances using the bridge factories even when import
    # paths differ between installed and local development copies.
    def _register_reducers():
        mapping = {
            'Quadrupole': ('length', 'k1', 'tilt'),
            'Dipole': ('length', 'angle', 'k1'),
            'Sextupole': ('length', 'k2', 'dx', 'dy', 'ds'),
        }
        for nm, fields in mapping.items():
            cls = getattr(pyegret, nm, None)
            if cls is None:
                continue
            def _make_reduce(nm, fields):
                def _reduce(self):
                    vals = tuple(getattr(self, f) for f in fields)
                    return (globals()[nm], vals)
                return _reduce
            try:
                setattr(cls, '__reduce__', _make_reduce(nm, fields))
            except Exception:
                pass

    _register_reducers()


def _to_pyegret_coordinate(py_cood) -> object:
    """Convert a Python `Coordinate`-like object to a `pyegret.Coordinate`."""
    if pyegret is None:
        raise ImportError('pyegret not available')
    c = pyegret.Coordinate()
    # copy vector elements (expect length-4)
    vec = np.asarray(py_cood.vector, dtype=float)
    c.vector = vec
    c.s = float(getattr(py_cood, 's', 0.0))
    c.z = float(getattr(py_cood, 'z', 0.0))
    c.delta = float(getattr(py_cood, 'delta', 0.0))
    return c


def make_envelope(beta_x, beta_y, s):
    """Construct an `Envelope` object backed by the compiled extension when available.

    Returns a `pyegret.Envelope` when compiled helpers are present, otherwise
    returns a simple tuple `(beta_x, beta_y, s)`.
    """
    bx = np.asarray(beta_x, dtype=float)
    by = np.asarray(beta_y, dtype=float)
    ss = np.asarray(s, dtype=float)
    if not _PYEGRET_AVAILABLE:
        return (bx, by, ss)
    return pyegret.Envelope(bx, by, ss)


def make_dispersion(eta_x, eta_y, s):
    """Construct a `Dispersion` object backed by the compiled extension when available.

    Returns a `pyegret.Dispersion` when compiled helpers are present, otherwise
    returns a simple tuple `(eta_x, eta_y, s)`.
    """
    ex = np.asarray(eta_x, dtype=float)
    ey = np.asarray(eta_y, dtype=float)
    ss = np.asarray(s, dtype=float)
    if not _PYEGRET_AVAILABLE:
        return (ex, ey, ss)
    return pyegret.Dispersion(ex, ey, ss)


def Quadrupole(length: float, k1: float, tilt: float = 0.0):
    """Factory for the compiled `Quadrupole` wrapper when available.

    When the compiled extension is not present this raises ImportError.
    """
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    # Return a small Python shim that holds the compiled object and
    # preserves constructor args for pickling and subclassability.
    class _QuadShim:
        def __init__(self, length, k1, tilt):
            self._inner = pyegret.Quadrupole(length, k1, tilt)
            self.length = float(length)
            self.k1 = float(k1)
            self.tilt = float(tilt)
        def __getattr__(self, name):
            return getattr(self._inner, name)
        def __reduce__(self):
            return (Quadrupole, (self.length, self.k1, self.tilt))
    return _QuadShim(length, k1, tilt)


def Dipole(length: float, angle: float, k1: float = 0.0):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    class _DipShim:
        def __init__(self, length, angle, k1):
            self._inner = pyegret.Dipole(length, angle, k1)
            self.length = float(length)
            self.angle = float(angle)
            self.k1 = float(k1)
        def __getattr__(self, name):
            return getattr(self._inner, name)
        def __reduce__(self):
            return (Dipole, (self.length, self.angle, self.k1))
    return _DipShim(length, angle, k1)


def Sextupole(length: float, k2: float, dx: float = 0.0, dy: float = 0.0, ds: float = 0.1):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    class _SextShim:
        def __init__(self, length, k2, dx, dy, ds):
            self._inner = pyegret.Sextupole(length, k2, dx, dy, ds)
            self.length = float(length)
            self.k2 = float(k2)
            self.dx = float(dx)
            self.dy = float(dy)
            self.ds = float(ds)
        def __getattr__(self, name):
            return getattr(self._inner, name)
        def __reduce__(self):
            return (Sextupole, (self.length, self.k2, self.dx, self.dy, self.ds))
    return _SextShim(length, k2, dx, dy, ds)


def quadrupole_transfer_matrix(length: float, k1: float, tilt: float = 0.0, delta: float = 0.0):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    return pyegret.quadrupole_transfer_matrix(length, k1, tilt, delta)


def quadrupole_transfer_matrix_array(length: float, k1: float, tilt: float = 0.0, delta: float = 0.0, ds: float = 0.1, endpoint: bool = True):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    # pyegret returns (ndarray, list[double])
    return pyegret.quadrupole_transfer_matrix_array(length, k1, tilt, delta, ds, endpoint)


def dipole_transfer_matrix(length: float, angle: float, k1: float = 0.0, delta: float = 0.0):
    if not _PYEGRET_AVAILABLE:
        raise ImportError('pyegret not available')
    return pyegret.dipole_transfer_matrix(length, angle, k1, delta)


def dipole_transfer_matrix_array(length: float, angle: float, k1: float = 0.0, delta: float = 0.0, ds: float = 0.1, endpoint: bool = True):
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


def sextupole_transfer_matrix_array(py_cood, length: float, k2: float, ds: float = 0.1, endpoint: bool = True):
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
