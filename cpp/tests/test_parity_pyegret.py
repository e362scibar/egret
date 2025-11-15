import os
import sys
import numpy as np

# Ensure repo root and built extension are importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
build_dir = os.path.join(repo_root, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, repo_root)

import importlib
import importlib.util
import glob
import sys
import os


def load_pyegret():
    # Prefer normal import; if the local package shadows the installed extension,
    # search sys.path entries for an `egret/pyegret*.so` and load it directly.
    try:
        return importlib.import_module('egret.pyegret')
    except ModuleNotFoundError:
        for p in list(sys.path):
            candidate_dir = os.path.join(p, 'egret')
            if not os.path.isdir(candidate_dir):
                continue
            matches = glob.glob(os.path.join(candidate_dir, 'pyegret*.so'))
            if not matches:
                # some systems may use different suffixes
                matches = glob.glob(os.path.join(candidate_dir, 'pyegret*.cpython-310-*.so'))
            if matches:
                path = matches[0]
                spec = importlib.util.spec_from_file_location('egret.pyegret', path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        # re-raise original error if not found
        raise


pyegret = load_pyegret()


def test_quadrupole_parity():
    from egret.quadrupole import Quadrupole as PyQuadrupole
    length = 0.5
    k1 = 1.2
    tilt = 0.13
    delta = 0.0
    q_py = PyQuadrupole('Q', length, k1, tilt=tilt)
    t_py = q_py.transfer_matrix()
    t_cpp = pyegret.quadrupole_transfer_matrix(length, k1, tilt, delta)
    assert np.allclose(t_py, t_cpp, atol=1e-12, rtol=1e-9)


def test_dipole_parity():
    from egret.dipole import Dipole
    length = 1.0
    angle = 0.2
    k1 = 0.0
    delta = 0.0
    d_py = Dipole('D', length, angle, k1=k1)
    t_py = d_py.transfer_matrix()
    t_cpp = pyegret.dipole_transfer_matrix(length, angle, k1, delta)
    assert np.allclose(t_py, t_cpp, atol=1e-12, rtol=1e-9)


def test_sextupole_parity_and_array():
    from egret.sextupole import Sextupole
    from egret.coordinate import Coordinate
    length = 0.2
    k2 = 10.0
    ds = 0.01
    S = Sextupole('S', length, k2)
    co = Coordinate(np.zeros(4), 0.0, 0.0, 0.0)

    # single transfer matrix parity
    t_py = S.transfer_matrix(co, ds=ds)
    c2 = pyegret.Coordinate(); c2.vector = np.zeros(4); c2.s = 0.0; c2.z = 0.0; c2.delta = 0.0
    t_cpp = pyegret.sextupole_transfer_matrix(c2, length, k2, ds)
    assert np.allclose(t_py, t_cpp, atol=1e-10)

    # transfer matrix array parity
    tarr_py, s_py = S.transfer_matrix_array(co, ds=ds, endpoint=False)
    tarr_cpp, s_cpp = pyegret.sextupole_transfer_matrix_array(c2, length, k2, ds, False)
    assert tarr_py.shape == tarr_cpp.shape
    assert np.allclose(tarr_py, tarr_cpp, atol=1e-10)
    # s vectors may be produced with slightly different step rounding; ensure same length
    s_cpp_arr = np.asarray(s_cpp, dtype=float)
    assert len(s_py) == len(s_cpp_arr)
    # check mean step size is comparable (allow generous tolerance because step rounding differs)
    step_py = float(np.mean(np.diff(s_py)))
    step_cpp = float(np.mean(np.diff(s_cpp_arr)))
    assert np.isclose(step_py, step_cpp, rtol=0.2, atol=1e-6)


def test_apply_transfer_matrix_array_matches_numpy():
    from egret.quadrupole import Quadrupole
    from egret.coordinate import Coordinate
    q = Quadrupole('Q', 0.5, 1.2, tilt=0.13)
    co = Coordinate(np.zeros(4), 0.0, 0.0, 0.0)
    tmat, s = q.transfer_matrix_array(co, ds=0.1, endpoint=False)
    out_cpp = pyegret.apply_transfer_matrix_array(tmat, co.vector)
    out_np = np.matmul(tmat.transpose(2, 0, 1), co.vector).T
    assert out_cpp.shape == out_np.shape
    assert np.allclose(out_cpp, out_np, atol=1e-12)
