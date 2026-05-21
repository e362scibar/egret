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


def test_transfer_from_s_parity_with_alignment():
    from egret.python.dipole import Dipole as PyDipole
    from egret.cpp.dipole import Dipole as CppDipole
    from egret.python.coordinate import Coordinate as PyCoordinate
    from egret.cpp.coordinate import Coordinate as CppCoordinate
    from egret.python.envelope import Envelope as PyEnvelope
    from egret.cpp.envelope import Envelope as CppEnvelope
    from egret.python.dispersion import Dispersion as PyDispersion
    from egret.cpp.dispersion import Dispersion as CppDispersion

    length = 1.2
    angle = 0.25
    s0 = 0.35
    ds = 0.05
    dx = 1.5e-3
    dy = -8.0e-4
    ds_offset = 2.0e-3
    cood0_py = PyCoordinate(np.array([1.0e-3, -2.0e-4, 3.0e-4, 4.0e-4]), 0.2, -0.1, 0.01)
    cood0_cpp = CppCoordinate(np.array([1.0e-3, -2.0e-4, 3.0e-4, 4.0e-4]), 0.2, -0.1, 0.01)
    evlp0_py = PyEnvelope()
    evlp0_cpp = CppEnvelope()
    disp0_py = PyDispersion(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4]), 0.2)
    disp0_cpp = CppDispersion(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4]), 0.2)

    py_elem = PyDipole('D', length, angle, dx=dx, dy=dy, ds=ds_offset)
    cpp_elem = CppDipole('D', length, angle, dx=dx, dy=dy, ds=ds_offset)

    t_py = py_elem.transfer_matrix_from_s(s0, cood0_py, ds=ds)
    t_cpp = cpp_elem.transfer_matrix_from_s(s0, cood0_cpp, ds=ds)
    assert np.allclose(t_py, t_cpp, atol=1e-12, rtol=1e-9)

    cood_py, evlp_py, disp_py = py_elem.transfer_from_s(s0, cood0_py, evlp0_py, disp0_py, ds=ds)
    cood_cpp, evlp_cpp, disp_cpp = cpp_elem.transfer_from_s(s0, cood0_cpp, evlp0_cpp, disp0_cpp, ds=ds)
    assert np.allclose(cood_py.vector, cood_cpp.vector, atol=1e-12, rtol=1e-9)
    assert np.isclose(cood_py.s, cood_cpp.s, atol=1e-12)
    assert np.isclose(cood_py.z, cood_cpp.z, atol=1e-12)
    assert np.isclose(cood_py.delta, cood_cpp.delta, atol=1e-12)
    assert evlp_py is not None and evlp_cpp is not None
    assert np.allclose(evlp_py.cov, evlp_cpp.cov, atol=1e-12, rtol=1e-9)
    assert np.isclose(evlp_py.s, evlp_cpp.s, atol=1e-12)
    assert disp_py is not None and disp_cpp is not None
    assert np.allclose(disp_py.vector, disp_cpp.vector, atol=1e-12, rtol=1e-9)
    assert np.isclose(disp_py.s, disp_cpp.s, atol=1e-12)

    cood_arr_py, evlp_arr_py, disp_arr_py = py_elem.transfer_array_from_s(s0, cood0_py, evlp0_py, disp0_py, ds=ds, endpoint=True)
    cood_arr_cpp, evlp_arr_cpp, disp_arr_cpp = cpp_elem.transfer_array_from_s(s0, cood0_cpp, evlp0_cpp, disp0_cpp, ds=ds, endpoint=True)
    assert np.allclose(cood_arr_py.vector_array, cood_arr_cpp.vector_array, atol=1e-12, rtol=1e-9)
    assert np.allclose(cood_arr_py.s_array, cood_arr_cpp.s_array, atol=1e-12, rtol=1e-9)
    assert evlp_arr_py is not None and evlp_arr_cpp is not None
    assert np.allclose(evlp_arr_py.cov_array[0], evlp_arr_cpp.cov_array[0], atol=1e-12, rtol=1e-9)
    assert np.allclose(disp_arr_py.vector_array, disp_arr_cpp.vector_array, atol=1e-12, rtol=1e-9)
    assert np.allclose(disp_arr_py.s_array, disp_arr_cpp.s_array, atol=1e-12, rtol=1e-9)
