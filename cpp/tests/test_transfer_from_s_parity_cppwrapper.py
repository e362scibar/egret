import numpy as np


def test_transfer_from_s_parity_with_alignment_cppwrapper():
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

    cood_vec = np.array([1.0e-3, -2.0e-4, 3.0e-4, 4.0e-4])
    disp_vec = np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4])

    cood0_py = PyCoordinate(cood_vec, 0.2, -0.1, 0.01)
    cood0_cpp = CppCoordinate(cood_vec, 0.2, -0.1, 0.01)
    evlp0_py = PyEnvelope()
    evlp0_cpp = CppEnvelope()
    disp0_py = PyDispersion(disp_vec, 0.2)
    disp0_cpp = CppDispersion(disp_vec, 0.2)

    py_elem = PyDipole("D", length, angle, dx=dx, dy=dy, ds=ds_offset)
    cpp_elem = CppDipole("D", length, angle, dx=dx, dy=dy, ds=ds_offset)

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

    cood_arr_py, evlp_arr_py, disp_arr_py = py_elem.transfer_array_from_s(
        s0, cood0_py, evlp0_py, disp0_py, ds=ds, endpoint=True
    )
    cood_arr_cpp, evlp_arr_cpp, disp_arr_cpp = cpp_elem.transfer_array_from_s(
        s0, cood0_cpp, evlp0_cpp, disp0_cpp, ds=ds, endpoint=True
    )

    assert np.allclose(cood_arr_py.vector, cood_arr_cpp.vector, atol=1e-12, rtol=1e-9)
    assert np.allclose(cood_arr_py.s, cood_arr_cpp.s, atol=1e-12, rtol=1e-9)

    assert evlp_arr_py is not None and evlp_arr_cpp is not None
    assert np.allclose(evlp_arr_py.cov, evlp_arr_cpp.cov, atol=1e-12, rtol=1e-9)

    assert disp_arr_py is not None and disp_arr_cpp is not None
    assert np.allclose(disp_arr_py.vector, disp_arr_cpp.vector, atol=1e-12, rtol=1e-9)
    assert np.allclose(disp_arr_py.s, disp_arr_cpp.s, atol=1e-12, rtol=1e-9)
