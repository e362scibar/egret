import importlib
import importlib.util
import glob
import sys
import os
import pytest


def load_pyegret():
    try:
        return importlib.import_module('egret.cppegret')
    except ModuleNotFoundError:
        for p in list(sys.path):
            candidate_dir = os.path.join(p, 'egret')
            if not os.path.isdir(candidate_dir):
                continue
            matches = glob.glob(os.path.join(candidate_dir, 'cppegret*.so'))
            if not matches:
                matches = glob.glob(os.path.join(candidate_dir, 'cppegret*.cpython-310-*.so'))
            if matches:
                path = matches[0]
                spec = importlib.util.spec_from_file_location('egret.cppegret', path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        raise


def get_pyegret():
    return load_pyegret()


def test_transfer_and_transfer_array_trampoline():
    mod = get_pyegret()
    Coordinate = mod.Coordinate
    c = Coordinate()

    class MyElem(mod.Element):
        def __init__(self):
            super().__init__("MyElem", 0.0)

        def transfer_matrix(self, cood0=None, ds=0.1, method=mod.IntegrationMethod.SYMPLECTIC4):
            import numpy as np
            m = np.eye(4)
            m[0, 1] = float(ds)
            return m

        def transfer_matrix_array(self, cood0=None, ds=0.1, endpoint=False, method=mod.IntegrationMethod.SYMPLECTIC4):
            import numpy as np
            return [np.eye(4)], np.array([0.0])

    me = MyElem()

    out = me.transfer_matrix(c, 0.123, mod.IntegrationMethod.SYMPLECTIC4)
    assert hasattr(out, "shape")
    assert out.shape == (4, 4)
    assert pytest.approx(out[0, 1], rel=1e-12) == 0.123

    out2 = me.transfer_matrix_array(c, 0.05, False, mod.IntegrationMethod.SYMPLECTIC4)
    assert isinstance(out2, tuple)
    mats, svec = out2
    assert isinstance(mats, list)
    assert len(mats) == 1
    assert hasattr(mats[0], "shape")
    assert mats[0].shape == (4, 4)
    assert hasattr(svec, "shape")
