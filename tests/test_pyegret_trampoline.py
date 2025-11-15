import importlib
import importlib.util
import glob
import sys
import os
import pytest


def load_pyegret():
    try:
        return importlib.import_module('egret.pyegret')
    except ModuleNotFoundError:
        for p in list(sys.path):
            candidate_dir = os.path.join(p, 'egret')
            if not os.path.isdir(candidate_dir):
                continue
            matches = glob.glob(os.path.join(candidate_dir, 'pyegret*.so'))
            if not matches:
                matches = glob.glob(os.path.join(candidate_dir, 'pyegret*.cpython-310-*.so'))
            if matches:
                path = matches[0]
                spec = importlib.util.spec_from_file_location('egret.pyegret', path)
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
            super().__init__()

        def transfer(self, cood0, ds):
            # return a recognizable marker + inputs
            return ("MARK", cood0.s, float(ds))

        def transfer_array(self, cood0, ds, endpoint):
            import numpy as np
            arr = np.zeros((4, 4, 1), dtype=float)
            svec = [cood0.s]
            return (arr, svec)

    me = MyElem()

    out = mod.call_element_transfer(me, c, 0.123)
    assert isinstance(out, tuple)
    assert out[0] == "MARK"
    assert pytest.approx(out[2], rel=1e-12) == 0.123

    out2 = mod.call_element_transfer_array(me, c, 0.05, False)
    assert isinstance(out2, tuple)
    arr, svec = out2
    assert hasattr(arr, "shape")
    assert arr.shape == (4, 4, 1)
    assert isinstance(svec, (list, tuple))
