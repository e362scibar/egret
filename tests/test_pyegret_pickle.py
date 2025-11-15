import importlib
import importlib.util
import glob
import sys
import os
import pickle
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


def test_quadrupole_pickle_roundtrip():
    mod = load_pyegret()
    q = mod.Quadrupole(1.1, 0.4, 0.0)
    data = pickle.dumps(q)
    q2 = pickle.loads(data)

    # attributes exposed as read/write on the wrapper
    assert pytest.approx(q2.length, rel=1e-12) == 1.1
    assert pytest.approx(q2.k1, rel=1e-12) == 0.4
    assert pytest.approx(q2.tilt, rel=1e-12) == 0.0
