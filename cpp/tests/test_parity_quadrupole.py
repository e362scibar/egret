import sys
import os
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Ensure Python egret package is importable
sys.path.insert(0, repo_root)
# Ensure built pyegret module in build directory is importable
build_dir = os.path.join(repo_root, 'build')
sys.path.insert(0, build_dir)

print('Repository root:', repo_root)
print('Build dir:', build_dir)

from egret.quadrupole import Quadrupole as PyQuadrupole
import importlib
import importlib.util
import glob
import sys
import os


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


pyegret = load_pyegret()

def compare_quadrupole(length=0.5, k1=1.2, tilt=0.13, delta=0.0):
    q_py = PyQuadrupole('Q', length, k1, tilt=tilt)
    t_py = q_py.transfer_matrix()

    t_cpp = pyegret.quadrupole_transfer_matrix(length, k1, tilt, delta)

    print('Python Quadrupole tmat:\n', t_py)
    print('C++ pyegret Quadrupole tmat:\n', t_cpp)
    close = np.allclose(t_py, t_cpp, atol=1e-12, rtol=1e-9)
    print('Allclose:', close)
    if not close:
        diff = t_py - t_cpp
        print('Max abs diff:', np.max(np.abs(diff)))
    return close

if __name__ == '__main__':
    ok = compare_quadrupole()
    sys.exit(0 if ok else 2)
