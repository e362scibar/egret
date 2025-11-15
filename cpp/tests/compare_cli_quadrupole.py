import subprocess
import sys
import os
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, repo_root)

import importlib.util

# Load Python quadrupole module directly (avoid importing full 'egret' package)
quad_path = os.path.join(repo_root, 'egret', 'quadrupole.py')
import types
# Create a lightweight package object to allow relative imports inside the module
pkg = types.ModuleType('egret')
pkg.__path__ = [os.path.join(repo_root, 'egret')]
sys.modules['egret'] = pkg

spec = importlib.util.spec_from_file_location('egret.quadrupole', quad_path)
py_quad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_quad)
PyQuadrupole = py_quad.Quadrupole


def run_cli(length=0.5, k1=1.2, tilt=0.13):
    # The CMake build directory is under the cpp subfolder
    bin_path = os.path.join(repo_root, 'cpp', 'build', 'egret_cli')
    if not os.path.exists(bin_path):
        print('CLI binary not found:', bin_path)
        return None
    out = subprocess.check_output([bin_path, str(length), str(k1), str(tilt)])
    txt = out.decode('utf-8').strip().splitlines()
    mat = np.array([[float(x) for x in line.split()] for line in txt])
    return mat

def compare():
    length = 0.5; k1 = 1.2; tilt = 0.13
    t_py = PyQuadrupole('Q', length, k1, tilt=tilt).transfer_matrix()
    t_cli = run_cli(length, k1, tilt)
    print('Python tmat:\n', t_py)
    print('CLI tmat:\n', t_cli)
    diff = np.abs(t_py - t_cli)
    maxdiff = np.max(diff)
    tol = 1e-6
    ok = maxdiff <= tol
    print('Max abs diff:', maxdiff)
    print('Within tol', tol, ':', ok)
    return 0 if ok else 2

if __name__ == '__main__':
    sys.exit(compare())
