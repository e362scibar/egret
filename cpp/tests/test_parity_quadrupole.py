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
import pyegret

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
