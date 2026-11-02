import subprocess
from pathlib import Path

import numpy as np
import pytest

from egret.python import Quadrupole as PyQuadrupole

CLI_BIN = Path(__file__).resolve().parents[1] / 'cpp' / 'build' / 'egret_cli'
LENGTH = 0.5
K1 = 1.2
TILT = 0.13
TOL = 1.0e-6
MATRIX_ROWS = 4


def run_cli(length: float, k1: float, tilt: float) -> np.ndarray:
    '''
    Run the C++ CLI and parse the first transfer matrix it prints.

    The CLI also prints diagnostic matrices after a "polymorphic call" header; those are not compared here.

    Args:
        length (float): Quadrupole length [m].
        k1 (float): Normalized quadrupole gradient [1/m^2].
        tilt (float): Roll angle [rad].

    Returns:
        np.ndarray: 4x4 transfer matrix printed by the CLI.
    '''
    out = subprocess.check_output([str(CLI_BIN), str(length), str(k1), str(tilt)])
    lines = out.decode('utf-8').strip().splitlines()[:MATRIX_ROWS]
    return np.array([[float(x) for x in line.split()] for line in lines])


def test_quadrupole_cli_parity():
    if not CLI_BIN.exists():
        pytest.skip(f'CLI binary not built: {CLI_BIN}')
    t_py = PyQuadrupole('Q', LENGTH, K1, tilt=TILT).transfer_matrix()
    t_cli = run_cli(LENGTH, K1, TILT)
    np.testing.assert_allclose(t_cli, t_py, atol=TOL, rtol=0.0)
