import pickle
import pytest

def test_quadrupole_pickle_roundtrip():
    from egret.python.quadrupole import Quadrupole

    q = Quadrupole('Q', 1.1, 0.4, tilt=0.0)
    data = pickle.dumps(q)
    q2 = pickle.loads(data)

    assert pytest.approx(q2.length, rel=1e-12) == 1.1
    assert pytest.approx(q2.k1, rel=1e-12) == 0.4
    assert pytest.approx(q2.tilt, rel=1e-12) == 0.0
