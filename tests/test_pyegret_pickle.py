# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: egret
#     language: python
#     name: python3
# ---

# %% [markdown]
# tests/test_pyegret_pickle.py
#
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
#
# This file is part of Egret: Engine for General Research in
# Energetic-beam Tracking.
#
# Egret is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# %%
import pickle
import pytest

# %%
def test_quadrupole_pickle_roundtrip():
    from egret.python.quadrupole import Quadrupole

    q = Quadrupole('Q', 1.1, 0.4, tilt=0.0)
    data = pickle.dumps(q)
    q2 = pickle.loads(data)

    assert pytest.approx(q2.length, rel=1e-12) == 1.1
    assert pytest.approx(q2.k1, rel=1e-12) == 0.4
    assert pytest.approx(q2.tilt, rel=1e-12) == 0.0
