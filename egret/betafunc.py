# betafunc.py
#
# Copyright (C) 2025 Hirokazu Maesaka (RIKEN SPring-8 Center)
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

from .object import Object

import numpy as np

class BetaFunc:
    """
    Beta function object.
    """
    def __init__(self, bx=1., ax=0., by=1., ay=0.):
        self.bx = bx
        self.ax = ax
        self.by = by
        self.ay = ay
        self.gx = (1. + ax**2) / bx
        self.gy = (1. + ay**2) / by
