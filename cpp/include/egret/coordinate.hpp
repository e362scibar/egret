// coordinate.hpp
// coordinate.hpp
//
// Copyright (C) 2025 Hirokazu Maesaka (RIKEN SPring-8 Center)
//
// This file is part of Egret: Engine for General Research in
// Energetic-beam Tracking.
//
// Egret is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <Eigen/Dense>

namespace egret {

struct Coordinate {
    Eigen::Vector4d vector; // x, xp, y, yp
    double s{0.0};
    double z{0.0};
    double delta{0.0};

    Coordinate() : vector(Eigen::Vector4d::Zero()), s(0.), z(0.), delta(0.) {}
    Coordinate(const Eigen::Vector4d &v, double s_, double z_ = 0., double d_ = 0.)
        : vector(v), s(s_), z(z_), delta(d_) {}
};

} // namespace egret
