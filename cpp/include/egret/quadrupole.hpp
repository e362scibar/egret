// quadrupole.hpp
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
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace egret {

class Quadrupole {
public:
    // transfer matrix for a quadrupole of length L and strength k (k=k1/(1+delta))
    static Eigen::Matrix4d transfer_matrix(double length, double k1, double tilt = 0.0, double delta = 0.0);

    // transfer matrix array along the element (4x4xN) and s positions
    static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array(double length, double k1, double tilt = 0.0, double delta = 0.0, double ds = 0.1, bool endpoint = false);
    // additive dispersion vector for given initial coordinate
    static Eigen::Vector4d dispersion(const Eigen::Vector4d &cood0vec, double length, double k1, double delta = 0.0);
};

} // namespace egret
