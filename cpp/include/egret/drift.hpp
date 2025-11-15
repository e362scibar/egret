// drift.hpp
// drift.hpp
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

namespace egret {

class Drift {
public:
    // Return 4x4 transfer matrix for a drift of given length
    static Eigen::Matrix4d transfer_matrix_from_length(double length);

    // Return transfer matrix array (4x4xN) and s array for given length and step ds
    static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array_from_length(double length, double ds = 0.1, bool endpoint = false);
};

} // namespace egret
