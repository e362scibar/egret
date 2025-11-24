/**
 * @file drift.cpp
 * @brief Drift element class implementation
 * @author Hirokazu Maesaka
 * @date 2025
 */
// drift.cpp
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

#include "egret/drift.hpp"
#include <ranges>

Eigen::Matrix4d egret::Drift::transfer_matrix_from_length(const double length) noexcept(false) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0,1) = length;
    M(2,3) = length;
    return M;
}

std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Drift::transfer_matrix_array_from_length(const double length, const double ds, const bool endpoint) noexcept(false) {
    const auto s_ary = s_array(length, ds, endpoint);
    const size_t n = s_ary.size();
    std::vector<Eigen::Matrix4d> M_array(n, Eigen::Matrix4d::Identity());
    for (const size_t i : std::views::iota(0u, n)) {
        auto &M = M_array[i]; // Matrix4d
        const auto s = s_ary[i]; // double
        M(0,1) = s;
        M(2,3) = s;
    }
    return std::make_tuple(M_array, s_ary);
}
