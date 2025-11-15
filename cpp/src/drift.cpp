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
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace egret {

Eigen::Matrix4d Drift::transfer_matrix_from_length(double length) {
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    tmat(0,1) = length;
    tmat(2,3) = length;
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Drift::transfer_matrix_array_from_length(double length, double ds, bool endpoint) {
    std::vector<double> s;
    if (std::abs(length) > 0.0) {
        int n_base = static_cast<int>(std::floor(length / ds));
        int n = n_base + static_cast<int>(endpoint) + 1;
        s.reserve(n);
        for (int i = 0; i < n; ++i) s.push_back((static_cast<double>(i) * length) / (n - 1));
    } else {
        s.push_back(0.0);
    }

    int N = static_cast<int>(s.size());
    Eigen::Tensor<double,3> tmat(4,4,N);
    tmat.setZero();
    for (int k=0;k<N;++k) {
        // identity
        for (int i=0;i<4;++i) tmat(i,i,k) = 1.0;
        tmat(0,1,k) = s[k];
        tmat(2,3,k) = s[k];
    }
    return {tmat, s};
}

} // namespace egret
