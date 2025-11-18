/**
 * @file envelope.cpp
 * @brief Implementation of the Envelope class representing an phase space envelope.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// envelope.cpp
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

#include "egret/envelope.hpp"
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <utility>

/**
 * @brief Construct a new egret::Envelope object.
 * @param cov Covariance matrices (4 x 4 x N tensor)
 * @param s_array Longitudinal positions
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::Envelope::Envelope(
    const Eigen::Matrix4d& cov,
    double s,
    std::optional<const Eigen::Matrix2d&> T) noexcept(false) :
    cov_(cov), s_(s), T_(T.value_or(Eigen::Matrix2d::Zero())) {
    calc_eigenmode(T)
}

void egret::Envelope::calc_eigenmode(
    std::optional<const Eigen::Matrix2d&> T) noexcept(false) {
    // If T is provided, use it
    if (T) {
        T_ = *T;
    } else {
        // Calculate T from the covariance matrix
        const Eigen::Matrix2d Sxx = cov_.block<2, 2>(0, 0);
        const Eigen::Matrix2d Sxy = cov_.block<2, 2>(0, 2);
        const Eigen::Matrix2d Syy = cov_.block<2, 2>(2, 2);

        Eigen::Matrix4d mat;
        mat << -Sxx(0,0), -Sxx(0,1)-Syy(0,1), 0., Syy(0,0),
               0., -Syy(1,1), -Sxx(0,0), -Sxx(0,1)+Syy(0,1),
               -Sxx(0,1)+Syy(0,1), -Sxx(1,1), -Syy(0,0), 0.,
               Syy(1,1), 0., -Sxx(0,1)-Syy(0,1), -Sxx(1,1);
        const Eigen::Vector4d vec = Eigen::Map<const Eigen::Vector4d>(Sxy.data());

        Eigen::Vector4d res;
        try {
            res = mat.colPivHouseholderQr().solve(vec);
        } catch (const std::exception& e) {
            res = Eigen::Vector4d::Zero();
        }
        T_ = Eigen::Map<const Eigen::Matrix2d>(res.data());
    }
    // calculate tau, U, V
    Eigen::Matrix2d T_s; // adjoint of T
    T_s << T_(1,1), -T_(0,1),
            -T_(1,0), T_(0,0);
    tau_ = std::sqrt(1.0 - T_.determinant());
    const double chi = 1.0 / (2.0 * tau_ * tau_ - 1.0);
    const double sqrtchi = std::sqrt(chi);
    const Eigen::Matrix2d Sxx = cov_.block<2, 2>(0, 0);
    const Eigen::Matrix2d Syy = cov_.block<2, 2>(2, 2);
    U_ = sqrtchi * (tau_ * tau_ * Sxx - T_s * Syy * T_s.transpose());
    V_ = sqrtchi * (tau_ * tau_ * Syy - T_ * Sxx * T_.transpose());
}
