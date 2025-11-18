/**
 * @brief Class representing a particle coordinate in phase space
 */
// envelope.hpp
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
#include <optional>

namespace egret {

class Envelope {
protected:
    //! 4x4 covariance matrix of the beam envelope
    Eigen::Matrix4d cov_;
    //! Longitudinal position s
    double s_;
    //! 2x2 coordinate transformation matrix for eigenmode
    Eigen::Matrix2d T_;
    //! Factor for eigenmode normalization
    double tau_{1.0};
    //! Matrices for eigenmode parameters
    Eigen::Matrix2d U_{Eigen::Matrix2d::Identity()};
    Eigen::Matrix2d V_{Eigen::Matrix2d::Identity()};

public:
    // Constructor
    Envelope(
        const Eigen::Matrix4d &cov = Eigen::Matrix4d::Identity(),
        double s = 0.,
        std::optional<const Eigen::Matrix2d&> T = std::nullopt)
        noexcept(false);

    // Calculate eigenmode parameters
    void calc_eigenmode(
        std::optional<const Eigen::Matrix2d&> T = std::nullopt)
        noexcept(false);
};

} // namespace egret
