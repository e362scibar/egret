/**
 * @file envelopearray.hpp
 * @brief Class representing an array of beam envelopes in phase space
 * @author Hirokazu Maesaka
 * @date 2025
 */
// envelopearray.hpp
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
#include <vector>
#include "egret/envelope.hpp"
#include "egret/basearray.hpp"

namespace egret {
    class EnvelopeArray;
}

/**
 * @brief Class representing an array of beam envelopes in phase space.
 */
class egret::EnvelopeArray: public BaseArray {
protected:
    //! std::vector of 4 x 4 covariance matrices
    std::vector<Eigen::Matrix4d> cov_array_;
    //! std::vector of 2x2 coordinate transformation matrix for eigenmode
    std::vector<Eigen::Matrix2d> T_array_;
    //! Factor for eigenmode normalization
    Eigen::ArrayXd tau_array_;
    //! std::vector of covariance matrices for the envelope of the eigenmode U
    std::vector<Eigen::Matrix2d> U_array_;
    //! std::vector of covariance matrices for the envelope of the eigenmode V
    std::vector<Eigen::Matrix2d> V_array_;

public:
    // Constructor
    EnvelopeArray(
        const std::vector<Eigen::Matrix4d>& cov_array,
        const Eigen::ArrayXd& s_array,
        const std::optional<std::vector<Eigen::Matrix2d>>& T_array = std::nullopt)
        noexcept(false);
    /**
    * @brief Destroy the EnvelopeArray object.
    */
    virtual ~EnvelopeArray() = default;

    // Efficient append (reserve + copy)
    void append(const EnvelopeArray &other);

    // linear interpolation like Python's from_s
    Envelope from_s(double sval) const noexcept(false);
};
