/**
 * @file envelopearray.cpp
 * @brief Implementation of the EnvelopeArray class representing an array of phase space envelopes.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// envelopearray.cpp
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

#include "egret/envelopearray.hpp"
#include <stdexcept>
#include <ranges>

/**
 * @brief Construct a new egret::EnvelopeArray object.
 * @param cov_array Array of covariance matrices (std::vector of 4 x 4 Matrices)
 * @param s_array Longitudinal positions
 * @param
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::EnvelopeArray::EnvelopeArray(
    const std::vector<Eigen::Matrix4d>& cov_array,
    const Eigen::ArrayXd& s_array,
    const std::optional<std::vector<Eigen::Matrix2d>>& T_array) noexcept(false) :
    BaseArray(s_array), cov_array_(cov_array), T_array_(T_array.value_or(std::vector<Eigen::Matrix2d>())),
    tau_array_(Eigen::ArrayXd::Ones(s_array.size())), U_array_(), V_array_() {
    // Check consistency of input sizes
    const auto n = BaseArray::size();
    if (cov_array_.size() != n) {
        throw std::invalid_argument("Size of s_array does not match the number of covariance matrices in cov_array");
    }
    if (T_array && T_array->size() != n) {
        throw std::invalid_argument("Size of s_array does not match the number of transformation matrices in T_array");
    }
    // To Do: T_array_ initialization and other member initializations can be added here.

}

/**
 * @brief Append another EnvelopeArray to this one.
 * @param other The other EnvelopeArray to append.
 */
void egret::EnvelopeArray::append(const EnvelopeArray &other) noexcept(false) {
    BaseArray::append(other);
    cov_array_.insert(cov_array_.end(), other.cov_array_.begin(), other.cov_array_.end());
}

/**
 * @brief Get Envelope from linear interpolation.
 * @param s Longitudinal position to interpolate at.
 * @return egret::Envelope
 * @throws std::out_of_range if s is out of the range of s_array.
 */
egret::Envelope egret::EnvelopeArray::from_s(double s) const noexcept(false) {
    const auto idx = BaseArray::index_from_s(s);
    const double s0 = s_array_(idx);
    const double s1 = s_array_(idx + 1);
    const double ds = s1 - s0;
    if (ds == 0.) {
        // Degenerate case: s0 == s1
        const Eigen::Matrix4d cov = 0.5 * (cov_array_[idx] + cov_array_[idx + 1]);
        const double zval = 0.5 * (z_array_(idx) + z_array_(idx + 1));
        const double dval = 0.5 * (delta_array_(idx) + delta_array_(idx + 1));
        return Envelope(cov, s, zval, dval);
    }
    const double a0 = (s1 - s) / ds;
    const double a1 = (s - s0) / ds;
    const Eigen::Matrix4d cov = a0 * cov_array_[idx] + a1 * cov_array_[idx + 1];
    const double zval = a0 * z_array_(idx) + a1 * z_array_(idx + 1);
    const double dval = a0 * delta_array_(idx) + a1 * delta_array_(idx + 1);
    return Envelope(cov, s, zval, dval);
}
