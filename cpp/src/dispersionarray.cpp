/**
 * @file dispersionarray.cpp
 * @brief Implementation of the DispersionArray class representing an array of dispersion functions.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// dispersionarray.cpp
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

#include "egret/dispersionarray.hpp"
#include <stdexcept>

/**
 * @brief Construct a new egret::DispersionArray object.
 * @param vector_array Array of particle coordinates (4 x N matrix)
 * @param s_array Longitudinal positions
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::DispersionArray::DispersionArray(
    const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array,
    const Eigen::ArrayXd& s_array) noexcept(false) :
    BaseArray(s_array), vector_array_(vector_array) {
    // Check consistency of input sizes
    if (vector_array_.cols() != size()) {
        throw std::invalid_argument("Size of s_array does not match number of columns in vector_array");
    }
}

/**
 * @brief Append another DispersionArray to this one.
 * @param other The other DispersionArray to append.
 */
void egret::DispersionArray::append(const DispersionArray &other) noexcept(false) {
    BaseArray::append(other);
    const auto n = size();
    Eigen::Matrix<double,4,Eigen::Dynamic> new_vector_array(4, n);
    new_vector_array << vector_array_, other.vector_array_;
    vector_array_.swap(new_vector_array);
}

/**
 * @brief Get Dispersion at given s by linear interpolation.
 * @param s Longitudinal position to interpolate at.
 * @return egret::Coordinate
 * @throws std::out_of_range if s is out of the range of s_array.
 */
egret::Dispersion egret::DispersionArray::from_s(double s) const noexcept(false) {
    const auto idx = index_from_s(s);
    const double s0 = s_array_[idx];
    const double s1 = s_array_[idx + 1];
    const double ds = s1 - s0;
    if (ds == 0.) {
        // Degenerate case: s0 == s1
        const Eigen::Vector4d vec = 0.5 * (vector_array_.col(idx) + vector_array_.col(idx + 1));
        return Dispersion(vec, s);
    }
    const double a0 = (s1 - s) / ds;
    const double a1 = (s - s0) / ds;
    const Eigen::Vector4d vec = a0 * vector_array_.col(idx) + a1 * vector_array_.col(idx + 1);
    return Dispersion(vec, s);
}
