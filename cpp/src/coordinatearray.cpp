/**
 * @file coordinatearray.cpp
 * @brief Implementation of the CoordinateArray class representing an array of particle coordinates in phase space.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// coordinatearray.cpp
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

#include "egret/coordinatearray.hpp"
#include <stdexcept>

/**
 * @brief Construct a new egret::CoordinateArray::CoordinateArray object.
 * @param vector_array Array of particle coordinates (4 x N matrix)
 * @param s_array Longitudinal positions
 * @param z_array Longitudinal displacements
 * @param delta_array Relative momentum deviations
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::CoordinateArray::CoordinateArray(
    const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array,
    const Eigen::ArrayXd& s_array,
    const Eigen::ArrayXd& z_array,
    const Eigen::ArrayXd& delta_array) noexcept(false) :
    BaseArray(s_array), vector_array_(vector_array),
    z_array_(z_array), delta_array_(delta_array) {
    // Check consistency of input sizes
    const auto n = BaseArray::size();
    if (vector_array_.cols() != n) {
        throw std::invalid_argument("Size of s_array does not match number of columns in vector_array");
    }
    if (z_array_.size() == 0) {
        z_array_ = Eigen::ArrayXd::Zero(n);
    } else if (z_array_.size() != n) {
        throw std::invalid_argument("Size of z_array does not match size of s_array");
    }
    if (delta_array_.size() == 0) {
        delta_array_ = Eigen::ArrayXd::Zero(n);
    } else if (delta_array_.size() != n) {
        throw std::invalid_argument("Size of delta_array does not match size of s_array");
    }
}

/**
 * @brief Append another CoordinateArray to this one.
 * @param other The other CoordinateArray to append.
 */
void egret::CoordinateArray::append(const CoordinateArray &other) noexcept(false) {
    BaseArray::append(other);
    const auto n = BaseArray::size();
    Eigen::Matrix<double,4,Eigen::Dynamic> new_vector_array(4, n);
    new_vector_array << vector_array_, other.vector_array_;
    vector_array_.swap(new_vector_array);
    Eigen::ArrayXd new_z_array(n);
    new_z_array << z_array_, other.z_array_;
    z_array_.swap(new_z_array);
    Eigen::ArrayXd new_delta_array(n);
    new_delta_array << delta_array_, other.delta_array_;
    delta_array_.swap(new_delta_array);
}

/**
 * @brief Get Coordinate from linear interpolation.
 * @param s Longitudinal position to interpolate at.
 * @return egret::Coordinate
 * @throws std::out_of_range if s is out of the range of s_array.
 */
egret::Coordinate egret::CoordinateArray::from_s(double s) const noexcept(false) {
    const auto idx = BaseArray::index_from_s(s);
    const double s0 = s_array_[idx];
    const double s1 = s_array_[idx + 1];
    const double ds = s1 - s0;
    if (ds == 0.) {
        // Degenerate case: s0 == s1
        const Eigen::Vector4d vec = 0.5 * (vector_array_.col(idx) + vector_array_.col(idx + 1));
        const double zval = 0.5 * (z_array_[idx] + z_array_[idx + 1]);
        const double dval = 0.5 * (delta_array_[idx] + delta_array_[idx + 1]);
        return Coordinate(vec, s, zval, dval);
    }
    const double a0 = (s1 - s) / ds;
    const double a1 = (s - s0) / ds;
    const Eigen::Vector4d vec = a0 * vector_array_.col(idx) + a1 * vector_array_.col(idx + 1);
    const double zval = a0 * z_array_[idx] + a1 * z_array_[idx + 1];
    const double dval = a0 * delta_array_[idx] + a1 * delta_array_[idx + 1];
    return Coordinate(vec, s, zval, dval);
}
