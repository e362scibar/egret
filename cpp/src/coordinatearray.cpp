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
#include <algorithm>
#include <optional>
#include <utility>

/**
 * @brief Construct a new egret::CoordinateArray::CoordinateArray object.
 * @param vector_array Array of particle coordinates (4 x N matrix)
 * @param s_array Longitudinal positions
 * @param z_array Longitudinal displacements
 * @param delta_array Relative momentum deviations
 */
egret::CoordinateArray::CoordinateArray(
    const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array,
    const Eigen::ArrayXd& s_array,
    const Eigen::ArrayXd& z_array,
    const Eigen::ArrayXd& delta_array) :
    vector_array_(vector_array), s_array_(s_array),
    z_array_(z_array), delta_array_(delta_array) {
    // Check consistency of input sizes
    const size_t n = vector_array_.cols();
    if (s_array_.size() != n) {
        throw std::invalid_argument("Size of s_array does not match number of columns in vector_array");
    } else if (!std::is_sorted(s_array_.data(), s_array_.data() + s_array_.size())) {
        throw std::invalid_argument("s_array must be non-decreasing");
    }
    if (z_array_.size() == 0) {
        z_array_ = Eigen::ArrayXd::Zero(n);
    } else if (z_array_.size() != n) {
        throw std::invalid_argument("Size of z_array does not match number of columns in vector_array");
    }
    if (delta_array_.size() == 0) {
        delta_array_ = Eigen::ArrayXd::Zero(n);
    } else if (delta_array_.size() != n) {
        throw std::invalid_argument("Size of delta_array does not match number of columns in vector_array");
    }
}

/**
 * @brief Append another CoordinateArray to this one.
 * @param other The other CoordinateArray to append.
 */
void egret::CoordinateArray::append(const CoordinateArray &other) {
    const auto n = vector_array_.cols() + other.vector_array_.cols();
    Eigen::Matrix<double,4,Eigen::Dynamic> new_vector_array(4, n);
    new_vector_array << vector_array_, other.vector_array_;
    vector_array_.swap(new_vector_array);
    Eigen::ArrayXd new_s_array(n);
    new_s_array << s_array_, other.s_array_;
    s_array_.swap(new_s_array);
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
 */
egret::Coordinate egret::CoordinateArray::from_s(double s) const {
    const auto n = s_array_.size();
    if (n < 2) {
        throw std::out_of_range("CoordinateArray must contain at least two points for interpolation");
    }
    if (s < s_array_(0) || s > s_array_(n - 1)) {
        throw std::out_of_range("s value is out of the range of the s_array");
    }
    // binary search to find the right interval
    const double *begin = s_array_.data();
    const double *end = s_array_.data() + n;
    const double *it = std::upper_bound(begin, end, s);
    size_t idx = std::distance(begin, it);
    if (idx == 0) {
        throw std::out_of_range("Out of range");
    }
    if (idx == n) {
        idx = n - 1;
    }
    if (idx == n - 1) {
        throw std::out_of_range("Out of range");
    }
    const double s0 = s_array_[idx - 1];
    const double s1 = s_array_[idx];
    const double ds = s1 - s0;
    if (ds == 0.) {
        // Degenerate case: s0 == s1
        Eigen::Vector4d vec = 0.5 * (vector_array_.col(idx - 1) + vector_array_.col(idx));
        double zval = 0.5 * (z_array_[idx - 1] + z_array_[idx]);
        double dval = 0.5 * (delta_array_[idx - 1] + delta_array_[idx]);
        return Coordinate(vec, s, zval, dval);
    }
    const double a0 = (s1 - s) / ds;
    const double a1 = (s - s0) / ds;
    const Eigen::Vector4d vec = a0 * vector_array_.col(idx - 1) + a1 * vector_array_.col(idx);
    const double zval = a0 * z_array_[idx - 1] + a1 * z_array_[idx];
    const double dval = a0 * delta_array_[idx - 1] + a1 * delta_array_[idx];
    return Coordinate(vec, s, zval, dval);
}
