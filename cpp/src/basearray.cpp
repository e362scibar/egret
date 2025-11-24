/**
 * @file basearray.cpp
 * @brief Implementation of the BaseArray class representing an array of phase space data.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// basearray.cpp
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

#include "egret/basearray.hpp"
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <utility>

/**
 * @brief Construct a new egret::BaseArray object.
 * @param s_array Longitudinal positions
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::BaseArray::BaseArray(const Eigen::ArrayXd& s_array) noexcept(false):
    s_array_(s_array){
    // Check s_array is non-decreasing
    if (!std::is_sorted(s_array_.data(), s_array_.data() + s_array_.size())) {
        throw std::invalid_argument("s_array must be non-decreasing");
    }
}

/**
 * @brief Append another BaseArray to this one.
 * @param other The other BaseArray to append.
 */
void egret::BaseArray::append(const BaseArray &other) noexcept(false) {
    const auto n = s_array_.size() + other.s_array_.size();
    Eigen::ArrayXd new_s_array(n);
    new_s_array << s_array_, other.s_array_;
    s_array_.swap(new_s_array);
}

/**
 * @brief Get index closest upstream of given s.
 * @param s Longitudinal position.
 * @return size_t Index of the closest upstream point.
 * @throws std::out_of_range if s is out of the range of s_array.
 */
size_t egret::BaseArray::index_from_s(const double s) const noexcept(false) {
    const auto n = s_array_.size();
    if (n < 2) {
        throw std::out_of_range("s_array must contain at least two points.");
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
    return idx - 1;
}
