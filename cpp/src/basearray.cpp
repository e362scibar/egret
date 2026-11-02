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
#include <string>
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
 * @brief Check that a per-element array has the same length as s_array.
 * @param n Length of the array to check.
 * @param name Name of the array used in the error message.
 * @throws std::invalid_argument if n does not match the size of s_array.
 */
void egret::BaseArray::check_length(const Eigen::Index n, const char *name) const noexcept(false) {
    if (n != static_cast<Eigen::Index>(size())) {
        throw std::invalid_argument(std::string(name) + " must have the same length as s_array");
    }
}

/**
 * @brief Set the array of longitudinal positions.
 * @param s_array Eigen::ArrayXd to copy from; must keep the current length and be non-decreasing.
 * @throws std::invalid_argument if the length changes or s_array is not non-decreasing.
 */
void egret::BaseArray::s_array(const Eigen::ArrayXd& s_array) noexcept(false) {
    if (s_array.size() != s_array_.size()) {
        throw std::invalid_argument("s_array must keep its current length");
    }
    if (!std::is_sorted(s_array.data(), s_array.data() + s_array.size())) {
        throw std::invalid_argument("s_array must be non-decreasing");
    }
    s_array_ = s_array;
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
 * @throws std::invalid_argument if s_array has fewer than two points or s is out of the range of s_array.
 */
size_t egret::BaseArray::index_from_s(const double s) const noexcept(false) {
    static constexpr double tol = 1.0e-6;
    const size_t n = s_array_.size();
    if (n < 2) {
        throw std::invalid_argument("s_array must contain at least two points.");
    }
    if (s < s_array_(0) - tol || s > s_array_(n - 1) + tol) {
        throw std::invalid_argument("s value is out of the range of the s_array");
    }
    // binary search to find the right interval
    const double *begin = s_array_.data();
    const double *end = s_array_.data() + n;
    const double *it = std::upper_bound(begin, end, s);
    size_t idx = std::distance(begin, it);
    if (idx == 0) {
        idx = 1;
    } else if (idx == n) {
        idx = n - 1;
    }
    return idx - 1;
}
