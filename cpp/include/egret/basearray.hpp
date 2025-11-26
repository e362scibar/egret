/**
 * @file basearray.hpp
 * @brief Base class of data arrays for particles in phase space.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// basearray.hpp
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
#include "egret/coordinate.hpp"

namespace egret {
    class BaseArray;
}

/**
 * @brief Base class of data arrays for particles in phase space.
 */
class egret::BaseArray {
protected:
    //! Longitudinal positions
    Eigen::ArrayXd s_array_;

public:
    // Constructor
    BaseArray(const Eigen::ArrayXd& s_array) noexcept(false);
    /**
     * @brief Destroy the BaseArray object.
     */
    virtual ~BaseArray() = default;

    /**
     * @brief Get the size of the BaseArray.
     * @return size_t
     */
    size_t size() const { return s_array_.size(); }

    /**
     * @brief Get the array of longitudinal positions.
     * @return const Eigen::ArrayXd& Array of longitudinal positions
     */
    const Eigen::ArrayXd& s_array() const { return s_array_; }

    /**
     * @brief Set the array of longitudinal positions.
     * @param other Eigen::ArrayXd to copy from
     */
    void s_array(const Eigen::ArrayXd& s_array) { s_array_ = s_array; }

    /**
     * @brief Get the step size between longitudinal positions.
     * @return double Step size
     * @throws std::runtime_error if the size of s_array is less than 2.
     */
    double ds() const noexcept(false) {
        const size_t n = size();
        if (n < 2) {
            throw std::runtime_error("ds() requires at least two elements in s_array");
        }
        return s_array_(1) - s_array_(0);
    }

    // Efficient append (reserve + copy)
    void append(const BaseArray &other) noexcept(false);

    // Get Coordinate from linear interpolation
    virtual size_t index_from_s(double s) const noexcept(false);
};
