/**
 * @file dispersionarray.hpp
 * @brief Definition of the DispersionArray class representing an array of dispersion functions.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// dispersionarray.hpp
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
#include <algorithm>
#include "egret/dispersion.hpp"
#include "egret/basearray.hpp"

namespace egret {
    class DispersionArray;
}

class egret::DispersionArray : public egret::BaseArray {
protected:
    // 4 x N matrix
    Eigen::Matrix<double, 4, Eigen::Dynamic> vector_array_;

public:
    // Constructor
    DispersionArray(const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array,
                    const Eigen::ArrayXd& s_array) noexcept(false);
    /**
     * @brief Destroy the DispersionArray object.
     */
    virtual ~DispersionArray() = default;

    /**
     * @brief Get the array of dispersion functions.
     * @return Eigen::Matrix<double,4,Eigen::Dynamic> 4 x N matrix of dispersion functions
     */
    const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array() const { return vector_array_; }

    /**
     * @brief Get the array of x components of the dispersion functions.
     * @return Eigen::ArrayXd Array of x components
     */
    Eigen::ArrayXd x_array() const { return vector_array_.row(0).array(); }
    /**
     * @brief Get the array of x' components of the dispersion functions.
     * @return Eigen::ArrayXd Array of x' components
     */
    Eigen::ArrayXd xp_array() const { return vector_array_.row(1).array(); }
    /**
     * @brief Get the array of y components of the dispersion functions.
     * @return Eigen::ArrayXd Array of y components
     */
    Eigen::ArrayXd y_array() const { return vector_array_.row(2).array(); }
    /**
     * @brief Get the array of y' components of the dispersion functions.
     * @return Eigen::ArrayXd Array of y' components
     */
    Eigen::ArrayXd yp_array() const { return vector_array_.row(3).array(); }

    // Efficient append (reserve + copy)
    void append(const DispersionArray &other);

    // Get Dispersion at given s by linear interpolation
    Dispersion from_s(double s) const;
};
