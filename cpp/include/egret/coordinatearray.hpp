/**
 * @file coordinatearray.hpp
 * @brief Class representing an array of particle coordinates in phase space.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// coordinatearray.hpp
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
#include "egret/basearray.hpp"
namespace egret {
    class CoordinateArray;
}

/**
 * @brief Class representing an array of particle coordinates in phase space.
 */
class egret::CoordinateArray: public egret::BaseArray {
protected:
    //! 4 x N matrix of particle coordinates (x, xp, y, yp)
    Eigen::Matrix<double, 4, Eigen::Dynamic> vector_array_;
    //! Longitudinal displacements
    Eigen::ArrayXd z_array_;
    //! Relative momentum deviations
    Eigen::ArrayXd delta_array_;

public:
    // Constructor
    CoordinateArray(
        const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array,
        const Eigen::ArrayXd& s_array,
        const Eigen::ArrayXd& z_array = Eigen::ArrayXd(),
        const Eigen::ArrayXd& delta_array = Eigen::ArrayXd()) noexcept(false);
    /**
     * @brief Destroy the CoordinateArray object.
     */
    virtual ~CoordinateArray() = default;

    /**
     * @brief Get the array of particle coordinate vectors.
     * @return const Eigen::Matrix<double,4,Eigen::Dynamic>& Array of particle coordinate vectors
     */
    const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array() const { return vector_array_; }
    /**
     * @brief Get the array of x coordinates.
     * @return const Eigen::ArrayXd Array of x coordinates
     */
    const Eigen::ArrayXd x_array() const { return vector_array_.row(0); }
    /**
     * @brief Get the array of x' coordinates.
     * @return const Eigen::ArrayXd Array of x' coordinates
     */
    const Eigen::ArrayXd xp_array() const { return vector_array_.row(1); }
    /**
     * @brief Get the array of y coordinates.
     * @return const Eigen::ArrayXd Array of y coordinates
     */
    const Eigen::ArrayXd y_array() const { return vector_array_.row(2); }
    /**
     * @brief Get the array of y' coordinates.
     * @return const Eigen::ArrayXd Array of y' coordinates
     */
    const Eigen::ArrayXd yp_array() const { return vector_array_.row(3); }
    /**
     * @brief Get the array of longitudinal displacements.
     * @return const Eigen::ArrayXd& Array of longitudinal displacements
    */
    const Eigen::ArrayXd& z_array() const { return z_array_; }
    /**
     * @brief Get the array of relative momentum deviations.
     * @return const Eigen::ArrayXd& Array of relative momentum deviations
     */
    const Eigen::ArrayXd& delta_array() const { return delta_array_; }

    /**
     * @brief Set the array of particle coordinate vectors.
     * @param vector_array Array of particle coordinate vectors
     */
    void vector_array(const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array) {
        vector_array_ = vector_array;
    }
    /**
     * @brief Set the array of x coordinates.
     * @param x_array Array of x coordinates
     */
    void x_array(const Eigen::ArrayXd& x_array) { vector_array_.row(0) = x_array; }
    /**
     * @brief Set the array of x' coordinates.
     * @param xp_array Array of x' coordinates
     */
    void xp_array(const Eigen::ArrayXd& xp_array) { vector_array_.row(1) = xp_array; }
    /**
     * @brief Set the array of y coordinates.
     * @param y_array Array of y coordinates
     */
    void y_array(const Eigen::ArrayXd& y_array) { vector_array_.row(2) = y_array; }
    /**
     * @brief Set the array of y' coordinates.
     * @param yp_array Array of y' coordinates
     */
    void yp_array(const Eigen::ArrayXd& yp_array) { vector_array_.row(3) = yp_array; }
    /**
     * @brief Set the array of longitudinal displacements.
     * @param z_array Array of longitudinal displacements
     */
    void z_array(const Eigen::ArrayXd& z_array) { z_array_ = z_array; }
    /**
     * @brief Set the array of relative momentum deviations.
     * @param delta_array Array of relative momentum deviations
     */
    void delta_array(const Eigen::ArrayXd& delta_array) { delta_array_ = delta_array; }

    // Efficient append (reserve + copy)
    void append(const CoordinateArray &other) noexcept(false);

    // Get Coordinate at given s by linear interpolation
    Coordinate from_s(double s) const noexcept(false);
};
