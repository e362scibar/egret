/**
 * @file dispersion.hpp
 * @brief Definition of the Dispersion class representing dispersion functions.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// dispersion.hpp
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

namespace egret {
    class Dispersion;
}

class egret::Dispersion {
protected:
    //! 4D vector representing dispersion function
    Eigen::Vector4d vector_;
    //! Longitudinal position s
    double s_;

public:
    /**
     * @brief Construct a new Dispersion object.
     * @param v 4D vector representing dispersion function
     * @param s Longitudinal position
     */
    Dispersion(const Eigen::Vector4d &vector=Eigen::Vector4d::Zero(), double s=0.) :
        vector_(vector), s_(s) {}
    /**
     * @brief Destroy the Dispersion object.
     */
    virtual ~Dispersion() = default;

    /**
     * @brief Get the 4D vector representing dispersion function.
     * @return Eigen::Vector4d 4D vector (Dx, Dpx, Dy, Dpy)
     */
    const Eigen::Vector4d& vector() const { return vector_; }
    /**
     * @brief Get the longitudinal position s.
     * @return double Longitudinal position s
     */
    double s() const { return s_; }
    /**
     * @brief Get the x components of the dispersion vector.
     * @return double x component
     */
    double x() const { return vector_(0); };
    /**
     * @brief Get the x' component of the dispersion vector.
     * @return double x' component
     */
    double xp() const { return vector_(1); };
    /**
     * @brief Get the y component of the dispersion vector.
     * @return double y component
     */
    double y() const { return vector_(2); };
    /**
     * @brief Get the y' component of the dispersion vector.
     * @return double y' component
     */
    double yp() const { return vector_(3); };

    /**
     * @brief Set the 4D vector representing dispersion function.
     * @param vector 4D vector representing dispersion function
     */
    void vector(const Eigen::Vector4d &vector) { vector_ = vector; }
    /**
     * @brief Set the longitudinal position s.
     * @param s Longitudinal position s
     */
    void s(const double s) { s_ = s; }
    /**
     * @brief Set the x component of the dispersion vector.
     * @param x x component
     */
    void x(const double x) { vector_(0) = x; };
    /**
     * @brief Set the x' component of the dispersion vector.
     * @param xp x' component
     */
    void xp(const double xp) { vector_(1) = xp; };
    /**
     * @brief Set the y component of the dispersion vector.
     * @param y y component
     */
    void y(const double y) { vector_(2) = y; };
    /**
     * @brief Set the y' component of the dispersion vector.
     * @param yp y' component
     */
    void yp(const double yp) { vector_(3) = yp; };
};
