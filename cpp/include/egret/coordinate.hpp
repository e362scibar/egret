/**
 * @file coordinate.hpp
 * @brief Class representing a particle coordinate in phase space
 * @author Hirokazu Maesaka
 * @date 2025
 */
// coordinate.hpp
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
    class Coordinate;
}

/**
 * @brief Class representing a particle coordinate in phase space.
 */
class egret::Coordinate {
protected:
    //! Particle coordinate vector: (x, xp, y, yp)
    Eigen::Vector4d vector_;
    //! Longitudinal position s, longitudinal displacement z, and relative momentum deviation delta
    double s_{0.0}, z_{0.0}, delta_{0.0};
public:
    /**
     * @brief Construct a new Coordinate object.
     * @param v Particle coordinate vector (x, xp, y, yp)
     * @param s Longitudinal position
     * @param z Longitudinal displacement
     * @param delta Relative momentum deviation
     */
    Coordinate(const Eigen::Vector4d &vector=Eigen::Vector4d::Zero(), double s=0., double z=0., double delta=0.)
        : vector_(vector), s_(s), z_(z), delta_(delta) {}
    /**
     * @brief Destroy the Coordinate object.
     */
    virtual ~Coordinate() = default;
    /**
     * @brief Get the x coordinate.
     * @return double x coordinate
     */
    double x() const { return vector_(0); }
    /**
     * @brief Get the x' coordinate.
     * @return double x' coordinate
     */
    double xp() const { return vector_(1); }
    /**
     * @brief Get the y coordinate.
     * @return double y coordinate
     */
    double y() const { return vector_(2); }
    /**
     * @brief Get the y' coordinate.
     * @return double y' coordinate
     */
    double yp() const { return vector_(3); }
    /**
     * @brief Get the longitudinal position s.
     * @return double Longitudinal position s
     */
    double s() const { return s_; }
    /**
     * @brief Get the longitudinal displacement z.
     * @return double Longitudinal displacement z
     */
    double z() const { return z_; }
    /**
     * @brief Get the relative momentum deviation delta.
     * @return double Relative momentum deviation delta
     */
    double delta() const { return delta_; }
    /**
     * @brief Set the x coordinate.
     * @param val x coordinate value
     */
    void x(double val) { vector_(0) = val; }
    /**
     * @brief Set the x' coordinate.
     * @param val x' coordinate value
     */
    void xp(double val) { vector_(1) = val; }
    /**
     * @brief Set the y coordinate.
     * @param val y coordinate value
     */
    void y(double val) { vector_(2) = val; }
    /**
     * @brief Set the y' coordinate.
     * @param val y' coordinate value
     */
    void yp(double val) { vector_(3) = val; }
    /**
     * @brief Set the longitudinal position s.
     * @param val longitudinal position s value
     */
    void s(double val) { s_ = val; }
    /**
     * @brief Set the longitudinal displacement z.
     * @param val longitudinal displacement z value
     */
    void z(double val) { z_ = val; }
    /**
     * @brief Set the relative momentum deviation delta.
     * @param val relative momentum deviation delta value
     */
    void delta(double val) { delta_ = val; }
};
