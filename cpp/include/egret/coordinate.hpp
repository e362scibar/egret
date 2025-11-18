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
    double s_, z_, delta_;
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
     * @brief Get the particle coordinate vector (x, xp, y, yp).
     * @return const Eigen::Vector4d& Particle coordinate vector.
     */
    const Eigen::Vector4d& vector() const { return vector_; }
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
     * @brief Set the particle coordinate vector (x, xp, y, yp).
     * @param vector Particle coordinate vector.
     */
    void vector(const Eigen::Vector4d &vector) { vector_ = vector; }
    /**
     * @brief Set the x coordinate.
     * @param x x coordinate value
     */
    void x(double x) { vector_(0) = x; }
    /**
     * @brief Set the x' coordinate.
     * @param xp x' coordinate value
     */
    void xp(double xp) { vector_(1) = xp; }
    /**
     * @brief Set the y coordinate.
     * @param y y coordinate value
     */
    void y(double y) { vector_(2) = y; }
    /**
     * @brief Set the y' coordinate.
     * @param yp y' coordinate value
     */
    void yp(double yp) { vector_(3) = yp; }
    /**
     * @brief Set the longitudinal position s.
     * @param s longitudinal position s value
     */
    void s(double s) { s_ = s; }
    /**
     * @brief Set the longitudinal displacement z.
     * @param z longitudinal displacement z value
     */
    void z(double z) { z_ = z; }
    /**
     * @brief Set the relative momentum deviation delta.
     * @param delta relative momentum deviation delta value
     */
    void delta(double delta) { delta_ = delta; }
};
