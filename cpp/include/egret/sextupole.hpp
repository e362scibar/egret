/**
 * @file sextupole.hpp
 * @brief Definition of the Sextupole element class.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// sextupole.hpp
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

#include "egret/element.hpp"

namespace egret {
    class Sextupole;
}

class egret::Sextupole : public egret::Element {
public:
    static constexpr double IGNORE_QUADRUPOLE_K1_THRESHOLD = 1.0e-20;

protected:
    //! Sextupole strength k2 (1/m^3)
    double k2_;
    //! Steering dipole strength in x direction
    double k0x_;
    //! Steering dipole strength in y direction
    double k0y_;

public:
    /**
     * @brief Construct a new Sextupole object.
     * @param name Name of the sextupole element
     * @param length Length of the sextupole element
     * @param k2 Sextupole strength (1/m^3)
     * @param kick_x Steering kick in x direction
     * @param kick_y Steering kick in y direction
     * @param dx Horizontal offset
     * @param dy Vertical offset
     * @param ds Longitudinal offset
     * @param tilt Tilt angle
     * @param info Additional info string
     */
    Sextupole(const std::string& name, const double length,
        const double k2, const double kick_x=0.0, const double kick_y=0.0,
        double dx = 0.0, double dy = 0.0, double ds = 0.0,
        double tilt = 0.0, const std::string& info = "") :
        Element(name, length, dx, dy, ds, tilt, info), k2_(k2) {
        set_steering(kick_x, kick_y);
    }
    /**
     * @brief Destroy the Sextupole object.
     */
    virtual ~Sextupole() noexcept = default;

    /**
     * @brief Get the sextupole strength k2.
     * @return double Sextupole strength k2 (1/m^3)
     */
    double k2() const { return k2_; }
    /**
     * @brief Get the steering dipole strength in x direction.
     * @return double Steering dipole strength in x direction (1/m)
     */
    double k0x() const { return k0x_; }
    /**
     * @brief Get the steering dipole strength in y direction.
     * @return double Steering dipole strength in y direction (1/m)
     */
    double k0y() const { return k0y_; }
    /**
     * @brief Get the steering kick angle in x direction.
     * @return double Steering kick in x direction (radians)
     */
    double kick_x() const { return -k0x_ * length_; }
    /**
     * @brief Get the steering kick angle in y direction.
     * @return double Steering kick in y direction (radians)
     */
    double kick_y() const { return -k0y_ * length_; }

    /**
     * @brief Set the sextupole strength k2.
     * @param k2 Sextupole strength k2 (1/m^3)
     */
    void k2(const double k2) { k2_ = k2; }
    /**
     * @brief Set the steering dipole strength in x direction.
     * @param k0x Steering dipole strength in x direction (1/m)
     */
    void k0x(const double k0x) { k0x_ = k0x; }
    /**
     * @brief Set the steering dipole strength in y direction.
     * @param k0y Steering dipole strength in y direction (1/m)
     */
    void k0y(const double k0y) { k0y_ = k0y; }
    /**
     * @brief Set the steering kick angle in x direction.
     * @param kick_x Steering kick in x direction (radians)
     */
    void kick_x(const double kick_x) { k0x_ = -kick_x / length_; }
    /**
     * @brief Set the steering kick angle in y direction.
     * @param kick_y Steering kick in y direction (radians)
     */
    void kick_y(const double kick_y) { k0y_ = -kick_y / length_; }

    /**
     * @brief Set the steering kick angles if provided.
     * @param kick_x Steering kick in x direction or std::nullopt to leave unchanged
     * @param kick_y Steering kick in y direction or std::nullopt to leave unchanged
     */
    void set_steering(const std::optional<double> &kick_x,
        const std::optional<double> &kick_y) {
        if (kick_x) {
            k0x_ = -(*kick_x) / length_;
        }
        if (kick_y) {
            k0y_ = -(*kick_y) / length_;
        }
    }

    // Calculate coordinate, transfer matrix, and dispersion by midpoint method.
    std::tuple<Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>>
    transfer_by_midpoint_method(const Coordinate &cood0, double ds=0.1,
        bool tmat_flag=true, bool disp_flag=false) const noexcept(false);

    // Calculate transfer matrix of the sextupole.
    Eigen::Matrix4d transfer_matrix(const std::optional<Coordinate> &cood0,
        double ds=0.1) const noexcept(false) override;

    // Calculate transfer matrix array of the sextupole.
    std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0,
        double ds=0.1, bool endpoint = false) const noexcept(false) override;

    // Calculate additive dispersion of the sextupole.
    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0,
        double ds=0.1) const noexcept(false) override;

    // Calculate additive dispersion array of the sextupole.
    std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
    dispersion_array(const std::optional<Coordinate> &cood0,
        double ds=0.1, bool endpoint = false) const noexcept(false) override;

    // Calculate coordinate, envelope, and dispersion after the sextupole.
    std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>
    transfer(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1) const noexcept(false) override;

    // Calculate coordinate array, envelope array, and dispersion array along the sextupole.
    std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>
    transfer_array(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, bool endpoint=false) const noexcept(false) override;
};
