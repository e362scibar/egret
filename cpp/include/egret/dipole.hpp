/**
 * @file dipole.hpp
 * @brief Dipole element class
 * @author Hirokazu Maesaka
 * @date 2025
 */
// dipole.hpp
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
    class Dipole;
}
class egret::Dipole : public egret::Element {
protected:
    //! Quadrupole component strength k1 (1/m^2)
    double k1_;
    //! Entrance/exit edge angle (radians)
    double e1_, e2_;
    //! Entrance/exit pole-face curvature (1/m)
    double h1_, h2_;

public:
    /**
     * @brief Construct a new Dipole object.
     * @param name Name of the dipole element
     * @param length Length of the dipole element
     * @param angle Bending angle (radians)
     * @param k1 Quadrupole component strength k1 (1/m^2)
     * @param e1 Entrance edge angle (radians)
     * @param e2 Exit edge angle (radians)
     * @param h1 Entrance pole-face curvature (1/m)
     * @param h2 Exit pole-face curvature (1/m)
     * @param dx Horizontal displacement (m)
     * @param dy Vertical displacement (m)
     * @param ds Longitudinal displacement (m)
     * @param tilt Tilt angle (radians)
     * @param info Additional information string
     */
    Dipole(const std::string &name, const double length, const double angle,
        const double k1=0.0, const double e1=0.0, const double e2=0.0,
        const double h1=0.0, const double h2=0.0,
        const double dx=0.0, const double dy=0.0, const double ds=0.0,
        const double tilt=0.0, const std::string &info="") :
        Element(name, length, angle, dx, dy, ds, tilt, info),
        k1_(k1), e1_(e1), e2_(e2), h1_(h1), h2_(h2) {}
    /**
     * @brief Destroy the Dipole object
     */
    virtual ~Dipole() noexcept = default;

    /**
     * @brief Get the quadrupole component strength k1 of the dipole.
     * @return double Quadrupole component strength k1 (1/m^2)
     */
    double k1() const { return k1_; }
    /**
     * @brief Get the entrance edge angle of the dipole.
     * @return double Entrance edge angle (radians)
     */
    double e1() const { return e1_; }
    /**
     * @brief Get the exit edge angle of the dipole.
     * @return double Exit edge angle (radians)
     */
    double e2() const { return e2_; }
    /**
     * @brief Get the entrance pole-face curvature of the dipole.
     * @return double Entrance pole-face curvature (1/m)
     */
    double h1() const { return h1_; }
    /**
     * @brief Get the exit pole-face curvature of the dipole.
     * @return double Exit pole-face curvature (1/m)
     */
    double h2() const { return h2_; }
    /**
     * @brief Get the bending radius of the dipole.
     * @return double Bending radius (m)
     */
    double rho() const { return length_ / angle_; }

    /**
     * @brief Set the quadrupole component strength k1 of the dipole.
     * @param k1 Quadrupole component strength k1 (1/m^2)
     */
    void k1(const double k1) { k1_ = k1; }
    /**
     * @brief Set the entrance edge angle of the dipole.
     * @param e1 Entrance edge angle (radians)
     */
    void e1(const double e1) { e1_ = e1; }
    /**
     * @brief Set the exit edge angle of the dipole.
     * @param e2 Exit edge angle (radians)
     */
    void e2(const double e2) { e2_ = e2; }
    /**
     * @brief Set the entrance pole-face curvature of the dipole.
     * @param h1 Entrance pole-face curvature (1/m)
     */
    void h1(const double h1) { h1_ = h1; }
    /**
     * @brief Set the exit pole-face curvature of the dipole.
     * @param h2 Exit pole-face curvature (1/m)
     */
    void h2(const double h2) { h2_ = h2; }

    // Calculate transfer matrix given initial coordinates
    Eigen::Matrix4d transfer_matrix(
        const std::optional<Coordinate> &cood0, double ds=0.1) const noexcept(false) override;

    // Calculate array of transfer matrices along the dipole
    std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> transfer_matrix_array(
        const std::optional<Coordinate> &cood0, double ds=0.1,
        bool endpoint=false) const noexcept(false) override;

    // Calculate additive dispersion function at the end of the dipole
    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0, double ds=0.1)
    const noexcept(false) override;

    // Calculate array of additive dispersion functions along the dipole
    std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> dispersion_array(
        const std::optional<Coordinate> &cood0, double ds=0.1, bool endpoint=false)
    const noexcept(false) override;

    // Transfer coordinate, envelope, and dispersion through the dipole
    std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>
    transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1) const noexcept(false) override;

    // Transfer arrays of coordinate, envelope, and dispersion through the dipole
    std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>
    transfer_array(const Coordinate &cood0, const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, bool endpoint=false) const noexcept(false) override;

    // Calculate radiation integrals contribution through the dipole
    std::tuple<double, double, double, double, double, double>
    radiation_integrals(const Coordinate &cood0, const Envelope &evlp0, const Dispersion &disp0,
        double ds=0.1) const noexcept(false) override;
};
