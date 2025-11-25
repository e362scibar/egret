/**
 * @file octupole.hpp
 * @brief Definition of the Octupole element class.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// octupole.hpp
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

#include "egret/nonlinearmultipole.hpp"

namespace egret {
    class Octupole;
}

/**
 * @brief Class for octupole elements.
 */
class egret::Octupole : public egret::NonlinearMultipole {
protected:
    //! Octupole strength k3 (1/m^4)
    double k3_;
    //! Quadrupole strength (1/m^2)
    double k1_;
    //! Tilt angle of the quadrupole component
    double tilt_quad_;

public:
    /**
     * @brief Construct a new Octupole object.
     * @param name Name of the octupole element
     * @param length Length of the octupole element
     * @param k3 Octupole strength (1/m^4)
     * @param k1 Quadrupole strength (1/m^2)
     * @param tilt_quad Tilt angle of the quadrupole component
     * @param kick_x Steering kick in x direction
     * @param kick_y Steering kick in y direction
     * @param dx Horizontal offset
     * @param dy Vertical offset
     * @param ds Longitudinal offset
     * @param tilt Tilt angle
     * @param info Additional info string
     */
    Octupole(const std::string& name, const double length,
        const double k3, const double k1=0.0, const double tilt_quad=0.0,
        const double kick_x=0.0, const double kick_y=0.0,
        double dx = 0.0, double dy = 0.0, double ds = 0.0,
        double tilt = 0.0, const std::string& info = "") :
        NonlinearMultipole(name, length, kick_x, kick_y, dx, dy, ds, tilt, info),
        k3_(k3), k1_(k1), tilt_quad_(tilt_quad) {}
    /**
     * @brief Destroy the Octupole object.
     */
    virtual ~Octupole() noexcept = default;

    /**
     * @brief Get the octupole strength k3.
     * @return double Octupole strength k3 (1/m^4)
     */
    double k3() const { return k3_; }
    /**
     * @brief Get the quadrupole strength.
     * @return double Quadrupole strength (1/m^2)
     */
    double k1() const { return k1_; }
    /**
     * @brief Get the tilt angle of the quadrupole component.
     * @return double Tilt angle of the quadrupole component
     */
    double tilt_quad() const { return tilt_quad_; }

    /**
     * @brief Set the octupole strength k3.
     * @param k3 Octupole strength k3 (1/m^4)
     */
    void k3(const double k3) { k3_ = k3; }
    /**
     * @brief Set the quadrupole strength.
     * @param k1 Quadrupole strength (1/m^2)
     */
    void k1(const double k1) { k1_ = k1; }
    /**
     * @brief Set the tilt angle of the quadrupole component.
     * @param tilt_quad Tilt angle of the quadrupole component
     */
    void tilt_quad(const double tilt_quad) { tilt_quad_ = tilt_quad; }

    /**
     * @brief Set the quadrupole parameters if provided.
     * @param k1 Quadrupole strength (1/m^2) or std::nullopt to leave unchanged
     * @param tilt_quad Tilt angle of the quadrupole component or std::nullopt to leave unchanged
     */
    void set_quadrupole(const std::optional<double> &k1 = std::nullopt,
        const std::optional<double> &tilt_quad = std::nullopt) {
        if (k1) {
            k1_ = *k1;
        }
        if (tilt_quad) {
            tilt_quad_ = *tilt_quad;
        }
    }

    // Calculate dipole and quadrupole field strengths.
    // (x'+jy' = - k0 L - k1 L x + j k1 L y)
    std::tuple<std::complex<double>, std::complex<double>>
    get_k(const Coordinate &cood) const noexcept override;
};
