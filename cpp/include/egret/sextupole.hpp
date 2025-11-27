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

#include "egret/nonlinearmultipole.hpp"

namespace egret {
    class Sextupole;
}

class egret::Sextupole : public egret::NonlinearMultipole {
protected:
    //! Sextupole strength k2 (1/m^3)
    double k2_;

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
        NonlinearMultipole(name, length, kick_x, kick_y, dx, dy, ds, tilt, info),
        k2_(k2) {}
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
     * @brief Set the sextupole strength k2.
     * @param k2 Sextupole strength k2 (1/m^3)
     */
    void k2(const double k2) { k2_ = k2; }

    // Calculate dipole and quadrupole field strengths
    // (x'+jy' = - k0 L - k1 L x + j k1 L y)
    std::tuple<std::complex<double>, std::complex<double>>
    get_k(const Coordinate &cood) const noexcept override;

    // Polymorphic clone
    std::shared_ptr<Element> clone() const override { return std::make_shared<Sextupole>(*this); }
};
