/**
 * @file quadrupole.hpp
 * @brief Quadrupole magnet class definition
 * @author Hirokazu Maesaka
 * @date 2025
 */
// quadrupole.hpp
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
#include "egret/element.hpp"

namespace egret {
    class Quadrupole;
}

class egret::Quadrupole : public egret::Element {
protected:
    //! Quadrupole strength k1 (1/m^2)
    double k1_;

public:
    /**
     * @brief Construct a new Quadrupole object
     * @param name Object name
     * @param length Quadrupole length
     * @param k1 Quadrupole strength k1 (1/m^2)
     * @param dx Horizontal offset
     * @param dy Vertical offset
     * @param ds Longitudinal offset
     * @param tilt Rotation angle (radians)
     * @param info Additional information string
     */
    Quadrupole(const std::string& name, const double length, const double k1,
        const double dx=0.0, const double dy=0.0, const double ds=0.0,
        const double tilt=0.0, const std::string &info="") :
        Element(name, length, dx, dy, ds, tilt, info), k1_(k1) {}
    /**
     * @brief Destroy the Quadrupole object
     */
    virtual ~Quadrupole() noexcept = default;

    /**
     * @brief Get the quadrupole strength k1.
     * @return double Quadrupole strength k1 (1/m^2)
     */
    double k1() const { return k1_; }

    /**
     * @brief Set the quadrupole strength k1.
     * @param k1 Quadrupole strength k1 (1/m^2)
     */
    void k1(const double k1) { k1_ = k1; }

    // Calculate transfer matrix for this quadrupole element
    Eigen::Matrix4d transfer_matrix(const std::optional<Coordinate> &cood0,
        double ds=0.1) const noexcept(false) override;

    // Calculate transfer matrix array along this quadrupole element
    std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0,
        double ds=0.1, bool endpoint = false) const noexcept(false) override;

    // Calculate additive dispersion vector for given initial coordinate
    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0,
        double ds=0.1) const noexcept(false) override;

    // Calculate additive dispersion vector array for given initial coordinate
    std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
    dispersion_array(const std::optional<Coordinate> &cood0,
        double ds=0.1, bool endpoint = false) const noexcept(false) override;
};
