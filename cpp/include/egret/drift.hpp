/**
 * @file drift.hpp
 * @brief Drift element class definition
 * @author Hirokazu Maesaka
 * @date 2025
 */
// drift.hpp
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
    class Drift;
}
class egret::Drift: public egret::Element {
public:
    /**
     * @brief Construct a new Drift object
     * @param name Object name
     * @param length Length of the drift space
     */
    Drift(const std::string &name, const double length,
        const double dx=0.0, const double dy=0.0, const double ds=0.0,
        const double tilt=0.0, const std::string &info="") :
        Element(name, length, 0.0, dx, dy, ds, tilt, info) {}
    /**
     * @brief Destroy the Drift object
     */
    virtual ~Drift() noexcept = default;

    // Return 4x4 transfer matrix for a drift of a given length
    static Eigen::Matrix4d transfer_matrix(double length) noexcept(false);

    // Return an array of 4x4 transfer matrices for a drift of a given length
    static std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(double length, double ds = 0.1, bool endpoint = false)
    noexcept(false);

    /**
     * @brief Get the transfer matrix for this drift element.
     * @return Eigen::Matrix4d Transfer matrix (4x4)
     */
    Eigen::Matrix4d transfer_matrix(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1) const noexcept(false) override {
        (void)cood0; // unused parameter
        (void)ds; // unused parameter
        return transfer_matrix(length_);
    }

    /**
     * @brief Get an array of transfer matrices for this drift element.
     * @param ds Step size
     * @param endpoint Include endpoint
     * @return std::vector<Eigen::Matrix4d> Array of transfer matrices
     */
    std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        const double ds = 0.1, const bool endpoint = false) const noexcept(false) override {
        (void)cood0; // unused parameter
        return transfer_matrix_array(length_, ds, endpoint);
    }

    /**
     * @brief Clone the Drift object.
     * @return std::shared_ptr<Element> Shared pointer to the cloned Drift object
     */
    std::shared_ptr<Element> clone() const noexcept override { return std::make_shared<Drift>(*this); }

    /**
     * @brief Calculate radiation integrals contribution through the drift element.
     * @param cood0 Initial coordinate
     * @param evlp0 Initial envelope
     * @param disp0 Initial dispersion
     * @param ds Step size for integration
     * @return std::tuple<double, double, double, double, double, double>
     *         Radiation integrals (I1, I2, I3, I4, I5, I6)
     */
    std::tuple<double, double, double, double, double, double>
    radiation_integrals(const Coordinate &cood0, const Envelope &evlp0, const Dispersion &disp0,
        double ds=0.1) const override{
        (void)cood0; // unused parameter
        (void)evlp0; // unused parameter
        (void)disp0; // unused parameter
        (void)ds; // unused parameter
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
};
