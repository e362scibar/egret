/**
 * @file steering.hpp
 * @brief Steering magnet class definition
 * @author Hirokazu Maesaka
 * @date 2025
 */
// steering.hpp
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
#include "egret/drift.hpp"

namespace egret {
    class Steering;
}

class egret::Steering : public egret::Element {
protected:
    //!< Kick angle in the x direction (radians)
    double kick_x_;
    //!< Kick angle in the y direction (radians)
    double kick_y_;

    // Compute effective kick angles considering tilt and momentum deviation
    std::tuple<double, double> tilted_kick(double delta=0.0) const noexcept;

public:
    /**
     * @brief Construct a new Steering object
     * @param name Object name
     * @param length Length of the steering magnet
     * @param kick_x Kick angle in the x direction (radians)
     * @param kick_y Kick angle in the y direction (radians
     * @param dx Transverse offset in x
     * @param dy Transverse offset in y
     * @param ds Longitudinal offset
     * @param tilt Tilt angle
     * @param info Additional info string
     */
    Steering(const std::string &name, const double length,
        const double kick_x=0.0, const double kick_y=0.0,
        const double dx=0.0, const double dy=0.0, const double ds=0.0,
        const double tilt=0.0, const std::string &info="") :
        Element(name, length, 0.0, dx, dy, ds, tilt, info),
        kick_x_(kick_x), kick_y_(kick_y) {}
    /**
     * @brief Destroy the Steering object
     */
    virtual ~Steering() noexcept = default;

    /**
     * @brief Get the kick angle in the x direction
     * @return double Kick angle in the x direction (radians)
     */
    double kick_x() const { return kick_x_; }
    /**
     * @brief Get the kick angle in the y direction
     * @return double Kick angle in the y direction (radians)
     */
    double kick_y() const { return kick_y_; }
    /**
     * @brief Get the kick angles in both x and y directions
     * @return std::tuple<double, double> Kick angles (kick_x, kick_y) in radians
     */
    std::tuple<double, double> kick() const {
        return std::make_tuple(kick_x_, kick_y_);
    }

    /**
     * @brief Set the kick angle in the x direction
     * @param kick_x Kick angle in the x direction (radians)
     */
    void kick_x(const double kick_x) { kick_x_ = kick_x; }
    /**
     * @brief Set the kick angle in the y direction
     * @param kick_y Kick angle in the y direction (radians)
     */
    void kick_y(const double kick_y) { kick_y_ = kick_y; }
    /**
     * @brief Set the kick angles in both x and y directions
     * @param kick_x Kick angle in the x direction (radians)
     * @param kick_y Kick angle in the y direction (radians)
     */
    void kick(const double kick_x, const double kick_y) {
        kick_x_ = kick_x;
        kick_y_ = kick_y;
    }

    /**
     * @brief Set the kick angles in both x and y directions if provided.
     * @param kick_x Kick angle in the x direction (radians) or std::nullopt to leave unchanged
     * @param kick_y Kick angle in the y direction (radians) or std::nullopt to leave unchanged
     */
    void set_steering(const std::optional<double> &kick_x,
        const std::optional<double> &kick_y) {
        if (kick_x) {
            kick_x_ = *kick_x;
        }
        if (kick_y) {
            kick_y_ = *kick_y;
        }
    }

    /**
     * @brief Get the transfer matrix for the steering magnet.
     * @param cood0 Input coordinate (not used here)
     * @param ds Maximum step size for integration (not used here)
     * @param method Integration method (not used here)
     * @return Eigen::Matrix4d Transfer matrix
     */
    Eigen::Matrix4d transfer_matrix(std::optional<Coordinate> const &cood0 = std::nullopt,
        const double ds=0.1, IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const noexcept(false) override {
        (void)cood0; // unused parameter
        (void)ds; // unused parameter
        (void)method; // unused parameter
        return Drift::transfer_matrix(length_);
    }

    /**
     * @brief Get an array of transfer matrices for the steering magnet.
     * @param cood0 Input coordinate (not used here)
     * @param ds Maximum step size for integration (not used here)
     * @param endpoint Whether to include the endpoint
     * @param method Integration method (not used here)
     * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Array of transfer matrices and s array
     */
    std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        const double ds = 0.1, const bool endpoint = false,
        const IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const noexcept(false) override {
        (void)cood0; // unused parameter
        (void)method; // unused parameter
        return Drift::transfer_matrix_array(length_, ds, endpoint);
    }

    // Additive dispersion function at the end of the steering magnet.
    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0 = std::nullopt, double ds=0.1,
        const IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Array of additive dispersion functions along the steering magnet.
    std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
    dispersion_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1, bool endpoint=false,
        IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Calculate Coordinate, Envelope, and Dispersion after the steering magnet.
    std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>
    transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt, double ds=0.1,
        IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Calculate CoordinateArray, EnvelopeArray, and DispersionArray along the steering magnet.
    std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>
    transfer_array(const Coordinate &cood0, const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, bool endpoint=false,
        IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    /**
     * @brief Clone the Steering object.
     * @return std::shared_ptr<Element> Shared pointer to the cloned Steering object
     */
    std::shared_ptr<Element> clone() const noexcept override { return std::make_shared<Steering>(*this); }

    /**
     * @brief Calculate radiation integrals contribution through the steering element.
     * @param cood0 Initial coordinate
     * @param evlp0 Initial envelope
     * @param disp0 Initial dispersion
     * @param ds Step size for integration
     * @param method Integration method (not used here)
     * @return std::tuple<double, double, double, double, double, double>
     *         Radiation integrals (I1, I2, I3, I4, I5, I6)
     */
    std::tuple<double, double, double, double, double, double>
    radiation_integrals(const Coordinate &cood0, const Envelope &evlp0, const Dispersion &disp0,
        double ds=0.1, const IntegrationMethod method = IntegrationMethod::SYMPLECTIC4) const override{
        (void)cood0; // unused parameter
        (void)evlp0; // unused parameter
        (void)disp0; // unused parameter
        (void)ds; // unused parameter
        (void)method; // unused parameter
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
};
