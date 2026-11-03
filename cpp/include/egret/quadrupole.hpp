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

#include "egret/element.hpp"

namespace egret {
    class Quadrupole;
}

class egret::Quadrupole : public egret::Element {
public:
    //! Threshold to ignore quadrupole strength
    static constexpr double IGNORE_K1_THRESHOLD = 1.0e-15;

protected:
    //! Quadrupole strength k1 (1/m^2)
    double k1_;
    //! Horizontal kick angle of the steering coil (rad)
    double kick_x_{0.0};
    //! Vertical kick angle of the steering coil (rad)
    double kick_y_{0.0};

    /**
     * @brief Steering kick angles at a relative momentum deviation. The kick scales as 1 / (1 + delta).
     * @param delta Relative momentum deviation
     * @return std::array<double, 2> Horizontal and vertical kick angles (rad)
     */
    std::array<double, 2> kick_at(double delta) const;

    /**
     * @brief Transfer matrix of a section of the given length at a relative momentum deviation.
     * @param length Length of the section (m)
     * @param delta Relative momentum deviation
     * @return Eigen::Matrix4d 4x4 transfer matrix
     */
    Eigen::Matrix4d matrix(double length, double delta) const noexcept(false);

    /**
     * @brief Derivative of the transfer matrix with respect to the relative momentum deviation.
     * @param length Length of the section (m)
     * @param delta Relative momentum deviation
     * @return Eigen::Matrix4d dM / d(delta)
     */
    Eigen::Matrix4d matrix_derivative(double length, double delta) const noexcept(false);

    /**
     * @brief Transverse offset of the magnetic center that realizes the steering kick [m].
     *        Evaluated once at the reference momentum (delta = 0); zero for a negligible gradient or no kick.
     * @return std::array<double, 2> Horizontal and vertical offset (m)
     * @throws std::invalid_argument if the transfer block is singular and a kick is requested.
     */
    std::array<double, 2> steering_offset() const noexcept(false);

    /**
     * @brief Distributed dipole kick accumulated over the first s meters, for a negligible gradient.
     * @param s Longitudinal position inside the element (m)
     * @param delta Relative momentum deviation
     * @return Eigen::Vector4d Vector [x, x', y, y'] added to the drifted coordinate
     */
    Eigen::Vector4d distributed_kick(double s, double delta) const;

    /**
     * @brief Additive dispersion of a section of the given length, including the momentum dependence of the
     *        steering offset and of the kick.
     * @param vector Vector relative to the element offset and steering offset
     * @param length Length of the section (m)
     * @param delta Relative momentum deviation
     * @return Eigen::Vector4d Dispersion vector
     */
    Eigen::Vector4d dispersion_at(const Eigen::Vector4d &vector, double length, double delta) const noexcept(false);

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
     * @param kick_x Horizontal kick angle of the steering coil (radians)
     * @param kick_y Vertical kick angle of the steering coil (radians)
     */
    Quadrupole(const std::string& name, const double length, const double k1,
        const double dx=0.0, const double dy=0.0, const double ds=0.0,
        const double tilt=0.0, const std::string &info="",
        const double kick_x=0.0, const double kick_y=0.0) :
        Element(name, length, 0.0, dx, dy, ds, tilt, info), k1_(k1), kick_x_(kick_x), kick_y_(kick_y) {}
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

    /**
     * @brief Get the horizontal kick angle of the steering coil.
     * @return double Kick angle (radians)
     */
    double kick_x() const { return kick_x_; }
    /**
     * @brief Set the horizontal kick angle of the steering coil.
     * @param kick_x Kick angle (radians)
     */
    void kick_x(const double kick_x) { kick_x_ = kick_x; }
    /**
     * @brief Get the vertical kick angle of the steering coil.
     * @return double Kick angle (radians)
     */
    double kick_y() const { return kick_y_; }
    /**
     * @brief Set the vertical kick angle of the steering coil.
     * @param kick_y Kick angle (radians)
     */
    void kick_y(const double kick_y) { kick_y_ = kick_y; }
    /**
     * @brief Get the horizontal steering strength k0x = -kick_x / length.
     * @return double Steering strength (1/m)
     */
    double k0x() const { return -kick_x_ / length_; }
    /**
     * @brief Set the horizontal steering strength; sets kick_x = -k0x * length.
     * @param k0x Steering strength (1/m)
     */
    void k0x(const double k0x) { kick_x_ = -k0x * length_; }
    /**
     * @brief Get the vertical steering strength k0y = -kick_y / length.
     * @return double Steering strength (1/m)
     */
    double k0y() const { return -kick_y_ / length_; }
    /**
     * @brief Set the vertical steering strength; sets kick_y = -k0y * length.
     * @param k0y Steering strength (1/m)
     */
    void k0y(const double k0y) { kick_y_ = -k0y * length_; }

    /**
     * @brief Set the steering coil kick angles. A kick that is std::nullopt is left unchanged.
     * @param kick_x Horizontal kick angle (radians), or std::nullopt
     * @param kick_y Vertical kick angle (radians), or std::nullopt
     */
    void set_steering(std::optional<double> kick_x, std::optional<double> kick_y) noexcept;

    // Calculate transfer matrix for a quadrupole magnet
    static Eigen::Matrix4d transfer_matrix(double length, double k1,
        const std::optional<Eigen::Matrix4d> &rmat = std::nullopt) noexcept(false);

    // Calculate rotation matrix of a transfer matrix for a given tilt angle
    static Eigen::Matrix4d rotation_matrix(double tilt) noexcept(false);

    // Calculate transfer matrix for this quadrupole element
    Eigen::Matrix4d transfer_matrix(const std::optional<Coordinate> &cood0, double ds=0.1,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Calculate transfer matrix array along this quadrupole element
    std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0,
        double ds=0.1,bool endpoint = false,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    static Eigen::Vector4d dispersion(const Eigen::Vector4d &cood0_vec,
        double length, double k1, double tilt=0.0) noexcept(false);

    // Calculate additive dispersion vector for given initial coordinate
    Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Calculate additive dispersion vector array for given initial coordinate
    std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
    dispersion_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1, bool endpoint = false,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Transfer a single coordinate through the quadrupole (steering kick realized by the magnetic-center offset)
    std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>
    transfer(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Transfer a coordinate array along the quadrupole
    std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>
    transfer_array(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, bool endpoint=false,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false) override;

    // Calculate transferred coordinates through a quadrupole magnet
    static std::tuple<Eigen::Vector4d, std::optional<Eigen::Matrix4d>,
        std::optional<Eigen::Vector4d>>
    transfer(const Eigen::Vector4d &cood0_vec, double length, double k1,
        double k0x=0.0, double k0y=0.0, double tilt=0.0,
        bool tmat_flag=false, bool disp_flag=false) noexcept(false);

    // Polymorphic clone
    std::shared_ptr<Element> clone() const override { return std::make_shared<Quadrupole>(*this); }

    /**
     * @brief Calculate radiation integrals contribution through the quadrupole element.
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
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const override{
        (void)cood0; // unused parameter
        (void)evlp0; // unused parameter
        (void)disp0; // unused parameter
        (void)ds; // unused parameter
        (void)method; // unused parameter
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
};
