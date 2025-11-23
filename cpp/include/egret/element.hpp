/**
 * @file element.hpp
 * @brief Base class for an accelerator element in the Egret framework.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// element.hpp
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

#include <memory>
#include <tuple>
#include <optional>
#include <list>
#include "egret/object.hpp"
#include "egret/coordinate.hpp"
#include "egret/coordinatearray.hpp"
#include "egret/envelope.hpp"
#include "egret/envelopearray.hpp"
#include "egret/dispersion.hpp"
#include "egret/dispersionarray.hpp"

namespace egret {
    class Element;
}

/**
 * @brief Base class for an accelerator element in the Egret framework.
 */
class egret::Element: public egret::Object {
protected:
    //! Length of the element
    double length_;
    //! Transverse and longitudinal offsets and tilt
    double dx_, dy_, ds_, tilt_;
    //! Additional info string
    std::string info_;
    //! List of child elements (for composite elements)
    std::optional<std::list<std::shared_ptr<Element>>> elements_{std::nullopt};

public:
    /**
     * @brief Construct a new Element object
     * @param name Object name
     * @param length Length of the element
     * @param dx Transverse offset in x
     * @param dy Transverse offset in y
     * @param ds Longitudinal offset
     * @param tilt Tilt angle
     * @param info Additional info string
     */
    Element(const std::string &name, double length,
            double dx=0.0, double dy=0.0, double ds=0.0, double tilt=0.0,
            const std::string &info="")
        : Object(name), length_(length), dx_(dx), dy_(dy), ds_(ds), tilt_(tilt), info_(info) {}
    /**
     * @brief Virtual destructor
     */
    virtual ~Element() noexcept = default;

    /**
     * @brief Get the length of the element.
     * @return double Length of the element
     */
    double length() const { return length_; }
    /**
     * @brief Get the transverse offset in x.
     * @return double Transverse offset in x
     */
    double dx() const { return dx_; }
    /**
     * @brief Get the transverse offset in y.
     * @return double Transverse offset in y
     */
    double dy() const { return dy_; }
    /**
     * @brief Get the longitudinal offset.
     * @return double Longitudinal offset
     */
    double ds() const { return ds_; }
    /**
     * @brief Get the tilt angle.
     * @return double Tilt angle
     */
    double tilt() const { return tilt_; }
    /**
     * @brief Get the additional info string.
     * @return const std::string& Additional info string
     */
    const std::string& info() const { return info_; }
    /**
     * @brief Get the list of child elements.
     * @return const std::optional<std::list<std::shared_ptr<Element>>>& List of child elements
     */
    const std::optional<std::list<std::shared_ptr<Element>>>& elements() const { return elements_; }

    /**
     * @brief Set the length of the element.
     * @param length Length of the element
     */
    void length(double length) { length_ = length; }
    /**
     * @brief Set the transverse offset in x.
     * @param dx Transverse offset in x
     */
    void dx(double dx) { dx_ = dx; }
    /**
     * @brief Set the transverse offset in y.
     * @param dy Transverse offset in y
     */
    void dy(double dy) { dy_ = dy; }
    /**
     * @brief Set the longitudinal offset.
     * @param ds Longitudinal offset
     */
    void ds(double ds) { ds_ = ds; }
    /**
     * @brief Set the tilt angle.
     * @param tilt Tilt angle
     */
    void tilt(double tilt) { tilt_ = tilt; }
    /**
     * @brief Set the additional info string.
     * @param info Additional info string
     */
    void info(const std::string &info) { info_ = info; }

    // Get s array based on length and step size.
    static Eigen::ArrayXd s_array(double length, double ds=0.1, bool endpoint=false) noexcept;

    // Get s array for the element based on length and step size.
    Eigen::ArrayXd s_array(double ds=0.1, bool endpoint=false) const noexcept {
        return s_array(length_, ds, endpoint);
    }

    /**
     * @brief Get the transfer matrix for a given coordinate and step size.
     * Just returns identity matrix in the base class.
     * @param cood0 Initial coordinate
     * @param ds Maximum step size
     * @return Eigen::Matrix4d Transfer matrix
     */
    virtual Eigen::Matrix4d transfer_matrix(
        const std::optional<Coordinate> &cood0 = std::nullopt, double ds=0.1) {
        (void)cood0; // unused parameter
        (void)ds; // unused parameter
        return Eigen::Matrix4d::Identity();
    }

    /**
     * @brief Get an array of transfer matrices for a given coordinate and step size.
     * @param cood0 Initial coordinate
     * @param ds Maximum step size
     * @param endpoint Whether to include the endpoint
     * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Array of transfer matrices and s array
     */
    virtual std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1, bool endpoint=false) {
        const auto s_ary = s_array(ds, endpoint);
        return {std::vector<Eigen::Matrix4d>(s_ary.size(), Eigen::Matrix4d::Identity()), s_ary};
    }

    /**
     * @brief Get the additive dispersion vector of the element.
     * @param cood0 Initial coordinate
     * @param ds Maximum step size
     * @return Eigen::Vector4d Additive dispersion vector
     */
    virtual Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1) const noexcept {
        (void)cood0; // unused parameter
        (void)ds; // unused parameter
        return Eigen::Vector4d::Zero();
    }

    /**
     * @brief Get an array of additive dispersion vectors for the element.
     * @param cood0 Initial coordinate
     * @param ds Maximum step size
     * @param endpoint Whether to include the endpoint
     * @return std::tuple<std::vector<Eigen::Vector4d>, Eigen::ArrayXd> Array of dispersion vectors and s array
     */
    virtual std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
    dispersion_array(const std::optional<Coordinate> &cood0 = std::nullopt, double ds=0.1,
        bool endpoint=false) const noexcept {
        (void)cood0; // unused parameter
        const auto s_ary = s_array(ds, endpoint);
        const auto disp_ary = Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4, s_ary.size());
        return {disp_ary, s_ary};
    }

    // Transfer a single coordinate through the element
    virtual std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>
    transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0, const double ds=0.1);

    // Transfer coordinate array through the element
    virtual std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>
    transfer_array(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0, const double ds, bool endpoint);

    /**
     * @brief Calculate radiation integrals through the element.
     * @param cood0 Initial coordinate
     * @param evlp0 Initial envelope
     * @param disp0 Initial dispersion
     * @param ds Maximum step size
     * @return std::tuple<double, double, double, double, double, double> Radiation integrals (I2, I4, I5u, I5v, I4u, I4v)
     */
    virtual std::tuple<double, double, double, double, double, double>
    radiation_integrals(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
        const std::optional<Dispersion> &disp0, const double ds=0.1) {
        (void)cood0; // unused parameter
        (void)evlp0; // unused parameter
        (void)disp0; // unused parameter
        (void)ds; // unused parameter
        return {0., 0., 0., 0., 0., 0.};
    }

    // Get element and local s from global s
    virtual std::tuple<const Element&, double> get_element_from_s(double s) const noexcept(false);

    // Get transfer matrix from s to the end of the element
    virtual Eigen::Matrix4d transfer_matrix_from_s(double s, const Coordinate &cood0,
        double ds=0.1) const noexcept(false);
};
