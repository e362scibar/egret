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
public:
    //! Integration methods
    enum IntegrationMethod {
        MIDPOINT = 0,
        RK4 = 1,
        SYMPLECTIC1 = 2,
        SYMPLECTIC2 = 3,
        SYMPLECTIC4 = 4
    };

protected:
    //! Length of the element
    double length_;
    //! Bending angle of the element
    double angle_;
    //! Transverse and longitudinal offsets and tilt
    double dx_, dy_, ds_, tilt_;
    //! Additional info string
    std::string info_;
    //! Vector of child elements (for composite elements)
    std::optional<std::vector<std::shared_ptr<Element>>> elements_{std::nullopt};
    //! indices
    std::vector<size_t> indices_;

public:
    /**
     * @brief Construct a new Element object
     * @param name Object name
     * @param length Length of the element
     * @param angle Bending angle of the element
     * @param dx Transverse offset in x
     * @param dy Transverse offset in y
     * @param ds Longitudinal offset
     * @param tilt Tilt angle
     * @param info Additional info string
     */
    Element(const std::string &name, const double length, const double angle=0.0,
            const double dx=0.0, const double dy=0.0, const double ds=0.0,
            const double tilt=0.0, const std::string &info="") :
        Object(name), length_(length), angle_(angle),
        dx_(dx), dy_(dy), ds_(ds), tilt_(tilt), info_(info) {}
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
     * @brief Get the bending angle of the element.
     * @return double Bending angle of the element
     */
    double angle() const { return angle_; }
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
     * @brief Get the vector of child elements.
     * @return const std::optional<std::vector<std::shared_ptr<Element>>>& Vector of child elements
     */
    const std::optional<std::vector<std::shared_ptr<Element>>>& elements() const { return elements_; }

    /**
     * @brief Set the length of the element.
     * @param length Length of the element
     */
    void length(const double length) { length_ = length; }
    /**
     * @brief Set the bending angle of the element.
     * @param angle Bending angle of the element
     */
    void angle(const double angle) { angle_ = angle; }
    /**
     * @brief Set the transverse offset in x.
     * @param dx Transverse offset in x
     */
    void dx(const double dx) { dx_ = dx; }
    /**
     * @brief Set the transverse offset in y.
     * @param dy Transverse offset in y
     */
    void dy(const double dy) { dy_ = dy; }
    /**
     * @brief Set the longitudinal offset.
     * @param ds Longitudinal offset
     */
    void ds(const double ds) { ds_ = ds; }
    /**
     * @brief Set the tilt angle.
     * @param tilt Tilt angle
     */
    void tilt(const double tilt) { tilt_ = tilt; }
    /**
     * @brief Set the additional info string.
     * @param info Additional info string
     */
    void info(const std::string &info) { info_ = info; }

    // Get s array based on length and step size.
    static Eigen::ArrayXd s_array(double length, double ds=0.1, bool endpoint=false) noexcept(false);

    // Get s array for the element based on length and step size.
    Eigen::ArrayXd s_array(const double ds=0.1, const bool endpoint=false) const noexcept(false) {
        return s_array(length_, ds, endpoint);
    }

    // Get the transfer matrix of the element
    virtual Eigen::Matrix4d transfer_matrix(
        const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Get an array of transfer matrices through the element
    virtual std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
    transfer_matrix_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        const double ds=0.1, const bool endpoint=false,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Get the additive dispersion vector of the element.
    virtual Eigen::Vector4d dispersion(const std::optional<Coordinate> &cood0 = std::nullopt,
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Get an array of additive dispersion vectors for the element.
    virtual std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
    dispersion_array(const std::optional<Coordinate> &cood0 = std::nullopt,
        const double ds=0.1, const bool endpoint=false,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Transfer a single coordinate through the element
    virtual std::tuple<Coordinate, std::optional<Envelope>, std::optional<Dispersion>>
    transfer(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Transfer coordinate array through the element
    virtual std::tuple<CoordinateArray, std::optional<EnvelopeArray>, std::optional<DispersionArray>>
    transfer_array(const Coordinate &cood0,
        const std::optional<Envelope> &evlp0 = std::nullopt,
        const std::optional<Dispersion> &disp0 = std::nullopt,
        double ds=0.1, bool endpoint=false,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Calculate radiation integrals through the element
    virtual std::tuple<double, double, double, double, double, double>
    radiation_integrals(const Coordinate &cood0, const Envelope &evlp0,
        const Dispersion &disp0, double ds=0.1,
        IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Simpson's rule integration
    static double simpson_integration(const Eigen::ArrayXd &y_array, double dx) noexcept(false);

    // Get element and local s from global s
    virtual std::tuple<const Element&, double> get_element_from_s(double s) const noexcept(false);

    // Get transfer matrix from s to the end of the element
    virtual Eigen::Matrix4d transfer_matrix_from_s(double s, const Coordinate &cood0,
        double ds=0.1, IntegrationMethod method=IntegrationMethod::SYMPLECTIC4) const noexcept(false);

    // Get element at given indices if elements_ is set
    std::shared_ptr<Element> get_element(const std::vector<size_t> &indices) noexcept(false);

    // Set element at given indices if elements_ is set
    void set_element(const std::vector<size_t> &indices,
        std::shared_ptr<Element> &element) noexcept(false);

    // Get longitudinal position at given indices if elements_ is set
    double get_s(const std::vector<size_t> &indices) const noexcept(false);

    // Find indices of elements with given names if elements_ is set
    std::vector<std::vector<size_t>> find_index(const std::vector<std::string> &names) const noexcept(false);

    // Set indices for all child elements
    void set_indices(const std::vector<size_t> &indices={}) noexcept;

    /**
     * @brief Get the indices of this element.
     * @return const std::vector<size_t>& Indices of this element.
     */
    const std::vector<size_t>& get_indices() const noexcept {
        return indices_;
    }

    /**
     * @brief Clone the Element object.
     * @return std::shared_ptr<Element> Shared pointer to the cloned Element object (of derived type)
     * @throws std::runtime_error if called on the base class.
     */
    virtual std::shared_ptr<Element> clone() const noexcept(false) {
        //return std::make_shared<Element>(*this);
        throw std::runtime_error("Do not call Element::clone() in the base class.");
    }
};
