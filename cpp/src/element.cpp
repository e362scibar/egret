/**
 * @file element.cpp
 * @brief Implementation of the Element class methods.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// element.cpp
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

#include "egret/element.hpp"
#include <vector>
#include <cmath>
#include <ranges>

/**
 * @brief Generate an array of s values based on length and step size.
 * @param length Length of the element
 * @param ds Maximum step size
 * @param endpoint Whether to include the endpoint
 * @return Eigen::ArrayXd Array of s values
 */
Eigen::ArrayXd egret::Element::s_array(const double length, const double ds,
    const bool endpoint) noexcept(false) {
    const double abs_len = std::abs(length);
    const double abs_ds = std::abs(ds);
    if ((abs_len > abs_ds) || (endpoint && (abs_len > 0.0))) {
        const size_t n = static_cast<size_t>(std::ceil(abs_len / abs_ds));
        if (endpoint) {
            return Eigen::ArrayXd::LinSpaced(n + 1, 0.0, length);
        } else {
            const double ds2 = length / static_cast<double>(n);
            return Eigen::ArrayXd::LinSpaced(n, 0.0, length - ds2);
        }
    }
    // length is zero or smaller than ds and endpoint is false
    return Eigen::ArrayXd::Zero(1);
}

/**
 * @brief Get the transfer matrix for a given coordinate and step size.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration
 * @return Eigen::Matrix4d Transfer matrix of the element
 */
Eigen::Matrix4d egret::Element::transfer_matrix(
    const std::optional<Coordinate> &cood0,
    const double ds) const noexcept(false) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    if (!elements_) {
        // Default: identity matrix
        //return M;
        throw std::runtime_error("Element::transfer_matrix: Do not call on base Element.");
    }
    Coordinate cood = cood0.value_or(Coordinate());
    for (const auto &elem : *elements_) {
        M = elem->transfer_matrix(cood, ds) * M;
        cood = std::get<0>(elem->transfer(cood, std::nullopt, std::nullopt, ds));
    }
    return M;
}

/**
 * @brief Get an array of transfer matrices for a given coordinate and step size.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size
 * @param endpoint Whether to include the endpoint
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Array of transfer matrices and s array
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Element::transfer_matrix_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint) const noexcept(false) {
    if (!elements_) {
        // Default: identity matrix array
        //Eigen::ArrayXd s_array = this->s_array(ds, endpoint);
        //std::vector<Eigen::Matrix4d> M_array(s_array.size(), Eigen::Matrix4d::Identity());
        //return std::make_tuple(M_array, s_array);
        throw std::runtime_error("Element::transfer_matrix_array: Do not call on base Element.");
    }
    Coordinate cood = cood0.value_or(Coordinate());
    double s = 0.0;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::ArrayXd s_array;
    std::vector<Eigen::Matrix4d> M_array;
    for (const auto &elem: *elements_) {
        const auto [M_sub_array, s_sub_array] = elem->transfer_matrix_array(cood, ds, false);
        M_array.insert(M_array.end(), M_sub_array.begin(), M_sub_array.end());
        const size_t prev_size = s_array.size();
        const size_t sub_size = s_sub_array.size();
        s_array.conservativeResize(prev_size + sub_size);
        s_array.tail(sub_size) = s + s_sub_array;
        s += elem->length();
        M = elem->transfer_matrix(cood, ds) * M;
        cood = std::get<0>(elem->transfer(cood, std::nullopt, std::nullopt, ds));
    }
    if (endpoint) {
        const size_t s_size = s_array.size();
        s_array.conservativeResize(s_size + 1);
        s_array(s_size - 1) = s;
        M_array.push_back(M);
    }
    return std::make_tuple(M_array, s_array);
}

/**
 * @brief Get the additive dispersion vector of the element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Element::dispersion(
    const std::optional<Coordinate> &cood0,
    const double ds) const noexcept(false) {
    if (!elements_) {
        // Default: zero dispersion
        return Eigen::Vector4d::Zero();
        //throw std::runtime_error("Element::dispersion: Do not call on base Element.");
    }
    Coordinate cood = cood0.value_or(Coordinate());
    Dispersion disp;
    for (const auto &elem : *elements_) {
        const auto results = elem->transfer(cood, std::nullopt, disp, ds);
        cood = std::get<0>(results);
        disp = *std::get<2>(results);
    }
    return disp.vector();
}

/**
 * @brief Get an array of additive dispersion vectors for a given coordinate and step size.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size
 * @param endpoint Whether to include the endpoint
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Array of additive dispersion vectors and s array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Element::dispersion_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint) const noexcept(false) {
    if (!elements_) {
        // Default: zero dispersion array
        const auto s_array = this->s_array(ds, endpoint);
        const auto disp_vec_array = Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4, s_array.size());
        return std::make_tuple(disp_vec_array, s_array);
        //throw std::runtime_error("Element::dispersion_array: Do not call on base Element.");
    }
    double s = 0.0;
    Coordinate cood = cood0.value_or(Coordinate());
    Dispersion disp;
    Eigen::ArrayXd s_array;
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vec_array(4, 0);
    for (const auto &elem : *elements_) {
        const auto [disp_sub_array, s_sub_array] = elem->dispersion_array(cood, ds, false);
        const size_t prev_size = s_array.size();
        const size_t sub_size = s_sub_array.size();
        disp_vec_array.conservativeResize(4, prev_size + sub_size);
        disp_vec_array.rightCols(sub_size) = disp_sub_array;
        const size_t s_prev_size = s_array.size();
        s_array.conservativeResize(s_prev_size + sub_size);
        s_array.tail(sub_size) = s + s_sub_array;
        s += elem->length();
        const auto results = elem->transfer(cood, std::nullopt, disp, ds);
        cood = std::get<0>(results);
        disp = *std::get<2>(results);
    }
    if (endpoint) {
        const size_t s_size = s_array.size();
        s_array.conservativeResize(s_size + 1);
        s_array(s_size - 1) = s;
        disp_vec_array.conservativeResize(4, s_size + 1);
        disp_vec_array.col(s_size - 1) = disp.vector();
    }
    return std::make_tuple(disp_vec_array, s_array);
}

/**
 * @brief Transfer a single coordinate through the element.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Transfer results
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Element::transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds) const noexcept(false) {
    Coordinate cood0err = cood0;
    cood0err.x(cood0.x() - dx_);
    cood0err.y(cood0.y() - dy_);
    cood0err.s(cood0.s() - ds_);
    if (elements_) {
        Coordinate cood = cood0err;
        std::optional<Envelope> evlp = evlp0;
        std::optional<Dispersion> disp = disp0;
        for (const auto &elem : *elements_) {
            std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds);
        }
        cood.x(cood.x() + dx_);
        cood.y(cood.y() + dy_);
        cood.s(cood.s() + ds_);
        return std::make_tuple(cood, evlp, disp);
    }
    const auto M = transfer_matrix(cood0err, ds); // Matrix4d
    const auto v_out = M * cood0err.vector(); // Vector4d
    Coordinate cood(v_out, cood0.s() + length_, cood0.z(), cood0.delta());
    cood.x(cood.x() + dx_);
    cood.y(cood.y() + dy_);
    cood.s(cood.s() + ds_);
    std::optional<Envelope> evlp = evlp0;
    if (evlp) {
        evlp->transfer(M, length_);
    }
    std::optional<Dispersion> disp = disp0;
    if (disp) {
        const Eigen::Vector4d disp_v_out = M * disp->vector() + dispersion(cood0err, ds);
        disp->vector(disp_v_out);
        disp->s(disp->s() + length_);
    }
    return std::make_tuple(cood, evlp, disp);
}

/**
* @brief Transfer an array of coordinates through the element.
* @param cood0 Initial coordinate
* @param evlp0 Initial envelope
* @param disp0 Initial dispersion
* @param ds Maximum step size
* @param endpoint Whether to include the endpoint
* @return std::tuple of CoordinateArray, EnvelopeArray, DispersionArray
*/
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>,
    std::optional<egret::DispersionArray>>
egret::Element::transfer_array(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds, const bool endpoint) const noexcept(false) {
    Coordinate cood0err = cood0;
    cood0err.x(cood0.x() - dx_);
    cood0err.y(cood0.y() - dy_);
    cood0err.s(cood0.s() - ds_);
    if (elements_) {
        Coordinate cood = cood0err;
        std::optional<Envelope> evlp = evlp0;
        std::optional<Dispersion> disp = disp0;
        std::optional<CoordinateArray> cood_array = std::nullopt;
        std::optional<EnvelopeArray> evlp_array = std::nullopt;
        std::optional<DispersionArray> disp_array = std::nullopt;
        for (const auto &elem : *elements_) {
            const auto results = elem->transfer_array(cood, evlp, disp, ds, false);
            if (cood_array) {
                cood_array->append(std::get<0>(results));
            } else {
                cood_array = std::get<0>(results);
            }
            if (evlp_array) {
                evlp_array->append(*std::get<1>(results));
            } else if (std::get<1>(results)) {
                evlp_array = *std::get<1>(results);
            }
            if (disp_array) {
                disp_array->append(*std::get<2>(results));
            } else if (std::get<2>(results)) {
                disp_array = *std::get<2>(results);
            }
            std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds);
        }
        if (endpoint) {
            const Eigen::ArrayXd s_array = Eigen::ArrayXd::Constant(1, cood.s());
            const Eigen::ArrayXd z_array = Eigen::ArrayXd::Constant(1, cood.z());
            const Eigen::ArrayXd delta_array = Eigen::ArrayXd::Constant(1, cood.delta());
            if (cood_array) {
                cood_array->append(CoordinateArray(cood.vector(), s_array, z_array, delta_array));
            } else {
                cood_array = CoordinateArray(cood.vector(), s_array, z_array, delta_array);
            }
            if (evlp) {
                const auto evlp_end = std::vector<Eigen::Matrix4d>{evlp->cov()};
                if (evlp_array) {
                    evlp_array->append(EnvelopeArray(evlp_end, s_array));
                } else {
                    evlp_array = EnvelopeArray(evlp_end, s_array);
                }
            }
            if (disp) {
                const auto disp_end = Eigen::Matrix<double, 4, Eigen::Dynamic>(disp->vector());
                if (disp_array) {
                    disp_array->append(DispersionArray(disp_end, s_array));
                } else {
                    disp_array = DispersionArray(disp_end, s_array);
                }
            }
        }
        cood_array->x_array(cood_array->x_array() + dx_);
        cood_array->y_array(cood_array->y_array() + dy_);
        cood_array->s_array(cood_array->s_array() + ds_);
        return std::make_tuple(*cood_array, evlp_array, disp_array);
    }
    const auto results = transfer_matrix_array(cood0err, ds, endpoint);
    const auto &M_array = std::get<0>(results); // vector<Matrix4d>
    const auto &s_array = std::get<1>(results); // ArrayXd
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> cood_vector_array(4, n);
    for (const size_t i : std::views::iota(0u, n)) {
        const auto &M = M_array[i]; // Matrix4d
        cood_vector_array.col(i) = M * cood0err.vector();
    }
    cood_vector_array.row(0).array() += dx_;
    cood_vector_array.row(2).array() += dy_;
    CoordinateArray cood_array(cood_vector_array, s_array + cood0.s(),
        Eigen::ArrayXd::Constant(n, cood0.z()),
        Eigen::ArrayXd::Constant(n, cood0.delta()));
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (evlp0) {
        evlp_array = EnvelopeArray::transport(*evlp0, M_array, s_array);
    }
    if (disp0) {
        const auto results = dispersion_array(cood0err, ds, endpoint);
        const auto &dispersion = std::get<0>(results); // Matrix<double, 4, Dynamic>
        Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
        for (const size_t i : std::views::iota(0u, n)) {
            const auto &M = M_array[i]; // Matrix4d
            disp_vector_array.col(i) = M * disp0->vector() + dispersion.col(i);
        }
        disp_array = DispersionArray(disp_vector_array, s_array + disp0->s());
    }
    return std::make_tuple(cood_array, evlp_array, disp_array);
}

/**
 * @brief Calculate radiation integrals through the element.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope
 * @param disp0 Initial dispersion
 * @param ds Maximum step size
 * @return std::tuple<double, double, double, double, double, double> Radiation integrals (I2, I4, I5u, I5v, I4u, I4v)
 */
std::tuple<double, double, double, double, double, double>
egret::Element::radiation_integrals(const Coordinate &cood0, const Envelope &evlp0,
    const Dispersion &disp0, const double ds) const noexcept(false) {
    if (!elements_) {
        // Default: zero radiation integrals
        //return std::make_tuple(0., 0., 0., 0., 0., 0.);
        throw std::runtime_error("Element::radiation_integrals: Do not call on base Element.");
    }
    Coordinate cood = cood0;
    std::optional<Envelope> evlp = evlp0;
    std::optional<Dispersion> disp = disp0;
    double I2 = 0.;
    double I4 = 0.;
    double I5u = 0.;
    double I5v = 0.;
    double I4u = 0.;
    double I4v = 0.;
    for (const auto &elem : *elements_) {
        double i2, i4, i5u, i5v, i4u, i4v;
        if (elem->length() == 0.0) {
            continue;
        }
        std::tie(i2, i4, i5u, i5v, i4u, i4v) = elem->radiation_integrals(cood, *evlp, *disp, ds);
        I2 += i2;
        I4 += i4;
        I5u += i5u;
        I5v += i5v;
        I4u += i4u;
        I4v += i4v;
        std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds);
    }
    return std::make_tuple(I2, I4, I5u, I5v, I4u, I4v);
}

/**
 * @brief Get the element and local s from a global s position.
 * @param s Global s position
 * @return std::tuple<const Element&, double> Tuple of the element reference and local s.
 */
std::tuple<const egret::Element&, double>
egret::Element::get_element_from_s(const double s) const noexcept(false) {
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    if (elements_) {
        double s_accum = 0.;
        for (const auto &elem : *elements_) {
            if (s >= s_accum && s < s_accum + elem->length()) {
                return elem->get_element_from_s(s - s_accum);
            }
            s_accum += elem->length();
        }
        // Should not reach here
        throw std::runtime_error("Internal error in get_element_from_s.");
    } else {
        return {(*this), s};
    }
}

/**
 * @brief Get the transfer matrix from a given s position to the end of the element.
 * @param s Longitudinal position within the element
 * @param cood0 Initial coordinate at position s
 * @param ds Maximum step size
 * @return Eigen::Matrix4d Transfer matrix from position s to the end of the element
 */
Eigen::Matrix4d egret::Element::transfer_matrix_from_s(const double s,
    const egret::Coordinate &cood0, const double ds) const noexcept(false) {
    (void)cood0; // suppress unused parameter warning
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    if (elements_) {
        double s_accum = 0.;
        Coordinate cood0_local = cood0;
        Eigen::Matrix4d M_total = Eigen::Matrix4d::Identity();
        for (const auto &elem : *elements_) {
            if (s >= s_accum && s < s_accum + elem->length()) {
                M_total = elem->transfer_matrix_from_s(s - s_accum, cood0_local, ds);
                const auto v_out = M_total * cood0_local.vector(); // Vector4d
                cood0_local = Coordinate(v_out,
                    cood0_local.s() + elem->length() - (s - s_accum),
                    cood0_local.z(), cood0_local.delta());
            } else if (s < s_accum) {
                const auto M = elem->transfer_matrix(cood0_local, ds); // Matrix4d
                M_total = M * M_total;
                std::tie(cood0_local, std::ignore, std::ignore) = elem->transfer(cood0_local,
                    std::nullopt, std::nullopt, ds);
            }
            s_accum += elem->length();
        }
        return M_total;
    } else {
        Element elem = *this;
        elem.length(elem.length() - s);
        return elem.transfer_matrix(cood0, ds);
    }
}

/**
 * @brief Perform numerical integration using Simpson's rule.
 * @param y_array Array of function values at equally spaced points
 * @param dx Spacing between points
 * @return double Approximate integral value
 * @throws std::runtime_error if the number of points is less than 2
 */
double egret::Element::simpson_integration(const Eigen::ArrayXd &y_array, const double dx) noexcept(false) {
    const size_t n = y_array.size();
    if (n < 2) throw std::runtime_error("Need at least 2 points.");
    // n == 2 --> trapezoidal rule
    if (n == 2) {
        return dx * 0.5 * (y_array[0] + y_array[1]);
    }
    // n is odd --> 1/3 rule
    if ((n - 1) % 2 == 0) {
        // I = dx/3 * (y0 + yn + 4*(odd index) + 2*(even index, except both ends))
        const auto odd = y_array(Eigen::seq(1, n-2, 2));
        const auto even = y_array(Eigen::seq(2, n-3, 2));
        return dx/3.0 * (y_array[0] + y_array[n-1] + 4.0 * odd.sum() + 2.0 * even.sum());
    }
    // n is even --> 1/3 rule up to n-3, and 3/8 rule for last 4 points
    int m = n - 4; // 1/3 rule end index
    double I = 0.;
    // 1/3 rule part
    if (m > 0) {
        const auto odd = y_array(Eigen::seq(1, m-1, 2));
        const auto even = y_array(Eigen::seq(2, m-2, 2));
        I += dx/3.0 * (y_array[0] + y_array[m] + 4.0 * odd.sum() + 2.0 * even.sum());
    }
    // 3/8 rule part
    // I = 3*dx/8 * (y0 + 3*y1 + 3*y2 + y3)
    I += 3.0*dx/8.0 * (y_array[m] + 3.0 * y_array[m+1] + 3.0 * y_array[m+2] + y_array[m+3]);
    return I;
}

/**
 * @brief Get element at given indices
 * @param indices Indices of the element
 * @return std::shared_ptr<egret::Element> Element at the given indices
 * @throws std::runtime_error if this element does not have child elements
 * @throws std::invalid_argument if indices are invalid
 * @throws std::out_of_range if indices are out of range
 */
std::shared_ptr<egret::Element> egret::Element::get_element(const std::vector<size_t> &indices) noexcept(false) {
    if (!elements_) {
        throw std::runtime_error("This element does not have child elements.");
    }
    if (indices.empty()) {
        throw std::invalid_argument("Indices vector is empty in Element::get_element.");
    }
    const size_t index = indices[0];
    const auto &elem = elements_->at(index);
    if (indices.size() == 1) {
        return elem;
    }
    return elem->get_element(std::vector<size_t>(indices.begin() + 1, indices.end()));
}

/**
 * @brief Set element at given indices if elements_ is set
 * @param indices Indices of the element 
 * @param new_element New element to set
 * @throws std::runtime_error if this element does not have child elements
 * @throws std::invalid_argument if indices are invalid
 * @throws std::out_of_range if indices are out of range
 */
void egret::Element::set_element(const std::vector<size_t> &indices,
    std::shared_ptr<egret::Element> &new_element) noexcept(false) {
    if (!elements_) {
        throw std::runtime_error("This element does not have child elements.");
    }
    if (indices.empty()) {
        throw std::invalid_argument("Indices vector is empty in Element::set_element.");
    }
    if (indices.size() >= new_element.get_indices().size()) {
        new_element->set_indices(indices);
    }
    const size_t index = indices[0];
    if (indices.size() == 1) {
        elements_->at(index) = new_element;
        return;
    }
    auto &elem = elements_->at(index);
    elem->set_element(std::vector<size_t>(indices.begin() + 1, indices.end()), new_element);
}

/**
 * @brief Get the longitudinal position at given indices
 * @param indices Indices of the element
 * @return double Longitudinal position at the given indices
 * @throws std::runtime_error if this element does not have child elements
 * @throws std::invalid_argument if indices are invalid
 * @throws std::out_of_range if indices are out of range
 */
double egret::Element::get_s(const std::vector<size_t> &indices) const noexcept(false) {
    if (!elements_) {
        throw std::runtime_error("This element does not have child elements.");
    }
    if (indices.empty()) {
        throw std::invalid_argument("Indices vector is empty in Element::get_s.");
    }
    const size_t index = indices[0];
    double s = 0.0;
    for (const size_t i : std::views::iota(0u, index)) {
        s += elements_->at(i)->length();
    }
    if (indices.size() > 1) {
        s += elements_->at(index)->get_s(
            std::vector<size_t>(indices.begin() + 1, indices.end()));
    }
    return s;
}

/**
 * @brief Find indices of every element whose name starts with one of the given names.
 * @param names Vector of names to search for
 * @return std::vector<std::vector<size_t>> Vector of indices of matching elements
 * @throws std::runtime_error if this element does not have child elements
 */
std::vector<std::vector<size_t>> egret::Element::find_index(
    const std::vector<std::string> &names) const noexcept(false) {
    if (!elements_) {
        throw std::runtime_error("This element does not have child elements.");
    }
    std::vector<std::vector<size_t>> indices;
    for (const size_t i : std::views::iota(0u, elements_->size())) {
        const auto &elem = elements_->at(i);
        if (elem->elements_) {
            std::vector<std::vector<size_t>> sub_indices = elem->find_index(names);
            for (auto &sub_index : sub_indices) {
                sub_index.insert(sub_index.begin(), i);
                indices.push_back(sub_index);
            }
        } else {
            for (const auto &name : names) {
                if (elem->name().starts_with(name)) {
                    indices.push_back({i});
                }
            }
        }
    }
    return indices;
}

/**
 * @brief Set the indices of this element and its child elements recursively.
 * @param indices Indices of the element
 */
void egret::Element::set_indices(const std::vector<size_t> &indices) noexcept {
    indices_ = indices;
    if (!elements_) {
        return;
    }
    for (size_t i : std::views::iota(0u, elements_->size())) {
        const auto &elem = elements_->at(i);
        std::vector<size_t> new_indices = indices;
        new_indices.push_back(i);
        elem->set_indices(new_indices);
    }
}
