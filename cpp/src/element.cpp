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
#include "egret/drift_util.hpp"
#include <vector>
#include <cmath>
#include <ranges>

namespace eu = egret::util;

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
 * @param method Integration method
 * @return Eigen::Matrix4d Transfer matrix of the element
 */
Eigen::Matrix4d egret::Element::transfer_matrix(
    const std::optional<Coordinate> &cood0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    if (!elements_) {
        // Default: identity matrix
        //return M;
        throw std::runtime_error("Element::transfer_matrix: Do not call on base Element.");
    }
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    for (const auto &elem : *elements_) {
        M = elem->transfer_matrix(cood, ds, method) * M;
        std::tie(cood, std::ignore, std::ignore) = elem->transfer(cood, std::nullopt, std::nullopt, ds, method);
    }
    return M;
}

/**
 * @brief Get an array of transfer matrices for a given coordinate and step size.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size
 * @param endpoint Whether to include the endpoint
 * @param method Integration method
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Array of transfer matrices and s array
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Element::transfer_matrix_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    if (!elements_) {
        // Default: identity matrix array
        //Eigen::ArrayXd s_array = this->s_array(ds, endpoint);
        //std::vector<Eigen::Matrix4d> M_array(s_array.size(), Eigen::Matrix4d::Identity());
        //return std::make_tuple(M_array, s_array);
        throw std::runtime_error("Element::transfer_matrix_array: Do not call on base Element.");
    }
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    double s = 0.0;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::ArrayXd s_array;
    std::vector<Eigen::Matrix4d> M_array;
    for (const auto &elem: *elements_) {
        const auto [M_sub_array, s_sub_array] = elem->transfer_matrix_array(cood, ds, false, method);
        M_array.insert(M_array.end(), M_sub_array.begin(), M_sub_array.end());
        const size_t prev_size = s_array.size();
        const size_t sub_size = s_sub_array.size();
        s_array.conservativeResize(prev_size + sub_size);
        s_array.tail(sub_size) = s + s_sub_array;
        s += elem->length();
        M = elem->transfer_matrix(cood, ds, method) * M;
        std::tie(cood, std::ignore, std::ignore) = elem->transfer(cood, std::nullopt, std::nullopt, ds, method);
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
 * @param method Integration method
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Element::dispersion(
    const std::optional<Coordinate> &cood0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    if (!elements_) {
        // Default: zero dispersion
        return Eigen::Vector4d::Zero();
        //throw std::runtime_error("Element::dispersion: Do not call on base Element.");
    }
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    std::optional<Dispersion> disp = Dispersion();
    for (const auto &elem : *elements_) {
        std::tie(cood, std::ignore, disp) = elem->transfer(cood, std::nullopt, disp, ds, method);
    }
    disp = eu::drift_dispersion(*disp, -ds_);
    return disp->vector();
}

/**
 * @brief Get an array of additive dispersion vectors for a given coordinate and step size.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size
 * @param endpoint Whether to include the endpoint
 * @param method Integration method
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Array of additive dispersion vectors and s array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Element::dispersion_array(const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    if (!elements_) {
        // Default: zero dispersion array
        const auto s_array = this->s_array(ds, endpoint);
        const auto disp_vec_array = Eigen::Matrix<double, 4, Eigen::Dynamic>::Zero(4, s_array.size());
        return std::make_tuple(disp_vec_array, s_array);
        //throw std::runtime_error("Element::dispersion_array: Do not call on base Element.");
    }
    double s = 0.0;
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    std::optional<Dispersion> disp = Dispersion();
    DispersionArray disp_array(Eigen::MatrixXd(4, 0), Eigen::ArrayXd(0));
    //Eigen::ArrayXd s_array;
    //Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vec_array(4, 0);
    for (const auto &elem : *elements_) {
        const auto [disp_sub_array, s_sub_array] = elem->dispersion_array(cood, ds, false, method);
        const auto [tmat_array, _] = elem->transfer_matrix_array(cood, ds, false, method);
        const size_t sub_size = s_sub_array.size();
        auto disp_upstream_array = Eigen::Matrix<double, 4, Eigen::Dynamic>(4, sub_size);
        for (const size_t i : std::views::iota(0u, sub_size)) {
            disp_upstream_array.col(i) = tmat_array[i] * disp->vector();
        }
        disp_array.append(DispersionArray(disp_sub_array + disp_upstream_array, s_sub_array + s));
        s += elem->length();
        std::tie(cood, std::ignore, disp) = elem->transfer(cood, std::nullopt, disp, ds, method);
    }
    if (endpoint) {
        auto disp_sub_array = Eigen::Matrix<double, 4, Eigen::Dynamic>(4, 1);
        disp_sub_array.col(0) = disp->vector();
        const auto s_sub_array = Eigen::ArrayXd::Constant(1, disp->s());
        disp_array.append(DispersionArray(disp_sub_array, s_sub_array));
    }
    disp_array = eu::drift_dispersion_array(disp_array, -ds_);
    return std::make_tuple(disp_array.vector_array(), disp_array.s_array());
}

/**
 * @brief Transfer a single coordinate through the element.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size
 * @param method Integration method
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Transfer results
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Element::transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    if (elements_) {
        for (const auto &elem : *elements_) {
            std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds, method);
        }
    } else {
        const auto M = transfer_matrix(cood, ds, method); // Matrix4d
        if (evlp) {
            evlp->transfer(M, length_);
        }
        if (disp) {
            const Eigen::Vector4d disp_v_out = M * disp->vector() + dispersion(cood, ds, method);
            disp->vector(disp_v_out);
            disp->s(disp->s() + length_);
        }
        cood.vector(M * cood.vector());
        cood.s(cood.s() + length_);
    }
    cood.x(cood.x() + dx_);
    cood.y(cood.y() + dy_);
    cood = eu::drift_coordinate(cood, -ds_);
    if (evlp) {
        *evlp = eu::drift_envelope(*evlp, -ds_);
    }
    if (disp) {
        *disp = eu::drift_dispersion(*disp, -ds_);
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
* @param method Integration method
* @return std::tuple of CoordinateArray, EnvelopeArray, DispersionArray
*/
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>,
    std::optional<egret::DispersionArray>>
egret::Element::transfer_array(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    std::optional<CoordinateArray> cood_array = std::nullopt;
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (elements_) {
        for (const auto &elem : *elements_) {
            const auto [cood_sub, evlp_sub, disp_sub] = elem->transfer_array(cood, evlp, disp, ds, false, method);
            if (cood_array) {
                cood_array->append(cood_sub);
            } else {
                cood_array = cood_sub;
            }
            if (evlp_array) {
                evlp_array->append(*evlp_sub);
            } else if (evlp_sub) {
                evlp_array = *evlp_sub;
            }
            if (disp_array) {
                disp_array->append(*disp_sub);
            } else if (disp_sub) {
                disp_array = *disp_sub;
            }
            std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds, method);
        }
        if (endpoint || !cood_array) {
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
                const auto T_end = std::vector<Eigen::Matrix2d>{evlp->T()};
                const auto psix_end = Eigen::ArrayXd::Constant(1, evlp->psix());
                const auto psiy_end = Eigen::ArrayXd::Constant(1, evlp->psiy());
                if (evlp_array) {
                    evlp_array->append(EnvelopeArray(evlp_end, s_array, T_end, psix_end, psiy_end));
                } else {
                    evlp_array = EnvelopeArray(evlp_end, s_array, T_end, psix_end, psiy_end);
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
    } else {
        const auto [M_array, s_array] = transfer_matrix_array(cood, ds, endpoint, method); // vector<Matrix4d>, ArrayXd
        const size_t n = s_array.size();
        Eigen::Matrix<double, 4, Eigen::Dynamic> cood_vector_array(4, n);
        for (const size_t i : std::views::iota(0u, n)) {
            cood_vector_array.col(i) = M_array[i] * cood.vector();
        }
        cood_array = CoordinateArray(cood_vector_array, s_array + cood.s(),
                                     Eigen::ArrayXd::Constant(n, cood.z()),
                                     Eigen::ArrayXd::Constant(n, cood.delta()));
        if (evlp) {
            evlp_array = EnvelopeArray::transport(*evlp, M_array, s_array);
        }
        if (disp) {
            const auto [dispersion, _] = dispersion_array(cood, ds, endpoint, method); // Matrix<double, 4, Dynamic>
            Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
            for (const size_t i : std::views::iota(0u, n)) {
                disp_vector_array.col(i) = M_array[i] * disp->vector() + dispersion.col(i);
            }
            disp_array = DispersionArray(disp_vector_array, s_array + disp->s());
        }
    }
    cood_array->x_array(cood_array->x_array() + dx_);
    cood_array->y_array(cood_array->y_array() + dy_);
    *cood_array = eu::drift_coordinate_array(*cood_array, -ds_);
    if (evlp_array) {
        *evlp_array = eu::drift_envelope_array(*evlp_array, -ds_);
    }
    if (disp_array) {
        *disp_array = eu::drift_dispersion_array(*disp_array, -ds_);
    }
    return std::make_tuple(*cood_array, evlp_array, disp_array);
}

/**
 * @brief Calculate radiation integrals through the element.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope
 * @param disp0 Initial dispersion
 * @param ds Maximum step size
 * @param method Integration method
 * @return std::tuple<double, double, double, double, double, double> Radiation integrals (I2, I4, I5u, I5v, I4u, I4v)
 */
std::tuple<double, double, double, double, double, double>
egret::Element::radiation_integrals(const Coordinate &cood0, const Envelope &evlp0,
    const Dispersion &disp0, const double ds, const IntegrationMethod method) const noexcept(false) {
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
        std::tie(i2, i4, i5u, i5v, i4u, i4v) = elem->radiation_integrals(cood, *evlp, *disp, ds, method);
        I2 += i2;
        I4 += i4;
        I5u += i5u;
        I5v += i5v;
        I4u += i4u;
        I4v += i4v;
        std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds, method);
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
 * @param method Integration method
 * @return Eigen::Matrix4d Transfer matrix from position s to the end of the element
 */
Eigen::Matrix4d egret::Element::transfer_matrix_from_s(const double s,
    const egret::Coordinate &cood0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    if (elements_) {
        double s_accum = 0.;
        Eigen::Matrix4d M_total = Eigen::Matrix4d::Identity();
        for (const auto &elem : *elements_) {
            if (s >= s_accum && s < s_accum + elem->length()) {
                M_total = elem->transfer_matrix_from_s(s - s_accum, cood, ds, method);
                std::tie(cood, std::ignore, std::ignore) = elem->transfer_from_s(s - s_accum,
                    cood, std::nullopt, std::nullopt, ds, method);
            } else if (s < s_accum) {
                const auto M = elem->transfer_matrix(cood, ds, method); // Matrix4d
                M_total = M * M_total;
                std::tie(cood, std::ignore, std::ignore) = elem->transfer(cood,
                    std::nullopt, std::nullopt, ds, method);
            }
            s_accum += elem->length();
        }
        return M_total;
    }
    const auto elem = partial_element_from_s(s);
    return elem->transfer_matrix(cood, ds, method);
}

/**
 * @brief Get a partial element starting from a given s position.
 * @param s Longitudinal position within the element
 * @return std::shared_ptr<egret::Element> Partial element
 */
std::shared_ptr<egret::Element> egret::Element::partial_element_from_s(const double s) const noexcept(false) {
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    auto elem = clone();
    elem->length(elem->length() - s);
    return elem;
}

/**
 * @brief Transfer coordinate, envelope, and dispersion from a given s position.
 * @param s Longitudinal position within the element
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size
 * @param method Integration method
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Transfer results
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Element::transfer_from_s(const double s, const Coordinate &cood0,
    const std::optional<Envelope> &evlp0, const std::optional<Dispersion> &disp0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    if (elements_) {
        double s_accum = 0.;
        for (const auto &elem : *elements_) {
            if (s >= s_accum && s < s_accum + elem->length()) {
                std::tie(cood, evlp, disp) = elem->transfer_from_s(s - s_accum, cood, evlp, disp, ds, method);
            } else if (s < s_accum) {
                std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds, method);
            }
            s_accum += elem->length();
        }
    } else {
        const auto elem = partial_element_from_s(s);
        const auto tmat = elem->transfer_matrix(cood, ds, method);
        const auto cood_vec = tmat * cood.vector();
        if (evlp) {
            evlp->transfer(tmat, length_ - s);
        }
        if (disp) {
            const auto disp_vec = tmat * disp->vector() + elem->dispersion(cood, ds, method);
            disp = Dispersion(disp_vec, disp->s() + length_ - s);
        }
        cood = Coordinate(cood_vec, cood.s() + length_ - s, cood.z(), cood.delta());
    }
    cood.x(cood.x() + dx_);
    cood.y(cood.y() + dy_);
    cood = eu::drift_coordinate(cood, -ds_);
    if (evlp) {
        *evlp = eu::drift_envelope(*evlp, -ds_);
    }
    if (disp) {
        *disp = eu::drift_dispersion(*disp, -ds_);
    }
    return std::make_tuple(cood, evlp, disp);
}

/**
 * @brief Transfer coordinate array, envelope array, and dispersion array from a given s position.
 * @param s Longitudinal position within the element
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size
 * @param endpoint Whether to include the endpoint
 * @param method Integration method
 * @return std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>> Transfer results
 */
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>>
egret::Element::transfer_array_from_s(const double s, const Coordinate &cood0,
    const std::optional<Envelope> &evlp0, const std::optional<Dispersion> &disp0,
    const double ds, const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    auto cood_array = CoordinateArray(Eigen::Matrix<double, 4, Eigen::Dynamic>(), Eigen::ArrayXd(),
                                        Eigen::ArrayXd(), Eigen::ArrayXd());
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (elements_) {
        double s0 = 0.;
        for (const auto &elem : *elements_) {
            if (s >= s0 && s < s0 + elem->length()) {
                const auto [cood_sub, evlp_sub, disp_sub] = elem->transfer_array_from_s(s - s0, cood, evlp, disp, ds, false, method);
                cood_array.append(cood_sub);
                if (evlp_sub) {
                    if (evlp_array) {
                        evlp_array->append(*evlp_sub);
                    } else {
                        evlp_array = *evlp_sub;
                    }
                }
                if (disp_sub) {
                    if (disp_array) {
                        disp_array->append(*disp_sub);
                    } else {
                        disp_array = *disp_sub;
                    }
                }
                std::tie(cood, evlp, disp) = elem->transfer_from_s(s - s0, cood, evlp, disp, ds, method);
            } else if (s < s0) {
                const auto [cood_sub, evlp_sub, disp_sub] = elem->transfer_array(cood, evlp, disp, ds, false, method);
                cood_array.append(cood_sub);
                if (evlp_sub) {
                    if (evlp_array) {
                        evlp_array->append(*evlp_sub);
                    } else {
                        evlp_array = *evlp_sub;
                    }
                }
                if (disp_sub) {
                    if (disp_array) {
                        disp_array->append(*disp_sub);
                    } else {
                        disp_array = *disp_sub;
                    }
                }
                std::tie(cood, evlp, disp) = elem->transfer(cood, evlp, disp, ds, method);
            }
            s0 += elem->length();
        }
        if (endpoint || cood_array.s_array().size() == 0) {
            const Eigen::ArrayXd s_array = Eigen::ArrayXd::Constant(1, cood.s());
            const Eigen::ArrayXd z_array = Eigen::ArrayXd::Constant(1, cood.z());
            const Eigen::ArrayXd delta_array = Eigen::ArrayXd::Constant(1, cood.delta());
            cood_array.append(CoordinateArray(cood.vector(), s_array, z_array, delta_array));
            if (evlp) {
                const auto cov_end = std::vector<Eigen::Matrix4d>{evlp->cov()};
                const std::vector<Eigen::Matrix2d> T_end{evlp->T()};
                const Eigen::ArrayXd psix_end = Eigen::ArrayXd::Constant(1, evlp->psix());
                const Eigen::ArrayXd psiy_end = Eigen::ArrayXd::Constant(1, evlp->psiy());
                const Eigen::ArrayXd evlp_s_array = Eigen::ArrayXd::Constant(1, evlp->s());
                if (evlp_array) {
                    evlp_array->append(EnvelopeArray(cov_end, evlp_s_array, T_end, psix_end, psiy_end));
                } else {
                    evlp_array = EnvelopeArray(cov_end, evlp_s_array, T_end, psix_end, psiy_end);
                }
            }
            if (disp) {
                const auto disp_end = Eigen::Matrix<double, 4, Eigen::Dynamic>(disp->vector());
                const auto disp_s_array = Eigen::ArrayXd::Constant(1, disp->s());
                if (disp_array) {
                    disp_array->append(DispersionArray(disp_end, disp_s_array));
                } else {
                    disp_array = DispersionArray(disp_end, disp_s_array);
                }
            }
        }
    } else {
        const auto elem = partial_element_from_s(s);
        const auto [M_array, s_array] = elem->transfer_matrix_array(cood, ds, endpoint, method);
        const size_t n = s_array.size();
        Eigen::Matrix<double, 4, Eigen::Dynamic> cood_vector_array(4, n);
        for (const size_t i : std::views::iota(0u, n)) {
            cood_vector_array.col(i) = M_array[i] * cood.vector();
        }
        cood_array = CoordinateArray(cood_vector_array, s_array + cood.s(),
                                        Eigen::ArrayXd::Constant(n, cood.z()),
                                        Eigen::ArrayXd::Constant(n, cood.delta()));
        if (evlp) {
            evlp_array = EnvelopeArray::transport(*evlp, M_array, s_array);
        }
        if (disp) {
            const auto [disp_add, _] = elem->dispersion_array(cood, ds, endpoint, method);
            Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
            for (const size_t i : std::views::iota(0u, n)) {
                disp_vector_array.col(i) = M_array[i] * disp->vector() + disp_add.col(i);
            }
            disp_array = DispersionArray(disp_vector_array, s_array + disp->s());
        }
    }
    cood_array.x_array(cood_array.x_array() + dx_);
    cood_array.y_array(cood_array.y_array() + dy_);
    cood_array = eu::drift_coordinate_array(cood_array, -ds_);
    if (evlp_array) {
        *evlp_array = eu::drift_envelope_array(*evlp_array, -ds_);
    }
    if (disp_array) {
        *disp_array = eu::drift_dispersion_array(*disp_array, -ds_);
    }
    return std::make_tuple(cood_array, evlp_array, disp_array);
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
    if (indices.size() >= new_element->get_indices().size()) {
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
