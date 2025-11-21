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
Eigen::ArrayXd egret::Element::s_array(double length, double ds=0.1, bool endpoint=false) noexcept {
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
 * @brief Transfer a single coordinate through the element.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Transfer results
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Element::transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds) {
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
        return {cood, evlp, disp};
    }
    const Eigen::Matrix4d M = transfer_matrix(cood0err, ds);
    const Eigen::Vector4d v_out = M * cood0err.vector();
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
    return {cood, evlp, disp};
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
    const std::optional<Dispersion> &disp0, const double ds, bool endpoint) {
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
            const Eigen::Matrix<double, 4, 1> vector_array = Eigen::Map<Eigen::Matrix<double, 4, 1>>(cood.vector().data());
            const Eigen::ArrayXd s_array{cood.s()};
            const Eigen::ArrayXd z_array{cood.z()};
            const Eigen::ArrayXd delta_array{cood.delta()};
            if (cood_array) {
                cood_array->append(CoordinateArray(vector_array, s_array, z_array, delta_array));
            } else {
                cood_array = CoordinateArray(vector_array, s_array, z_array, delta_array);
            }
        }
        cood_array->x_array(cood_array->x_array() + dx_);
        cood_array->y_array(cood_array->y_array() + dy_);
        cood_array->s_array(cood_array->s_array() + ds_);
        return {*cood_array, evlp_array, disp_array};
    }
    const auto results = transfer_matrix_array(cood0err, ds, endpoint);
    const std::vector<Eigen::Matrix4d> &M_array = std::get<0>(results);
    const Eigen::ArrayXd &s_array = std::get<1>(results);
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> cood_vector_array(4, n);
    for (const size_t i : std::views::iota(0u, n)) {
        const Eigen::Matrix4d &M = M_array[i];
        cood_vector_array.col(i) = M * cood0err.vector();
    }
    cood_vector_array.row(0).array() += dx_;
    cood_vector_array.row(2).array() += dy_;
    CoordinateArray cood_array(cood_vector_array, s_array + ds_,
        Eigen::ArrayXd::Constant(n, cood0.z()),
        Eigen::ArrayXd::Constant(n, cood0.delta()));
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (evlp0) {
        evlp_array = EnvelopeArray::transport(*evlp0, M_array, s_array);
    }
    if (disp0) {
        const auto results = dispersion_array(cood0err, ds, endpoint);
        const Eigen::Matrix<double, 4, Eigen::Dynamic> &dispersion = std::get<0>(results);
        Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
        for (const size_t i : std::views::iota(0u, n)) {
            const Eigen::Matrix4d &M = M_array[i];
            disp_vector_array.col(i) = M * disp0->vector() + dispersion.col(i);
        }
        disp_array = DispersionArray(disp_vector_array, s_array + disp0->s());
    }
    return {cood_array, evlp_array, disp_array};
}

/**
 * @brief Get the element and local s from a global s position.
 * @param s Global s position
 * @return std::tuple<const Element&, double> Tuple of the element reference and local s.
 */
std::tuple<const egret::Element&, double>
egret::Element::get_element_from_s(double s) const noexcept(false) {
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
Eigen::Matrix4d egret::Element::transfer_matrix_from_s(double s,
    const egret::Coordinate &cood0, double ds) const noexcept(false) {
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
                const Eigen::Vector4d v_out = M_total * cood0_local.vector();
                cood0_local = Coordinate(v_out,
                    cood0_local.s() + elem->length() - (s - s_accum),
                    cood0_local.z(), cood0_local.delta());
            } else if (s < s_accum) {
                const Eigen::Matrix4d M = elem->transfer_matrix(cood0_local, ds);
                M_total = M * M_total;
                const Eigen::Vector4d v_out = M * cood0_local.vector();
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

#if 0
std::pair<Eigen::Tensor<double,3>, std::vector<double>> Drift::transfer_matrix_array_from_length(double length, double ds, bool endpoint) {
    std::vector<double> s;
    if (std::abs(length) > 0.0) {
        int n_base = static_cast<int>(std::floor(length / ds));
        int n = n_base + static_cast<int>(endpoint) + 1;
        s.reserve(n);
        for (int i = 0; i < n; ++i) s.push_back((static_cast<double>(i) * length) / (n - 1));
    } else {
        s.push_back(0.0);
    }

    int N = static_cast<int>(s.size());
    Eigen::Tensor<double,3> tmat(4,4,N);
    tmat.setZero();
    for (int k=0;k<N;++k) {
        // identity
        for (int i=0;i<4;++i) tmat(i,i,k) = 1.0;
        tmat(0,1,k) = s[k];
        tmat(2,3,k) = s[k];
    }
    return {tmat, s};
}
#endif
