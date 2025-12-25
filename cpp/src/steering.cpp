/**
 * @file steering.cpp
 * @brief Steering magnet class implementation
 * @author Hirokazu Maesaka
 * @date 2025
 */
// steering.cpp
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

#include "egret/steering.hpp"
#include <ranges>

/**
 * @brief Compute effective kick angles considering tilt and momentum deviation.
 * @param delta Relative momentum deviation
 * @return std::tuple<double, double> Effective kick angles (kick_x_eff, kick_y_eff)
 */
std::tuple<double, double> egret::Steering::tilted_kick(const double delta) const noexcept {
    const double cos_tilt = std::cos(tilt_);
    const double sin_tilt = std::sin(tilt_);
    const double denom = 1.0 + delta;
    const double kick_x_eff = (kick_x_ * cos_tilt - kick_y_ * sin_tilt) / denom;
    const double kick_y_eff = (kick_x_ * sin_tilt + kick_y_ * cos_tilt) / denom;
    return std::make_tuple(kick_x_eff, kick_y_eff);
}

/**
 * @brief Calculate the additive dispersion function at the end of the steering magnet.
 * @param cood0 Optional input coordinate
 * @param ds Step size (unused)
 * @param method Integration method (unused)
 * @return Eigen::Vector4d Additive dispersion vector [eta_x, eta'_x, eta_y, eta'_y]
 */
Eigen::Vector4d egret::Steering::dispersion(const std::optional<Coordinate> &cood0,
    const double ds, IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const auto [kick_x_eff, kick_y_eff] = tilted_kick(delta);
    const double eta_x = -0.5 * length_ * kick_x_eff;
    const double eta_y = -0.5 * length_ * kick_y_eff;
    const double etap_x = -kick_x_eff;
    const double etap_y = -kick_y_eff;
    Eigen::Vector4d eta;
    eta << eta_x, etap_x, eta_y, etap_y;
    return eta;
}

/**
 * @brief Calculate the additive dispersion function array through the steering magnet.
 * @param cood0 Optional input coordinate
 * @param ds Step size
 * @param endpoint Whether to include the endpoint
 * @param method Integration method (unused)
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Array of additive dispersion vectors and corresponding s values
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Steering::dispersion_array(const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint, IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const auto s_array = Element::s_array(ds, endpoint); // ArrayXd
    const size_t n = s_array.size();
    const double delta = cood0 ? cood0->delta() : 0.0;
    const auto [kick_x_eff, kick_y_eff] = tilted_kick(delta);
    Eigen::Matrix<double, 4, Eigen::Dynamic> eta(4, n);
    if (n == 1) {
        eta << 0.0, -kick_x_eff, 0.0, -kick_y_eff;
    } else if (std::abs(length_) > 0.) {
        const auto eta_x = -0.5 * kick_x_eff * s_array * s_array / length_; // ArrayXd
        const auto eta_y = -0.5 * kick_y_eff * s_array * s_array / length_; // ArrayXd
        const auto etap_x = -kick_x_eff * s_array / length_; // ArrayXd
        const auto etap_y = -kick_y_eff * s_array / length_; // ArrayXd
        eta.row(0) = eta_x;
        eta.row(1) = etap_x;
        eta.row(2) = eta_y;
        eta.row(3) = etap_y;
    } else {
        throw std::runtime_error("Steering length is zero and array length is more than 1 in dispersion_array calculation.");
    }
    return std::make_tuple(eta, s_array);
}

/**
 * @brief Transfer the coordinate, envelope, and dispersion through the steering magnet.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Step size for integration
 * @param method Integration method (unused)
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Coordinate, envelope, and dispersion after transfer
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Steering::transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds, IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0.delta();
    const auto [kick_x_eff, kick_y_eff] = tilted_kick(delta);
    const double dx = 0.5 * length_ * kick_x_eff;
    const double dy = 0.5 * length_ * kick_y_eff;
    const auto M = transfer_matrix(cood0, ds);
    const auto kick_vec = Eigen::Vector4d(dx, kick_x_eff, dy, kick_y_eff); // Vector4d
    const auto cood_vec = M * cood0.vector() + kick_vec;
    const Coordinate cood(cood_vec, cood0.s() + length_, cood0.z(), cood0.delta());
    std::optional<Envelope> evlp = evlp0;
    if (evlp) {
        evlp->transfer(M, length_);
    }
    std::optional<Dispersion> disp = disp0;
    if (disp) {
        const auto disp_add = dispersion(cood0, ds); // Vector4d
        const auto disp_vec = M * disp0->vector() + disp_add; // Vector4d
        disp->vector(disp_vec);
        disp->s(disp0->s() + length_);
    }
    return std::make_tuple(cood, evlp, disp);
}

/**
 * @brief Transfer the coordinate, envelope, and dispersion through the steering magnet for an array of steps.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Step size for integration
 * @param endpoint Whether to include the endpoint in the array
 * @param method Integration method (unused)
 * @return std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>> Coordinate, envelope, and dispersion arrays after transfer
 */
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>>
egret::Steering::transfer_array(const Coordinate &cood0,
    const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0,
    const double ds, const bool endpoint,
    IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0.delta();
    const auto [kick_x_eff, kick_y_eff] = tilted_kick(delta);
    const auto [M_array, s_array] = transfer_matrix_array(cood0, ds, endpoint); // vector<Matrix4d>, ArrayXd
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> kick_vector_array(4, n);
    if (n == 1) {
        kick_vector_array << 0.0, kick_x_eff, 0.0, kick_y_eff;
    } else if (std::abs(length_) > 0.) {
        const auto dx_array = 0.5 * kick_x_eff * s_array; // ArrayXd
        const auto dy_array = 0.5 * kick_y_eff * s_array; // ArrayXd
        const auto kick_x_array = kick_x_eff * s_array / length_; // ArrayXd
        const auto kick_y_array = kick_y_eff * s_array / length_; // ArrayXd
        kick_vector_array.row(0) = dx_array;
        kick_vector_array.row(1) = kick_x_array;
        kick_vector_array.row(2) = dy_array;
        kick_vector_array.row(3) = kick_y_array;
    } else {
        throw std::runtime_error("Steering length is zero and array length is more than 1 in transfer_array calculation.");
    }
    Eigen::Matrix<double, Eigen::Dynamic, 4> M_combined(4 * n, 4);
    for (const size_t i : std::views::iota(0u, n)) {
        M_combined.block<4, 4>(4 * i, 0) = M_array[i];
    }
    const auto vector_array = (M_combined * cood0.vector()).reshaped(4, n)
        + kick_vector_array; // Matrix 4xn
    const CoordinateArray cood_array(vector_array, cood0.s() + s_array,
        Eigen::ArrayXd::Constant(n, cood0.z()),
        Eigen::ArrayXd::Constant(n, cood0.delta()));
    std::optional<EnvelopeArray> evlp_array;
    if (evlp0) {
        evlp_array = EnvelopeArray::transport(*evlp0, M_array, s_array);
    }
    std::optional<DispersionArray> disp_array;
    if (disp0) {
        const auto [disp_add_array, _] = dispersion_array(cood0, ds, endpoint); // Matrix 4xn, ArrayXd
        const auto disp_vector_array = (M_combined * disp0->vector()).reshaped(4, n)
            + disp_add_array; // Matrix 4xn
        disp_array = DispersionArray(disp_vector_array, disp0->s() + s_array);
    }
    return std::make_tuple(cood_array, evlp_array, disp_array);
}
