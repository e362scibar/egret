/**
 * @file nonlinearmultipole.cpp
 * @brief Implementation of the NonlinearMultipole element class.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// nonlinearmultipole.cpp
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

#include "egret/nonlinearmultipole.hpp"
#include "egret/quadrupole.hpp"
#include "egret/drift.hpp"
#include <cmath>
#include <ranges>

/**
 * @brief Compute coordinate, transfer matrix, and dispersion using the midpoint integration method.
 * @param cood0 Initial coordinate.
 * @param ds Step size for the integration.
 * @param tmat_flag Whether to compute the transfer matrix.
 * @param disp_flag Whether to compute the dispersion vector.
 * @return std::tuple<Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>> Tuple of the final coordinate, transfer matrix, and dispersion vector.
 */
std::tuple<egret::Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>>
egret::NonlinearMultipole::transfer_by_midpoint_method(const Coordinate &cood0,
    const double ds, const bool tmat_flag, const bool disp_flag) const noexcept(false) {
    // dipole and quadrupole strengths at the entrance.
    // (x'+jy' = - k0 L - k1 L x + j k1 L y)
    const auto [k0_a, k1_a] = get_k(cood0); // tuple of std::complex<double>
    // tilt angle of the quadrupole component at the entrance
    const double tilt_a = 0.5 * std::arg(k1_a);
    // Coordinates after the first half step
    Eigen::Vector4d cood0a_vec = cood0.vector();
    cood0a_vec(0) = 0.0;
    cood0a_vec(2) = 0.0;
    auto cood1a_vec = std::get<0>(Quadrupole::transfer(cood0a_vec, ds,
        std::abs(k1_a), k0_a.real(), k0_a.imag(), tilt_a, false, false)); // Vector4d
    cood1a_vec(0) += cood0.x();
    cood1a_vec(2) += cood0.y();
    const Coordinate cood1a(cood1a_vec, cood0.s() + ds, cood0.z(), cood0.delta());
    // dipole and quadrupole strengths at the exit. (x'+jy' = - k0 L - k1 L x + j k1 L y)
    const auto [k0_b, k1_b] = get_k(cood1a); // tuple of std::complex<double>
    // average dipole and quadrupole strengths
    const std::complex<double> k0_ab = 0.5 * (k0_a + k0_b);
    const std::complex<double> k1_ab = 0.5 * (k1_a + k1_b);
    // tilt angle of the quadrupole component
    const double tilt_ab = 0.5 * std::arg(k1_ab);
    // final coordinates after the second half step
    const auto [cood1_vec, tmat, disp] = Quadrupole::transfer(cood0a_vec, ds,
        std::abs(k1_ab), k0_ab.real(), k0_ab.imag(), tilt_ab, tmat_flag, disp_flag);
    auto cood1_vec_mod = cood1_vec;
    cood1_vec_mod(0) += cood0.x();
    cood1_vec_mod(2) += cood0.y();
    const Coordinate cood1(cood1_vec_mod, cood0.s() + ds, cood0.z(), cood0.delta());
    return std::make_tuple(cood1, tmat, disp);
}

/**
 * @brief Compute coordinate, transfer matrix, and dispersion using the RK4 integration method.
 * @param cood0 Initial coordinate.
 * @param ds Step size for the integration.
 * @param tmat_flag Whether to compute the transfer matrix.
 * @param disp_flag Whether to compute the dispersion vector.
 * @return std::tuple<Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>> Tuple of the final coordinate, transfer matrix, and dispersion vector.
 */
std::tuple<egret::Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>>
egret::NonlinearMultipole::transfer_by_rk4_method(const Coordinate &cood0,
    const double ds, const bool tmat_flag, const bool disp_flag) const noexcept(false) {
    // dipole and quadrupole strengths at the entrance.
    // (x'+jy' = - k0 L - k1 L x + j k1 L y)
    const auto [k0_a, k1_a] = get_k(cood0); // tuple of std::complex<double>
    // tilt angle of the quadrupole component at the entrance
    const double tilt_a = 0.5 * std::arg(k1_a);
    // Coordinates after the first quarter step
    Eigen::Vector4d cood0_vec = cood0.vector();
    cood0_vec(0) = 0.0;
    cood0_vec(2) = 0.0;
    auto cood1_vec = std::get<0>(Quadrupole::transfer(cood0_vec, 0.5 * ds,
        std::abs(k1_a), k0_a.real(), k0_a.imag(), tilt_a, false, false)); // Vector4d
    cood1_vec(0) += cood0.x();
    cood1_vec(2) += cood0.y();
    const Coordinate cood1(cood1_vec, cood0.s() + 0.5 * ds, cood0.z(), cood0.delta());
    // dipole and quadrupole strengths after the first quarter step.
    const auto [k0_b, k1_b] = get_k(cood1); // tuple of std::complex<double>
    // tilt angle of the quadrupole component after the first quarter step
    const double tilt_b = 0.5 * std::arg(k1_b);
    // Coordinates after the second quarter step
    auto cood2_vec = std::get<0>(Quadrupole::transfer(cood0_vec, 0.5 * ds,
        std::abs(k1_b), k0_b.real(), k0_b.imag(), tilt_b, false, false)); // Vector4d
    cood2_vec(0) += cood0.x();
    cood2_vec(2) += cood0.y();
    const Coordinate cood2(cood2_vec, cood0.s() + 0.5 * ds, cood0.z(), cood0.delta());
    // dipole and quadrupole strengths after the second quarter step.
    const auto [k0_c, k1_c] = get_k(cood2); // tuple of std::complex<double>
    // tilt angle of the quadrupole component after the second quarter step
    const double tilt_c = 0.5 * std::arg(k1_c);
    // Coordinates after the third quarter step
    auto cood3_vec = std::get<0>(Quadrupole::transfer(cood0_vec, ds,
        std::abs(k1_c), k0_c.real(), k0_c.imag(), tilt_c, false, false)); // Vector4d
    cood3_vec(0) += cood0.x();
    cood3_vec(2) += cood0.y();
    const Coordinate cood3(cood3_vec, cood0.s() + ds, cood0.z(), cood0.delta());
    // dipole and quadrupole strengths at the exit.
    const auto [k0_d, k1_d] = get_k(cood3); // tuple of std::complex<double>
    // average dipole and quadrupole strengths
    const std::complex<double> k0 = (k0_a + 2.0 * k0_b + 2.0 * k0_c + k0_d) / 6.0;
    const std::complex<double> k1 = (k1_a + 2.0 * k1_b + 2.0 * k1_c + k1_d) / 6.0;
    // tilt angle of the quadrupole component
    const double tilt = 0.5 * std::arg(k1);
    // final coordinates after the fourth quarter step
    const auto [cood4_vec, tmat, disp] = Quadrupole::transfer(cood0_vec, ds,
        std::abs(k1), k0.real(), k0.imag(), tilt, tmat_flag, disp_flag);
    auto cood4_vec_mod = cood4_vec;
    cood4_vec_mod(0) += cood0.x();
    cood4_vec_mod(2) += cood0.y();
    const Coordinate cood4(cood4_vec_mod, cood0.s() + ds, cood0.z(), cood0.delta());
    return std::make_tuple(cood4, tmat, disp);
}

/**
 * @brief Compute coordinate, transfer matrix, and dispersion using the symplectic 1st order integration method.
 * @param cood0 Initial coordinate.
 * @param ds Step size for the integration.
 * @param tmat_flag Whether to compute the transfer matrix.
 * @param disp_flag Whether to compute the dispersion vector.
 * @return std::tuple<Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>> Tuple of the final coordinate, transfer matrix, and dispersion vector.
 */
std::tuple<egret::Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>>
egret::NonlinearMultipole::transfer_by_symplectic1_method(const Coordinate &cood0,
    const double ds, const bool tmat_flag, const bool disp_flag) const noexcept(false) {
    // initial coordinates
    const auto &x0 = cood0.x(); // double &
    const auto &xp0 = cood0.xp(); // double &
    const auto &y0 = cood0.y(); // double &
    const auto &yp0 = cood0.yp(); // double &
    // dipole and quadrupole strengths at the entrance.
    // (x'+jy' = - k0 L - k1 L x + j k1 L y)
    const auto [k0, k1] = get_k(cood0); // tuple of std::complex<double>
    // update momenta
    const auto dxp = -k0.real() * ds;
    const auto dyp = -k0.imag() * ds;
    const auto xp1 = xp0 + dxp;
    const auto yp1 = yp0 + dyp;
    // update positions
    const auto x1 = x0 + xp1 * ds;
    const auto y1 = y0 + yp1 * ds;
    // final coordinates
    const auto cood1_vec = Eigen::Vector4d(x1, xp1, y1, yp1);
    const Coordinate cood1(cood1_vec, cood0.s() + ds, cood0.z(), cood0.delta());
    // transfer matrix and dispersion
    std::optional<Eigen::Matrix4d> tmat = std::nullopt;
    std::optional<Eigen::Vector4d> disp = std::nullopt;
    if (tmat_flag) {
        if (std::abs(k1.imag()) == 0.0) {
            tmat = Quadrupole::transfer_matrix(ds, k1.real());
        } else {
            const auto rmat = Quadrupole::rotation_matrix(std::arg(k1) * 0.5);
            tmat = Quadrupole::transfer_matrix(ds, std::abs(k1), rmat);
        }
    }
    if (disp_flag) {
        disp = Eigen::Vector4d(-dxp * ds, -dxp, -dyp * ds, -dyp);
    }
    return std::make_tuple(cood1, tmat, disp);
}

/**
 * @brief Compute coordinate, transfer matrix, and dispersion using the symplectic 2nd order integration method.
 * @param cood0 Initial coordinate.
 * @param ds Step size for the integration.
 * @param tmat_flag Whether to compute the transfer matrix.
 * @param disp_flag Whether to compute the dispersion vector.
 * @return std::tuple<Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>> Tuple of the final coordinate, transfer matrix, and dispersion vector.
 */
std::tuple<egret::Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>>
egret::NonlinearMultipole::transfer_by_symplectic2_method(const Coordinate &cood0,
    const double ds, const bool tmat_flag, const bool disp_flag) const noexcept(false) {
    // initial coordinates
    const auto &x0 = cood0.x(); // double &
    const auto &xp0 = cood0.xp(); // double &
    const auto &y0 = cood0.y(); // double &
    const auto &yp0 = cood0.yp(); // double &
    // update displacements (first half step)
    const auto ds_half = 0.5 * ds; // double
    const auto x1 = x0 + xp0 * ds_half; // double
    const auto y1 = y0 + yp0 * ds_half; // double
    // dipole and quadrupole strengths at the half step.
    // (x'+jy' = - k0 L - k1 L x + j k1 L y)
    const auto cood1_vec = Eigen::Vector4d(x1, xp0, y1, yp0);
    const Coordinate cood1(cood1_vec, cood0.s() + ds_half, cood0.z(), cood0.delta());
    const auto [k0, k1] = get_k(cood1); // tuple of std::complex<double>
    // update momenta
    const auto dxp = -k0.real() * ds; // double
    const auto dyp = -k0.imag() * ds; // double
    const auto xp1 = xp0 + dxp; // double
    const auto yp1 = yp0 + dyp; // double
    // update displacements (second half step)
    const auto x2 = x1 + xp1 * ds_half; // double
    const auto y2 = y1 + yp1 * ds_half; // double
    // final coordinates
    const auto cood2_vec = Eigen::Vector4d(x2, xp1, y2, yp1);
    const Coordinate cood2(cood2_vec, cood0.s() + ds, cood0.z(), cood0.delta());
    // transfer matrix and dispersion
    std::optional<Eigen::Matrix4d> tmat = std::nullopt;
    std::optional<Eigen::Vector4d> disp = std::nullopt;
    if (tmat_flag) {
        if (std::abs(k1.imag()) == 0.0) {
            tmat = Quadrupole::transfer_matrix(ds, k1.real());
        } else {
            const auto rmat = Quadrupole::rotation_matrix(std::arg(k1) * 0.5);
            tmat = Quadrupole::transfer_matrix(ds, std::abs(k1), rmat);
        }
    }
    if (disp_flag) {
        disp = Eigen::Vector4d(-dxp * ds_half, -dxp, -dyp * ds_half, -dyp);
    }
    return std::make_tuple(cood2, tmat, disp);
}

/**
 * @brief Compute coordinate, transfer matrix, and dispersion using the symplectic 4th order integration method.
 * @param cood0 Initial coordinate.
 * @param ds Step size for the integration.
 * @param tmat_flag Whether to compute the transfer matrix.
 * @param disp_flag Whether to compute the dispersion vector.
 * @return std::tuple<Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>> Tuple of the final coordinate, transfer matrix, and dispersion vector.
 */
std::tuple<egret::Coordinate, std::optional<Eigen::Matrix4d>, std::optional<Eigen::Vector4d>>
egret::NonlinearMultipole::transfer_by_symplectic4_method(const Coordinate &cood0,
    const double ds, const bool tmat_flag, const bool disp_flag) const noexcept(false) {
    static const double a1 = 1.0 / (2.0 - std::cbrt(2.0));
    static const double a0 = 1.0 - 2.0 * a1;
    const auto ds0 = a0 * ds; // double
    const auto ds1 = a1 * ds; // double
    const auto [cood1, tmat1, disp1] = transfer_by_symplectic2_method(
        cood0, ds1, tmat_flag, disp_flag);
    const auto [cood2, tmat2, disp2] = transfer_by_symplectic2_method(
        cood1, ds0, tmat_flag || disp_flag, disp_flag);
    const auto [cood3, tmat3, disp3] = transfer_by_symplectic2_method(
        cood2, ds1, tmat_flag || disp_flag, disp_flag);
    // transfer matrix and dispersion
    std::optional<Eigen::Matrix4d> tmat = std::nullopt;
    std::optional<Eigen::Vector4d> disp = std::nullopt;
    if (tmat_flag) {
        tmat = (*tmat3) * (*tmat2) * (*tmat1);
    }
    if (disp_flag) {
        disp = (*tmat2) * (*disp1) + (*disp2);
        disp = (*tmat3) * (*disp) + (*disp3);
    }
    return std::make_tuple(cood3, tmat, disp);
}

/**
 * @brief Compute the transfer matrix of the nonlinear multipole element.
 * @param cood0 Initial coordinate. (optional)
 * @param ds Step size for the transfer matrix calculation.
 * @param method Integration method to use.
 * @return Eigen::Matrix4d The transfer matrix.
 */
Eigen::Matrix4d egret::NonlinearMultipole::transfer_matrix(
    const std::optional<Coordinate> &cood0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    const size_t n_step = static_cast<size_t>(std::ceil(length_ / ds));
    const double ds_step = length_ / n_step;
    Coordinate cood = cood0 ? *cood0 : Coordinate();
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    auto transfer_by_integration = get_transfer_func_ptr(method);
    for (const size_t i : std::views::iota(0u, n_step)) {
        (void)i; // unused variable
        const auto results = (this->*transfer_by_integration)(cood, ds_step, true, false);
        cood = std::get<0>(results);
        const auto tmat_step = std::get<1>(results);
        tmat = *tmat_step * tmat;
    }
    return tmat;
}

/**
 * @brief Compute the transfer matrix array along the nonlinear multipole element.
 * @param cood0 Initial coordinate. (optional)
 * @param ds Step size for the transfer matrix calculation.
 * @param endpoint Whether to include the endpoint in the s array.
 * @param method Integration method to use.
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Tuple of the array of transfer matrices and the corresponding s positions.
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::NonlinearMultipole::transfer_matrix_array(const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    auto s_array = Element::s_array(ds, endpoint);
    Coordinate cood = cood0 ? *cood0 : Coordinate();
    std::vector<Eigen::Matrix4d> tmat_array;
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    tmat_array.push_back(tmat);
    auto transfer_by_integration = get_transfer_func_ptr(method);
    for (const size_t i : std::views::iota(0u, static_cast<size_t>(s_array.size() - 1))) {
        const double ds_step = s_array[i + 1] - s_array[i];
        const auto results = (this->*transfer_by_integration)(cood, ds_step, true, false);
        cood = std::get<0>(results);
        const auto tmat_step = std::get<1>(results);
        tmat = *tmat_step * tmat;
        tmat_array.push_back(tmat);
    }
    return std::make_tuple(tmat_array, s_array);
}

/**
 * @brief Calculate the additive dispersion vector of the nonlinear multipole element.
 * @param cood0 Initial coordinate. (optional)
 * @param ds Step size for the dispersion calculation.
 * @param method Integration method to use.
 * @return Eigen::Vector4d The additive dispersion vector.
 */
Eigen::Vector4d egret::NonlinearMultipole::dispersion(
    const std::optional<Coordinate> &cood0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    const size_t n_step = static_cast<size_t>(std::ceil(length_ / ds));
    const double ds_step = length_ / n_step;
    Coordinate cood = cood0 ? *cood0 : Coordinate();
    Eigen::Vector4d dispout = Eigen::Vector4d::Zero();
    auto transfer_by_integration = get_transfer_func_ptr(method);
    for (const size_t i : std::views::iota(0u, n_step)) {
        (void)i; // unused variable
        const auto results = (this->*transfer_by_integration)(cood, ds_step, true, true);
        const auto tmat_step = std::get<1>(results);
        const auto disp_step = std::get<2>(results);
        cood = std::get<0>(results);
        dispout = (*tmat_step) * dispout + *disp_step;
    }
    return dispout;
}

/**
 * @brief Calculate the additive dispersion array along the nonlinear multipole element.
 * @param cood0 Initial coordinate. (optional)
 * @param ds Step size for the dispersion calculation.
 * @param endpoint Whether to include the endpoint in the s array.
 * @param method Integration method to use.
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Tuple of the dispersion array and the corresponding s positions.
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::NonlinearMultipole::dispersion_array(
    const std::optional<Coordinate> &cood0, const double ds,
    const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    auto s_array = Element::s_array(ds, endpoint);
    Coordinate cood = cood0 ? *cood0 : Coordinate();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_array(4, s_array.size());
    disp_array.setZero();
    Eigen::Vector4d dispout = Eigen::Vector4d::Zero();
    auto transfer_by_integration = get_transfer_func_ptr(method);
    for (const size_t i : std::views::iota(0u, static_cast<size_t>(s_array.size() - 1))) {
        const double ds_step = s_array[i+1] - s_array[i];
        const auto results = (this->*transfer_by_integration)(cood, ds_step, true, true);
        const auto tmat_step = std::get<1>(results);
        const auto disp_step = std::get<2>(results);
        cood = std::get<0>(results);
        dispout = (*tmat_step) * dispout + *disp_step;
        disp_array.col(i+1) = dispout;
    }
    return std::make_tuple(disp_array, s_array);
}

/**
 * @brief Transfer a particle through the nonlinear multipole element.
 * @param cood0 Initial coordinate.
 * @param evlp0 Initial envelope. (optional)
 * @param disp0 Initial dispersion. (optional)
 * @param ds Step size for the transfer calculation.
 * @param method Integration method to use.
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Tuple of final coordinate, envelope, and dispersion.
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::NonlinearMultipole::transfer(const Coordinate &cood0,
    const std::optional<Envelope> &evlp0, const std::optional<Dispersion> &disp0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    Coordinate cood = cood0;
    cood.x(cood0.x() - dx_);
    cood.y(cood0.y() - dy_);
    cood.s(cood0.s() - ds_);
    const size_t n_step = static_cast<size_t>(std::ceil(length_ / ds));
    const double ds_step = length_ / n_step;
    std::optional<Eigen::Matrix4d> tmat = std::nullopt;
    std::optional<Eigen::Vector4d> dispout = std::nullopt;
    if (evlp0) {
        tmat = Eigen::Matrix4d::Identity();
    }
    if (disp0) {
        dispout = disp0->vector();
    }
    auto transfer_by_integration = get_transfer_func_ptr(method);
    for (const size_t i : std::views::iota(0u, n_step)) {
        (void)i; // unused variable
        const auto results = (this->*transfer_by_integration)(cood, ds_step,
            tmat.has_value() || dispout.has_value(), dispout.has_value());
        cood = std::get<0>(results);
        const auto tmat_step = std::get<1>(results);
        const auto disp_step = std::get<2>(results);
        if (tmat) {
            *tmat = (*tmat_step) * (*tmat);
        }
        if (dispout) {
            *dispout = (*tmat_step) * (*dispout) + *disp_step;
        }
    }
    cood.x(cood.x() + dx_);
    cood.y(cood.y() + dy_);
    cood.s(cood.s() + ds_);
    std::optional<Envelope> evlp1 = std::nullopt;
    if (tmat) {
        evlp1 = evlp0;
        evlp1->transfer(*tmat, length_);
    }
    std::optional<Dispersion> disp1 = std::nullopt;
    if (dispout) {
        disp1 = Dispersion(*dispout, disp0->s() + length_);
    }
    return std::make_tuple(cood, evlp1, disp1);
}

/**
 * @brief Transfer a particle through the nonlinear multipole element and obtain arrays of coordinates, envelopes, and dispersions.
 * @param cood0 Initial coordinate.
 * @param evlp0 Initial envelope. (optional)
 * @param disp0 Initial dispersion. (optional)
 * @param ds Maximum step size for the transfer calculation.
 * @param endpoint Whether to include the endpoint in the output arrays.
 * @param method Integration method to use.
 * @return std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>> Tuple of coordinate array, envelope array, and dispersion array.
 */
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>>
egret::NonlinearMultipole::transfer_array(const Coordinate &cood0,
    const std::optional<Envelope> &evlp0, const std::optional<Dispersion> &disp0,
    const double ds, const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    const auto s_array = Element::s_array(ds, endpoint); // ArrayXd
    Coordinate cood = cood0;
    cood.x(cood0.x() - dx_);
    cood.y(cood0.y() - dy_);
    cood.s(cood0.s() - ds_);
    const size_t n_step = static_cast<size_t>(s_array.size() - 1);
    Eigen::Matrix<double, 4, Eigen::Dynamic> cood_array_out(4, s_array.size());
    Eigen::ArrayXd s_array_out(s_array.size());
    cood_array_out.col(0) = cood0.vector();
    s_array_out(0) = cood0.s();
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    std::optional<std::vector<Eigen::Matrix4d>> tmat_array = std::nullopt;
    std::optional<Eigen::Matrix<double, 4, Eigen::Dynamic>> disp_array = std::nullopt;
    if (evlp0) {
        tmat_array = std::vector<Eigen::Matrix4d>();
        tmat_array->reserve(s_array.size());
        tmat_array->push_back(Eigen::Matrix4d::Identity());
    }
    if (disp0) {
        disp_array = Eigen::Matrix<double, 4, Eigen::Dynamic>(4, s_array.size());
        disp_array->col(0) = disp0->vector();
    }
    auto transfer_by_integration = get_transfer_func_ptr(method);
    for (const size_t i : std::views::iota(0u, n_step)) {
        const double ds_step = s_array[i+1] - s_array[i];
        const auto results = (this->*transfer_by_integration)(cood, ds_step,
            evlp0.has_value() || disp0.has_value(), disp0.has_value());
        cood = std::get<0>(results);
        const auto tmat_step = std::get<1>(results);
        const auto disp_step = std::get<2>(results);
        if (tmat_array) {
            tmat = (*tmat_step) * tmat;
            tmat_array->push_back(tmat);
        }
        if (disp_array) {
            disp_array->col(i+1) = (*tmat_step) * disp_array->col(i) + *disp_step;
        }
        auto cood_vec_out = cood.vector();
        cood_vec_out(0) += dx_;
        cood_vec_out(2) += dy_;
        cood_array_out.col(i+1) = cood_vec_out;
        s_array_out(i+1) = cood.s() + ds_;
    }
    CoordinateArray cood1_array(cood_array_out, s_array_out,
        Eigen::ArrayXd::Constant(s_array.size(), cood0.z()),
        Eigen::ArrayXd::Constant(s_array.size(), cood0.delta()));
    std::optional<EnvelopeArray> evlp1_array = std::nullopt;
    if (tmat_array) {
        evlp1_array = EnvelopeArray::transport(*evlp0, *tmat_array, s_array);
    }
    std::optional<DispersionArray> disp1_array = std::nullopt;
    if (disp_array) {
        disp1_array = DispersionArray(*disp_array, s_array + disp0->s());
    }
    return std::make_tuple(cood1_array, evlp1_array, disp1_array);
}
