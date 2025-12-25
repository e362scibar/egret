/**
 * @file quadrupole.cpp
 * @brief Quadrupole magnet class implementation
 * @author Hirokazu Maesaka
 * @date 2025
 */
// quadrupole.cpp
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

#include "egret/quadrupole.hpp"
#include "egret/drift.hpp"
#include <ranges>

/**
 * @brief Calculate the transfer matrix for a quadrupole magnet.
 * @param length Quadrupole length
 * @param k1 Quadrupole strength k1 (1/m^2)
 * @param tilt Rotation angle (radians)
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::transfer_matrix(const double length, const double k1,
    const std::optional<Eigen::Matrix4d> &rmat) noexcept(false) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    if (k1 == 0.0) { // drift case
        M(0,1) = length;
        M(2,3) = length;
        return M;
    }
    const double sqrtk = std::sqrt(std::abs(k1));
    const double psi = sqrtk * length;
    const double cospsi = std::cos(psi);
    const double sinpsi = std::sin(psi);
    const double coshpsi = std::cosh(psi);
    const double sinhpsi = std::sinh(psi);
    Eigen::Matrix2d mf, md; // focusing and defocusing matrices
    mf << cospsi, sinpsi / sqrtk, -sqrtk * sinpsi, cospsi;
    md << coshpsi, sinhpsi / sqrtk, sqrtk * sinhpsi, coshpsi;
    if (k1 < 0.0) { // defocusing in x
        M.block<2,2>(0,0) = md;
        M.block<2,2>(2,2) = mf;
    } else { // focusing in x
        M.block<2,2>(0,0) = mf;
        M.block<2,2>(2,2) = md;
    }
    if (rmat) {
        M = rmat->transpose() * M * (*rmat);
    }
    return M;
}

/**
 * @brief Calculate the rotation matrix of a transfer matrix for a given tilt angle.
 * @param tilt Rotation angle (radians)
 * @return Eigen::Matrix4d Rotation matrix
 */
Eigen::Matrix4d egret::Quadrupole::rotation_matrix(const double tilt) noexcept(false) {
    Eigen::Matrix4d rmat = Eigen::Matrix4d::Zero();
    const double ct = std::cos(tilt);
    const double st = std::sin(tilt);
    rmat(0,0) = ct;
    rmat(0,2) = st;
    rmat(1,1) = ct;
    rmat(1,3) = st;
    rmat(2,0) = -st;
    rmat(2,2) = ct;
    rmat(3,1) = -st;
    rmat(3,3) = ct;
    return rmat;
}

/**
 * @brief Calculate the transfer matrix for the quadrupole element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Step size (unused)
 * @param method Integration method (not used here)
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::transfer_matrix(
    const std::optional<Coordinate> &cood0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    double k = k1_ / (1.0 + delta);
    std::optional<Eigen::Matrix4d> rmat = std::nullopt;
    if (tilt_ != 0.0) {
        rmat = rotation_matrix(tilt_);
    }
    return transfer_matrix(length_, k, rmat);
}

/**
 * @brief Calculate the transfer matrix array through the quadrupole element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Step size (unused)
 * @param endpoint Include endpoint in s_array
 * @param method Integration method (not used here)
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Tuple of transfer matrix array and s_array
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Quadrupole::transfer_matrix_array(
    const std::optional<Coordinate> &cood0, const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double k = k1_ / (1.0 + delta);
    const auto s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    std::optional<Eigen::Matrix4d> rmat = std::nullopt;
    if (tilt_ != 0.0) {
        rmat = rotation_matrix(tilt_);
    }
    std::vector<Eigen::Matrix4d> M_array;
    M_array.reserve(n);
    for (const double s : s_array) {
        const auto M = transfer_matrix(s, k, rmat);
        M_array.push_back(M);
    }
    return std::make_tuple(M_array, s_array);
}

/**
 * @brief Calculate the additive dispersion function at the end of a quadrupole.
 * @param cood0_vec Initial coordinate vector
 * @param length Quadrupole length
 * @param k1 Quadrupole strength
 * @param tilt Rotation angle (radians)
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Quadrupole::dispersion(const Eigen::Vector4d &cood0_vec,
    double length, double k1, double tilt) noexcept(false) {
    if (std::abs(k1) < IGNORE_K1_THRESHOLD) { // drift case
        return Eigen::Vector4d::Zero();
    }
    const double sqrtk = std::sqrt(std::abs(k1));
    const double psi = sqrtk * length;
    const double cospsi = std::cos(psi);
    const double sinpsi = std::sin(psi);
    const double coshpsi = std::cosh(psi);
    const double sinhpsi = std::sinh(psi);
    Eigen::Matrix2d mf1;
    mf1 << sinpsi, -cospsi/sqrtk, sqrtk*cospsi, sinpsi;
    mf1 *= 0.5 * length * sqrtk;
    Eigen::Matrix2d mf2;
    mf2 << 0., sinpsi/sqrtk, sqrtk*sinpsi, 0.;
    mf2 *= 0.5;
    Eigen::Matrix2d md1;
    md1 << -sinhpsi, -coshpsi/sqrtk, -sqrtk*coshpsi, -sinhpsi;
    md1 *= 0.5 * length * sqrtk;
    Eigen::Matrix2d md2;
    md2 << 0., sinhpsi/sqrtk, -sqrtk*sinhpsi, 0.;
    md2 *= 0.5;
    const auto mf = mf1 + mf2; // Matrix2d
    const auto md = md1 + md2; // Matrix2d
    Eigen::Vector4d disp;
    if (tilt == 0.0) {
        if (k1 < 0.0) { // defocusing in x
            disp.head<2>() = md * cood0_vec.head<2>();
            disp.tail<2>() = mf * cood0_vec.tail<2>();
        } else { // focusing in x
            disp.head<2>() = mf * cood0_vec.head<2>();
            disp.tail<2>() = md * cood0_vec.tail<2>();
        }
    } else {
        const auto R = rotation_matrix(tilt); // Matrix4d
        Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
        if (k1 < 0.0) { // defocusing in x
            M.block<2,2>(0,0) = md;
            M.block<2,2>(2,2) = mf;
        } else { // focusing in x
            M.block<2,2>(0,0) = mf;
            M.block<2,2>(2,2) = md;
        }
        disp = R.transpose() * M * R * cood0_vec;
    }
    return disp;
}

/**
 * @brief Calculate the additive dispersion function at the end of the quadrupole.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m) (unused)
 * @param method Integration method (not used here)
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Quadrupole::dispersion(
    const std::optional<Coordinate> &cood0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double k = k1_ / (1.0 + delta);
    const auto vector0 = cood0 ? cood0->vector() : Eigen::Vector4d::Zero();
    return dispersion(vector0, length_, k, tilt_);
}

/**
 * @brief Calculate the additive dispersion function array along the quadrupole.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size (m)
 * @param endpoint Whether to include the endpoint in the array
 * @param method Integration method (not used here)
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Tuple of dispersion array and s_array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Quadrupole::dispersion_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double k = k1_ / (1.0 + delta);
    const auto cood0_vec = cood0 ? cood0->vector() : Eigen::Vector4d::Zero();
    const auto s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
    for (const size_t i : std::views::iota(size_t{0}, n)) {
        disp_vector_array.col(i) = dispersion(cood0_vec, s_array(i), k, tilt_);
    }
    return std::make_tuple(disp_vector_array, s_array);
}

/**
 * @brief Calculate transferred coordinates through a quadrupole magnet.
 * @param cood0_vec Initial coordinate vector
 * @param length Length of the quadrupole magnet
 * @param k1 Quadrupole strength (1/m^2)
 * @param k0x Steering dipole strength in x direction (1/m) (negative for positive kick)
 * @param k0y Steering dipole strength in y direction (1/m) (negative for positive kick)
 * @param tilt Tilt angle (radians)
 * @return Eigen::Vector4d Transferred coordinate vector
 */
std::tuple<Eigen::Vector4d, std::optional<Eigen::Matrix4d>,
    std::optional<Eigen::Vector4d>>
egret::Quadrupole::transfer(
    const Eigen::Vector4d &cood0_vec, const double length, const double k1,
    const double k0x, const double k0y, const double tilt,
    const bool tmat_flag, const bool disp_flag) noexcept(false) {
    const double x0 = cood0_vec(0);
    const double xp0 = cood0_vec(1);
    const double y0 = cood0_vec(2);
    const double yp0 = cood0_vec(3);
    double x1, xp1, y1, yp1;
    std::optional<Eigen::Matrix4d> tmat = std::nullopt;
    std::optional<Eigen::Vector4d> disp = std::nullopt;
    if (std::abs(k1) < IGNORE_K1_THRESHOLD) { // no quadrupole, just dipole kick
        x1 = x0 + (xp0 - 0.5 * k0x * length) * length;
        y1 = y0 + (yp0 - 0.5 * k0y * length) * length;
        xp1 = xp0 - k0x * length;
        yp1 = yp0 - k0y * length;
        if (tmat_flag) {
            tmat = Drift::transfer_matrix(length);
        }
        if (disp_flag) {
            const double eta_x = 0.5 * k0x * length * length;
            const double eta_xp = k0x * length;
            const double eta_y = 0.5 * k0y * length * length;
            const double eta_yp = k0y * length;
            disp = Eigen::Vector4d(eta_x, eta_xp, eta_y, eta_yp);
        }
    } else { // with quadrupole component
        // transverse offset to generate dipole kick
        const std::complex<double> k0(k0x, k0y);
        const std::complex<double> offset =
            std::exp(std::complex<double>(0.0, 2.0 * tilt)) * std::conj(k0) / std::abs(k1);
        const double ofs_x = offset.real();
        const double ofs_y = offset.imag();
        // transfer matrix
        const auto R = rotation_matrix(tilt); // Matrix4d
        const auto M = transfer_matrix(length, std::abs(k1), R); // Matrix4d
        // coordinate after the quadrupole
        const Eigen::Vector4d cood0a_vec(ofs_x, xp0, ofs_y, yp0);
        const auto cood1a_vec = M * cood0a_vec; // Vector4d
        x1 = cood1a_vec(0) - ofs_x + x0;
        xp1 = cood1a_vec(1);
        y1 = cood1a_vec(2) - ofs_y + y0;
        yp1 = cood1a_vec(3);
        if (tmat_flag) {
            tmat = M;
        }
        if (disp_flag) {
            disp = Quadrupole::dispersion(cood0a_vec, length, std::abs(k1), tilt);
        }
    }
    const Eigen::Vector4d cood1_vec(x1, xp1, y1, yp1);
    return std::make_tuple(cood1_vec, tmat, disp);
}
