/**
 * @file dipole.cpp
 * @brief Dipole element implementation
 * @author Hirokazu Maesaka
 * @date 2025
 */
// dipole.cpp
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

#include "egret/dipole.hpp"
#include <cmath>
#include <ranges>

/**
 * @brief Calculate the transfer matrix for the dipole element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m) (not used here)
 * @param method Integration method (not used here)
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Dipole::transfer_matrix(
    const std::optional<Coordinate> &cood0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    // simple dipole transfer matrix
    if (k1_ == 0.0) {
        const double phi = angle_ / (1.0 + delta);
        const double cosphi = std::cos(phi);
        const double sinphi = std::sin(phi);
        M(0,0) = cosphi;
        M(0,1) = rho * sinphi;
        M(1,0) = -sinphi / rho;
        M(1,1) = cosphi;
        M(2,3) = length_;
        return M;
    }
    // combined-function dipole transfer matrix
    const double k1 = k1_ / (1.0 + delta);
    const double kx = k1 + 1.0 / (rho * rho);
    const double sqrtkx = std::sqrt(std::abs(kx));
    const double psix = sqrtkx * length_;
    const double ky = -k1;
    const double sqrtky = std::sqrt(std::abs(ky));
    const double psiy = sqrtky * length_;
    if (kx < 0.0) { // defocusing dipole
        const double coshx = std::cosh(psix);
        const double sinhx = std::sinh(psix);
        const double cosy = std::cos(psiy);
        const double siny = std::sin(psiy);
        M(0,0) = coshx;
        M(0,1) = sinhx/sqrtkx;
        M(1,0) = sqrtkx*sinhx;
        M(1,1) = coshx;
        M(2,2) = cosy;
        M(2,3) = siny/sqrtky;
        M(3,2) = -sqrtky*siny;
        M(3,3) = cosy;
    } else if (ky < 0.0) { // focusing dipole
        const double cosx = std::cos(psix);
        const double sinx = std::sin(psix);
        const double coshy = std::cosh(psiy);
        const double sinhy = std::sinh(psiy);
        M(0,0) = cosx;
        M(0,1) = sinx/sqrtkx;
        M(1,0) = -sqrtkx*sinx;
        M(1,1) = cosx;
        M(2,2) = coshy;
        M(2,3) = sinhy/sqrtky;
        M(3,2) = sqrtky*sinhy;
        M(3,3) = coshy;
    } else { // both focusing dipole
        const double cosx = std::cos(psix);
        const double sinx = std::sin(psix);
        const double cosy = std::cos(psiy);
        const double siny = std::sin(psiy);
        M(0,0) = cosx;
        M(0,1) = sinx/sqrtkx;
        M(1,0) = -sqrtkx*sinx;
        M(1,1) = cosx;
        M(2,2) = cosy;
        M(2,3) = siny/sqrtky;
        M(3,2) = -sqrtky*siny;
        M(3,3) = cosy;
    }
    return M;
}

/**
 * @brief Calculate the transfer matrix array for the dipole element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m)
 * @param endpoint Whether to include the endpoint in the array
 * @param method Integration method (not used here)
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Transfer matrix array and corresponding s positions
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> egret::Dipole::transfer_matrix_array(
    const std::optional<Coordinate> &cood0, const double ds,
    const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    const Eigen::ArrayXd s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    std::vector<Eigen::Matrix4d> M_array(n, Eigen::Matrix4d::Identity());
    // simple dipole transfer matrix
    if (k1_ == 0.0) {
        const auto phi_array = s_array / rho; // ArrayXd
        const auto cosphi_array = phi_array.cos(); // ArrayXd
        const auto sinphi_array = phi_array.sin(); // ArrayXd
        const auto M00_array = cosphi_array; // ArrayXd
        const auto M01_array = rho * sinphi_array; // ArrayXd
        const auto M10_array = -sinphi_array / rho; // ArrayXd
        const auto M11_array = cosphi_array; // ArrayXd
        const auto M23_array = s_array; // ArrayXd
        for (const auto i : std::views::iota(0u, n)) {
            M_array[i](0,0) = M00_array(i);
            M_array[i](0,1) = M01_array(i);
            M_array[i](1,0) = M10_array(i);
            M_array[i](1,1) = M11_array(i);
            M_array[i](2,3) = M23_array(i);
        }
        return std::make_tuple(M_array, s_array);
    }
    // combined-function dipole transfer matrix
    const double k1 = k1_ / (1.0 + delta);
    const double kx = k1 + 1.0 / (rho * rho);
    const double sqrtkx = std::sqrt(std::abs(kx));
    const double ky = -k1;
    const double sqrtky = std::sqrt(std::abs(ky));
    const auto psix_array = sqrtkx * s_array; // ArrayXd
    const auto psiy_array = sqrtky * s_array; // ArrayXd
    if (kx < 0.0) { // defocusing dipole
        const auto coshx_array = psix_array.cosh(); // ArrayXd
        const auto sinhx_array = psix_array.sinh(); // ArrayXd
        const auto cosy_array = psiy_array.cos(); // ArrayXd
        const auto siny_array = psiy_array.sin(); // ArrayXd
        const auto M00_array = coshx_array; // ArrayXd
        const auto M01_array = sinhx_array / sqrtkx; // ArrayXd
        const auto M10_array = sqrtkx * sinhx_array; // ArrayXd
        const auto M11_array = coshx_array; // ArrayXd
        const auto M22_array = cosy_array; // ArrayXd
        const auto M23_array = siny_array / sqrtky; // ArrayXd
        const auto M32_array = -sqrtky * siny_array; // ArrayXd
        const auto M33_array = cosy_array; // ArrayXd
        for (const auto i : std::views::iota(0u, n)) {
            M_array[i](0,0) = M00_array(i);
            M_array[i](0,1) = M01_array(i);
            M_array[i](1,0) = M10_array(i);
            M_array[i](1,1) = M11_array(i);
            M_array[i](2,2) = M22_array(i);
            M_array[i](2,3) = M23_array(i);
            M_array[i](3,2) = M32_array(i);
            M_array[i](3,3) = M33_array(i);
        }
    } else if (ky < 0.0) { // focusing dipole
        const auto cosx_array = psix_array.cos(); // ArrayXd
        const auto sinx_array = psix_array.sin(); // ArrayXd
        const auto coshy_array = psiy_array.cosh(); // ArrayXd
        const auto sinhy_array = psiy_array.sinh(); // ArrayXd
        const auto M00_array = cosx_array; // ArrayXd
        const auto M01_array = sinx_array / sqrtkx; // ArrayXd
        const auto M10_array = -sqrtkx * sinx_array; // ArrayXd
        const auto M11_array = cosx_array; // ArrayXd
        const auto M22_array = coshy_array; // ArrayXd
        const auto M23_array = sinhy_array / sqrtky; // ArrayXd
        const auto M32_array = sqrtky * sinhy_array; // ArrayXd
        const auto M33_array = coshy_array; // ArrayXd
        for (const auto i : std::views::iota(0u, n)) {
            M_array[i](0,0) = M00_array(i);
            M_array[i](0,1) = M01_array(i);
            M_array[i](1,0) = M10_array(i);
            M_array[i](1,1) = M11_array(i);
            M_array[i](2,2) = M22_array(i);
            M_array[i](2,3) = M23_array(i);
            M_array[i](3,2) = M32_array(i);
            M_array[i](3,3) = M33_array(i);
        }
    } else { // both focusing dipole
        const auto cosx_array = psix_array.cos(); // ArrayXd
        const auto sinx_array = psix_array.sin(); // ArrayXd
        const auto cosy_array = psiy_array.cos(); // ArrayXd
        const auto siny_array = psiy_array.sin(); // ArrayXd
        const auto M00_array = cosx_array; // ArrayXd
        const auto M01_array = sinx_array / sqrtkx; // ArrayXd
        const auto M10_array = -sqrtkx * sinx_array; // ArrayXd
        const auto M11_array = cosx_array; // ArrayXd
        const auto M22_array = cosy_array; // ArrayXd
        const auto M23_array = siny_array / sqrtky; // ArrayXd
        const auto M32_array = -sqrtky * siny_array; // ArrayXd
        const auto M33_array = cosy_array; // ArrayXd
        for (const auto i : std::views::iota(0u, n)) {
            M_array[i](0,0) = M00_array(i);
            M_array[i](0,1) = M01_array(i);
            M_array[i](1,0) = M10_array(i);
            M_array[i](1,1) = M11_array(i);
            M_array[i](2,2) = M22_array(i);
            M_array[i](2,3) = M23_array(i);
            M_array[i](3,2) = M32_array(i);
            M_array[i](3,3) = M33_array(i);
        }
    }
    return std::make_tuple(M_array, s_array);
}

/**
 * @brief Calculate the additive dispersion function at the end of the dipole.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m) (not used here)
 * @param method Integration method (not used here)
 * @return Eigen::Vector4d
 */
Eigen::Vector4d egret::Dipole::dispersion(const std::optional<Coordinate> &cood0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    const auto vector0 = cood0 ? cood0->vector() : Eigen::Vector4d::Zero(); // Vector4d
    Eigen::Vector4d disp = Eigen::Vector4d::Zero();
    // simple dipole dispersion
    if (k1_ == 0.0) {
        const double phi = angle_ / (1.0 + delta);
        const double cosphi = std::cos(phi);
        const double sinphi = std::sin(phi);
        disp(0) = rho * (1.0 - cosphi);
        disp(1) = sinphi;
        Eigen::Matrix2d Mx1;
        Mx1 << sinphi, -rho * cosphi, cosphi / rho, sinphi;
        Mx1 *= 0.5 * length_ / rho;
        Eigen::Matrix2d Mx2;
        Mx2 << 0., rho * sinphi, sinphi / rho, 0.;
        Mx2 *= 0.5;
        disp.head<2>() += (Mx1 + Mx2) * vector0.head<2>();
        return disp;
    }
    // combined-function dipole dispersion
    const double k1 = k1_ / (1.0 + delta);
    const double kx = k1 + 1.0 / (rho * rho);
    const double sqrtkx = std::sqrt(std::abs(kx));
    const double psix = sqrtkx * length_;
    const double ky = -k1;
    const double sqrtky = std::sqrt(std::abs(ky));
    const double psiy = sqrtky * length_;
    if (kx < 0.0) { // defocusing dipole
        const double coshx = std::cosh(psix);
        const double sinhx = std::sinh(psix);
        const double cosy = std::cos(psiy);
        const double siny = std::sin(psiy);
        disp(0) = (1.0 - coshx) / (kx * rho);
        disp(1) = sinhx / (sqrtkx * rho);
        Eigen::Matrix2d Mx1;
        Mx1 << -sinhx, -coshx/sqrtkx, -sqrtkx*coshx, -sinhx;
        Mx1 *= 0.5 * length_ * sqrtkx;
        Eigen::Matrix2d Mx2;
        Mx2 << 0., sinhx/sqrtkx, -sqrtkx*sinhx, 0.;
        Mx2 *= 0.5;
        disp.head<2>() += (Mx1 + Mx2) * vector0.head<2>();
        Eigen::Matrix2d My1;
        My1 << siny, -cosy/sqrtky, sqrtky*cosy, siny;
        My1 *= 0.5 * length_ * sqrtky;
        Eigen::Matrix2d My2;
        My2 << 0., siny/sqrtky, sqrtky*siny, 0.;
        My2 *= 0.5;
        disp.tail<2>() += (My1 + My2) * vector0.tail<2>();
    } else if (ky < 0.0) { // focusing dipole
        const double cosx = std::cos(psix);
        const double sinx = std::sin(psix);
        const double coshy = std::cosh(psiy);
        const double sinhy = std::sinh(psiy);
        disp(0) = (1.0 - cosx) / (kx * rho);
        disp(1) = sinx / (sqrtkx * rho);
        Eigen::Matrix2d Mx1;
        Mx1 << sinx, -cosx/sqrtkx, sqrtkx*cosx, sinx;
        Mx1 *= 0.5 * length_ * sqrtkx;
        Eigen::Matrix2d Mx2;
        Mx2 << 0., sinx/sqrtkx, sqrtkx*sinx, 0.;
        Mx2 *= 0.5;
        disp.head<2>() += (Mx1 + Mx2) * vector0.head<2>();
        Eigen::Matrix2d My1;
        My1 << -sinhy, -coshy/sqrtky, -sqrtky*coshy, -sinhy;
        My1 *= 0.5 * length_ * sqrtky;
        Eigen::Matrix2d My2;
        My2 << 0., sinhy/sqrtky, -sqrtky*sinhy, 0.;
        My2 *= 0.5;
        disp.tail<2>() += (My1 + My2) * vector0.tail<2>();
    } else { // both focusing dipole
        const double cosx = std::cos(psix);
        const double sinx = std::sin(psix);
        const double cosy = std::cos(psiy);
        const double siny = std::sin(psiy);
        disp(0) = (1.0 - cosx) / (kx * rho);
        disp(1) = sinx / (sqrtkx * rho);
        Eigen::Matrix2d Mx1;
        Mx1 << sinx, -cosx/sqrtkx, sqrtkx*cosx, sinx;
        Mx1 *= 0.5 * length_ * sqrtkx;
        Eigen::Matrix2d Mx2;
        Mx2 << 0., sinx/sqrtkx, sqrtkx*sinx, 0.;
        Mx2 *= 0.5;
        disp.head<2>() += (Mx1 + Mx2) * vector0.head<2>();
        Eigen::Matrix2d My1;
        My1 << siny, -cosy/sqrtky, sqrtky*cosy, siny;
        My1 *= 0.5 * length_ * sqrtky;
        Eigen::Matrix2d My2;
        My2 << 0., siny/sqrtky, sqrtky*siny, 0.;
        My2 *= 0.5;
        disp.tail<2>() += (My1 + My2) * vector0.tail<2>();
    }
    return disp;
}

/**
 * @brief Calculate array of additive dispersion functions along the dipole.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m)
 * @param endpoint Whether to include the endpoint in the array
 * @param method Integration method (not used here)
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Array of dispersion vectors and s array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> egret::Dipole::dispersion_array(
    const std::optional<Coordinate> &cood0, const double ds,
    const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    const auto vector0 = cood0 ? cood0->vector() : Eigen::Vector4d::Zero(); // Vector4d
    const auto s_array = Element::s_array(ds, endpoint); // ArrayXd
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_array(4, n);
    disp_array.setZero();
    // simple dipole dispersion
    if (k1_ == 0.0) {
        const auto phi_array = s_array / rho; // ArrayXd
        const auto cosphi_array = phi_array.cos(); // ArrayXd
        const auto sinphi_array = phi_array.sin(); // ArrayXd
        const auto s_cosphi_array = s_array * cosphi_array; // ArrayXd
        const auto s_sinphi_array = s_array * sinphi_array; // ArrayXd
        disp_array.row(0) = rho * (1.0 - cosphi_array);
        disp_array.row(1) = sinphi_array;
        Eigen::Matrix<double, Eigen::Dynamic, 2> Mx1(n*2, 2);
        auto Mx1_00 = Mx1.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
        auto Mx1_10 = Mx1.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
        auto Mx1_01 = Mx1.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
        auto Mx1_11 = Mx1.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
        Mx1_00 = 0.5 * s_sinphi_array / rho;
        Mx1_01 = -0.5 * s_cosphi_array;
        Mx1_10 = 0.5 * s_cosphi_array / (rho * rho);
        Mx1_11 = 0.5 * s_sinphi_array / rho;
        Eigen::Matrix<double, Eigen::Dynamic, 2> Mx2(n*2, 2);
        auto Mx2_00 = Mx2.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
        auto Mx2_10 = Mx2.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
        auto Mx2_01 = Mx2.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
        auto Mx2_11 = Mx2.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
        Mx2_00.setZero();
        Mx2_01 = 0.5 * rho * sinphi_array;
        Mx2_10 = 0.5 * sinphi_array / rho;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        return std::make_tuple(disp_array, s_array);
    }
    // combined-function dipole dispersion
    const double k1 = k1_ / (1.0 + delta);
    const double kx = k1 + 1.0 / (rho * rho);
    const double sqrtkx = std::sqrt(std::abs(kx));
    const double ky = -k1;
    const double sqrtky = std::sqrt(std::abs(ky));
    const auto psix_array = sqrtkx * s_array;
    const auto psiy_array = sqrtky * s_array;
    Eigen::Matrix<double, Eigen::Dynamic, 2> Mx1(n*2, 2);
    auto Mx1_00 = Mx1.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto Mx1_10 = Mx1.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto Mx1_01 = Mx1.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto Mx1_11 = Mx1.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    Eigen::Matrix<double, Eigen::Dynamic, 2> Mx2(n*2, 2);
    auto Mx2_00 = Mx2.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto Mx2_10 = Mx2.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto Mx2_01 = Mx2.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto Mx2_11 = Mx2.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    Eigen::Matrix<double, Eigen::Dynamic, 2> My1(n*2, 2);
    auto My1_00 = My1.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto My1_10 = My1.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto My1_01 = My1.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto My1_11 = My1.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    Eigen::Matrix<double, Eigen::Dynamic, 2> My2(n*2, 2);
    auto My2_00 = My2.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto My2_10 = My2.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto My2_01 = My2.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto My2_11 = My2.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    if (kx < 0.0) { // defocusing dipole
        const auto coshx_array = psix_array.cosh(); // ArrayXd
        const auto sinhx_array = psix_array.sinh(); // ArrayXd
        const auto cosy_array = psiy_array.cos(); // ArrayXd
        const auto siny_array = psiy_array.sin(); // ArrayXd
        const auto s_coshx_array = s_array * coshx_array; // ArrayXd
        const auto s_sinhx_array = s_array * sinhx_array; // ArrayXd
        const auto s_cosy_array = s_array * cosy_array; // ArrayXd
        const auto s_siny_array = s_array * siny_array; // ArrayXd
        disp_array.row(0) = (1.0 - coshx_array) / (kx * rho);
        disp_array.row(1) = sinhx_array / (sqrtkx * rho);
        Mx1_00 = -0.5 * sqrtkx * s_sinhx_array;
        Mx1_01 = -0.5 * s_coshx_array;
        Mx1_10 = -0.5 * std::abs(kx) * s_coshx_array;
        Mx1_11 = -0.5 * sqrtkx * s_sinhx_array;
        Mx2_00.setZero();
        Mx2_01 = 0.5 * sinhx_array / sqrtkx;
        Mx2_10 = -0.5 * sqrtkx * sinhx_array;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        My1_00 = 0.5 * sqrtky * s_siny_array;
        My1_01 = -0.5 * s_cosy_array;
        My1_10 = 0.5 * std::abs(ky) * s_cosy_array;
        My1_11 = 0.5 * sqrtky * s_siny_array;
        My2_00.setZero();
        My2_01 = 0.5 * siny_array / sqrtky;
        My2_10 = 0.5 * sqrtky * siny_array;
        My2_11.setZero();
        disp_array.bottomRows<2>() += ((My1 + My2) * vector0.tail<2>()).reshaped(2, n);
    } else if (ky < 0.0) { // focusing dipole
        const auto cosx_array = psix_array.cos(); // ArrayXd
        const auto sinx_array = psix_array.sin(); // ArrayXd
        const auto coshy_array = psiy_array.cosh(); // ArrayXd
        const auto sinhy_array = psiy_array.sinh(); // ArrayXd
        const auto s_cosx_array = s_array * cosx_array; // ArrayXd
        const auto s_sinx_array = s_array * sinx_array; // ArrayXd
        const auto s_coshy_array = s_array * coshy_array; // ArrayXd
        const auto s_sinhy_array = s_array * sinhy_array; // ArrayXd
        disp_array.row(0) = (1.0 - cosx_array) / (kx * rho);
        disp_array.row(1) = sinx_array / (sqrtkx * rho);
        Mx1_00 = 0.5 * sqrtkx * s_sinx_array;
        Mx1_01 = -0.5 * s_cosx_array;
        Mx1_10 = 0.5 * std::abs(ky) * s_cosx_array;
        Mx1_11 = 0.5 * sqrtkx * s_sinx_array;
        Mx2_00.setZero();
        Mx2_01 = 0.5 * sinx_array / sqrtkx;
        Mx2_10 = 0.5 * sqrtkx * sinx_array;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        My1_00 = -0.5 * sqrtky * s_sinhy_array;
        My1_01 = -0.5 * s_coshy_array;
        My1_10 = -0.5 * std::abs(ky) * s_coshy_array;
        My1_11 = -0.5 * sqrtky * s_sinhy_array;
        My2_00.setZero();
        My2_01 = 0.5 * sinhy_array / sqrtky;
        My2_10 = -0.5 * sqrtky * sinhy_array;
        My2_11.setZero();
        disp_array.bottomRows<2>() += ((My1 + My2) * vector0.tail<2>()).reshaped(2, n);
    } else { // both focusing dipole
        const auto cosx_array = psix_array.cos(); // ArrayXd
        const auto sinx_array = psix_array.sin(); // ArrayXd
        const auto cosy_array = psiy_array.cos(); // ArrayXd
        const auto siny_array = psiy_array.sin(); // ArrayXd
        const auto s_cosx_array = s_array * cosx_array; // ArrayXd
        const auto s_sinx_array = s_array * sinx_array; // ArrayXd
        const auto s_cosy_array = s_array * cosy_array; // ArrayXd
        const auto s_siny_array = s_array * siny_array; // ArrayXd
        disp_array.row(0) = (1.0 - cosx_array) / (kx * rho);
        disp_array.row(1) = sinx_array / (sqrtkx * rho);
        Mx1_00 = 0.5 * sqrtkx * s_sinx_array;
        Mx1_01 = -0.5 * s_cosx_array;
        Mx1_10 = 0.5 * std::abs(ky) * s_cosx_array;
        Mx1_11 = 0.5 * sqrtkx * s_sinx_array;
        Mx2_00.setZero();
        Mx2_01 = 0.5 * sinx_array / sqrtkx;
        Mx2_10 = 0.5 * sqrtkx * sinx_array;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        My1_00 = 0.5 * sqrtky * s_siny_array;
        My1_01 = -0.5 * s_cosy_array;
        My1_10 = 0.5 * std::abs(ky) * s_cosy_array;
        My1_11 = 0.5 * sqrtky * s_siny_array;
        My2_00.setZero();
        My2_01 = 0.5 * siny_array / sqrtky;
        My2_10 = 0.5 * sqrtky * siny_array;
        My2_11.setZero();
        disp_array.bottomRows<2>() += ((My1 + My2) * vector0.tail<2>()).reshaped(2, n);
    }
    return std::make_tuple(disp_array, s_array);
}

/**
 * @brief Transfer the coordinate, envelope, and dispersion through the dipole element.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size for integration (m)
 * @param method Integration method (not used here)
 * @return std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>> Coordinate, envelope, and dispersion after transfer
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Dipole::transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    Coordinate cood = cood0;
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    cood.s(cood.s() - ds_);
    const Coordinate cood0err = cood;
    const auto M = transfer_matrix(cood0err, ds); // Matrix4d
    const auto disp_add = dispersion(Coordinate(), ds); // Vector4d
    cood.vector(M * cood0err.vector() + disp_add * cood.delta());
    cood.x(cood.x() + dx_);
    cood.y(cood.y() + dy_);
    cood.s(cood.s() + ds_ + length_);
    std::optional<Envelope> evlp = evlp0;
    if (evlp) {
        evlp->transfer(M, length_);
    }
    std::optional<Dispersion> disp = disp0;
    if (disp) {
        const auto disp_add2 = dispersion(cood0err, ds); // Vector4d
        const auto disp_vec = M * disp->vector() + disp_add2; // Vector4d
        disp = Dispersion(disp_vec, disp->s() + length_);
    }
    return std::make_tuple(cood, evlp, disp);
}

/**
 * @brief Calculate coordinate array, envelope array, and dispersion array after transfer through the dipole.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size (m)
 * @param endpoint Whether to include the endpoint in the calculation
 * @param method Integration method (not used here)
 * @return std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>> Coordinate array, envelope array, and dispersion array after transfer
 */
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>>
egret::Dipole::transfer_array(const Coordinate &cood0,
    const std::optional<Envelope> &evlp0, const std::optional<Dispersion> &disp0,
    const double ds, const bool endpoint, const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    Coordinate cood = cood0;
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    cood.s(cood.s() - ds_);
    const Coordinate cood0err = cood;
    const auto [M_array, s_array] = transfer_matrix_array(cood0err, ds, endpoint); // vector<Matrix4d>, ArrayXd
    const size_t n = M_array.size();
    Eigen::Matrix<double, Eigen::Dynamic, 4> M_combined(n*4, 4);
    for (const auto i : std::views::iota(0u, n)) {
        M_combined.block(i*4, 0, 4, 4) = M_array[i];
    }
    const auto [disp_array_mat, _] = dispersion_array(Coordinate(), ds, endpoint); // Matrix, ArrayXd
    Eigen::Matrix<double, 4, Eigen::Dynamic> vector_array =
        (M_combined * cood0err.vector()).reshaped(4, n) + disp_array_mat * cood0err.delta();
    vector_array.row(0).array() += dx_;
    vector_array.row(2).array() += dy_;
    const CoordinateArray cood_array(vector_array, cood0err.s() + ds_ + s_array,
        Eigen::ArrayXd::Constant(n, cood0err.delta()),
        Eigen::ArrayXd::Constant(n, cood0err.z()));
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    if (evlp0) {
        evlp_array = EnvelopeArray::transport(*evlp0, M_array, s_array);
    }
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (disp0) {
        const auto [disp_add, _] = dispersion_array(cood0err, ds, endpoint); // Matrix, ArrayXd
        const auto disp_vector_array = (M_combined * disp0->vector()).reshaped(4, n)
            + disp_add; // Matrix 4 x n
        disp_array = DispersionArray(disp_vector_array, disp0->s() + s_array);
    }
    return std::make_tuple(cood_array, evlp_array, disp_array);
}

/**
 * @brief Calculate radiation integrals through the dipole.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope
 * @param disp0 Initial dispersion
 * @param ds Maximum step size (m)
 * @param method Integration method
 * @return std::tuple<double, double, double, double, double, double> Tuple of radiation integrals I2, I4, I5u, I5v, I4u, I4v
 */
std::tuple<double, double, double, double, double, double>
egret::Dipole::radiation_integrals(const Coordinate &cood0, const Envelope &evlp0,
    const Dispersion &disp0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double kappa = angle_ / length_;
    const auto &[cood_array, evlp_array, disp_array] =
        transfer_array(cood0, evlp0, disp0, ds, true);
    const size_t n = evlp_array->size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> dispuv(4, n);
    for (const size_t i : std::views::iota(0u, n)) {
        dispuv.col(i) = evlp_array->T_matrix(i) * disp_array->vector_array().col(i);
    }
    const double dz = evlp_array->ds();
    const double kappa2 = kappa * kappa;
    const double kappa3 = kappa2 * kappa;
    const auto tau_array = evlp_array->tau_array(); // ArrayXd
    const auto eta_x_array = disp_array->x_array(); // ArrayXd
    const Eigen::ArrayXd eta_u_array = dispuv.row(0);
    const Eigen::ArrayXd etap_u_array = dispuv.row(1);
    const Eigen::ArrayXd eta_v_array = dispuv.row(2);
    const Eigen::ArrayXd etap_v_array = dispuv.row(3);
    const auto bu_array = evlp_array->bu_array(); // ArrayXd
    const auto au_array = evlp_array->au_array(); // ArrayXd
    const auto gu_array = evlp_array->gu_array(); // ArrayXd
    const auto bv_array = evlp_array->bv_array(); // ArrayXd
    const auto av_array = evlp_array->av_array(); // ArrayXd
    const auto gv_array = evlp_array->gv_array(); // ArrayXd
    const double I2 = length_ * kappa2;
    const double I4 = simpson_integration(
        eta_x_array * kappa * (kappa2 + 2.0 * k1_), dz);
    const double I4u = simpson_integration(
        tau_array * eta_u_array * kappa * (kappa2 + 2.0 * k1_), dz);
    const double I4v = I4 - I4u;
    const double I5u = simpson_integration(kappa3 *
        ( bu_array * etap_u_array * etap_u_array
        + 2.0 * au_array * eta_u_array * etap_u_array
        + gu_array * eta_u_array * eta_u_array), dz);
    const double I5v = simpson_integration(kappa3 *
        ( bv_array * etap_v_array * etap_v_array
        + 2.0 * av_array * eta_v_array * etap_v_array
        + gv_array * eta_v_array * eta_v_array), dz);
    return std::make_tuple(I2, I4, I5u, I5v, I4u, I4v);
}
