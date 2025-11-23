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
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Dipole::transfer_matrix(
    const std::optional<Coordinate> &cood0, double ds) const noexcept {
    (void)ds; // unused parameter
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
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Transfer matrix array and corresponding s positions
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> egret::Dipole::transfer_matrix_array(
    const std::optional<Coordinate> &cood0, double ds,
    bool endpoint) const noexcept {
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    const Eigen::ArrayXd s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    std::vector<Eigen::Matrix4d> M_array(n, Eigen::Matrix4d::Identity());
    // simple dipole transfer matrix
    if (k1_ == 0.0) {
        for (const auto i : std::views::iota(0u, n)) {
            const double s = s_array(i);
            const double phi = s / rho;
            const double cosphi = std::cos(phi);
            const double sinphi = std::sin(phi);
            M_array[i](0,0) = cosphi;
            M_array[i](0,1) = rho * sinphi;
            M_array[i](1,0) = -sinphi / rho;
            M_array[i](1,1) = cosphi;
            M_array[i](2,3) = s;
        }
    }
    // combined-function dipole transfer matrix
    else {
        const double k1 = k1_ / (1.0 + delta);
        const double kx = k1 + 1.0 / (rho * rho);
        const double sqrtkx = std::sqrt(std::abs(kx));
        const double ky = -k1;
        const double sqrtky = std::sqrt(std::abs(ky));
        for (const auto i : std::views::iota(0u, n)) {
            const double s = s_array(i);
            const double psix = sqrtkx * s;
            const double psiy = sqrtky * s;
            if (kx < 0.0) { // defocusing dipole
                const double coshx = std::cosh(psix);
                const double sinhx = std::sinh(psix);
                const double cosy = std::cos(psiy);
                const double siny = std::sin(psiy);
                M_array[i](0,0) = coshx;
                M_array[i](0,1) = sinhx/sqrtkx;
                M_array[i](1,0) = sqrtkx*sinhx;
                M_array[i](1,1) = coshx;
                M_array[i](2,2) = cosy;
                M_array[i](2,3) = siny/sqrtky;
                M_array[i](3,2) = -sqrtky*siny;
                M_array[i](3,3) = cosy;
            } else if (ky < 0.0) { // focusing dipole
                const double cosx = std::cos(psix);
                const double sinx = std::sin(psix);
                const double coshy = std::cosh(psiy);
                const double sinhy = std::sinh(psiy);
                M_array[i](0,0) = cosx;
                M_array[i](0,1) = sinx/sqrtkx;
                M_array[i](1,0) = -sqrtkx*sinx;
                M_array[i](1,1) = cosx;
                M_array[i](2,2) = coshy;
                M_array[i](2,3) = sinhy/sqrtky;
                M_array[i](3,2) = sqrtky*sinhy;
                M_array[i](3,3) = coshy;
            } else { // both focusing dipole
                const double cosx = std::cos(psix);
                const double sinx = std::sin(psix);
                const double cosy = std::cos(psiy);
                const double siny = std::sin(psiy);
                M_array[i](0,0) = cosx;
                M_array[i](0,1) = sinx/sqrtkx;
                M_array[i](1,0) = -sqrtkx*sinx;
                M_array[i](1,1) = cosx;
                M_array[i](2,2) = cosy;
                M_array[i](2,3) = siny/sqrtky;
                M_array[i](3,2) = -sqrtky*siny;
                M_array[i](3,3) = cosy;
            }
        }
    }
    return {M_array, s_array};
}

/**
 * @brief Calculate the additive dispersion function at the end of the dipole.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m) (not used here)
 * @return Eigen::Vector4d
 */
Eigen::Vector4d egret::Dipole::dispersion(const std::optional<Coordinate> &cood0, double ds) const noexcept {
    (void)ds; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    const Eigen::Vector4d vector0 = cood0 ? cood0->vector() : Eigen::Vector4d::Zero();
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
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Array of dispersion vectors and s array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> egret::Dipole::dispersion_array(
    const std::optional<Coordinate> &cood0, double ds,
    bool endpoint) const noexcept {
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double rho = (length_ / angle_) * (1.0 + delta);
    const Eigen::Vector4d vector0 = cood0 ? cood0->vector() : Eigen::Vector4d::Zero();
    const Eigen::ArrayXd s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_array(4, n);
    disp_array.setZero();
    // simple dipole dispersion
    if (k1_ == 0.0) {
        const Eigen::ArrayXd phi_array = s_array / rho;
        const Eigen::ArrayXd cosphi_array = phi_array.cos();
        const Eigen::ArrayXd sinphi_array = phi_array.sin();
        disp_array.row(0) = rho * (1.0 - cosphi_array);
        disp_array.row(1) = sinphi_array;
        Eigen::Matrix<double, Eigen::Dynamic, 2> Mx1(n*2, 2);
        auto Mx1_00 = Mx1.col(0)(Eigen::seq(0, Eigen::last, 2));
        auto Mx1_10 = Mx1.col(0)(Eigen::seq(1, Eigen::last, 2));
        auto Mx1_01 = Mx1.col(1)(Eigen::seq(0, Eigen::last, 2));
        auto Mx1_11 = Mx1.col(1)(Eigen::seq(1, Eigen::last, 2));
        Mx1_00 = 0.5 * s_array * sinphi_array / rho;
        Mx1_01 = -0.5 * s_array * cosphi_array;
        Mx1_10 = 0.5 * s_array * cosphi_array / (rho * rho);
        Mx1_11 = 0.5 * s_array * sinphi_array / rho;
        Eigen::Matrix<double, Eigen::Dynamic, 2> Mx2(n*2, 2);
        auto Mx2_00 = Mx2.col(0)(Eigen::seq(0, Eigen::last, 2));
        auto Mx2_10 = Mx2.col(0)(Eigen::seq(1, Eigen::last, 2));
        auto Mx2_01 = Mx2.col(1)(Eigen::seq(0, Eigen::last, 2));
        auto Mx2_11 = Mx2.col(1)(Eigen::seq(1, Eigen::last, 2));
        Mx2_00.setZero();
        Mx2_01 = 0.5 * rho * sinphi_array;
        Mx2_10 = 0.5 * sinphi_array / rho;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        return {disp_array, s_array};
    }
    // combined-function dipole dispersion
    const double k1 = k1_ / (1.0 + delta);
    const double kx = k1 + 1.0 / (rho * rho);
    const double sqrtkx = std::sqrt(std::abs(kx));
    const double ky = -k1;
    const double sqrtky = std::sqrt(std::abs(ky));
    const Eigen::ArrayXd psix_array = sqrtkx * s_array;
    const Eigen::ArrayXd psiy_array = sqrtky * s_array;
    Eigen::Matrix<double, Eigen::Dynamic, 2> Mx1(n*2, 2);
    auto Mx1_00 = Mx1.col(0)(Eigen::seq(0, Eigen::last, 2));
    auto Mx1_10 = Mx1.col(0)(Eigen::seq(1, Eigen::last, 2));
    auto Mx1_01 = Mx1.col(1)(Eigen::seq(0, Eigen::last, 2));
    auto Mx1_11 = Mx1.col(1)(Eigen::seq(1, Eigen::last, 2));
    Eigen::Matrix<double, Eigen::Dynamic, 2> Mx2(n*2, 2);
    auto Mx2_00 = Mx2.col(0)(Eigen::seq(0, Eigen::last, 2));
    auto Mx2_10 = Mx2.col(0)(Eigen::seq(1, Eigen::last, 2));
    auto Mx2_01 = Mx2.col(1)(Eigen::seq(0, Eigen::last, 2));
    auto Mx2_11 = Mx2.col(1)(Eigen::seq(1, Eigen::last, 2));
    Eigen::Matrix<double, Eigen::Dynamic, 2> My1(n*2, 2);
    auto My1_00 = My1.col(0)(Eigen::seq(0, Eigen::last, 2));
    auto My1_10 = My1.col(0)(Eigen::seq(1, Eigen::last, 2));
    auto My1_01 = My1.col(1)(Eigen::seq(0, Eigen::last, 2));
    auto My1_11 = My1.col(1)(Eigen::seq(1, Eigen::last, 2));
    Eigen::Matrix<double, Eigen::Dynamic, 2> My2(n*2, 2);
    auto My2_00 = My2.col(0)(Eigen::seq(0, Eigen::last, 2));
    auto My2_10 = My2.col(0)(Eigen::seq(1, Eigen::last, 2));
    auto My2_01 = My2.col(1)(Eigen::seq(0, Eigen::last, 2));
    auto My2_11 = My2.col(1)(Eigen::seq(1, Eigen::last, 2));
    if (kx < 0.0) { // defocusing dipole
        const Eigen::ArrayXd coshx_array = psix_array.cosh();
        const Eigen::ArrayXd sinhx_array = psix_array.sinh();
        const Eigen::ArrayXd cosy_array = psiy_array.cos();
        const Eigen::ArrayXd siny_array = psiy_array.sin();
        disp_array.row(0) = (1.0 - coshx_array) / (kx * rho);
        disp_array.row(1) = sinhx_array / (sqrtkx * rho);
        Mx1_00 = -0.5 * sqrtkx * s_array * sinhx_array;
        Mx1_01 = -0.5 * s_array * coshx_array;
        Mx1_10 = -0.5 * std::abs(kx) * s_array * coshx_array;
        Mx1_11 = -0.5 * sqrtkx * s_array * sinhx_array;
        Mx2_00.setZero();
        Mx2_01 = 0.5 * sinhx_array / sqrtkx;
        Mx2_10 = -0.5 * sqrtkx * sinhx_array;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        My1_00 = 0.5 * sqrtky * s_array * siny_array;
        My1_01 = -0.5 * s_array * cosy_array;
        My1_10 = 0.5 * std::abs(ky) * s_array * cosy_array;
        My1_11 = 0.5 * sqrtky * s_array * siny_array;
        My2_00.setZero();
        My2_01 = 0.5 * siny_array / sqrtky;
        My2_10 = 0.5 * sqrtky * siny_array;
        My2_11.setZero();
        disp_array.bottomRows<2>() += ((My1 + My2) * vector0.tail<2>()).reshaped(2, n);
    } else if (ky < 0.0) { // focusing dipole
        const Eigen::ArrayXd cosx_array = psix_array.cos();
        const Eigen::ArrayXd sinx_array = psix_array.sin();
        const Eigen::ArrayXd coshy_array = psiy_array.cosh();
        const Eigen::ArrayXd sinhy_array = psiy_array.sinh();
        disp_array.row(0) = (1.0 - cosx_array) / (kx * rho);
        disp_array.row(1) = sinx_array / (sqrtkx * rho);
        Mx1_00 = 0.5 * sqrtkx * s_array * sinx_array;
        Mx1_01 = -0.5 * s_array * cosx_array;
        Mx1_10 = 0.5 * std::abs(ky) * s_array * cosx_array;
        Mx1_11 = 0.5 * sqrtkx * s_array * sinx_array;
        Mx2_00.setZero();
        Mx2_01 = 0.5 * sinx_array / sqrtkx;
        Mx2_10 = 0.5 * sqrtkx * sinx_array;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        My1_00 = -0.5 * sqrtky * s_array * sinhy_array;
        My1_01 = -0.5 * s_array * coshy_array;
        My1_10 = -0.5 * std::abs(ky) * s_array * coshy_array;
        My1_11 = -0.5 * sqrtky * s_array * sinhy_array;
        My2_00.setZero();
        My2_01 = 0.5 * sinhy_array / sqrtky;
        My2_10 = -0.5 * sqrtky * sinhy_array;
        My2_11.setZero();
        disp_array.bottomRows<2>() += ((My1 + My2) * vector0.tail<2>()).reshaped(2, n);
    } else { // both focusing dipole
        const Eigen::ArrayXd cosx_array = psix_array.cos();
        const Eigen::ArrayXd sinx_array = psix_array.sin();
        const Eigen::ArrayXd cosy_array = psiy_array.cos();
        const Eigen::ArrayXd siny_array = psiy_array.sin();
        disp_array.row(0) = (1.0 - cosx_array) / (kx * rho);
        disp_array.row(1) = sinx_array / (sqrtkx * rho);
        Mx1_00 = 0.5 * sqrtkx * s_array * sinx_array;
        Mx1_01 = -0.5 * s_array * cosx_array;
        Mx1_10 = 0.5 * std::abs(ky) * s_array * cosx_array;
        Mx1_11 = 0.5 * sqrtkx * s_array * sinx_array;
        Mx2_00.setZero();
        Mx2_01 = 0.5 * sinx_array / sqrtkx;
        Mx2_10 = 0.5 * sqrtkx * sinx_array;
        Mx2_11.setZero();
        disp_array.topRows<2>() += ((Mx1 + Mx2) * vector0.head<2>()).reshaped(2, n);
        My1_00 = 0.5 * sqrtky * s_array * siny_array;
        My1_01 = -0.5 * s_array * cosy_array;
        My1_10 = 0.5 * std::abs(ky) * s_array * cosy_array;
        My1_11 = 0.5 * sqrtky * s_array * siny_array;
        My2_00.setZero();
        My2_01 = 0.5 * siny_array / sqrtky;
        My2_10 = 0.5 * sqrtky * siny_array;
        My2_11.setZero();
        disp_array.bottomRows<2>() += ((My1 + My2) * vector0.tail<2>()).reshaped(2, n);
    }
    return {disp_array, s_array};
}

std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Dipole::transfer(
    const Coordinate &cood0, double ds,
    const std::optional<Envelope> &env0,
    const std::optional<Dispersion> &disp0) const noexcept {
    const Eigen::Matrix4d M = transfer_matrix(cood0, ds);
    const Eigen::Vector4d new_vector = M * cood0.vector();
    std::optional<Envelope> new_env = std::nullopt;
    if (env0) {
        const Eigen::Matrix4d new_sigma = M * env0->sigma() * M.transpose();
        new_env = Envelope(new_sigma);
    }
    std::optional<Dispersion> new_disp = std::nullopt;
    if (disp0) {
        const Eigen::Vector4d disp_vec = dispersion(cood0, ds);
        new_disp = Dispersion(disp_vec + M * disp0->vector());
    }
    return {Coordinate(new_vector, cood0.charge(), cood0.mass(), cood0.energy()), new_env, new_disp};
}



#if 0
Eigen::Matrix4d Dipole::transfer_matrix(double length, double angle, double k1, double delta) {
    double rho = (length / angle) * (1.0 + delta);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (k1 == 0.0) {
        double phi = angle / (1.0 + delta);
        double cosphi = std::cos(phi), sinphi = std::sin(phi);
        tmat(0,0) = cosphi; tmat(0,1) = rho * sinphi;
        tmat(1,0) = -sinphi / rho; tmat(1,1) = cosphi;
        tmat(2,3) = length;
        return tmat;
    }
    double k = k1 / (1.0 + delta);
    double kx = k + 1.0 / (rho * rho);
    double sqrtkx = std::sqrt(std::abs(kx));
    double psix = sqrtkx * length;
    double ky = -k;
    double sqrtky = std::sqrt(std::abs(ky));
    double psiy = sqrtky * length;
    // handle cases similar to Python implementation
    if (kx < 0.0) {
        double coshx = std::cosh(psix), sinhx = std::sinh(psix);
        double cosy = std::cos(psiy), siny = std::sin(psiy);
        tmat(0,0)=coshx; tmat(0,1)=sinhx/sqrtkx;
        tmat(1,0)=sqrtkx*sinhx; tmat(1,1)=coshx;
        tmat(2,2)=cosy; tmat(2,3)=siny/sqrtky;
        tmat(3,2)=-sqrtky*siny; tmat(3,3)=cosy;
    } else if (ky < 0.0) {
        double cosx = std::cos(psix), sinx = std::sin(psix);
        double coshy = std::cosh(psiy), sinhy = std::sinh(psiy);
        tmat(0,0)=cosx; tmat(0,1)=sinx/sqrtkx;
        tmat(1,0)=-sqrtkx*sinx; tmat(1,1)=cosx;
        tmat(2,2)=coshy; tmat(2,3)=sinhy/sqrtky;
        tmat(3,2)=sqrtky*sinhy; tmat(3,3)=coshy;
    } else {
        double cosx = std::cos(psix), sinx = std::sin(psix);
        double cosy = std::cos(psiy), siny = std::sin(psiy);
        tmat(0,0)=cosx; tmat(0,1)=sinx/sqrtkx;
        tmat(1,0)=-sqrtkx*sinx; tmat(1,1)=cosx;
        tmat(2,2)=cosy; tmat(2,3)=siny/sqrtky;
        tmat(3,2)=-sqrtky*siny; tmat(3,3)=cosy;
    }
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Dipole::transfer_matrix_array(double length, double angle, double k1, double delta, double ds, bool endpoint) {
    double rho = (length / angle) * (1.0 + delta);
    int n_base = static_cast<int>(std::floor(length / ds));
    int n = n_base + static_cast<int>(endpoint) + 1;
    if (length == 0.0) n = 1;
    std::vector<double> s(n);
    if (n==1) s[0]=0.0; else for (int i=0;i<n;++i) s[i] = (static_cast<double>(i) * length) / (n - 1);
    Eigen::Tensor<double,3> tmat(4,4,n);
    tmat.setZero();
    for (int i=0;i<n;++i) for (int j=0;j<4;++j) tmat(j,j,i)=1.0;

    if (k1 == 0.0) {
        for (int idx=0; idx<n; ++idx) {
            double phi = s[idx] / rho;
            double cosphi = std::cos(phi), sinphi = std::sin(phi);
            tmat(0,0,idx) = cosphi; tmat(0,1,idx) = rho * sinphi;
            tmat(1,0,idx) = -sinphi / rho; tmat(1,1,idx) = cosphi;
            tmat(2,3,idx) = s[idx];
        }
        return {tmat, s};
    }

    double k = k1 / (1.0 + delta);
    for (int idx=0; idx<n; ++idx) {
        double kx = k + 1.0 / (rho * rho);
        double sqrtkx = std::sqrt(std::abs(kx));
        double psix = sqrtkx * s[idx];
        double ky = -k;
        double sqrtky = std::sqrt(std::abs(ky));
        double psiy = sqrtky * s[idx];
        if (kx < 0.0) {
            double coshx = std::cosh(psix), sinhx = std::sinh(psix);
            double cosy = std::cos(psiy), siny = std::sin(psiy);
            tmat(0,0,idx)=coshx; tmat(0,1,idx)=sinhx/sqrtkx;
            tmat(1,0,idx)=sqrtkx*sinhx; tmat(1,1,idx)=coshx;
            tmat(2,2,idx)=cosy; tmat(2,3,idx)=siny/sqrtky;
            tmat(3,2,idx)=-sqrtky*siny; tmat(3,3,idx)=cosy;
        } else if (ky < 0.0) {
            double cosx = std::cos(psix), sinx = std::sin(psix);
            double coshy = std::cosh(psiy), sinhy = std::sinh(psiy);
            tmat(0,0,idx)=cosx; tmat(0,1,idx)=sinx/sqrtkx;
            tmat(1,0,idx)=-sqrtkx*sinx; tmat(1,1,idx)=cosx;
            tmat(2,2,idx)=coshy; tmat(2,3,idx)=sinhy/sqrtky;
            tmat(3,2,idx)=sqrtky*sinhy; tmat(3,3,idx)=coshy;
        } else {
            double cosx = std::cos(psix), sinx = std::sin(psix);
            double cosy = std::cos(psiy), siny = std::sin(psiy);
            tmat(0,0,idx)=cosx; tmat(0,1,idx)=sinx/sqrtkx;
            tmat(1,0,idx)=-sqrtkx*sinx; tmat(1,1,idx)=cosx;
            tmat(2,2,idx)=cosy; tmat(2,3,idx)=siny/sqrtky;
            tmat(3,2,idx)=-sqrtky*siny; tmat(3,3,idx)=cosy;
        }
    }
    return {tmat, s};
}

#endif
