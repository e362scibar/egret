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
#include <ranges>

/**
 * @brief Calculate the transfer matrix for the quadrupole element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Step size (unused)
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::transfer_matrix(
    const std::optional<Coordinate> &cood0,
    const double ds) const noexcept(false) {
    (void)ds; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    double k = k1_ / (1.0 + delta);
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    if (k == 0.0) { // drift case
        M(0,1) = length_;
        M(2,3) = length_;
        return M;
    }
    const double sqrtk = std::sqrt(std::abs(k));
    const double psi = sqrtk * length_;
    const double cospsi = std::cos(psi);
    const double sinpsi = std::sin(psi);
    const double coshpsi = std::cosh(psi);
    const double sinhpsi = std::sinh(psi);
    Eigen::Matrix2d mf, md; // focusing and defocusing matrices
    mf << cospsi, sinpsi / sqrtk, -sqrtk * sinpsi, cospsi;
    md << coshpsi, sinhpsi / sqrtk, sqrtk * sinhpsi, coshpsi;
    if (k < 0.0) { // devofocusing in x
        M.block<2,2>(0,0) = md;
        M.block<2,2>(2,2) = mf;
    } else { // focusing in x
        M.block<2,2>(0,0) = mf;
        M.block<2,2>(2,2) = md;
    }
    if (tilt_ != 0.0) {
        const double ct = std::cos(tilt_);
        const double st = std::sin(tilt_);
        Eigen::Matrix4d rmat = Eigen::Matrix4d::Zero();
        rmat(0,0) = ct;
        rmat(0,2) = st;
        rmat(1,1) = ct;
        rmat(1,3) = st;
        rmat(2,0) = -st;
        rmat(2,2) = ct;
        rmat(3,1) = -st;
        rmat(3,3) = ct;
        M = rmat.transpose() * M * rmat;
    }
    return M;
}

std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Quadrupole::transfer_matrix_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint) const noexcept(false) {
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double k = k1_ / (1.0 + delta);
    const auto s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    std::vector<Eigen::Matrix4d> M_array(n, Eigen::Matrix4d::Identity());
    if (k == 0.0) { // drift case
        for (const size_t i : std::views::iota(0u, n)) {
            const double s = s_array(i);
            M_array[i](0,1) = s;
            M_array[i](2,3) = s;
        }
        return std::make_tuple(M_array, s_array);
    }
}

/**
 * @brief Calculate the additive dispersion function at the end of the quadrupole.
 * @param cood0 Initial coordinate
 * @param ds Maximum step size for integration (m) (unused)
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Quadrupole::dispersion(
    const std::optional<Coordinate> &cood0,
    const double ds) const noexcept(false) {
    (void)ds; // unused parameter
    if (k1_ == 0.0 || !cood0) { // drift case or no initial coordinate
        return Eigen::Vector4d::Zero();
    }
    const double delta = cood0->delta();
    const double k = k1_ / (1.0 + delta);
    const auto vector0 = cood0->vector(); // Vector4d
    const double sqrtk = std::sqrt(std::abs(k));
    const double psi = sqrtk * length_;
    const double cospsi = std::cos(psi);
    const double sinpsi = std::sin(psi);
    const double coshpsi = std::cosh(psi);
    const double sinhpsi = std::sinh(psi);
    const auto mf1 = Eigen::Matrix2d(sinpsi, -cospsi/sqrtk, sqrtk*cospsi, sinpsi)
        * 0.5 * length_ * sqrtk; // Matrix2d
    const auto mf2 = Eigen::Matrix2d(0., sinpsi/sqrtk, sqrtk*sinpsi, 0.) * 0.5; // Matrix2d
    const auto md1 = Eigen::Matrix2d(sinhpsi, -coshpsi/sqrtk, -sqrtk*coshpsi, -sinhpsi)
        * 0.5 * length_ * sqrtk; // Matrix2d
    const auto md2 = Eigen::Matrix2d(0., sinhpsi/sqrtk, -sqrtk*sinhpsi, 0.) * 0.5; // Matrix2d
    Eigen::Vector4d disp;
    if (k < 0.0) { // defocusing in x
        disp.head<2>() = (md1 + md2) * vector0.head<2>();
        disp.tail<2>() = (mf1 + mf2) * vector0.tail<2>();
    } else { // focusing in x
        disp.head<2>() = (mf1 + mf2) * vector0.head<2>();
        disp.tail<2>() = (md1 + md2) * vector0.tail<2>();
    }
    return disp;
}

std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Quadrupole::dispersion_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint) const noexcept(false) {
    const auto s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
    if (k1_ == 0.0 || !cood0) { // drift case or no initial coordinate
        disp_vector_array.setZero();
        return std::make_tuple(disp_vector_array, s_array);
    }
    const double delta = cood0->delta();
    const double k = k1_ / (1.0 + delta);
    const auto vector0 = cood0->vector(); // Vector4d
    const double sqrtk = std::sqrt(std::abs(k));
    const auto psi_array = sqrtk * s_array; // ArrayXd
    const auto sinpsi_array = psi_array.sin(); // ArrayXd
    const auto sinhpsi_array = psi_array.sinh(); // ArrayXd
    const auto s_cospsi_array = s_array * psi_array.cos(); // ArrayXd
    const auto s_sinpsi_array = s_array * sinpsi_array; // ArrayXd
    const auto s_coshpsi_array = s_array * psi_array.cosh(); // ArrayXd
    const auto s_sinhpsi_array = s_array * sinhpsi_array; // ArrayXd
    auto mf1_combined = Eigen::Matrix<double, Eigen::Dynamic, 2>(n*2, 2);
    auto mf2_combined = Eigen::Matrix<double, Eigen::Dynamic, 2>(n*2, 2);
    auto md1_combined = Eigen::Matrix<double, Eigen::Dynamic, 2>(n*2, 2);
    auto md2_combined = Eigen::Matrix<double, Eigen::Dynamic, 2>(n*2, 2);
    auto mf1_00 = mf1_combined.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto mf1_10 = mf1_combined.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto mf1_01 = mf1_combined.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto mf1_11 = mf1_combined.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    mf1_00 = 0.5 * sqrtk * s_sinpsi_array;
    mf1_01 = -0.5 * s_cospsi_array;
    mf1_10 = 0.5 * std::abs(k) * s_cospsi_array;
    mf1_11 = 0.5 * sqrtk * s_sinpsi_array;
    auto mf2_00 = mf2_combined.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto mf2_10 = mf2_combined.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto mf2_01 = mf2_combined.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto mf2_11 = mf2_combined.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    mf2_00.setZero();
    mf2_01 = 0.5 * sinpsi_array / sqrtk;
    mf2_10 = 0.5 * sqrtk * sinpsi_array;
    mf2_11.setZero();
    auto md1_00 = md1_combined.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto md1_10 = md1_combined.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto md1_01 = md1_combined.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto md1_11 = md1_combined.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    md1_00 = -0.5 * sqrtk * s_sinhpsi_array;
    md1_01 = -0.5 * s_coshpsi_array;
    md1_10 = -0.5 * std::abs(k) * s_coshpsi_array;
    md1_11 = -0.5 * sqrtk * s_sinhpsi_array;
    auto md2_00 = md2_combined.col(0)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto md2_10 = md2_combined.col(0)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    auto md2_01 = md2_combined.col(1)(Eigen::seq(0, Eigen::last, 2)); // ArrayXd-like
    auto md2_11 = md2_combined.col(1)(Eigen::seq(1, Eigen::last, 2)); // ArrayXd-like
    md2_00.setZero();
    md2_01 = 0.5 * sinhpsi_array / sqrtk;
    md2_10 = -0.5 * sqrtk * sinhpsi_array;
    md2_11.setZero();
    if (k < 0.0) { // defocusing in x
        disp_vector_array.topRows<2>() = ((md1_combined + md2_combined) * vector0.head<2>()).reshaped(2, n);
        disp_vector_array.bottomRows<2>() = ((mf1_combined + mf2_combined) * vector0.tail<2>()).reshaped(2, n);
    } else { // focusing in x
        disp_vector_array.topRows<2>() = ((mf1_combined + mf2_combined) * vector0.head<2>()).reshaped(2, n);
        disp_vector_array.bottomRows<2>() = ((md1_combined + md2_combined) * vector0.tail<2>()).reshaped(2, n);
    }
    return std::make_tuple(disp_vector_array, s_array);
}


#if 0
namespace egret {

Eigen::Matrix4d Quadrupole::transfer_matrix(double length, double k1, double tilt, double delta) {
    double k = k1 / (1.0 + delta);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (k == 0.0) {
        tmat(0,1) = length;
        tmat(2,3) = length;
        return tmat;
    }
    double sqrtk = std::sqrt(std::abs(k));
    double psi = sqrtk * length;
    double cospsi = std::cos(psi), sinpsi = std::sin(psi);
    double coshpsi = std::cosh(psi), sinhpsi = std::sinh(psi);
    Eigen::Matrix2d mf, md;
    mf << cospsi, sinpsi / sqrtk,
          -sqrtk * sinpsi, cospsi;
    md << coshpsi, sinhpsi / sqrtk,
          sqrtk * sinhpsi, coshpsi;
    if (k < 0.0) {
        tmat.block<2,2>(0,0) = md;
        tmat.block<2,2>(2,2) = mf;
    } else {
        tmat.block<2,2>(0,0) = mf;
        tmat.block<2,2>(2,2) = md;
    }
    if (tilt != 0.0) {
        double ct = std::cos(tilt), st = std::sin(tilt);
        Eigen::Matrix4d rmat = Eigen::Matrix4d::Zero();
        rmat(0,0)=ct; rmat(0,2)=st;
        rmat(1,1)=ct; rmat(1,3)=st;
        rmat(2,0)=-st; rmat(2,2)=ct;
        rmat(3,1)=-st; rmat(3,3)=ct;
        tmat = rmat.transpose() * tmat * rmat;
    }
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Quadrupole::transfer_matrix_array(double length, double k1, double tilt, double delta, double ds, bool endpoint) {
    double k = k1 / (1.0 + delta);
    int n_base = static_cast<int>(std::floor(length / ds));
    int n = n_base + static_cast<int>(endpoint) + 1;
    if (length == 0.0) {
        n = 1;
    }
    std::vector<double> s(n);
    if (n==1) s[0]=0.0;
    else for (int i=0;i<n;++i) s[i] = (static_cast<double>(i) * length) / (n - 1);
    Eigen::Tensor<double,3> tmat(4,4,n);
    // initialize to identity
    tmat.setZero();
    for (int i=0;i<n;++i) for (int j=0;j<4;++j) tmat(j,j,i)=1.0;

    if (k == 0.0) {
        for (int i=0;i<n;++i) {
            tmat(0,1,i) = s[i];
            tmat(2,3,i) = s[i];
        }
        return {tmat, s};
    }

    // compute per-s psi values
    for (int idx=0; idx<n; ++idx) {
        double sqrtk = std::sqrt(std::abs(k));
        double psi = sqrtk * s[idx];
        double cospsi = std::cos(psi), sinpsi = std::sin(psi);
        double coshpsi = std::cosh(psi), sinhpsi = std::sinh(psi);
        Eigen::Matrix2d mf, md;
        mf << cospsi, sinpsi / sqrtk,
              -sqrtk * sinpsi, cospsi;
        md << coshpsi, sinhpsi / sqrtk,
              sqrtk * sinhpsi, coshpsi;
        if (k < 0.0) {
            // place md in 0:2,0:2 and mf in 2:4,2:4
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r,c,idx)=md(r,c);
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r+2,c+2,idx)=mf(r,c);
        } else {
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r,c,idx)=mf(r,c);
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r+2,c+2,idx)=md(r,c);
        }
    }

    if (tilt != 0.0) {
        // apply rotation rmat.T @ tmat @ rmat for each slice
        double ct = std::cos(tilt), st = std::sin(tilt);
        Eigen::Matrix4d rmat = Eigen::Matrix4d::Zero();
        rmat(0,0)=ct; rmat(0,2)=st;
        rmat(1,1)=ct; rmat(1,3)=st;
        rmat(2,0)=-st; rmat(2,2)=ct;
        rmat(3,1)=-st; rmat(3,3)=ct;
        Eigen::Matrix4d tmp;
        for (int idx=0; idx<n; ++idx) {
            // load slice into tmp
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) tmp(i,j) = tmat(i,j,idx);
            tmp = rmat.transpose() * tmp * rmat;
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) tmat(i,j,idx) = tmp(i,j);
        }
    }
    return {tmat, s};
}

Eigen::Vector4d Quadrupole::dispersion(const Eigen::Vector4d &cood0vec, double length, double k1, double delta) {
    Eigen::Vector4d disp = Eigen::Vector4d::Zero();
    double k = k1 / (1.0 + delta);
    if (k == 0.0) return disp;
    double sqrtk = std::sqrt(std::abs(k));
    double psi = sqrtk * length;
    double cospsi = std::cos(psi), sinpsi = std::sin(psi);
    double coshpsi = std::cosh(psi), sinhpsi = std::sinh(psi);
    Eigen::Matrix2d Mf1, Mf2, Md1, Md2;
    Mf1 = (Eigen::Matrix2d() << sinpsi, -cospsi / sqrtk, sqrtk * cospsi, sinpsi).finished();
    Mf1 *= 0.5 * length * sqrtk;
    Mf2 = Eigen::Matrix2d::Zero();
    Mf2(0,1) = sinpsi / sqrtk;
    Mf2(1,0) = sqrtk * sinpsi;
    Mf2 *= 0.5;
    Md1 = (Eigen::Matrix2d() << -sinhpsi, -coshpsi / sqrtk, -sqrtk * coshpsi, -sinhpsi).finished();
    Md1 *= 0.5 * length * sqrtk;
    Md2 = Eigen::Matrix2d::Zero();
    Md2(0,1) = sinhpsi / sqrtk;
    Md2(1,0) = -sqrtk * sinhpsi;
    Md2 *= 0.5;
    if (k < 0.0) {
        Eigen::Vector2d v0 = cood0vec.segment<2>(0);
        Eigen::Vector2d v1 = cood0vec.segment<2>(2);
        Eigen::Vector2d r0 = (Md1 + Md2) * v0;
        Eigen::Vector2d r1 = (Mf1 + Mf2) * v1;
        disp.segment<2>(0) = r0;
        disp.segment<2>(2) = r1;
    } else {
        Eigen::Vector2d v0 = cood0vec.segment<2>(0);
        Eigen::Vector2d v1 = cood0vec.segment<2>(2);
        Eigen::Vector2d r0 = (Mf1 + Mf2) * v0;
        Eigen::Vector2d r1 = (Md1 + Md2) * v1;
        disp.segment<2>(0) = r0;
        disp.segment<2>(2) = r1;
    }
    return disp;
}

} // namespace egret
#endif
