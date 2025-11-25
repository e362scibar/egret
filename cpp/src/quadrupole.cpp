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
Eigen::Matrix4d egret::Quadrupole::transfer_matrix( const double length, const double k1,
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
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::transfer_matrix(
    const std::optional<Coordinate> &cood0,
    const double ds) const noexcept(false) {
    (void)ds; // unused parameter
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
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Tuple of transfer matrix array and s_array
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Quadrupole::transfer_matrix_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint) const noexcept(false) {
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
    const auto mf1 = Eigen::Matrix2d(sinpsi, -cospsi/sqrtk, sqrtk*cospsi, sinpsi)
        * 0.5 * length * sqrtk; // Matrix2d
    const auto mf2 = Eigen::Matrix2d(0., sinpsi/sqrtk, sqrtk*sinpsi, 0.) * 0.5; // Matrix2d
    const auto md1 = Eigen::Matrix2d(sinhpsi, -coshpsi/sqrtk, -sqrtk*coshpsi, -sinhpsi)
        * 0.5 * length * sqrtk; // Matrix2d
    const auto md2 = Eigen::Matrix2d(0., sinhpsi/sqrtk, -sqrtk*sinhpsi, 0.) * 0.5; // Matrix2d
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
        auto M = Eigen::Matrix4d::Zero(); // Matrix4d
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
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Quadrupole::dispersion(
    const std::optional<Coordinate> &cood0,
    const double ds) const noexcept(false) {
    (void)ds; // unused parameter
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
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Tuple of dispersion array and s_array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Quadrupole::dispersion_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint) const noexcept(false) {
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
