/**
 * @file envelope.cpp
 * @brief Implementation of the Envelope class representing an phase space envelope.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// envelope.cpp
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

#include "egret/envelope.hpp"
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <utility>

/**
 * @brief Construct a new egret::Envelope object.
 * @param cov Covariance matrices (4 x 4 x N tensor)
 * @param s_array Longitudinal positions
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::Envelope::Envelope(
    const Eigen::Matrix4d& cov,
    const double s,
    const std::optional<const Eigen::Matrix2d> &T) noexcept(false) :
    cov_(cov), s_(s), T_(T.value_or(Eigen::Matrix2d::Zero())) {
    calc_eigenmode(T);
}

/**
 * @brief Get the adjoint of the given matrix.
 * @param M Input 2x2 matrix.
 * @return Eigen::Matrix2d Adjoint of M.
 */
Eigen::Matrix2d egret::Envelope::adjoint(const Eigen::Matrix2d& M) noexcept {
    Eigen::Matrix2d M_s;
    M_s << M(1,1), -M(0,1), -M(1,0), M(0,0);
    return M_s;
}

/**
 * @brief Calculate the eigenmode of the envelope.
 * @param T Transformation matrix (optional)
 * @throws std::runtime_error if eigenmode calculation fails.
 */
void egret::Envelope::calc_eigenmode(
    const std::optional<const Eigen::Matrix2d> &T) noexcept(false) {
    // If T is provided, use it
    if (T) {
        T_ = *T;
    } else {
        // Calculate T from the covariance matrix
        T_ = estimate_T(cov_);
    }
    // calculate tau, U, V
    const auto T_s = adjoint(T_); // adjoint of T (Matrix2d)
    tau_ = std::sqrt(1.0 - T_.determinant());
    const double chi = 1.0 / (2.0 * tau_ * tau_ - 1.0);
    const double sqrtchi = std::sqrt(chi);
    const auto Sxx = cov_.block<2, 2>(0, 0); // Matrix2d
    const auto Syy = cov_.block<2, 2>(2, 2); // Matrix2d
    U_ = sqrtchi * (tau_ * tau_ * Sxx - T_s * Syy * T_s.transpose());
    V_ = sqrtchi * (tau_ * tau_ * Syy - T_ * Sxx * T_.transpose());
}

/**
 * @brief Get 4 x 4 Transformation matrix for full phase space.
 * @return Eigen::Matrix4d Transformation matrix.
 */
Eigen::Matrix4d egret::Envelope::T_matrix() const noexcept {
    const auto T_s = adjoint(T_); // adjoint of T_ (Matrix2d)
    Eigen::Matrix4d T_full = Eigen::Matrix4d::Identity() * tau_;
    T_full.block<2,2>(0,2) = -T_s;
    T_full.block<2,2>(2,0) = T_;
    return T_full;
}

/**
 * @brief Transfer the envelope by a given transfer matrix.
 * @param M Transfer matrix (4 x 4)
 * @param length Length of the element
 */
void egret::Envelope::transfer(const Eigen::Matrix4d &M, const double length) noexcept {
    // Update covariance matrix
    cov_ = M * cov_ * M.transpose();
    // Update longitudinal position
    s_ += length;
    // Update T_, tau_, U_, V_
    const auto Mxx = M.block<2,2>(0,0); // Matrix2d
    const auto Mxy = M.block<2,2>(0,2); // Matrix2d
    const auto Myx = M.block<2,2>(2,0); // Matrix2d
    const auto Myy = M.block<2,2>(2,2); // Matrix2d
    const auto Mxx_s = adjoint(Mxx); // adjoint of Mxx
    const auto Mxy_s = adjoint(Mxy); // adjoint of Mxy
    const auto T_s = adjoint(T_); // adjoint of T_ (Matrix2d)
    const auto tauMu = tau_ * Mxx - Mxy * T_; // Matrix2d
    const auto tauMv = tau_ * Myy + Myx * T_s; // Matrix2d
    const double tau = std::sqrt(0.5 * (tauMu.determinant() + tauMv.determinant()));
    const auto Mu = tauMu / tau; // Matrix2d
    const auto Mv = tauMv / tau; // Matrix2d
    const auto Mu_s = adjoint(Mu); // adjoint of Mu (Matrix2d)
    const auto Mv_T1 = tau_ * Mxy_s + T_ * Mxx_s; // Matrix2d
    const auto T1Mu = -tau_ * Myx + Myy * T_; // Matrix2d
    T_ = 0.5 * (Mv * Mv_T1 + T1Mu * Mu_s);
    tau_ = tau;
    U_ = Mu * U_ * Mu.transpose();
    V_ = Mv * V_ * Mv.transpose();
}

/**
 * @brief Estimate T matrix from covariance matrix.
 * @param cov Covariance matrix (4 x 4)
 * @return Eigen::Matrix2d Estimated T matrix.
 * @throws std::runtime_error if estimation fails.
 */
Eigen::Matrix2d egret::Envelope::estimate_T(const Eigen::Matrix4d &cov) noexcept(false) {
    const auto Sxx = cov.block<2, 2>(0, 0); // Matrix2d
    const auto Sxy = cov.block<2, 2>(0, 2); // Matrix2d
    const auto Syy = cov.block<2, 2>(2, 2); // Matrix2d
    // Calculate T from the covariance matrix
    Eigen::Matrix4d mat;
    mat << -Sxx(0,0), -Sxx(0,1)-Syy(0,1), 0., Syy(0,0),
            0., -Syy(1,1), -Sxx(0,0), -Sxx(0,1)+Syy(0,1),
            -Sxx(0,1)+Syy(0,1), -Sxx(1,1), -Syy(0,0), 0.,
            Syy(1,1), 0., -Sxx(0,1)-Syy(0,1), -Sxx(1,1);
    const auto vec = Eigen::Map<const Eigen::Vector4d>(Sxy.data()); // Vector4d
    Eigen::Vector4d res;
    try {
        res = mat.colPivHouseholderQr().solve(vec);
    } catch (const std::exception& e) {
        res = Eigen::Vector4d::Zero();
    }
    return Eigen::Map<const Eigen::Matrix2d>(res.data());
}
