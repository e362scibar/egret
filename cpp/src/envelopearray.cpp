/**
 * @file envelopearray.cpp
 * @brief Implementation of the EnvelopeArray class representing an array of phase space envelopes.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// envelopearray.cpp
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

#include "egret/envelopearray.hpp"
#include <stdexcept>
#include <ranges>
#include <numbers>

/**
 * @brief Construct a new egret::EnvelopeArray object.
 * @param cov_array Array of covariance matrices (std::vector of 4 x 4 Matrices)
 * @param s_array Longitudinal positions
 * @param
 * @throws std::invalid_argument if input sizes are inconsistent.
 * @throws std::invalid_argument if s_array is not non-decreasing.
 */
egret::EnvelopeArray::EnvelopeArray(
    const std::vector<Eigen::Matrix4d> &cov_array,
    const Eigen::ArrayXd &s_array,
    const std::optional<std::vector<Eigen::Matrix2d>> &T_array,
    const std::optional<Eigen::ArrayXd> &psix_array,
    const std::optional<Eigen::ArrayXd> &psiy_array) noexcept(false):
    BaseArray(s_array), cov_array_(cov_array),
    T_array_(T_array.value_or(std::vector<Eigen::Matrix2d>())),
    tau_array_(Eigen::ArrayXd::Ones(s_array.size())), U_array_(), V_array_(),
    psix_array_(psix_array.value_or(Eigen::ArrayXd())),
    psiy_array_(psiy_array.value_or(Eigen::ArrayXd())) {
    // Check consistency of input sizes
    const auto n = size();
    if (cov_array_.size() != n) {
        throw std::invalid_argument("Size of s_array does not match the number of covariance matrices in cov_array");
    }
    const size_t psix_size = psix_array_.size();
    if (psix_size != 0 && psix_size != n) {
        throw std::invalid_argument("Size of s_array does not match the size of psix_array");
    }
    const size_t psiy_size = psiy_array_.size();
    if (psiy_size != 0 && psiy_size != n) {
        throw std::invalid_argument("Size of s_array does not match the size of psiy_array");
    }
    calc_eigenmode(T_array);
}

/**
 * @brief Get the array of beta functions in the x direction.
 * @return Eigen::ArrayXd Array of beta_x values.
 */
Eigen::ArrayXd egret::EnvelopeArray::bx_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd bx_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        bx_array(i) = cov_array_[i](0, 0);
    }
    return bx_array;
}

/**
 * @brief Get the array of alpha functions in the x direction.
 * @return Eigen::ArrayXd Array of alpha_x values.
 */
Eigen::ArrayXd egret::EnvelopeArray::ax_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd ax_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        ax_array(i) = -0.5 * (cov_array_[i](0, 1) + cov_array_[i](1, 0));
    }
    return ax_array;
}

/**
 * @brief Get the array of gamma functions in the x direction.
 * @return Eigen::ArrayXd Array of gamma_x values.
 */
Eigen::ArrayXd egret::EnvelopeArray::gx_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd gx_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        gx_array(i) = cov_array_[i](1, 1);
    }
    return gx_array;
}

/**
 * @brief Get the array of beta functions in the y direction.
 * @return Eigen::ArrayXd Array of beta_y values.
 */
Eigen::ArrayXd egret::EnvelopeArray::by_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd by_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        by_array(i) = cov_array_[i](2, 2);
    }
    return by_array;
}

/**
 * @brief Get the array of alpha functions in the y direction.
 * @return Eigen::ArrayXd Array of alpha_y values.
 */
Eigen::ArrayXd egret::EnvelopeArray::ay_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd ay_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        ay_array(i) = -0.5 * (cov_array_[i](2, 3) + cov_array_[i](3, 2));
    }
    return ay_array;
}

/**
 * @brief Get the array of gamma functions in the y direction.
 * @return Eigen::ArrayXd Array of gamma_y values.
 */
Eigen::ArrayXd egret::EnvelopeArray::gy_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd gy_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        gy_array(i) = cov_array_[i](3, 3);
    }
    return gy_array;
}

/**
 * @brief Get the array of beta functions in the eigenmode U.
 * @return Eigen::ArrayXd Array of beta_u values.
 */
Eigen::ArrayXd egret::EnvelopeArray::bu_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd bu_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        bu_array(i) = U_array_[i](0, 0);
    }
    return bu_array;
}

/**
 * @brief Get the array of alpha functions in the eigenmode U.
 * @return Eigen::ArrayXd Array of alpha_u values.
 */
Eigen::ArrayXd egret::EnvelopeArray::au_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd au_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        au_array(i) = -0.5 * (U_array_[i](0, 1) + U_array_[i](1, 0));
    }
    return au_array;
}

/**
 * @brief Get the array of gamma functions in the eigenmode U.
 * @return Eigen::ArrayXd Array of gamma_u values.
 */
Eigen::ArrayXd egret::EnvelopeArray::gu_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd gu_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        gu_array(i) = U_array_[i](1, 1);
    }
    return gu_array;
}

/**
 * @brief Get the array of beta functions in the eigenmode V.
 * @return Eigen::ArrayXd Array of beta_v values.
 */
Eigen::ArrayXd egret::EnvelopeArray::bv_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd bv_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        bv_array(i) = V_array_[i](0, 0);
    }
    return bv_array;
}

/**
 * @brief Get the array of alpha functions in the eigenmode V.
 * @return Eigen::ArrayXd Array of alpha_v values.
 */
Eigen::ArrayXd egret::EnvelopeArray::av_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd av_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        av_array(i) = -0.5 * (V_array_[i](0, 1) + V_array_[i](1, 0));
    }
    return av_array;
}

/**
 * @brief Get the array of gamma functions in the eigenmode V.
 * @return Eigen::ArrayXd Array of gamma_v values.
 */
Eigen::ArrayXd egret::EnvelopeArray::gv_array() const noexcept(false) {
    const size_t n = size();
    Eigen::ArrayXd gv_array = Eigen::ArrayXd::Zero(n);
    for (const size_t i : std::views::iota(0u, n)) {
        gv_array(i) = V_array_[i](1, 1);
    }
    return gv_array;
}

namespace {
    /**
     * @brief Compute the cumulative integral using the Hermite cubic spline interpolation method.
     * @param x Array of x values (independent variable).
     * @param y Array of y values (dependent variable).
     * @param dy Array of derivatives of y with respect to x.
     * @return Eigen::ArrayXd Cumulative integral values at each x.
     */
    Eigen::ArrayXd cumulative_hermite_integral(
        const Eigen::ArrayXd &x,
        const Eigen::ArrayXd &y,
        const Eigen::ArrayXd &dy) {
        const size_t n = x.size();
        Eigen::ArrayXd integral = Eigen::ArrayXd::Zero(n);
        if (n < 2) {
            return integral;
        }
        for (size_t i = 0; i + 1 < n; ++i) {
            const double dx = x(i + 1) - x(i);
            const double seg = dx * (0.5 * (y(i) + y(i+1)) + (dy(i) - dy(i+1)) * dx / 12.0);
            integral(i+1) = integral(i) + seg;
        }
        return integral;
    }
}

/**
 * @brief Calculate eigenmode parameters for all envelopes.
 * @param T_array Optional array of transformation matrices. If not provided, they will be estimated from covariances.
 */
void egret::EnvelopeArray::calc_eigenmode(const std::optional<std::vector<Eigen::Matrix2d>> &T_array) noexcept(false) {
    const size_t n = size();
    // If T_array is provided, update T_array_
    if (T_array) {
        if (T_array->size() != n) {
            throw std::invalid_argument("Size of T_array does not match the size of s_array.");
        }
        T_array_ = *T_array;
    } else {
        // If T_array is not provided, estimate T from covariances
        T_array_.clear();
        T_array_.reserve(n);
        for (const auto &cov: cov_array_) {
            T_array_.push_back(Envelope::estimate_T(cov));
        }
    }
    // calculate tau, U, V
    U_array_.clear();
    V_array_.clear();
    U_array_.reserve(n);
    V_array_.reserve(n);
    for (const size_t i : std::views::iota(0u, n)) {
        const auto &cov = cov_array_[i]; // Matrix4d
        const auto &T = T_array_[i]; // Matrix2d
        const auto T_s = Envelope::adjoint(T); // adjoint of T (Matrix2d)
        const double tau = std::sqrt(1.0 - T.determinant());
        const double chi = 1.0 / (2.0 * tau * tau - 1.0);
        const double sqrtchi = std::sqrt(chi);
        const auto Sxx = cov.block<2, 2>(0, 0); // Matrix2d
        const auto Syy = cov.block<2, 2>(2, 2); // Matrix2d
        const auto U = sqrtchi * (tau * tau * Sxx - T_s * Syy * T_s.transpose()); // Matrix2d
        const auto V = sqrtchi * (tau * tau * Syy - T * Sxx * T.transpose()); // Matrix2d
        tau_array_(i) = tau;
        U_array_.push_back(U);
        V_array_.push_back(V);
    }
    // calculate psix, psiy if not provided
    if (psix_array_.size() == 0) {
        const Eigen::ArrayXd b_array = bu_array();
        const Eigen::ArrayXd bp_array = -2.0 * au_array();
        psix_array_ = cumulative_hermite_integral(s_array_, 1.0/b_array, -bp_array/(b_array*b_array));
    }
    if (psiy_array_.size() == 0) {
        const Eigen::ArrayXd b_array = bv_array();
        const Eigen::ArrayXd bp_array = -2.0 * av_array();
        psiy_array_ = cumulative_hermite_integral(s_array_, 1.0/b_array, -bp_array/(b_array*b_array));
    }
}

/**
 * @brief Append another EnvelopeArray to this one.
 * @param other The other EnvelopeArray to append.
 */
void egret::EnvelopeArray::append(const EnvelopeArray &other) noexcept(false) {
    BaseArray::append(other);
    cov_array_.insert(cov_array_.end(), other.cov_array_.begin(), other.cov_array_.end());
    T_array_.insert(T_array_.end(), other.T_array_.begin(), other.T_array_.end());
    tau_array_.conservativeResize(tau_array_.size() + other.tau_array_.size());
    tau_array_.tail(other.tau_array_.size()) = other.tau_array_;
    U_array_.insert(U_array_.end(), other.U_array_.begin(), other.U_array_.end());
    V_array_.insert(V_array_.end(), other.V_array_.begin(), other.V_array_.end());
    psix_array_.conservativeResize(psix_array_.size() + other.psix_array_.size());
    psix_array_.tail(other.psix_array_.size()) = other.psix_array_;
    psiy_array_.conservativeResize(psiy_array_.size() + other.psiy_array_.size());
    psiy_array_.tail(other.psiy_array_.size()) = other.psiy_array_;
}

/**
 * @brief Get Envelope from linear interpolation.
 * @param s Longitudinal position to interpolate at.
 * @return egret::Envelope
 * @throws std::out_of_range if s is out of the range of s_array.
 */
egret::Envelope egret::EnvelopeArray::from_s(const double s) const noexcept(false) {
    const auto idx = index_from_s(s);
    const double s0 = s_array_(idx);
    const double s1 = s_array_(idx + 1);
    const double ds = s1 - s0;
    if (s < s0 || s1 < s) {
        std::ostringstream ss;
        ss << "s = " << s << " is out of range of s_array: [" << s0 << ", " << s1 << "]";
        throw std::out_of_range(ss.str());
    }
    if (ds == 0.) {
        // Degenerate case: s0 == s1
        const auto cov = 0.5 * (cov_array_[idx] + cov_array_[idx + 1]); // Matrix4d
        const auto T = 0.5 * (T_array_[idx] + T_array_[idx + 1]); // Matrix2d
        const double psix = 0.5 * (psix_array_(idx) + psix_array_(idx + 1));
        const double psiy = 0.5 * (psiy_array_(idx) + psiy_array_(idx + 1));
        return Envelope(cov, s, T, psix, psiy);
    }
    const double a0 = (s1 - s) / ds;
    const double a1 = (s - s0) / ds;
    const auto cov = a0 * cov_array_[idx] + a1 * cov_array_[idx + 1]; // Matrix4d
    const auto T = a0 * T_array_[idx] + a1 * T_array_[idx + 1]; // Matrix2d
    const double psix = a0 * psix_array_(idx) + a1 * psix_array_(idx + 1);
    const double psiy = a0 * psiy_array_(idx) + a1 * psiy_array_(idx + 1);
    return Envelope(cov, s, T, psix, psiy);
}

/**
 * @brief Get the full transformation matrix T at the specified index.
 * @param index Array index.
 * @return Eigen::Matrix4d Full transformation matrix at the given index.
 * @throws std::out_of_range if the index is out of range.
 */
Eigen::Matrix4d egret::EnvelopeArray::T_matrix(const size_t index) const noexcept(false) {
    check_index(index);
    Eigen::Matrix4d T_full = Eigen::Matrix4d::Identity() * tau_array_(index);
    T_full.block<2,2>(2,0) = T_array_[index];
    T_full.block<2,2>(0,2) = -Envelope::adjoint(T_array_[index]);
    return T_full;
}

/**
 * @brief Get the array of coordinate transformation matrices for eigenmode.
 * @return std::vector<Eigen::Matrix4d> Array of 4 x 4 coordinate transformation matrices for eigenmode.
 */
std::vector<Eigen::Matrix4d> egret::EnvelopeArray::T_matrix_array() const noexcept(false) {
    const size_t n = size();
    std::vector<Eigen::Matrix4d> T_matrices;
    T_matrices.reserve(n);
    for (const size_t i : std::views::iota(0u, n)) {
        T_matrices.push_back(T_matrix(i));
    }
    return T_matrices;
}

#if 0
namespace {
    /**
     * @brief Unwrap phase array.
     * @param phase_array Input phase array.
     * @return Eigen::ArrayXd Unwrapped phase array.
     */
    Eigen::ArrayXd unwrap_phase(const Eigen::ArrayXd &phase_array) {
        const size_t n = phase_array.size();
        if (n < 2) {
            return phase_array;
        }
        Eigen::ArrayXd unwrapped = phase_array;
        double offset = 0.0;
        for (const size_t i : std::views::iota(1u, n)) {
            const double diff = unwrapped(i) - unwrapped(i-1);
            if (diff > std::numbers::pi) {
                offset -= 2.0 * std::numbers::pi;
            } else if (diff < -std::numbers::pi) {
                offset += 2.0 * std::numbers::pi;
            }
            unwrapped(i) += offset;
        }
        return unwrapped;
    }

    /**
     * @brief Shift phase array to be non-negative.
     * @param phase_array Input phase array.
     * @return Eigen::ArrayXd Shifted phase array.
     */
    Eigen::ArrayXd shift_phase_non_negative(const Eigen::ArrayXd &phase_array) {
        const size_t n = phase_array.size();
        if (n == 0) {
            return phase_array;
        }
        Eigen::ArrayXd shifted = phase_array;
        for (const size_t i : std::views::iota(0u, n)) {
            if (shifted(i) < 0.) {
                shifted(i) += 2. * std::numbers::pi;
            }
        }
        return shifted;
    }
}
#endif

/**
 * @brief Transport an EnvelopeArray using a series of transfer matrices.
 * @param evlp0 Initial Envelope.
 * @param M_array Array of transfer matrices.
 * @param s_array Array of longitudinal positions corresponding to each transfer matrix.
 * @return egret::EnvelopeArray Transported EnvelopeArray.
 */
egret::EnvelopeArray egret::EnvelopeArray::transport(
    const Envelope &evlp0,
    const std::vector<Eigen::Matrix4d> &M_array,
    const Eigen::ArrayXd &s_array) noexcept(false) {
    const size_t n = M_array.size();
    if (s_array.size() != static_cast<int>(n)) {
        throw std::invalid_argument("Size of s_array does not match number of transfer matrices in M_array");
    }
    std::vector<Eigen::Matrix4d> cov_array;
    std::vector<Eigen::Matrix2d> T_array;
    std::vector<Eigen::Matrix2d> Mu_array;
    std::vector<Eigen::Matrix2d> Mv_array;
    cov_array.reserve(n);
    T_array.reserve(n);
    Mu_array.reserve(n);
    Mv_array.reserve(n);
    const auto &cov0 = evlp0.cov(); // Matrix4d
    const auto &T0 = evlp0.T(); // Matrix2d
    const auto T0_s = Envelope::adjoint(T0); // Matrix2d
    const double tau0 = evlp0.tau();
    for (const auto &M : M_array) {
        const auto cov = M * cov0 * M.transpose(); // Matrix4d
        const auto Mxx = M.block<2,2>(0,0); // Matrix2d
        const auto Mxy = M.block<2,2>(0,2); // Matrix2d
        const auto Myx = M.block<2,2>(2,0); // Matrix2d
        const auto Myy = M.block<2,2>(2,2); // Matrix2d
        const auto Mxx_s = Envelope::adjoint(Mxx); // Matrix2d
        const auto Mxy_s = Envelope::adjoint(Mxy); // Matrix2d
        const auto tauMu = tau0 * Mxx - Mxy * T0; // Matrix2d
        const auto tauMv = tau0 * Myy + Myx * T0_s; // Matrix2d
        const double tau = std::sqrt(0.5 * (tauMu.determinant() + tauMv.determinant()));
        const auto Mu = tauMu / tau; // Matrix2d
        const auto Mv = tauMv / tau; // Matrix2d
        const auto Mu_s = Envelope::adjoint(Mu); // Matrix2d
        const auto Mv_T1 = tau0 * Mxy_s + T0 * Mxx_s; // Matrix2d
        const auto T1Mu = -tau0 * Myx + Myy * T0; // Matrix2d
        const auto T = 0.5 * (Mv * Mv_T1 + T1Mu * Mu_s); // Matrix2d
        cov_array.push_back(cov);
        T_array.push_back(T);
        Mu_array.push_back(Mu);
        Mv_array.push_back(Mv);
    }
    Eigen::ArrayXd dpsix_array = Eigen::ArrayXd::Zero(n);
    Eigen::ArrayXd dpsiy_array = Eigen::ArrayXd::Zero(n);
    const double bu0 = evlp0.bu();
    const double bv0 = evlp0.bv();
    const double au0 = evlp0.au();
    const double av0 = evlp0.av();
    for (const size_t i : std::views::iota(0u, n)) {
        const double mu11 = Mu_array[i](0,0);
        const double mu12 = Mu_array[i](0,1);
        const double mv11 = Mv_array[i](0,0);
        const double mv12 = Mv_array[i](0,1);
        dpsix_array(i) = std::atan2(mu12, bu0 * mu11 - au0 * mu12);
        dpsiy_array(i) = std::atan2(mv12, bv0 * mv11 - av0 * mv12);
    }
    // Materialize temporaries to avoid complex expression templates
#if 0
    const auto shifted_dpsix = shift_phase_non_negative(dpsix_array); // Eigen::ArrayXd
    const auto unwrapped_dpsix = unwrap_phase(shifted_dpsix); // Eigen::ArrayXd
    const auto psix_array = psix_base + unwrapped_dpsix; // Eigen::ArrayXd
    const auto shifted_dpsiy = shift_phase_non_negative(dpsiy_array); // Eigen::ArrayXd
    const auto unwrapped_dpsiy = unwrap_phase(shifted_dpsiy); // Eigen::ArrayXd
    const auto psiy_array = psiy_base + unwrapped_dpsiy; // Eigen::ArrayXd
#else
    const auto psix_base = Eigen::ArrayXd::Constant(n, evlp0.psix()); // Eigen::ArrayXd
    const auto psix_array = psix_base + dpsix_array; // Eigen::ArrayXd
    const auto psiy_base = Eigen::ArrayXd::Constant(n, evlp0.psiy()); // Eigen::ArrayXd
    const auto psiy_array = psiy_base + dpsiy_array; // Eigen::ArrayXd
#endif
    return EnvelopeArray(cov_array, s_array + evlp0.s(), T_array, psix_array, psiy_array);
}
