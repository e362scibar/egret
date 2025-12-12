/**
 * @brief Class representing a particle coordinate in phase space
 */
// envelope.hpp
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

#pragma once

#include <Eigen/Dense>
#include <optional>

namespace egret {
    class Envelope;
}

class egret::Envelope {
protected:
    //! 4x4 covariance matrix of the beam envelope
    Eigen::Matrix4d cov_;
    //! Longitudinal position s
    double s_;
    //! 2x2 coordinate transformation matrix for eigenmode
    Eigen::Matrix2d T_;
    //! Factor for eigenmode normalization
    double tau_{1.0};
    //! Covariance matrix for the envelope of the eigenmode U
    Eigen::Matrix2d U_{Eigen::Matrix2d::Identity()};
    //! Covariance matrix for the envelope of the eigenmode V
    Eigen::Matrix2d V_{Eigen::Matrix2d::Identity()};
    //! Horizontal betatron phase (Eigenmode U)
    double psix_{0.0};
    //! Vertical betatron phase (Eigenmode V)
    double psiy_{0.0};

public:
    // Constructor
    Envelope(
        const Eigen::Matrix4d &cov = Eigen::Matrix4d::Identity(),
        double s = 0.,
        const std::optional<const Eigen::Matrix2d> &T = std::nullopt,
        const std::optional<double> &psix = std::nullopt,
        const std::optional<double> &psiy = std::nullopt)
        noexcept(false);
    /**
     * @brief Destroy the Envelope object.
     */
    virtual ~Envelope() = default;

    /**
     * @brief Get the covariance matrix of the envelope.
     * @return const Eigen::Matrix4d& Covariance matrix.
     */
    const Eigen::Matrix4d& cov() const { return cov_; }
    /**
     * @brief Get the longitudinal position of the envelope.
     * @return double Longitudinal position.
     */
    double s() const { return s_; }
    /**
     * @brief Get the coordinate transformation matrix for eigenmode.
     * @return const Eigen::Matrix2d& Transformation matrix.
     */
    const Eigen::Matrix2d& T() const { return T_; }
    /**
     * @brief Get the factor for eigenmode normalization.
     * @return double Normalization factor.
     */
    double tau() const { return tau_; }
    /**
     * @brief Get the U matrix for eigenmode parameters.
     * @return const Eigen::Matrix2d& U matrix.
     */
    const Eigen::Matrix2d& U() const { return U_; }
    /**
     * @brief Get the V matrix for eigenmode parameters.
     * @return const Eigen::Matrix2d& V matrix.
     */
    const Eigen::Matrix2d& V() const { return V_; }

    /**
     * @brief Get the beta function in the x direction.
     * @return double beta_x
     */
    double bx() const { return cov_(0,0); }
    /**
     * @brief Get the alpha function in the x direction.
     * @return double alpha_x
     */
    double ax() const { return -0.5 * (cov_(0,1) + cov_(1,0)); }
    /**
     * @brief Get the gamma function in the x direction.
     * @return double gamma_x
     */
    double gx() const { return cov_(1,1); }
    /**
     * @brief Get the beta function in the y direction.
     * @return double beta_y
     */
    double by() const { return cov_(2,2); }
    /**
     * @brief Get the alpha function in the y direction.
     * @return double alpha_y
     */
    double ay() const { return -0.5 * (cov_(2,3) + cov_(3,2)); }
    /**
     * @brief Get the gamma function in the y direction.
     * @return double gamma_y
     */
    double gy() const { return cov_(3,3); }
    /**
     * @brief Get the beta function in the eigenmode U.
     * @return double beta_u
     */
    double bu() const { return U_(0,0); }
    /**
     * @brief Get the alpha function in the eigenmode U.
     * @return double alpha_u
     */
    double au() const { return -0.5 * (U_(0,1) + U_(1,0)); }
    /**
     * @brief Get the gamma function in the eigenmode U.
     * @return double gamma_u
     */
    double gu() const { return U_(1,1); }
    /**
     * @brief Get the beta function in the eigenmode V.
     * @return double beta_v
     */
    double bv() const { return V_(0,0); }
    /**
     * @brief Get the alpha function in the eigenmode V.
     * @return double alpha_v
     */
    double av() const { return -0.5 * (V_(0,1) + V_(1,0)); }
    /**
     * @brief Get the gamma function in the eigenmode V.
     * @return double gamma_v
     */
    double gv() const { return V_(1,1); }
    /**
     * @brief Get the horizontal betatron phase. (Eigenmode U)
     */
    double psix() const { return psix_; }
    /**
     * @brief Get the vertical betatron phase. (Eigenmode V)
     */
    double psiy() const { return psiy_; }

    // Helper to get the adjoint
    static Eigen::Matrix2d adjoint(const Eigen::Matrix2d& M) noexcept;

    // Calculate eigenmode parameters
    void calc_eigenmode(
        const std::optional<const Eigen::Matrix2d> &T = std::nullopt)
        noexcept(false);

    // Estimate T matrix from covariance matrix
    static Eigen::Matrix2d estimate_T(const Eigen::Matrix4d &cov) noexcept(false);

    // Get 4 x 4 Transformation matrix for full phase space
    Eigen::Matrix4d T_matrix() const noexcept;

    // Transfer the envelope by a given transfer matrix.
    void transfer(const Eigen::Matrix4d &M, double length) noexcept;
};
