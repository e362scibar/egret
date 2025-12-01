/**
 * @file envelopearray.hpp
 * @brief Class representing an array of beam envelopes in phase space
 * @author Hirokazu Maesaka
 * @date 2025
 */
// envelopearray.hpp
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
#include <vector>
#include "egret/envelope.hpp"
#include "egret/basearray.hpp"

namespace egret {
    class EnvelopeArray;
}

/**
 * @brief Class representing an array of beam envelopes in phase space.
 */
class egret::EnvelopeArray: public BaseArray {
protected:
    //! std::vector of 4 x 4 covariance matrices
    std::vector<Eigen::Matrix4d> cov_array_;
    //! std::vector of 2x2 coordinate transformation matrix for eigenmode
    std::vector<Eigen::Matrix2d> T_array_;
    //! Factor for eigenmode normalization
    Eigen::ArrayXd tau_array_;
    //! std::vector of covariance matrices for the envelope of the eigenmode U
    std::vector<Eigen::Matrix2d> U_array_;
    //! std::vector of covariance matrices for the envelope of the eigenmode V
    std::vector<Eigen::Matrix2d> V_array_;

public:
    // Constructor
    EnvelopeArray(
        const std::vector<Eigen::Matrix4d>& cov_array,
        const Eigen::ArrayXd& s_array,
        const std::optional<std::vector<Eigen::Matrix2d>>& T_array = std::nullopt)
        noexcept(false);
    /**
    * @brief Destroy the EnvelopeArray object.
    */
    virtual ~EnvelopeArray() = default;

    /**
     * @brief Check if the given index is within the valid range.
     * @param index Array index to check.
     * @throws std::out_of_range if the index is out of range.
     */
    void check_index(const size_t index) const noexcept(false) {
        if (index >= size()) {
            throw std::out_of_range("Index out of range in EnvelopeArray.");
        }
    }

    /**
     * @brief Get the covariance matrix at the specified index.
     * @param index Array index.
     * @return const Eigen::Matrix4d& Covariance matrix at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    const Eigen::Matrix4d& cov(const size_t index) const noexcept(false) {
        check_index(index);
        return cov_array_[index];
    }

    /**
     * @brief Get the coordinate transformation matrix for eigenmode at the specified index.
     * @param index Array index.
     * @return const Eigen::Matrix2d& Transformation matrix at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    const Eigen::Matrix2d& T(const size_t index) const noexcept(false) {
        check_index(index);
        return T_array_[index];
    }

    /**
     * @brief Get the factor for eigenmode normalization at the specified index.
     * @param index Array index.
     * @return double Normalization factor at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double tau(const size_t index) const noexcept(false) {
        check_index(index);
        return tau_array_(index);
    }

    /**
     * @brief Get the U matrix for eigenmode parameters at the specified index.
     * @param index Array index.
     * @return const Eigen::Matrix2d& U matrix at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    const Eigen::Matrix2d& U(const size_t index) const noexcept(false) {
        check_index(index);
        return U_array_[index];
    }

    /**
     * @brief Get the V matrix for eigenmode parameters at the specified index.
     * @param index Array index.
     * @return const Eigen::Matrix2d& V matrix at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    const Eigen::Matrix2d& V(const size_t index) const noexcept(false) {
        check_index(index);
        return V_array_[index];
    }

    /**
     * @brief Get the beta function in the x direction at the specified index.
     * @param index Array index.
     * @return double beta_x at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double bx(const size_t index) const noexcept(false) {
        check_index(index);
        return cov_array_[index](0,0);
    }

    /**
     * @brief Get the alpha function in the x direction at the specified index.
     * @param index Array index.
     * @return double alpha_x at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double ax(const size_t index) const noexcept(false) {
        check_index(index);
        return -0.5 * (cov_array_[index](0,1) + cov_array_[index](1,0));
    }

    /**
     * @brief Get the gamma function in the x direction at the specified index.
     * @param index Array index.
     * @return double gamma_x at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double gx(const size_t index) const noexcept(false) {
        check_index(index);
        return cov_array_[index](1,1);
    }

    /**
     * @brief Get the beta function in the y direction at the specified index.
     * @param index Array index.
     * @return double beta_y at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double by(const size_t index) const noexcept(false) {
        check_index(index);
        return cov_array_[index](2,2);
    }

    /**
     * @brief Get the alpha function in the y direction at the specified index.
     * @param index Array index.
     * @return double alpha_y at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double ay(const size_t index) const noexcept(false) {
        check_index(index);
        return -0.5 * (cov_array_[index](2,3) + cov_array_[index](3,2));
    }

    /**
     * @brief Get the gamma function in the y direction at the specified index.
     * @param index Array index.
     * @return double gamma_y at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double gy(const size_t index) const noexcept(false) {
        check_index(index);
        return cov_array_[index](3,3);
    }

    /**
     * @brief Get the beta function in the eigenmode U at the specified index.
     * @param index Array index.
     * @return double beta_u at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double bu(const size_t index) const noexcept(false) {
        check_index(index);
        return U_array_[index](0,0);
    }

    /**
     * @brief Get the alpha function in the eigenmode U at the specified index.
     * @param index Array index.
     * @return double alpha_u at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double au(const size_t index) const noexcept(false) {
        check_index(index);
        return -0.5 * (U_array_[index](0,1) + U_array_[index](1,0));
    }

    /**
     * @brief Get the gamma function in the eigenmode U at the specified index.
     * @param index Array index.
     * @return double gamma_u at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double gu(const size_t index) const noexcept(false) {
        check_index(index);
        return U_array_[index](1,1);
    }

    /**
     * @brief Get the beta function in the eigenmode V at the specified index.
     * @param index Array index.
     * @return double beta_v at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double bv(const size_t index) const noexcept(false) {
        check_index(index);
        return V_array_[index](0,0);
    }

    /**
     * @brief Get the alpha function in the eigenmode V at the specified index.
     * @param index Array index.
     * @return double alpha_v at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double av(const size_t index) const noexcept(false) {
        check_index(index);
        return -0.5 * (V_array_[index](0,1) + V_array_[index](1,0));
    }

    /**
     * @brief Get the gamma function in the eigenmode V at the specified index.
     * @param index Array index.
     * @return double gamma_v at the given index.
     * @throws std::out_of_range if the index is out of range.
     */
    double gv(const size_t index) const noexcept(false) {
        check_index(index);
        return V_array_[index](1,1);
    }

    /**
     * @brief Get the array of covariance matrices.
     * @return const std::vector<Eigen::Matrix4d>& Array of covariance matrices.
     */
    const std::vector<Eigen::Matrix4d>& cov_array() const { return cov_array_; }
    /**
     * @brief Get the array of coordinate transformation matrices for eigenmode.
     * @return const std::vector<Eigen::Matrix2d>& Array of transformation matrices.
     */
    const std::vector<Eigen::Matrix2d>& T_array() const { return T_array_; }
    /**
     * @brief Get the array of normalization factors.
     * @return Eigen::ArrayXd Array of normalization factors.
     */
    const Eigen::ArrayXd& tau_array() const { return tau_array_; }
    /**
     * @brief Get the array of U matrices for eigenmode parameters.
     * @return const std::vector<Eigen::Matrix2d>& Array of U matrices.
     */
    const std::vector<Eigen::Matrix2d>& U_array() const { return U_array_; }
    /**
     * @brief Get the array of V matrices for eigenmode parameters.
     * @return const std::vector<Eigen::Matrix2d>& Array of V matrices.
     */
    const std::vector<Eigen::Matrix2d>& V_array() const { return V_array_; }

    //! Get the array of beta functions in the x direction.
    Eigen::ArrayXd bx_array() const noexcept(false);
    //! Get the array of alpha functions in the x direction.
    Eigen::ArrayXd ax_array() const noexcept(false);
    //! Get the array of gamma functions in the x direction.
    Eigen::ArrayXd gx_array() const noexcept(false);
    //! Get the array of beta functions in the y direction.
    Eigen::ArrayXd by_array() const noexcept(false);
    //! Get the array of alpha functions in the y direction.
    Eigen::ArrayXd ay_array() const noexcept(false);
    //! Get the array of gamma functions in the y direction.
    Eigen::ArrayXd gy_array() const noexcept(false);
    //! Get the array of beta functions in the eigenmode U.
    Eigen::ArrayXd bu_array() const noexcept(false);
    //! Get the array of alpha functions in the eigenmode U.
    Eigen::ArrayXd au_array() const noexcept(false);
    //! Get the array of gamma functions in the eigenmode U.
    Eigen::ArrayXd gu_array() const noexcept(false);
    //! Get the array of beta functions in the eigenmode V.
    Eigen::ArrayXd bv_array() const noexcept(false);
    //! Get the array of alpha functions in the eigenmode V.
    Eigen::ArrayXd av_array() const noexcept(false);
    //! Get the array of gamma functions in the eigenmode V.
    Eigen::ArrayXd gv_array() const noexcept(false);

    // Efficient append (reserve + copy)
    void append(const EnvelopeArray &other);

    // linear interpolation like Python's from_s
    Envelope from_s(double sval) const noexcept(false);

    // Get transformation matrix for full phase space at the specified index
    Eigen::Matrix4d T_matrix(size_t index) const noexcept(false);

    // Transport an initial envelope through a series of transfer matrices
    static EnvelopeArray transport(const Envelope &evlp0, const std::vector<Eigen::Matrix4d> &M_array, const Eigen::ArrayXd &s_array) noexcept(false);
};
