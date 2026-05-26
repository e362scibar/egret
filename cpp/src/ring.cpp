/**
 * @file ring.cpp
 * @brief Ring element class implementation
 * @author Hirokazu Maesaka
 * @date 2025
 */
// ring.cpp
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

#include "egret/ring.hpp"
#include "egret/lattice.hpp"
#include <iostream>
#include <ranges>
#include <gsl/gsl_multimin.h>

//! Tolerance for finding initial coordinates of the closed orbit
double egret::Ring::tol_cod = 1.0e-12;
//! Maximum iterations for finding closed orbit
size_t egret::Ring::max_iter_cod = 1000;

/**
 * @brief Construct a new egret::Ring object.
 * @param name Ring name
 * @param elements Vector of child elements
 * @param energy Beam energy
 * @param info Additional info string
 */
egret::Ring::Ring(const std::string &name,
    const std::vector<std::shared_ptr<Element>> &elements,
    const double energy, const std::string &info) noexcept(false) :
    Element(name, Lattice::length(elements), Lattice::angle(elements), 0.0, 0.0, 0.0, 0.0, info),
    energy_(energy), tune_x_(0.0), tune_y_(0.0), cood0_(), evlp0_(), disp0_() {
    elements_ = std::vector<std::shared_ptr<Element>>(); // std::optional<std::vector<std::shared_ptr<Element>>>
    for (const auto &elem: elements) {
        // Deep-copy each element to preserve dynamic type
        elements_->push_back(elem->clone());
    }
    set_indices();
}

/**
 * @brief Clone the Ring object.
 * @return std::shared_ptr<egret::Element> Shared pointer to the cloned Ring object
 */
std::shared_ptr<egret::Element> egret::Ring::clone() const noexcept(false) {
    auto newring = std::make_shared<Ring>(name_, *elements_, energy_, info_);
    newring->cood0_ = cood0_;
    newring->evlp0_ = evlp0_;
    newring->disp0_ = disp0_;
    newring->emittance_x_ = emittance_x_;
    newring->emittance_y_ = emittance_y_;
    newring->Jx_ = Jx_;
    newring->Jy_ = Jy_;
    newring->Jz_ = Jz_;
    newring->tune_x_ = tune_x_;
    newring->tune_y_ = tune_y_;
    newring->set_indices();
    return newring;
}

/**
 * @brief Update the ring parameters (tunes, etc.)
 * @param delta Relative momentum deviation
 * @param ds Longitudinal step size
 * @param method Integration method
 */
void egret::Ring::update(const double delta, const double ds,
    const IntegrationMethod method) noexcept(false) {
    // find closed orbit
    try {
        const Coordinate cood_guess(Eigen::Vector4d::Zero(), 0.0, 0.0, delta);
        cood0_ = find_initial_coordinate_of_closed_orbit(cood_guess, ds, method);
    } catch (const std::runtime_error &e) {
        std::cout << "Error in finding closed orbit: " << e.what() << std::endl;
        std::cout << "Using previous closed orbit guess." << std::endl;
        cood0_ = Coordinate(Eigen::Vector4d::Zero(), 0.0, 0.0, delta);
    }
    // One-turn transfer matrix
    const auto M = transfer_matrix(cood0_, ds, method); // Matrix4d
    // Initial dispersion
    const auto disp = dispersion(cood0_, ds, method); // Vector4d
    const auto disp0 = (Eigen::Matrix4d::Identity() - M).inverse() * disp;
    disp0_ = Dispersion(disp0, 0.0);
    // Initial betatron function
    const auto Mxx = M.block<2,2>(0,0); // Matrix2d
    const auto Mxy = M.block<2,2>(0,2); // Matrix2d
    const auto Myx = M.block<2,2>(2,0); // Matrix2d
    const auto Myy = M.block<2,2>(2,2); // Matrix2d
    const auto Mxy_s = Envelope::adjoint(Mxy); // Matrix2d
    const double diff_trace_M = Mxx.trace() - Myy.trace();
    const double chi = 1.0 + 4.0 * (Myx + Mxy_s).determinant() / (diff_trace_M * diff_trace_M);
    const double sqrtchi = std::sqrt(std::abs(chi));
    const double tau = std::sqrt(0.5 * (1.0 + 1.0 / sqrtchi));
    const auto T = -(Myx + Mxy_s) / (sqrtchi * tau * diff_trace_M); // Matrix2d
    const auto T_s = Envelope::adjoint(T); // Matrix2d
    const auto U = sqrtchi * (tau*tau * Mxx - T_s * Myy * T); // Matrix2d
    const auto V = sqrtchi * (tau*tau * Myy - T * Mxx * T_s); // Matrix2d
    const double cos_u = 0.5 * U.trace();
    const double sin_u = (std::signbit(U(0,1)-U(1,0)) ? -1.0 : 1.0) *
        std::sqrt((U - cos_u * Eigen::Matrix2d::Identity()).determinant());
    const double cos_v = 0.5 * V.trace();
    const double sin_v = (std::signbit(V(0,1)-V(1,0)) ? -1.0 : 1.0) *
        std::sqrt((V - cos_v * Eigen::Matrix2d::Identity()).determinant());
    const double mu_u = std::atan2(sin_u, cos_u);
    const double mu_v = std::atan2(sin_v, cos_v);
    const double beta_u = U(0,1) / sin_u;
    const double alpha_u = (U(0,0) - U(1,1)) * 0.5 / sin_u;
    const double gamma_u = -U(1,0) / sin_u;
    const double beta_v = V(0,1) / sin_v;
    const double alpha_v = (V(0,0) - V(1,1)) * 0.5 / sin_v;
    const double gamma_v = -V(1,0) / sin_v;
    Eigen::Matrix4d Tmat_inv;
    Tmat_inv.block<2,2>(0,0) = tau * Eigen::Matrix2d::Identity();
    Tmat_inv.block<2,2>(0,2) = T_s;
    Tmat_inv.block<2,2>(2,0) = -T;
    Tmat_inv.block<2,2>(2,2) = tau * Eigen::Matrix2d::Identity();
    Eigen::Matrix2d Su;
    Su << beta_u, -alpha_u, -alpha_u, gamma_u;
    Eigen::Matrix2d Sv;
    Sv << beta_v, -alpha_v, -alpha_v, gamma_v;
    Eigen::Matrix4d SSuv = Eigen::Matrix4d::Zero();
    SSuv.block<2,2>(0,0) = Su;
    SSuv.block<2,2>(2,2) = Sv;
    const Eigen::Matrix4d SSxy = Tmat_inv * SSuv * Tmat_inv.transpose();
    evlp0_ = Envelope(SSxy, 0.0, T);
    tune_x_ = mu_u / (2.0 * M_PI);
    tune_y_ = mu_v / (2.0 * M_PI);
    if (tune_x_ < 0.0) {
        tune_x_ += 1.0;
    }
    if (tune_y_ < 0.0) {
        tune_y_ += 1.0;
    }
    std::tie(I2_, I4_, I5u_, I5v_, I4u_, I4v_) = radiation_integrals(cood0_, evlp0_, disp0_, ds, method);
    const double lgamma = energy_ / m_e_eV;
    emittance_x_ = C_q * lgamma * lgamma * I5u_ / (I2_ - I4u_);
    emittance_y_ = C_q * lgamma * lgamma * I5v_ / (I2_ - I4v_);
    Jx_ = 1.0 - I4u_ / I2_;
    Jy_ = 1.0 - I4v_ / I2_;
    Jz_ = 2.0 + I4_ / I2_;
}

namespace {
    // Dimension of transverse phase space
    constexpr size_t DIM = 4;
    // Initial step size for GSL minimizer
    constexpr double INITIAL_STEP_SIZE = 1.0e-4;
    // flag to indicate whether the function evaluation failed
    bool func_eval_failed = false;
    // Objective function for GSL minimizer to find closed orbit
    double eval_func_cod(const gsl_vector *v, void *params) {
        using IntegrationMethod = egret::Element::IntegrationMethod;
        Eigen::Vector4d x;
        for (size_t i = 0; i < ::DIM; ++i) {
            x(i) = gsl_vector_get(v, i);
        }
        const std::tuple<const egret::Ring*, double, double, IntegrationMethod> *eval_params =
            static_cast<std::tuple<const egret::Ring*, double, double, IntegrationMethod>*>(params);
        const egret::Ring *ring = std::get<0>(*eval_params);
        const double delta = std::get<1>(*eval_params);
        const double ds = std::get<2>(*eval_params);
        const egret::Element::IntegrationMethod method = std::get<3>(*eval_params);
        const egret::Coordinate cood0(x, 0.0, 0.0, delta);
        const auto cood1 = std::get<0>(ring->transfer(cood0, std::nullopt, std::nullopt, ds, method)); // Coordinate after one turn
        const double norm = (cood1.vector() - x).norm();
        if (std::isnan(norm) || std::isinf(norm)) {
            func_eval_failed = true;
            return 1.0e20; // large value to indicate failure
        }
        return norm;
    }
}

/**
 * @brief Find the initial coordinate of the closed orbit
 * @param cood_guess Initial guess coordinate
 * @return egret::Coordinate Initial coordinate of the closed orbit
 * @throws std::runtime_error if the minimization fails to converge
 */
egret::Coordinate egret::Ring::find_initial_coordinate_of_closed_orbit(
    const Coordinate &cood_guess, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    // Set up GSL minimizer
    std::tuple<const Ring*, double, double, IntegrationMethod> params
        = std::make_tuple(this, cood_guess.delta(), ds, method);
    ::gsl_multimin_function f;
    f.n = ::DIM;
    f.f = &eval_func_cod;
    f.params = &params;
    func_eval_failed = false;
    // Initial guess
    gsl_vector *x = gsl_vector_alloc(::DIM);
    for (size_t i = 0; i < ::DIM; ++i) {
        gsl_vector_set(x, i, cood_guess.vector()(i));
    }
    // Step sizes
    gsl_vector *step_size = gsl_vector_alloc(::DIM);
    gsl_vector_set_all(step_size, ::INITIAL_STEP_SIZE);
    // Minimizer (Nelder-Mead Simplex)
    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, ::DIM);
    gsl_multimin_fminimizer_set(s, &f, x, step_size);
    // Minimization loop
    size_t iter = 0;
    int status;
    do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);
        if (status) {
            break;
        }
        if (func_eval_failed) {
            gsl_multimin_fminimizer_free(s);
            gsl_vector_free(x);
            gsl_vector_free(step_size);
            throw std::runtime_error("Function evaluation failed during GSL minimization.");
        }
        double size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, tol_cod);
        if (status == GSL_SUCCESS) {
            break;
        }
    } while (iter < max_iter_cod);
    // Check result
    if (status != GSL_SUCCESS) {
        gsl_multimin_fminimizer_free(s);
        gsl_vector_free(x);
        gsl_vector_free(step_size);
        throw std::runtime_error("GSL minimization failed to converge.");
    }
    // Construct Coordinate from result
    Eigen::Vector4d result_vector;
    for (size_t i = 0; i < ::DIM; ++i) {
        result_vector(i) = gsl_vector_get(s->x, i);
    }
    gsl_multimin_fminimizer_free(s);
    gsl_vector_free(x);
    gsl_vector_free(step_size);
    return Coordinate(result_vector, cood_guess.s(), cood_guess.z(), cood_guess.delta());
}
