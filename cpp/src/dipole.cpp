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
#include "egret/drift_util.hpp"
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace eu = egret::util;

/**
 * @brief Construct a new egret::Dipole object.
 * @param name Name of the dipole element
 * @param length Length of the dipole element
 * @param angle Bending angle (radians)
 * @param k1 Quadrupole component strength k1 (1/m^2)
 * @param e1 Entrance edge angle (radians)
 * @param e2 Exit edge angle (radians)
 * @param h1 Entrance pole-face curvature (1/m)
 * @param h2 Exit pole-face curvature (1/m)
 * @param dx Horizontal displacement (m)
 * @param dy Vertical displacement (m)
 * @param ds Longitudinal displacement (m)
 * @param tilt Tilt angle (radians)
 * @param info Additional information string
 * @throws std::invalid_argument if angle is zero.
 */
egret::Dipole::Dipole(const std::string &name, const double length, const double angle,
    const double k1, const double e1, const double e2,
    const double h1, const double h2,
    const double dx, const double dy, const double ds,
    const double tilt, const std::string &info) noexcept(false) :
    Element(name, length, angle, dx, dy, ds, tilt, info),
    k1_(k1), e1_(e1), e2_(e2), h1_(h1), h2_(h2) {
    if (angle == 0.0) {
        throw std::invalid_argument("Angle is zero.");
    }
}

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
 * @brief Calculate the transfer block of x'' + K x = 0 over a section, and its derivative with respect to K.
 * @param strength Focusing strength K (1/m^2); negative values are defocusing
 * @param length Length of the section (m)
 * @return ThickBlock Transfer block [[c, s], [-K s, c]] and its derivative with respect to K
 */
egret::Dipole::ThickBlock egret::Dipole::thick_block(const double strength, const double length) {
    double c = 1.0;
    double s = length;
    double ds_dk = -length * length * length / 6.0; // drift limit; O(K) corrections are negligible
    if (std::abs(strength) >= FOCUSING_IGNORE_THRESHOLD) {
        const double root = std::sqrt(std::abs(strength));
        const double psi = root * length;
        if (strength > 0.0) { // focusing
            c = std::cos(psi);
            s = std::sin(psi) / root;
        } else { // defocusing
            c = std::cosh(psi);
            s = std::sinh(psi) / root;
        }
        ds_dk = (length * c - s) / (2.0 * strength);
    }
    ThickBlock out;
    out.block << c, s, -strength * s, c;
    out.derivative << -0.5 * length * s, ds_dk, -0.5 * (s + length * c), -0.5 * length * s;
    return out;
}

/**
 * @brief Derivative of the transfer matrix of a section with respect to the relative momentum deviation.
 * @param delta Relative momentum deviation
 * @param length Length of the section (m)
 * @return Eigen::Matrix4d dM / d(delta)
 */
Eigen::Matrix4d egret::Dipole::matrix_derivative(const double delta, const double length) const {
    const double k = k1_ / (1.0 + delta);
    const double rho = (length_ / angle_) * (1.0 + delta);
    const double dk_x = -(k + 2.0 / (rho * rho)) / (1.0 + delta);
    const double dk_y = k / (1.0 + delta);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Zero();
    tmat.topLeftCorner<2, 2>() = thick_block(k + 1.0 / (rho * rho), length).derivative * dk_x;
    tmat.bottomRightCorner<2, 2>() = thick_block(-k, length).derivative * dk_y;
    return tmat;
}

/**
 * @brief Dispersion generated by the bending field at a given relative momentum deviation, for a zero orbit.
 * @param delta Relative momentum deviation
 * @param length Length of the section (m)
 * @return Eigen::Vector4d Dispersion [eta_x, eta_x', eta_y, eta_y']
 */
Eigen::Vector4d egret::Dipole::inhomogeneous_dispersion(const double delta, const double length) const {
    const double rho = (length_ / angle_) * (1.0 + delta);
    const double kx = k1_ / (1.0 + delta) + 1.0 / (rho * rho);
    const ThickBlock tb = thick_block(kx, length);
    Eigen::Vector4d disp = Eigen::Vector4d::Zero();
    disp(0) = (1.0 - tb.block(0, 0)) / (kx * rho);
    disp(1) = tb.block(0, 1) / rho;
    return disp;
}

/**
 * @brief Array of the zero-orbit dispersion along the dipole.
 * @param delta Relative momentum deviation
 * @param s_array Longitudinal positions inside the dipole (m)
 * @return Eigen::Matrix<double, 4, Eigen::Dynamic> Dispersion array of shape (4, N)
 */
Eigen::Matrix<double, 4, Eigen::Dynamic> egret::Dipole::inhomogeneous_dispersion_array(
    const double delta, const Eigen::ArrayXd &s_array) const {
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_array(4, n);
    for (const auto i : std::views::iota(0u, n)) {
        disp_array.col(i) = inhomogeneous_dispersion(delta, s_array[i]);
    }
    return disp_array;
}

/**
 * @brief Calculate the additive dispersion function at the end of the dipole.
 *
 * The dispersion is the momentum derivative of the transfer map at fixed input: the zero-orbit part plus M'(delta) v.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size for integration (m) (not used here)
 * @param method Integration method (not used here)
 * @return Eigen::Vector4d
 */
Eigen::Vector4d egret::Dipole::dispersion(const std::optional<Coordinate> &cood0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    return inhomogeneous_dispersion(cood.delta(), length_) + matrix_derivative(cood.delta(), length_) * cood.vector();
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
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    const auto s_array = Element::s_array(ds, endpoint); // ArrayXd
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_array = inhomogeneous_dispersion_array(cood.delta(), s_array);
    const size_t n = s_array.size();
    for (const auto i : std::views::iota(0u, n)) {
        disp_array.col(i) += matrix_derivative(cood.delta(), s_array[i]) * cood.vector();
    }
    return std::make_tuple(disp_array, s_array);
}

/**
 * @brief Get a partial dipole starting from the specified longitudinal position.
 * @param s Start position along the element (m)
 * @return std::shared_ptr<egret::Element> Shared pointer to the generated partial dipole
 * @throw std::out_of_range If s is outside the range [0, length_]
 */
std::shared_ptr<egret::Element> egret::Dipole::partial_element_from_s(const double s) const noexcept(false) {
    if (s < 0. || s > length_) {
        throw std::out_of_range("s is out of range in this element.");
    }
    const double length = length_ - s;
    const double angle = (length_ == 0.0) ? angle_ : angle_ * length / length_;
    return std::make_shared<Dipole>(name() + "_part_from_" + std::to_string(s),
        length, angle, k1_, e1_, e2_, h1_, h2_, dx_, dy_, ds_ + s, tilt_, info_);
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
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    const auto M = transfer_matrix(cood0, ds); // Matrix4d
    const auto disp_add = inhomogeneous_dispersion(cood.delta(), length_); // Vector4d
    cood.vector(M * cood.vector() + disp_add * cood.delta());
    cood.s(cood.s() + length_);
    std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    if (evlp) {
        evlp->transfer(M, length_);
    }
    std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    if (disp) {
        const auto disp_vec = M * disp->vector() + dispersion(cood0, ds); // Vector4d
        disp = Dispersion(disp_vec, disp->s() + length_);
    }
    cood.x(cood.x() + dx_);
    cood.y(cood.y() + dy_);
    cood = eu::drift_coordinate(cood, -ds_);
    if (evlp) {
        *evlp = eu::drift_envelope(*evlp, -ds_);
    }
    if (disp) {
        *disp = eu::drift_dispersion(*disp, -ds_);
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
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    cood.x(cood.x() - dx_);
    cood.y(cood.y() - dy_);
    const auto [M_array, s_array] = transfer_matrix_array(cood0, ds, endpoint); // vector<Matrix4d>, ArrayXd
    const size_t n = M_array.size();
    Eigen::Matrix<double, Eigen::Dynamic, 4> M_combined(n*4, 4);
    for (const auto i : std::views::iota(0u, n)) {
        M_combined.block(i*4, 0, 4, 4) = M_array[i];
    }
    const auto disp_add_mat = inhomogeneous_dispersion_array(cood.delta(), s_array); // Matrix 4 x n
    Eigen::Matrix<double, 4, Eigen::Dynamic> vector_array =
        (M_combined * cood.vector()).reshaped(4, n) + disp_add_mat * cood.delta();
    vector_array.row(0).array() += dx_;
    vector_array.row(2).array() += dy_;
    CoordinateArray cood_array(vector_array, cood.s() + s_array,
        Eigen::ArrayXd::Constant(n, cood.delta()),
        Eigen::ArrayXd::Constant(n, cood.z()));
    const std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    const std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    if (evlp) {
        evlp_array = EnvelopeArray::transport(*evlp, M_array, s_array);
    }
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (disp) {
        const auto [disp_add, _] = dispersion_array(cood0, ds, endpoint); // Matrix, ArrayXd
        const auto disp_vector_array = (M_combined * disp->vector()).reshaped(4, n)
            + disp_add; // Matrix 4 x n
        disp_array = DispersionArray(disp_vector_array, disp->s() + s_array);
    }
    cood_array = eu::drift_coordinate_array(cood_array, -ds_);
    if (evlp_array) {
        *evlp_array = eu::drift_envelope_array(*evlp_array, -ds_);
    }
    if (disp_array) {
        *disp_array = eu::drift_dispersion_array(*disp_array, -ds_);
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
        transfer_array(cood0, evlp0, disp0, ds, true, method);
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
