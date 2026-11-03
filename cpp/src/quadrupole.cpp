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
#include "egret/drift_util.hpp"
#include <ranges>

namespace eu = egret::util;

/**
 * @brief Calculate the transfer matrix for a quadrupole magnet.
 * @param length Quadrupole length
 * @param k1 Quadrupole strength k1 (1/m^2)
 * @param tilt Rotation angle (radians)
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::transfer_matrix(const double length, const double k1,
    const std::optional<Eigen::Matrix4d> &rmat) noexcept(false) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    if (std::abs(k1) == 0.0) { // drift case
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
 * @param method Integration method (not used here)
 * @return Eigen::Matrix4d Transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::transfer_matrix(
    const std::optional<Coordinate> &cood0,
    const double ds, const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    double k = k1_ / (1.0 + delta);
    std::optional<Eigen::Matrix4d> rmat = std::nullopt;
    if (std::abs(tilt_) != 0.0) {
        rmat = rotation_matrix(tilt_);
    }
    return transfer_matrix(length_, k, rmat);
}

/**
 * @brief Calculate the transfer matrix array through the quadrupole element.
 * @param cood0 Initial coordinate (optional)
 * @param ds Step size (unused)
 * @param endpoint Include endpoint in s_array
 * @param method Integration method (not used here)
 * @return std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd> Tuple of transfer matrix array and s_array
 */
std::tuple<std::vector<Eigen::Matrix4d>, Eigen::ArrayXd>
egret::Quadrupole::transfer_matrix_array(
    const std::optional<Coordinate> &cood0, const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    const double k = k1_ / (1.0 + delta);
    const auto s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    std::optional<Eigen::Matrix4d> rmat = std::nullopt;
    if (std::abs(tilt_) != 0.0) {
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
    Eigen::Matrix2d mf1;
    mf1 << sinpsi, -cospsi/sqrtk, sqrtk*cospsi, sinpsi;
    mf1 *= 0.5 * length * sqrtk;
    Eigen::Matrix2d mf2;
    mf2 << 0., sinpsi/sqrtk, sqrtk*sinpsi, 0.;
    mf2 *= 0.5;
    Eigen::Matrix2d md1;
    md1 << -sinhpsi, -coshpsi/sqrtk, -sqrtk*coshpsi, -sinhpsi;
    md1 *= 0.5 * length * sqrtk;
    Eigen::Matrix2d md2;
    md2 << 0., sinhpsi/sqrtk, -sqrtk*sinhpsi, 0.;
    md2 *= 0.5;
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
        Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
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
 * @param method Integration method (not used here)
 * @return Eigen::Vector4d Additive dispersion vector
 */
Eigen::Vector4d egret::Quadrupole::dispersion(
    const std::optional<Coordinate> &cood0, const double ds,
    const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    const double delta = cood0 ? cood0->delta() : 0.0;
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    const auto steer = steering_offset();
    cood.x(cood.x() - dx_ - steer[0]);
    cood.y(cood.y() - dy_ - steer[1]);
    return dispersion_at(cood.vector(), length_, delta);
}

/**
 * @brief Calculate the additive dispersion function array along the quadrupole.
 * @param cood0 Initial coordinate (optional)
 * @param ds Maximum step size (m)
 * @param endpoint Whether to include the endpoint in the array
 * @param method Integration method (not used here)
 * @return std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd> Tuple of dispersion array and s_array
 */
std::tuple<Eigen::Matrix<double, 4, Eigen::Dynamic>, Eigen::ArrayXd>
egret::Quadrupole::dispersion_array(
    const std::optional<Coordinate> &cood0,
    const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    (void)method; // unused parameter
    Coordinate cood = eu::drift_coordinate(cood0.value_or(Coordinate()), ds_);
    const double delta = cood.delta();
    const auto steer = steering_offset();
    cood.x(cood.x() - dx_ - steer[0]);
    cood.y(cood.y() - dy_ - steer[1]);
    const auto cood0_vec = cood.vector();
    const auto s_array = Element::s_array(ds, endpoint);
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
    for (const size_t i : std::views::iota(size_t{0}, n)) {
        disp_vector_array.col(i) = dispersion_at(cood0_vec, s_array(i), delta);
    }
    return std::make_tuple(disp_vector_array, s_array);
}

/**
 * @brief Steering kick angles at a relative momentum deviation. The kick scales as 1 / (1 + delta).
 * @param delta Relative momentum deviation
 * @return std::array<double, 2> Horizontal and vertical kick angles (rad)
 */
std::array<double, 2> egret::Quadrupole::kick_at(const double delta) const {
    const double scale = 1.0 / (1.0 + delta);
    return {kick_x_ * scale, kick_y_ * scale};
}

/**
 * @brief Transfer matrix of a section of the given length at a relative momentum deviation.
 * @param length Length of the section (m)
 * @param delta Relative momentum deviation
 * @return Eigen::Matrix4d 4x4 transfer matrix
 */
Eigen::Matrix4d egret::Quadrupole::matrix(const double length, const double delta) const noexcept(false) {
    std::optional<Eigen::Matrix4d> rmat = std::nullopt;
    if (std::abs(tilt_) != 0.0) {
        rmat = rotation_matrix(tilt_);
    }
    return transfer_matrix(length, k1_ / (1.0 + delta), rmat);
}

/**
 * @brief Derivative of the transfer matrix with respect to the relative momentum deviation.
 * @param length Length of the section (m)
 * @param delta Relative momentum deviation
 * @return Eigen::Matrix4d dM / d(delta)
 */
Eigen::Matrix4d egret::Quadrupole::matrix_derivative(const double length, const double delta) const noexcept(false) {
    const double k = k1_ / (1.0 + delta);
    Eigen::Matrix4d derivative = Eigen::Matrix4d::Zero();
    for (const int j : std::views::iota(0, 4)) {
        Eigen::Vector4d unit = Eigen::Vector4d::Zero(); // Vector4d
        unit(j) = 1.0;
        derivative.col(j) = dispersion(unit, length, k, tilt_);
    }
    return derivative;
}

/**
 * @brief Transverse offset of the magnetic center that realizes the steering kick [m].
 *
 * The offset is a mechanical displacement, so it is evaluated once at the reference momentum (delta = 0).
 * A particle entering on axis leaves with an angle change of +kick at delta = 0.
 * @return std::array<double, 2> Horizontal and vertical offset (m). Zero for a negligible gradient or no kick.
 * @throws std::invalid_argument if the transfer block is singular and a kick is requested.
 */
std::array<double, 2> egret::Quadrupole::steering_offset() const noexcept(false) {
    if (std::abs(k1_) < IGNORE_K1_THRESHOLD || (kick_x_ == 0.0 && kick_y_ == 0.0) || length_ == 0.0) {
        return {0.0, 0.0};
    }
    const Eigen::Matrix4d M = matrix(length_, 0.0); // Matrix4d
    Eigen::Matrix2d B;
    B << M(1, 0), M(1, 2),
         M(3, 0), M(3, 2);
    if (std::abs(B.determinant()) < IGNORE_K1_THRESHOLD) {
        throw std::invalid_argument("Steering kick cannot be realized: the quadrupole transfer block is singular.");
    }
    const Eigen::Vector2d offset = -(B.inverse() * Eigen::Vector2d(kick_x_, kick_y_)); // Vector2d
    return {offset(0), offset(1)};
}

/**
 * @brief Distributed dipole kick accumulated over the first s meters, for a negligible gradient.
 *
 * The angle changes by kick * s / L and the position by kick * s^2 / (2 L), at the given momentum.
 * A zero-length quadrupole has no kick.
 * @param s Longitudinal position inside the element (m)
 * @param delta Relative momentum deviation
 * @return Eigen::Vector4d Vector [x, x', y, y'] added to the drifted coordinate
 */
Eigen::Vector4d egret::Quadrupole::distributed_kick(const double s, const double delta) const {
    if (length_ == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    const auto [kx, ky] = kick_at(delta);
    return Eigen::Vector4d(0.5 * kx * s * s / length_, kx * s / length_,
        0.5 * ky * s * s / length_, ky * s / length_);
}

/**
 * @brief Additive dispersion of a section of the given length, for a vector relative to the magnetic center.
 *
 * The steering offset is fixed at its reference value, so the derivative with respect to delta is M'(delta)(v - c).
 * For a negligible gradient the distributed kick scales as 1 / (1 + delta), which adds -kick / (1 + delta).
 * @param vector Vector relative to the element offset and steering offset
 * @param length Length of the section (m)
 * @param delta Relative momentum deviation
 * @return Eigen::Vector4d Dispersion vector
 */
Eigen::Vector4d egret::Quadrupole::dispersion_at(const Eigen::Vector4d &vector, const double length,
    const double delta) const noexcept(false) {
    Eigen::Vector4d disp = dispersion(vector, length, k1_ / (1.0 + delta), tilt_); // Vector4d
    if (std::abs(k1_) < IGNORE_K1_THRESHOLD) {
        disp -= distributed_kick(length, delta) / (1.0 + delta);
    }
    return disp;
}

/**
 * @brief Set the steering coil kick angles. A kick that is std::nullopt is left unchanged.
 * @param kick_x Horizontal kick angle (radians), or std::nullopt
 * @param kick_y Vertical kick angle (radians), or std::nullopt
 */
void egret::Quadrupole::set_steering(const std::optional<double> kick_x,
    const std::optional<double> kick_y) noexcept {
    if (kick_x) {
        kick_x_ = *kick_x;
    }
    if (kick_y) {
        kick_y_ = *kick_y;
    }
}

/**
 * @brief Transfer a single coordinate through the quadrupole.
 *
 * The steering kick is realized by the magnetic-center offset (k1 != 0) or by a distributed dipole kick (k1 = 0).
 * Both scale with the momentum deviation as 1 / (1 + delta).
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Longitudinal step (unused; the quadrupole map does not depend on it)
 * @param method Integration method (not used here)
 * @return Transferred coordinate, envelope, and dispersion
 */
std::tuple<egret::Coordinate, std::optional<egret::Envelope>, std::optional<egret::Dispersion>>
egret::Quadrupole::transfer(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds, const IntegrationMethod method) const noexcept(false) {
    (void)ds; // unused parameter
    (void)method; // unused parameter
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    const double delta = cood.delta();
    const auto steer = steering_offset();
    cood.x(cood.x() - dx_ - steer[0]);
    cood.y(cood.y() - dy_ - steer[1]);
    std::optional<Envelope> evlp = evlp0 ? std::optional<Envelope>(eu::drift_envelope(*evlp0, ds_)) : std::nullopt;
    std::optional<Dispersion> disp = disp0 ? std::optional<Dispersion>(eu::drift_dispersion(*disp0, ds_)) : std::nullopt;
    const Eigen::Matrix4d M = matrix(length_, delta); // Matrix4d
    if (evlp) {
        evlp->transfer(M, length_);
    }
    if (disp) {
        const Eigen::Vector4d disp_v_out = M * disp->vector() + dispersion_at(cood.vector(), length_, delta);
        disp->vector(disp_v_out);
        disp->s(disp->s() + length_);
    }
    Eigen::Vector4d vec = M * cood.vector(); // Vector4d
    if (std::abs(k1_) < IGNORE_K1_THRESHOLD) {
        vec += distributed_kick(length_, delta);
    }
    cood.vector(vec);
    cood.s(cood.s() + length_);
    cood.x(cood.x() + dx_ + steer[0]);
    cood.y(cood.y() + dy_ + steer[1]);
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
 * @brief Transfer a coordinate array along the quadrupole.
 * @param cood0 Initial coordinate
 * @param evlp0 Initial envelope (optional)
 * @param disp0 Initial dispersion (optional)
 * @param ds Maximum step size (m)
 * @param endpoint Whether to include the endpoint
 * @param method Integration method (not used here)
 * @return Coordinate, envelope, and dispersion arrays along the quadrupole
 */
std::tuple<egret::CoordinateArray, std::optional<egret::EnvelopeArray>, std::optional<egret::DispersionArray>>
egret::Quadrupole::transfer_array(const Coordinate &cood0, const std::optional<Envelope> &evlp0,
    const std::optional<Dispersion> &disp0, const double ds, const bool endpoint,
    const IntegrationMethod method) const noexcept(false) {
    Coordinate cood = eu::drift_coordinate(cood0, ds_);
    const double delta = cood.delta();
    const auto steer = steering_offset();
    cood.x(cood.x() - dx_ - steer[0]);
    cood.y(cood.y() - dy_ - steer[1]);
    const Eigen::Vector4d relative = cood.vector(); // relative to the magnetic center
    const auto [M_array, s_array] = transfer_matrix_array(cood, ds, endpoint, method); // vector<Matrix4d>, ArrayXd
    const size_t n = s_array.size();
    Eigen::Matrix<double, 4, Eigen::Dynamic> cood_vector_array(4, n);
    for (const size_t i : std::views::iota(size_t{0}, n)) {
        Eigen::Vector4d vec = M_array[i] * relative; // Vector4d
        if (std::abs(k1_) < IGNORE_K1_THRESHOLD) {
            vec += distributed_kick(s_array(i), delta);
        }
        vec(0) += dx_ + steer[0];
        vec(2) += dy_ + steer[1];
        cood_vector_array.col(i) = vec;
    }
    CoordinateArray cood_array(cood_vector_array, s_array + cood.s(),
        Eigen::ArrayXd::Constant(n, cood.z()), Eigen::ArrayXd::Constant(n, cood.delta()));
    std::optional<EnvelopeArray> evlp_array = std::nullopt;
    if (evlp0) {
        const Envelope evlp = eu::drift_envelope(*evlp0, ds_);
        evlp_array = EnvelopeArray::transport(evlp, M_array, s_array);
    }
    std::optional<DispersionArray> disp_array = std::nullopt;
    if (disp0) {
        const Dispersion disp = eu::drift_dispersion(*disp0, ds_);
        Eigen::Matrix<double, 4, Eigen::Dynamic> disp_vector_array(4, n);
        for (const size_t i : std::views::iota(size_t{0}, n)) {
            disp_vector_array.col(i) = M_array[i] * disp.vector() + dispersion_at(relative, s_array(i), delta);
        }
        disp_array = DispersionArray(disp_vector_array, s_array + disp.s());
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
