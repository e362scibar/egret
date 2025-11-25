/**
 * @file octupole.cpp
 * @brief Implementation of the Octupole element class.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// octupole.cpp
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

#include "egret/octupole.hpp"

/**
 * @brief Calculate dipole and quadrupole field strengths at given coordinate.
 * (x'+jy' = - k0 L - k1 L x + j k1 L y)
 * @param cood Particle coordinate
 * @return std::tuple<std::complex<double>, std::complex<double>> Dipole and quadrupole field strengths
 */
std::tuple<std::complex<double>, std::complex<double>>
egret::Octupole::get_k(const Coordinate &cood) const noexcept {
    const double delta = cood.delta();
    const double x = cood.x();
    const double y = cood.y();
    // Octupole contributions
    const double k3 = k3_ / (1.0 + delta);
    const double k0x_oct = k3 * (x*x*x/6.0 - 0.5*x*y*y);
    const double k0y_oct = k3 * (y*y*y/6.0 - 0.5*x*x*y);
    const std::complex<double> k0_oct(k0x_oct, k0y_oct);
    const std::complex<double> k1_oct = k3 * std::complex<double>(0.5 * (x*x - y*y), -x*y);
    // Quadrupole contributions
    const double k1amp = k1_ / (1.0 + delta);
    const std::complex<double> k1_quad = k1amp * std::exp(std::complex<double>(0.0, 2.0*tilt_quad_));
    const std::complex<double> k0_quad = k1_quad * std::conj(std::complex<double>(x, y));
    // Steering dipole contributions
    const double k0x_str = k0x_ / (1.0 + delta);
    const double k0y_str = k0y_ / (1.0 + delta);
    const std::complex<double> k0_str(k0x_str, k0y_str);
    // Total contributions
    const std::complex<double> k0 = k0_oct + k0_quad + k0_str;
    const std::complex<double> k1 = k1_oct + k1_quad;
    return std::make_tuple(k0, k1);
}
