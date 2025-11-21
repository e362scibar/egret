/**
 * @file dipole.hpp
 * @brief Dipole element class
 * @author Hirokazu Maesaka
 * @date 2025
 */
// dipole.hpp
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

#include "egret/element.hpp"

namespace egret {
    class Dipole;
}
class egret::Dipole : public egret::Element {
protected:
    //! Bending angle (radians)
    double angle_;
    //! Quadrupole component strength k1 (1/m^2)
    double k1_;
    //! Entrance/exit edge angle (radians)
public:







    // simple dipole or combined-function dipole transfer matrix
    //static Eigen::Matrix4d transfer_matrix(double length, double angle, double k1=0.0, double delta=0.0);

    //static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array(double length, double angle, double k1=0.0, double delta=0.0, double ds=0.1, bool endpoint=false);
};
