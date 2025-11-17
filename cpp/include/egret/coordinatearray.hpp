/**
 * @file coordinatearray.hpp
 * @brief Class representing an array of particle coordinates in phase space.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// coordinatearray.hpp
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
#include <algorithm>
#include "egret/coordinate.hpp"

namespace egret {
    class CoordinateArray;
}

/**
 * @brief Class representing an array of particle coordinates in phase space.
 */
class egret::CoordinateArray {
protected:
    //! 4 x N matrix of particle coordinates (x, xp, y, yp)
    Eigen::Matrix<double, 4, Eigen::Dynamic> vector_array_;
    //! Longitudinal positions
    Eigen::ArrayXd s_array_;
    //! Longitudinal displacements
    Eigen::ArrayXd z_array_;
    //! Relative momentum deviations
    Eigen::ArrayXd delta_array_;

public:
    // Constructor
    CoordinateArray(
        const Eigen::Matrix<double,4,Eigen::Dynamic>& vector_array,
        const Eigen::ArrayXd& s_array,
        const Eigen::ArrayXd& z_array = Eigen::ArrayXd(),
        const Eigen::ArrayXd& delta_array = Eigen::ArrayXd());

    // Efficient append (reserve + copy)
    void append(const CoordinateArray &other);

    // Get Coordinate from linear interpolation
    Coordinate from_s(double s) const;
};
