// cli_quadrupole.cpp
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

#include <iostream>
#include <Eigen/Dense>
#include "egret/quadrupole.hpp"

int main(int argc, char **argv) {
    double length = 0.5;
    double k1 = 1.2;
    double tilt = 0.13;
    double delta = 0.0;
    if (argc > 1) length = std::atof(argv[1]);
    if (argc > 2) k1 = std::atof(argv[2]);
    if (argc > 3) tilt = std::atof(argv[3]);
    Eigen::Matrix4d t = egret::Quadrupole::transfer_matrix(length, k1, tilt, delta);
    // print 4x4 matrix as rows
    std::cout.setf(std::ios::scientific);
    for (int i=0;i<4;++i) {
        for (int j=0;j<4;++j) {
            std::cout << t(i,j);
            if (j<3) std::cout << ' ';
        }
        std::cout << '\n';
    }
    return 0;
}
