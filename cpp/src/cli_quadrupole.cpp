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
