// dipole.hpp
#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace egret {

class Dipole {
public:
    // simple dipole or combined-function dipole transfer matrix
    static Eigen::Matrix4d transfer_matrix(double length, double angle, double k1=0.0, double delta=0.0);

    static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array(double length, double angle, double k1=0.0, double delta=0.0, double ds=0.1, bool endpoint=false);
};

} // namespace egret
