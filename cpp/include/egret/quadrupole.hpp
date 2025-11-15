// quadrupole.hpp
#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace egret {

class Quadrupole {
public:
    // transfer matrix for a quadrupole of length L and strength k (k=k1/(1+delta))
    static Eigen::Matrix4d transfer_matrix(double length, double k1, double tilt = 0.0, double delta = 0.0);

    // transfer matrix array along the element (4x4xN) and s positions
    static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array(double length, double k1, double tilt = 0.0, double delta = 0.0, double ds = 0.1, bool endpoint = false);
    // additive dispersion vector for given initial coordinate
    static Eigen::Vector4d dispersion(const Eigen::Vector4d &cood0vec, double length, double k1, double delta = 0.0);
};

} // namespace egret
