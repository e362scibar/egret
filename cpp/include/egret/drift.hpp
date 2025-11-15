// drift.hpp
#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace egret {

class Drift {
public:
    // Return 4x4 transfer matrix for a drift of given length
    static Eigen::Matrix4d transfer_matrix_from_length(double length);

    // Return transfer matrix array (4x4xN) and s array for given length and step ds
    static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array_from_length(double length, double ds = 0.1, bool endpoint = false);
};

} // namespace egret
