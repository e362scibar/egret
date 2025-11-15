// drift.cpp
#include "egret/drift.hpp"
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace egret {

Eigen::Matrix4d Drift::transfer_matrix_from_length(double length) {
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    tmat(0,1) = length;
    tmat(2,3) = length;
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Drift::transfer_matrix_array_from_length(double length, double ds, bool endpoint) {
    std::vector<double> s;
    if (std::abs(length) > 0.0) {
        int n = static_cast<int>(length / ds) + static_cast<int>(endpoint) + 1;
        s.reserve(n);
        for (int i = 0; i < n; ++i) s.push_back((static_cast<double>(i) * length) / (n - 1));
    } else {
        s.push_back(0.0);
    }

    int N = static_cast<int>(s.size());
    Eigen::Tensor<double,3> tmat(4,4,N);
    tmat.setZero();
    for (int k=0;k<N;++k) {
        // identity
        for (int i=0;i<4;++i) tmat(i,i,k) = 1.0;
        tmat(0,1,k) = s[k];
        tmat(2,3,k) = s[k];
    }
    return {tmat, s};
}

} // namespace egret
