// coordinatearray.cpp
#include "egret/coordinatearray.hpp"
#include <stdexcept>

namespace egret {

CoordinateArray::CoordinateArray()
    : vector(Eigen::Matrix<double,4,Eigen::Dynamic>(4,0)) {}

CoordinateArray::CoordinateArray(const Eigen::Matrix<double,4,Eigen::Dynamic>& vec,
                                 const std::vector<double>& s_,
                                 const std::vector<double>& z_,
                                 const std::vector<double>& delta_)
    : vector(vec), s(s_), z(z_), delta(delta_) {}

void CoordinateArray::append(const CoordinateArray &other) {
    auto n0 = vector.cols();
    auto n1 = other.vector.cols();
    Eigen::Matrix<double,4,Eigen::Dynamic> tmp(4, n0 + n1);
    tmp << vector, other.vector;
    vector.swap(tmp);
    s.insert(s.end(), other.s.begin(), other.s.end());
    z.insert(z.end(), other.z.begin(), other.z.end());
    delta.insert(delta.end(), other.delta.begin(), other.delta.end());
}

Coordinate CoordinateArray::from_s(double sval) const {
    if (s.empty()) throw std::out_of_range("CoordinateArray is empty");
    auto it = std::lower_bound(s.begin(), s.end(), sval);
    size_t idx = std::distance(s.begin(), it);
    if (idx == s.size()) {
        if (idx == 0) throw std::out_of_range("Out of range");
        idx = s.size() - 1;
    }
    if (idx == s.size() - 1) throw std::out_of_range("Out of range");
    double s0 = s[idx];
    double s1 = s[idx+1];
    double ds = s1 - s0;
    double a0 = (s1 - sval) / (ds == 0. ? 2. : ds);
    double a1 = (sval - s0) / (ds == 0. ? 2. : ds);
    Eigen::Vector4d vec = a0 * vector.col(idx) + a1 * vector.col(idx+1);
    double zval = a0 * z[idx] + a1 * z[idx+1];
    double dval = a0 * delta[idx] + a1 * delta[idx+1];
    return Coordinate(vec, sval, zval, dval);
}

} // namespace egret
