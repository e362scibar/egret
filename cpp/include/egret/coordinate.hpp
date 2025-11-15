// coordinate.hpp
#pragma once

#include <Eigen/Dense>

namespace egret {

struct Coordinate {
    Eigen::Vector4d vector; // x, xp, y, yp
    double s{0.0};
    double z{0.0};
    double delta{0.0};

    Coordinate() : vector(Eigen::Vector4d::Zero()), s(0.), z(0.), delta(0.) {}
    Coordinate(const Eigen::Vector4d &v, double s_, double z_ = 0., double d_ = 0.)
        : vector(v), s(s_), z(z_), delta(d_) {}
};

} // namespace egret
