// coordinatearray.hpp
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include "coordinate.hpp"

namespace egret {

class CoordinateArray {
public:
    // 4 x N matrix
    Eigen::Matrix<double, 4, Eigen::Dynamic> vector;
    std::vector<double> s;
    std::vector<double> z;
    std::vector<double> delta;

    CoordinateArray();
    CoordinateArray(const Eigen::Matrix<double,4,Eigen::Dynamic>& vec,
                   const std::vector<double>& s_,
                   const std::vector<double>& z_ = {},
                   const std::vector<double>& delta_ = {});

    // Efficient append (reserve + copy)
    void append(const CoordinateArray &other);

    // linear interpolation like Python's from_s
    Coordinate from_s(double sval) const;
};

} // namespace egret
