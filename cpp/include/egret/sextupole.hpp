// sextupole.hpp
#pragma once

#include "egret/coordinate.hpp"
#include "egret/quadrupole.hpp"
#include "egret/drift.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include <vector>

namespace egret {

class Sextupole {
public:
    Sextupole(double length, double k2, double dx = 0.0, double dy = 0.0, double ds = 0.0,
             double tilt = 0.0, double dxp = 0.0, double dyp = 0.0);

    // midpoint integrator single-step: returns (tmat, final_coordinate, dispersion)
    static std::tuple<Eigen::Matrix4d, Coordinate, Eigen::Vector4d>
    transfer_matrix_by_midpoint_method(const Coordinate &cood0, double length, double k2,
                                       double k0x = 0.0, double k0y = 0.0,
                                       double dx = 0.0, double dy = 0.0, double ds = 0.1,
                                       bool tmatflag = true, bool dispflag = false);

    static Eigen::Matrix4d transfer_matrix(const Coordinate &cood0, double length, double k2, double ds = 0.1);

    static std::pair<Eigen::Tensor<double,3>, std::vector<double>> transfer_matrix_array(const Coordinate &cood0, double length, double k2, double ds = 0.1, bool endpoint = false);

    static Eigen::Vector4d dispersion(const Coordinate &cood0, double length, double k2, double ds = 0.1);
};

} // namespace egret
