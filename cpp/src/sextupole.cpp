// sextupole.cpp
// sextupole.cpp
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

#include "egret/sextupole.hpp"
#include <complex>
#include <cmath>

namespace egret {

Sextupole::Sextupole(double length, double k2, double dx, double dy, double ds,
                     double tilt, double dxp, double dyp) {
    (void)length; (void)k2; (void)dx; (void)dy; (void)ds; (void)tilt; (void)dxp; (void)dyp;
}

std::tuple<Eigen::Matrix4d, Coordinate, Eigen::Vector4d>
Sextupole::transfer_matrix_by_midpoint_method(const Coordinate &cood0, double length, double k2,
                                              double k0x, double k0y,
                                              double dx, double dy, double ds,
                                              bool tmatflag, bool dispflag) {
    using cplx = std::complex<double>;
    double k2loc = k2 / (1.0 + cood0.delta);
    double k0xloc = k0x / (1.0 + cood0.delta);
    double k0yloc = k0y / (1.0 + cood0.delta);
    double x0 = cood0.vector(0);
    double y0 = cood0.vector(2);
    double xp0 = cood0.vector(1);
    double yp0 = cood0.vector(3);
    cplx k0a = k2loc * (0.5 * (x0*x0 - y0*y0) - cplx(0.0,1.0) * x0 * y0) + cplx(k0xloc, k0yloc);
    cplx k1a = k2loc * (cplx(x0, -y0));
    double tilt = 0.5 * std::arg(k1a);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    Coordinate cood2;
    Eigen::Vector4d disp = Eigen::Vector4d::Zero();
    const double eps = 1e-20;
    double abs_k1a = std::abs(k1a);
    double x1=0.0, y1=0.0;
    if (abs_k1a < eps) {
        x1 = x0 + (xp0 - 0.5 * k0a.real() * ds) * ds;
        y1 = y0 + (yp0 - 0.5 * k0a.imag() * ds) * ds;
    } else {
        cplx offset = - std::exp(cplx(0,1)*tilt) * std::conj(std::exp(cplx(0,-1)*tilt) * k0a) / abs_k1a;
        double offx = offset.real();
        double offy = offset.imag();
        // first quad of length ds with k1 = abs(k1a)
        Eigen::Matrix4d tquad1 = Quadrupole::transfer_matrix(ds, std::abs(k1a), tilt, 0.0);
        Eigen::Vector4d cood_in; cood_in << 0.0, xp0, 0.0, yp0;
        Eigen::Vector4d cood_out = tquad1 * cood_in;
        x1 = cood_out(0) + x0;
        y1 = cood_out(2) + y0;
    }
    // second step
    cplx k0b = k2loc * (0.5 * (x1*x1 - y1*y1) - cplx(0.0,1.0) * x1 * y1) + cplx(k0xloc, k0yloc);
    cplx k1b = k2loc * cplx(x1, -y1);
    cplx k0 = 0.5 * (k0a + k0b);
    cplx k1 = 0.5 * (k1a + k1b);
    tilt = 0.5 * std::arg(k1);
    double abs_k1 = std::abs(k1);
    if (abs_k1 < eps) {
        // no quadrupole: dipole kick
        Eigen::Vector4d coodvec;
        coodvec << x0 + (xp0 - 0.5 * k0.real() * ds) * ds,
                  xp0 - k0.real() * ds,
                  y0 + (yp0 - 0.5 * k0.imag() * ds) * ds,
                  yp0 - k0.imag() * ds;
        cood2 = Coordinate(coodvec, cood0.s + ds, cood0.z, cood0.delta);
        if (tmatflag) tmat = Drift::transfer_matrix_from_length(ds);
        if (dispflag) {
            disp << 0.5 * k0.real() * ds*ds, k0.real() * ds, 0.5 * k0.imag() * ds*ds, k0.imag() * ds;
        }
    } else {
        cplx offset = - std::exp(cplx(0,1)*tilt) * std::conj(std::exp(cplx(0,-1)*tilt) * k0) / std::abs(k1);
        double offx = offset.real();
        double offy = offset.imag();
        Eigen::Matrix4d tquad2 = Quadrupole::transfer_matrix(ds, abs_k1, tilt, 0.0);
        Eigen::Vector4d cood_in; cood_in << 0.0, xp0, 0.0, yp0;
        Eigen::Vector4d cood_out = tquad2 * cood_in;
        cood_out(0) += x0; cood_out(2) += y0;
        cood_out(1) = cood_out(1); // keep delta
        cood2 = Coordinate(cood_out, cood0.s + ds, cood0.z, cood0.delta);
        if (tmatflag) tmat = Quadrupole::transfer_matrix(ds, abs_k1, tilt, 0.0);
        if (dispflag) {
            disp = Quadrupole::dispersion(cood_in, ds, abs_k1, 0.0);
        }
    }
    return {tmat, cood2, disp};
}

Eigen::Matrix4d Sextupole::transfer_matrix(const Coordinate &cood0, double length, double k2, double ds) {
    int n_step = static_cast<int>(length / ds) + 1;
    double s_step = length / n_step;
    Coordinate cood = cood0;
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    for (int i=0;i<n_step;++i) {
        Eigen::Matrix4d tstep; Coordinate ctmp; Eigen::Vector4d dtmp;
        std::tie(tstep, ctmp, dtmp) = Sextupole::transfer_matrix_by_midpoint_method(cood, length, k2, 0.0, 0.0, 0.0, 0.0, s_step);
        tmat = tstep * tmat;
        cood = ctmp;
    }
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Sextupole::transfer_matrix_array(const Coordinate &cood0, double length, double k2, double ds, bool endpoint) {
    int n_base = static_cast<int>(std::floor(length / ds));
    int n_step = n_base + 1;
    double s_step = length / n_step;
    int n = n_base + static_cast<int>(endpoint) + 1;
    std::vector<double> s(n);
    for (int i=0;i<n;++i) s[i] = (static_cast<double>(i) * length) / (n - 1);
    Eigen::Tensor<double,3> tmat(4,4,n);
    for (int i=0;i<n;++i) for (int j=0;j<4;++j) { for (int k=0;k<4;++k) tmat(k,j,i) = (j==k)?1.0:0.0; }
    Coordinate cood = cood0;
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    tmat.setZero();
    for (int i=0;i<n;++i) {
        if (i==0) {
            for (int r=0;r<4;++r) tmat(r,r,i)=1.0;
        } else {
            Eigen::Matrix4d tstep; Coordinate ctmp; Eigen::Vector4d dtmp;
            std::tie(tstep, ctmp, dtmp) = Sextupole::transfer_matrix_by_midpoint_method(cood, length, k2, 0.0, 0.0, 0.0, 0.0, s_step);
            t = tstep * t;
            cood = ctmp;
            for (int r=0;r<4;++r) for (int c=0;c<4;++c) tmat(r,c,i) = t(r,c);
        }
    }
    return {tmat, s};
}

Eigen::Vector4d Sextupole::dispersion(const Coordinate &cood0, double length, double k2, double ds) {
    int n_base = static_cast<int>(std::floor(length / ds));
    int n_step = n_base + 1;
    double s_step = length / n_step;
    Coordinate cood = cood0;
    Eigen::Vector4d dispout = Eigen::Vector4d::Zero();
    for (int i=0;i<n_step;++i) {
        Eigen::Matrix4d tmat; Coordinate ctmp; Eigen::Vector4d disp;
        std::tie(tmat, ctmp, disp) = Sextupole::transfer_matrix_by_midpoint_method(cood, length, k2, 0.0, 0.0, 0.0, 0.0, s_step, true, true);
        dispout = tmat * dispout + disp;
        cood = ctmp;
    }
    return dispout;
}

} // namespace egret
