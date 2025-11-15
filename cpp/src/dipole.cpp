// dipole.cpp
#include "egret/dipole.hpp"
#include <cmath>

namespace egret {

Eigen::Matrix4d Dipole::transfer_matrix(double length, double angle, double k1, double delta) {
    double rho = (length / angle) * (1.0 + delta);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (k1 == 0.0) {
        double phi = angle / (1.0 + delta);
        double cosphi = std::cos(phi), sinphi = std::sin(phi);
        tmat(0,0) = cosphi; tmat(0,1) = rho * sinphi;
        tmat(1,0) = -sinphi / rho; tmat(1,1) = cosphi;
        tmat(2,3) = length;
        return tmat;
    }
    double k = k1 / (1.0 + delta);
    double kx = k + 1.0 / (rho * rho);
    double sqrtkx = std::sqrt(std::abs(kx));
    double psix = sqrtkx * length;
    double ky = -k;
    double sqrtky = std::sqrt(std::abs(ky));
    double psiy = sqrtky * length;
    // handle cases similar to Python implementation
    if (kx < 0.0) {
        double coshx = std::cosh(psix), sinhx = std::sinh(psix);
        double cosy = std::cos(psiy), siny = std::sin(psiy);
        tmat(0,0)=coshx; tmat(0,1)=sinhx/sqrtkx;
        tmat(1,0)=sqrtkx*sinhx; tmat(1,1)=coshx;
        tmat(2,2)=cosy; tmat(2,3)=siny/sqrtky;
        tmat(3,2)=-sqrtky*siny; tmat(3,3)=cosy;
    } else if (ky < 0.0) {
        double cosx = std::cos(psix), sinx = std::sin(psix);
        double coshy = std::cosh(psiy), sinhy = std::sinh(psiy);
        tmat(0,0)=cosx; tmat(0,1)=sinx/sqrtkx;
        tmat(1,0)=-sqrtkx*sinx; tmat(1,1)=cosx;
        tmat(2,2)=coshy; tmat(2,3)=sinhy/sqrtky;
        tmat(3,2)=sqrtky*sinhy; tmat(3,3)=coshy;
    } else {
        double cosx = std::cos(psix), sinx = std::sin(psix);
        double cosy = std::cos(psiy), siny = std::sin(psiy);
        tmat(0,0)=cosx; tmat(0,1)=sinx/sqrtkx;
        tmat(1,0)=-sqrtkx*sinx; tmat(1,1)=cosx;
        tmat(2,2)=cosy; tmat(2,3)=siny/sqrtky;
        tmat(3,2)=-sqrtky*siny; tmat(3,3)=cosy;
    }
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Dipole::transfer_matrix_array(double length, double angle, double k1, double delta, double ds, bool endpoint) {
    double rho = (length / angle) * (1.0 + delta);
    int n = static_cast<int>(length / ds) + static_cast<int>(endpoint) + 1;
    if (length == 0.0) n = 1;
    std::vector<double> s(n);
    if (n==1) s[0]=0.0; else for (int i=0;i<n;++i) s[i] = (static_cast<double>(i) * length) / (n - 1);
    Eigen::Tensor<double,3> tmat(4,4,n);
    tmat.setZero();
    for (int i=0;i<n;++i) for (int j=0;j<4;++j) tmat(j,j,i)=1.0;

    if (k1 == 0.0) {
        for (int idx=0; idx<n; ++idx) {
            double phi = s[idx] / rho;
            double cosphi = std::cos(phi), sinphi = std::sin(phi);
            tmat(0,0,idx) = cosphi; tmat(0,1,idx) = rho * sinphi;
            tmat(1,0,idx) = -sinphi / rho; tmat(1,1,idx) = cosphi;
            tmat(2,3,idx) = s[idx];
        }
        return {tmat, s};
    }

    double k = k1 / (1.0 + delta);
    for (int idx=0; idx<n; ++idx) {
        double kx = k + 1.0 / (rho * rho);
        double sqrtkx = std::sqrt(std::abs(kx));
        double psix = sqrtkx * s[idx];
        double ky = -k;
        double sqrtky = std::sqrt(std::abs(ky));
        double psiy = sqrtky * s[idx];
        if (kx < 0.0) {
            double coshx = std::cosh(psix), sinhx = std::sinh(psix);
            double cosy = std::cos(psiy), siny = std::sin(psiy);
            tmat(0,0,idx)=coshx; tmat(0,1,idx)=sinhx/sqrtkx;
            tmat(1,0,idx)=sqrtkx*sinhx; tmat(1,1,idx)=coshx;
            tmat(2,2,idx)=cosy; tmat(2,3,idx)=siny/sqrtky;
            tmat(3,2,idx)=-sqrtky*siny; tmat(3,3,idx)=cosy;
        } else if (ky < 0.0) {
            double cosx = std::cos(psix), sinx = std::sin(psix);
            double coshy = std::cosh(psiy), sinhy = std::sinh(psiy);
            tmat(0,0,idx)=cosx; tmat(0,1,idx)=sinx/sqrtkx;
            tmat(1,0,idx)=-sqrtkx*sinx; tmat(1,1,idx)=cosx;
            tmat(2,2,idx)=coshy; tmat(2,3,idx)=sinhy/sqrtky;
            tmat(3,2,idx)=sqrtky*sinhy; tmat(3,3,idx)=coshy;
        } else {
            double cosx = std::cos(psix), sinx = std::sin(psix);
            double cosy = std::cos(psiy), siny = std::sin(psiy);
            tmat(0,0,idx)=cosx; tmat(0,1,idx)=sinx/sqrtkx;
            tmat(1,0,idx)=-sqrtkx*sinx; tmat(1,1,idx)=cosx;
            tmat(2,2,idx)=cosy; tmat(2,3,idx)=siny/sqrtky;
            tmat(3,2,idx)=-sqrtky*siny; tmat(3,3,idx)=cosy;
        }
    }
    return {tmat, s};
}

} // namespace egret
