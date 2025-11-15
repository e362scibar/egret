// quadrupole.cpp
#include "egret/quadrupole.hpp"
#include <cmath>

namespace egret {

Eigen::Matrix4d Quadrupole::transfer_matrix(double length, double k1, double tilt, double delta) {
    double k = k1 / (1.0 + delta);
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (k == 0.0) {
        tmat(0,1) = length;
        tmat(2,3) = length;
        return tmat;
    }
    double sqrtk = std::sqrt(std::abs(k));
    double psi = sqrtk * length;
    double cospsi = std::cos(psi), sinpsi = std::sin(psi);
    double coshpsi = std::cosh(psi), sinhpsi = std::sinh(psi);
    Eigen::Matrix2d mf, md;
    mf << cospsi, sinpsi / sqrtk,
          -sqrtk * sinpsi, cospsi;
    md << coshpsi, sinhpsi / sqrtk,
          sqrtk * sinhpsi, coshpsi;
    if (k < 0.0) {
        tmat.block<2,2>(0,0) = md;
        tmat.block<2,2>(2,2) = mf;
    } else {
        tmat.block<2,2>(0,0) = mf;
        tmat.block<2,2>(2,2) = md;
    }
    if (tilt != 0.0) {
        double ct = std::cos(tilt), st = std::sin(tilt);
        Eigen::Matrix4d rmat = Eigen::Matrix4d::Zero();
        rmat(0,0)=ct; rmat(0,2)=st;
        rmat(1,1)=ct; rmat(1,3)=st;
        rmat(2,0)=-st; rmat(2,2)=ct;
        rmat(3,1)=-st; rmat(3,3)=ct;
        tmat = rmat.transpose() * tmat * rmat;
    }
    return tmat;
}

std::pair<Eigen::Tensor<double,3>, std::vector<double>> Quadrupole::transfer_matrix_array(double length, double k1, double tilt, double delta, double ds, bool endpoint) {
    double k = k1 / (1.0 + delta);
    int n_base = static_cast<int>(std::floor(length / ds));
    int n = n_base + static_cast<int>(endpoint) + 1;
    if (length == 0.0) {
        n = 1;
    }
    std::vector<double> s(n);
    if (n==1) s[0]=0.0;
    else for (int i=0;i<n;++i) s[i] = (static_cast<double>(i) * length) / (n - 1);
    Eigen::Tensor<double,3> tmat(4,4,n);
    // initialize to identity
    tmat.setZero();
    for (int i=0;i<n;++i) for (int j=0;j<4;++j) tmat(j,j,i)=1.0;

    if (k == 0.0) {
        for (int i=0;i<n;++i) {
            tmat(0,1,i) = s[i];
            tmat(2,3,i) = s[i];
        }
        return {tmat, s};
    }

    // compute per-s psi values
    for (int idx=0; idx<n; ++idx) {
        double sqrtk = std::sqrt(std::abs(k));
        double psi = sqrtk * s[idx];
        double cospsi = std::cos(psi), sinpsi = std::sin(psi);
        double coshpsi = std::cosh(psi), sinhpsi = std::sinh(psi);
        Eigen::Matrix2d mf, md;
        mf << cospsi, sinpsi / sqrtk,
              -sqrtk * sinpsi, cospsi;
        md << coshpsi, sinhpsi / sqrtk,
              sqrtk * sinhpsi, coshpsi;
        if (k < 0.0) {
            // place md in 0:2,0:2 and mf in 2:4,2:4
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r,c,idx)=md(r,c);
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r+2,c+2,idx)=mf(r,c);
        } else {
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r,c,idx)=mf(r,c);
            for (int r=0;r<2;++r) for (int c=0;c<2;++c) tmat(r+2,c+2,idx)=md(r,c);
        }
    }

    if (tilt != 0.0) {
        // apply rotation rmat.T @ tmat @ rmat for each slice
        double ct = std::cos(tilt), st = std::sin(tilt);
        Eigen::Matrix4d rmat = Eigen::Matrix4d::Zero();
        rmat(0,0)=ct; rmat(0,2)=st;
        rmat(1,1)=ct; rmat(1,3)=st;
        rmat(2,0)=-st; rmat(2,2)=ct;
        rmat(3,1)=-st; rmat(3,3)=ct;
        Eigen::Matrix4d tmp;
        for (int idx=0; idx<n; ++idx) {
            // load slice into tmp
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) tmp(i,j) = tmat(i,j,idx);
            tmp = rmat.transpose() * tmp * rmat;
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) tmat(i,j,idx) = tmp(i,j);
        }
    }
    return {tmat, s};
}

Eigen::Vector4d Quadrupole::dispersion(const Eigen::Vector4d &cood0vec, double length, double k1, double delta) {
    Eigen::Vector4d disp = Eigen::Vector4d::Zero();
    double k = k1 / (1.0 + delta);
    if (k == 0.0) return disp;
    double sqrtk = std::sqrt(std::abs(k));
    double psi = sqrtk * length;
    double cospsi = std::cos(psi), sinpsi = std::sin(psi);
    double coshpsi = std::cosh(psi), sinhpsi = std::sinh(psi);
    Eigen::Matrix2d Mf1, Mf2, Md1, Md2;
    Mf1 = (Eigen::Matrix2d() << sinpsi, -cospsi / sqrtk, sqrtk * cospsi, sinpsi).finished();
    Mf1 *= 0.5 * length * sqrtk;
    Mf2 = Eigen::Matrix2d::Zero();
    Mf2(0,1) = sinpsi / sqrtk;
    Mf2(1,0) = sqrtk * sinpsi;
    Mf2 *= 0.5;
    Md1 = (Eigen::Matrix2d() << -sinhpsi, -coshpsi / sqrtk, -sqrtk * coshpsi, -sinhpsi).finished();
    Md1 *= 0.5 * length * sqrtk;
    Md2 = Eigen::Matrix2d::Zero();
    Md2(0,1) = sinhpsi / sqrtk;
    Md2(1,0) = -sqrtk * sinhpsi;
    Md2 *= 0.5;
    if (k < 0.0) {
        Eigen::Vector2d v0 = cood0vec.segment<2>(0);
        Eigen::Vector2d v1 = cood0vec.segment<2>(2);
        Eigen::Vector2d r0 = (Md1 + Md2) * v0;
        Eigen::Vector2d r1 = (Mf1 + Mf2) * v1;
        disp.segment<2>(0) = r0;
        disp.segment<2>(2) = r1;
    } else {
        Eigen::Vector2d v0 = cood0vec.segment<2>(0);
        Eigen::Vector2d v1 = cood0vec.segment<2>(2);
        Eigen::Vector2d r0 = (Mf1 + Mf2) * v0;
        Eigen::Vector2d r1 = (Md1 + Md2) * v1;
        disp.segment<2>(0) = r0;
        disp.segment<2>(2) = r1;
    }
    return disp;
}

} // namespace egret
