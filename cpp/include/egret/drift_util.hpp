/**
 * @file drift_util.hpp
 * @brief Shared drift helper utilities.
 * @author Hirokazu Maesaka
 * @date 2025
 */
// drift_util.hpp
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

#pragma once

#include "egret/coordinate.hpp"
#include "egret/coordinatearray.hpp"
#include "egret/dispersion.hpp"
#include "egret/dispersionarray.hpp"
#include "egret/envelope.hpp"
#include "egret/envelopearray.hpp"
#include <cmath>
#include <ranges>

namespace egret::util {
inline Eigen::Matrix4d drift_matrix(const double length) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 1) = length;
    M(2, 3) = length;
    return M;
}

inline egret::Coordinate drift_coordinate(const egret::Coordinate &cood, const double length) {
    if (length == 0.0) {
        return cood;
    }
    const auto M = drift_matrix(length);
    return egret::Coordinate(M * cood.vector(), cood.s(), cood.z(), cood.delta());
}

inline egret::CoordinateArray drift_coordinate_array(
    const egret::CoordinateArray &cood_array, const double length) {
    if (length == 0.0) {
        return cood_array;
    }
    const auto M = drift_matrix(length);
    return egret::CoordinateArray(M * cood_array.vector_array(), cood_array.s_array(),
        cood_array.z_array(), cood_array.delta_array());
}

inline egret::Dispersion drift_dispersion(const egret::Dispersion &disp, const double length) {
    if (length == 0.0) {
        return disp;
    }
    const auto M = drift_matrix(length);
    return egret::Dispersion(M * disp.vector(), disp.s());
}

inline egret::DispersionArray drift_dispersion_array(
    const egret::DispersionArray &disp_array, const double length) {
    if (length == 0.0) {
        return disp_array;
    }
    const auto M = drift_matrix(length);
    return egret::DispersionArray(M * disp_array.vector_array(), disp_array.s_array());
}

inline egret::Envelope drift_envelope(const egret::Envelope &evlp, const double length) {
    if (length == 0.0) {
        return evlp;
    }
    const auto M4 = drift_matrix(length);
    Eigen::Matrix2d M2 = Eigen::Matrix2d::Identity();
    M2(0, 1) = length;
    Eigen::Matrix2d M2m = Eigen::Matrix2d::Identity();
    M2m(0, 1) = -length;
    const auto cov = M4 * evlp.cov() * M4.transpose();
    const auto T = M2 * evlp.T() * M2m;
    const double psix = evlp.psix() + std::atan2(length, evlp.bu() - evlp.au() * length);
    const double psiy = evlp.psiy() + std::atan2(length, evlp.bv() - evlp.av() * length);
    return egret::Envelope(cov, evlp.s(), T, psix, psiy);
}

inline egret::EnvelopeArray drift_envelope_array(
    const egret::EnvelopeArray &evlp_array, const double length) {
    if (length == 0.0) {
        return evlp_array;
    }
    const auto M4 = drift_matrix(length);
    Eigen::Matrix2d M2 = Eigen::Matrix2d::Identity();
    M2(0, 1) = length;
    Eigen::Matrix2d M2m = Eigen::Matrix2d::Identity();
    M2m(0, 1) = -length;
    auto cov_array = evlp_array.cov_array();
    auto T_array = evlp_array.T_array();
    auto psix_array = evlp_array.psix_array();
    auto psiy_array = evlp_array.psiy_array();
    const auto bu_array = evlp_array.bu_array();
    const auto au_array = evlp_array.au_array();
    const auto bv_array = evlp_array.bv_array();
    const auto av_array = evlp_array.av_array();
    const size_t n = cov_array.size();
    for (const size_t i : std::views::iota(0u, n)) {
        cov_array[i] = M4 * cov_array[i] * M4.transpose();
        if (i < T_array.size()) {
            T_array[i] = M2 * T_array[i] * M2m;
        }
        psix_array(i) += std::atan2(length, bu_array(i) - au_array(i) * length);
        psiy_array(i) += std::atan2(length, bv_array(i) - av_array(i) * length);
    }
    return egret::EnvelopeArray(cov_array, evlp_array.s_array(), T_array, psix_array, psiy_array);
}
} // namespace egret::util
