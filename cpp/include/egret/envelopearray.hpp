// envelopearray.hpp
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

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include "egret/envelope.hpp"

namespace egret {

class EnvelopeArray {
protected:
    // 4 x N matrix
    Eigen::Matrix<double, 4, Eigen::Dynamic> cov;
    std::vector<double> s;
    std::vector<double> z;
    std::vector<double> delta;

public:
    EnvelopeArray();
    EnvelopeArray(const Eigen::Matrix<double,4,Eigen::Dynamic>& cov,
                    const std::vector<double>& s_,
                    const std::vector<double>& z_ = {},
                    const std::vector<double>& delta_ = {});

    // Efficient append (reserve + copy)
    void append(const EnvelopeArray &other);

    // linear interpolation like Python's from_s
    Envelope from_s(double sval) const;
};

} // namespace egret
