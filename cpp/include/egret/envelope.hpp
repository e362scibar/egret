/**
 * @brief Class representing a particle coordinate in phase space
 */
// envelope.hpp
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

namespace egret {

class Envelope {
protected:
    Eigen::Matrix4d cov;
public:
    Envelope() : cov(Eigen::Matrix4d::Identity()) {}
    Envelope(const Eigen::Matrix4d &cov): cov(cov) {}
};

} // namespace egret
