/**
 * @file lattice.hpp
 * @brief Lattice element class definition
 * @author Hirokazu Maesaka
 * @date 2025
 */
// lattice.hpp
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

#include "egret/element.hpp"

namespace egret {
    class Lattice;
}
class egret::Lattice: public egret::Element {
public:
    // Construct a new Lattice object
    Lattice(const std::string &name, const std::vector<std::shared_ptr<Element>> &elements,
        double dx=0.0, double dy=0.0, double ds=0.0,
        double tilt=0.0, const std::string &info="") noexcept(false);
    /**
     * @brief Destroy the Lattice object
     */
    virtual ~Lattice() noexcept = default;

    // Calculate total length of a vector of elements
    static double length(const std::vector<std::shared_ptr<Element>> &elements) noexcept;

    // Calculate total bending angle of a vector of elements
    static double angle(const std::vector<std::shared_ptr<Element>> &elements) noexcept;

    // Polymorphic clone
    std::shared_ptr<Element> clone() const override;
};
